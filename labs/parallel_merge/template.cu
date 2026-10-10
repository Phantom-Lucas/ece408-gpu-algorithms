#include <cstdio>
#include <cstdlib>
#include <stdio.h>

#include "template.hu"

#define BLOCK_SIZE 512
#define TILE_SIZE 512
#define ELEMENTS_PER_THREAD 8

// Ceiling funciton for X / Y.
__host__ __device__ static inline int ceil_div(int x, int y) {
    return (x - 1) / y + 1;
}
/******************************************************************************
 GPU kernels
*******************************************************************************/
__device__ int co_rank(int k, const float* A, int A_len,
                       const float* B, int B_len) {
    if(k >= A_len + B_len) {
        return A_len;
    }
    int left = max(0, k - B_len);
    int right = min(k, A_len);

    while (left <= right) {
        int i = left + (right - left) / 2;
        int j = k - i;

        if (i > 0 && j < B_len && A[i - 1] > B[j]) {
            right = i - 1;
        }
        else if (j > 0 && i < A_len && B[j - 1] >= A[i]) {
            left = i + 1;
        }
        else {
            return i;
        }
    }

    return left;
}
/*
 * Sequential merge implementation is given. You can use it in your kernels.
 */
__device__ void merge_sequential(float* A, int A_len, float* B, int B_len, float* C) {
    int i = 0, j = 0, k = 0;

    while ((i < A_len) && (j < B_len)) {
        C[k++] = A[i] <= B[j] ? A[i++] : B[j++];
    }

    if (i == A_len) {
        while (j < B_len) {
            C[k++] = B[j++];
        }
    } else {
        while (i < A_len) {
            C[k++] = A[i++];
        }
    }
}

/*
 * Basic parallel merge kernel using co-rank function
 * A, A_len - input array A and its length
 * B, B_len - input array B and its length
 * C - output array holding the merged elements.
 *      Length of C is A_len + B_len (size pre-allocated for you)
 */
__global__ void gpu_merge_basic_kernel(
    float* A, int A_len,
    float* B, int B_len,
    float* C) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int stride = gridDim.x * blockDim.x
               * ELEMENTS_PER_THREAD;

    int total_len = A_len + B_len;

    for (int start = idx * ELEMENTS_PER_THREAD;
         start < total_len;
         start += stride) {

        int end = min(start + ELEMENTS_PER_THREAD, total_len);

        int i = co_rank(start, A, A_len, B, B_len);
        int j = start - i;

        for (int t = start; t < end; t++) {

            if (i < A_len &&
                (j >= B_len || A[i] <= B[j])) {

                C[t] = A[i];
                i++;

            } else {

                C[t] = B[j];
                j++;
            }
        }
    }
}

/*
 * Arguments are the same as gpu_merge_basic_kernel.
 * In this kernel, use shared memory to increase the reuse.
 */
#define THREADS_PER_TILE (TILE_SIZE / ELEMENTS_PER_THREAD)
#define TILES_PER_BLOCK (BLOCK_SIZE / THREADS_PER_TILE)
#define OUTPUTS_PER_ROUND (TILES_PER_BLOCK * TILE_SIZE)

__global__ void gpu_merge_tiled_kernel(
    float* A, int A_len,
    float* B, int B_len,
    float* C) {

    // 8 个 Tile，每个 Tile 512 个 float
    __shared__ float smem[TILES_PER_BLOCK][TILE_SIZE];

    // 每个 Tile 的输入起点和长度
    __shared__ int A_start[TILES_PER_BLOCK];
    __shared__ int B_start[TILES_PER_BLOCK];
    __shared__ int A_count[TILES_PER_BLOCK];
    __shared__ int tile_count[TILES_PER_BLOCK];

    int tx = threadIdx.x;

    // 当前线程属于哪个 Tile
    int tile_id = tx / THREADS_PER_TILE;

    // 当前线程在这个 Tile 内的编号：0~63
    int lane = tx % THREADS_PER_TILE;

    int total = A_len + B_len;

    // 将整个输出数组分给 128 个 Block
    int block_start =
        (long long)total * blockIdx.x / gridDim.x;

    int block_end =
        (long long)total * (blockIdx.x + 1) / gridDim.x;

    // 每轮处理最多 4096 个输出
    for (int group_start = block_start;
         group_start < block_end;
         group_start += OUTPUTS_PER_ROUND) {

        // 1. 前 8 个线程分别计算 8 个 Tile 的边界
        if (tx < TILES_PER_BLOCK) {

            int start = group_start + tx * TILE_SIZE;
            int len = max(0, min(TILE_SIZE, block_end - start));

            tile_count[tx] = len;

            if (len > 0) {

                int a0 = co_rank(
                    start, A, A_len, B, B_len);

                int a1 = co_rank(
                    start + len, A, A_len, B, B_len);

                A_start[tx] = a0;
                B_start[tx] = start - a0;
                A_count[tx] = a1 - a0;
            } else {
                A_start[tx] = 0;
                B_start[tx] = 0;
                A_count[tx] = 0;
            }
        }

        __syncthreads();

        int len = tile_count[tile_id];
        int na = A_count[tile_id];
        int nb = len - na;

        float* As = smem[tile_id];
        float* Bs = smem[tile_id] + na;

        // 2. 每组 64 个线程协作加载自己的 Tile
        for (int p = lane; p < len; p += THREADS_PER_TILE) {

            if (p < na) {
                smem[tile_id][p] = A[A_start[tile_id] + p];
            } else {
                smem[tile_id][p] =
                    B[B_start[tile_id] + p - na];
            }
        }

        __syncthreads();

        // 3. 每个线程负责 8 个输出
        int local_start = lane * ELEMENTS_PER_THREAD;
        int local_end = min(
            local_start + ELEMENTS_PER_THREAD, len);

        if (local_start < len) {

            // 在 Shared Memory 中 co_rank
            int i = co_rank(local_start, As, na, Bs, nb);
            int j = local_start - i;

            int output_start =
                group_start + tile_id * TILE_SIZE;

            // 4. 顺序归并 8 个元素
            for (int p = local_start; p < local_end; p++) {

                if (i < na && (j >= nb || As[i] <= Bs[j])) {
                    C[output_start + p] = As[i++];
                } else {
                    C[output_start + p] = Bs[j++];
                }
            }
        }

        // 确保所有线程完成后再覆盖 Shared Memory
        __syncthreads();
    }
}

/*
 * gpu_merge_circular_buffer_kernel is optional.
 * The implementation will be similar to tiled merge kernel.
 * You'll have to modify co-rank function and sequential_merge
 * to accommodate circular buffer.
 */
__global__ void gpu_merge_circular_buffer_kernel(float* A, int A_len, float* B, int B_len, float* C) {
    /* Your code here */
}

/******************************************************************************
 Functions
*******************************************************************************/

void gpu_basic_merge(float* A, int A_len, float* B, int B_len, float* C) {
    const int numBlocks = 128;
    gpu_merge_basic_kernel<<<numBlocks, BLOCK_SIZE>>>(A, A_len, B, B_len, C);
}

void gpu_tiled_merge(float* A, int A_len, float* B, int B_len, float* C) {
    const int numBlocks = 128;
    gpu_merge_tiled_kernel<<<numBlocks, BLOCK_SIZE>>>(A, A_len, B, B_len, C);
}

void gpu_circular_buffer_merge(float* A, int A_len, float* B, int B_len, float* C) {
    const int numBlocks = 128;
    gpu_merge_circular_buffer_kernel<<<numBlocks, BLOCK_SIZE>>>(A, A_len, B, B_len, C);
}
