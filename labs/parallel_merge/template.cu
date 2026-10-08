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
    int total_len = A_len + B_len;

    int start = ELEMENTS_PER_THREAD * idx;

    if (start >= total_len) return;

    int end = min(start + ELEMENTS_PER_THREAD, total_len);

    int i = co_rank(start, A, A_len, B, B_len);
    int j = start - i;

    for (int t = start; t < end; t++) {

        if (i < A_len && j < B_len) {

            if (A[i] <= B[j]) {
                C[t] = A[i];
                i++;
            }
            else {
                C[t] = B[j];
                j++;
            }
        }
        else if (i < A_len) {
            C[t] = A[i];
            i++;
        }
        else if (j < B_len) {
            C[t] = B[j];
            j++;
        }
    }
}

/*
 * Arguments are the same as gpu_merge_basic_kernel.
 * In this kernel, use shared memory to increase the reuse.
 */
__global__ void gpu_merge_tiled_kernel(float* A, int A_len, float* B, int B_len, float* C) {
    /* Your code here */
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
