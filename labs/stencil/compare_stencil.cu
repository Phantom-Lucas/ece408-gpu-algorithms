#include <cuda_runtime.h>

#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define TILE_SIZE 30

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            std::exit(EXIT_FAILURE);                                     \
        }                                                                 \
    } while (0)

// A: Explicit register tiling along z (sliding prev/curr/next window).
__global__ void stencil_register(const int* __restrict__ A0,
                                 int* __restrict__ Anext,
                                 int nx, int ny, int nz) {
    __shared__ int smem[TILE_SIZE + 2][TILE_SIZE + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * TILE_SIZE + tx;
    int y = blockIdx.y * TILE_SIZE + ty;

    int plane = nx * ny;
    int offset = y * nx + x;
    bool valid = (x < nx && y < ny);
    bool output = (tx > 0 && tx <= TILE_SIZE &&
                   ty > 0 && ty <= TILE_SIZE &&
                   x > 0 && x < nx - 1 && y > 0 && y < ny - 1);

    int prev = 0, curr = 0, next = 0;
    if (valid) {
        prev = A0[offset];
        curr = A0[plane + offset];
        next = A0[2 * plane + offset];
    }

    for (int z = 1; z < nz - 1; z++) {
        smem[ty][tx] = curr;
        __syncthreads();

        if (output) {
            int sum = smem[ty - 1][tx] + smem[ty + 1][tx]
                    + smem[ty][tx - 1] + smem[ty][tx + 1]
                    + prev + next - 6 * smem[ty][tx];
            Anext[z * plane + offset] = sum;
        }
        __syncthreads();

        prev = curr;
        curr = next;
        if (valid && z < nz - 2) {
            next = A0[(z + 2) * plane + offset];
        }
    }
}

// B: No EXPLICIT register tiling along z. For every z, write three
// Global Memory loads in the source. The compiler may still optimize them.
__global__ void stencil_no_register_tiling(const int* __restrict__ A0,
                                           int* __restrict__ Anext,
                                           int nx, int ny, int nz) {
    __shared__ int smem[TILE_SIZE + 2][TILE_SIZE + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * TILE_SIZE + tx;
    int y = blockIdx.y * TILE_SIZE + ty;

    int plane = nx * ny;
    int offset = y * nx + x;
    bool valid = (x < nx && y < ny);
    bool output = (tx > 0 && tx <= TILE_SIZE &&
                   ty > 0 && ty <= TILE_SIZE &&
                   x > 0 && x < nx - 1 && y > 0 && y < ny - 1);

    for (int z = 1; z < nz - 1; z++) {
        int prev = 0, curr = 0, next = 0;
        if (valid) {
            prev = A0[(z - 1) * plane + offset];
            curr = A0[z * plane + offset];
            next = A0[(z + 1) * plane + offset];
        }

        smem[ty][tx] = curr;
        __syncthreads();

        if (output) {
            int sum = smem[ty - 1][tx] + smem[ty + 1][tx]
                    + smem[ty][tx - 1] + smem[ty][tx + 1]
                    + prev + next - 6 * smem[ty][tx];
            Anext[z * plane + offset] = sum;
        }
        __syncthreads();
    }
}

void reference_stencil(const std::vector<int>& input,
                       std::vector<int>& output,
                       int nx, int ny, int nz) {
    int plane = nx * ny;
    for (int z = 1; z < nz - 1; z++) {
        for (int y = 1; y < ny - 1; y++) {
            for (int x = 1; x < nx - 1; x++) {
                int i = z * plane + y * nx + x;
                output[i] = input[i - 1] + input[i + 1]
                          + input[i - nx] + input[i + nx]
                          + input[i - plane] + input[i + plane]
                          - 6 * input[i];
            }
        }
    }
}

bool check_output(const char* label, const std::vector<int>& ref,
                  const std::vector<int>& actual,
                  int nx, int ny) {
    for (std::size_t i = 0; i < ref.size(); ++i) {
        if (ref[i] != actual[i]) {
            int plane = nx * ny;
            std::printf("[FAIL] %s at (x=%zu,y=%zu,z=%zu): "
                        "expected=%d, got=%d\n",
                        label, (i % plane) % nx, (i % plane) / nx,
                        i / plane, ref[i], actual[i]);
            return false;
        }
    }
    std::printf("[PASS] %s: matches CPU reference (including unchanged boundaries)\n",
                label);
    return true;
}

template <typename Launch>
float benchmark_ms(Launch launch, int warmup, int repeats) {
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t begin, end;
    CUDA_CHECK(cudaEventCreate(&begin));
    CUDA_CHECK(cudaEventCreate(&end));

    CUDA_CHECK(cudaEventRecord(begin));
    for (int i = 0; i < repeats; ++i) launch();
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    CUDA_CHECK(cudaGetLastError());

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, begin, end));
    CUDA_CHECK(cudaEventDestroy(begin));
    CUDA_CHECK(cudaEventDestroy(end));
    return elapsed_ms / repeats;
}

int main(int argc, char** argv) {
    int nx = 512, ny = 512, nz = 64, repeats = 100;
    if (argc == 4 || argc == 5) {
        nx = std::atoi(argv[1]);
        ny = std::atoi(argv[2]);
        nz = std::atoi(argv[3]);
        if (argc == 5) repeats = std::atoi(argv[4]);
    } else if (argc != 1) {
        std::fprintf(stderr, "Usage: %s [nx ny nz [repeats]]\n", argv[0]);
        return EXIT_FAILURE;
    }
    if (nx < 3 || ny < 3 || nz < 3 || repeats <= 0 ||
        1LL * nx * ny * nz > INT_MAX) {
        std::fprintf(stderr, "Require nx,ny,nz>=3, repeats>0, nx*ny*nz<=INT_MAX\n");
        return EXIT_FAILURE;
    }

    std::size_t count = static_cast<std::size_t>(nx) * ny * nz;
    std::size_t bytes = count * sizeof(int);
    std::vector<int> host_in(count), host_ref(count, 0);
    std::vector<int> host_reg(count, 0), host_no_reg(count, 0);

    for (std::size_t i = 0; i < count; ++i) {
        std::uint32_t v = static_cast<std::uint32_t>(i) * 1664525u + 1013904223u;
        host_in[i] = static_cast<int>((v >> 16) % 201u) - 100;
    }
    reference_stencil(host_in, host_ref, nx, ny, nz);

    int *d_input = nullptr, *d_reg = nullptr, *d_no_reg = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_input), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_reg), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_no_reg), bytes));
    CUDA_CHECK(cudaMemcpy(d_input, host_in.data(), bytes, cudaMemcpyHostToDevice));
    // Both kernels compute interior only. Boundaries remain zero for verification.
    CUDA_CHECK(cudaMemset(d_reg, 0, bytes));
    CUDA_CHECK(cudaMemset(d_no_reg, 0, bytes));

    dim3 block(TILE_SIZE + 2, TILE_SIZE + 2);
    dim3 grid((nx - 2 + TILE_SIZE - 1) / TILE_SIZE,
              (ny - 2 + TILE_SIZE - 1) / TILE_SIZE);
    auto launch_reg = [&]() {
        stencil_register<<<grid, block>>>(d_input, d_reg, nx, ny, nz);
    };
    auto launch_no_reg = [&]() {
        stencil_no_register_tiling<<<grid, block>>>(d_input, d_no_reg, nx, ny, nz);
    };

    launch_reg();
    launch_no_reg();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_reg.data(), d_reg, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_no_reg.data(), d_no_reg, bytes, cudaMemcpyDeviceToHost));

    std::printf("Input dimensions: %d x %d x %d; repetitions: %d\n",
                nx, ny, nz, repeats);
    bool ok_reg = check_output("register tiling", host_ref, host_reg, nx, ny);
    bool ok_no_reg = check_output("no register tiling", host_ref, host_no_reg, nx, ny);
    if (!ok_reg || !ok_no_reg) {
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_reg));
        CUDA_CHECK(cudaFree(d_no_reg));
        return EXIT_FAILURE;
    }

    const int warmup = 10;
    float no_reg_ms = benchmark_ms(launch_no_reg, warmup, repeats);
    float reg_ms = benchmark_ms(launch_reg, warmup, repeats);
    std::printf("No explicit register tiling: %.6f ms / kernel\n", no_reg_ms);
    std::printf("With register tiling:       %.6f ms / kernel\n", reg_ms);
    std::printf("Speedup (no-reg / reg):     %.3fx\n", no_reg_ms / reg_ms);
    std::printf("Note: compiler optimization and GPU cache can hide differences.\n");

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_reg));
    CUDA_CHECK(cudaFree(d_no_reg));
    return EXIT_SUCCESS;
}