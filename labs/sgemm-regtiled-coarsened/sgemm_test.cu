// ECE408 SGEMM: independent TILE_SZ_A / TILE_SZ_B / TILE_SZ_RATIO sweep.
// No CPU reference computation, no device-to-host copy of C.
// A and C: column-major; B: row-major.
// Build: nvcc -O3 -std=c++17 -arch=sm_89 sgemm_sweep.cu -o sgemm_test
// Run:   ./sgemm_test 4096 --repeat 5 --warmup 2
//        ./sgemm_test 1024 2048 4096 8192 --repeat 5 --full

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t error = (call);                                           \
        if (error != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,     \
                         __LINE__, cudaGetErrorString(error));                \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                     \
    } while (0)

// TILE_SZ_A: threads per block AND output rows per block.
// TILE_SZ_B: outputs per thread (thread coarsening factor).
// TILE_SZ_RATIO: K dimension tile width (independent from A/B).
// Use template parameters rather than #define so all configurations
// are compiled in one binary, with fixed-size register arrays.
template <int TILE_SZ_A, int TILE_SZ_B, int TILE_SZ_RATIO>
__global__ void mysgemm_tuned(int m, int n, int k,
                               const float *__restrict__ A,
                               const float *__restrict__ B,
                               float *__restrict__ C) {
    __shared__ float Bs[TILE_SZ_RATIO][TILE_SZ_B];

    const int tid = threadIdx.x;
    const int row = blockIdx.x * TILE_SZ_A + tid;
    const int colstart = blockIdx.y * TILE_SZ_B;

    float sum[TILE_SZ_B] = {0.0f};

    for (int i = 0; i < k; i += TILE_SZ_RATIO) {
        // All A threads load the K_TILE * COL_TILE values cooperatively.
        // Grid-stride loading is essential when A != B * RATIO.
        for (int idx = tid; idx < TILE_SZ_RATIO * TILE_SZ_B;
             idx += TILE_SZ_A) {
            const int r = idx / TILE_SZ_B;
            const int c = idx % TILE_SZ_B;
            const int global_k = i + r;
            const int global_col = colstart + c;
            Bs[r][c] = (global_k < k && global_col < n)
                           ? B[static_cast<size_t>(global_k) * n + global_col]
                           : 0.0f;
        }
        __syncthreads();

        for (int j = 0; j < TILE_SZ_RATIO; ++j) {
            const int global_k = i + j;
            const float a = (row < m && global_k < k)
                                ? A[static_cast<size_t>(global_k) * m + row]
                                : 0.0f;
#pragma unroll
            for (int c = 0; c < TILE_SZ_B; ++c) {
                sum[c] += a * Bs[j][c];
            }
        }
        __syncthreads();
    }

    if (row < m) {
#pragma unroll
        for (int c = 0; c < TILE_SZ_B; ++c) {
            const int global_col = colstart + c;
            if (global_col < n) {
                C[static_cast<size_t>(global_col) * m + row] = sum[c];
            }
        }
    }
}

__global__ void initialize_input(float *dst, size_t num_elements, unsigned seed) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        // Simple deterministic input data; no costly host-side generation.
        dst[i] = 0.001f * static_cast<float>((i + seed) % 251 + 1);
    }
}

struct Result {
    int dim;
    int tile_a;
    int tile_b;
    int tile_ratio;
    int grid_blocks;
    int registers_per_thread;
    size_t local_bytes_per_thread;
    size_t shared_bytes_per_block;
    double wall_ms;
    double event_median_ms;
    double gflops;
};

static double median(std::vector<float> values) {
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    return values.size() % 2 ? values[mid]
                             : 0.5 * (values[mid - 1] + values[mid]);
}

template <int TA, int TB, int TK>
Result bench_one(int dim, const float *A, const float *B, float *C,
                 int warmup, int repeats) {
    const dim3 block(TA);
    const dim3 grid((dim + TA - 1) / TA, (dim + TB - 1) / TB);

    auto launch = [&]() {
        mysgemm_tuned<TA, TB, TK><<<grid, block>>>(dim, dim, dim, A, B, C);
    };

    // Warm-up before both measurements.
    for (int t = 0; t < warmup; ++t) {
        launch();
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Same scope as the previous eval.cu "Performing GPU sgemm" timer:
    // one kernel launch + cudaDeviceSynchronize(), measured on the host.
    const auto host_start = std::chrono::steady_clock::now();
    launch();
    cudaError_t sync_status = cudaDeviceSynchronize();
    const auto host_stop = std::chrono::steady_clock::now();
    CUDA_CHECK(sync_status);
    CUDA_CHECK(cudaGetLastError());
    const double wall_ms =
        std::chrono::duration<double, std::milli>(host_stop - host_start).count();

    // CUDA-event timing excludes most of the host launch overhead.
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> elapsed_ms;
    elapsed_ms.reserve(repeats);

    for (int t = 0; t < repeats; ++t) {
        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        elapsed_ms.push_back(ms);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    cudaFuncAttributes attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, mysgemm_tuned<TA, TB, TK>));

    const double kernel_ms = median(elapsed_ms);
    const double operations = 2.0 * dim * dim * dim;
    return Result{dim, TA, TB, TK,
                  static_cast<int>(grid.x * grid.y),
                  attributes.numRegs, attributes.localSizeBytes,
                  attributes.sharedSizeBytes, wall_ms, kernel_ms,
                  operations / (kernel_ms * 1.0e6)};
}

struct RunContext {
    int dim;
    const float *A;
    const float *B;
    float *C;
    int warmup;
    int repeats;
};

template <int TA, int TB>
Result dispatch_ratio(int ratio, const RunContext &ctx) {
    switch (ratio) {
        case 4:  return bench_one<TA, TB, 4>(ctx.dim, ctx.A, ctx.B, ctx.C,
                                              ctx.warmup, ctx.repeats);
        case 8:  return bench_one<TA, TB, 8>(ctx.dim, ctx.A, ctx.B, ctx.C,
                                              ctx.warmup, ctx.repeats);
        case 16: return bench_one<TA, TB, 16>(ctx.dim, ctx.A, ctx.B, ctx.C,
                                               ctx.warmup, ctx.repeats);
        default: throw std::invalid_argument("unsupported TILE_SZ_RATIO");
    }
}

template <int TA>
Result dispatch_b(int b, int ratio, const RunContext &ctx) {
    switch (b) {
        case 8:  return dispatch_ratio<TA, 8>(ratio, ctx);
        case 16: return dispatch_ratio<TA, 16>(ratio, ctx);
        case 32: return dispatch_ratio<TA, 32>(ratio, ctx);
        case 64: return dispatch_ratio<TA, 64>(ratio, ctx);
        default: throw std::invalid_argument("unsupported TILE_SZ_B");
    }
}

Result dispatch_a(int a, int b, int ratio, const RunContext &ctx) {
    switch (a) {
        case 64:  return dispatch_b<64>(b, ratio, ctx);
        case 128: return dispatch_b<128>(b, ratio, ctx);
        case 256: return dispatch_b<256>(b, ratio, ctx);
        default: throw std::invalid_argument("unsupported TILE_SZ_A");
    }
}

struct Config { int a, b, ratio; };

std::vector<Config> make_configs(bool full) {
    if (!full) {
        // One-factor-at-a-time experiment around the baseline (128,16,8).
        // Tests TILE_SZ_A, TILE_SZ_B, TILE_SZ_RATIO independently.
        return {{64,16,8}, {128,16,8}, {256,16,8},
                {128,8,8}, {128,32,8}, {128,64,8},
                {128,16,4}, {128,16,16}};
    }
    std::vector<Config> configs;
    for (int a : {64, 128, 256}) {
        for (int b : {8, 16, 32, 64}) {
            for (int ratio : {4, 8, 16}) {
                configs.push_back({a, b, ratio});
            }
        }
    }
    return configs;
}

void usage(const char *program) {
    std::cout << "Usage: " << program
              << " [dim ...] [--repeat N] [--warmup N] [--full] [--csv PATH]\n"
              << "  default: dim=4096, repeat=5, warmup=2, 8 configurations\n"
              << "  --full: test all 36 configurations (A=64/128/256, "
                 "B=8/16/32/64, RATIO=4/8/16)\n"
              << "  examples:\n"
              << "    " << program << " 4096\n"
              << "    " << program << " 2048 4096 --repeat 3 --full\n"
              << "    " << program << " 8192 --repeat 2 --warmup 1\n";
}

int main(int argc, char **argv) {
    std::vector<int> dimensions;
    int repeats = 5;
    int warmup = 2;
    bool full = false;
    std::string csv_path = "sgemm_autotune_results.csv";

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            if (arg == "--help" || arg == "-h") {
                usage(argv[0]);
                return 0;
            } else if (arg == "--full") {
                full = true;
            } else if (arg == "--repeat" && i + 1 < argc) {
                repeats = std::stoi(argv[++i]);
            } else if (arg == "--warmup" && i + 1 < argc) {
                warmup = std::stoi(argv[++i]);
            } else if (arg == "--csv" && i + 1 < argc) {
                csv_path = argv[++i];
            } else if (!arg.empty() && arg[0] == '-') {
                throw std::invalid_argument("unknown argument " + arg);
            } else {
                dimensions.push_back(std::stoi(arg));
            }
        }
        if (repeats < 1 || warmup < 0)
            throw std::invalid_argument("repeat must be >=1 and warmup >=0");
        if (dimensions.empty()) dimensions.push_back(4096);
        for (int dim : dimensions)
            if (dim <= 0) throw std::invalid_argument("dimensions must be positive");

        std::ofstream csv(csv_path);
        if (!csv) throw std::runtime_error("cannot open CSV: " + csv_path);
        csv << "dim,tile_a,tile_b,tile_ratio,grid_blocks,registers_per_thread,"
               "local_bytes_per_thread,shared_bytes_per_block,wall_ms,event_median_ms,gflops\n";
        csv << std::fixed << std::setprecision(6);

        CUDA_CHECK(cudaSetDevice(0));
        const auto configs = make_configs(full);
        std::cout << "Configurations: " << configs.size()
                  << ", repeats: " << repeats << ", warmup: " << warmup << "\n";
        std::cout << "wall_ms: single launch + cudaDeviceSynchronize (like eval.cu)\n"
                  << "event_median_ms: median kernel time measured by CUDA events\n";

        for (int dim : dimensions) {
            const size_t elements = static_cast<size_t>(dim) * dim;
            if (elements > static_cast<size_t>(2147483647)) {
                throw std::invalid_argument("dimension too large for this benchmark");
            }
            const size_t one_matrix_bytes = elements * sizeof(float);
            const size_t required_bytes = 3 * one_matrix_bytes;

            size_t free_bytes = 0, total_bytes = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
            const size_t reserve_bytes = static_cast<size_t>(256) * 1024 * 1024;
            if (required_bytes > free_bytes ||
                free_bytes - required_bytes < reserve_bytes) {
                std::cerr << "Skipping dim=" << dim << ": needs "
                          << required_bytes / (1024.0 * 1024 * 1024)
                          << " GiB + reserve; free="
                          << free_bytes / (1024.0 * 1024 * 1024) << " GiB\n";
                continue;
            }

            float *A = nullptr, *B = nullptr, *C = nullptr;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&A), one_matrix_bytes));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&B), one_matrix_bytes));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&C), one_matrix_bytes));

            const int threads = 256;
            const int blocks = static_cast<int>((elements + threads - 1) / threads);
            initialize_input<<<blocks, threads>>>(A, elements, 17);
            CUDA_CHECK(cudaGetLastError());
            initialize_input<<<blocks, threads>>>(B, elements, 47);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            const RunContext ctx{dim, A, B, C, warmup, repeats};
            std::vector<Result> results;
            std::cout << "\n===== dim=" << dim << " =====\n";
            for (const auto &cfg : configs) {
                std::cout << "Testing A=" << cfg.a << " B=" << cfg.b
                          << " RATIO=" << cfg.ratio << " ... " << std::flush;
                const Result r = dispatch_a(cfg.a, cfg.b, cfg.ratio, ctx);
                results.push_back(r);
                std::cout << std::fixed << std::setprecision(3)
                          << "wall=" << r.wall_ms << " ms, median="
                          << r.event_median_ms << " ms, "
                          << r.gflops << " GFLOPS, "
                          << "regs=" << r.registers_per_thread << "\n";
                csv << r.dim << ',' << r.tile_a << ',' << r.tile_b << ','
                    << r.tile_ratio << ',' << r.grid_blocks << ','
                    << r.registers_per_thread << ',' << r.local_bytes_per_thread
                    << ',' << r.shared_bytes_per_block << ',' << r.wall_ms << ','
                    << r.event_median_ms << ',' << r.gflops << '\n';
                csv.flush();
            }

            std::sort(results.begin(), results.end(), [](const Result &x,
                                                          const Result &y) {
                return x.event_median_ms < y.event_median_ms;
            });
            std::cout << "Best for dim=" << dim << ": A=" << results[0].tile_a
                      << " B=" << results[0].tile_b
                      << " RATIO=" << results[0].tile_ratio
                      << ", median=" << results[0].event_median_ms
                      << " ms, GFLOPS=" << results[0].gflops << "\n";

            CUDA_CHECK(cudaFree(A));
            CUDA_CHECK(cudaFree(B));
            CUDA_CHECK(cudaFree(C));
        }

        std::cout << "\nSaved CSV: " << csv_path << "\n";
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        usage(argv[0]);
        return 1;
    }
    return 0;
}