#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do {                                         \
  cudaError_t error = (call);                                        \
  if (error != cudaSuccess) {                                        \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,          \
            __LINE__, cudaGetErrorString(error));                    \
    std::exit(EXIT_FAILURE);                                         \
  }                                                                  \
} while (0)

// 1. CPU reference: one row at a time.
static void spmvCPU(
    float *out, const int *cols, const int *rows,
    const float *data, const float *vec, int dim) {
  for (int row = 0; row < dim; row++) {
    float sum = 0.0f;
    for (int j = rows[row]; j < rows[row + 1]; j++) {
      sum += data[j] * vec[cols[j]];
    }
    out[row] = sum;
  }
}

// 2. CSR Scalar: one thread per row.
__global__ void spmvCSRScalarKernel(
    float *out, const int *cols, const int *rows,
    const float *data, const float *vec, int dim) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= dim) {
    return;
  }

  float sum = 0.0f;
  for (int j = rows[row]; j < rows[row + 1]; j++) {
    sum += data[j] * vec[cols[j]];
  }
  out[row] = sum;
}

// 3. CSR Warp Shared: a warp per row, with shared-memory reduction.
// Launch geometry is block(32, 8), i.e. 8 warps per block.
__global__ void spmvCSRWarpSharedKernel(
    float *out, const int *cols, const int *rows,
    const float *data, const float *vec, int dim) {
  __shared__ float shared[8][32];

  int row = blockIdx.x * blockDim.y + threadIdx.y;
  int lane = threadIdx.x;
  int warp = threadIdx.y;

  // All 32 lanes of a given warp agree on whether its row is valid.
  if (row >= dim) {
    return;
  }

  float sum = 0.0f;
  for (int j = rows[row] + lane; j < rows[row + 1]; j += 32) {
    sum += data[j] * vec[cols[j]];
  }

  shared[warp][lane] = sum;
  __syncwarp();

  for (int stride = 16; stride > 0; stride /= 2) {
    if (lane < stride) {
      shared[warp][lane] += shared[warp][lane + stride];
    }
    __syncwarp();
  }

  if (lane == 0) {
    out[row] = shared[warp][0];
  }
}

// 4. CSR Warp Shuffle: a warp per row, register shuffle reduction.
__global__ void spmvCSRWarpShuffleKernel(
    float *out, const int *cols, const int *rows,
    const float *data, const float *vec, int dim) {
  int row = blockIdx.x * blockDim.y + threadIdx.y;
  int lane = threadIdx.x;

  if (row >= dim) {
    return;
  }

  float sum = 0.0f;
  for (int j = rows[row] + lane; j < rows[row + 1]; j += 32) {
    sum += data[j] * vec[cols[j]];
  }

  // Every lane in a valid warp participates in all shuffle steps.
  for (int offset = 16; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  }

  if (lane == 0) {
    out[row] = sum;
  }
}

using Launcher = void (*)(float *, const int *, const int *,
                          const float *, const float *, int);

static void launchScalar(
    float *out, const int *cols, const int *rows,
    const float *data, const float *vec, int dim) {
  const int blockSize = 256;
  int gridSize = (dim + blockSize - 1) / blockSize;
  spmvCSRScalarKernel<<<gridSize, blockSize>>>(out, cols, rows, data, vec, dim);
  CUDA_CHECK(cudaGetLastError());
}

static void launchShared(
    float *out, const int *cols, const int *rows,
    const float *data, const float *vec, int dim) {
  dim3 block(32, 8);
  int gridSize = (dim + block.y - 1) / block.y;
  spmvCSRWarpSharedKernel<<<gridSize, block>>>(out, cols, rows, data, vec, dim);
  CUDA_CHECK(cudaGetLastError());
}

static void launchShuffle(
    float *out, const int *cols, const int *rows,
    const float *data, const float *vec, int dim) {
  dim3 block(32, 8);
  int gridSize = (dim + block.y - 1) / block.y;
  spmvCSRWarpShuffleKernel<<<gridSize, block>>>(out, cols, rows, data, vec, dim);
  CUDA_CHECK(cudaGetLastError());
}

// Verify each GPU output against the independently calculated CPU result.
static bool verify(
    const char *name, const std::vector<float> &actual,
    const std::vector<float> &expected) {
  constexpr float absTol = 1e-4f;
  constexpr float relTol = 1e-4f;
  int mismatches = 0;
  float maxAbsError = 0.0f;

  for (size_t row = 0; row < expected.size(); row++) {
    float diff = std::fabs(actual[row] - expected[row]);
    float tolerance = absTol + relTol * std::fabs(expected[row]);
    bool equal = std::isfinite(actual[row]) &&
                 std::isfinite(expected[row]) && diff <= tolerance;

    if (!equal) {
      if (mismatches < 5) {
        printf("    row %zu: actual=%.9g, expected=%.9g, diff=%.9g\n",
               row, actual[row], expected[row], diff);
      }
      mismatches++;
    }
    if (std::isfinite(diff)) {
      maxAbsError = std::max(maxAbsError, diff);
    }
  }

  printf("  %-15s %s  mismatches=%d  maxAbsError=%.9g\n",
         name, mismatches == 0 ? "PASS" : "FAIL",
         mismatches, maxAbsError);
  return mismatches == 0;
}

template <typename T>
static std::vector<T> readRaw(const std::string &path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("Cannot open: " + path);
  }

  long long count = -1;
  if (!(file >> count) || count < 0) {
    throw std::runtime_error("Invalid array length in: " + path);
  }

  std::vector<T> result(static_cast<size_t>(count));
  for (long long i = 0; i < count; i++) {
    if (!(file >> result[static_cast<size_t>(i)])) {
      throw std::runtime_error("Invalid data in: " + path);
    }
  }
  return result;
}

// CUDA events, 5 warm-ups; take median of repeated kernel timings.
static float benchmarkGPU(
    Launcher launch, float *out, const int *cols, const int *rows,
    const float *data, const float *vec, int dim, int repeats) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  for (int i = 0; i < 5; i++) {
    launch(out, cols, rows, data, vec, dim);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> times;
  times.reserve(repeats);
  for (int i = 0; i < repeats; i++) {
    CUDA_CHECK(cudaEventRecord(start));
    launch(out, cols, rows, data, vec, dim);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start, stop));
    times.push_back(elapsedMs);
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  std::sort(times.begin(), times.end());
  return times[times.size() / 2];
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <data_folder> [repeats=100]\n", argv[0]);
    return 1;
  }

  try {
    std::string folder = argv[1];
    if (folder.empty()) {
      throw std::runtime_error("Empty data directory");
    }
    if (folder.back() != '/') {
      folder += '/';
    }

    int repeats = argc >= 3 ? std::atoi(argv[2]) : 100;
    if (repeats <= 0) {
      throw std::runtime_error("repeats must be positive");
    }

    std::vector<int> cols = readRaw<int>(folder + "col.raw");
    std::vector<int> rows = readRaw<int>(folder + "row.raw");
    std::vector<float> data = readRaw<float>(folder + "data.raw");
    std::vector<float> vec = readRaw<float>(folder + "vec.raw");

    int dim = static_cast<int>(vec.size());
    int nnz = static_cast<int>(data.size());
    if (dim <= 0 || cols.size() != data.size() ||
        rows.size() != static_cast<size_t>(dim + 1) ||
        rows.front() != 0 || rows.back() != nnz) {
      throw std::runtime_error("Invalid CSR dimensions");
    }

    int maxRowNNZ = 0;
    for (int row = 0; row < dim; row++) {
      if (rows[row] < 0 || rows[row] > rows[row + 1] ||
          rows[row + 1] > nnz) {
        throw std::runtime_error("Invalid CSR row pointers");
      }
      maxRowNNZ = std::max(maxRowNNZ, rows[row + 1] - rows[row]);
    }
    for (int col : cols) {
      if (col < 0 || col >= dim) {
        throw std::runtime_error("Invalid CSR column index");
      }
    }

    printf("CSR SpMV: dim=%d, nnz=%d, avgNNZ/row=%.2f, maxNNZ/row=%d\n",
           dim, nnz, double(nnz) / dim, maxRowNNZ);
    printf("GPU: warmup=5, repetitions=%d, median\n", repeats);

    // Independent CPU reference, computed before any GPU kernels.
    std::vector<float> cpuOut(dim, 0.0f);
    auto cpuStart = std::chrono::steady_clock::now();
    spmvCPU(cpuOut.data(), cols.data(), rows.data(),
            data.data(), vec.data(), dim);
    auto cpuStop = std::chrono::steady_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(
        cpuStop - cpuStart).count();

    bool passed = true;
    std::ifstream officialCheck(folder + "output.raw");
    if (officialCheck.good()) {
      officialCheck.close();
      std::vector<float> official = readRaw<float>(folder + "output.raw");
      if (official.size() != cpuOut.size()) {
        throw std::runtime_error("output.raw size mismatch");
      }
      printf("CPU vs official output.raw:\n");
      passed = verify("CPU Reference", cpuOut, official) && passed;
    } else {
      printf("No output.raw found; checking GPU versions against CPU only.\n");
    }

    int *dCols = nullptr;
    int *dRows = nullptr;
    float *dData = nullptr;
    float *dVec = nullptr;
    float *dOut = nullptr;

    // Allocate at least one element so an all-zero sparse matrix is valid.
    CUDA_CHECK(cudaMalloc((void **)&dCols,
                          std::max<size_t>(1, cols.size()) * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&dRows, rows.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&dData,
                          std::max<size_t>(1, data.size()) * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&dVec, vec.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&dOut, dim * sizeof(float)));

    if (!cols.empty()) {
      CUDA_CHECK(cudaMemcpy(dCols, cols.data(), cols.size() * sizeof(int),
                            cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(dRows, rows.data(), rows.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    if (!data.empty()) {
      CUDA_CHECK(cudaMemcpy(dData, data.data(), data.size() * sizeof(float),
                            cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(dVec, vec.data(), vec.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    struct Version {
      const char *name;
      Launcher launch;
      float ms;
    };

    Version versions[] = {
      {"CSR Scalar", launchScalar, 0.0f},
      {"Warp Shared", launchShared, 0.0f},
      {"Warp Shuffle", launchShuffle, 0.0f}
    };

    printf("GPU vs independent CPU reference:\n");
    std::vector<float> gpuOut(dim, 0.0f);
    for (auto &version : versions) {
      // Run a separate, fully synchronized correctness check.
      CUDA_CHECK(cudaMemset(dOut, 0, dim * sizeof(float)));
      version.launch(dOut, dCols, dRows, dData, dVec, dim);
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(gpuOut.data(), dOut, dim * sizeof(float),
                            cudaMemcpyDeviceToHost));
      passed = verify(version.name, gpuOut, cpuOut) && passed;

      // Only kernel execution time is counted.
      version.ms = benchmarkGPU(version.launch, dOut, dCols, dRows,
                                dData, dVec, dim, repeats);
    }

    printf("\n%-16s %13s %13s %13s\n",
           "Version", "Time (ms)", "vs Scalar", "GFLOP/s");
    printf("--------------------------------------------------------------\n");
    printf("%-16s %13.5f %13s %13.2f\n",
           "CPU Reference", cpuMs, "-", 2.0 * nnz / (cpuMs * 1e6));

    for (const auto &version : versions) {
      printf("%-16s %13.5f %12.2fx %13.2f\n",
             version.name, version.ms, versions[0].ms / version.ms,
             2.0 * nnz / (version.ms * 1e6));
    }
    printf("\nOverall correctness: %s\n", passed ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(dCols));
    CUDA_CHECK(cudaFree(dRows));
    CUDA_CHECK(cudaFree(dData));
    CUDA_CHECK(cudaFree(dVec));
    CUDA_CHECK(cudaFree(dOut));
    return passed ? 0 : 2;
  } catch (const std::exception &error) {
    fprintf(stderr, "Error: %s\n", error.what());
    return 1;
  }
}