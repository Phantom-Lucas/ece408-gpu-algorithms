
#include "helper.hpp"
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

// ============================================================
// 1. CPU Scatter
// ============================================================

static void s2g_cpu_scatter(uint32_t *in, uint32_t *out, int len) {
  for (int inIdx = 0; inIdx < len; ++inIdx) {
    uint32_t intermediate = outInvariant(in[inIdx]);

    for (int outIdx = 0; outIdx < len; ++outIdx) {
      out[outIdx] += outDependent(intermediate, inIdx, outIdx);
    }
  }
}

// ============================================================
// 2. GPU Scatter - AtomicAdd
// ============================================================

__global__ void s2g_gpu_scatter_kernel(uint32_t *in, uint32_t *out, int len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < len) {
    uint32_t intermediate = outInvariant(in[idx]);

    for (int outIdx = 0; outIdx < len; ++outIdx) {
      atomicAdd(&out[outIdx], outDependent(intermediate, idx, outIdx));
    }
  }
}

// ============================================================
// 3. GPU Gather - Global Memory Baseline
// ============================================================

__global__ void s2g_gpu_gather_kernel(uint32_t *in, uint32_t *out, int len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= len) {
    return;
  }

  uint32_t sum = 0;

  for (int inIdx = 0; inIdx < len; ++inIdx) {
    uint32_t intermediate = outInvariant(in[inIdx]);
    sum += outDependent(intermediate, inIdx, idx);
  }

  out[idx] = sum;
}

// ============================================================
// 4. GPU Kernel Launch
// ============================================================

static void s2g_gpu_scatter(uint32_t *in, uint32_t *out, int len) {
  const int blockSize = 256;
  const int gridSize = (len + blockSize - 1) / blockSize;

  s2g_gpu_scatter_kernel<<<gridSize, blockSize>>>(in, out, len);
  THROW_IF_ERROR(cudaGetLastError());
}

static void s2g_gpu_gather(uint32_t *in, uint32_t *out, int len) {
  const int blockSize = 256;
  const int gridSize = (len + blockSize - 1) / blockSize;

  s2g_gpu_gather_kernel<<<gridSize, blockSize>>>(in, out, len);
  THROW_IF_ERROR(cudaGetLastError());
}

// ============================================================
// 5. GPU Benchmark - CUDA Events
// ============================================================

static float benchmark_gpu(uint32_t *in, uint32_t *out, int len, bool scatter) {
  cudaEvent_t start, stop;

  THROW_IF_ERROR(cudaEventCreate(&start));
  THROW_IF_ERROR(cudaEventCreate(&stop));

  size_t byteCount = len * sizeof(uint32_t);

  // Scatter 使用 atomicAdd，需要将输出初始化为 0
  THROW_IF_ERROR(cudaMemset(out, 0, byteCount));

  // 开始计时，不包含 cudaMemset
  THROW_IF_ERROR(cudaEventRecord(start));

  if (scatter) {
    s2g_gpu_scatter(in, out, len);
  } else {
    s2g_gpu_gather(in, out, len);
  }

  THROW_IF_ERROR(cudaEventRecord(stop));
  THROW_IF_ERROR(cudaEventSynchronize(stop));

  float elapsed = 0.0f;
  THROW_IF_ERROR(cudaEventElapsedTime(&elapsed, start, stop));

  THROW_IF_ERROR(cudaEventDestroy(start));
  THROW_IF_ERROR(cudaEventDestroy(stop));

  return elapsed;
}

// ============================================================
// 6. Evaluation
// ============================================================

static int eval(int inputLength) {
  uint32_t *deviceInput = nullptr;
  uint32_t *deviceOutput = nullptr;

  const std::string conf_info =
    std::string("scatter[len:") + std::to_string(inputLength) + "]";

  INFO("Running " << conf_info);

  auto hostInput = generate_input(inputLength);

  const size_t byteCount = inputLength * sizeof(uint32_t);

  // ----------------------------------------------------------
  // GPU Memory Allocation
  // ----------------------------------------------------------

  THROW_IF_ERROR(cudaMalloc((void **)&deviceInput, byteCount));
  THROW_IF_ERROR(cudaMalloc((void **)&deviceOutput, byteCount));

  THROW_IF_ERROR(cudaMemcpy(
    deviceInput,
    hostInput.data(),
    byteCount,
    cudaMemcpyHostToDevice
  ));

  // ----------------------------------------------------------
  // CPU Scatter
  // ----------------------------------------------------------

  std::vector<uint32_t> expected(inputLength, 0);

  auto cpuStart = std::chrono::steady_clock::now();

  s2g_cpu_scatter(hostInput.data(), expected.data(), inputLength);

  auto cpuStop = std::chrono::steady_clock::now();

  double cpuMs = std::chrono::duration<double, std::milli>(
    cpuStop - cpuStart
  ).count();

  // ----------------------------------------------------------
  // GPU Scatter
  // ----------------------------------------------------------

  float scatterMs = benchmark_gpu(
    deviceInput,
    deviceOutput,
    inputLength,
    true
  );

  std::vector<uint32_t> scatterOutput(inputLength);

  THROW_IF_ERROR(cudaMemcpy(
    scatterOutput.data(),
    deviceOutput,
    byteCount,
    cudaMemcpyDeviceToHost
  ));

  // ----------------------------------------------------------
  // GPU Gather
  // ----------------------------------------------------------

  float gatherMs = benchmark_gpu(
    deviceInput,
    deviceOutput,
    inputLength,
    false
  );

  std::vector<uint32_t> gatherOutput(inputLength);

  THROW_IF_ERROR(cudaMemcpy(
    gatherOutput.data(),
    deviceOutput,
    byteCount,
    cudaMemcpyDeviceToHost
  ));

  // ----------------------------------------------------------
  // Correctness
  // ----------------------------------------------------------

  verify(expected, scatterOutput);
  verify(expected, gatherOutput);

  // ----------------------------------------------------------
  // Performance Comparison
  // ----------------------------------------------------------

  printf(
    "len=%6d | CPU=%9.3f ms | Scatter=%9.3f ms | Gather=%9.3f ms | CPU/Scatter=%7.2fx | Scatter/Gather=%7.2fx\n",
    inputLength,
    cpuMs,
    scatterMs,
    gatherMs,
    cpuMs / scatterMs,
    scatterMs / gatherMs
  );

  // ----------------------------------------------------------
  // Free Memory
  // ----------------------------------------------------------

  THROW_IF_ERROR(cudaFree(deviceInput));
  THROW_IF_ERROR(cudaFree(deviceOutput));

  return 0;
}

// ============================================================
// 7. Tests
// ============================================================

TEST_CASE("Scatter", "[scatter]") {
  SECTION("[inputSize:1024]") {
    eval(1024);
  }
  SECTION("[inputSize:2048]") {
    eval(2048);
  }
  SECTION("[inputSize:2047]") {
    eval(2047);
  }
  SECTION("[inputSize:2049]") {
    eval(2049);
  }
  SECTION("[inputSize:9101]") {
    eval(9101);
  }
  SECTION("[inputSize:9910]") {
    eval(9910);
  }
  SECTION("[inputSize:8192]") {
    eval(8192);
  }
  SECTION("[inputSize:8193]") {
    eval(8193);
  }
  SECTION("[inputSize:8191]") {
    eval(8191);
  }
  SECTION("[inputSize:16191]") {
    eval(16191);
  }
}
