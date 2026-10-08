
#include "helper.hpp"
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

// ============================================================
// 1. CPU Baseline
// ============================================================

static void s2g_cpu_gather(uint32_t *in, uint32_t *out, int len) {
  for (int outIdx = 0; outIdx < len; ++outIdx) {
    uint32_t out_reg = 0;

    for (int inIdx = 0; inIdx < len; ++inIdx) {
      uint32_t intermediate = outInvariant(in[inIdx]);
      out_reg += outDependent(intermediate, inIdx, outIdx);
    }

    out[outIdx] += out_reg;
  }
}

// ============================================================
// 2. GPU Baseline：直接访问 Global Memory
// ============================================================

__global__ void s2g_gpu_gather_baseline_kernel(uint32_t *in, uint32_t *out, int len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= len) {
    return;
  }

  uint32_t sum = 0;

  for (int j = 0; j < len; j++) {
    uint32_t intermediate = outInvariant(in[j]);
    sum += outDependent(intermediate, j, idx);
  }

  out[idx] = sum;
}

// ============================================================
// 3. GPU Optimized：Shared Memory + 预计算
// ============================================================

__global__ void s2g_gpu_gather_kernel(uint32_t *in, uint32_t *out, int len) {
  __shared__ uint32_t shared_in[256];

  int tx = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tx;

  uint32_t sum = 0;

  for (int i = 0; i < len; i += blockDim.x) {
    int count = min(blockDim.x, len - i);

    // Cooperative loading + outInvariant
    if (tx < count) {
      shared_in[tx] = outInvariant(in[i + tx]);
    }

    __syncthreads();

    // Gather computation
    if (idx < len) {
      for (int j = 0; j < count; j++) {
        sum += outDependent(shared_in[j], i + j, idx);
      }
    }

    // 防止下一轮覆盖 Shared Memory
    __syncthreads();
  }

  if (idx < len) {
    out[idx] = sum;
  }
}

// ============================================================
// 4. GPU Kernel Launch
// ============================================================

static void s2g_gpu_gather_baseline(uint32_t *in, uint32_t *out, int len) {
  const int blockSize = 256;
  const int gridSize = (len + blockSize - 1) / blockSize;

  s2g_gpu_gather_baseline_kernel<<<gridSize, blockSize>>>(in, out, len);
  THROW_IF_ERROR(cudaGetLastError());
}

static void s2g_gpu_gather(uint32_t *in, uint32_t *out, int len) {
  const int blockSize = 256;
  const int gridSize = (len + blockSize - 1) / blockSize;

  s2g_gpu_gather_kernel<<<gridSize, blockSize>>>(in, out, len);
  THROW_IF_ERROR(cudaGetLastError());
}

// ============================================================
// 5. GPU Benchmark：CUDA Event
// ============================================================

static float benchmark_gpu(uint32_t *in, uint32_t *out, int len, bool optimized) {
  const int repeats = 5;

  cudaEvent_t start, stop;

  THROW_IF_ERROR(cudaEventCreate(&start));
  THROW_IF_ERROR(cudaEventCreate(&stop));

  // Warm-up
  if (optimized) {
    s2g_gpu_gather(in, out, len);
  } else {
    s2g_gpu_gather_baseline(in, out, len);
  }

  THROW_IF_ERROR(cudaDeviceSynchronize());

  // Start timing
  THROW_IF_ERROR(cudaEventRecord(start));

  for (int i = 0; i < repeats; i++) {
    if (optimized) {
      s2g_gpu_gather(in, out, len);
    } else {
      s2g_gpu_gather_baseline(in, out, len);
    }
  }

  THROW_IF_ERROR(cudaEventRecord(stop));
  THROW_IF_ERROR(cudaEventSynchronize(stop));

  float elapsed = 0.0f;

  THROW_IF_ERROR(cudaEventElapsedTime(&elapsed, start, stop));

  THROW_IF_ERROR(cudaEventDestroy(start));
  THROW_IF_ERROR(cudaEventDestroy(stop));

  return elapsed / repeats;
}

// ============================================================
// 6. Evaluation：CPU vs GPU Baseline vs GPU Optimized
// ============================================================

static int eval(int inputLength) {
  uint32_t *deviceInput = nullptr;
  uint32_t *deviceOutput = nullptr;

  const std::string conf_info =
    std::string("gather[len:") + std::to_string(inputLength) + "]";

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

  THROW_IF_ERROR(cudaMemset(deviceOutput, 0, byteCount));

  // ----------------------------------------------------------
  // CPU Baseline
  // ----------------------------------------------------------

  std::vector<uint32_t> expected(inputLength, 0);

  auto cpuStart = std::chrono::steady_clock::now();

  s2g_cpu_gather(hostInput.data(), expected.data(), inputLength);

  auto cpuStop = std::chrono::steady_clock::now();

  double cpuMs = std::chrono::duration<double, std::milli>(
    cpuStop - cpuStart
  ).count();

  // ----------------------------------------------------------
  // GPU Baseline
  // ----------------------------------------------------------

  float baselineMs = benchmark_gpu(
    deviceInput,
    deviceOutput,
    inputLength,
    false
  );

  std::vector<uint32_t> baselineOutput(inputLength);

  THROW_IF_ERROR(cudaMemcpy(
    baselineOutput.data(),
    deviceOutput,
    byteCount,
    cudaMemcpyDeviceToHost
  ));

  // ----------------------------------------------------------
  // GPU Optimized
  // ----------------------------------------------------------

  float optimizedMs = benchmark_gpu(
    deviceInput,
    deviceOutput,
    inputLength,
    true
  );

  std::vector<uint32_t> optimizedOutput(inputLength);

  THROW_IF_ERROR(cudaMemcpy(
    optimizedOutput.data(),
    deviceOutput,
    byteCount,
    cudaMemcpyDeviceToHost
  ));

  // ----------------------------------------------------------
  // Correctness Verification
  // ----------------------------------------------------------

  verify(expected, baselineOutput);
  verify(expected, optimizedOutput);

  // ----------------------------------------------------------
  // Performance Report
  // ----------------------------------------------------------

  printf(
    "len=%6d | CPU=%9.3f ms | Baseline=%9.3f ms | Optimized=%9.3f ms | GPU Speedup=%7.2fx | CPU/GPU=%7.2fx\n",
    inputLength,
    cpuMs,
    baselineMs,
    optimizedMs,
    baselineMs / optimizedMs,
    cpuMs / optimizedMs
  );

  // ----------------------------------------------------------
  // Free GPU Memory
  // ----------------------------------------------------------

  THROW_IF_ERROR(cudaFree(deviceInput));
  THROW_IF_ERROR(cudaFree(deviceOutput));

  return 0;
}

// ============================================================
// 7. Tests
// ============================================================

TEST_CASE("Gather", "[gather]") {
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
