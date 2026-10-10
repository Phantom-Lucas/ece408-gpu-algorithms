// Example Nsight Compute run (Uniform CSR, one warm GPU launch):
//   sudo "$(command -v ncu)" --profile-from-start off --launch-count 1 \
//     --section SpeedOfLight --section MemoryWorkloadAnalysis \
//     ./spmv_profile --case Uniform --profile csr
// Repeat with --case Extreme and --profile jds, etc.
//
// ECE408 SpMV benchmark (adds targeted Nsight Compute profiling): CSR (warp-per-row) vs JDS (thread-per-row).
// Four synthetic matrices have the SAME dimensions and total NNZ,
// but different NNZ-per-row distributions.
//
// Build (RTX 4050 / sm_89):
//   nvcc -O3 -std=c++17 -arch=sm_89 spmv_benchmark.cu -o spmv_benchmark
// Run:
//   ./spmv_benchmark
//   ./spmv_benchmark --rows 4096 --cols 4096 --repeats 100 --trials 3
//
// Kernel-only timing with CUDA events. JDS conversion / host-to-device copies
// are excluded from timing and reported separately where relevant.

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#define CUDA_CHECK(expr)                                                        \
  do {                                                                          \
    cudaError_t err__ = (expr);                                                  \
    if (err__ != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ':' << __LINE__             \
                << " (" << #expr << "): " << cudaGetErrorString(err__)      \
                << std::endl;                                                  \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

struct Options {
  int rows = 65536;
  int cols = 65536;
  int repeats = 500;
  int trials = 7;
  int warmup = 50;
  unsigned seed = 20261010;
  std::string output = "spmv_results.csv";
  int selectedCase = -1; // -1: run all four cases
  std::string profileKernel; // empty: CUDA event benchmark; csr/jds: ncu single kernel
};

struct CSRMatrix {
  int rows = 0;
  int cols = 0;
  std::vector<int> rowPtr; // rows + 1
  std::vector<int> colIdx; // NNZ
  std::vector<float> data; // NNZ
};

struct JDSMatrix {
  std::vector<int> colStart; // maxRowNNZ + 1; final element = total NNZ
  std::vector<int> colIdx;   // NNZ
  std::vector<int> rowPerm;  // sorted row -> original row
  std::vector<int> rowNNZ;   // NNZ for each sorted row
  std::vector<float> data;   // NNZ
};

struct NNZStats {
  double mean = 0.0;
  double stddev = 0.0; // population standard deviation
  int minValue = 0;
  int maxValue = 0;
  int emptyRows = 0;
};

struct Result {
  std::string name;
  NNZStats stats;
  std::size_t nnz = 0;
  double csrUs = 0.0;
  double jdsUs = 0.0;
  double csrGflops = 0.0;
  double jdsGflops = 0.0;
  double conversionMs = 0.0;
};

template <typename T>
class DeviceArray {
 public:
  explicit DeviceArray(std::size_t count) : count_(count) {
    if (count_ != 0) {
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&ptr_),
                            count_ * sizeof(T)));
    }
  }

  DeviceArray(const DeviceArray &) = delete;
  DeviceArray &operator=(const DeviceArray &) = delete;

  ~DeviceArray() {
    if (ptr_ != nullptr) cudaFree(ptr_);
  }

  T *get() const { return ptr_; }

  void copyFrom(const std::vector<T> &src) {
    if (src.size() != count_) throw std::runtime_error("H2D size mismatch");
    if (count_ != 0) {
      CUDA_CHECK(cudaMemcpy(ptr_, src.data(), count_ * sizeof(T),
                            cudaMemcpyHostToDevice));
    }
  }

  void copyTo(std::vector<T> &dst) const {
    if (dst.size() != count_) throw std::runtime_error("D2H size mismatch");
    if (count_ != 0) {
      CUDA_CHECK(cudaMemcpy(dst.data(), ptr_, count_ * sizeof(T),
                            cudaMemcpyDeviceToHost));
    }
  }

 private:
  T *ptr_ = nullptr;
  std::size_t count_;
};

class CudaEvent {
 public:
  CudaEvent() { CUDA_CHECK(cudaEventCreate(&event_)); }
  ~CudaEvent() { cudaEventDestroy(event_); }
  CudaEvent(const CudaEvent &) = delete;
  CudaEvent &operator=(const CudaEvent &) = delete;
  cudaEvent_t get() const { return event_; }

 private:
  cudaEvent_t event_{};
};

// Original ECE408 code: one warp (32 lanes) calculates one CSR row.
__global__ void spmvCSRKernel(float *out, const int *matCols,
                              const int *matRows, const float *matData,
                              const float *vec, int dim) {
  int row = blockIdx.x * blockDim.y + threadIdx.y;
  int lane = threadIdx.x;
  if (row >= dim) return; // warp-uniform with block(32,8)

  float sum = 0.0f;
  for (int j = matRows[row] + lane; j < matRows[row + 1]; j += 32) {
    sum += matData[j] * vec[matCols[j]];
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  }
  if (lane == 0) out[row] = sum;
}

// Original ECE408 code: one thread calculates one sorted JDS row.
__global__ void spmvJDSKernel(float *out, const int *matColStart,
                              const int *matCols, const int *matRowPerm,
                              const int *matRows, const float *matData,
                              const float *vec, int dim) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= dim) return;

  float sum = 0.0f;
  for (int j = 0; j < matRows[row]; j++) {
    int idx = matColStart[j] + row;
    sum += matData[idx] * vec[matCols[idx]];
  }
  out[matRowPerm[row]] = sum;
}

// Return NNZ-per-row counts; each case totals 32 * rows NNZ.
std::vector<int> makeRowCounts(int rows, int caseIndex, std::mt19937 &rng) {
  std::vector<int> counts(rows, 32);
  int low = 32, high = 32;
  switch (caseIndex) {
    case 0: low = 32; high = 32; break;
    case 1: low = 16; high = 48; break;
    case 2: low = 4;  high = 60; break;
    case 3: low = 0;  high = 64; break;
    default: throw std::runtime_error("unknown case");
  }
  for (int i = 0; i < rows / 2; ++i) counts[i] = low;
  for (int i = rows / 2; i < rows; ++i) counts[i] = high;
  std::shuffle(counts.begin(), counts.end(), rng);
  return counts;
}

NNZStats analyzeNNZ(const std::vector<int> &counts) {
  NNZStats s;
  const int rows = static_cast<int>(counts.size());
  const std::int64_t sum = std::accumulate(
      counts.begin(), counts.end(), std::int64_t{0});
  s.mean = static_cast<double>(sum) / rows;
  s.minValue = *std::min_element(counts.begin(), counts.end());
  s.maxValue = *std::max_element(counts.begin(), counts.end());
  double sq = 0.0;
  for (int n : counts) {
    double diff = n - s.mean;
    sq += diff * diff;
    if (n == 0) ++s.emptyRows;
  }
  s.stddev = std::sqrt(sq / rows);
  return s;
}

CSRMatrix generateCSR(int rows, int cols, const std::vector<int> &counts,
                      std::mt19937 &rng) {
  CSRMatrix a;
  a.rows = rows;
  a.cols = cols;
  a.rowPtr.resize(rows + 1, 0);
  for (int r = 0; r < rows; ++r) {
    a.rowPtr[r + 1] = a.rowPtr[r] + counts[r];
  }
  const std::size_t nnz = static_cast<std::size_t>(a.rowPtr.back());
  a.colIdx.resize(nnz);
  a.data.resize(nnz);

  std::uniform_int_distribution<int> colDist(0, cols - 1);
  std::uniform_real_distribution<float> valueDist(-1.0f, 1.0f);
  std::vector<std::pair<int, float>> entries;
  entries.reserve(64);

  for (int r = 0; r < rows; ++r) {
    entries.clear();
    const int base = colDist(rng);
    const int stride = colDist(rng) | 1; // coprime to power-of-two cols
    for (int k = 0; k < counts[r]; ++k) {
      // For arbitrary cols, explicitly reject repeated columns below.
      const int col = static_cast<int>(
          (static_cast<std::int64_t>(base) +
           static_cast<std::int64_t>(stride) * k) % cols);
      entries.emplace_back(col, valueDist(rng));
    }
    // For non-power-of-two cols, unusual strides could produce duplicates.
    // Replace duplicate column indices deterministically if necessary.
    std::vector<int> used;
    used.reserve(entries.size());
    for (auto &entry : entries) {
      int col = entry.first;
      while (std::find(used.begin(), used.end(), col) != used.end()) {
        col = (col + 1) % cols;
      }
      entry.first = col;
      used.push_back(col);
    }
    std::sort(entries.begin(), entries.end(),
              [](const std::pair<int, float> &x,
                 const std::pair<int, float> &y) {
                return x.first < y.first;
              });
    for (int k = 0; k < counts[r]; ++k) {
      const int pos = a.rowPtr[r] + k;
      a.colIdx[pos] = entries[k].first;
      a.data[pos] = entries[k].second;
    }
  }
  return a;
}

// Convert the SAME CSR matrix into JDS (the conversion is not GPU-timed).
JDSMatrix csrToJDS(const CSRMatrix &a) {
  JDSMatrix jds;
  const int rows = a.rows;
  const int nnz = a.rowPtr[rows];
  jds.rowPerm.resize(rows);
  std::iota(jds.rowPerm.begin(), jds.rowPerm.end(), 0);
  std::stable_sort(jds.rowPerm.begin(), jds.rowPerm.end(),
                   [&](int r1, int r2) {
                     return (a.rowPtr[r1 + 1] - a.rowPtr[r1]) >
                            (a.rowPtr[r2 + 1] - a.rowPtr[r2]);
                   });

  jds.rowNNZ.resize(rows);
  int maxRowNNZ = 0;
  for (int sortedRow = 0; sortedRow < rows; ++sortedRow) {
    int oldRow = jds.rowPerm[sortedRow];
    int rowNNZ = a.rowPtr[oldRow + 1] - a.rowPtr[oldRow];
    jds.rowNNZ[sortedRow] = rowNNZ;
    maxRowNNZ = std::max(maxRowNNZ, rowNNZ);
  }

  // Each jagged diagonal j contains all rows with rowNNZ > j.
  std::vector<int> histogram(maxRowNNZ + 1, 0);
  for (int count : jds.rowNNZ) ++histogram[count];
  jds.colStart.assign(maxRowNNZ + 1, 0);
  int activeRows = rows - histogram[0];
  for (int j = 0; j < maxRowNNZ; ++j) {
    jds.colStart[j + 1] = jds.colStart[j] + activeRows;
    activeRows -= histogram[j + 1];
  }
  if (jds.colStart.back() != nnz)
    throw std::runtime_error("JDS offset construction error");

  jds.colIdx.resize(nnz);
  jds.data.resize(nnz);
  for (int sortedRow = 0; sortedRow < rows; ++sortedRow) {
    const int originalRow = jds.rowPerm[sortedRow];
    const int csrStart = a.rowPtr[originalRow];
    for (int j = 0; j < jds.rowNNZ[sortedRow]; ++j) {
      const int jdsPos = jds.colStart[j] + sortedRow;
      jds.colIdx[jdsPos] = a.colIdx[csrStart + j];
      jds.data[jdsPos] = a.data[csrStart + j];
    }
  }
  return jds;
}

std::vector<float> cpuReference(const CSRMatrix &a,
                                const std::vector<float> &x) {
  std::vector<float> out(a.rows, 0.0f);
  for (int r = 0; r < a.rows; ++r) {
    double sum = 0.0;
    for (int j = a.rowPtr[r]; j < a.rowPtr[r + 1]; ++j) {
      sum += static_cast<double>(a.data[j]) * x[a.colIdx[j]];
    }
    out[r] = static_cast<float>(sum);
  }
  return out;
}

void verify(const char *algorithm, const std::vector<float> &actual,
            const std::vector<float> &reference) {
  const double atol = 1e-4;
  const double rtol = 1e-4;
  double maxAbsError = 0.0;
  int errors = 0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    double got = actual[i];
    double expected = reference[i];
    double diff = std::abs(got - expected);
    maxAbsError = std::max(maxAbsError, diff);
    if (!std::isfinite(got) || diff > atol + rtol * std::abs(expected)) {
      if (errors < 5) {
        std::cerr << "  " << algorithm << " mismatch row " << i
                  << ": got " << got << ", expected " << expected
                  << ", abs error " << diff << std::endl;
      }
      ++errors;
    }
  }
  if (errors != 0) {
    throw std::runtime_error(std::string(algorithm) + " correctness FAILED (" +
                             std::to_string(errors) + " mismatches)");
  }
  std::cout << "  " << algorithm << " correctness: PASS (max abs error "
            << maxAbsError << ")" << std::endl;
}

// Returns average microseconds per launch from CUDA event elapsed time.
// Repeated launches are bracketed by one pair of events to reduce noise.
template <typename Launcher>
double timeKernels(Launcher launch, int warmup, int repeats) {
  for (int i = 0; i < warmup; ++i) launch();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CudaEvent begin, end;
  CUDA_CHECK(cudaEventRecord(begin.get()));
  for (int i = 0; i < repeats; ++i) launch();
  CUDA_CHECK(cudaEventRecord(end.get()));
  CUDA_CHECK(cudaEventSynchronize(end.get()));
  CUDA_CHECK(cudaGetLastError());
  float totalMs = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&totalMs, begin.get(), end.get()));
  return static_cast<double>(totalMs) * 1000.0 / repeats;
}

double median(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  const std::size_t mid = samples.size() / 2;
  if (samples.size() % 2 == 1) return samples[mid];
  return (samples[mid - 1] + samples[mid]) / 2.0;
}

Result runCase(const std::string &caseName, int caseIndex,
               const Options &opts) {
  std::mt19937 rng(opts.seed + static_cast<unsigned>(caseIndex) * 1009u);
  std::vector<int> counts = makeRowCounts(opts.rows, caseIndex, rng);
  NNZStats stats = analyzeNNZ(counts);
  CSRMatrix csr = generateCSR(opts.rows, opts.cols, counts, rng);

  auto conversionStart = std::chrono::steady_clock::now();
  JDSMatrix jds = csrToJDS(csr);
  auto conversionEnd = std::chrono::steady_clock::now();
  double conversionMs = std::chrono::duration<double, std::milli>(
                            conversionEnd - conversionStart).count();

  std::mt19937 vecRng(opts.seed + 777u); // same vector across all cases
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> x(opts.cols);
  for (float &v : x) v = dist(vecRng);
  std::vector<float> reference = cpuReference(csr, x);

  std::cout << "\n[" << caseName << "] rows=" << opts.rows
            << " cols=" << opts.cols << " NNZ=" << csr.data.size()
            << " avg=" << stats.mean << " stddev=" << stats.stddev
            << " min=" << stats.minValue << " max=" << stats.maxValue
            << " empty=" << stats.emptyRows << std::endl;

  DeviceArray<int> dCSRRows(csr.rowPtr.size());
  DeviceArray<int> dCSRCols(csr.colIdx.size());
  DeviceArray<float> dCSRData(csr.data.size());
  DeviceArray<int> dJDSColStart(jds.colStart.size());
  DeviceArray<int> dJDSCols(jds.colIdx.size());
  DeviceArray<int> dJDSRowPerm(jds.rowPerm.size());
  DeviceArray<int> dJDSRows(jds.rowNNZ.size());
  DeviceArray<float> dJDSData(jds.data.size());
  DeviceArray<float> dVec(x.size());
  DeviceArray<float> dCSROut(opts.rows);
  DeviceArray<float> dJDSOut(opts.rows);

  dCSRRows.copyFrom(csr.rowPtr);
  dCSRCols.copyFrom(csr.colIdx);
  dCSRData.copyFrom(csr.data);
  dJDSColStart.copyFrom(jds.colStart);
  dJDSCols.copyFrom(jds.colIdx);
  dJDSRowPerm.copyFrom(jds.rowPerm);
  dJDSRows.copyFrom(jds.rowNNZ);
  dJDSData.copyFrom(jds.data);
  dVec.copyFrom(x);

  auto launchCSR = [&]() {
    dim3 block(32, 8);
    int gridSize = (opts.rows + block.y - 1) / block.y;
    spmvCSRKernel<<<gridSize, block>>>(dCSROut.get(), dCSRCols.get(),
                                       dCSRRows.get(), dCSRData.get(),
                                       dVec.get(), opts.rows);
  };
  auto launchJDS = [&]() {
    dim3 block(256);
    int gridSize = (opts.rows + block.x - 1) / block.x;
    spmvJDSKernel<<<gridSize, block>>>(dJDSOut.get(), dJDSColStart.get(),
                                       dJDSCols.get(), dJDSRowPerm.get(),
                                       dJDSRows.get(), dJDSData.get(),
                                       dVec.get(), opts.rows);
  };

  // Verify BOTH implementations on the identical matrix before benchmarking.
  launchCSR();
  launchJDS();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> yCSR(opts.rows), yJDS(opts.rows);
  dCSROut.copyTo(yCSR);
  dJDSOut.copyTo(yJDS);
  verify("CSR", yCSR, reference);
  verify("JDS", yJDS, reference);

  // ncu mode: warm up the selected kernel, then profile ONE marked launch.
  // Run with ncu --profile-from-start off so validation/warm-up are excluded.
  if (!opts.profileKernel.empty()) {
    auto selectedLaunch = [&]() {
      if (opts.profileKernel == "csr") launchCSR();
      else launchJDS();
    };
    for (int i = 0; i < opts.warmup; ++i) selectedLaunch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "  Profiling one " << opts.profileKernel
              << " launch (after " << opts.warmup << " warm-ups)..." << std::endl;
    CUDA_CHECK(cudaProfilerStart());
    selectedLaunch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaProfilerStop());
    Result result;
    result.name = caseName;
    result.stats = stats;
    result.nnz = csr.data.size();
    return result;
  }

  std::vector<double> csrSamples, jdsSamples;
  csrSamples.reserve(opts.trials);
  jdsSamples.reserve(opts.trials);
  for (int t = 0; t < opts.trials; ++t) {
    // Alternate measurement order to reduce order-dependent clock bias.
    if (t % 2 == 0) {
      csrSamples.push_back(timeKernels(launchCSR, opts.warmup, opts.repeats));
      jdsSamples.push_back(timeKernels(launchJDS, opts.warmup, opts.repeats));
    } else {
      jdsSamples.push_back(timeKernels(launchJDS, opts.warmup, opts.repeats));
      csrSamples.push_back(timeKernels(launchCSR, opts.warmup, opts.repeats));
    }
  }

  Result result;
  result.name = caseName;
  result.stats = stats;
  result.nnz = csr.data.size();
  result.csrUs = median(csrSamples);
  result.jdsUs = median(jdsSamples);
  // 2 floating-point operations per nonzero, duration is in us.
  result.csrGflops = (2.0 * result.nnz) / (result.csrUs * 1000.0);
  result.jdsGflops = (2.0 * result.nnz) / (result.jdsUs * 1000.0);
  result.conversionMs = conversionMs;
  std::cout << std::fixed << std::setprecision(3)
            << "  CSR=" << result.csrUs << " us, JDS=" << result.jdsUs
            << " us, CSR/JDS=" << (result.csrUs / result.jdsUs)
            << "x; conversion=" << result.conversionMs << " ms"
            << std::endl;
  return result;
}

int positiveInt(const std::string &value, const char *name) {
  try {
    std::size_t parsed = 0;
    long long n = std::stoll(value, &parsed);
    if (parsed != value.size() || n <= 0 ||
        n > std::numeric_limits<int>::max()) throw std::invalid_argument("range");
    return static_cast<int>(n);
  } catch (...) {
    throw std::runtime_error(std::string("invalid ") + name + ": " + value);
  }
}

Options parseArgs(int argc, char **argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: " << argv[0]
                << " [--rows N] [--cols N] [--repeats N] [--trials N]"
                << " [--warmup N] [--seed N] [--output FILE]"
                << " [--case Uniform|Moderate|High|Extreme] [--profile csr|jds]\n"
                << "Rows must be even and >= 2; columns must be >= 64.\n"
                << "Use --case and --profile together for ncu profiling.\n";
      std::exit(0);
    }
    if (i + 1 >= argc) throw std::runtime_error("missing value for " + arg);
    std::string value = argv[++i];
    if (arg == "--rows") o.rows = positiveInt(value, "rows");
    else if (arg == "--cols") o.cols = positiveInt(value, "cols");
    else if (arg == "--repeats") o.repeats = positiveInt(value, "repeats");
    else if (arg == "--trials") o.trials = positiveInt(value, "trials");
    else if (arg == "--warmup") o.warmup = positiveInt(value, "warmup");
    else if (arg == "--seed") o.seed = static_cast<unsigned>(positiveInt(value, "seed"));
    else if (arg == "--output") o.output = value;
    else if (arg == "--case") {
      const std::vector<std::string> names = {"Uniform", "Moderate", "High", "Extreme"};
      auto it = std::find(names.begin(), names.end(), value);
      if (it == names.end())
        throw std::runtime_error("--case must be Uniform, Moderate, High or Extreme");
      o.selectedCase = static_cast<int>(it - names.begin());
    }
    else if (arg == "--profile") {
      if (value != "csr" && value != "jds")
        throw std::runtime_error("--profile must be csr or jds");
      o.profileKernel = value;
    }
    else throw std::runtime_error("unknown option " + arg);
  }
  if (!o.profileKernel.empty() && o.selectedCase < 0)
    throw std::runtime_error("--profile requires --case to select one matrix");
  if (o.rows < 2 || (o.rows % 2) != 0)
    throw std::runtime_error("--rows must be even and >= 2");
  if (o.cols < 64) throw std::runtime_error("--cols must be >= 64");
  if (static_cast<std::int64_t>(o.rows) * 32 >
      std::numeric_limits<int>::max())
    throw std::runtime_error("rows too large for 32-bit NNZ indices");
  return o;
}

void writeCSV(const std::string &filename, const std::vector<Result> &results,
              const Options &opts) {
  std::ofstream f(filename);
  if (!f) throw std::runtime_error("cannot write " + filename);
  f << "case,rows,cols,nnz,avg_nnz_per_row,stddev_nnz_per_row,"
       "min_row_nnz,max_row_nnz,empty_rows,csr_us,jds_us,"
       "csr_gflops,jds_gflops,csr_over_jds_speedup,jds_conversion_ms\n";
  f << std::fixed << std::setprecision(6);
  for (const Result &r : results) {
    f << r.name << ',' << opts.rows << ',' << opts.cols << ',' << r.nnz
      << ',' << r.stats.mean << ',' << r.stats.stddev << ','
      << r.stats.minValue << ',' << r.stats.maxValue << ','
      << r.stats.emptyRows << ',' << r.csrUs << ',' << r.jdsUs
      << ',' << r.csrGflops << ',' << r.jdsGflops << ','
      << r.csrUs / r.jdsUs << ',' << r.conversionMs << '\n';
  }
  if (!f) throw std::runtime_error("failed writing " + filename);
}

int main(int argc, char **argv) {
  try {
    const Options opts = parseArgs(argc, argv);
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "Device: " << prop.name << " (sm_"
              << prop.major << prop.minor << ")\n"
              << "Matrix: " << opts.rows << " x " << opts.cols
              << ", target NNZ: " << static_cast<std::int64_t>(opts.rows) * 32
              << ", warmup=" << opts.warmup
              << ", repeats=" << opts.repeats
              << ", trials=" << opts.trials << std::endl;

    const std::vector<std::string> names = {
        "Uniform", "Moderate", "High", "Extreme"};
    std::vector<Result> results;
    results.reserve(names.size());
    for (int i = 0; i < static_cast<int>(names.size()); ++i) {
      if (opts.selectedCase >= 0 && i != opts.selectedCase) continue;
      results.push_back(runCase(names[i], i, opts));
    }
    if (!opts.profileKernel.empty()) {
      std::cout << "One launch profiled; skipping CUDA-event timing and CSV.\n";
      return EXIT_SUCCESS;
    }

    std::cout << "\nSummary (kernel-only, trial median):\n"
              << std::left << std::setw(12) << "Case"
              << std::right << std::setw(10) << "StdNNZ"
              << std::setw(13) << "CSR (us)"
              << std::setw(13) << "JDS (us)"
              << std::setw(14) << "CSR GFLOPS"
              << std::setw(14) << "JDS GFLOPS"
              << std::setw(12) << "CSR/JDS" << std::endl;
    for (const Result &r : results) {
      std::cout << std::fixed << std::setprecision(3)
                << std::left << std::setw(12) << r.name
                << std::right << std::setw(10) << r.stats.stddev
                << std::setw(13) << r.csrUs
                << std::setw(13) << r.jdsUs
                << std::setw(14) << r.csrGflops
                << std::setw(14) << r.jdsGflops
                << std::setw(12) << r.csrUs / r.jdsUs << std::endl;
    }
    writeCSV(opts.output, results, opts);
    std::cout << "\nResults saved to " << opts.output << '\n'
              << "Speedup = CSR time / JDS time (>1 means JDS is faster).\n"
              << "Note: This compares CSR warp-per-row with JDS thread-per-row;"
              << " kernel times exclude matrix conversion and transfers.\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
}