# ECE408 SGEMM：线程粗化与分块优化实验技术报告

> **实验项目：** Matrix Multiplication with Thread Coarsening and Register Tiling  
> **实验平台：** NVIDIA GeForce RTX 4050 Laptop GPU（6 GB，Compute Capability 8.9），Ubuntu / CUDA  
> **测试工具：** CMake、实验自带 `eval.cu`、NVIDIA Nsight Compute

## 1. 实验目标与实现方案

本实验实现单精度矩阵乘法 $C_{m\times n}=A_{m\times k}B_{k\times n}$，利用 **Thread Coarsening、Register Tiling 和 Shared Memory Tiling** 提高数据复用效率。输入矩阵 A 与输出矩阵 C 为 **Column-major**，输入矩阵 B 为 **Row-major**。

采用以下固定分块参数：

| 参数 | 数值 | 含义 |
|---|---:|---|
| `TILE_SZ_A` | 128 | 每个 Block 负责 C 的 128 行；Block 有 128 个线程 |
| `TILE_SZ_B` | 16 | 每个 Block 负责 C 的 16 列 |
| `TILE_SZ_RATIO` | 8 | 沿 K 维每轮处理 8 个元素 |

每个 Block 计算一个 **128×16 的 C Tile**：128 个线程各负责 1 行 × 16 列；沿 K 维循环，每轮进行 $A_{128\times8}\times B_{8\times16}$ 的分块乘加。

- **A — Register Tiling：** 每线程从 A 取一个数 `a` 到寄存器，复用该值更新 16 个输出累加器；按 K 维逐次加载，不需要同时保存完整 A Tile。
- **B — Shared Memory Tiling：** 128 个线程合作加载一个 8×16 的 `Bs` Tile（512 B），每线程加载 1 个元素，随后供同一 Block 的线程反复使用。
- **C — Register Tiling：** 每线程维护 `sum[16]`，计算结束后按列主序写回 C；不同线程写入的行互不重叠，**不需要原子操作**。同一列上 Warp 内线程访问连续地址，有利于合并访存。
- **同步与边界：** 每轮加载 B 后及覆盖下一轮 B 前分别执行一次 `__syncthreads()`；对矩阵尾部 Tile 进行越界保护，B 越界位置补零，写回 C 时检查边界。

## 2. 测试方法与正确性

使用 Release 模式编译（目标架构 `sm_89`），通过实验测试程序测量 `Performing GPU sgemm` 阶段的时间，并通过 Nsight Compute 采集 Kernel Duration、Occupancy、SM/DRAM 吞吐率、寄存器等指标。

在保留 CPU 参考结果验证的测试中，**1024³ 与 2048³ 均通过逐元素断言检查**。随后为避免 CPU 三重循环验证在超大矩阵上耗时过长，4096³、8192³、16384³ 的性能运行关闭了 CPU 验证；因此这三组记录只代表**完成运行和计时，不代表已完成正确性验证**。

> 计时口径：下表大矩阵时间来自测试程序的 `Performing GPU sgemm`，与 Nsight Compute 报告中的 Kernel Duration 不同；不应直接混用进行性能对比。

## 3. 性能结果

### 3.1 大矩阵运行时间与有效计算吞吐率

理论浮点运算量近似为 $2N^3$，计算吞吐率为 $\mathrm{GFLOPS}=2N^3/(T_{\mathrm{ms}}\times10^6)$。

| 矩阵规模 $N\times N\times N$ | GPU 阶段时间 | 有效吞吐率 |
|---|---:|---:|
| 4096³ | 59.05 ms | 2.33 TFLOPS |
| 8192³ | 418.32 ms | **2.63 TFLOPS** |
| 16384³ | 5941.60 ms | 1.48 TFLOPS |

当矩阵从 4096³ 增至 8192³ 时，运算量增长 8 倍，时间增长约 **7.08 倍**；但从 8192³ 增至 16384³ 时，时间增长约 **14.20 倍**，有效吞吐率显著下降。现有数据不足以确定下降的具体原因，需要通过重复计时、时钟/功耗记录及更大尺寸的 Profiling 进一步定位。

### 3.2 Nsight Compute 指标（不同矩阵规模）

| 指标 | 256³ | 1024³ | 2048³ |
|---|---:|---:|---:|
| Grid Blocks | 32 | 512 | 2048 |
| Achieved Occupancy | 13.33% | 82.98% | **96.51%** |
| Compute (SM) Throughput | 9.03% | 63.68% | **84.31%** |
| DRAM Throughput | 1.51% | 2.83% | 3.58% |
| Registers / Thread | 40 | 40 | 40 |
| Kernel Duration（ncu） | 91.01 μs | 815.26 μs | 4.93 ms |

**关键发现：**

1. **小矩阵受并行规模限制。** 256³ 仅 32 个 Block，而 GPU 有 20 个 SM，导致实际 Occupancy 只有 13.33%；扩展至 2048³ 后提升至 96.51%。
2. **大矩阵 SM 吞吐显著提高。** 2048³ 时 Compute (SM) Throughput 达 84.31%。该指标是 Nsight Compute 的计算资源吞吐率，**不等同于 FP32 峰值 FLOPS 利用率**。
3. **暂无明显 DRAM 带宽饱和迹象。** 2048³ 的 DRAM Throughput 为 3.58%，但 L1/TEX Cache Throughput 达 84.60%、L2 Hit Rate 为 97.56%；进一步优化应关注缓存/访存管线和计算指令，而非只盯显存带宽。
4. **寄存器未出现明显容量问题。** 每线程 40 个寄存器、Local Memory Spilling 为 0，理论 Occupancy 为 100%。

## 4. Thread Coarsening 与 Tile 参数探索

固定测试平台，分别扫描 `TILE_SZ_A∈{64,128,256}`、`TILE_SZ_B∈{8,16,32,64}` 和 `TILE_SZ_RATIO∈{4,8,16}`，共 **36 种组合**；在每种矩阵规模下预热 2 次、重复计时 5 次，取 CUDA Event 耗时中位数。此独立调参实现允许三项参数分别变化；这些测试**不进行 CPU 正确性验证**。

| 矩阵规模 | 最优 A / B / RATIO | Kernel 中位时间 | 吞吐率 |
|---|---|---:|---:|
| 1024³ | 256 / 32 / 16 | 0.582 ms | **3.69 TFLOPS** |
| 2048³ | 256 / 64 / 16 | 4.705 ms | 3.65 TFLOPS |
| 4096³ | 256 / 32 / 8 | 42.020 ms | 3.27 TFLOPS |
| 8192³ | 256 / 64 / 16 | 336.745 ms | 3.27 TFLOPS |

**实验现象与分析：**

1. **适度粗化有效，但并非越大越好。** `B=8` 普遍较慢，`B=32` 整体表现稳定；`B=64` 在部分尺寸更快，但寄存器占用由 `B=32` 的约 65 个/线程增至约 104 个/线程，存在资源压力的权衡。
2. **最优配置与矩阵规模相关。** 本轮四个尺寸的最快配置均采用 `A=256`，但 `B` 和 `RATIO` 并不相同；`RATIO=8/16` 通常优于 4。
3. **同一测试口径下有可观收益。** 在 4096³ 的自动调参实验中，`128/16/8` 耗时 **56.790 ms**，最优 `256/32/8` 为 **42.020 ms**，耗时下降约 **26.0%**（加速约 **1.35×**）。

## 5. 总结

通过 Register Tiling 复用 A 并累加 C、Shared Memory Tiling 在 Block 内复用 B，Kernel 能实现有效的数据复用和合并访存。**小矩阵主要受到并行规模限制，大矩阵能显著提高 SM 利用率；进一步优化的关键在于粗化程度、K 维分块和寄存器压力之间的平衡。** 本轮最高测得约 **3.69 TFLOPS**。自动调参结果属于性能测试，选定配置后仍需恢复 CPU/GPU 对照验证。