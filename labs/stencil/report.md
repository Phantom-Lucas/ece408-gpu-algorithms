# 3D 7-point Stencil：沿 z 方向 Register Tiling 的性能分析报告

> 实验类型：CUDA Kernel 优化对比实验  
> 测试平台：NVIDIA GeForce RTX 4050 Laptop GPU（Compute Capability 8.9）  
> 实验日期：2026-10-09  
> 对比源码：`compare_stencil.cu`  
> 说明：本文以已完成的实际运行日志、`ptxas` 报告和 Nsight Compute 指标为依据；推测性解释均明确标注。

## 摘要

本实验研究 3D 7-point Stencil 中，沿 z 方向采用**显式 Register Tiling（寄存器滑动窗口）**能否改善 GPU Kernel 性能。两个版本均采用二维 CUDA Grid、`32×32` 线程块、`30×30` 有效计算 Tile，以及 x-y 平面的 Shared Memory Tiling；唯一主动控制的变量是 z 方向是否通过 `prev/curr/next` 在相邻迭代间显式复用输入值。

在 `512×512×64` 输入规模下，两个版本都与 CPU 参考实现完全一致。Nsight Compute 显示，显式 Register Tiling 使 Global Load 指令数与 L1/TEX Global Load sectors 均下降约 **65.6%**，每线程寄存器数由 **40** 降至 **34**，但 DRAM 读取量分别为 **22.38 MB** 和 **22.40 MB**，基本没有改变。5 组、每组 100 次的 CUDA Events 测试中，两个版本平均耗时分别为 **0.8291 ms** 和 **0.8580 ms**，未体现出明确的加速收益。

**结论：在该实现、输入规模和运行条件下，显式寄存器复用明显减少了执行层面的全局加载指令，却没有明显减少实际 DRAM 读取，也未提升端到端 Kernel 执行性能。** 该结果说明，理论访存指令优化与最终性能之间存在缓存、同步和执行资源等多方面的权衡。

## 1. 实验目标与问题

3D 7-point Stencil 以一个中心元素及其 x、y、z 三个方向的六个邻居计算输出：

\[
\begin{aligned}
\operatorname{out}(x,y,z)={}&-6\operatorname{in}(x,y,z)\\
&+\operatorname{in}(x-1,y,z)+\operatorname{in}(x+1,y,z)\\
&+\operatorname{in}(x,y-1,z)+\operatorname{in}(x,y+1,z)\\
&+\operatorname{in}(x,y,z-1)+\operatorname{in}(x,y,z+1).
\end{aligned}
\]

该实验采用 `C0=-6`、`C1=1`。三维输入以 x 为连续维度，展平索引为：

```cpp
index = (z * ny + y) * nx + x;
```

仅计算内部元素：`1 <= x < nx-1`、`1 <= y < ny-1`、`1 <= z < nz-1`。因此，`512×512×64` 测试实际更新 `510×510×62 = 16,126,200` 个输出元素。

实验聚焦三个问题：

1. 显式 Register Tiling 是否降低 GPU 实际执行的 Global Memory Load 指令？
2. Global Load 指令下降能否转化成实际 DRAM 读取量下降？
3. 上述变化是否带来 Kernel 执行时间的改善？

## 2. 实验环境与控制变量

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| GPU Compute Capability | 8.9（`sm_89`） |
| 开发环境 | Ubuntu、CUDA `nvcc`、Nsight Compute（`ncu`） |
| 编译优化 | `-O3 -arch=sm_89 -std=c++17 -lineinfo` |
| 输入尺寸 | `nx=512, ny=512, nz=64` |
| 数据类型 | `int`（32 bit） |
| 输入占用 | 67,108,864 bytes（64 MiB） |
| CUDA Grid | `(17,17,1)` |
| CUDA Block | `(32,32,1)`，每 Block 1024 线程 |
| 有效输出 Tile | 每 Block 最多 `30×30` 个 x-y 位置 |
| Shared Memory | `32×32×4 = 4096 bytes / Block` |
| Benchmark | CUDA Events；每版本预热 10 次、正式计时 100 次；重复整个程序 5 轮 |

**两版保持一致的机制：**

- 相同的输入、输出布局和边界判断；
- 相同的 Grid / Block 配置；
- 相同的 x-y Shared Memory Tiling 和 Halo 加载；
- 每层两个 `__syncthreads()`；
- 每个线程沿 z 方向处理一整列数据（Thread Coarsening）；
- 相同的 Stencil 数值公式和输出位置。

**唯一主动更改的因素：** z 方向的数据读取与跨迭代复用方式。

## 3. Kernel 设计

### 3.1 对照组：不显式进行 Register Tiling

每次 z 迭代在源代码层面读取 z-1、z、z+1 三个数据：

```cpp
for (int z = 1; z < nz - 1; ++z) {
    int prev = 0, curr = 0, next = 0;
    if (valid) {
        prev = A0[(z - 1) * plane + offset];
        curr = A0[z * plane + offset];
        next = A0[(z + 1) * plane + offset];
    }

    smem[ty][tx] = curr;
    __syncthreads();
    // 使用 x-y Shared Memory 邻居、prev 和 next 计算输出
    __syncthreads();
}
```

这并不意味着 GPU 完全“不使用寄存器”，而仅表示**程序没有手动保留跨 z 迭代的 `prev/curr/next` 数据**。编译器仍会分配寄存器，并可能进行自动优化。

### 3.2 实验组：显式 Register Tiling

先初始化相邻三层，计算后滑动寄存器窗口：

```cpp
int prev = 0, curr = 0, next = 0;
if (valid) {
    prev = A0[offset];
    curr = A0[plane + offset];
    next = A0[2 * plane + offset];
}

for (int z = 1; z < nz - 1; ++z) {
    smem[ty][tx] = curr;
    __syncthreads();
    // 使用 x-y Shared Memory 邻居、prev 和 next 计算输出
    __syncthreads();

    prev = curr;
    curr = next;
    if (valid && z < nz - 2) {
        next = A0[(z + 2) * plane + offset];
    }
}
```

从**源代码读取次数**看，对于一个有效 `(x,y)` 列、`nz=64`：

- 对照组：`3×62 = 186` 次 z 相关全局读取表达式；
- 实验组：初始化 3 次，后续 61 次读取，共 `64` 次；
- 理论减少 `(186-64)/186 ≈ 65.6%`。

这只是源代码级的访问计数，并不等同于实际 SASS 指令或 DRAM 流量；下文使用硬件计数器验证。

## 4. 正确性验证

`compare_stencil.cu` 包含 CPU 顺序参考实现，分别运行两个 GPU Kernel，并比较完整的三维输出数组。两个 GPU 输出数组在执行前清零，边界保持不变（值为零）。

实际输出：

```text
[PASS] register tiling: matches CPU reference (including unchanged boundaries)
[PASS] no register tiling: matches CPU reference (including unchanged boundaries)
```

**验证结果：两种实现均通过 CPU 参考结果逐元素比较（包括未计算的边界）。**

## 5. Benchmark 结果

### 5.1 五轮重复测试

测试命令：

```bash
for i in {1..5}; do
    ./compare_stencil 512 512 64 100
done
```

每轮中，每种 Kernel 均预热 10 次，再以 CUDA Events 对 100 次 Kernel 启动计时并计算单次平均时间。以下数值单位均为 **ms / Kernel**：

| 轮次 | 不显式 Register Tiling | 显式 Register Tiling | 速度比（无显式 / 显式） |
|---:|---:|---:|---:|
| 1 | 0.804997 | 0.819046 | 0.983× |
| 2 | 0.800922 | 0.852142 | 0.940× |
| 3 | 0.898232 | 0.962253 | 0.933× |
| 4 | 0.834068 | 0.850175 | 0.981× |
| 5 | 0.807250 | 0.806380 | 1.001× |
| **平均** | **0.829094** | **0.857999** | **0.966×** |

平均差值：`0.857999 - 0.829094 = 0.028905 ms`。显式 Register Tiling 的平均耗时约高 **3.5%**；由于测试之间存在波动，并且每轮固定先测对照组、后测实验组，**本实验不足以支持“显式 Register Tiling 在统计意义上更慢”的强结论**。可以谨慎判断：在当前测试条件下没有观察到明显加速。

### 5.2 ptxas 资源报告

编译命令：

```bash
nvcc -O3 -arch=sm_89 -std=c++17 \
  -lineinfo -Xptxas=-v \
  compare_stencil.cu -o compare_stencil
```

| 指标 | 不显式 Register Tiling | 显式 Register Tiling |
|---|---:|---:|
| Registers / Thread | 40 | 34 |
| Shared Memory / Block | 4096 B | 4096 B |
| Spill Loads | 0 | 0 |
| Spill Stores | 0 | 0 |
| Stack Frame | 0 B | 0 B |

显式版本每线程寄存器数量下降 **15%**，且两个版本均未发生寄存器溢出。寄存器数下降不保证 Occupancy 增长：对于 1024 线程的大 Block，实际同时驻留的 Block 数还受到 SM 寄存器容量、线程数等资源约束。

## 6. Nsight Compute 性能计数器结果

由于普通用户无法访问性能计数器（`ERR_NVGPUCTRPERM`），实际通过 `sudo ncu` 采集。以下为用户提供的原始实测计数。

### 6.1 Global Load 指令与 L1/TEX sectors

| 指标 | 不显式 Register Tiling | 显式 Register Tiling | 减少比例 |
|---|---:|---:|---:|
| `smsp__sass_inst_executed_op_global_ld.sum` | 1,720,128 | 591,872 | **65.6%** |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | 8,094,720 | 2,785,280 | **65.6%** |

结论：**显式 Register Tiling 的优化确实体现在 GPU 执行的 Global Load 指令数量上**，不能再将性能未提升简单归因于“编译器把两个版本编成了一样的 Global Load 行为”。

注意：`sectors` 是 L1/TEX 路径请求量，不是 DRAM 读取量；不能直接据此声称显存带宽消耗下降 65.6%。

### 6.2 DRAM 实际读写量

| 指标 | 不显式 Register Tiling | 显式 Register Tiling | 差异 |
|---|---:|---:|---:|
| `dram__bytes_read.sum` | 22.38 MB | 22.40 MB | +0.02 MB（约 +0.09%） |
| `dram__bytes_write.sum` | 18.57 MB | 18.79 MB | +0.22 MB（约 +1.2%） |

两种版本的 DRAM Read **几乎相同**。这是本实验最关键的观察：更少的 Global Load 指令没有带来对应的显存读取量下降。

合理解释是，额外的加载请求可能被 L1/L2 Cache 满足，或者出现其他缓存与调度行为，使它们没有转化为额外 DRAM 传输。不过，当前只测量了 Load、L1/TEX sectors 和 DRAM 字节数，**没有直接测量 L1/L2 命中率**，因此尚不能定量确认各级 Cache 的贡献。

还应注意：输入本身占 64 MiB，但测量到的 DRAM Read 仅约 22.4 MB。这说明不能把本次测得的 DRAM 字节数解读为“从显存完整冷读一次输入”的流量。计时程序在同一输入上反复执行 Kernel，且此前已完成 HtoD 拷贝和正确性检查；Profiler 捕获时的缓存状态可能明显影响 DRAM 计数。

### 6.3 指标采集命令

Global Load 和 L1/TEX sectors：

```bash
sudo "$(command -v ncu)" \
  --kernel-name-base function \
  --kernel-name "regex:stencil_no_register_tiling" \
  --launch-count 1 \
  --metrics smsp__sass_inst_executed_op_global_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum \
  ./compare_stencil 512 512 64 1
```

将 `stencil_no_register_tiling` 改成 `stencil_register`，采集另一个版本。

DRAM 读写量：

```bash
sudo "$(command -v ncu)" \
  --kernel-name-base function \
  --kernel-name "regex:stencil_no_register_tiling" \
  --launch-count 1 \
  --metrics dram__bytes_read.sum,dram__bytes_write.sum \
  ./compare_stencil 512 512 64 1
```

同样修改 Kernel 名称以采集实验组。

## 7. 结果讨论

### 7.1 已被证实的事实

1. 两种 Kernel 计算结果一致且正确。
2. 显式 Register Tiling 减少了约 **65.6%** 的 Global Load 指令和 L1/TEX Global Load sectors。
3. 显式版本每线程寄存器使用量由 **40** 降为 **34**，两者均无 Spill。
4. DRAM 读取量分别为 **22.38 MB** 与 **22.40 MB**，在本次采样中近乎相等。
5. 五轮 CUDA Events 计时没有显示显式 Register Tiling 的明确加速收益。

### 7.2 可能原因（尚待进一步验证）

- **缓存命中：** 对照组额外的 Global Load 可能主要命中缓存，因此没有形成相应的 DRAM 流量。
- **同步与计算开销：** 两个版本每层均有两次 `__syncthreads()`；`nz=64` 时每个 Block 在循环中执行 124 次同步。即使 Global Load 指令数减少，其他开销仍可能主导执行时间。
- **资源与调度：** 寄存器减少未必改变 Occupancy，指令依赖、发射调度以及内存延迟隐藏也可能影响最终表现。

目前**无法仅凭这些计数器判定哪个原因是唯一或主要瓶颈**。需要进一步采集缓存命中率、Warp Stall、Occupancy 和 Memory Throughput 才能区分。

### 7.3 实验局限

- 仅测试一个主要输入规模：`512×512×64`。
- 计时固定先运行无显式 Register Tiling，再运行显式版本；尚未交替或随机化顺序。
- 5 轮观测数量有限，未控制 GPU 频率、温度和背景负载。
- Nsight Compute 的计数器采样与 CUDA Events 计时并非同一性能环境；Profiler 可能引入 replay 与缓存状态变化。
- 尚未直接采集 L1/L2 Hit Rate、DRAM 带宽利用率、Warp Stall 原因等指标。

## 8. 结论

本实验实现并比较了两种 3D 7-point Stencil CUDA Kernel：一版在 z 方向显式使用 `prev/curr/next` 寄存器滑动窗口，另一版每轮直接使用三个全局数组读取表达式。两者共享相同的 x-y Shared Memory Tiling、线程粗化和同步结构。

**Register Tiling 的访存指令优化是真实且显著的**：Global Load 指令减少约 **65.6%**，寄存器使用量反而下降 **15%**。但在当前测试中，DRAM Read 保持在约 **22.4 MB**，Kernel 耗时也没有得到可确认的改善。

因此，这个实验并不表明 Register Tiling “无用”，而是说明：**优化源代码层面的数据复用或减少 Global Load 指令，并不必然降低显存流量，也不保证缩短 Kernel 时间。** CUDA 性能优化应结合实际硬件计数器识别瓶颈，而不只根据理论 Load 次数判断。

## 9. 后续改进方向（可选）

1. **更公平的 Benchmark**：交替测试顺序，增加运行轮数，记录中位数、波动和 GPU 频率。
2. **进一步 Profiling**：采集 L1/L2 Cache 相关命中统计、Achieved Occupancy、Warp Stall 原因与 DRAM Throughput。
3. **隔离同步成本**：在保证线程间数据安全的前提下，设计专门实验量化 Shared Memory Barrier 开销。
4. **不同尺寸验证**：改变 x-y 面积和 z 深度，判断 Register Tiling 在不同工作集与缓存压力下是否出现收益。

---

### 附录：复现实验

编译：

```bash
nvcc -O3 -arch=sm_89 -std=c++17 -lineinfo \
  -Xptxas=-v compare_stencil.cu -o compare_stencil
```

运行与计时：

```bash
./compare_stencil 512 512 64 100
for i in {1..5}; do ./compare_stencil 512 512 64 100; done
```

> 本报告数据来自本次实验输出；不包含未执行的对照实验，也不将缓存、Occupancy 或同步瓶颈推测写成已验证事实。