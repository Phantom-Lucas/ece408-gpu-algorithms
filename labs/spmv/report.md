# ECE408 SpMV：CSR 优化、JDS 与 NNZ 分布性能分析

## 1. 实验目标与方法

实现单精度稀疏矩阵向量乘法（SpMV）：

\[
y_i=\sum_{j:A_{ij}\ne0} A_{ij}x_j,\qquad \mathrm{GFLOPS}=\frac{2\,\mathrm{NNZ}}{T(\mathrm{s})\times10^9}.
\]

**环境：** NVIDIA GeForce RTX 4050 Laptop GPU（Compute Capability 8.9），CUDA `-O3 -arch=sm_89`。只比较 **Kernel 执行时间**（不含数据传输和格式转换），所有版本均通过 CPU 参考结果验证。

实验分为两阶段：①在 ECE408 原始测试集上优化 CSR 的线程分配和归约；②固定矩阵规模与总 NNZ，分析 CSR 和 JDS 对行 NNZ 分布的敏感性，并使用 Nsight Compute 解释性能差异。

## 2. CSR 三种实现与优化结果

| 实现 | 并行方式 | 归约方法 |
|---|---|---|
| **CSR Scalar** | 一个线程处理一行，顺序累加 NNZ | 线程内累加 |
| **CSR Warp Shared** | 一个 Warp 处理一行，Lane 按步长 32 分担 NNZ | Shared Memory + Warp 同步 |
| **CSR Warp Shuffle** | 同样一个 Warp 处理一行 | 寄存器 + `__shfl_down_sync()` |

原始数据集 `data/0–9`：5 次预热、100 次 CUDA Event 计时，取中位数。以下为找回的**四组真实测量记录**（单位：μs；其余数据集的耗时未保留）：

| 数据集 | Dim / NNZ | Scalar | Warp Shared | Warp Shuffle | Shuffle / Scalar 加速比 |
|---|---:|---:|---:|---:|---:|
| `data/0` | 16 / 17 | 3.07 | 3.07 | 3.07 | 1.00× |
| `data/1` | 256 / 3,372 | 4.96 | 3.90 | 3.07 | 1.61× |
| `data/3` | 912 / 43,669 | 10.24 | 4.93 | 4.10 | **2.50×** |
| `data/9` | 892 / 39,658 | 10.18 | 5.12 | 4.10 | **2.48×** |

**结论：** 行内协作把串行遍历分摊到 32 个 Lane；Shuffle 不需要 Shared Memory 中间存储与反复同步，在这些测试中优于 Warp Shared。极小矩阵（`data/0`）的微秒级时间接近固定启动开销，优化收益不明显。后续 CSR 基线采用 **Warp Shuffle**。

## 3. NNZ 分布实验：CSR Warp Shuffle vs JDS Scalar

JDS 按每行 NNZ 降序重排，再按“第 0、1、2……个非零元素”连续存储；其索引为 `idx = matColStart[j] + sortedRow`，最终通过行置换恢复输出顺序。**JDS 使用每线程一行，与 CSR 的每 Warp 一行不同，故本节比较的是两个具体实现，而非纯粹的存储格式。**

固定矩阵 **65,536 × 65,536**，总 **NNZ = 2,097,152**，平均 **32 NNZ/row**；使用 50 次预热、每轮 500 次重复、7 轮中位数。各行 NNZ 分布如下：Uniform=全 32；Moderate=一半 16、一半 48；High=一半 4、一半 60；Extreme=一半 0、一半 64。

| 分布 | NNZ 标准差 | CSR (μs) | JDS (μs) | JDS 加速比 | CSR / JDS (GFLOPS) | CSR→JDS 转换 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Uniform | 0 | 90.740 | 62.614 | **1.449×** | 46.22 / 66.99 | 15.456 |
| Moderate | 16 | 112.605 | 52.380 | **2.150×** | 37.25 / 80.08 | 16.061 |
| High | 28 | 96.946 | 40.858 | **2.373×** | 43.26 / 102.66 | 11.867 |
| Extreme | 32 | 101.294 | 40.415 | **2.506×** | 41.41 / 103.78 | 12.578 |

**观察：** JDS 四组均更快；随着 NNZ 分布变得极端，JDS 从 **62.614 μs 降至 40.415 μs（−35.5%）**，而 CSR 没有单调趋势。两阶段数据规模和计时方法不同，不应直接比较其绝对耗时。

## 4. Nsight Compute：硬件计数器

以下只列 Uniform 和 Extreme 两种端点分布。`Load Requests / Sectors` 是全局加载的汇总指标，**Sectors/Request 不能单独代表某一个数组的合并访存效率**。

| 场景 / Kernel | Load Requests | Load Sectors | Sectors/Req | Warp 指令数 | L1 Hit | Achieved Occupancy |
|---|---:|---:|---:|---:|---:|---:|
| Uniform CSR | 327,680 | 2,750,129 | 8.39 | 4,653,056 | 18.98% | 83.22% |
| Uniform JDS | 266,240 | 2,668,042 | 10.02 | 763,904 | 31.56% | 83.26% |
| Extreme CSR | 327,680 | 2,748,107 | 8.39 | 4,030,464 | 18.93% | 70.44% |
| Extreme JDS | 266,240 | 2,641,561 | 9.92 | 748,544 | 64.98% | 92.62% |

- **指令与请求：** 对比 CSR，JDS 的 Global Load Requests 少 **18.75%**，Warp 指令数少约 **81%–84%**；但 JDS 的 Sectors/Request **更高**，不能直接声称它的所有加载都更合并。主要差别涉及 Warp-per-row 的协作/Shuffle 与 Thread-per-row 的执行组织。
- **Extreme JDS 的额外收益：** 相对 Uniform，Kernel 时间降 **35.5%**，但 Warp 指令仅降 **2.0%**、Load Sectors 仅降 **1.0%**、Load Requests 不变。此前完整采样中 L1 Hit 从 **31.56% 升至 64.98%**、Occupancy 从 **83.26% 升至 92.62%**，表明缓存行为与工作组织可能比指令总量更能解释差异，但**尚未证明因果关系**。
- **瓶颈特征：** Nsight Compute 指出显著的 Long Scoreboard Stall（等待内存加载依赖）；Uniform JDS 还存在 LG Throttle。观测到的 DRAM Throughput 远低于峰值，更应关注缓存层次、依赖延迟和调度，而非只追求 DRAM 带宽。

**测量限制：** Nsight Compute 使用 `--cache-control none`，并在 3～15 个 Replay Pass 中收集计数器；缓存状态可能波动（完整报告中甚至出现大于 100% 的 L2 Hit Rate）。因此以 CUDA Event 结果判断时间趋势，缓存/占用率指标仅用于提出合理解释。

## 5. 格式转换成本与结论

若矩阵原本为 CSR，先转换成 JDS，再重复做 \(K\) 次 SpMV，忽略传输成本时，JDS 摊销转换开销的临界条件为：

\[
K>\frac{T_{\mathrm{convert}}}{T_{\mathrm{CSR}}-T_{\mathrm{JDS}}}.
\]

四组数据对应约 **550 / 267 / 212 / 207 次**重复 SpMV（Uniform / Moderate / High / Extreme）才能抵消测得的转换时间；只计算单次 SpMV 时，不能仅凭 Kernel 时间认定 JDS 的端到端性能更优。

**总结：** ① CSR 的行内 Warp 并行和 Shuffle 归约在代表性数据集上带来最高 **2.50×** 加速；② 当前实现中 JDS 相对 CSR Warp Shuffle 获得 **1.45–2.51×** 加速，且对行 NNZ 分布更敏感；③ 访存请求、Warp 指令数、缓存行为与线程负载组织共同决定 SpMV 性能，不能用单一的 Sectors/Request 或 NNZ 标准差解释全部差异。