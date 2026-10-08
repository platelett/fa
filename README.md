# Flash Attention on Ascend

本仓库比较同一个 attention 的多种实现：**Torch** 调用 NPU 内置算子；**TileLang** 保留原流式版本，并新增把输出累加留在片上的 FP32 版本；**CCE** 使用底层接口手工实现，提供 FP16、FP32 两种中间计算精度。各版输入、输出均为 FP16。

想了解这些实现怎样一步步变快，可以阅读[中文优化历程](docs/optimization/README.md)，其中包括[新版 TileLang FP32](docs/optimization/tilelang-fp32.md)及没有采用的尝试。

## 性能怎么看

表中每一行是一组输入。`Batch` 表示一次处理几个样本，`Length` 表示每个样本的序列长度。各版本处理相同的输入规模，并通过了正确性检查。本次新增 TileLang FP32 列，其余保留历史实测值。

**用时越小越快，单位为毫秒。**

| Batch | Length | Torch | TileLang 流式（历史） | TileLang FP32 | CCE FP16 | CCE FP32 |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 131072 | 1207.513 | 1118.169 | 954.812 | 572.472 | 714.630 |
| 2 | 65536 | 300.555 | 276.640 | 233.315 | 143.603 | 180.099 |
| 1 | 32768 | 37.564 | 34.508 | 29.161 | 18.440 | 23.018 |

**下面的百分比，用来比较实际速度与这台机器的理论矩阵乘法速度。越高，越接近理论上限。**

Cube 是昇腾中负责矩阵乘法的计算单元。我们用 attention 中两次矩阵乘法的计算量，估算它们在 Cube 满速运行时需要多久，再除以完整 attention 的实测用时。

例如，第一组输入的两次矩阵乘法，理想情况下需要约 **557 毫秒**；CCE FP16 完成整个 attention 实际用了约 **572 毫秒**。因此它达到的比例约为 `557 / 572 = 97.3%`。这个比例比较的是计算速度，不是硬件忙碌时间。

| Batch | Length | Torch | TileLang 流式（历史） | TileLang FP32 | CCE FP16 | CCE FP32 |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 131072 | 46.1% | 49.8% | 58.4% | 97.3% | 78.0% |
| 2 | 65536 | 46.3% | 50.4% | 59.7% | 97.0% | 77.3% |
| 1 | 32768 | 46.4% | 50.5% | 59.7% | 94.4% | 75.6% |

两张表都来自同一台 Atlas A3，各列不是同一轮测量。新增 TileLang FP32 在 2026-10-08（UTC）预热 3 次、测 5 次、取中位数；其余列是此前 10 次采样的中位数。流式 TileLang 一列对应原发布版本，当前本地源码已更新，不能把旧成绩当成更新后的实测。所有百分比使用同一个参照——这台机器 25 个 Cube 的 FP16 理论峰值，共 378.88 TFLOPS；TileLang 实际仍使用 24 组。

## 当前主线 Expert 对照

2026-10-08，原 `core.bench` 的三个标准输入、同一批测量，设备内核用时均值如下。
三版共 9 项检查全部通过；v3 相对旧 v1 约为 1.92 倍加速。
这张表不与上面的历史中位数混用。

| Batch | 序列长度 | expert_v1，ms | expert_v2，ms | expert_v3，ms |
|---:|---:|---:|---:|---:|
| 2 | 131072 | 1792.043 | 1116.421 | 934.714 |
| 2 | 65536 | 447.936 | 276.739 | 233.308 |
| 1 | 32768 | 56.062 | 34.541 | 29.194 |

条件、源码身份和原报告见[完整对照](backends/expert_v3/mainline_comparison.md)。

## 怎么运行

先准备好已有的 PyTorch、`torch_npu`、CANN 和最新 `ascendc_pto` 主线的 TileLang 环境；本仓库不安装这些外部依赖。2026-10-08 核实的主线 HEAD 为 `4c726701`。上面的历史性能表仍保留原测量日期与条件，不代表每次主线更新后的成绩。

将 `env.local.sh.example` 复制为 `env.local.sh`，填入已有 TileLang 环境脚本的路径，然后在 Bash 中运行：

```bash
source env.sh
npu-smi info
# 选择空闲设备；这里以设备 0 为例。
python -m core.bench --cases all --devices 0
```

这会运行全部已注册版本（含 Expert 对照），检查结果，并生成本地计时报告。默认配置使用 24 组计算单元。只运行新增版本时使用：

```bash
python -m core.bench --backends expert_v3 --cases all --devices 0
```

框架会自动加入 Torch 参考实现。

**要使用上表的测试配置，在本机已验证的 25-Cube 设备上运行：**

```bash
FA_LOGICAL_BLOCKS=0 python -m core.bench --cases all --devices 0
```

这个设置只改变两个 CCE 版本的任务分配，允许许多小任务接续执行。代码会先检查设备是否适用；TileLang 仍使用 24 组。上表使用的就是这个配置，因此默认命令的 CCE 用时可能不同。

以下是供查看代码和调整实现时使用的细节。

<details>
<summary>测试参数与峰值比例公式</summary>

测试环境：Atlas A3 `Ascend910_9392`、CANN 9.1。历史流式 TileLang 使用 PR #1852 提交 `77a444b2`；新增 TileLang FP32 使用已合并该 PR 的提交 `4c726701`。均预热 3 次。原四列分别沿用其 10 样本记录，新增列使用与其优化前版本正反交错的 5 样本记录。输入为非因果 attention，12 个 query heads、1 个 KV head，head dimension 为 128。原始样本留在本地研究档案，不放入公开仓库。

仅统计 QK 和 PV 两次矩阵乘法的 FLOPs：

```text
QK/PV FLOPs = 4 * B * Hq * Nq * Nk * D
peak ratio = QK/PV FLOPs / (time_seconds * 378.88e12)
nominal peak = 25 * 1.85e9 * 8192 FLOP/cycle
```

两个 TileLang 实现均使用三个工作区槽，通过 `reduce_max`、减法、`exp` 和 `reduce_sum` 组合完成在线 softmax，不需要额外的融合 softmax 接口。新增版固定 query/key 分块为 256/512，并保留 FP32 输出累加在片上。若只按它实际使用的 24 个 Cube 计算，三个 case 的峰值比例分别为 60.8%、62.2%、62.2%；上表统一使用 25 个 Cube 作参照。

</details>

A small benchmark framework for non-causal, FP16-input grouped-query attention
on Ascend A3 (`ascend910_93`), with head dimension 128. It contains selected
implementations, their source code, and a common correctness/timing runner.
No raw benchmark output files, compiled binaries, external dependencies, or dependency
installation scripts are included.

## Implementations

三个 TileLang backend 使用与 [FA PR #1863](https://github.com/tile-ai/tilelang-ascend/pull/1863)
相同的版本名。`expert_v1` 保留主线旧实现；`expert_v2` 从 TileLang 写法出发组织流水；
`expert_v3` 先用 CCE 验证调度，再用 TileLang 表达。测试点、标准输入和
`core.bench` 的计时/正确性流程不改。直接运行完整三点对照：

```bash
source env.sh
python -m core.bench --backends expert_v1,expert_v2,expert_v3 --cases all --devices 8
```

设备 8 只是示例，应先确认空闲。结果与 `npu_fa` 检查并由原 profiler 统计 20 次设备
内核用时的平均值，报告写入 `results/`；新旧比较不使用上面的历史跨批次数据。

| Backend | Implementation |
|---|---|
| `npu_fa` | `torch_npu.npu_fusion_attention`, native GQA; correctness reference |
| `expert_v1` | [`fa_opt/expert_v1/kernel.py`](https://github.com/tile-ai/tilelang-ascend/blob/cbfd67e9b7ccd227730d20e1e6d6a2ae08cde156/examples/flash_attention/fa_opt/expert_v1/kernel.py); earlier Expert implementation retained for comparison |
| `expert_v2` | [`fa_opt/expert_v2/kernel.py`](https://github.com/tile-ai/tilelang-ascend/blob/cbfd67e9b7ccd227730d20e1e6d6a2ae08cde156/examples/flash_attention/fa_opt/expert_v2/kernel.py); TileLang-first three-slot online softmax |
| `expert_v3` | [`fa_opt/expert_v3/kernel.py`](https://github.com/tile-ai/tilelang-ascend/blob/cbfd67e9b7ccd227730d20e1e6d6a2ae08cde156/examples/flash_attention/fa_opt/expert_v3/kernel.py); CCE-derived FP32 resident output accumulation |
| `cce_fa_nqkq_nz_resident_v13_q768_pool_i4_p128` | CCE Normal-NZ Q-K-Q, Q_L1=768, WS_Q=256, WS_K=512; resident FP16 O |
| `cce_fa_nk_nz_vector_fp32_v5_sumpack` | CCE Normal-NZ K-first, Q_L1=WS_Q=256, WS_K=512; resident FP32 O; reused reduction scratch and staged P conversion; no BAR.V in the Vector payload |

All return normalized attention output in FP16 with shape `[B,Hq,Nq,128]`.
The CCE FP32 variant uses FP32 Vector arithmetic except FP16 max comparisons;
Cube communication (S, P and partial O) remains FP16. The FP16 variant can
overflow for sufficiently large unnormalized accumulators.

The TL entries implement online softmax with TileLang-Ascend reduction,
broadcast, subtraction and exponential primitives. Running maximum, denominator
updates and delayed O accumulation are included. They need no added fused-softmax
operator or compiler patch. The CCE projects are self-contained operator sources;
they do not import generators or other backends from the research repository.

## Environment

Provide an existing working environment with Python, PyTorch, `torch_npu`,
CANN/BiSheng, CMake, and (for the TL entries) compatible TileLang-Ascend.
The CCE sources target the CANN 9.1 / dav-c220 API. Dependencies are not managed
by this repository.

Use the latest [TileLang-Ascend `ascendc_pto` mainline](https://github.com/tile-ai/tilelang-ascend/tree/ascendc_pto),
not a PR-specific compiler branch. Its HEAD was
[`4c7267018af02284ff28f1cd090a1492c7985bed`](https://github.com/tile-ai/tilelang-ascend/commit/4c7267018af02284ff28f1cd090a1492c7985bed)
when checked on 2026-10-08, the same compiler revision used for the recorded comparison.
After updating mainline, rerun the original matrix before publishing new performance claims.
The TL entries use its
existing Expert layout/allocation, synchronization and BRCB/row-expand APIs.
No extra fused-softmax interface or local compiler patch is required.
Activate that external TileLang environment yourself; it is not vendored or
installed by this repository. The other three backends run independently
without TileLang.

Using Bash, copy `env.local.sh.example` to the ignored `env.local.sh` and point
it at your existing TileLang activation script when needed. Then:

```bash
source env.sh
npu-smi info
# Replace 0 with an idle device on your machine.
python -m core.bench --cases all --devices 0
```

This builds the selected implementations, profiles device execution and checks
outputs against `npu_fa`. Local reports go into the ignored `results/` directory.
They are generated at runtime and are not part of the repository.

To run only the Torch reference and a CCE implementation:

```bash
python -m core.bench \
  --backends npu_fa,cce_fa_nk_nz_vector_fp32_v5_sumpack \
  --cases all --devices 0
```

For a single correctness check, first expose only the intended idle device:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python -m core.runner \
  --backend cce_fa_nqkq_nz_resident_v13_q768_pool_i4_p128 \
  --case-index 2 --mode check
```

The three built-in workloads are defined in `problem/cases.py`. They use sequence
lengths 131072, 65536 and 32768, 12 query heads, one KV head, and head dimension
128. Backend-specific alignment constraints are enforced by each implementation.

## Launch configuration

Public defaults use 24 logical Cube groups (paired with 48 Vector contexts),
including the CCE entries. The selected CCE kernels also retain their original
many-block scheduling mode for explicitly qualified local 25-Cube/50-Vector
installations:

```bash
FA_LOGICAL_BLOCKS=0 python -m core.bench \
  --backends cce_fa_nqkq_nz_resident_v13_q768_pool_i4_p128,cce_fa_nk_nz_vector_fp32_v5_sumpack \
  --cases all --devices 0
```

`0` selects the original automatic logical grid. This opt-in performs a device
identity/data qualification and rejects an incompatible physical mapping.
Do not assume that another Ascend deployment exposes 25 Cube cores.

## Source layout and checks

```text
backends/    selected implementations, comparison entries and local CANN projects
core/        discovery, build/load drivers, checks and profiling
problem/     canonical inputs and workloads
tests/       host-only framework regressions
```

Edit TL's `impl_tl.py`, then regenerate its runnable file:

```bash
python backends/expert_v2/preprocess.py \
  backends/expert_v2/impl_tl.py -o backends/expert_v2/impl.py
python backends/expert_v3/preprocess.py \
  backends/expert_v3/impl_tl.py -o backends/expert_v3/impl.py
python -m unittest discover -s tests -v
```

The CCE build drivers use source hashes and keep local artifacts under each
project's ignored `.build/` directory. They do not install a system-wide OPP
package. Keep package names distinct when adding another backend.
