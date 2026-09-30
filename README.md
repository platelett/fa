# Flash Attention on Ascend

这个仓库在昇腾 NPU 上比较四种 Flash Attention 实现。它们使用同样的输入、完成同样的计算，输入和输出都是半精度（FP16）。

- **Torch**：调用 `torch_npu` 提供的现成 attention 算子，作为比较基准。
- **TileLang**：用 TileLang 编写 attention 内核，可以自己安排计算和数据搬运。
- **CCE FP16**：用昇腾的底层编程接口手工实现，中间计算和各分块结果的累加主要使用半精度。
- **CCE FP32**：同样是手工实现，将矩阵乘法之外的计算和中间输出累加主要改为单精度（FP32）。两个 CCE 版本的矩阵乘法仍使用 FP16。

## 性能怎么看

表中每一行是一组输入。`Batch` 表示一次处理几个样本，`Length` 表示每个样本的序列长度。四个版本处理的输入相同，结果都通过了正确性检查。

**用时越小越快，单位为毫秒。**

| Batch | Length | Torch | TileLang | CCE FP16 | CCE FP32 |
|---:|---:|---:|---:|---:|---:|
| 2 | 131072 | 1207.513 | 1118.169 | 572.472 | 793.552 |
| 2 | 65536 | 300.555 | 276.640 | 143.603 | 199.889 |
| 1 | 32768 | 37.564 | 34.508 | 18.440 | 25.524 |

**下面的百分比，用来比较实际速度与这台机器的理论矩阵乘法速度。越高，越接近理论上限。**

Cube 是昇腾中负责矩阵乘法的计算单元。我们用 attention 中两次矩阵乘法的计算量，估算它们在 Cube 满速运行时需要多久，再除以完整 attention 的实测用时。

例如，第一组输入的两次矩阵乘法，理想情况下需要约 **557 毫秒**；CCE FP16 完成整个 attention 实际用了约 **572 毫秒**。因此它达到的比例约为 `557 / 572 = 97.3%`。这个比例比较的是计算速度，不是硬件忙碌时间。

| Batch | Length | Torch | TileLang | CCE FP16 | CCE FP32 |
|---:|---:|---:|---:|---:|---:|
| 2 | 131072 | 46.1% | 49.8% | 97.3% | 70.2% |
| 2 | 65536 | 46.3% | 50.4% | 97.0% | 69.7% |
| 1 | 32768 | 46.4% | 50.5% | 94.4% | 68.2% |

两张表都来自同一台 Atlas A3：预热后各测 10 次，取中位数。四个版本的百分比使用同一个参照——这台机器 25 个 Cube 的 FP16 理论峰值，共 378.88 TFLOPS。

## 怎么运行

先准备好已有的 PyTorch、`torch_npu`、CANN 和 TileLang 环境；本仓库不安装这些外部依赖。TileLang 使用 [PR #1852](https://github.com/tile-ai/tilelang-ascend/pull/1852)，具体版本见后面的 Environment 部分。

将 `env.local.sh.example` 复制为 `env.local.sh`，填入已有 TileLang 环境脚本的路径，然后在 Bash 中运行：

```bash
source env.sh
npu-smi info
# 选择空闲设备；这里以设备 0 为例。
python -m core.bench --cases all --devices 0
```

这会运行四个版本，检查结果，并生成本地计时报告。默认配置使用 24 组计算单元。

**要使用上表的测试配置，在本机已验证的 25-Cube 设备上运行：**

```bash
FA_LOGICAL_BLOCKS=0 python -m core.bench --cases all --devices 0
```

这个设置只改变两个 CCE 版本的任务分配，允许许多小任务接续执行。代码会先检查设备是否适用；TileLang 仍使用 24 组。上表使用的就是这个配置，因此默认命令的 CCE 用时可能不同。

以下是供查看代码和调整实现时使用的细节。

<details>
<summary>测试参数与峰值比例公式</summary>

测试环境：Atlas A3 `Ascend910_9392`、CANN 9.1、TileLang PR #1852 提交 `77a444b2`。每轮预热 3 次，按正反顺序各测 5 次，保留全部 10 个样本。输入为非因果 attention，12 个 query heads、1 个 KV head，head dimension 为 128。

仅统计 QK 和 PV 两次矩阵乘法的 FLOPs：

```text
QK/PV FLOPs = 4 * B * Hq * Nq * Nk * D
peak ratio = QK/PV FLOPs / (time_seconds * 378.88e12)
nominal peak = 25 * 1.85e9 * 8192 FLOP/cycle
```

选中的 TileLang 配置使用三个工作区槽（ns3），通过 `reduce_max`、减法、`exp` 和 `reduce_sum` 组合完成在线 softmax，不需要额外的融合 softmax 接口。

</details>

A small benchmark framework for non-causal, FP16-input grouped-query attention
on Ascend A3 (`ascend910_93`), with head dimension 128. It contains four selected
implementations, their source code, and a common correctness/timing runner.
No raw benchmark output files, compiled binaries, external dependencies, or dependency
installation scripts are included.

## Implementations

| Backend | Implementation |
|---|---|
| `npu_fa` | `torch_npu.npu_fusion_attention`, native GQA; correctness reference |
| `tl_stream_v3` | TL v3 with three workspace slots (ns3) and composed reduce-max/subtract/exp/reduce-sum softmax |
| `cce_fa_nqkq_nz_resident_v13_q768_pool_i4_p128` | CCE Normal-NZ Q-K-Q, Q_L1=768, WS_Q=256, WS_K=512; resident FP16 O |
| `cce_fa_nk_nz_vector_fp32_v3` | CCE Normal-NZ K-first, Q_L1=WS_Q=256, WS_K=512; resident FP32 O; no BAR.V in the Vector payload |

All four return normalized attention output in FP16 with shape `[B,Hq,Nq,128]`.
The CCE FP32 variant uses FP32 Vector arithmetic except FP16 max comparisons;
Cube communication (S, P and partial O) remains FP16. The FP16 variant can
overflow for sufficiently large unnormalized accumulators.

The TL entry implements online softmax with TileLang-Ascend reduction,
broadcast, subtraction and exponential primitives. Running maximum, denominator
updates and delayed O accumulation are included. It needs no added fused-softmax
operator or compiler patch. The CCE projects are self-contained operator sources;
they do not import generators or other backends from the research repository.

## Environment

Provide an existing working environment with Python, PyTorch, `torch_npu`,
CANN/BiSheng, CMake, and (for the TL entry only) compatible TileLang-Ascend.
The CCE sources target the CANN 9.1 / dav-c220 API. Dependencies are not managed
by this repository.

Use the TileLang-Ascend implementation from
[PR #1852](https://github.com/tile-ai/tilelang-ascend/pull/1852), revision
`77a444b2b7e5b721e976d2918a3923155d80ca3f`, which provides the refactored
FP32 `T.reduce_max` / `T.reduce_sum` implementation. The TL entry uses its
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
  --backends npu_fa,cce_fa_nk_nz_vector_fp32_v3 \
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
  --backends cce_fa_nqkq_nz_resident_v13_q768_pool_i4_p128,cce_fa_nk_nz_vector_fp32_v3 \
  --cases all --devices 0
```

`0` selects the original automatic logical grid. This opt-in performs a device
identity/data qualification and rejects an incompatible physical mapping.
Do not assume that another Ascend deployment exposes 25 Cube cores.

## Source layout and checks

```text
backends/    four implementations and local CANN projects
core/        discovery, build/load drivers, checks and profiling
problem/     canonical inputs and workloads
tests/       host-only framework regressions
```

Edit TL's `impl_tl.py`, then regenerate its runnable file:

```bash
python backends/tl_stream_v3/preprocess.py \
  backends/tl_stream_v3/impl_tl.py -o backends/tl_stream_v3/impl.py
python -m unittest discover -s tests -v
```

The CCE build drivers use source hashes and keep local artifacts under each
project's ignored `.build/` directory. They do not install a system-wide OPP
package. Keep package names distinct when adding another backend.
