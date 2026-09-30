# Flash Attention on Ascend

四个版本做的是同一个 attention 计算，输入和输出都是 FP16：

- **Torch**：直接调用 `torch_npu` 提供的 attention 算子。
- **TileLang**：用 TileLang 实现，使用三个流水缓存槽（ns3），基于 PR #1852。
- **CCE FP16**：直接写底层代码，中间的输出累加使用半精度。
- **CCE FP32**：直接写底层代码，向量计算和中间的输出累加使用单精度。

下面是三个输入规模的实测结果。`Batch` 是一次处理的样本数，`Length` 是每个样本的序列长度。

**用时：单位为毫秒，越小越快。**

| Batch | Length | Torch | TileLang ns3 | CCE FP16 | CCE FP32 |
|---:|---:|---:|---:|---:|---:|
| 2 | 131072 | 1207.513 | 1118.169 | 572.472 | 793.552 |
| 2 | 65536 | 300.555 | 276.640 | 143.603 | 199.889 |
| 1 | 32768 | 37.564 | 34.508 | 18.440 | 25.524 |

**达到理论 Cube 峰值的比例：越高，越接近这台机器的矩阵乘法速度上限。**

| Batch | Length | Torch | TileLang ns3 | CCE FP16 | CCE FP32 |
|---:|---:|---:|---:|---:|---:|
| 2 | 131072 | 46.1% | 49.8% | 97.3% | 70.2% |
| 2 | 65536 | 46.3% | 50.4% | 97.0% | 69.7% |
| 1 | 32768 | 46.4% | 50.5% | 94.4% | 68.2% |

例如，97.3% 表示整个 attention 的速度已接近矩阵乘法的理论上限；它不表示 Cube 有 97.3% 的时间在工作。四个版本都按这台机器的同一个上限计算：**25 个 Cube 的 FP16 峰值，合计 378.88 TFLOPS**。CCE FP32 的矩阵乘法仍使用 FP16，因此也采用这个上限。

测试使用 Atlas A3、CANN 9.1 和 TileLang PR #1852。每项预热后测 10 次，表中取中位数；三个输入规模都通过了正确性检查。TileLang 使用 24 组，CCE 使用本机已验证的多 block 模式（`FA_LOGICAL_BLOCKS=0`）。代码默认使用 24 组，性能模式需先通过设备检查。

<details>
<summary>峰值比例的计算方法</summary>

只计 QK 和 PV 两次矩阵乘法的计算量。所有测试均为非因果 attention，12 个 query heads，head dimension 为 128。

```text
QK/PV FLOPs = 4 * B * Hq * Nq * Nk * D
peak ratio = QK/PV FLOPs / (time_seconds * 378.88e12)
nominal peak = 25 * 1.85e9 * 8192 FLOP/cycle
```

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
| `tl_stream_v3` | TL v3 with three pipeline stages and composed reduce-max/subtract/exp/reduce-sum softmax |
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
