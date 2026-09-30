# Flash Attention on Ascend

A small benchmark framework for non-causal, FP16-input grouped-query attention
on Ascend A3 (`ascend910_93`), with head dimension 128. It contains four selected
implementations, their source code, and a common correctness/timing runner.
No benchmark results, compiled binaries, external dependencies, or dependency
installation scripts are included.

## Implementations

| Backend | Implementation |
|---|---|
| `npu_fa` | `torch_npu.npu_fusion_attention`, native GQA; correctness reference |
| `tl_stream_v3` | TL v3 with two pipeline stages and composed reduce-max/subtract/exp/reduce-sum softmax |
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
