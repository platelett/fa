"""Canonical flash-attention inputs -- backend-agnostic, a pure function of the Case.

Both backends consume the *same* canonical tensors, so their outputs are directly
comparable. Depends only on Case, never on any backend's tiling config.

Shapes (non-causal full attention, GQA via heads_q / heads_kv):
  q  [batch, heads_q,  seq_len, dim]
  k  [batch, heads_kv, seq_len, dim]
  v  [batch, heads_kv, seq_len, dim]

Dtype is float16 (the tilelang kernel's dtype); the reference consumes the same
tensors and handles GQA natively.
"""

import torch


def build_canonical(case, seed: int = 0) -> dict:
    """Allocate the canonical q/k/v inputs on the NPU."""
    torch.manual_seed(seed)
    torch.set_default_device("npu")
    rand = lambda *shape: torch.randn(shape, dtype=torch.float16)

    return {
        "q": rand(case.batch, case.heads_q, case.seq_len, case.dim),
        "k": rand(case.batch, case.heads_kv, case.seq_len, case.dim),
        "v": rand(case.batch, case.heads_kv, case.seq_len, case.dim),
    }
