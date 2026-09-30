"""The flash-attention test matrix -- the *problem*, decoupled from any backend.

A Case carries only logical workload dimensions (no tiling, no derived kernel
constants). It is the shared, editable definition every backend must solve:
one non-causal (GQA) flash-attention forward over a full [B, H, S, D] tensor.

The three cases mirror fa_opt/bench.sh (zip mode).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    batch: int                   # B
    seq_len: int                 # S (both query and kv length; non-causal full attention)
    heads_q: int                 # query heads
    heads_kv: int                # kv heads (GQA: heads_q % heads_kv == 0)
    dim: int = 128               # head dim (kernel requires 128)

    @property
    def group_size(self) -> int:
        return self.heads_q // self.heads_kv

    @property
    def scale(self) -> float:
        return self.dim ** -0.5

    @property
    def name(self) -> str:
        return f"B{self.batch}_S{self.seq_len}_Q{self.heads_q}_KV{self.heads_kv}_D{self.dim}"


# The 3-case matrix, from fa_opt/bench.sh (--B / --S / --q-heads / --kv-heads / --D, zip).
CASES = (
    Case(2, 131072, 12, 1, 128),
    Case(2, 65536, 12, 1, 128),
    Case(1, 32768, 12, 1, 128),
)


def index_of(case: Case) -> int:
    return CASES.index(case)
