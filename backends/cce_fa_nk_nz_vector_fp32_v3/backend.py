"""Complete attention from the common parameterized resident-O implementation."""

from dataclasses import dataclass, replace
from pathlib import Path
import sys
import os
from core.contract import Meta
from core.drivers.ascendc import Project
from .project import build
from .parameters import MODE, WS_Q, TRANSPOSE, SP_NZ

META = Meta(
    name="cce_fa_nk_nz_vector_fp32_v3",
    type="ascendc",
    op_type="FaFaNkNzVectorFp32V3",
    diagnostic=False,
    notes="Q_L1=256, WS_Q=256, WS_K=512; FP32 O resident across KV; independent complete O and softmax phases; selected public implementation",
)


@dataclass(frozen=True)
class AttentionConfig:
    q_l1: int = 256
    q_block: int = 0
    k_block: int = 512
    groups: int = int(os.environ.get("FA_LOGICAL_BLOCKS", "24"))  # 0: maximum grid up to 8192, with balanced task refill
    capture: bool = False
    phase: int = 0

    def __post_init__(self):
        expected = WS_Q or self.q_l1
        if not self.q_block:
            object.__setattr__(self, "q_block", expected)
        if self.q_l1 != 256 or self.q_block != expected or self.k_block != 512:
            raise ValueError("unsupported panel-mode geometry")
        if not 0 <= self.groups <= 8192 or not isinstance(self.capture, bool) or self.phase != 0:
            raise ValueError("invalid groups/capture/phase")

    def resolved(self, q_shape):
        if self.groups:
            return self
        b, h, nq, _ = q_shape
        groups = min(8192, b * h * ((nq + self.q_l1 - 1) // self.q_l1))
        if not 1 <= groups <= 8192:
            raise ValueError("maximum logical grid exceeds the qualified 1..8192 bound")
        return replace(self, groups=groups)

    transpose = TRANSPOSE
    sp_layout = "NZ" if SP_NZ else "ND"
    packed_sp = SP_NZ
    k_first = True  # compatibility: macro KV traversal; qk_order is authoritative
    qk_order = MODE
    query_alignment = 256 if TRANSPOSE else 128


def project(case):
    root = Path(__file__).with_name("project")
    rel = build.directory().relative_to(root)
    return Project(
        root,
        tuple(
            str(rel / p)
            for p in (
                "libfa_fa_nk_nz_vector_fp32_v3.so",
                "op_host/libfa_fa_nk_nz_vector_fp32_v3_ascendc_cust_optiling.so",
                "op_kernel/ascendc_kernels/binary/config/ascend910_93/binary_info_config.json",
                "build.json",
            )
        ),
        (sys.executable, str(root / "build.py")),
    )


def load(case, project):
    from .binding import Operator

    return Operator(project.artifact_paths()[0], case if isinstance(case, AttentionConfig) else AttentionConfig())


def adapt(case, canonical):
    return tuple(canonical[k] for k in ("q", "k", "v"))
