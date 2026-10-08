"""ND attention with FP32 Vector math and resident output accumulation."""

from core.contract import Meta

META = Meta(
    name="expert_v3",
    type="tilelang",
    op_type="main_kernel",
    is_reference=False,
    author="platelet",
    date="2026-10-08",
    notes="FP32 Vector and resident O; useful-work scheduling and phased broadcast; ND, UnitFlag, static 24 groups.",
)


def kernel(case, *, kernel_name: str):
    from backends.expert_v3.impl import flash_attention_fwd

    return flash_attention_fwd(
        batch=case.batch,
        seq_len=case.seq_len,
        heads_q=case.heads_q,
        heads_kv=case.heads_kv,
        dim=case.dim,
        kernel_name=kernel_name,
    )


def assemble(case, canonical):
    return canonical["q"], canonical["k"], canonical["v"]
