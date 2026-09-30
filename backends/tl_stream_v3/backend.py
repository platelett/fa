"""tl_stream_v3 -- composed online softmax using reduction primitives, fully manual sync."""

from core.contract import Meta

META = Meta(
    name="tl_stream_v3",
    type="tilelang",
    op_type="main_kernel",
    is_reference=False,
    author="platelet",
    date="2026-09-08",
    notes="Three-stage online reduce-max/subtract/exp/reduce-sum softmax; explicit synchronization; no fused softmax extension.",
)


def kernel(case, *, kernel_name: str):
    from backends.tl_stream_v3 import impl

    return impl.flash_attention_fwd(
        kernel_name=kernel_name,
        batch=case.batch,
        seq_len=case.seq_len,
        heads_q=case.heads_q,
        heads_kv=case.heads_kv,
        dim=case.dim,
    )


def assemble(case, canonical):
    from backends.tl_stream_v3 import impl

    return impl.make_args(case, canonical)
