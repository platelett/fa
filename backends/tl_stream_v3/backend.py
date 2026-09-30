"""TL v3 three-stage SoftmaxFlashV2 configuration."""

from core.contract import Meta

META = Meta(
    name="tl_stream_v3",
    type="tilelang",
    op_type="main_kernel",
    is_reference=False,
    author="platelet",
    date="2026-09-08",
    notes="Three-stage v3 with fused SoftmaxFlashV2; requires the matching TileLang-Ascend API.",
)


def kernel(case, *, kernel_name: str):
    from tilelang import language as T
    if not hasattr(T, "softmax_flash_v2"):
        raise RuntimeError(
            "This TL v3 requires a TileLang-Ascend build exposing T.softmax_flash_v2. "
            "Activate the compatible external environment before running this backend."
        )
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
