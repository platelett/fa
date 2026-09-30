"""npu_fa -- torch_npu's fused flash-attention op, as the reference backend.

Pure declaration: how to adapt the canonical inputs to the op and how to call it.
The op (`torch_npu.npu_fusion_attention`) handles GQA natively via head_num, so
the canonical q/k/v are passed verbatim (no repeat_interleave). It matches an
fp32 `scaled_dot_product_attention` reference to within ~1e-4, and profiles as
the `FlashAttentionScore` op.

Running the op uses the external driver; it needs only torch_npu (see env.sh).
"""

from core.contract import Meta

META = Meta(
    name="npu_fa",
    type="external",
    op_type="FlashAttentionScore",
    is_reference=True,
    author="torch_npu",
    date="2026-07-01",
    notes="torch_npu.npu_fusion_attention (fp16, BNSD, non-causal, native GQA); "
          "aligned with fp32 scaled_dot_product_attention within 1e-2.",
)


def adapt(case, canonical):
    return {
        "q": canonical["q"],
        "k": canonical["k"],
        "v": canonical["v"],
        "heads_q": case.heads_q,
        "scale": case.scale,
    }


def call(op_args):
    import torch_npu

    return torch_npu.npu_fusion_attention(
        op_args["q"], op_args["k"], op_args["v"], op_args["heads_q"],
        padding_mask=None,
        atten_mask=None,
        scale=op_args["scale"],
        keep_prob=1.0,
        input_layout="BNSD",
        pre_tockens=65535,
        next_tockens=65535,
        sparse_mode=0,
    )[0]
