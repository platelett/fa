"""Legacy Expert baseline from TileLang-Ascend ascendc_pto@4c726701."""
from core.contract import Meta
META = Meta(name="expert_v1", type="tilelang", op_type="main_kernel", date="2026-10-08",
            notes="Original expert_v1 kernel from FA PR #1863; only symbol/cache/host contract adapted.")
def kernel(case, *, kernel_name: str):
    from backends.expert_v1 import impl
    return impl.flash_attention_fwd(case.batch, case.seq_len, case.heads_q, case.heads_kv, case.dim, kernel_name=kernel_name)
def assemble(case, canonical):
    from backends.expert_v1 import impl
    return impl.make_args(case, canonical)
