import os

import tilelang
from tilelang import language as T
from tilelang.intrinsics import make_zn_layout, make_nz_layout

if os.environ.get("FA_DISABLE_CACHE", "0") == "1":
    tilelang.disable_cache()

# ===========================================================================
# Lock declarations (preprocessor expands to T.set_flag / T.wait_flag)
# ===========================================================================

# Cube scope
LOCK A_L1   0 MTE2 MTE1 2
LOCK KV_L1  2 MTE2 MTE1
LOCK L0AB   3 MTE1 M     2
LOCK L0C    5 M    FIX   2

# Vector scope: shared load/store buffers across softmax and O_acc
LOCK LD_BUF 0 MTE2 V
LOCK ST_BUF 1 V    MTE3
LOCK ST_ON  2 V    MTE3
LOCK OA_WB  3 MTE2 MTE3

# ===========================================================================
# Constants
# ===========================================================================
NUM_CORES = 24          # 910B AI Cores
DIM = 128               # head dimension (fixed)

TILE_Q_L2 = 256         # query tile cached in L2
TILE_KV_L1 = 1024       # key/value tile staged in L1
TILE_KV_L2 = 1024       # key/value tile at L2 granularity

TILE_M = 128            # L0 tile on M (query) axis
TILE_K = 128            # L0 tile on contraction / output-split axis

TILE_Q_UB = 8           # narrow strip for softmax (transcendentals)
TILE_Q_ACC = 64         # wide strip for O_acc update (MACs)

NUM_STAGES = 3          # pipeline depth / token-ring size
NUM_L1_CHUNKS = TILE_KV_L2 // TILE_KV_L1   # currently 1

# L0 iteration counts (derived, used in GEMM macros)
M_ITERS = TILE_Q_L2 // TILE_M    # 2  — m-tiles per GEMM invocation
N_ITERS = TILE_KV_L1 // TILE_K   # 8  — output-split tiles (GEMM1)
K_ITERS = TILE_KV_L1 // TILE_K   # 8  — contraction tiles  (GEMM2)

# Cross-core semaphore IDs
SEM_CUBE = 0    # Vector → Cube: "P ready" / "slot free"
SEM_VEC  = 1    # Cube → Vector: "S ready" / "O ready"

# SoftmaxFlashV2 block alignment: float32 → 8 elements per 32-byte block
ELEM_PER_BLK = 8
SFM_WORKSPACE_BYTES = 2304  # M=8: 8 * (8+64) * 4 bytes

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
}


# ===========================================================================
# Layer 2: Building-block macros
# ===========================================================================

# ---------------------------------------------------------------------------
# gemm_output_split: Q[TILE_Q_L2, DIM] @ K[TILE_KV_L1, DIM]^T → S[TILE_Q_L2, TILE_KV_L1]
# ---------------------------------------------------------------------------
@T.macro(hygienic=False)
def gemm_output_split(
    a_src, a_i0, a_offset,
    b_src, b_i0, b_offset,
    out, out_i0, out_i1,
    a_l1, kv_l1, l0a, l0b, l0c,
):
    # Phase 1: bulk-load all K tiles to L1
    ACQ MTE2 KV_L1
    for ni in T.serial(N_ITERS):
        T.copy(b_src[b_i0, b_offset + ni * TILE_K : b_offset + (ni + 1) * TILE_K, :], kv_l1[ni, :, :])
    REL MTE2 KV_L1

    # Phase 2: tiled MMA (Q loaded per m-tile, a_l1 double-buffered on mi)
    ACQ MTE1 KV_L1
    for mi in T.serial(M_ITERS):
        mi_side = mi % 2
        LOCKED MTE2 A_L1(mi_side): T.copy(a_src[a_i0, a_offset + mi * TILE_M : a_offset + (mi + 1) * TILE_M, :], a_l1[mi_side, :, :])

        ACQ MTE1 A_L1(mi_side)
        for ni in T.serial(N_ITERS):
            side = ni % 2
            ACQ MTE1 L0AB(side)
            if ni < 2:
                T.copy(a_l1[mi_side, :, :], l0a[side, :, :])
            T.copy(kv_l1[ni, :, :], l0b[side, :, :], transpose=True)
            REL MTE1 L0AB(side)

            LOCKED M L0AB(side), L0C(side): T.mma(l0a[side, :, :], l0b[side, :, :], l0c[side, :, :], init=True)
            LOCKED FIX L0C(side): T.copy(l0c[side, :, :], out[out_i0, out_i1, mi * TILE_M : (mi + 1) * TILE_M, ni * TILE_K : (ni + 1) * TILE_K])
        REL MTE1 A_L1(mi_side)
    REL MTE1 KV_L1

# ---------------------------------------------------------------------------
# gemm_contraction_split: P[TILE_Q_L2, TILE_KV_L1] @ V[TILE_KV_L1, DIM] → O[TILE_Q_L2, DIM]
# ---------------------------------------------------------------------------
@T.macro(hygienic=False)
def gemm_contraction_split(
    a_src, a_i0, a_i1,
    b_src, b_i0, b_offset,
    out, out_i0, out_i1,
    a_l1, kv_l1, l0a, l0b, l0c,
):
    # Phase 1: load all V tiles to L1
    ACQ MTE2 KV_L1
    for ki in T.serial(K_ITERS):
        T.copy(b_src[b_i0, b_offset + ki * TILE_K : b_offset + (ki + 1) * TILE_K, :], kv_l1[ki, :, :])
    REL MTE2 KV_L1

    # Phase 2: contraction-split tiled MMA (a_l1 double-buffered on ki)
    ACQ MTE1 KV_L1
    for mi in T.serial(M_ITERS):
        c_side = mi % 2
        ACQ M L0C(c_side)
        for ki in T.serial(K_ITERS):
            side = ki % 2
            LOCKED MTE2 A_L1(side): T.copy(a_src[a_i0, a_i1, mi * TILE_M : (mi + 1) * TILE_M, ki * TILE_K : (ki + 1) * TILE_K], a_l1[side, :, :])

            ACQ MTE1 L0AB(side)
            LOCKED MTE1 A_L1(side): T.copy(a_l1[side, :, :], l0a[side, :, :])
            T.copy(kv_l1[ki, :, :], l0b[side, :, :])
            REL MTE1 L0AB(side)

            LOCKED M L0AB(side): T.mma(l0a[side, :, :], l0b[side, :, :], l0c[c_side, :, :], init=(ki == 0))
        REL M L0C(c_side)
        LOCKED FIX L0C(c_side): T.copy(l0c[c_side, :, :], out[out_i0, out_i1, mi * TILE_M : (mi + 1) * TILE_M, :])
    REL MTE1 KV_L1
# ===========================================================================
# Layer 3: Task macros (Vector scope)
# ===========================================================================

HALF_Q = TILE_Q_L2 // 2
SOFTMAX_STRIPS = HALF_Q // TILE_Q_UB       # 16
O_ACC_STRIPS   = HALF_Q // TILE_Q_ACC      # 2

# ---------------------------------------------------------------------------
# vec_softmax: TASK 2 — S → P + stats + early release
#
# Phase 2A: 16 narrow strips [TILE_Q_UB=8, TILE_KV_L2=1024]
#   load S → SoftmaxFlashV2 → store P
# Early release: signal Cube after all P stored
# Phase 2B: scale_ring for O_acc rescaling — no DMA
# ---------------------------------------------------------------------------
@T.macro(hygienic=False)
def vec_softmax(step, my_start, num_q_blocks, num_kv_blocks, num_q_stages, sm_scale):
    stage = step % NUM_STAGES
    q_task = step // num_kv_blocks
    global_q = my_start + q_task
    kv_idx = step % num_kv_blocks
    q_ring = global_q % num_q_stages

    if kv_idx == 0:
        T.tile.fill(m_stats, -(2 ** 30))
        T.tile.fill(l_stats_ring[q_ring, :, :], 0.0)

    # --- Phase 2A: narrow strip softmax via SoftmaxFlashV2 ---
    for r in T.serial(SOFTMAX_STRIPS):
        row = vid * HALF_Q + r * TILE_Q_UB

        # MTE2: load S strip from workspace
        LOCKED MTE2 LD_BUF: T.copy(ws_sp[cid, stage, row : row + TILE_Q_UB, :], ld_score)

        # V: fp16 → fp32
        LOCKED V LD_BUF: T.copy(ld_score, score_local)

        # Pre-scale scores for SoftmaxFlashV2
        T.tile.mul(score_local, score_local, sm_scale)

        # SoftmaxFlashV2: P = exp(scaled_S - max), in-place (isReuseSource).
        # expMax = exp(old_max - new_max) is written straight into scale_ring
        # (the O_acc rescale factor) — no separate Phase 2B recompute needed.
        T.softmax_flash_v2(
            score_local,
            l_stats_ring[q_ring, r * TILE_Q_UB : (r + 1) * TILE_Q_UB, :],
            m_stats[r * TILE_Q_UB : (r + 1) * TILE_Q_UB, :],
            score_local,
            scale_ring[stage, r * TILE_Q_UB : (r + 1) * TILE_Q_UB, :],
            l_stats_ring[q_ring, r * TILE_Q_UB : (r + 1) * TILE_Q_UB, :],
            m_stats[r * TILE_Q_UB : (r + 1) * TILE_Q_UB, :],
            is_update=True,
            is_reuse_source=True,
            tmp=sfm_tmp,
        )

        # V → MTE3: store P strip
        LOCKED V ST_BUF: T.copy(score_local, st_prob)
        LOCKED MTE3 ST_BUF: T.copy(st_prob, ws_sp[cid, stage, row : row + TILE_Q_UB, :])

    # --- Early release: P is complete, let Cube start P@V ---
    T.set_cross_flag("MTE3", SEM_CUBE)

# ---------------------------------------------------------------------------
# vec_o_acc: TASK 1 — O_acc update (consume O_partial)
#
# 2 wide strips [TILE_Q_ACC=64, DIM=128]:
#   load O_acc + O_partial → o_acc = o_acc * scale + o_partial → store O_acc
#   If last KV block: normalize by l_stats and store to GM
# ---------------------------------------------------------------------------
@T.macro(hygienic=False)
def vec_o_acc(step, my_start, num_q_blocks, num_kv_blocks, num_q_stages):
    old_step = step - NUM_STAGES
    stage = step % NUM_STAGES
    old_q_task = old_step // num_kv_blocks
    old_global_q = my_start + old_q_task
    bh = old_global_q // num_q_blocks
    local_q = old_global_q % num_q_blocks
    kv_idx = old_step % num_kv_blocks
    q_ring = old_global_q % num_q_stages

    for r in T.serial(O_ACC_STRIPS):
        row = vid * HALF_Q + r * TILE_Q_ACC

        if kv_idx == 0:
            LOCKED MTE2 LD_BUF: T.copy(ws_op[cid, stage, row : row + TILE_Q_ACC, :], ld_o_partial)

            ACQ V ST_BUF
            LOCKED V LD_BUF: T.copy(ld_o_partial, st_o_acc)
        else:
            ACQ MTE2 LD_BUF
            T.copy(ws_oa[cid, row : row + TILE_Q_ACC, :], ld_o_acc)
            T.copy(ws_op[cid, stage, row : row + TILE_Q_ACC, :], ld_o_partial)
            REL MTE2 LD_BUF

            ACQ V LD_BUF
            ACQ V ST_BUF
            T.tile.row_expand_mul_experiment(
                st_o_acc, ld_o_acc,
                scale_ring[stage, r * TILE_Q_ACC : (r + 1) * TILE_Q_ACC, :],
            )
            T.tile.add(st_o_acc, st_o_acc, ld_o_partial)
            REL V LD_BUF

        if kv_idx == num_kv_blocks - 1:
            T.tile.row_expand_div_experiment(
                st_o_acc, st_o_acc,
                l_stats_ring[q_ring, r * TILE_Q_ACC : (r + 1) * TILE_Q_ACC, :],
            )

            q_row = local_q * TILE_Q_L2 + row
            LOCKED V ST_ON: T.copy(st_o_acc, st_o_norm)
            LOCKED MTE3 ST_ON: T.copy(st_o_norm, O_bh[bh, q_row : q_row + TILE_Q_ACC, :])

        REL V ST_BUF
        ACQ MTE3 ST_BUF
        if kv_idx != num_kv_blocks - 1:
            T.copy(st_o_acc, ws_oa[cid, row : row + TILE_Q_ACC, :])
        REL MTE3 ST_BUF
# ===========================================================================

# ===========================================================================
# Main kernel
# ===========================================================================
@tilelang.jit(out_idx=[3], workspace_idx=[4, 5, 6], pass_configs=pass_configs)
def flash_attention_fwd(
    kernel_name,
    batch,
    seq_len,
    heads_q,
    heads_kv,
    dim,
):
    assert heads_q % heads_kv == 0
    assert dim == DIM
    assert seq_len % TILE_KV_L2 == 0
    assert seq_len % TILE_Q_L2 == 0

    dtype = "float16"
    accum_dtype = "float"

    sm_scale = (1.0 / dim) ** 0.5

    shape_q = [batch, heads_q, seq_len, dim]
    shape_kv = [batch, heads_kv, seq_len, dim]

    # --- GQA → MHA fold (identical to tl_expert) ---
    q_per_kv = heads_q // heads_kv
    num_bh = batch * heads_kv
    q_seq = seq_len * q_per_kv
    kv_seq = seq_len

    # --- Task geometry ---
    num_q_blocks = q_seq // TILE_Q_L2
    num_kv_blocks = kv_seq // TILE_KV_L2
    global_q_tasks = num_bh * num_q_blocks
    total_tasks = global_q_tasks * num_kv_blocks

    num_q_stages = 1 + (NUM_STAGES + num_kv_blocks - 1) // num_kv_blocks

    q_tasks_per_core = global_q_tasks // NUM_CORES
    r_tasks = global_q_tasks % NUM_CORES

    @T.prim_func
    def main(
        Q: T.Tensor(shape_q, dtype),
        K: T.Tensor(shape_kv, dtype),
        V: T.Tensor(shape_kv, dtype),
        Output: T.Tensor(shape_q, dtype),
        ws_sp: T.Tensor([NUM_CORES, NUM_STAGES, TILE_Q_L2, TILE_KV_L2], dtype),
        ws_op: T.Tensor([NUM_CORES, NUM_STAGES, TILE_Q_L2, DIM], accum_dtype),
        ws_oa: T.Tensor([NUM_CORES, TILE_Q_L2, DIM], accum_dtype),
    ):
        T.func_attr({"global_symbol": kernel_name})
        with T.Kernel(NUM_CORES, is_npu=True) as (cid, vid):
            # --- GQA reshape aliases (no copy) ---
            Q_bh = T.decl_buffer([num_bh, q_seq, dim], dtype, data=Q.data, scope="global")
            K_bh = T.decl_buffer([num_bh, kv_seq, dim], dtype, data=K.data, scope="global")
            V_bh = T.decl_buffer([num_bh, kv_seq, dim], dtype, data=V.data, scope="global")
            O_bh = T.decl_buffer([num_bh, q_seq, dim], dtype, data=Output.data, scope="global")

            # --- L1 buffers (Cube data path) ---
            a_l1  = T.alloc_L1([2, TILE_M, TILE_K], dtype)
            kv_l1 = T.alloc_L1([N_ITERS, TILE_K, DIM], dtype)

            T.annotate_layout(
                {
                    a_l1:  make_zn_layout(a_l1),
                    kv_l1: make_nz_layout(kv_l1),
                }
            )

            # --- L0 buffers (Cube compute, double-buffered) ---
            l0a = T.alloc_L0A([2, TILE_M, TILE_K], dtype)
            l0b = T.alloc_L0B([2, TILE_K, DIM], dtype)
            l0c = T.alloc_L0C([2, TILE_M, DIM], accum_dtype)

            # --- UB buffers (Vector scope, per sub-core: 192KB each) ---
            #
            # [0K,   32K)  work  — score_local: SoftmaxFlashV2 in-place src/dst
            # [32K,  48K)  store — st_o_norm: final O output (fp16)
            # [48K,  80K)  store — st_prob / st_o_acc (alias, fp16/fp32)
            # [80K,  96K)  load  — ld_score (fp16, alias ld_o_acc)
            # [80K, 112K)  load  — ld_o_acc (fp32, alias ld_score)
            # [112K,144K)  load  — ld_o_partial (fp32)
            # [144K,~160K) auto  — stats + SoftmaxFlashV2 internals

            # --- Work [0K, 32K) — V-pipe only, no Lock ---
            score_local = T.alloc_ub([TILE_Q_UB, TILE_KV_L2], accum_dtype)   # 32KB

            # --- Store [32K, 80K) — ST_ON + ST_BUF Lock ---
            st_o_norm   = T.alloc_ub([TILE_Q_ACC, DIM], dtype)               # 16KB
            st_prob     = T.alloc_ub([TILE_Q_UB, TILE_KV_L2], dtype)         # 16KB
            st_o_acc    = T.alloc_ub([TILE_Q_ACC, DIM], accum_dtype)          # 32KB (alias st_prob)

            # --- Load [80K, 144K) — LD_BUF Lock ---
            ld_score    = T.alloc_ub([TILE_Q_UB, TILE_KV_L2], dtype)         # 16KB
            ld_o_acc    = T.alloc_ub([TILE_Q_ACC, DIM], accum_dtype)          # 32KB (alias ld_score)
            ld_o_partial = T.alloc_ub([TILE_Q_ACC, DIM], accum_dtype)         # 32KB

            T.annotate_address({
                # work [0K, 32K)
                score_local: 0,
                # store [32K, 80K)
                st_o_norm: 32768,
                st_prob: 49152, st_o_acc: 49152,
                # load [80K, 144K)
                ld_score: 81920, ld_o_acc: 81920,
                ld_o_partial: 114688,
            })

            # --- Auto [144K, ~158K) — stats + SoftmaxFlashV2 workspace ---
            # SoftmaxFlashV2 writes expMax straight into scale_ring, so no
            # m_stats_old / sfm_exp_max buffers are needed.
            m_stats      = T.alloc_ub([HALF_Q, ELEM_PER_BLK], accum_dtype)
            l_stats_ring = T.alloc_ub([num_q_stages, HALF_Q, ELEM_PER_BLK], accum_dtype)
            scale_ring   = T.alloc_ub([NUM_STAGES, HALF_Q, ELEM_PER_BLK], accum_dtype)
            sfm_tmp      = T.alloc_ub([SFM_WORKSPACE_BYTES], "uint8")

            my_start = cid * q_tasks_per_core + T.if_then_else(cid < r_tasks, cid, r_tasks)
            my_count = q_tasks_per_core + T.if_then_else(cid < r_tasks, 1, 0)
            my_total_steps = my_count * num_kv_blocks

            # ===============================================================
            # Cube scope
            # ===============================================================
            with T.Scope("C"):
                INIT_LOCKS A_L1, KV_L1, L0AB, L0C

                for step in T.serial(my_total_steps + NUM_STAGES):
                    T.wait_cross_flag(SEM_CUBE)
                    stage = step % NUM_STAGES

                    if step >= NUM_STAGES:
                        old_step = step - NUM_STAGES
                        bh_a = (my_start + old_step // num_kv_blocks) // num_q_blocks
                        kv_off_a = (old_step % num_kv_blocks) * TILE_KV_L2
                        gemm_contraction_split(
                            ws_sp, cid, stage,
                            V_bh, bh_a, kv_off_a,
                            ws_op, cid, stage,
                            a_l1, kv_l1, l0a, l0b, l0c,
                        )

                    if step < my_total_steps:
                        global_q = my_start + step // num_kv_blocks
                        bh_b = global_q // num_q_blocks
                        q_off = (global_q % num_q_blocks) * TILE_Q_L2
                        kv_off_b = (step % num_kv_blocks) * TILE_KV_L2
                        gemm_output_split(
                            Q_bh, bh_b, q_off,
                            K_bh, bh_b, kv_off_b,
                            ws_sp, cid, stage,
                            a_l1, kv_l1, l0a, l0b, l0c,
                        )

                    T.set_cross_flag("FIX", SEM_VEC)

                DESTROY_LOCKS A_L1, KV_L1, L0AB, L0C

            # ===============================================================
            # Vector scope
            # ===============================================================
            with T.Scope("V"):
                for _ in range(NUM_STAGES):
                    T.set_cross_flag("MTE2", SEM_CUBE)
                INIT_LOCKS LD_BUF, ST_BUF, ST_ON, OA_WB

                for step in T.serial(my_total_steps + NUM_STAGES):
                    T.wait_cross_flag(SEM_VEC)

                    if step >= NUM_STAGES:
                        ACQ MTE2 OA_WB
                        vec_o_acc(step, my_start, num_q_blocks, num_kv_blocks, num_q_stages)
                        REL MTE3 OA_WB

                    if step < my_total_steps:
                        vec_softmax(step, my_start, num_q_blocks, num_kv_blocks, num_q_stages, sm_scale)

                DESTROY_LOCKS LD_BUF, ST_BUF, ST_ON, OA_WB

    return main


def make_args(case, canonical):
    return (canonical["q"], canonical["k"], canonical["v"])
