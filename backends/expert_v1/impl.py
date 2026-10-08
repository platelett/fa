"""First-generation hand-synchronized Expert Flash Attention example."""

import os
import tilelang
from tilelang import language as T
from tilelang.intrinsics import make_zn_layout, make_nz_layout



if os.environ.get("FA_DISABLE_CACHE", "0") == "1":
    tilelang.disable_cache()


L0_MAX_SIZE = 64 * 1024  # 64KB
NUM_CORES = 24  # 910B has 24 AI Cores

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
}


@tilelang.jit(out_idx=[3], workspace_idx=[4, 5, 6], pass_configs=pass_configs)
def flash_attention_fwd(
    batch,
    seq_len,
    heads_q,
    heads_kv,
    dim,
    num_stages=14,
    cross_interval=2,
    *,
    kernel_name: str,
):
    assert heads_q % heads_kv == 0, "heads_q must be a multiple of heads_kv"
    block_M, block_N = 128, 128
    assert dim == 128, "dim must be 128"
    assert seq_len % block_N == 0, f"seq_len ({seq_len}) must be divisible by block_N ({block_N})"
    assert num_stages % 2 == 0, "num_stages must be even for double buffering"

    dtype = "float16"
    accum_dtype = "float"

    sm_scale = (1.0 / dim) ** 0.5

    shape_q = [batch, heads_q, seq_len, dim]
    shape_kv = [batch, heads_kv, seq_len, dim]

    # Total number of output tiles (logical tasks)
    num_seq_blocks = seq_len // block_M
    block_num = num_seq_blocks * heads_q * batch
    num_iters = T.ceildiv(seq_len, block_N)
    num_outer = T.ceildiv(num_iters, num_stages)

    # ---------------------------------------------------------------------------
    # Static task distribution: evenly split block_num across NUM_CORES.
    #   - Cores 0 .. r-1       get (q+1) tasks each
    #   - Cores r .. NUM_CORES-1 get  q   tasks each
    # where q = block_num // NUM_CORES, r = block_num % NUM_CORES
    # ---------------------------------------------------------------------------
    q_tasks = block_num // NUM_CORES
    r_tasks = block_num % NUM_CORES

    # Cross-core semaphore IDs
    SEM_WS1_C2V = 0
    SEM_WS1_V2C = 1
    SEM_WS2_V2C = 2
    SEM_WS2_C2V = 3
    SEM_WS3_C2V = 4
    SEM_WS3_V2C = 5

    # Local event IDs are allocated per directed pipe pair. Different meanings keep
    # distinct names even when their directed pairs allow the same numeric ID.
    # MTE2 <-> MTE1
    SIG_K_L1 = 0
    SIG_P_L1 = 1
    SIG_V_L1 = 2
    SIG_Q_L1 = 3

    # MTE1 <-> M
    SIG_L0AB = 0  # double-buffer slots 0 and 1

    # M <-> FIX
    SIG_L0C = 0  # double-buffer slots 0 and 1

    # MTE2 <-> V
    SIG_IO_UB = 0

    # V <-> MTE3
    SIG_S_HALF = 0

    def task_range(cid_val):
        """Return (start, count) for core cid_val."""
        start = cid_val * q_tasks + T.if_then_else(cid_val < r_tasks, cid_val, r_tasks)
        count = q_tasks + T.if_then_else(cid_val < r_tasks, 1, 0)
        return start, count

    @T.prim_func
    def main(
        Q: T.Tensor(shape_q, dtype),  # type: ignore
        K: T.Tensor(shape_kv, dtype),  # type: ignore
        V: T.Tensor(shape_kv, dtype),  # type: ignore
        Output: T.Tensor(shape_q, dtype),  # type: ignore
        workspace_1: T.Tensor([NUM_CORES, num_stages, block_M, block_N], dtype),
        workspace_2: T.Tensor([NUM_CORES, num_stages, block_M, block_N], dtype),
        workspace_3: T.Tensor([NUM_CORES, num_stages, block_M, dim], dtype),
    ):
        T.func_attr({"global_symbol": kernel_name})
        with T.Kernel(NUM_CORES, is_npu=True) as (cid, vid):
            q_l1 = T.alloc_L1([block_M, dim], dtype)
            k_l1 = T.alloc_L1([block_N, dim], dtype)
            v_l1 = T.alloc_L1([block_N, dim], dtype)
            p_l1 = T.alloc_L1([block_M, block_N], dtype)

            T.annotate_layout(
                {
                    q_l1: make_zn_layout(q_l1),
                    k_l1: make_nz_layout(k_l1),
                    p_l1: make_zn_layout(p_l1),
                    v_l1: make_zn_layout(v_l1),
                }
            )

            l0a = T.alloc_L0A([2, block_M, dim], dtype)
            l0b = T.alloc_L0B([2, dim, block_N], dtype)
            l0c = T.alloc_L0C([2, block_M, block_N], accum_dtype)

            acc_o = T.alloc_ub([block_M // 2, dim], accum_dtype)

            half_M = block_M // 2

            r_factors = T.alloc_ub([num_stages, block_M // 2, 1], accum_dtype)
            sumexp_is = T.alloc_ub([num_stages, block_M // 2, 1], accum_dtype)

            sumexp = T.alloc_ub([block_M // 2, 1], accum_dtype)
            neg_sm = T.alloc_ub([2, block_M // 2, 1], accum_dtype)

            io_buf = T.alloc_ub([block_M // 2, block_N], dtype)
            acc_s_half = T.alloc_ub([block_M // 2, block_N], dtype)

            work_ub = T.alloc_ub([block_M // 2, block_N], accum_dtype)
            buf_2d = T.alloc_ub([block_M // 2, block_N], accum_dtype)

            my_start, my_count = task_range(cid)

            with T.Scope("C"):
                T.set_cross_flag("MTE2", SEM_WS2_C2V)
                # init: pretend consumer already released
                T.set_flag("MTE1", "MTE2", SIG_K_L1)
                T.set_flag("MTE1", "MTE2", SIG_P_L1)
                T.set_flag("MTE1", "MTE2", SIG_V_L1)
                T.set_flag("MTE1", "MTE2", SIG_Q_L1)
                T.set_flag("M", "MTE1", SIG_L0AB)
                T.set_flag("M", "MTE1", SIG_L0AB + 1)
                T.set_flag("FIX", "M", SIG_L0C)
                T.set_flag("FIX", "M", SIG_L0C + 1)

                for t in T.serial(my_count):
                    task_id = my_start + t
                    bx = task_id % num_seq_blocks
                    by = (task_id // num_seq_blocks) % heads_q
                    bz = task_id // (num_seq_blocks * heads_q)
                    kv_by = by // (heads_q // heads_kv)

                    T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
                    T.copy(Q[bz, by, bx * block_M : (bx + 1) * block_M, :], q_l1)
                    T.set_flag("MTE2", "MTE1", SIG_Q_L1)
                    T.wait_flag("MTE2", "MTE1", SIG_Q_L1)

                    for k in T.serial(num_outer):
                        _remaining = num_iters - k * num_stages
                        batch_iters = T.if_then_else(_remaining < num_stages, _remaining, num_stages)

                        # --- GEMM1: produce S into ws1 ---
                        T.wait_cross_flag(SEM_WS1_V2C)
                        for i in T.serial(batch_iters):
                            side = i % 2
                            idx = k * num_stages + i

                            T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                            T.copy(K[bz, kv_by, idx * block_N : (idx + 1) * block_N, :], k_l1)
                            T.set_flag("MTE2", "MTE1", SIG_K_L1)

                            T.wait_flag("M", "MTE1", SIG_L0AB + side)
                            if i < 2:
                                T.copy(q_l1, l0a[side, :, :])

                            T.wait_flag("MTE2", "MTE1", SIG_K_L1)
                            T.copy(k_l1, l0b[side, :, :], transpose=True)
                            T.set_flag("MTE1", "MTE2", SIG_K_L1)
                            T.set_flag("MTE1", "M", SIG_L0AB + side)

                            T.wait_flag("MTE1", "M", SIG_L0AB + side)
                            T.wait_flag("FIX", "M", SIG_L0C + side)
                            T.mma(l0a[side, :, :], l0b[side, :, :], l0c[side, :, :], init=True)
                            T.set_flag("M", "MTE1", SIG_L0AB + side)
                            T.set_flag("M", "FIX", SIG_L0C + side)

                            T.wait_flag("M", "FIX", SIG_L0C + side)
                            T.copy(l0c[side, :, :], workspace_1[cid, i, :, :])
                            T.set_flag("FIX", "M", SIG_L0C + side)
                            if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
                                T.set_cross_flag("FIX", SEM_WS1_C2V)

                        # --- GEMM2: consume P from ws2, produce O into ws3 ---
                        T.wait_cross_flag(SEM_WS3_V2C)
                        for i in T.serial(batch_iters):
                            side = i % 2
                            idx = k * num_stages + i

                            T.wait_flag("MTE1", "MTE2", SIG_V_L1)
                            T.copy(V[bz, kv_by, idx * block_N : (idx + 1) * block_N, :], v_l1)
                            T.set_flag("MTE2", "MTE1", SIG_V_L1)

                            T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                            if i % cross_interval == 0:
                                T.wait_cross_flag(SEM_WS2_V2C)
                            T.copy(workspace_2[cid, i, :, :], p_l1)
                            T.set_flag("MTE2", "MTE1", SIG_P_L1)

                            T.wait_flag("MTE2", "MTE1", SIG_V_L1)
                            T.wait_flag("M", "MTE1", SIG_L0AB + side)
                            T.copy(v_l1, l0b[side, :, :])
                            T.set_flag("MTE1", "MTE2", SIG_V_L1)

                            T.wait_flag("MTE2", "MTE1", SIG_P_L1)
                            T.copy(p_l1, l0a[side, :, :])
                            T.set_flag("MTE1", "MTE2", SIG_P_L1)
                            T.set_flag("MTE1", "M", SIG_L0AB + side)

                            T.wait_flag("MTE1", "M", SIG_L0AB + side)
                            T.wait_flag("FIX", "M", SIG_L0C + side)
                            T.mma(l0a[side, :, :], l0b[side, :, :], l0c[side, :, :], init=True)
                            T.set_flag("M", "MTE1", SIG_L0AB + side)
                            T.set_flag("M", "FIX", SIG_L0C + side)

                            T.wait_flag("M", "FIX", SIG_L0C + side)
                            T.copy(l0c[side, :, :], workspace_3[cid, i, :, :])
                            T.set_flag("FIX", "M", SIG_L0C + side)
                            if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
                                T.set_cross_flag("FIX", SEM_WS3_C2V)

                        T.set_cross_flag("MTE2", SEM_WS2_C2V)

                    # MTE1 no longer reads q_l1; return it before the next task reloads Q.
                    T.set_flag("MTE1", "MTE2", SIG_Q_L1)

                # destroy: consume outstanding init-direction flags
                T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                T.wait_flag("MTE1", "MTE2", SIG_V_L1)
                T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
                T.wait_flag("M", "MTE1", SIG_L0AB)
                T.wait_flag("M", "MTE1", SIG_L0AB + 1)
                T.wait_flag("FIX", "M", SIG_L0C)
                T.wait_flag("FIX", "M", SIG_L0C + 1)

            with T.Scope("V"):
                T.set_cross_flag("MTE2", SEM_WS1_V2C)
                T.set_cross_flag("MTE2", SEM_WS3_V2C)
                # init
                T.set_flag("V", "MTE2", SIG_IO_UB)
                T.set_flag("MTE3", "V", SIG_S_HALF)

                for t in T.serial(my_count):
                    task_id = my_start + t
                    bx = task_id % num_seq_blocks
                    by = (task_id // num_seq_blocks) % heads_q
                    bz = task_id // (num_seq_blocks * heads_q)

                    T.tile.fill(acc_o, 0.0)
                    T.tile.fill(sumexp, 0.0)
                    T.tile.fill(neg_sm, 2**30)

                    for k in T.serial(num_outer):
                        _remaining = num_iters - k * num_stages
                        batch_iters = T.if_then_else(_remaining < num_stages, _remaining, num_stages)

                        # --- softmax batch ---
                        T.wait_cross_flag(SEM_WS2_C2V)
                        for i in T.serial(batch_iters):
                            cur = i % 2
                            prv = 1 - cur

                            T.wait_flag("V", "MTE2", SIG_IO_UB)
                            if i % cross_interval == 0:
                                T.wait_cross_flag(SEM_WS1_C2V)
                            T.copy(workspace_1[cid, i, vid * half_M : vid * half_M + half_M, :], io_buf)
                            T.set_flag("MTE2", "V", SIG_IO_UB)

                            T.wait_flag("MTE2", "V", SIG_IO_UB)
                            T.copy(io_buf, work_ub)
                            T.set_flag("V", "MTE2", SIG_IO_UB)

                            T.reduce_max(work_ub, neg_sm[cur, :, :], dim=-1)
                            T.tile.mul(neg_sm[cur, :, :], neg_sm[cur, :, :], -sm_scale)
                            T.tile.min(neg_sm[cur, :, :], neg_sm[cur, :, :], neg_sm[prv, :, :])
                            T.tile.broadcast(buf_2d, neg_sm[cur, :, :])
                            T.tile.axpy(buf_2d, work_ub, sm_scale)
                            T.tile.exp(work_ub, buf_2d)

                            T.wait_flag("MTE3", "V", SIG_S_HALF)
                            T.copy(work_ub, acc_s_half)
                            T.set_flag("V", "MTE3", SIG_S_HALF)

                            T.wait_flag("V", "MTE3", SIG_S_HALF)
                            T.copy(acc_s_half, workspace_2[cid, i, vid * half_M : vid * half_M + half_M, :])
                            T.set_flag("MTE3", "V", SIG_S_HALF)
                            if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
                                T.set_cross_flag("MTE3", SEM_WS2_V2C)

                            T.reduce_sum(work_ub, sumexp_is[i, :, :], dim=-1)

                            T.tile.sub(r_factors[i, :, :], neg_sm[cur, :, :], neg_sm[prv, :, :])

                        T.set_cross_flag("MTE2", SEM_WS1_V2C)

                        # --- O accumulation batch ---
                        for i in T.serial(batch_iters):
                            T.tile.exp(r_factors[i, :, :], r_factors[i, :, :])
                            T.tile.mul(sumexp, sumexp, r_factors[i, :, :])
                            T.tile.add(sumexp, sumexp, sumexp_is[i, :, :])
                            T.tile.broadcast(buf_2d, r_factors[i, :, :])
                            T.tile.mul(acc_o, acc_o, buf_2d)

                            T.wait_flag("V", "MTE2", SIG_IO_UB)
                            if i % cross_interval == 0:
                                T.wait_cross_flag(SEM_WS3_C2V)
                            T.copy(workspace_3[cid, i, vid * half_M : vid * half_M + half_M, :], io_buf)
                            T.set_flag("MTE2", "V", SIG_IO_UB)

                            T.wait_flag("MTE2", "V", SIG_IO_UB)
                            T.copy(io_buf, work_ub)
                            T.set_flag("V", "MTE2", SIG_IO_UB)

                            T.tile.add(acc_o, acc_o, work_ub)

                        T.set_cross_flag("MTE2", SEM_WS3_V2C)

                    T.tile.broadcast(buf_2d, sumexp)
                    T.tile.div(acc_o, acc_o, buf_2d)

                    # Reuse the softmax store buffer only after MTE3 returns ownership.
                    T.wait_flag("MTE3", "V", SIG_S_HALF)
                    T.copy(acc_o, acc_s_half)
                    T.set_flag("V", "MTE3", SIG_S_HALF)
                    T.wait_flag("V", "MTE3", SIG_S_HALF)
                    T.copy(
                        acc_s_half, Output[bz, by, bx * block_M + vid * block_M // 2 : bx * block_M + vid * block_M // 2 + block_M // 2, :]
                    )
                    T.set_flag("MTE3", "V", SIG_S_HALF)

                # destroy: consume outstanding init-direction flags
                T.wait_flag("V", "MTE2", SIG_IO_UB)
                T.wait_flag("MTE3", "V", SIG_S_HALF)

    return main




def make_args(case, canonical):
    return canonical["q"], canonical["k"], canonical["v"]
