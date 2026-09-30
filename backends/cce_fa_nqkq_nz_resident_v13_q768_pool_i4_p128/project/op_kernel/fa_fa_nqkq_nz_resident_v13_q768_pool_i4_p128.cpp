#include "cce_sync.h"
#include "cce_task.h"
#include "fa_fa_nqkq_nz_resident_v13_q768_pool_i4_p128_tiling.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "online_softmax.h"
#include "tiling_key_fa_fa_nqkq_nz_resident_v13_q768_pool_i4_p128.h"
#include "panel_kernel.h"

template <uint32_t ROW_TILES, uint32_t GRAY_REUSE>
__global__ __aicore__ void fa_fa_nqkq_nz_resident_v13_q768_pool_i4_p128(GM_ADDR q_gm, GM_ADDR k_gm, GM_ADDR v_gm,
    GM_ADDR sp_gm, GM_ADDR partial_gm, GM_ADDR trace_gm, GM_ADDR stats_gm,
    GM_ADDR running_gm, GM_ADDR output_gm, GM_ADDR scratch, GM_ADDR tiling) {
  REGISTER_TILING_DEFAULT(FaFaNqkqNzResidentV13Q768PoolI4P128TilingData);
  GET_TILING_DATA_WITH_STRUCT(FaFaNqkqNzResidentV13Q768PoolI4P128TilingData,d,tiling);
  panels::Run<(ROW_TILES+1)*256,(GRAY_REUSE!=0)>(q_gm,k_gm,v_gm,sp_gm,partial_gm,trace_gm,stats_gm,running_gm,output_gm,d);
}
