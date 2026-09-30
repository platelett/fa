#include "cce_sync.h"
#include "cce_task.h"
#include "fa_fa_nk_nz_vector_fp32_v3_tiling.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "online_softmax.h"
#include "tiling_key_fa_fa_nk_nz_vector_fp32_v3.h"
#include "panel_kernel.h"

template <uint32_t ROW_TILES>
__global__ __aicore__ void fa_fa_nk_nz_vector_fp32_v3(GM_ADDR q_gm, GM_ADDR k_gm, GM_ADDR v_gm,
    GM_ADDR sp_gm, GM_ADDR partial_gm, GM_ADDR trace_gm, GM_ADDR stats_gm,
    GM_ADDR running_gm, GM_ADDR output_gm, GM_ADDR scratch, GM_ADDR tiling) {
  REGISTER_TILING_DEFAULT(FaFaNkNzVectorFp32V3TilingData);
  GET_TILING_DATA_WITH_STRUCT(FaFaNkNzVectorFp32V3TilingData,d,tiling);
  panels::Run<(ROW_TILES+1)*256>(q_gm,k_gm,v_gm,sp_gm,partial_gm,trace_gm,stats_gm,running_gm,output_gm,d);
}
