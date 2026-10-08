#pragma once
namespace fa_config {
constexpr bool TRANSPOSE=false, SP_NZ=true;
constexpr bool QK_M=false, PV_M=true;
constexpr unsigned Q_L1_LEN=256, WS_Q=0, WS_K=512;
constexpr bool SYNCHRONOUS=false;
constexpr bool BULK_O=false;
constexpr bool INPLACE_SP=true;
constexpr unsigned INPUT_SLOTS=2;
constexpr unsigned P_LOAD_K=128;
}
