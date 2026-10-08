#pragma once
#include <cstdint>
#include "variant_config.h"
#include "cce_sync.h"

namespace vector_layout {
namespace D=cce::c220;
constexpr uint32_t QLEN=256, PACKET_Q=fa_config::WS_Q?fa_config::WS_Q:QLEN;
constexpr uint32_t WORK=0, WORK_BYTES=33024, REDUCE_B=33024, REDUCE_B_BYTES=8448;
constexpr uint32_t O_BASE=41472, O_BYTES=65536, COEF_BASE=107008;
constexpr uint32_t OUTPUT_BASE=107520, OUTPUT_BYTES=16384;
constexpr uint32_t REDUCE_A=140288, REDUCE_A_BYTES=16640;
constexpr uint32_t S_BASE=156928, SLOT_BYTES=16640;
constexpr uint32_t M_BASE=190208, L_BASE=190720, SUM_BASE=191232;
constexpr uint32_t ALPHA_BASE=191744, ALPHA_STRIDE=512, TEMP=193280, END=193536;
constexpr float SCALE=0.08838834764831845f;
static_assert(fa_config::Q_L1_LEN==QLEN && QLEN%PACKET_Q==0);
static_assert(WORK+WORK_BYTES<=REDUCE_B && REDUCE_B+REDUCE_B_BYTES<=O_BASE);
static_assert(REDUCE_A+REDUCE_A_BYTES<=S_BASE && END<=196352);
static_assert(REDUCE_B/65536==0 && (REDUCE_B+REDUCE_B_BYTES-1)/65536==0);
static_assert(REDUCE_A/65536==2 && (REDUCE_A+REDUCE_A_BYTES-1)/65536==2);
template<typename T> __aicore__ __attribute__((always_inline)) inline __ubuf__ T *At(uint32_t b) {
  return reinterpret_cast<__ubuf__ T *>(b);
}
__aicore__ inline void HalfMask() { set_vector_mask(~0ULL,~0ULL); }
__aicore__ inline void FloatMask() { set_vector_mask(0,~0ULL); }
__aicore__ inline void Rows() { set_vector_mask(0,0xffffULL); }
// Empty descriptors supply measured schedule spacing, not a completion fence.
// Keep the caller's mask; all emitted mask contexts are included in validation.
template<uint32_t Count=16> __aicore__ inline void RawGap() {
  _Pragma("unroll") for (uint32_t i=0;i<Count;++i)
    vadd(At<float>(TEMP),At<float>(TEMP),At<float>(TEMP),0,1,1,1,8,8,8);
}
// Exact descriptor counts validated with the 8192-element fixed traversal.
// They are NOT cycles or a general opcode RAW-latency table.
constexpr uint32_t MAX_SPACING[2][9]={
  {0,8,8,8,8,8,8,8,8},
  {0,8,12,14,15,16,16,16,16}
};
constexpr uint32_t SUM_SPACING[2][9]={
  {0,0,8,8,8,8,8,8,8},
  {0,0,8,12,14,15,16,16,16}
};
// Addresses and locks travel together; only the two physical slots rotate.
template<pipe_t Producer,pipe_t Consumer,uint32_t Base,uint32_t Stride>
struct Queue {
  using Ownership=D::Lock<Producer,Consumer>;
  const Ownership lock{2,0,0};
  uint32_t serial=0;
  struct Handle {
    __ubuf__ half *addr;
    D::LockSlot<Ownership> token;
    __aicore__ void AcquireProducer() const { token.lock.template Acquire<Producer>(token.side); }
    __aicore__ void ReleaseProducer() const { token.lock.template Release<Producer>(token.side); }
    __aicore__ void AcquireConsumer() const { token.lock.template Acquire<Consumer>(token.side); }
    __aicore__ void ReleaseConsumer() const { token.lock.template Release<Consumer>(token.side); }
  };
  __aicore__ Handle Next() {
    uint32_t slot=serial; serial^=1;
    return {At<half>(Base+slot*Stride),lock[slot]};
  }
  __aicore__ void Drain() const {
    lock.template Acquire<Producer>(0); lock.template Acquire<Producer>(1);
  }
};
using InputQueue=Queue<PIPE_MTE2,PIPE_V,S_BASE,SLOT_BYTES>;
using OutputQueue=Queue<PIPE_V,PIPE_MTE3,OUTPUT_BASE,OUTPUT_BYTES>;
}
