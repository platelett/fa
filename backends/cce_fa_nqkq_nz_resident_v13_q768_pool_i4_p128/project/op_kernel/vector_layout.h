#pragma once
#include <cstdint>
#include "variant_config.h"
#include "cce_sync.h"
namespace vector_layout {
constexpr bool INPLACE=fa_config::INPLACE_SP;
constexpr uint32_t QLEN=fa_config::Q_L1_LEN, PACKET_Q=fa_config::WS_Q?fa_config::WS_Q:QLEN;
constexpr bool GAP=fa_config::SP_NZ && !fa_config::TRANSPOSE;
constexpr uint32_t STRIP_BYTES=16384, SLOT_BYTES=STRIP_BYTES+(GAP?256:0);
constexpr uint32_t TREE_BIG_STRIDE=8192+(GAP?256:0), TREE_SMALL_STRIDE=4096+(GAP?256:0);
constexpr uint32_t BIG_BYTES=2*TREE_BIG_STRIDE, SMALL_BYTES=2*TREE_SMALL_STRIDE;
constexpr uint32_t O_ROWS=64, O_BYTES=16384, O_GROUP_BYTES=INPLACE?32768:16384;
constexpr uint32_t O_GROUPS=QLEN/(INPLACE?256:128);
constexpr uint32_t META_BYTES=4*QLEN+3*PACKET_Q+512;
constexpr uint32_t MEMORY_SLOTS=(196352-O_GROUPS*O_GROUP_BYTES-BIG_BYTES-SMALL_BYTES-META_BYTES)/SLOT_BYTES;
constexpr uint32_t INPUT_CAP=INPLACE?(MEMORY_SLOTS<8-O_GROUPS?MEMORY_SLOTS:8-O_GROUPS):4;
constexpr uint32_t INPUT_SLOTS=INPLACE?fa_config::INPUT_SLOTS:4;
constexpr uint32_t S_BASE=0, INPUT_BYTES=INPLACE?INPUT_CAP*SLOT_BYTES:2*SLOT_BYTES;
constexpr bool SMALL_FIRST=INPLACE && INPUT_CAP>=5;
constexpr uint32_t EARLY_GROUPS=INPUT_CAP<=2?1:0;
constexpr uint32_t TREE_A=INPLACE?
  (SMALL_FIRST?INPUT_BYTES+SMALL_BYTES+O_GROUP_BYTES:INPUT_BYTES+EARLY_GROUPS*O_GROUP_BYTES):
  INPUT_BYTES+2*O_BYTES;
constexpr uint32_t P_BASE=INPLACE?S_BASE:TREE_A+BIG_BYTES+O_GROUPS*O_GROUP_BYTES;
constexpr uint32_t TREE_B=INPLACE?
  (SMALL_FIRST?INPUT_BYTES:TREE_A+BIG_BYTES+2*O_GROUP_BYTES):P_BASE+2*SLOT_BYTES;
constexpr uint32_t M_BASE=INPLACE?INPUT_BYTES+BIG_BYTES+SMALL_BYTES+O_GROUPS*O_GROUP_BYTES:TREE_B+SMALL_BYTES;
constexpr uint32_t L_BASE=M_BASE+QLEN, SUM_BASE=L_BASE+2*QLEN, ALPHA_HALF_BASE=SUM_BASE+QLEN;
constexpr uint32_t ALPHA_STRIDE=PACKET_Q/2, SCALE_BASE=ALPHA_HALF_BASE+3*PACKET_Q;
constexpr uint32_t ROW_BASE=SCALE_BASE+256, RESULT_BASE=ROW_BASE+128;
constexpr uint32_t BIAS_SLOT_BYTES=INPLACE?TREE_SMALL_STRIDE:(fa_config::SP_NZ?512:1024);
constexpr uint32_t BIAS_BASE=INPLACE?TREE_B:RESULT_BASE+128;
constexpr uint32_t END=RESULT_BASE+128+(INPLACE || fa_config::TRANSPOSE?0:2*BIAS_SLOT_BYTES);
constexpr uint32_t TEMP_BASE=TREE_A, ALPHA_BASE=TREE_A+4096, SUM_FLOAT_BASE=TREE_A+6144;
constexpr uint32_t FLOAT_BASE=TREE_B, BC_BASE=TREE_A;
constexpr uint32_t ALPHA_BC_LO=TREE_A+16384-PACKET_Q*16, ALPHA_BC_HI=ALPHA_BC_LO+8192;
constexpr uint32_t RESIDENT_EVENT_BASE=INPLACE?INPUT_CAP:2;
static_assert(QLEN>=512 && QLEN<=1024 && QLEN%256==0 && QLEN%PACKET_Q==0);
static_assert(INPLACE || QLEN==512,"independent input domains require Q512");
static_assert(INPUT_SLOTS>=2 && INPUT_SLOTS<=INPUT_CAP && RESIDENT_EVENT_BASE+O_GROUPS<=8);
static_assert(END<=196352 && INPUT_CAP>=2,"resident UB capacity exceeded");
static_assert(TREE_A/65536!=TREE_B/65536,"reduction scratch domains must differ");
static_assert((TREE_A+BIG_BYTES-1)/65536==TREE_A/65536);
static_assert((TREE_B+SMALL_BYTES-1)/65536==TREE_B/65536);

// The physical region lifecycle determines the lock's pipe sequence.
template<bool InPlace> struct InputOwnership;
template<> struct InputOwnership<true>:cce::c220::Lock<PIPE_MTE2,PIPE_V,PIPE_MTE3> {
  using Base=cce::c220::Lock<PIPE_MTE2,PIPE_V,PIPE_MTE3>;
  // One FREE occurrence per physical slot. This bit selects its producer pipe.
  mutable uint32_t vector_free=0;
  __aicore__ InputOwnership():Base(INPUT_SLOTS,0,0,0) {}
  __aicore__ __attribute__((always_inline)) inline cce::c220::LockSlot<InputOwnership> operator[](uint32_t slot) const {
    return {*this,slot};
  }
  template<pipe_t Pipe>
  __aicore__ __attribute__((always_inline)) inline void Acquire(uint32_t slot=0) const {
    if constexpr(Pipe==PIPE_MTE2) {
      if(vector_free&(1U<<slot)) cce::c220::Await<PIPE_V,PIPE_MTE2>(static_cast<event_t>(slot));
      else Base::Acquire<PIPE_MTE2>(slot);
    } else Base::Acquire<Pipe>(slot);
  }
  template<pipe_t Pipe>
  __aicore__ __attribute__((always_inline)) inline void Release(uint32_t slot=0) const {
    if constexpr(Pipe==PIPE_V) vector_free&=~(1U<<slot);
    Base::Release<Pipe>(slot);
  }
  __aicore__ __attribute__((always_inline)) inline void ReturnReadOnly(uint32_t slot) const {
    vector_free|=1U<<slot;
    cce::c220::Signal<PIPE_V,PIPE_MTE2>(static_cast<event_t>(slot));
  }
};
template<> struct InputOwnership<false>:cce::c220::Lock<PIPE_MTE2,PIPE_V> {
  __aicore__ InputOwnership():Lock(INPUT_SLOTS,0,0) {}
};
using InputLock=InputOwnership<INPLACE>;
using ProbabilityLock=cce::c220::Lock<PIPE_V,PIPE_MTE3>;
}
