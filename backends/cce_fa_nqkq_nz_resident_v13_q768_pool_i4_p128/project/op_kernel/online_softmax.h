#pragma once
#include "c_api/asc_simd.h"
#include "cce_copy.h"
#include "cce_sync.h"
#include "vector_layout.h"

namespace online {
using namespace vector_layout;
constexpr float SCALE=0.08837890625f;
constexpr bool SYNCHRONOUS=fa_config::SYNCHRONOUS;
// Qualified on the exact paired FP16 schedule; zero retains all stage boundaries.
constexpr uint32_t RAW_WIDE_STAGES=2;

__aicore__ inline void Full() { set_vector_mask(~0ULL,~0ULL); }
__aicore__ inline void Rows() { set_vector_mask(0,0xffffULL); }
__aicore__ inline __ubuf__ half *Half(uint32_t bytes) {
  return reinterpret_cast<__ubuf__ half *>(bytes);
}
__aicore__ inline void Init() {
  set_mask_norm(); Full();
  vector_dup(Half(M_BASE),half(-65504),fa_config::Q_L1_LEN/256,1,1,8,0);
  vector_dup(Half(SCALE_BASE),half(SCALE),1,1,1,8,0);
  pipe_barrier(PIPE_V);
}

template<bool SUM>
__aicore__ __attribute__((always_inline)) inline void Combine(__ubuf__ half *d,__ubuf__ half *x,__ubuf__ half *y,
    uint8_t n,uint8_t db,uint8_t xb,uint8_t yb,uint8_t dr,uint8_t xr,uint8_t yr) {
  if constexpr(SUM) vadd(d,x,y,n,db,xb,yb,dr,xr,yr);
  else vmax(d,x,y,n,db,xb,yb,dr,xr,yr);
}

template<bool TRANSPOSE,bool NZ,uint32_t SLOT,bool InPlace=INPLACE> struct Strip {
  static_assert(SLOT<2 && (!TRANSPOSE || NZ));
  static constexpr bool GAP=NZ && !TRANSPOSE;
  static constexpr uint32_t LEVELS=TRANSPOSE?9:5;
  const InputOwnership<InPlace> &input;
  const ProbabilityLock &probability;
  __gm__ half *gm;
  uint32_t owned,kv,id;
  __ubuf__ half *s;
  __ubuf__ half *p;
  __ubuf__ half *row=Half(ROW_BASE+SLOT*64);
  __ubuf__ half *bc=Half(BIAS_BASE+SLOT*BIAS_SLOT_BYTES);
  static constexpr event_t E=SLOT==0?EVENT_ID0:EVENT_ID1;

  __aicore__ Strip(__gm__ half *slot,uint32_t query,uint32_t owned_row,uint32_t key,const InputOwnership<InPlace> &io,const ProbabilityLock &po,uint32_t slot_id)
      :input(io),probability(po),gm(slot+(GAP?(query/128)*128*512+(query%128)*16:query*512)),owned(owned_row),kv(key),id(slot_id),s(Half(S_BASE+slot_id*SLOT_BYTES)),p(INPLACE?s:Half(P_BASE+SLOT*SLOT_BYTES)) {}

  __aicore__ void Load() {
    input.template Acquire<PIPE_MTE2>(id);
    if constexpr(GAP) {
      copy_gm_to_ubuf(s,gm,0,16,16,112,0);
      copy_gm_to_ubuf(s+4224,gm+32768,0,16,16,112,0);
    } else cce::c220::Load1D(s,gm,STRIP_BYTES);
    input.template Release<PIPE_MTE2>(id);
  }
  __aicore__ void Acquire() {
    input.template Acquire<PIPE_V>(id);
    if constexpr(SYNCHRONOUS && GAP) {
      Full(); vector_dup(s+4096,half(12345),1,1,1,8,0);
      pipe_barrier(PIPE_V);
    }
  }

  template<bool SUM,uint32_t LEVEL>
  __aicore__ __attribute__((always_inline)) __ubuf__ half *StageResult() {
    if constexpr(LEVEL&1) return Half(TREE_B+SLOT*TREE_SMALL_STRIDE);
    else return Half(TREE_A+SLOT*TREE_BIG_STRIDE);
  }
  template<bool SUM,uint32_t LEVEL>
  __aicore__ __attribute__((always_inline)) void Stage() {
    static_assert(LEVEL<LEVELS);
    constexpr uint32_t OUT=4096>>LEVEL;
    auto *src=SUM?p:s;
    if constexpr(LEVEL>0) src=StageResult<SUM,LEVEL-1>();
    auto *dst=StageResult<SUM,LEVEL>();
    Full();
    if constexpr(GAP) {
      auto *right=src+OUT+128;
      if constexpr(OUT>256) {
        if constexpr(SYNCHRONOUS) vector_dup(dst+OUT/2,half(12345),1,1,1,8,0);
        Combine<SUM>(dst,src,right,OUT/256,1,1,1,8,8,8);
        Combine<SUM>(dst+OUT/2+128,src+OUT/2,right+OUT/2,OUT/256,1,1,1,8,8,8);
      } else Combine<SUM>(dst,src,right,2,1,1,1,8,8,8);
    } else {
      if constexpr(OUT==64) set_vector_mask(0,~0ULL);
      else if constexpr(OUT<64) set_vector_mask(0,(1ULL<<OUT)-1);
      Combine<SUM>(dst,src,src+16,(OUT+127)/128,1,2,2,8,16,16);
    }
    if constexpr(INPLACE && SUM && LEVEL==0) {
      input.template Release<PIPE_V>(id);
      with_locks(PIPE_MTE3,input[id]) StoreProbability();
    }
  }
  template<bool SUM>
  __aicore__ __attribute__((always_inline)) __ubuf__ half *Result() {
    if constexpr(TRANSPOSE) return StageResult<SUM,LEVELS-1>();
    else return Half(RESULT_BASE+SLOT*64);
  }
  template<bool SUM>
  __aicore__ __attribute__((always_inline)) void Finish() {
    if constexpr(!TRANSPOSE) {
      Full();
      if constexpr(SUM) vcgadd(Result<SUM>(),StageResult<SUM,4>(),2,1,1,8);
      else vcgmax(Result<SUM>(),StageResult<SUM,4>(),2,1,1,8);
    }
  }
  __aicore__ void ScaleMax() {
    Rows(); vmuls(row,Result<false>(),half(SCALE),1,1,1,8,8);
  }
  __aicore__ void UpdateMax() {
    Rows(); vmax(Half(M_BASE)+owned,row,
                Half(M_BASE)+owned,1,1,1,1,8,8,8);
  }
  __aicore__ void NegateMax() {
    Rows(); vmuls(row,Half(M_BASE)+owned,half(-1),1,1,1,8,8);
  }
  __aicore__ void Broadcast() {
    if constexpr(!TRANSPOSE) {
      Full();
      if constexpr(NZ) vbrcb(reinterpret_cast<__ubuf__ uint16_t *>(bc),reinterpret_cast<__ubuf__ uint16_t *>(row),1,8,2);
      else {
        vbrcb(reinterpret_cast<__ubuf__ uint16_t *>(bc),reinterpret_cast<__ubuf__ uint16_t *>(row),2,16,2);
        vbrcb(reinterpret_cast<__ubuf__ uint16_t *>(bc+16),reinterpret_cast<__ubuf__ uint16_t *>(row),2,16,2);
      }
    }
  }
  __aicore__ void Affine() {
    Full(); auto *scale=Half(SCALE_BASE);
    if constexpr(TRANSPOSE) vmadd(s,scale,row,64,1,0,0,8,0,0);
    else if constexpr(NZ) {
      _Pragma("unroll") for(uint32_t h=0;h<2;++h) {
        vmadd(s+h*4224,scale,bc,16,1,0,1,16,0,0);
        vmadd(s+h*4224+128,scale,bc+128,16,1,0,1,16,0,0);
      }
    } else {
      _Pragma("unroll") for(uint32_t chunk=0;chunk<4;++chunk) {
        uint32_t col=(chunk/2)*256+(chunk%2)*16;
        vmadd(s+col,scale+((col&16)?32:48),bc+((col&16)?0:16),16,2,0,0,32,0,2);
      }
    }
  }
  __aicore__ void StoreProbability() {
    if constexpr(GAP) {
      copy_ubuf_to_gm(gm,p,0,16,16,0,112);
      copy_ubuf_to_gm(gm+32768,p+4224,0,16,16,0,112);
    } else cce::c220::Store1D(gm,p,STRIP_BYTES);
  }
  __aicore__ void ExpStore() {
    Full();
    if constexpr(!INPLACE) {
      probability.Acquire<PIPE_V>(SLOT);
      if constexpr(SYNCHRONOUS && GAP) {
        vector_dup(p+4096,half(12345),1,1,1,8,0);
        pipe_barrier(PIPE_V);
      }
    }
    if constexpr(GAP) {
      vexp(p,s,32,1,1,8,8);
      vexp(p+4224,s+4224,32,1,1,8,8);
    } else vexp(p,s,64,1,1,8,8);
    if constexpr(!INPLACE) {
      input.template Release<PIPE_V>(id);
      probability.Release<PIPE_V>(SLOT);
      with_locks(PIPE_MTE3,probability[SLOT]) StoreProbability();
    }
  }
  __aicore__ void SaveSum() {
    Rows(); vcopy(reinterpret_cast<__ubuf__ uint16_t *>(Half(SUM_BASE)+owned),
                  reinterpret_cast<__ubuf__ uint16_t *>(Result<true>()),1,1,1,8,8);
  }
};

template<bool SUM,uint32_t LEVEL,bool T,bool NZ>
__aicore__ __attribute__((always_inline)) inline void Stages(Strip<T,NZ,0> &a,Strip<T,NZ,1> &b) {
  a.template Stage<SUM,LEVEL>();
  if constexpr(SYNCHRONOUS) pipe_barrier(PIPE_V);
  if constexpr(!SUM && LEVEL==0) b.Acquire();
  b.template Stage<SUM,LEVEL>();
  if constexpr(SYNCHRONOUS || LEVEL>=RAW_WIDE_STAGES) pipe_barrier(PIPE_V);
  if constexpr(LEVEL+1<Strip<T,NZ,0>::LEVELS) Stages<SUM,LEVEL+1>(a,b);
}

// A is one sum level ahead. Each context keeps its own two-plane scratch.
template<uint32_t LEVEL,bool T,bool NZ>
__aicore__ __attribute__((always_inline)) inline void SumTail(Strip<T,NZ,0> &a,Strip<T,NZ,1> &b) {
  a.template Stage<true,LEVEL>();
  if constexpr(SYNCHRONOUS) pipe_barrier(PIPE_V);
  b.template Stage<true,LEVEL-1>();
  pipe_barrier(PIPE_V);
  if constexpr(LEVEL+1<Strip<T,NZ,0>::LEVELS) SumTail<LEVEL+1>(a,b);
  else {
    a.template Finish<true>();
    b.template Stage<true,LEVEL>();
    pipe_barrier(PIPE_V);
    a.SaveSum(); b.template Finish<true>();
    pipe_barrier(PIPE_V);
    b.SaveSum();
  }
}

template<bool T,bool NZ>
__aicore__ inline void Pair(__gm__ half *slot,uint32_t query,uint32_t owned,uint32_t kv,const InputLock &input,const ProbabilityLock &probability,uint32_t first,uint32_t second) {
  Strip<T,NZ,0> a(slot,query,owned,kv,input,probability,first);
  Strip<T,NZ,1> b(slot,query+16,owned+16,kv,input,probability,second);
  a.Load(); b.Load(); a.Acquire();
  Stages<false,0>(a,b);
  a.template Finish<false>(); b.template Finish<false>();
  if constexpr(!T) pipe_barrier(PIPE_V);
  a.ScaleMax(); b.ScaleMax(); pipe_barrier(PIPE_V);
  a.UpdateMax(); b.UpdateMax(); pipe_barrier(PIPE_V);
  a.NegateMax(); b.NegateMax(); pipe_barrier(PIPE_V);
  a.Broadcast(); b.Broadcast();
  if constexpr(!T) pipe_barrier(PIPE_V);
  a.Affine(); pipe_barrier(PIPE_V);
  // B's full-strip affine separates A's Exp from sum. A's first sum level
  // separates B's affine from Exp; B's Exp separates A's first two levels.
  a.ExpStore(); b.Affine();
  if constexpr(SYNCHRONOUS) pipe_barrier(PIPE_V);
  a.template Stage<true,0>();
  if constexpr(SYNCHRONOUS) pipe_barrier(PIPE_V);
  b.ExpStore();
  if constexpr(SYNCHRONOUS) pipe_barrier(PIPE_V);
  a.template Stage<true,1>();
  if constexpr(SYNCHRONOUS) pipe_barrier(PIPE_V);
  b.template Stage<true,0>();
  pipe_barrier(PIPE_V);
  SumTail<2>(a,b);
  pipe_barrier(PIPE_V); // Both P readers and scratch readers finish before reuse.
  Full();
}
} // namespace online
