#pragma once
#include "vector_layout.h"

namespace attention {
namespace D=cce::c220;
using namespace vector_layout;


template<typename T> __aicore__ __attribute__((always_inline)) inline __ubuf__ T *At(uint32_t bytes) {
  return reinterpret_cast<__ubuf__ T *>(bytes);
}
__aicore__ __attribute__((always_inline)) inline void Full() { set_vector_mask(~0ULL,~0ULL); }
__aicore__ __attribute__((always_inline)) inline void Rows() { set_vector_mask(0,0xffffULL); }
__aicore__ __attribute__((always_inline)) inline void CopyFloat(__ubuf__ float *d,__ubuf__ float *s,uint8_t repeats=1) {
  vcopy(reinterpret_cast<__ubuf__ uint32_t *>(d),reinterpret_cast<__ubuf__ uint32_t *>(s),repeats,1,1,8,8);
}

template<uint32_t WQ,bool InPlace=INPLACE> struct Vector {
  static_assert(WQ==128 || WQ==256 || WQ==512 || WQ==768 || WQ==1024,"unsupported resident packet");
  // Each final-store region stays V-owned throughout its KV recurrence.
  const D::Lock<PIPE_V,PIPE_MTE3> resident{O_GROUPS,RESIDENT_EVENT_BASE,RESIDENT_EVENT_BASE};
  const InputOwnership<InPlace> input;
  const ProbabilityLock probability{INPLACE?0U:2U,0,0};
  uint32_t input_serial=0, active_panels=0;
  __aicore__ void BeginTask(uint32_t count) { active_panels=count/128; }
  __aicore__ __attribute__((always_inline)) inline uint32_t NextInput() {
    uint32_t slot=input_serial;
    input_serial=slot+1==INPUT_SLOTS?0:slot+1;
    return slot;
  }
  __aicore__ __attribute__((always_inline)) inline void ReturnInput(uint32_t slot) {
    if constexpr(InPlace) input.ReturnReadOnly(slot);
    else input.template Release<PIPE_V>(slot);
  }

  __aicore__ void BeforeDebugStore() {
    D::Handoff<PIPE_V,PIPE_MTE2>(EVENT_ID7);
    D::Handoff<PIPE_MTE2,PIPE_MTE3>(EVENT_ID7);
  }
  __aicore__ void AfterDebugStore() {
    D::Handoff<PIPE_MTE3,PIPE_MTE2>(EVENT_ID7);
    D::Handoff<PIPE_MTE2,PIPE_V>(EVENT_ID7);
  }
  __aicore__ __attribute__((always_inline)) inline __ubuf__ half *OutputGroup(uint32_t group) {
    uint32_t address;
    if constexpr(SMALL_FIRST)
      address=group==0?TREE_B+SMALL_BYTES:TREE_A+BIG_BYTES+(group-1)*O_GROUP_BYTES;
    else
      address=group<EARLY_GROUPS?INPUT_BYTES+group*O_GROUP_BYTES:
        group<EARLY_GROUPS+2?TREE_A+BIG_BYTES+(group-EARLY_GROUPS)*O_GROUP_BYTES:
        TREE_B+SMALL_BYTES+(group-EARLY_GROUPS-2)*O_GROUP_BYTES;
    return At<half>(address);
  }
  __aicore__ __attribute__((always_inline)) inline __ubuf__ half *OutputPanel(uint32_t panel) {
    if constexpr(INPLACE) return OutputGroup(panel/2)+(panel%2)*(O_BYTES/2);
    else return At<half>(TREE_A+BIG_BYTES+panel*O_BYTES);
  }

  __aicore__ void Snapshot(uint32_t packet,uint32_t g,uint32_t count) {
    auto *dst=At<uint16_t>(ALPHA_HALF_BASE)+(packet%3)*ALPHA_STRIDE;
    auto *src=At<uint16_t>(M_BASE)+g*WQ/2;
    uint32_t rows=count/2, full=rows/128;
    Full();
    if(full) vcopy(dst,src,full,1,1,8,8);
    if(rows%128) {
      set_vector_mask(0,~0ULL);
      vcopy(dst+full*128,src+full*128,1,1,1,8,8);
    }
    Full();
  }

  __aicore__ void Stats(uint32_t packet,uint32_t g,uint32_t count,uint32_t kv,
                       uint32_t nk,uint32_t sub,uint64_t row_base,uint64_t all_rows,
                       __gm__ float *debug,bool capture) {
    auto *l=At<float>(L_BASE), *z=At<float>(SUM_FLOAT_BASE);
    auto *m=At<half>(M_BASE), *old=At<half>(ALPHA_HALF_BASE)+(packet%3)*ALPHA_STRIDE;
    auto *a=At<float>(ALPHA_BASE), *tmp=At<float>(TEMP_BASE);
    uint32_t owned=g*WQ/2, repeats=count/128;
    set_vector_mask(0,~0ULL);
    vconv_f162f32(z,At<half>(SUM_BASE)+owned,repeats,1,1,8,4);
    if(kv==0) vector_dup(a,0.0f,repeats,1,1,8,0);
    else {
      vconv_f162f32(tmp,old,repeats,1,1,8,4);
      vconv_f162f32(tmp+512,m+owned,repeats,1,1,8,4);
      pipe_barrier(PIPE_V);
      vsub(tmp,tmp,tmp+512,repeats,1,1,1,8,8,8);
      pipe_barrier(PIPE_V);
      vexp(a,tmp,repeats,1,1,8,8);
    }
    pipe_barrier(PIPE_V);
    if(kv==0) CopyFloat(l+owned,z,repeats);
    else vmadd(l+owned,a,z,repeats,1,1,1,8,8,8);
    pipe_barrier(PIPE_V);
    if(capture) {
      for(uint32_t r=0;r<count/2;r+=16) {
        Rows(); vconv_f162f32(tmp,m+owned+r,1,1,1,8,4);
        uint32_t query=g*WQ+(r/64)*128+sub*64+r%64;
        uint64_t dest=row_base+query;
        BeforeDebugStore();
        D::Store1D(debug+dest,tmp,64);
        D::Store1D(debug+all_rows+dest,l+owned+r,64);
        D::Store1D(debug+2*all_rows+dest,a+r,64);
        D::Store1D(debug+3*all_rows+dest,z+r,64);
        AfterDebugStore();
      }
    }
    if(kv) {
      set_vector_mask(0,~0ULL);
      vconv_f322f16(old,a,repeats,1,1,4,8);
    }
    pipe_barrier(PIPE_V); // New alpha ready; pure V scratch is free for the next Pair.
    Full();
  }

  __aicore__ void Capture(__gm__ float *dst,__ubuf__ half *src) {
    auto *f=At<float>(FLOAT_BASE);
    pipe_barrier(PIPE_V); // Diagnostic staging shares the completed V scratch arena.
    set_vector_mask(0,~0ULL);
    vconv_f162f32(f,src,32,1,1,8,4);
    BeforeDebugStore();
    D::Store1D(dst,f,8192);
    AfterDebugStore();
    Full();
  }

  __aicore__ __attribute__((always_inline)) inline __ubuf__ half *Coefficient(uint32_t row) {
    return row<256 ? At<half>(ALPHA_BC_LO)+row*16 : At<half>(ALPHA_BC_HI)+(row-256)*16;
  }

  __aicore__ __attribute__((always_inline)) inline void BeginMerge(uint32_t packet,uint32_t count,uint32_t kv) {
    if(kv) {
      Full();
      uint32_t first=count/2<256?count/2:256;
      auto *a=At<uint16_t>(ALPHA_HALF_BASE)+(packet%3)*ALPHA_STRIDE;
      vbrcb(At<uint16_t>(ALPHA_BC_LO),a,1,8,first/8);
      if(count/2>256) vbrcb(At<uint16_t>(ALPHA_BC_HI),a+256,1,8,(count/2-256)/8);
      pipe_barrier(PIPE_V); // Old compact alpha is now free for Snapshot.
    }
    wait_flag_dev(4);
  }

  __aicore__ __attribute__((always_inline)) inline void MergeStrip(uint32_t packet,uint32_t g,uint32_t count,uint32_t kv,uint32_t nk,
                       uint32_t sub,__gm__ half *op,__gm__ half *running,__gm__ half *out,
                       uint64_t row_base,uint64_t all_rows,__gm__ float *debug,bool capture,uint32_t tile,uint32_t r) {
    uint32_t os=g*WQ/128+tile, ps=INPLACE?0:input_serial++%2;
    uint32_t is=INPLACE?NextInput():ps+2;
    uint32_t query=g*WQ+tile*128+sub*64+r, owned=tile*64+r;
    auto *o=OutputPanel(os);
    auto *p=At<half>(INPLACE?S_BASE+is*SLOT_BYTES:2*SLOT_BYTES+ps*O_BYTES);
    with_locks(PIPE_MTE2,input[is]) D::Load1D(p,op+(tile*128+sub*64+r)*128,O_BYTES);
    if(kv==0 && (!INPLACE || os%2==0)) resident.Acquire<PIPE_V>(INPLACE?os/2:os);
    input.template Acquire<PIPE_V>(is);
    if constexpr(fa_config::BULK_O) {
      Full();
      if(kv==0) vcopy(reinterpret_cast<__ubuf__ uint16_t *>(o),reinterpret_cast<__ubuf__ uint16_t *>(p),64,1,1,8,8);
      else vmadd(o,Coefficient(owned),p,64,1,0,1,8,1,8);
      if(!capture) ReturnInput(is);
    }
    if(!fa_config::BULK_O || capture || kv+1==nk) for(uint32_t h=0;h<O_ROWS;h+=16) {
      auto *oh=o+h*128, *ph=p+h*128;
      if constexpr(!fa_config::BULK_O) {
        Full();
        if(kv==0) vcopy(reinterpret_cast<__ubuf__ uint16_t *>(oh),reinterpret_cast<__ubuf__ uint16_t *>(ph),16,1,1,8,8);
        else vmadd(oh,Coefficient(owned+h),ph,16,1,0,1,8,1,8);
      }
      if(capture) Capture(debug+4*all_rows+(row_base+query+h)*128,ph);
      if(h+16==O_ROWS && (!fa_config::BULK_O || capture)) ReturnInput(is);
      if(capture) Capture(debug+132*all_rows+(row_base+query+h)*128,oh);
      if(kv+1==nk) {
        // Final KV leaves l immutable until all O returns for this query task finish.
        auto *f=At<float>(FLOAT_BASE), *bc=At<float>(BC_BASE);
        set_vector_mask(0,~0ULL);
        pipe_barrier(PIPE_V);
        vconv_f162f32(f,oh,32,1,1,8,4);
        vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(bc),
              reinterpret_cast<__ubuf__ uint32_t *>(At<float>(L_BASE)+g*WQ/2+owned+h),1,8,2);
        pipe_barrier(PIPE_V);
        for(uint32_t col=0;col<128;col+=64) vdiv(f+col,f+col,bc,16,1,1,0,16,16,1);
        pipe_barrier(PIPE_V);
        vconv_f322f16(oh,f,32,1,1,4,8);
        Full();
      }
    }
    if constexpr(!INPLACE) {
      if(kv+1==nk) {
        resident.Release<PIPE_V>(os);
        with_locks(PIPE_MTE3,resident[os]) D::Store1D(out+query*128,o,O_BYTES);
      }
    } else if(kv+1==nk && (os%2 || os+1==active_panels)) {
      uint32_t group=os/2, blocks=os%2?2:1;
      resident.Release<PIPE_V>(group);
      with_locks(PIPE_MTE3,resident[group])
        copy_ubuf_to_gm(out+(query-(os%2)*128)*128,OutputGroup(group),0,blocks,O_BYTES/32,0,O_BYTES/32);
    }
  }

  __aicore__ __attribute__((always_inline)) inline void EndMerge() {
    pipe_barrier(PIPE_V); // Last alpha/final-l reader before that metadata slot is reused.
    Full();
  }

  __aicore__ __attribute__((always_inline)) inline void Merge(uint32_t packet,uint32_t g,uint32_t count,uint32_t kv,uint32_t nk,
                       uint32_t sub,__gm__ half *op,__gm__ half *running,__gm__ half *out,
                       uint64_t row_base,uint64_t all_rows,__gm__ float *debug,bool capture) {
    BeginMerge(packet,count,kv);
    for(uint32_t tile=0;tile<count/128;++tile)
      for(uint32_t r=0;r<64;r+=O_ROWS)
        MergeStrip(packet,g,count,kv,nk,sub,op,running,out,row_base,all_rows,debug,capture,tile,r);
    EndMerge();
  }

  __aicore__ void Drain() {
    if constexpr(!INPLACE) for(uint32_t i=0;i<2;++i) probability.Acquire<PIPE_V>(i);
    for(uint32_t i=0;i<O_GROUPS;++i) resident.Acquire<PIPE_V>(i);
    for(uint32_t i=0;i<INPUT_SLOTS;++i) input.template Acquire<PIPE_MTE2>(i);
    D::Handoff<PIPE_MTE2,PIPE_MTE3>(EVENT_ID7);
  }
};
} // namespace attention
