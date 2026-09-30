#pragma once
#include "c_api/asc_simd.h"
#include "cce_copy.h"
#include "cce_reduction_tree.h"
#include "vector_layout.h"

namespace online {
using namespace vector_layout;
__aicore__ inline void Init() {
  set_mask_norm(); FloatMask();
  vector_dup(At<float>(M_BASE),0.0f,2,1,1,8,0);
}

template<bool T,bool NZ> struct Strip {
  static constexpr bool G=NZ&&!T;
  using MaxTree=D::fp16::ReductionTree<8192,T?16:256,G,!T>;
  InputQueue::Handle input;
  OutputQueue &outputs;
  __gm__ half *gm;
  uint32_t owned,kv;
  __ubuf__ float *alpha;
  MaxTree max_tree;
  __aicore__ Strip(__gm__ half *base,uint32_t query,uint32_t row,uint32_t key,
      __ubuf__ float *a,InputQueue::Handle in,OutputQueue &out)
    :input(in),outputs(out),gm(base+(G?(query/128)*128*512+(query%128)*16:query*512)),
     owned(row),kv(key),alpha(a),max_tree(At<half>(REDUCE_B),At<half>(REDUCE_A),At<half>(COEF_BASE)) {}
  __aicore__ void Load() {
    with_locks(PIPE_MTE2,input.token) {
      if constexpr(G) {
        copy_gm_to_ubuf(input.addr,gm,0,16,16,112,0);
        copy_gm_to_ubuf(input.addr+4224,gm+32768,0,16,16,112,0);
      } else D::Load1D(input.addr,gm,16384);
    }
  }
  __aicore__ void Widen() {
    FloatMask();
    if constexpr(NZ) {
      vconv_f162f32(At<float>(WORK),input.addr,64,1,1,8,4);
      vconv_f162f32(At<float>(WORK)+4160,input.addr+(G?4224:4096),64,1,1,8,4);
    } else vconv_f162f32(At<float>(WORK),input.addr,128,1,1,8,4);
    // The first max layer and conversion are the only input readers.
    input.ReleaseConsumer();
  }
  template<uint32_t L> __aicore__ void MaxStages() {
    max_tree.template Stage<false,L>(input.addr);
    if constexpr(L+2==MaxTree::LEVELS) { Widen(); }
    else RawGap<MAX_SPACING[T][L]>();
    if constexpr(L+1<MaxTree::LEVELS) MaxStages<L+1>();
  }
  __aicore__ void UpdateMax() {
    Rows(); vconv_f162f32(At<float>(TEMP),max_tree.Result(),1,1,1,8,4);
    ScaleScores(); Rows();
    vmuls(At<float>(TEMP),At<float>(TEMP),float(SCALE),1,1,1,8,8); RawGap();
    auto *m=At<float>(M_BASE)+owned;
    if(kv) vmax(m,m,At<float>(TEMP),1,1,1,1,8,8,8);
    else vcopy(reinterpret_cast<__ubuf__ uint32_t *>(m),At<uint32_t>(TEMP),1,1,1,8,8);
    RawGap();
  }
  __aicore__ void Coefficients() {
    auto *m=reinterpret_cast<__ubuf__ uint32_t *>(At<float>(M_BASE)+owned);
    FloatMask();
    if constexpr(!NZ) {
      // Two sparse views keep the bias read outside the active data half-ring.
      vbrcb(At<uint32_t>(REDUCE_A),m,16,128,2);
      vbrcb(At<uint32_t>(REDUCE_A+256),m,16,128,2);
    } else if constexpr(!T) {
      // Eight-key halves use opposite-parity coefficient blocks.
      vbrcb(At<uint32_t>(REDUCE_A),m,2,16,2);
      vbrcb(At<uint32_t>(REDUCE_A+32),m,2,16,2);
    } else {
      set_vector_mask(0,0xffULL);
      vcopy(At<uint32_t>(COEF_BASE)+8,m,1,1,1,8,8);
      vcopy(At<uint32_t>(COEF_BASE),m+8,1,1,1,8,8);
    }
    RawGap();
  }
  __aicore__ void ScaleScores() {
    auto *f=At<float>(WORK);
    FloatMask();
    if constexpr(NZ) {
      vmuls(f,f,float(SCALE),64,1,1,8,8);
      vmuls(f+4160,f+4160,float(SCALE),64,1,1,8,8);
    } else vmuls(f,f,float(SCALE),128,1,1,8,8);
   
  }
  __aicore__ void AffineExp() {
    auto *f=At<float>(WORK); FloatMask();
    if constexpr(!NZ) {
      _Pragma("unroll") for(uint32_t c=0;c<512;c+=64)
        vsub(f+c,f+c,At<float>(REDUCE_A)+((c&64)?0:64),16,1,1,0,64,64,16);
    } else if constexpr(T) {
      _Pragma("unroll") for(uint32_t part=0;part<2;++part)
        _Pragma("unroll") for(uint32_t q=0;q<16;q+=8)
          vsub(f+part*4160+q,f+part*4160+q,At<float>(COEF_BASE)+(q?0:8),32,2,2,0,16,16,0);
    } else {
      _Pragma("unroll") for(uint32_t part=0;part<2;++part)
        _Pragma("unroll") for(uint32_t q=0;q<16;q+=8)
          _Pragma("unroll") for(uint32_t k=0;k<16;k+=8)
            vsub(f+part*4160+q*16+k,f+part*4160+q*16+k,
                 At<float>(REDUCE_A)+q*16+(k?0:8),16,2,2,2,32,32,0);
    }
    FloatMask();
    if constexpr(NZ) { vexp(f,f,64,1,1,8,8); vexp(f+4160,f+4160,64,1,1,8,8); }
    else vexp(f,f,128,1,1,8,8);
   
  }
  __aicore__ void Pack() {
    auto out=outputs.Next(); out.AcquireProducer(); FloatMask();
    if constexpr(NZ) {
      vconv_f322f16(out.addr,At<float>(WORK),64,1,1,4,8);
      vconv_f322f16(out.addr+4096,At<float>(WORK)+4160,64,1,1,4,8);
    } else vconv_f322f16(out.addr,At<float>(WORK),128,1,1,4,8);
    out.ReleaseProducer();
    with_locks(PIPE_MTE3,out.token) {
      if constexpr(G) {
        copy_ubuf_to_gm(gm,out.addr,0,16,16,0,112);
        copy_ubuf_to_gm(gm+32768,out.addr+4096,0,16,16,0,112);
      } else D::Store1D(gm,out.addr,16384);
    }
  }
  template<uint32_t Out,bool KeepGap>
  __aicore__ void HalfPair(__ubuf__ float *dst,__ubuf__ float *src) {
    FloatMask(); auto *right=src+Out+(Out>=128?64:0);
    if constexpr(KeepGap) {
      vadd(dst,src,right,Out/128,1,1,1,8,8,8);
      vadd(dst+Out/2+64,src+Out/2,right+Out/2,Out/128,1,1,1,8,8,8);
    } else {
      if constexpr(Out<64) set_vector_mask(0,(1ULL<<Out)-1);
      vadd(dst,src,right,(Out+63)/64,1,1,1,8,8,8);
    }
  }
  static constexpr uint32_t SUM_LEVELS=T?9:6;
  template<uint32_t L> __aicore__ void SumStages() {
    auto *src=At<float>(L==0?WORK:((L&1)?REDUCE_A:REDUCE_B));
    auto *dst=At<float>((L&1)?REDUCE_B:REDUCE_A);
    constexpr uint32_t out=4096>>L;
    if constexpr(T) HalfPair<out,(out>=256)>(dst,src);
    else if constexpr(G && L<5) HalfPair<out,(L<4)>(dst,src);
    else { FloatMask(); vadd(dst,src,src+8,out/64,1,2,2,8,16,16); }
    if constexpr(L+2==SUM_LEVELS) { Pack(); }
    else if constexpr(L==1) {
      Rows();
      if(kv) vsub(alpha,alpha,At<float>(M_BASE)+owned,1,1,1,1,8,8,8);
      else vector_dup(alpha,0.0f,1,1,1,8,0);
      RawGap<SUM_SPACING[T][L]>();
    } else if constexpr(L==2) {
      if(kv) { Rows(); vexp(alpha,alpha,1,1,1,8,8); }
      RawGap<SUM_SPACING[T][L]>();
    } else RawGap<SUM_SPACING[T][L]>();
    if constexpr(L+1<SUM_LEVELS) SumStages<L+1>();
    else {
      auto *z=At<float>(SUM_BASE)+owned;
      if constexpr(T) { Rows(); vcopy(reinterpret_cast<__ubuf__ uint32_t *>(z),reinterpret_cast<__ubuf__ uint32_t *>(dst),1,1,1,8,8); }
      else { FloatMask(); vcgadd(z,dst,2,1,1,8); }
    }
  }
  __aicore__ void Compute() {
    input.AcquireConsumer(); MaxStages<0>();
    max_tree.template Finish<false>(); RawGap();
    UpdateMax(); Coefficients(); AffineExp(); SumStages<0>();
  }
};
}
