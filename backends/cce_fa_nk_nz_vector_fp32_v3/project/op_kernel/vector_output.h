#pragma once
#include "vector_layout.h"

namespace attention {
using namespace vector_layout;
template<uint32_t WQ> struct Vector {
  InputQueue inputs;
  OutputQueue outputs;
  __aicore__ void BeginTask(uint32_t) {}
  __aicore__ __ubuf__ float *Alpha(uint32_t packet) {
    return At<float>(ALPHA_BASE+(packet%3)*ALPHA_STRIDE);
  }
  __aicore__ void Debug(__gm__ float *dst,__ubuf__ float *src,uint32_t bytes) {
    D::Handoff<PIPE_V,PIPE_MTE3>(EVENT_ID7);
    D::Store1D(dst,src,bytes);
    D::Handoff<PIPE_MTE3,PIPE_V>(EVENT_ID7);
  }
  __aicore__ void Snapshot(uint32_t packet,uint32_t g,uint32_t count) {
    FloatMask();
    vcopy(reinterpret_cast<__ubuf__ uint32_t *>(Alpha(packet)),At<uint32_t>(M_BASE)+g*WQ/2,count/128,1,1,8,8);
  }
  __aicore__ void Stats(uint32_t packet,uint32_t g,uint32_t count,uint32_t kv,
      uint32_t,uint32_t sub,uint64_t row_base,uint64_t all_rows,__gm__ float *debug,bool capture) {
    uint32_t owned=g*WQ/2;
    auto *a=Alpha(packet),*l=At<float>(L_BASE)+owned,*z=At<float>(SUM_BASE)+owned;
    RawGap(); FloatMask(); // Last strip sum -> packet statistics.
    if(kv) vmadd(l,a,z,count/128,1,1,1,8,8,8);
    else vcopy(reinterpret_cast<__ubuf__ uint32_t *>(l),reinterpret_cast<__ubuf__ uint32_t *>(z),count/128,1,1,8,8);
    if(capture) for(uint32_t r=0;r<count/2;r+=16) {
      uint32_t query=g*WQ+(r/64)*128+sub*64+r%64;
      uint64_t dest=row_base+query;
      Debug(debug+dest,At<float>(M_BASE)+owned+r,64);
      Debug(debug+all_rows+dest,l+r,64);
      Debug(debug+2*all_rows+dest,a+r,64);
      Debug(debug+3*all_rows+dest,z+r,64);
    }
    FloatMask();
  }
  __aicore__ void Broadcast(__ubuf__ float *src) {
    FloatMask();
    vbrcb(At<uint32_t>(REDUCE_A),reinterpret_cast<__ubuf__ uint32_t *>(src),16,128,4);
    vbrcb(At<uint32_t>(REDUCE_A+256),reinterpret_cast<__ubuf__ uint32_t *>(src),16,128,4);
    RawGap();
  }
  __aicore__ void Merge(uint32_t packet,uint32_t g,uint32_t count,uint32_t kv,uint32_t nk,
      uint32_t sub,__gm__ half *op,__gm__ half *,__gm__ half *out,
      uint64_t row_base,uint64_t all_rows,__gm__ float *debug,bool capture) {
    wait_flag_dev(4);
    for(uint32_t tile=0;tile<count/128;++tile) {
      uint32_t query=g*WQ+tile*128+sub*64,owned=g*WQ/2+tile*64;
      auto in=inputs.Next(); auto *o=At<float>(O_BASE)+owned*128;
      with_locks(PIPE_MTE2,in.token) D::Load1D(in.addr,op+(tile*128+sub*64)*128,16384);
      if(kv) {
        _Pragma("unroll") for(uint32_t r=0;r<64;r+=32) {
          Broadcast(Alpha(packet)+tile*64+r); FloatMask();
          _Pragma("unroll") for(uint32_t c=0;c<128;c+=64)
            vmul(o+r*128+c,o+r*128+c,At<float>(REDUCE_A)+(c?0:64),32,1,1,0,16,16,16);
          // Two 32-repeat consumers finish reading the old broadcast view
          // well before the next generation reaches its first overwritten block.
        }
      }
      in.AcquireConsumer(); FloatMask();
      if(kv) vaxpy(o,in.addr,half(1),128,1,1,8,4);
      else vconv_f162f32(o,in.addr,128,1,1,8,4);
      if(capture) {
        // Diagnostic-only widening; production mixed update has no FP32 partial.
        vconv_f162f32(At<float>(WORK),in.addr,128,1,1,8,4);
        Debug(debug+4*all_rows+(row_base+query)*128,At<float>(WORK),32768);
        Debug(debug+132*all_rows+(row_base+query)*128,o,32768);
      }
      in.ReleaseConsumer();
      if(kv+1==nk) {
        _Pragma("unroll") for(uint32_t r=0;r<64;r+=32) {
          Broadcast(At<float>(L_BASE)+owned+r); FloatMask();
          _Pragma("unroll") for(uint32_t c=0;c<128;c+=64)
            vdiv(o+r*128+c,o+r*128+c,At<float>(REDUCE_A)+(c?0:64),32,1,1,0,16,16,16);
          RawGap();
        }
        auto result=outputs.Next(); result.AcquireProducer(); FloatMask();
        vconv_f322f16(result.addr,o,128,1,1,4,8);
        result.ReleaseProducer();
        with_locks(PIPE_MTE3,result.token) D::Store1D(out+query*128,result.addr,16384);
      }
    }
    // Alpha was last read by VMUL, before the full 128-repeat mixed add.
  }
  __aicore__ void Drain() { inputs.Drain(); outputs.Drain(); }
};
}
