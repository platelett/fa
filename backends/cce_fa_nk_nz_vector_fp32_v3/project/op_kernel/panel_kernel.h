#pragma once
#include "variant_config.h"

namespace panels {
template<pipe_t Pipe,uint32_t Flag>
__aicore__ __attribute__((always_inline)) inline void NotifyGroup() {
  static_assert(Flag<=10,"leave library cross-core flags reserved");
  constexpr uint16_t message=1U|(2U<<4)|(Flag<<8);
  ffts_cross_core_sync(Pipe,message);
}
namespace D = cce::c220;
using L1Lock = D::Lock<PIPE_MTE2,PIPE_MTE1>;
using L0Lock = D::Lock<PIPE_MTE1,PIPE_M>;
constexpr uint32_t WK=fa_config::WS_K, DIM=128, PANEL=128*128, PANEL_BYTES=2*PANEL;
constexpr bool TRANSPOSE=fa_config::TRANSPOSE, SP_NZ=fa_config::SP_NZ;
constexpr bool QK_M=fa_config::QK_M, PV_M=fa_config::PV_M;
static_assert(!TRANSPOSE || SP_NZ,"Transpose requires NZ S/P");
// Zero means WS_Q follows the resident Q_L1 template constant.
constexpr uint32_t FIXED_WQ=fa_config::WS_Q;

__aicore__ inline void LoadNZ(__cbuf__ half *dst,__gm__ half *src,uint32_t rows,uint32_t pitch) {
  copy_gm_to_cbuf_multi_nd2nz_b16(dst,src,0,1,rows,DIM,0,pitch,rows,1,0);
}
__aicore__ inline void LoadA(__ca__ half *dst,__cbuf__ half *src,uint32_t rows,uint32_t row) {
  for(uint32_t m=0;m<8;++m)
    load_cbuf_to_ca(dst+m*2048,src+(row/16+m)*256,0,8,rows/16,0,0,false,inc);
}
// Independent NZ(X128) is exactly the physical input for B=X128^T.
// Preserve its 64 fractals; never reinterpret a slice of a larger NZ panel.
__aicore__ inline void LoadBT(__cb__ half *dst,__cbuf__ half *src) {
  load_cbuf_to_cb(dst,src,0,64,1,0,0,false,inc);
}
__aicore__ inline void LoadV(__cb__ half *dst,__cbuf__ half *src) {
  for(uint32_t k=0;k<8;++k)
    load_cbuf_to_cb(dst+k*2048,src+k*256,0,8,8,0,0,true,inc);
}
__aicore__ inline void LoadP(__ca__ half *dst,__cbuf__ half *src) {
  if constexpr(TRANSPOSE) load_cbuf_to_ca(dst,src,0,64,1,0,0,true,inc);
  else LoadA(dst,src,128,0);
}
__aicore__ inline void StoreND(__gm__ half *dst,__cc__ float *src,uint32_t n,uint32_t pitch) {
  set_nd_para(1ULL);
  copy_matrix_cc_to_gm(dst,src,0,n,128,pitch,128,3,QuantMode_t::F322F16,0,false,true);
}

template<uint32_t QLEN> struct Cube {
  static_assert(QLEN==fa_config::Q_L1_LEN,"resident query capacity mismatch");
  static constexpr uint32_t WQ=FIXED_WQ?FIXED_WQ:QLEN;
  static constexpr bool STREAM_K=WQ==QLEN;
  static constexpr bool STATIONARY_Q=TRANSPOSE?WQ==256:WQ==128;
  static constexpr bool REUSE_Q=!TRANSPOSE && !STATIONARY_Q;
  static constexpr bool ROTATE_Q=!TRANSPOSE && WQ==128;
  static constexpr uint32_t KS=STREAM_K?2:WK/128;
  static constexpr uint32_t KBASE=QLEN*DIM*2, VBASE=KBASE+KS*PANEL_BYTES;
  static constexpr uint32_t PK=fa_config::P_LOAD_K, PSTEP=PK/128, PSTRIDE=128*PK;
  static_assert(PK==128 || PK==256);
  static constexpr uint32_t PBASE=VBASE+WK*DIM*2, L1_END=PBASE+2*PSTRIDE*2;
  static_assert(L1_END<=524288,"L1 capacity exceeded");
  const L1Lock pl{2,0,0}, ql{1,2,2}, kl{KS,3,3};
  const D::RelayedLock<PIPE_MTE2,PIPE_MTE3,PIPE_MTE1,WK/128> vl;
  const L0Lock al{2,0,0}, bl{2,2,2};
  __cbuf__ half *q=reinterpret_cast<__cbuf__ half *>(0);
  __cbuf__ half *k=reinterpret_cast<__cbuf__ half *>(KBASE);
  __cbuf__ half *v=reinterpret_cast<__cbuf__ half *>(VBASE);
  __cbuf__ half *p=reinterpret_cast<__cbuf__ half *>(PBASE);
  __ca__ half *a=reinterpret_cast<__ca__ half *>(0);
  __cb__ half *b=reinterpret_cast<__cb__ half *>(0);
  __cc__ float *c=reinterpret_cast<__cc__ float *>(0);
  uint32_t kid[KS], kheld[KS], vid=~0U, vloaded=0;
  bool qlive=false;
  uint32_t vheld=0, q_side=0;
  __gm__ half *vsrc=nullptr;
  __aicore__ Cube() {
    for(uint32_t i=0;i<KS;++i) { kid[i]=~0U; kheld[i]=0; }
  }

  __aicore__ void BeginQ(__gm__ half *src,uint32_t rows) {
    with_locks(PIPE_MTE2,ql) {
      if constexpr(TRANSPOSE) {
        for(uint32_t i=0;i<rows/128;++i) LoadNZ(q+i*PANEL,src+i*PANEL,128,DIM);
      } else LoadNZ(q,src,rows,DIM);
    }
    ql.Acquire<PIPE_MTE1>(); qlive=true;
  }
  __aicore__ void FinishQRead(bool last_read) {
    if(last_read) { ql.Release<PIPE_MTE1>(); qlive=false; }
  }
  __aicore__ void IssueK(uint32_t kv,uint32_t ki,__gm__ half *src) {
    uint32_t slot=STREAM_K?ki%2:ki, identity=kv*(WK/128)+ki;
    if(kid[slot]==identity) return;
    if(kid[slot]!=~0U) {
      if(!kheld[slot]) kl.Acquire<PIPE_MTE1>(slot);
      kl.Release<PIPE_MTE1>(slot);
    }
    with_locks(PIPE_MTE2,kl[slot]) LoadNZ(k+slot*PANEL,src+ki*PANEL,128,DIM);
    kid[slot]=identity; kheld[slot]=0;
  }
  __aicore__ __cbuf__ half *UseK(uint32_t ki) {
    uint32_t slot=STREAM_K?ki%2:ki;
    if(!kheld[slot]) { kl.Acquire<PIPE_MTE1>(slot); kheld[slot]=1; }
    return k+slot*PANEL;
  }
  // K has reached L0; its remaining MAD consumers no longer read L1.
  // Non-streamed panels must survive until the final resident-Q packet.
  __aicore__ void FinishKRead(uint32_t ki,bool last_q) {
    if(STREAM_K || last_q) {
      uint32_t slot=STREAM_K?ki%2:ki;
      kl.Release<PIPE_MTE1>(slot);
      kid[slot]=~0U; kheld[slot]=0;
    }
  }
  __aicore__ void IssueV(uint32_t kv,__gm__ half *src) {
    if(vid==kv) return;
    // The previous KV's last complete Partial returned the full V region.
    vsrc=src; vid=kv; vloaded=1;
    with_locks(PIPE_MTE2,vl[0]) LoadNZ(v,src,128,DIM);
  }
  __aicore__ void TakeV(uint32_t ki,uint32_t side,bool last_read) {
    uint32_t bit=1U<<ki;
    if(!(vheld&bit)) { vl.Acquire<PIPE_MTE1>(ki); vheld|=bit; }
    with_locks(PIPE_MTE1,bl[side]) LoadV(b+side*PANEL,v+ki*PANEL);
    if(last_read) { vl.Release<PIPE_MTE1>(ki); vheld&=~bit; }
    if(ki+1==vloaded && vloaded<WK/128) {
      with_locks(PIPE_MTE2,vl[vloaded]) LoadNZ(v+vloaded*PANEL,vsrc+vloaded*PANEL,128,DIM);
      ++vloaded;
    }
  }
  __aicore__ void StoreScore(__gm__ half *sp,uint32_t qr,uint32_t kr) {
    if constexpr(TRANSPOSE)
      copy_matrix_cc_to_gm(sp+qr*WK+kr*16,c,0,256,128,WK,128,3,
                          QuantMode_t::F322F16,0,false,false);
    else if constexpr(SP_NZ)
      copy_matrix_cc_to_gm(sp+qr*WK+(kr/128)*PANEL,c,0,256,128,128,128,3,
                          QuantMode_t::F322F16,0,false,false);
    else StoreND(sp+qr*WK+kr,c,256,WK);
  }
  #include "qk_schedule.h"
  __aicore__ void Partial(__gm__ half *sp,__gm__ half *op,
                         uint32_t count,bool last_q) {
    set_ctrl(PV_M?sbitset1(get_ctrl(),51):sbitset0(get_ctrl(),51));
    // P_READY also returns this packet's old O slot: Vector merges i-3 before
    // producing P_i. No independently recycled half-tile ring is needed.
    wait_flag_dev(0);
    for(uint32_t qr=0;qr<count;qr+=256) {
      uint32_t halves=count-qr>=256?2:1;
      for(uint32_t ki=0;ki<WK/128;++ki) {
        uint32_t side=ki&1;
        if(ki%PSTEP==0) for(uint32_t h=0;h<halves;++h) {
          uint32_t ps=halves==1?(ki/PSTEP)%2:h;
          with_locks(PIPE_MTE2,pl[ps]) {
            if constexpr(TRANSPOSE)
              copy_gm_to_cbuf(p+ps*PSTRIDE,sp+(qr+h*128)*WK+ki*128*16,
                             0,8,PK,WK-PK,0,PAD_NONE);
            else if constexpr(SP_NZ)
              copy_gm_to_cbuf(p+ps*PSTRIDE,sp+(qr+h*128)*WK+ki*PANEL,
                             0,1,PSTRIDE*2/32,0,0,PAD_NONE);
            else copy_gm_to_cbuf_multi_nd2nz_b16(p+ps*PSTRIDE,sp+(qr+h*128)*WK+ki*128,
                                                0,1,128,PK,0,WK,128,1,0);
          }
        }
        TakeV(ki,side,last_q && qr+256>=count);
        with_locks(PIPE_M,bl[side]) {
          for(uint32_t h=0;h<halves;++h) {
            uint32_t ps=halves==1?(ki/PSTEP)%2:h;
            uint32_t as=halves==1?(ROTATE_Q?(side^q_side^1):side):
              (REUSE_Q && !TRANSPOSE ? h^q_side^1 : h);
            with_locks(PIPE_MTE1,al[as]) {
              if(ki%PSTEP==0) pl.Acquire<PIPE_MTE1>(ps);
              if constexpr(TRANSPOSE && PK==256) {
                for(uint32_t m=0;m<8;++m)
                  load_cbuf_to_ca(a+as*PANEL+m*2048,p+ps*PSTRIDE+m*4096+(ki%PSTEP)*2048,
                                 0,8,1,0,0,true,inc);
              } else LoadP(a+as*PANEL,p+ps*PSTRIDE+(ki%PSTEP)*PANEL);
              if(ki%PSTEP+1==PSTEP) pl.Release<PIPE_MTE1>(ps);
            }
            with_locks(PIPE_M,al[as])
              mad(c+h*PANEL,a+as*PANEL,b+side*PANEL,128,128,128,
                  ki+1==WK/128?3:2,false,false,ki==0);
          }
        }
      }
      for(uint32_t h=0;h<halves;++h)
        StoreND(op+(qr+h*128)*DIM,c+h*PANEL,128,128);
    }
    NotifyGroup<PIPE_FIX,4>(); // One complete workspace, at most three pending.
    if(last_q) vid=~0U;
    D::Handoff<PIPE_MTE2,PIPE_FIX>(EVENT_ID0);
  }
  __aicore__ void EndTask() {
    if(qlive) { ql.Release<PIPE_MTE1>(); qlive=false; }
    for(uint32_t i=0;i<KS;++i) {
      if(kid[i]!=~0U) {
        if(!kheld[i]) kl.Acquire<PIPE_MTE1>(i);
        kl.Release<PIPE_MTE1>(i); kid[i]=~0U; kheld[i]=0;
      }
    }
  }
  __aicore__ void Drain() {
    EndTask(); ql.Acquire<PIPE_MTE2>();
    for(uint32_t i=0;i<WK/128;++i) vl.Acquire<PIPE_MTE2>(i);
    for(uint32_t i=0;i<KS;++i) kl.Acquire<PIPE_MTE2>(i);
    for(uint32_t i=0;i<2;++i) {
      pl.Acquire<PIPE_MTE2>(i); al.Acquire<PIPE_MTE1>(i); bl.Acquire<PIPE_MTE1>(i);
    }
    D::Handoff<PIPE_FIX,PIPE_M>(EVENT_ID0);
    D::Handoff<PIPE_FIX,PIPE_S>(EVENT_ID0);
  }
};

} // namespace panels
#include "vector_output.h"
namespace panels {
// This helper contains ASCEND_IS_AIC/AIV target-specific bodies. It must not
// escape as one same-named COMDAT function in both MIX device objects.
template<uint32_t QLEN,typename Tiling>
__aicore__ __attribute__((always_inline)) inline void Run(GM_ADDR qgm,GM_ADDR kgm,GM_ADDR vgm,GM_ADDR spgm,
                    GM_ADDR opgm,GM_ADDR tracegm,GM_ADDR statsgm,GM_ADDR rungm,GM_ADDR ogm,const Tiling &d) {
  constexpr uint32_t WQ=Cube<QLEN>::WQ, SE=WQ*WK;
  auto *q=reinterpret_cast<__gm__ half *>(qgm), *k=reinterpret_cast<__gm__ half *>(kgm);
  auto *v=reinterpret_cast<__gm__ half *>(vgm);

  uint64_t start=get_sys_cnt();
  uint32_t role=0; if ASCEND_IS_AIV { role=1+get_subblockid(); }
  auto *trace=reinterpret_cast<__gm__ uint64_t *>(tracegm);
  auto *rec=trace+(get_block_idx()*3+role)*128;
  uint32_t workspace_id=get_block_idx();
  if(d.groups>25) {
    // This local candidate requires a host-qualified raw Cube ID set [0,25).
    // No high bits are masked. The sentinel is propagated to both consumers.
    auto *map=trace+d.groups*3*128;
    if ASCEND_IS_AIC {
      workspace_id=get_coreid();
      if(workspace_id>=25) workspace_id=25;
      auto *header=reinterpret_cast<__cbuf__ uint8_t *>(0);
      copy_gm_to_cbuf(header,reinterpret_cast<__gm__ uint8_t *>(map+workspace_id*4),
                     0,1,1,0,0,PAD_NONE);
      D::Handoff<PIPE_MTE2,PIPE_MTE3>(EVENT_ID7);
      // Keep this DMA header outside the Scalar trace's 512-byte domain.
      copy_cbuf_to_gm(reinterpret_cast<__gm__ uint8_t *>(trace+get_block_idx()*384+64),
                     header,0,1,1,0,0);
      NotifyGroup<PIPE_MTE3,2>();
      D::Handoff<PIPE_MTE3,PIPE_MTE2>(EVENT_ID7);
      // Finish the bootstrap lifecycle before seeding the main L1 lock set.
      D::Handoff<PIPE_MTE2,PIPE_S>(EVENT_ID7);
    }
    if ASCEND_IS_AIV {
      wait_flag_dev(2);
      auto *header=reinterpret_cast<__ubuf__ uint64_t *>(0);
      copy_gm_to_ubuf(header,trace+get_block_idx()*384+64,0,1,1,0,0);
      D::Handoff<PIPE_MTE2,PIPE_S>(EVENT_ID7);
      workspace_id=static_cast<uint32_t>(*header);
    }
    if(workspace_id>=25) {
      rec[0]=get_coreid(); rec[1]=start; rec[2]=get_sys_cnt(); rec[8]=25;
      return;
    }
  }

  auto *sp=reinterpret_cast<__gm__ half *>(spgm)+uint64_t(workspace_id)*3*SE;
  auto *op=reinterpret_cast<__gm__ half *>(opgm)+uint64_t(workspace_id)*3*WQ*DIM;
  auto *running=reinterpret_cast<__gm__ half *>(rungm)+uint64_t(workspace_id)*QLEN*DIM;
  auto *out=reinterpret_cast<__gm__ half *>(ogm);
  auto *debug=reinterpret_cast<__gm__ float *>(statsgm);
  uint32_t perhead=(d.nq+QLEN-1)/QLEN, nk=d.nk/WK;
  uint64_t all_rows=uint64_t(d.batch)*d.heads_q*nk*d.nq;
  auto tasks=cce::PartitionContiguous(d.batch*d.heads_q*perhead,d.groups,get_block_idx());
  if ASCEND_IS_AIC {
    Cube<QLEN> cube;
    set_ctrl(sbitset1(get_ctrl(),51)); set_fpc(1ULL<<63);
    for(uint32_t ti=0;ti<tasks.count;++ti) {
      uint32_t id=tasks.At(ti), bh=id/perhead, qb=(id%perhead)*QLEN;
      uint32_t nr=d.nq-qb; if(nr>QLEN) nr=QLEN;
      uint32_t ng=(nr+WQ-1)/WQ, total=ng*nk, generated=0;
      uint32_t bkv=(bh/d.heads_q)*d.heads_kv+(bh%d.heads_q)/(d.heads_q/d.heads_kv);
      cube.BeginQ(q+(uint64_t(bh)*d.nq+qb)*DIM,nr);
      for(;generated<3 && generated<total;++generated) {
        uint32_t g=generated%ng, kv=generated/ng, count=nr-g*WQ;
        if(count>WQ) count=WQ;
        cube.Score(sp+(generated%3)*SE,k+(uint64_t(bkv)*d.nk+kv*WK)*DIM,
                   kv,g*WQ,count,nr,generated+1==total);
      }
      for(uint32_t consumed=0;consumed<total;++consumed) {
        uint32_t g=consumed%ng,kv=consumed/ng,count=nr-g*WQ;
        if(count>WQ) count=WQ;
        cube.IssueV(kv,v+(uint64_t(bkv)*d.nk+kv*WK)*DIM);
        if(generated<total) cube.IssueK(generated/ng,0,k+(uint64_t(bkv)*d.nk+(generated/ng)*WK)*DIM);
        cube.Partial(sp+(consumed%3)*SE,op+(consumed%3)*WQ*DIM,count,g+1==ng);
        if(generated<total) {
          uint32_t nextg=generated%ng,nextkv=generated/ng,nc=nr-nextg*WQ;
          if(nc>WQ) nc=WQ;
          cube.Score(sp+(generated%3)*SE,k+(uint64_t(bkv)*d.nk+nextkv*WK)*DIM,
                     nextkv,nextg*WQ,nc,nr,generated+1==total);
          ++generated;
        }
      }
      cube.EndTask();
    }
    cube.Drain();
    wait_flag_dev(3);
  }
  if ASCEND_IS_AIV {
    uint32_t sub=get_subblockid();
    attention::Vector<WQ> vec;
    for(uint32_t ti=0;ti<tasks.count;++ti) {
      uint32_t id=tasks.At(ti),bh=id/perhead,qb=(id%perhead)*QLEN;
      uint32_t nr=d.nq-qb; if(nr>QLEN) nr=QLEN;
      uint32_t ng=(nr+WQ-1)/WQ,total=ng*nk;
      vec.BeginTask(nr);
      online::Init();
      // First generate up to three P packets; only then can O be returned.
      for(uint32_t step=0;step<total+3;++step) {
        if(step>=3) {
          uint32_t old=step-3,g=old%ng,kv=old/ng,count=nr-g*WQ;
          if(count>WQ) count=WQ;
          vec.Merge(old,g,count,kv,nk,sub,op+(old%3)*WQ*DIM,running,out+(uint64_t(bh)*d.nq+qb)*DIM,
                    (uint64_t(bh)*nk+kv)*d.nq+qb,all_rows,debug,d.capture);
        }
        if(step<total) {
          uint32_t g=step%ng,kv=step/ng,count=nr-g*WQ;
          if(count>WQ) count=WQ;
          vec.Snapshot(step,g,count);
          wait_flag_dev(1);
          for(uint32_t tile=0;tile<count/128;++tile) {
            for(uint32_t r=0;r<64;r+=16) {
              uint32_t query=tile*128+sub*64+r,owned=g*WQ/2+tile*64+r;
              online::Strip<TRANSPOSE,SP_NZ> strip(sp+(step%3)*SE,query,owned,kv,
                  vec.Alpha(step)+tile*64+r,vec.inputs.Next(),vec.outputs);
              strip.Load(); strip.Compute();
            }
          }
          NotifyGroup<PIPE_MTE3,0>();
          vec.Stats(step,g,count,kv,nk,sub,(uint64_t(bh)*nk+kv)*d.nq+qb,all_rows,debug,d.capture);
        }
      }
    }
    vec.Drain();
    NotifyGroup<PIPE_MTE3,3>();
    D::Handoff<PIPE_MTE3,PIPE_S>(EVENT_ID0);
  }
  rec[0]=get_coreid();rec[1]=start;rec[2]=get_sys_cnt();rec[3]=tasks.begin;
  rec[4]=tasks.count;rec[5]=WQ;rec[6]=WK;rec[7]=QLEN;rec[8]=workspace_id;
  rec[9]=Cube<QLEN>::REUSE_Q;
}
} // namespace panels
