// One QK implementation; all eight modes select compile-time geometry branches.
__aicore__ void Score(__gm__ half *sp,__gm__ half *ksrc,uint32_t kv,
                     uint32_t qoff,uint32_t count,uint32_t qrows,bool last_score) {
  if constexpr(TRANSPOSE) {
set_ctrl(QK_M?sbitset1(get_ctrl(),51):sbitset0(get_ctrl(),51));
  if constexpr(STATIONARY_Q) {
    with_locks(PIPE_MTE1,bl[0]) LoadBT(b,q+(qoff/128)*PANEL);
    for(uint32_t ki=0;ki<WK/128;++ki) {
      IssueK(kv,ki,ksrc); auto *kp=UseK(ki); uint32_t side=ki&1;
      with_locks(PIPE_MTE1,al[side]) LoadA(a+side*PANEL,kp,128,0);
      FinishKRead(ki,qoff+count==qrows);
      with_locks(PIPE_M,al[side]) {
        if(ki==0) bl.Acquire<PIPE_M>(0);
        mad(c,a+side*PANEL,b,128,128,128,3,false,false,true);
        if(ki==0) {
          with_locks(PIPE_MTE1,bl[1]) LoadBT(b+PANEL,q+(qoff/128+1)*PANEL);
          FinishQRead(last_score);
          bl.Acquire<PIPE_M>(1);
        }
        if(ki+1==WK/128) bl.Release<PIPE_M>(0);
        mad(c+PANEL,a+side*PANEL,b+PANEL,128,128,128,3,false,false,true);
        if(ki+1==WK/128) bl.Release<PIPE_M>(1);
      }
      StoreScore(sp,0,ki*128);
    }
  } else {
    for(uint32_t ki=0;ki<WK/128;++ki) {
      IssueK(kv,ki,ksrc); auto *kp=UseK(ki); uint32_t side=ki&1;
      with_locks(PIPE_MTE1,al[side]) LoadA(a+side*PANEL,kp,128,0);
      FinishKRead(ki,qoff+count==qrows);
      al.Acquire<PIPE_M>(side);
      for(uint32_t step=0;step<count;step+=256) {
        uint32_t qr=REUSE_Q && (ki&1)?count-256-step:step;
        bool reuse_q=REUSE_Q && ki!=0 && step==0;
        bool retain_q=REUSE_Q && ki+1<WK/128 && step+256==count;
        if(!reuse_q) {
          with_locks(PIPE_MTE1,bl[0]) LoadBT(b,q+((qoff+qr)/128)*PANEL);
          bl.Acquire<PIPE_M>(0);
        }
        mad(c,a+side*PANEL,b,128,128,128,3,false,false,true);
        if(!retain_q) bl.Release<PIPE_M>(0);
        if(!reuse_q) {
          with_locks(PIPE_MTE1,bl[1]) LoadBT(b+PANEL,q+((qoff+qr)/128+1)*PANEL);
          FinishQRead(last_score &&
                      ((REUSE_Q && count==256) || (ki+1==WK/128 && step+256==count)));
          bl.Acquire<PIPE_M>(1);
        }
        mad(c+PANEL,a+side*PANEL,b+PANEL,128,128,128,3,false,false,true);
        if(!retain_q) bl.Release<PIPE_M>(1);
        if(step+256==count) al.Release<PIPE_M>(side);
        StoreScore(sp,qr,ki*128);
      }
    }
  }
  ffts_cross_core_sync(PIPE_FIX,0x121);
  } else {
if constexpr(REUSE_Q) q_side^=1;
  set_ctrl(QK_M?sbitset1(get_ctrl(),51):sbitset0(get_ctrl(),51));
  if constexpr(STATIONARY_Q) {
    q_side=(qoff/128)&1;
    with_locks(PIPE_MTE1,al[q_side]) LoadA(a+q_side*PANEL,q,qrows,qoff);
    FinishQRead(last_score);
    al.Acquire<PIPE_M>(q_side);
  }
  for(uint32_t ki=0;ki<WK/128;ki+=2) {
    IssueK(kv,ki,ksrc);
    auto *k0=UseK(ki);
    with_locks(PIPE_MTE1,bl[0]) LoadBT(b,k0);
    FinishKRead(ki,qoff+count==qrows);
    for(uint32_t step=0;step<count;step+=128) {
      uint32_t qr=REUSE_Q && (ki&2)?count-128-step:step;
      uint32_t side=STATIONARY_Q?q_side:(((qr/128)&1)^(REUSE_Q?q_side:0));
      if constexpr(!STATIONARY_Q) {
        // The last two Q halves survive the turn into the next K pair.
        if(!REUSE_Q || ki==0 || step>=256) {
          with_locks(PIPE_MTE1,al[side]) LoadA(a+side*PANEL,q,qrows,qoff+qr);
          FinishQRead(last_score && step+128==count &&
                      (ki+2==WK/128 || (REUSE_Q && count==256)));
          al.Acquire<PIPE_M>(side);
        }
      }
      if(step==0) bl.Acquire<PIPE_M>(0);
      mad(c,a+side*PANEL,b,128,128,128,3,false,false,true);
      // B0 has no reader in the following MAD or B1 READY wait.
      if(step+128==count) bl.Release<PIPE_M>(0);
      if(step==0) {
        IssueK(kv,ki+1,ksrc);
        auto *k1=UseK(ki+1);
        with_locks(PIPE_MTE1,bl[1]) LoadBT(b+PANEL,k1);
        FinishKRead(ki+1,qoff+count==qrows);
        bl.Acquire<PIPE_M>(1);
      }
      mad(c+PANEL,a+side*PANEL,b+PANEL,128,128,128,3,false,false,true);
      if(step+128==count) bl.Release<PIPE_M>(1);
      if constexpr(!STATIONARY_Q) {
        if(!REUSE_Q || ki+2==WK/128 || step+256<count) al.Release<PIPE_M>(side);
      }
      else if(ki+2==WK/128) al.Release<PIPE_M>(side);
      StoreScore(sp,qr,ki*128);
    }
  }
  ffts_cross_core_sync(PIPE_FIX,0x121);
  }
}
