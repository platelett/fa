#pragma once
#include "c_api/asc_simd.h"

namespace cce::c220::fp16 {

// Stage leaves, NOT a complete synchronized reduction. Caller owns disjoint
// destination storage and schedules RAW/WAR dependencies, ping-pong scratch,
// and the final consumer. These leaves change NORMAL-mode mask bits; caller
// must establish NORMAL mode. No empirical delay constants are embedded here.
template<bool Sum>
__aicore__ __attribute__((always_inline)) inline void Combine(
    __ubuf__ half *dst, __ubuf__ half *x, __ubuf__ half *y, uint8_t repeats,
    uint8_t db, uint8_t xb, uint8_t yb, uint8_t dr, uint8_t xr, uint8_t yr) {
    if constexpr(Sum) vadd(dst,x,y,repeats,db,xb,yb,dr,xr,yr);
    else vmax(dst,x,y,repeats,db,xb,yb,dr,xr,yr);
}

// Pair adjacent 16-element groups: [2*Out/16,16] -> [Out/16,16].
// Used by both contiguous row reductions and key-major 16-query reductions.
template<bool Sum, uint32_t Out>
__aicore__ __attribute__((always_inline)) inline void ReduceAdjacent16(
    __ubuf__ half *dst, __ubuf__ half *src) {
    static_assert(Out>=16 && Out<=4096 && (Out & (Out-1))==0);
    set_vector_mask(~0ULL,~0ULL);
    if constexpr(Out==64) set_vector_mask(0,~0ULL);
    else if constexpr(Out<64) set_vector_mask(0,(1ULL<<Out)-1);
    Combine<Sum>(dst,src,src+16,(Out+127)/128,1,2,2,8,16,16);
}

// Input: two Out-element halves separated by 128 half elements (256 B).
// Output keeps the same midpoint gap if Out>256; Out=256 produces a dense
// 256-element result suitable for two-repeat VCGMAX/VCGADD. Scratch requires
// Out+128 half elements above that base case. PoisonGap is a diagnostic write
// matching the original synchronous control, not a synchronization primitive.
template<bool Sum, uint32_t Out, bool PoisonGap=false>
__aicore__ __attribute__((always_inline)) inline void ReduceHalfPairGap(
    __ubuf__ half *dst, __ubuf__ half *src) {
    static_assert(Out>=256 && Out<=4096 && (Out & (Out-1))==0);
    set_vector_mask(~0ULL,~0ULL);
    auto *right=src+Out+128;
    if constexpr(Out>256) {
        if constexpr(PoisonGap) vector_dup(dst+Out/2,half(12345),1,1,1,8,0);
        Combine<Sum>(dst,src,right,Out/256,1,1,1,8,8,8);
        Combine<Sum>(dst+Out/2+128,src+Out/2,right+Out/2,Out/256,1,1,1,8,8,8);
    } else Combine<Sum>(dst,src,right,2,1,1,1,8,8,8);
}

} // namespace cce::c220::fp16
