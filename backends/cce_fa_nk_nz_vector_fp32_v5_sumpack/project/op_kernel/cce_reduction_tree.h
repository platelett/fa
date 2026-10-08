#pragma once
#include "cce_fp16_reduce.h"

namespace cce::c220::fp16 {

// Complete binary-tree geometry, independent of an operator's issue schedule.
// Elements counts logical half values, excluding a midpoint gap. Each level
// halves this count until TerminalElements; GroupFinish additionally reduces
// each group of 16 to one result with VCGMAX/VCGADD.
//
// Adjacent pairing retains the 16 lanes in each group. Gap pairing combines
// halves separated by 128 half values and preserves that gap until 256 remain.
// The caller supplies disjoint, 32-byte aligned input/scratch/result regions,
// establishes NORMAL mask mode, and schedules all RAW/WAR dependencies when
// using individual stages. Scratch bank/phase placement is caller-controlled.
template<uint32_t Elements, uint32_t TerminalElements, bool Gap, bool GroupFinish>
class ReductionTree {
    static_assert(Elements>=32 && Elements<=8192 && (Elements & (Elements-1))==0);
    static_assert(TerminalElements>=16 && TerminalElements<=Elements/2 &&
                  (TerminalElements & (TerminalElements-1))==0);
    static_assert(!Gap || TerminalElements>=256);
    static_assert(!GroupFinish || TerminalElements==256,
                  "group finish currently implements the qualified 256-element form");
    __ubuf__ half *a_, *b_, *output_;

    static constexpr uint32_t Levels() {
        uint32_t levels=1;
        for(uint32_t n=Elements/2;n>TerminalElements;n/=2) ++levels;
        return levels;
    }
    static constexpr uint32_t Footprint(uint32_t elements) {
        return 2*(elements+(Gap && elements>256?128:0));
    }
    template<bool Sum,uint32_t Level>
    __aicore__ __attribute__((always_inline)) void RunLevels(__ubuf__ half *input) const {
        Stage<Sum,Level>(input);
        pipe_barrier(PIPE_V);
        if constexpr(Level+1<LEVELS) RunLevels<Sum,Level+1>(input);
    }

  public:
    static constexpr uint32_t LEVELS=Levels();
    static constexpr uint32_t INPUT_BYTES=Footprint(Elements);
    static constexpr uint32_t SCRATCH_A_BYTES=Footprint(Elements/2);
    static constexpr uint32_t SCRATCH_B_BYTES=LEVELS>1?Footprint(Elements/4):0;
    static constexpr uint32_t RESULT_ELEMENTS=GroupFinish?TerminalElements/16:TerminalElements;

    __aicore__ __attribute__((always_inline)) ReductionTree(
        __ubuf__ half *a, __ubuf__ half *b, __ubuf__ half *output)
        : a_(a), b_(b), output_(output) {}

    template<uint32_t Level>
    __aicore__ __attribute__((always_inline)) __ubuf__ half *StageResult() const {
        static_assert(Level<LEVELS);
        if constexpr(Level&1) return b_;
        else return a_;
    }
    template<bool Sum,uint32_t Level,bool PoisonGap=false>
    __aicore__ __attribute__((always_inline)) void Stage(__ubuf__ half *input) const {
        static_assert(Level<LEVELS);
        auto *src=input;
        if constexpr(Level>0) src=StageResult<Level-1>();
        auto *dst=StageResult<Level>();
        if constexpr(Gap) ReduceHalfPairGap<Sum,((Elements/2)>>Level),PoisonGap>(dst,src);
        else ReduceAdjacent16<Sum,((Elements/2)>>Level)>(dst,src);
    }
    __aicore__ __attribute__((always_inline)) __ubuf__ half *Result() const {
        if constexpr(GroupFinish) return output_;
        else return StageResult<LEVELS-1>();
    }
    template<bool Sum>
    __aicore__ __attribute__((always_inline)) void Finish() const {
        if constexpr(GroupFinish) {
            set_vector_mask(~0ULL,~0ULL);
            if constexpr(Sum) vcgadd(Result(),StageResult<LEVELS-1>(),2,1,1,8);
            else vcgmax(Result(),StageResult<LEVELS-1>(),2,1,1,8);
        }
    }

    // Simple synchronized execution for callers without an interleaved schedule.
    // Input readiness must already be established by the caller. The final
    // barrier orders later V consumers; other pipes still need a handoff.
    template<bool Sum>
    __aicore__ __attribute__((always_inline)) void Run(__ubuf__ half *input) const {
        RunLevels<Sum,0>(input);
        Finish<Sum>();
        if constexpr(GroupFinish) pipe_barrier(PIPE_V);
    }
};

} // namespace cce::c220::fp16
