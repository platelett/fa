#pragma once
#include "cce_sync.h"

namespace cce::c220 {

// Reusable AIV input slots with two possible generation lifecycles:
//   read-only: MTE2 -> V -> MTE2
//   stored:    MTE2 -> V -> MTE3 -> MTE2
// IndependentReaders=true lets V continue READING the immutable result after
// Release<V>, concurrently with MTE3. Call FinishRead after V's final read;
// the next MTE2 acquire joins both returns. Never write after Release<V>.
//
// EventBase+[0,Slots) is reserved on MTE3->MTE2, MTE2->V, V->MTE3 and
// V->MTE2. Caller owns addresses, DMA endpoints and mask/scratch lifecycles.
// Construction seeds empty slots. Drain consumes the last generation's
// returns, including unused seeds; destruction is inert. No PIPE_ALL.
template<uint32_t Slots, bool IndependentReaders = false, uint32_t EventBase = 0>
class SharedSlotLock : public Lock<PIPE_MTE2, PIPE_V, PIPE_MTE3> {
    static_assert(Slots > 0 && EventBase + Slots <= 8);
    using Base = Lock<PIPE_MTE2, PIPE_V, PIPE_MTE3>;
    mutable uint32_t read_only_ = 0;

    __aicore__ static constexpr event_t Event(uint32_t slot) {
        return static_cast<event_t>(EventBase + slot);
    }

  public:
    __aicore__ __attribute__((always_inline)) SharedSlotLock()
        : Base(Slots, EventBase, EventBase, EventBase) {
        if constexpr(IndependentReaders)
            for(uint32_t i = 0; i < Slots; ++i) Signal<PIPE_V, PIPE_MTE2>(Event(i));
    }

    __aicore__ __attribute__((always_inline)) LockSlot<SharedSlotLock> operator[](uint32_t slot) const {
        return {*this, slot};
    }

    template<pipe_t Pipe>
    __aicore__ __attribute__((always_inline)) void Acquire(uint32_t slot = 0) const {
        static_assert(Pipe == PIPE_MTE2 || Pipe == PIPE_V || Pipe == PIPE_MTE3);
        if constexpr(Pipe == PIPE_MTE2) {
            if constexpr(IndependentReaders) {
                Await<PIPE_V, PIPE_MTE2>(Event(slot));
                if(!(read_only_ & (1U << slot))) Base::Acquire<PIPE_MTE2>(slot);
            } else {
                if(read_only_ & (1U << slot)) Await<PIPE_V, PIPE_MTE2>(Event(slot));
                else Base::Acquire<PIPE_MTE2>(slot);
            }
        } else Base::Acquire<Pipe>(slot);
    }

    template<pipe_t Pipe>
    __aicore__ __attribute__((always_inline)) void Release(uint32_t slot = 0) const {
        if constexpr(Pipe == PIPE_V) read_only_ &= ~(1U << slot);
        Base::Release<Pipe>(slot);
    }

    __aicore__ __attribute__((always_inline)) void FinishRead(uint32_t slot) const {
        static_assert(IndependentReaders, "FinishRead requires independent readers");
        Signal<PIPE_V, PIPE_MTE2>(Event(slot));
    }

    // Ends a read-only generation INSTEAD OF Release<V>/FinishRead/store.
    __aicore__ __attribute__((always_inline)) void ReturnReadOnly(uint32_t slot) const {
        read_only_ |= 1U << slot;
        Signal<PIPE_V, PIPE_MTE2>(Event(slot));
    }

    // Call once after all accesses have been issued. Do not release afterwards.
    __aicore__ __attribute__((always_inline)) void Drain() const {
        for(uint32_t i = 0; i < Slots; ++i) Acquire<PIPE_MTE2>(i);
    }
};

} // namespace cce::c220
