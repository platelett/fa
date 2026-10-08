#pragma once

#include "kernel_operator.h"
#include <cstdint>

// dav-c220 ownership helpers. All operations enqueue on their named pipes.
// Reserve event IDs per directed pipe pair and protect the actual physical
// buffer region. Scope exit publishes RELEASE; it is not a Scalar completion
// fence. A Lock destructor is inert: consume ordinary terminal tokens explicitly
// before ending their lifecycle. No implicit PIPE_ALL or event-ID allocator.
namespace cce {
namespace c220 {


template <class LockType> struct LockSlot {
    const LockType &lock;
    uint32_t side;
};

template <pipe_t Src, pipe_t Dst> __aicore__ __attribute__((always_inline)) inline void Signal(event_t event) {
    set_flag(Src, Dst, event);
}

template <pipe_t Src, pipe_t Dst> __aicore__ __attribute__((always_inline)) inline void Await(event_t event) {
    wait_flag(Src, Dst, event);
}

// A Lock is a circular ownership token.  Bases are supplied in pipe order:
// bases[i] names the edge entering pipe i.  Construction seeds every replica
// on the last->first edge; destruction deliberately emits no event.
template <pipe_t... Pipes> class Lock {
    static constexpr uint32_t PIPE_COUNT = sizeof...(Pipes);
    static_assert(PIPE_COUNT >= 2U, "a lock needs at least two pipe positions");

    const event_t bases_[PIPE_COUNT];
    const uint32_t replicas_;

    static constexpr pipe_t PIPES[PIPE_COUNT] = {Pipes...};

    static constexpr bool UniquePipes() {
        for (uint32_t i = 0; i < PIPE_COUNT; ++i)
            for (uint32_t j = i + 1U; j < PIPE_COUNT; ++j)
                if (PIPES[i] == PIPES[j]) return false;
        return true;
    }
    static_assert(UniquePipes(), "each pipe must occur once in a lock");

    template <pipe_t Target> __aicore__ static constexpr uint32_t Index() {
        for (uint32_t i = 0; i < PIPE_COUNT; ++i) {
            if (PIPES[i] == Target) return i;
        }
        return PIPE_COUNT;
    }

  public:
    __aicore__ __attribute__((always_inline)) inline LockSlot<Lock> operator[](uint32_t side) const {
        return {*this, side};
    }

    template <typename... Bases>
    __aicore__ __attribute__((always_inline)) explicit Lock(uint32_t replicas, Bases... bases)
        : bases_{static_cast<event_t>(bases)...}, replicas_(replicas) {
        static_assert(sizeof...(Bases) == PIPE_COUNT, "provide one event base for each pipe position");
        ASCENDC_DEBUG_ASSERT(replicas <= 8U &&
            ((static_cast<uint64_t>(bases) <= 7U &&
              static_cast<uint64_t>(bases) + replicas <= 8U) && ...),
            KERNEL_LOG_INTERNAL(KERNEL_ERROR, "invalid local event range"));
        for (uint32_t replica = 0; replica < replicas; ++replica) {
            Signal<PIPES[PIPE_COUNT - 1U], PIPES[0U]>(static_cast<event_t>(bases_[0U] + replica));
        }
    }

    template <pipe_t Pipe_>
    __aicore__ __attribute__((always_inline)) inline void Acquire(uint32_t replica = 0U) const {
        ASCENDC_DEBUG_ASSERT(replica < replicas_,
            KERNEL_LOG_INTERNAL(KERNEL_ERROR, "invalid lock slot"));
        constexpr uint32_t index = Index<Pipe_>();
        static_assert(index < PIPE_COUNT, "acquiring pipe is not in this lock");
        constexpr uint32_t previous = (index + PIPE_COUNT - 1U) % PIPE_COUNT;
        Await<PIPES[previous], Pipe_>(static_cast<event_t>(bases_[index] + replica));
    }

    template <pipe_t Pipe_>
    __aicore__ __attribute__((always_inline)) inline void Release(uint32_t replica = 0U) const {
        ASCENDC_DEBUG_ASSERT(replica < replicas_,
            KERNEL_LOG_INTERNAL(KERNEL_ERROR, "invalid lock slot"));
        constexpr uint32_t index = Index<Pipe_>();
        static_assert(index < PIPE_COUNT, "releasing pipe is not in this lock");
        constexpr uint32_t next = (index + 1U) % PIPE_COUNT;
        Signal<Pipe_, PIPES[next]>(static_cast<event_t>(bases_[next] + replica));
    }
};



// Each physical panel has its own READY/FREE lifecycle. The relay supplies
// separate event namespaces; it does not move data or grant another panel.
template<pipe_t Producer,pipe_t Via,pipe_t Consumer,uint32_t Slots>
class RelayedLock {
    static_assert(Slots<=8 && Producer!=Via && Via!=Consumer && Producer!=Consumer);
  public:
    __aicore__ __attribute__((always_inline)) RelayedLock() {
        for(uint32_t i=0;i<Slots;++i) Signal<Via,Producer>(static_cast<event_t>(i));
    }
    __aicore__ __attribute__((always_inline)) LockSlot<RelayedLock> operator[](uint32_t slot) const {
        return {*this,slot};
    }
    template<pipe_t Pipe>
    __aicore__ __attribute__((always_inline)) void Acquire(uint32_t slot=0) const {
        static_assert(Pipe==Producer || Pipe==Consumer);
        Await<Via,Pipe>(static_cast<event_t>(slot));
    }
    template<pipe_t Pipe>
    __aicore__ __attribute__((always_inline)) void Release(uint32_t slot=0) const {
        static_assert(Pipe==Producer || Pipe==Consumer);
        constexpr pipe_t next=Pipe==Producer?Consumer:Producer;
        auto event=static_cast<event_t>(slot);
        Signal<Pipe,Via>(event);
        Await<Pipe,Via>(event);
        Signal<Via,next>(event);
    }
};

template <class L>
__aicore__ __attribute__((always_inline)) inline LockSlot<L> AsLockSlot(const L &lock) {
    return {lock, 0U};
}

template <class L>
__aicore__ __attribute__((always_inline)) inline LockSlot<L> AsLockSlot(const LockSlot<L> &slot) {
    return slot;
}

// A slot reference has no side effects; only its scope acquires ownership.
// Owning guards cannot be copied or moved. C++17 elision constructs them in place.
template <pipe_t Pipe, class Slot> class ScopedLock {
    Slot slot_;
  public:
    __aicore__ __attribute__((always_inline)) explicit ScopedLock(Slot slot) : slot_(slot) {
        slot_.lock.template Acquire<Pipe>(slot_.side);
    }
    __aicore__ __attribute__((always_inline)) ~ScopedLock() {
        slot_.lock.template Release<Pipe>(slot_.side);
    }
    ScopedLock(const ScopedLock &) = delete;
    ScopedLock &operator=(const ScopedLock &) = delete;
};

template <pipe_t Pipe, class... Slots> class LockScopeObj;

template <pipe_t Pipe> class LockScopeObj<Pipe> {
  public:
    __aicore__ __attribute__((always_inline)) LockScopeObj() {}
    __aicore__ __attribute__((always_inline)) ~LockScopeObj() {}
};

template <pipe_t Pipe, class Head, class... Tail>
class LockScopeObj<Pipe, Head, Tail...> {
    ScopedLock<Pipe, Head> head_;
    LockScopeObj<Pipe, Tail...> tail_;
  public:
    __aicore__ __attribute__((always_inline)) explicit LockScopeObj(Head head, Tail... tail)
        : head_(head), tail_(tail...) {}
    __aicore__ __attribute__((always_inline)) ~LockScopeObj() {}
    __aicore__ __attribute__((always_inline)) explicit operator bool() const { return false; }
    LockScopeObj(const LockScopeObj &) = delete;
    LockScopeObj &operator=(const LockScopeObj &) = delete;
};

template <pipe_t Pipe, class... Locks>
__aicore__ __attribute__((always_inline)) inline auto MakeLockScope(const Locks &...locks) {
    static_assert(sizeof...(Locks) > 0U, "with_locks requires at least one lock");
    return LockScopeObj<Pipe, decltype(AsLockSlot(locks))...>{AsLockSlot(locks)...};
}

template <pipe_t Src, pipe_t Dst> __aicore__ __attribute__((always_inline)) inline void Handoff(event_t event) {
    set_flag(Src, Dst, event);
    wait_flag(Src, Dst, event);
}

template <pipe_t Pipe> __aicore__ __attribute__((always_inline)) inline void Barrier() {
    pipe_barrier(Pipe);
}

// Physical H events: AIC only, immediate IDs 0..3. Prearm immediately before
// the final access to the named resource; wait before its next consumer or
// overwrite. These markers are not free-standing local SET/WAIT credits and
// must not use the scope-exit Release rule of an ordinary Lock.
__aicore__ __attribute__((always_inline)) inline void PrearmAReady(event_t event) {
    hset_flag(PIPE_MTE1, PIPE_M, event, L0A, false);
}

__aicore__ __attribute__((always_inline)) inline void HWaitAReady(event_t event) {
    hwait_flag(PIPE_MTE1, PIPE_M, event, L0A, false);
}

__aicore__ __attribute__((always_inline)) inline void PrearmAFree(event_t event) {
    hset_flag(PIPE_M, PIPE_MTE1, event, L0A, false);
}

__aicore__ __attribute__((always_inline)) inline void HWaitAFree(event_t event) {
    hwait_flag(PIPE_M, PIPE_MTE1, event, L0A, false);
}

__aicore__ __attribute__((always_inline)) inline void PrearmBReady(event_t event) {
    hset_flag(PIPE_MTE1, PIPE_M, event, L0B, false);
}

__aicore__ __attribute__((always_inline)) inline void HWaitBReady(event_t event) {
    hwait_flag(PIPE_MTE1, PIPE_M, event, L0B, false);
}

__aicore__ __attribute__((always_inline)) inline void PrearmBFree(event_t event) {
    hset_flag(PIPE_M, PIPE_MTE1, event, L0B, false);
}

__aicore__ __attribute__((always_inline)) inline void HWaitBFree(event_t event) {
    hwait_flag(PIPE_M, PIPE_MTE1, event, L0B, false);
}

// Formal MIX mode-2 channel; requires the launcher's FFTS control region.
// Cube publishes READY after ReadyPipe; all participating paired AIVs publish
// FREE after FreePipe. The receiver flag namespace is shared with other modes;
// the caller must reserve the ID (CANN SyncAll may use IDs 11..14).
// InitialCredits are emitted by AIV construction only. With zero initial
// credits, the caller skips producer waits for initially owned empty stages.
// Keep every wait before stage reuse. No destructor consumes terminal FREEs.
// Stage count, physical buffers, topology and bounded outstanding work belong
// to the caller; this class does not allocate storage or establish a grid.
template <uint32_t InitialCredits, pipe_t ReadyPipe, pipe_t FreePipe> class MixChannel {
    uint8_t event_;

    template <pipe_t Pipe> __aicore__ __attribute__((always_inline)) inline void Set() const {
        ffts_cross_core_sync(Pipe, 1U + (2U << 4U) + (static_cast<uint16_t>(event_) << 8U));
    }

  public:
    __aicore__ __attribute__((always_inline)) explicit MixChannel(uint8_t event) : event_(event) {
        ASCENDC_DEBUG_ASSERT(event < 16U,
            KERNEL_LOG_INTERNAL(KERNEL_ERROR, "invalid cross-core flag"));
        if ASCEND_IS_AIV {
            for (uint32_t token = 0; token < InitialCredits; ++token) Set<FreePipe>();
        }
    }

    // The receiver-local wait instruction is identical on both endpoints;
    // the endpoint name records whether this occurrence consumes FREE or
    // READY from the channel protocol.
    __aicore__ __attribute__((always_inline)) inline void ProducerAcquire() const { wait_flag_dev(event_); }
    __aicore__ __attribute__((always_inline)) inline void ProducerRelease() const { Set<ReadyPipe>(); }
    __aicore__ __attribute__((always_inline)) inline void ConsumerAcquire() const { wait_flag_dev(event_); }
    __aicore__ __attribute__((always_inline)) inline void ConsumerRelease() const { Set<FreePipe>(); }
};


} // namespace c220
} // namespace cce

#define CCE_LOCK_JOIN_IMPL(a, b) a##b
#define CCE_LOCK_JOIN(a, b) CCE_LOCK_JOIN_IMPL(a, b)
// No hidden loop: break/continue target the caller's loop. The false condition
// keeps the guard alive through the else body, including early return.
#define with_locks(pipe, ...) \
    if (auto CCE_LOCK_JOIN(cce_lock_scope_, __COUNTER__) = \
            ::cce::c220::MakeLockScope<pipe>(__VA_ARGS__)) {} else
