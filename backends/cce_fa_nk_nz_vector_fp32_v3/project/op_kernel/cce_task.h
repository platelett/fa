#pragma once

#include <cstdint>

// Pure host/device partitioning. No CANN, tiling, core query or allocation.
#if defined(__CCE__)
#define CCE_TASK_FN __aicore__ __attribute__((always_inline)) inline constexpr
#else
#define CCE_TASK_FN inline constexpr
#endif

namespace cce {

// Enumerate with local<count, then At(local). Do not form an unchecked end
// as begin+count*step: the sentinel may overflow even when every task fits.
struct TaskRange {
    uint32_t begin;
    uint32_t count;
    uint32_t step;

    // Precondition: local < count. The partition factories guarantee that
    // each valid result is below total, even when total is UINT32_MAX.
    CCE_TASK_FN uint32_t At(uint32_t local) const { return begin + local * step; }
};

// Invalid worker IDs, zero workers and zero tasks produce an empty range.
// Remainder tasks go to the first workers; counts differ by at most one.
CCE_TASK_FN TaskRange PartitionContiguous(uint32_t total, uint32_t workers,
                                         uint32_t worker) {
    if (workers == 0U || worker >= workers) return {0U, 0U, 1U};
    const uint32_t base = total / workers;
    const uint32_t extra = total - base * workers;
    return {worker * base + (worker < extra ? worker : extra),
            base + (worker < extra ? 1U : 0U), 1U};
}

CCE_TASK_FN TaskRange PartitionCyclic(uint32_t total, uint32_t workers,
                                     uint32_t worker) {
    if (workers == 0U || worker >= workers || worker >= total)
        return {0U, 0U, workers == 0U ? 1U : workers};
    return {worker, (total - 1U - worker) / workers + 1U, workers};
}

} // namespace cce

#undef CCE_TASK_FN
