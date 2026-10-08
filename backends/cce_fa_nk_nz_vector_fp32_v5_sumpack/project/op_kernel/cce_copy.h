#pragma once

#include "kernel_operator.h"
#include "cce_sync.h"
#include <cstdint>

// dav-c220 asynchronous movement. Ordinary GM/UB DMA uses typed byte pitches;
// matrix loads, ND2NZ and FIX below retain separate, explicitly documented
// descriptor units and INT8/INT32/FP16 layout contracts. No storage allocation.
namespace cce {
namespace c220 {

constexpr uint32_t BLOCK_BYTES = 32U;


template <class Tag> struct DmaArgument {
    uint64_t value;
    __aicore__ explicit constexpr DmaArgument(uint64_t v) : value(v) {}
};
using Rows = DmaArgument<struct RowsTag>;
using RowBytes = DmaArgument<struct RowBytesTag>;
using SrcPitchBytes = DmaArgument<struct SrcPitchBytesTag>;
using DstPitchBytes = DmaArgument<struct DstPitchBytesTag>;

// Pitch is the distance between row starts; gap is the distance after the
// transferred bytes. Validate wide values before subtraction or narrowing.
__aicore__ constexpr bool IsValidDma2D(
    Rows rows, RowBytes bytes, SrcPitchBytes src, DstPitchBytes dst) {
    return rows.value >= 1U && rows.value <= 4095U &&
           bytes.value >= 32U && bytes.value <= 65535U * 32U &&
           bytes.value % 32U == 0U &&
           src.value >= bytes.value && dst.value >= bytes.value &&
           src.value % 32U == 0U && dst.value % 32U == 0U &&
           (src.value - bytes.value) / 32U <= 65535U &&
           (dst.value - bytes.value) / 32U <= 65535U;
}

struct Dma2DDescriptor {
    uint16_t rows;
    uint16_t blocks;
    uint16_t srcGap;
    uint16_t dstGap;
};

__aicore__ __attribute__((always_inline)) inline Dma2DDescriptor EncodeDma2D(
    Rows rows, RowBytes bytes, SrcPitchBytes src, DstPitchBytes dst) {
    // Match CANN's debug checking policy: release kernels retain the direct
    // descriptor path. IsValidDma2D is also available to host/static checks.
    ASCENDC_DEBUG_ASSERT(IsValidDma2D(rows, bytes, src, dst),
        KERNEL_LOG_INTERNAL(KERNEL_ERROR, "invalid GM/UB 2D DMA geometry"));
    return {static_cast<uint16_t>(rows.value),
            static_cast<uint16_t>(bytes.value / 32U),
            static_cast<uint16_t>((src.value - bytes.value) / 32U),
            static_cast<uint16_t>((dst.value - bytes.value) / 32U)};
}

template <typename T>
__aicore__ __attribute__((always_inline)) inline void Load2D(
    __ubuf__ T *dst, __gm__ T *src, Rows rows, RowBytes bytes,
    SrcPitchBytes srcPitch, DstPitchBytes dstPitch) {
    const auto d = EncodeDma2D(rows, bytes, srcPitch, dstPitch);
    copy_gm_to_ubuf(dst, src, 0, d.rows, d.blocks, d.srcGap, d.dstGap);
}

template <typename T>
__aicore__ __attribute__((always_inline)) inline void Store2D(
    __gm__ T *dst, __ubuf__ T *src, Rows rows, RowBytes bytes,
    SrcPitchBytes srcPitch, DstPitchBytes dstPitch) {
    const auto d = EncodeDma2D(rows, bytes, srcPitch, dstPitch);
    copy_ubuf_to_gm(dst, src, 0, d.rows, d.blocks, d.srcGap, d.dstGap);
}

// One-dimensional exact-byte copies. UB is 32-byte aligned and owns the
// rounded-up 32-byte extent, including an input tail's extra UB lanes. GM and
// UB have the same dtype; there is no cast. Zero bytes performs no operation.
// This leaf supports up to 65535*32 bytes. Larger work must be tiled by caller.
template <typename T>
__aicore__ __attribute__((always_inline)) inline void Load1D(__ubuf__ T *dst, __gm__ T *src, uint32_t bytes) {
    if (bytes == 0U) return;
    ASCENDC_DEBUG_ASSERT(bytes <= 65535U * BLOCK_BYTES,
        KERNEL_LOG_INTERNAL(KERNEL_ERROR, "1D DMA must be tiled"));
    if ((bytes & (BLOCK_BYTES - 1U)) == 0U) {
        copy_gm_to_ubuf(dst, src, 0, 1, static_cast<uint16_t>(bytes / BLOCK_BYTES), 0, 0);
    } else {
        copy_gm_to_ubuf_align_b32(dst, src, 0, 1, bytes, 0, 0, 0, 0);
    }
}

template <typename T>
__aicore__ __attribute__((always_inline)) inline void Store1D(__gm__ T *dst, __ubuf__ T *src, uint32_t bytes) {
    if (bytes == 0U) return;
    ASCENDC_DEBUG_ASSERT(bytes <= 65535U * BLOCK_BYTES,
        KERNEL_LOG_INTERNAL(KERNEL_ERROR, "1D DMA must be tiled"));
    if ((bytes & (BLOCK_BYTES - 1U)) == 0U) {
        copy_ubuf_to_gm(dst, src, 0, 1, static_cast<uint16_t>(bytes / BLOCK_BYTES), 0, 0);
    } else {
        copy_ubuf_to_gm_align_b32(dst, src, 0, 1, bytes, 0, 0, 0, 0);
    }
}

// Ordinary GM/L1 byte DMA. Burst length and both gaps are in 32-byte blocks;
// bursts is the row count, not bytes. These int8 pointers carry storage bytes.
// Caller validates descriptor fields, address alignment, bounds and ownership.
__aicore__ __attribute__((always_inline)) inline void LoadL1(__cbuf__ int8_t *dst, __gm__ int8_t *src, uint16_t bursts,
                                                             uint16_t burstBlocks, uint16_t srcGapBlocks,
                                                             uint16_t dstGapBlocks = 0U) {
    copy_gm_to_cbuf(dst, src, 0, bursts, burstBlocks, srcGapBlocks, dstGapBlocks, PAD_NONE);
}

__aicore__ __attribute__((always_inline)) inline void StoreL1(__gm__ int8_t *dst, __cbuf__ int8_t *src,
                                                              uint16_t bursts, uint16_t burstBlocks) {
    copy_cbuf_to_gm(dst, src, 0, bursts, burstBlocks, 0, 0);
}

// INT8 ND -> L1 NZ. rows/cols/srcRowStride are element counts; alignedRows
// is the padded M extent (multiple of 16). The full source row pitch must fit
// uint16_t. The destination owns alignedRows*align32(cols) bytes.
__aicore__ __attribute__((always_inline)) inline void LoadNd2Nz(__cbuf__ int8_t *dst, __gm__ int8_t *src, uint16_t rows,
                                                                uint16_t cols, uint16_t srcRowStride,
                                                                uint16_t alignedRows) {
    copy_gm_to_cbuf_multi_nd2nz_b8(dst, src, 0, 1, rows, cols, 0, srcRowStride, alignedRows, 1, 0);
}

__aicore__ __attribute__((always_inline)) inline void LoadAPanel(
    __cbuf__ int8_t *dst, __gm__ int8_t *src, uint32_t row, uint32_t k,
    uint16_t rows, uint16_t cols, uint32_t fullK, uint16_t alignedRows) {
    LoadNd2Nz(dst, src + static_cast<uint64_t>(row) * fullK + k, rows, cols,
              static_cast<uint16_t>(fullK), alignedRows);
}

// INT8 L1 NZ -> L0A zZ. Dimensions and rowOffset count elements; M is padded
// to 16 and K to 32. ExtractLeftRows slices a fullM-pitch NZ panel. These
// functions set FMATRIX; caller owns its state and orders producer/consumer.
__aicore__ __attribute__((always_inline)) inline void ExtractLeft(__ca__ int8_t *dst, __cbuf__ int8_t *src,
                                                                  uint16_t alignedM, uint16_t alignedK) {
    set_fmatrix(static_cast<uint64_t>(alignedM) | (static_cast<uint64_t>(1U) << 16U));
    img2colv2_cbuf_to_ca(dst, src, alignedK, alignedM, 0, 0, 1, 1, 1, 1, 1, 1, false, false, false, false, alignedK);
}

__aicore__ __attribute__((always_inline)) inline void ExtractLeftRows(
    __ca__ int8_t *dst, __cbuf__ int8_t *src, uint16_t fullM,
    uint16_t tileM, uint16_t rowOffset, uint16_t alignedK) {
    set_fmatrix(static_cast<uint64_t>(fullM) |
                (static_cast<uint64_t>(1U) << 16U));
    img2colv2_cbuf_to_ca(dst, src, alignedK, tileM, 0, rowOffset,
                         1, 1, 1, 1, 1, 1,
                         false, false, false, false, alignedK);
}

// INT8 L1 B -> L0B nZ by the c220 transpose load. kBlocks/nBlocks count
// K32/N32 supertiles; l1KBlocks is the source K32 panel pitch, dstKStride is
// bytes. kBlocks and nBlocks must be positive and repeat/stride fields must
// fit the raw instruction. Each supertile occupies 1024 bytes.
// Ready variants PREARM immediately before the final L0B write; the caller
// waits READY before MMAD and returns FREE before overwriting that slot.
__aicore__ __attribute__((always_inline)) inline void ExtractRight(__cb__ int8_t *dst, __cbuf__ int8_t *src,
                                                                   uint32_t kBlocks, uint32_t nBlocks,
                                                                   uint16_t l1KBlocks, uint32_t dstKStride) {
    for (uint32_t kBlock = 0; kBlock < kBlocks; ++kBlock) {
        load_cbuf_to_cb_transpose(dst + kBlock * dstKStride, src + kBlock * 32U * 32U, 0, static_cast<uint8_t>(nBlocks),
                                  l1KBlocks, 1, inc, 0);
    }
}

__aicore__ __attribute__((always_inline)) inline void ExtractRightReady(
    __cb__ int8_t *dst, __cbuf__ int8_t *src, uint32_t kBlocks,
    uint32_t nBlocks, uint16_t l1KBlocks, uint32_t dstKStride,
    event_t event) {
    for (uint32_t kBlock = 0U; kBlock + 1U < kBlocks; ++kBlock) {
        load_cbuf_to_cb_transpose(
            dst + kBlock * dstKStride, src + kBlock * 32U * 32U, 0,
            static_cast<uint8_t>(nBlocks), l1KBlocks, 1, inc, 0);
    }
    PrearmBReady(event);
    const uint32_t finalBlock = kBlocks - 1U;
    load_cbuf_to_cb_transpose(
        dst + finalBlock * dstKStride,
        src + finalBlock * 32U * 32U, 0,
        static_cast<uint8_t>(nBlocks), l1KBlocks, 1, inc, 0);
}

// Specialized descriptor geometry: exactly K=256 (eight K32 supertiles).
__aicore__ __attribute__((always_inline)) inline void ExtractRightK256ColumnsReady(
    __cb__ int8_t *dst, __cbuf__ int8_t *src, uint32_t nBlocks,
    uint16_t l1KBlocks, event_t event) {
    for (uint32_t nBlock = 0U; nBlock < nBlocks; ++nBlock) {
        if (nBlock + 1U == nBlocks) PrearmBReady(event);
        load_cbuf_to_cb_transpose(
            dst + nBlock * 1024U,
            src + nBlock * static_cast<uint32_t>(l1KBlocks) * 1024U,
            0, 8, 1, static_cast<uint16_t>(2U * nBlocks - 1U), inc, 0);
    }
}

// INT32 L0C -> ND GM. validM/validN and destination pitch count elements;
// srcStride is the padded L0C M extent. MMAD direction/layout and FIX geometry
// must agree. UnitFlag form uses flag 3 and requires a matched MMAD protocol.
// All FIX leaves set ND_PARA and return asynchronously; no completion drain.
__aicore__ __attribute__((always_inline)) inline void StoreAcc(__gm__ int32_t *dst, __cc__ int32_t *src,
                                                               uint16_t validN, uint16_t validM, uint32_t dstStride,
                                                               uint16_t srcStride) {
    set_nd_para(1ULL);
    copy_matrix_cc_to_gm(dst, src, 0, validN, validM, dstStride, srcStride, 0, QuantMode_t::NoQuant, 0, false, true);
}

__aicore__ __attribute__((always_inline)) inline void StoreAccUnitFlag(
    __gm__ int32_t *dst, __cc__ int32_t *src, uint16_t validN,
    uint16_t validM, uint32_t dstStride, uint16_t srcStride) {
    set_nd_para(1ULL);
    copy_matrix_cc_to_gm(dst, src, 0, validN, validM, dstStride,
                         srcStride, 3U, QuantMode_t::NoQuant, 0,
                         false, true);
}

// INT32 -> FP16 FIX with scalar DEQF16. Caller sets QUANT_PRE and proves the
// multiplier's range/accuracy. No task-specific multiplier is selected here.
__aicore__ __attribute__((always_inline)) inline void StoreAccFp16(
    __gm__ half *dst, __cc__ int32_t *src, uint16_t validN,
    uint16_t validM, uint32_t dstStride, uint16_t srcStride) {
    set_nd_para(1ULL);
    copy_matrix_cc_to_gm(dst, src, 0, validN, validM, dstStride,
                         srcStride, 0U, QuantMode_t::DEQF16, 0,
                         false, true);
}

} // namespace c220
} // namespace cce
