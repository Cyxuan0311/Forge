#pragma once
// GEMM micro-kernel for quantized types.
// Phase 1: Q4_0 decode GEMM (RM=4, RN=1) using sign_epi8 + updot.
//
// Architecture: process RM=4 weight rows simultaneously, sharing
// the Q8_0_act activation loading. Uses sign_epi8 trick to avoid
// the maddubs + correction overhead (saves ~3 instructions/block).

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#ifdef USE_AVX2
#    include <immintrin.h>
#    include "../../common/quant_helpers.h"  // block_q8_K, block_q6_K, quantize_row_q8_K, get_scale_shuffle
#    include "scales.h"                       // get_scale_shuffle*, decode_q4_k_scales
#    include "thread_pool.h"
#    include "vec_dot.h"          // block_q8_0_act, quantize_row_q8_0_act, vdot::fp16_to_fp32
#endif

#ifdef _OPENMP
#    include <omp.h>
#endif

namespace forge {
namespace cpu {

#ifdef USE_AVX2

// ============================================================================
// Block Loaders: decode a quantized block into 32 signed int8 values
// ============================================================================

// Tag types for weight block layouts (stored as raw bytes, no struct)
struct block_q4_0_tag {};  // Q4_0: 18 bytes (d[2] fp16 + qs[16] nibbles)

template <typename T>
struct BlockLoader;  // primary template — must be specialized

// Q4_0 block loader: denibble 16 bytes → 32 signed int8 values [-8..7]
template <>
struct BlockLoader<block_q4_0_tag> {
    // Decode 16 bytes of 4-bit nibbles into 32 unsigned bytes [0..15]
    static inline __m256i denibble(const uint8_t* qs) {
        const __m128i x = _mm_loadu_si128((const __m128i*)qs);
        __m256i expanded = _mm256_insertf128_si256(
            _mm256_castsi128_si256(x),
            _mm_srli_epi16(x, 4),
            1);
        return _mm256_and_si256(_mm256_set1_epi8(0x0F), expanded);
    }

    // Load a Q4_0 block and return 32 signed int8 values
    static inline __m256i load(const uint8_t* block_ptr) {
        return _mm256_sub_epi8(denibble(block_ptr + 2), _mm256_set1_epi8(8));
    }

    // Read the fp16 scale from a Q4_0 block
    static inline float scale(const uint8_t* block_ptr) {
        uint16_t d;
        memcpy(&d, block_ptr, 2);
        return vdot::fp16_to_fp32(d);
    }

    static constexpr int BLOCK_BYTES = 18;
    static constexpr int BLOCK_SIZE = 32;
};

// Q8_0_act block loader: 32 signed int8 values, already in memory
template <>
struct BlockLoader<block_q8_0_act> {
    static inline __m256i load(const block_q8_0_act* b) {
        return _mm256_loadu_si256((const __m256i*)b->qs);
    }

    static inline float scale(const block_q8_0_act* b) {
        return b->d;
    }

    static constexpr int BLOCK_BYTES = sizeof(block_q8_0_act);  // 36
    static constexpr int BLOCK_SIZE = 32;
};

// ============================================================================
// Helper: updot — maddubs + madd_epi16 → 8 float32 values
// ============================================================================

// sign_epi8 + updot: convert signed×signed to unsigned×unsigned via sign trick
// _mm256_sign_epi8(a, a) = a >= 0 ? 0x01 : 0xFF  (for a != 0; 0 stays 0)
// Actually: sign_epi8(a, a) returns |a| with original sign encoding:
//   sign_epi8(a, a) = a (identity for sign extraction)
//   sign_epi8(b, a) = a >= 0 ? b : -b
// Combined: sign(a,a) gives |a|'s sign mask, sign(b,a) conditionally negates b
// This makes maddubs produce the correct signed result without correction.

static inline __m256 updot_avx2(__m256i u, __m256i s) {
    __m256i p16 = _mm256_maddubs_epi16(u, s);
    __m256i p32 = _mm256_madd_epi16(p16, _mm256_set1_epi16(1));
    return _mm256_cvtepi32_ps(p32);
}

// ============================================================================
// Q4_0 Decode GEMM Micro-Kernel (RM=4, RN=1)
// ============================================================================
// Processes 4 weight rows (output rows) simultaneously.
// For decode (M=1), there is only 1 activation row (quantized to Q8_0_act).
// The activation is shared across all 4 weight rows.
//
// Loop structure:
//   for each tile of 4 output rows (n0 .. n_end):
//     init 4 accumulators (one per weight row)
//     for each block l in [0, nb):
//       load 4 weight blocks (4 rows × block l)
//       load 1 activation block (shared)
//       for each of 4 rows: sign_epi8 + updot → accumulate
//     write 4 output values

static void gemm_q4_0_decode_avx2(const float* a_data,
                                   const uint8_t* w_data,
                                   float* o_data,
                                   int64_t K,
                                   int64_t N) {
    constexpr int RM = 4;
    constexpr int BLOCK_SIZE = 32;
    constexpr int BLOCK_BYTES = 18;
    const int64_t nb = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Quantize activation once — shared by all weight rows
    scratch_vec<block_q8_0_act> q8_act(nb);
    quantize_row_q8_0_act(a_data, q8_act.data(), (int)K);

    // Process RM=4 weight rows at a time
    #pragma omp parallel for schedule(static)
    for (int64_t n = 0; n < N; n += RM) {
        int64_t rows = (n + RM <= N) ? RM : (N - n);

        // Initialize accumulators for up to 4 output rows
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        // Pointers to weight rows
        const uint8_t* w0 = w_data + (size_t)(n + 0) * nb * BLOCK_BYTES;
        const uint8_t* w1 = w_data + (size_t)(n + 1) * nb * BLOCK_BYTES;
        const uint8_t* w2 = w_data + (size_t)(n + 2) * nb * BLOCK_BYTES;
        const uint8_t* w3 = w_data + (size_t)(n + 3) * nb * BLOCK_BYTES;

        for (int64_t l = 0; l < nb; ++l) {
            // Prefetch next blocks
            if (l + 4 < nb) {
                _mm_prefetch((const char*)(w0 + (l + 4) * BLOCK_BYTES), _MM_HINT_T0);
                _mm_prefetch((const char*)(w1 + (l + 4) * BLOCK_BYTES), _MM_HINT_T0);
                _mm_prefetch((const char*)(w2 + (l + 4) * BLOCK_BYTES), _MM_HINT_T0);
                _mm_prefetch((const char*)(w3 + (l + 4) * BLOCK_BYTES), _MM_HINT_T0);
            }

            // Load weight blocks (Q4_0 → 32 signed int8)
            __m256i wvec0 = BlockLoader<block_q4_0_tag>::load(w0 + l * BLOCK_BYTES);
            __m256i wvec1 = (rows > 1) ? BlockLoader<block_q4_0_tag>::load(w1 + l * BLOCK_BYTES) : _mm256_setzero_si256();
            __m256i wvec2 = (rows > 2) ? BlockLoader<block_q4_0_tag>::load(w2 + l * BLOCK_BYTES) : _mm256_setzero_si256();
            __m256i wvec3 = (rows > 3) ? BlockLoader<block_q4_0_tag>::load(w3 + l * BLOCK_BYTES) : _mm256_setzero_si256();

            // Load activation block (Q8_0_act → 32 signed int8)
            __m256i avec = BlockLoader<block_q8_0_act>::load(&q8_act[l]);

            // Combined scale = weight_scale * act_scale
            float sw0 = BlockLoader<block_q4_0_tag>::scale(w0 + l * BLOCK_BYTES) * q8_act[l].d;
            float sw1 = (rows > 1) ? BlockLoader<block_q4_0_tag>::scale(w1 + l * BLOCK_BYTES) * q8_act[l].d : 0.0f;
            float sw2 = (rows > 2) ? BlockLoader<block_q4_0_tag>::scale(w2 + l * BLOCK_BYTES) * q8_act[l].d : 0.0f;
            float sw3 = (rows > 3) ? BlockLoader<block_q4_0_tag>::scale(w3 + l * BLOCK_BYTES) * q8_act[l].d : 0.0f;

            // sign_epi8 + updot for each row
            // sign_epi8(a, a) extracts sign of a: positive→0x01, negative→0xFF
            // sign_epi8(b, a) conditionally negates b based on a's sign
            // maddubs then produces correct unsigned×unsigned = signed result
            __m256i sa0 = _mm256_sign_epi8(wvec0, wvec0);
            __m256i sa1 = _mm256_sign_epi8(wvec1, wvec1);
            __m256i sa2 = _mm256_sign_epi8(wvec2, wvec2);
            __m256i sa3 = _mm256_sign_epi8(wvec3, wvec3);

            __m256i sb0 = _mm256_sign_epi8(avec, wvec0);
            __m256i sb1 = (rows > 1) ? _mm256_sign_epi8(avec, wvec1) : _mm256_setzero_si256();
            __m256i sb2 = (rows > 2) ? _mm256_sign_epi8(avec, wvec2) : _mm256_setzero_si256();
            __m256i sb3 = (rows > 3) ? _mm256_sign_epi8(avec, wvec3) : _mm256_setzero_si256();

            __m256 dot0 = updot_avx2(sa0, sb0);
            __m256 dot1 = updot_avx2(sa1, sb1);
            __m256 dot2 = updot_avx2(sa2, sb2);
            __m256 dot3 = updot_avx2(sa3, sb3);

            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(sw0), dot0, acc0);
            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(sw1), dot1, acc1);
            acc2 = _mm256_fmadd_ps(_mm256_set1_ps(sw2), dot2, acc2);
            acc3 = _mm256_fmadd_ps(_mm256_set1_ps(sw3), dot3, acc3);
        }

        // Horizontal sum and write outputs
        o_data[n + 0] = vdot::hsum_ps256(acc0);
        if (rows > 1) o_data[n + 1] = vdot::hsum_ps256(acc1);
        if (rows > 2) o_data[n + 2] = vdot::hsum_ps256(acc2);
        if (rows > 3) o_data[n + 3] = vdot::hsum_ps256(acc3);
    }
}

// ============================================================================
// F16C-optimized variant: batch-convert 4 fp16 scales at once
// ============================================================================
// Uses _mm_cvtph_ps to convert 4 fp16 weight scales in one instruction.
// Packs 4 scales from consecutive weight rows into a 64-bit register,
// then multiplies with the FP32 activation scale.

// ============================================================================
// GEMM tile: RM=4 weight rows × 1 activation row
// ============================================================================
// Processes up to 4 weight rows against a single Q8_0_act activation row.
// This is the core GEMM tile that replaces the per-row GEMV inner loop.
// Uses F16C for batch scale conversion + sign_epi8 + updot.
//
// w_ptrs: array of up to 4 pointers to weight rows (each row = nb * 18 bytes)
// q8_act: pre-quantized activation (nb blocks)
// out: output array (at least RM floats)
// nb: number of blocks
// rows: actual number of valid rows (1-4)

static inline void gemm_q4_0_tile_4x1_f16c(
    const uint8_t* const* w_ptrs,    // [RM] weight row pointers
    const block_q8_0_act* q8_act,    // activation blocks
    float* out,                       // output [RM]
    int64_t nb,                       // number of blocks
    int64_t rows)                     // actual rows (1-4)
{
    constexpr int BLOCK_BYTES = 18;

    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    const uint8_t* w0 = w_ptrs[0];
    const uint8_t* w1 = (rows > 1) ? w_ptrs[1] : nullptr;
    const uint8_t* w2 = (rows > 2) ? w_ptrs[2] : nullptr;
    const uint8_t* w3 = (rows > 3) ? w_ptrs[3] : nullptr;

    for (int64_t l = 0; l < nb; ++l) {
        // Prefetch next blocks
        if (l + 4 < nb) {
            _mm_prefetch((const char*)(w0 + (l + 4) * BLOCK_BYTES), _MM_HINT_T0);
            if (rows > 1) _mm_prefetch((const char*)(w1 + (l + 4) * BLOCK_BYTES), _MM_HINT_T0);
            if (rows > 2) _mm_prefetch((const char*)(w2 + (l + 4) * BLOCK_BYTES), _MM_HINT_T0);
            if (rows > 3) _mm_prefetch((const char*)(w3 + (l + 4) * BLOCK_BYTES), _MM_HINT_T0);
        }

        // Load weight blocks (Q4_0 → 32 signed int8)
        __m256i wvec0 = BlockLoader<block_q4_0_tag>::load(w0 + l * BLOCK_BYTES);
        __m256i wvec1 = (rows > 1) ? BlockLoader<block_q4_0_tag>::load(w1 + l * BLOCK_BYTES) : _mm256_setzero_si256();
        __m256i wvec2 = (rows > 2) ? BlockLoader<block_q4_0_tag>::load(w2 + l * BLOCK_BYTES) : _mm256_setzero_si256();
        __m256i wvec3 = (rows > 3) ? BlockLoader<block_q4_0_tag>::load(w3 + l * BLOCK_BYTES) : _mm256_setzero_si256();

        // Load activation block (Q8_0_act → 32 signed int8)
        __m256i avec = BlockLoader<block_q8_0_act>::load(&q8_act[l]);

        // F16C: pack 4 fp16 weight scales, convert to 4 fp32 at once
        uint64_t packed_scales = 0;
        uint16_t d0, d1 = 0, d2 = 0, d3 = 0;
        memcpy(&d0, w0 + l * BLOCK_BYTES, 2);
        if (rows > 1) memcpy(&d1, w1 + l * BLOCK_BYTES, 2);
        if (rows > 2) memcpy(&d2, w2 + l * BLOCK_BYTES, 2);
        if (rows > 3) memcpy(&d3, w3 + l * BLOCK_BYTES, 2);
        packed_scales = (uint64_t)d0 | ((uint64_t)d1 << 16) | ((uint64_t)d2 << 32) | ((uint64_t)d3 << 48);

        __m128 sw_f16 = _mm_cvtph_ps(_mm_set_epi64x(0, packed_scales));
        float act_scale = q8_act[l].d;

        // Multiply weight scales by activation scale
        __m128 sw_all = _mm_mul_ps(sw_f16, _mm_set1_ps(act_scale));

        // Broadcast each scale to __m256 for FMADD
        __m256 sc0 = _mm256_castps128_ps256(_mm_shuffle_ps(sw_all, sw_all, 0x00));
        sc0 = _mm256_permute2f128_ps(sc0, sc0, 0x00);
        __m256 sc1 = _mm256_castps128_ps256(_mm_shuffle_ps(sw_all, sw_all, 0x55));
        sc1 = _mm256_permute2f128_ps(sc1, sc1, 0x00);
        __m256 sc2 = _mm256_castps128_ps256(_mm_shuffle_ps(sw_all, sw_all, 0xAA));
        sc2 = _mm256_permute2f128_ps(sc2, sc2, 0x00);
        __m256 sc3 = _mm256_castps128_ps256(_mm_shuffle_ps(sw_all, sw_all, 0xFF));
        sc3 = _mm256_permute2f128_ps(sc3, sc3, 0x00);

        // sign_epi8 + updot
        __m256i sa0 = _mm256_sign_epi8(wvec0, wvec0);
        __m256i sa1 = (rows > 1) ? _mm256_sign_epi8(wvec1, wvec1) : _mm256_setzero_si256();
        __m256i sa2 = (rows > 2) ? _mm256_sign_epi8(wvec2, wvec2) : _mm256_setzero_si256();
        __m256i sa3 = (rows > 3) ? _mm256_sign_epi8(wvec3, wvec3) : _mm256_setzero_si256();

        __m256i sb0 = _mm256_sign_epi8(avec, wvec0);
        __m256i sb1 = (rows > 1) ? _mm256_sign_epi8(avec, wvec1) : _mm256_setzero_si256();
        __m256i sb2 = (rows > 2) ? _mm256_sign_epi8(avec, wvec2) : _mm256_setzero_si256();
        __m256i sb3 = (rows > 3) ? _mm256_sign_epi8(avec, wvec3) : _mm256_setzero_si256();

        acc0 = _mm256_fmadd_ps(sc0, updot_avx2(sa0, sb0), acc0);
        acc1 = _mm256_fmadd_ps(sc1, updot_avx2(sa1, sb1), acc1);
        acc2 = _mm256_fmadd_ps(sc2, updot_avx2(sa2, sb2), acc2);
        acc3 = _mm256_fmadd_ps(sc3, updot_avx2(sa3, sb3), acc3);
    }

    // Horizontal sum and write outputs
    out[0] = vdot::hsum_ps256(acc0);
    if (rows > 1) out[1] = vdot::hsum_ps256(acc1);
    if (rows > 2) out[2] = vdot::hsum_ps256(acc2);
    if (rows > 3) out[3] = vdot::hsum_ps256(acc3);
}

// ============================================================================
// F16C-optimized variant: batch-convert 4 fp16 scales at once
// ============================================================================

static void gemm_q4_0_decode_f16c_avx2(const float* a_data,
                                         const uint8_t* w_data,
                                         float* o_data,
                                         int64_t K,
                                         int64_t N) {
    constexpr int RM = 4;
    constexpr int BLOCK_SIZE = 32;
    constexpr int BLOCK_BYTES = 18;
    const int64_t nb = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Quantize activation once
    scratch_vec<block_q8_0_act> q8_act(nb);
    quantize_row_q8_0_act(a_data, q8_act.data(), (int)K);

    #pragma omp parallel for schedule(static)
    for (int64_t n = 0; n < N; n += RM) {
        int64_t rows = (n + RM <= N) ? RM : (N - n);

        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        const uint8_t* w0 = w_data + (size_t)(n + 0) * nb * BLOCK_BYTES;
        const uint8_t* w1 = w_data + (size_t)(n + 1) * nb * BLOCK_BYTES;
        const uint8_t* w2 = w_data + (size_t)(n + 2) * nb * BLOCK_BYTES;
        const uint8_t* w3 = w_data + (size_t)(n + 3) * nb * BLOCK_BYTES;

        for (int64_t l = 0; l < nb; ++l) {
            // Prefetch
            if (l + 4 < nb) {
                _mm_prefetch((const char*)(w0 + (l + 4) * BLOCK_BYTES), _MM_HINT_T0);
                _mm_prefetch((const char*)(w1 + (l + 4) * BLOCK_BYTES), _MM_HINT_T0);
                _mm_prefetch((const char*)(w2 + (l + 4) * BLOCK_BYTES), _MM_HINT_T0);
                _mm_prefetch((const char*)(w3 + (l + 4) * BLOCK_BYTES), _MM_HINT_T0);
            }

            // Load weight blocks
            __m256i wvec0 = BlockLoader<block_q4_0_tag>::load(w0 + l * BLOCK_BYTES);
            __m256i wvec1 = (rows > 1) ? BlockLoader<block_q4_0_tag>::load(w1 + l * BLOCK_BYTES) : _mm256_setzero_si256();
            __m256i wvec2 = (rows > 2) ? BlockLoader<block_q4_0_tag>::load(w2 + l * BLOCK_BYTES) : _mm256_setzero_si256();
            __m256i wvec3 = (rows > 3) ? BlockLoader<block_q4_0_tag>::load(w3 + l * BLOCK_BYTES) : _mm256_setzero_si256();

            // Load activation block
            __m256i avec = BlockLoader<block_q8_0_act>::load(&q8_act[l]);

            // F16C: pack 4 fp16 weight scales, convert to 4 fp32 at once
            // Each Q4_0 block starts with uint16_t d (fp16 scale)
            uint64_t packed_scales = 0;
            uint16_t d0, d1, d2, d3;
            memcpy(&d0, w0 + l * BLOCK_BYTES, 2);
            memcpy(&d1, w1 + l * BLOCK_BYTES, 2);
            memcpy(&d2, w2 + l * BLOCK_BYTES, 2);
            memcpy(&d3, w3 + l * BLOCK_BYTES, 2);
            packed_scales = (uint64_t)d0 | ((uint64_t)d1 << 16) | ((uint64_t)d2 << 32) | ((uint64_t)d3 << 48);

            __m128 sw_f16 = _mm_cvtph_ps(_mm_set_epi64x(0, packed_scales));
            // sw_f16 = [d0_fp32, d1_fp32, d2_fp32, d3_fp32]
            float act_scale = q8_act[l].d;

            // Multiply weight scales by activation scale
            __m128 sw_all = _mm_mul_ps(sw_f16, _mm_set1_ps(act_scale));
            // sw_all = [d0*s, d1*s, d2*s, d3*s]

            // Broadcast each scale to __m256 for FMADD
            // Use shuffle to replicate each element across 8 lanes
            __m256 sc0 = _mm256_castps128_ps256(_mm_shuffle_ps(sw_all, sw_all, 0x00));  // d0*s
            sc0 = _mm256_permute2f128_ps(sc0, sc0, 0x00);
            __m256 sc1 = _mm256_castps128_ps256(_mm_shuffle_ps(sw_all, sw_all, 0x55));  // d1*s
            sc1 = _mm256_permute2f128_ps(sc1, sc1, 0x00);
            __m256 sc2 = _mm256_castps128_ps256(_mm_shuffle_ps(sw_all, sw_all, 0xAA));  // d2*s
            sc2 = _mm256_permute2f128_ps(sc2, sc2, 0x00);
            __m256 sc3 = _mm256_castps128_ps256(_mm_shuffle_ps(sw_all, sw_all, 0xFF));  // d3*s
            sc3 = _mm256_permute2f128_ps(sc3, sc3, 0x00);

            // sign_epi8 + updot
            __m256i sa0 = _mm256_sign_epi8(wvec0, wvec0);
            __m256i sa1 = _mm256_sign_epi8(wvec1, wvec1);
            __m256i sa2 = _mm256_sign_epi8(wvec2, wvec2);
            __m256i sa3 = _mm256_sign_epi8(wvec3, wvec3);

            __m256i sb0 = _mm256_sign_epi8(avec, wvec0);
            __m256i sb1 = (rows > 1) ? _mm256_sign_epi8(avec, wvec1) : _mm256_setzero_si256();
            __m256i sb2 = (rows > 2) ? _mm256_sign_epi8(avec, wvec2) : _mm256_setzero_si256();
            __m256i sb3 = (rows > 3) ? _mm256_sign_epi8(avec, wvec3) : _mm256_setzero_si256();

            acc0 = _mm256_fmadd_ps(sc0, updot_avx2(sa0, sb0), acc0);
            acc1 = _mm256_fmadd_ps(sc1, updot_avx2(sa1, sb1), acc1);
            acc2 = _mm256_fmadd_ps(sc2, updot_avx2(sa2, sb2), acc2);
            acc3 = _mm256_fmadd_ps(sc3, updot_avx2(sa3, sb3), acc3);
        }

        o_data[n + 0] = vdot::hsum_ps256(acc0);
        if (rows > 1) o_data[n + 1] = vdot::hsum_ps256(acc1);
        if (rows > 2) o_data[n + 2] = vdot::hsum_ps256(acc2);
        if (rows > 3) o_data[n + 3] = vdot::hsum_ps256(acc3);
    }
}

// ============================================================================
// Q4_0 Weight Repack for Decode GEMM
// ============================================================================
// Original layout (row-major):
//   Row n, block l: w[n * nb * 18 + l * 18]
//   RM=4 GEMM accesses w0[l], w1[l], w2[l], w3[l] — each separated by nb*18 bytes
//
// Repacked layout (4-row tile groups, blocks from same column contiguous):
//   Tile group g = n / 4,  row r = n % 4,  block l:
//     repack[(g * nb + l) * 4 * 18 + r * 18]
//
//   For a given g and l, the 4 blocks are at offsets 0, 18, 36, 54 — 72 bytes total.
//   This fits in ~1.1 cache lines, vs 4 separate cache lines in the original layout.
//
// Repack is a one-time cost at model load. Total memory usage is identical.

// Repack Q4_0 weight matrix for decode-optimized access.
// Returns a newly allocated buffer (caller must free with delete[]).
static inline std::pair<uint8_t*, size_t> repack_q4_0_weights(
    const uint8_t* w_data, int64_t K, int64_t N)
{
    constexpr int RM = 4;
    constexpr int BLOCK_SIZE = 32;
    constexpr int BLOCK_BYTES = 18;
    const int64_t nb = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int64_t row_stride = nb * BLOCK_BYTES;  // bytes per weight row

    // Same total size, just rearranged
    const size_t total_bytes = (size_t)N * row_stride;
    uint8_t* repacked = new uint8_t[total_bytes];

    const int64_t num_groups = (N + RM - 1) / RM;

    for (int64_t g = 0; g < num_groups; ++g) {
        int64_t n0 = g * RM;
        int64_t rows = std::min(n0 + RM, N) - n0;

        for (int64_t l = 0; l < nb; ++l) {
            // Destination: all 4 blocks for this (group, block) contiguous
            uint8_t* dst = repacked + (g * nb + l) * RM * BLOCK_BYTES;
            for (int64_t r = 0; r < rows; ++r) {
                const uint8_t* src = w_data + (n0 + r) * row_stride + l * BLOCK_BYTES;
                std::memcpy(dst + r * BLOCK_BYTES, src, BLOCK_BYTES);
            }
            // Zero-fill remaining rows (for partial groups)
            for (int64_t r = rows; r < RM; ++r) {
                std::memset(dst + r * BLOCK_BYTES, 0, BLOCK_BYTES);
            }
        }
    }

    return {repacked, total_bytes};
}

// ============================================================================
// Q4_0 8-row interleaved repack (llama.cpp block_q4_0x8, MIT) + 8x8 GEMV
// ============================================================================
// Ported from llama.cpp (ggml/src/ggml-cpu/repack.cpp: make_block_q4_0x8 and
// arch/x86/repack.cpp: gemv_q4_b32_8x8_q8_0_lut_avx). Interleaves 8 rows'
// same Q4_0 block so 4 contiguous 256-bit loads fetch 8 rows' quants at once
// (better prefetch/bandwidth than the RM=4 layout), and sign-extends nibbles
// via a LUT. Quants are XOR'd with 0x88 during repack to convert the Q4_0
// "nibble - 8" encoding to two's complement so the LUT maps directly.
// Total size is unchanged (144 B per 8 rows = 8 * 18 B).

struct block_q4_0x8 {
    uint16_t d[8];    // fp16 scale per row
    uint8_t  qs[128]; // 4-bit quants, 8 bytes per row, 8-way interleaved
};
static_assert(sizeof(block_q4_0x8) == 8 * 18, "block_q4_0x8 must be 144 bytes");

static inline std::pair<uint8_t*, size_t> repack_q4_0_weights_8x8(
    const uint8_t* w_data, int64_t K, int64_t N)
{
    constexpr int RM = 8;
    constexpr int BLOCK_SIZE = 32;
    constexpr int BLOCK_BYTES = 18;
    if (N % RM != 0 || K % BLOCK_SIZE != 0)
        return {nullptr, 0};
    const int64_t nb = K / BLOCK_SIZE;
    const int64_t row_stride = nb * BLOCK_BYTES;
    const size_t total_bytes = (size_t)(N / RM) * nb * sizeof(block_q4_0x8);
    uint8_t* repacked = new uint8_t[total_bytes];
    block_q4_0x8* out = reinterpret_cast<block_q4_0x8*>(repacked);
    constexpr uint8_t XOR = 0x88;

    for (int64_t g = 0; g < N / RM; ++g) {
        const uint8_t* rows[RM];
        for (int r = 0; r < RM; ++r)
            rows[r] = w_data + (size_t)(g * RM + r) * row_stride;
        for (int64_t l = 0; l < nb; ++l) {
            block_q4_0x8& dst = out[(size_t)g * nb + l];
            for (int r = 0; r < RM; ++r) {
                uint16_t d;
                std::memcpy(&d, rows[r] + l * BLOCK_BYTES, 2);
                dst.d[r] = d;
            }
            // Interleave 8 rows' 16 bytes of nibbles in 8-byte chunks.
            for (int i = 0; i < 16; ++i) {
                const int src_id = i % 8;
                const int src_off = (i / 8) * 8;
                const int dst_off = i * 8;
                const uint8_t* src = rows[src_id] + l * BLOCK_BYTES + 2 + src_off;
                for (int k = 0; k < 8; ++k)
                    dst.qs[dst_off + k] = src[k] ^ XOR;
            }
        }
    }
    return {repacked, total_bytes};
}

// int8 x int8 dot into 8 int32 lanes, sign trick (llama mul_sum_i8 helpers)
static inline __m256i mul_sum_i8_pairs_acc_int32x8(const __m256i acc,
                                                   const __m256i x,
                                                   const __m256i y) {
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
    const __m256i dot = _mm256_maddubs_epi16(ax, sy);
    return _mm256_add_epi32(acc, _mm256_madd_epi16(dot, _mm256_set1_epi16(1)));
}

// Load 8 fp16 scales and rearrange lanes to [d0,d4,d2,d6,d1,d5,d3,d7] to
// match the [B0,B4,B1,B5,B2,B6,B3,B7] lane order of the 8x8 accumulator.
static inline __m256 load_col_scale_q4_0x8_rearranged(const uint16_t* d8) {
    const __m128i mask = _mm_set_epi8(15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0);
    return _mm256_cvtph_ps(_mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)d8), mask));
}

// 8-lane int8 dot of one Q8_0 activation block against pre-decoded 8-row
// weights (the v_* vectors from a block_q4_0x8). Returns the int32 lanes in
// [B0,B4,B1,B5,B2,B6,B3,B7] order (callers permute via finalperm before store).
static inline __m256i gemv_q4_0_8x8_dots_8lanes(
    const __m256i& v_0123_0, const __m256i& v_4567_0,
    const __m256i& v_0123_1, const __m256i& v_4567_1,
    const __m256i& v_0123_2, const __m256i& v_4567_2,
    const __m256i& v_0123_3, const __m256i& v_4567_3,
    const int8_t* aqs) {
    const __m128i a_lo = _mm_loadu_si128((const __m128i*)aqs);
    const __m128i a_hi = _mm_loadu_si128((const __m128i*)(aqs + 16));
    const __m256i lhs_0 = _mm256_permute2f128_si256(
        _mm256_castsi128_si256(a_lo), _mm256_castsi128_si256(a_lo), 0);
    const __m256i lhs_1 = _mm256_permute2f128_si256(
        _mm256_castsi128_si256(a_hi), _mm256_castsi128_si256(a_hi), 0);

    __m256i iacc = _mm256_setzero_si256();
    iacc = mul_sum_i8_pairs_acc_int32x8(iacc, _mm256_blend_epi32(v_0123_0, _mm256_shuffle_epi32(v_4567_0, 177), 170), _mm256_shuffle_epi32(lhs_0, 0));
    iacc = mul_sum_i8_pairs_acc_int32x8(iacc, _mm256_blend_epi32(_mm256_shuffle_epi32(v_0123_0, 177), v_4567_0, 170), _mm256_shuffle_epi32(lhs_0, 85));
    iacc = mul_sum_i8_pairs_acc_int32x8(iacc, _mm256_blend_epi32(v_0123_1, _mm256_shuffle_epi32(v_4567_1, 177), 170), _mm256_shuffle_epi32(lhs_0, 170));
    iacc = mul_sum_i8_pairs_acc_int32x8(iacc, _mm256_blend_epi32(_mm256_shuffle_epi32(v_0123_1, 177), v_4567_1, 170), _mm256_shuffle_epi32(lhs_0, 255));
    iacc = mul_sum_i8_pairs_acc_int32x8(iacc, _mm256_blend_epi32(v_0123_2, _mm256_shuffle_epi32(v_4567_2, 177), 170), _mm256_shuffle_epi32(lhs_1, 0));
    iacc = mul_sum_i8_pairs_acc_int32x8(iacc, _mm256_blend_epi32(_mm256_shuffle_epi32(v_0123_2, 177), v_4567_2, 170), _mm256_shuffle_epi32(lhs_1, 85));
    iacc = mul_sum_i8_pairs_acc_int32x8(iacc, _mm256_blend_epi32(v_0123_3, _mm256_shuffle_epi32(v_4567_3, 177), 170), _mm256_shuffle_epi32(lhs_1, 170));
    iacc = mul_sum_i8_pairs_acc_int32x8(iacc, _mm256_blend_epi32(_mm256_shuffle_epi32(v_0123_3, 177), v_4567_3, 170), _mm256_shuffle_epi32(lhs_1, 255));
    return iacc;
}

// Decode a block_q4_0x8 into the 8 row vectors + lane-ordered fp32 col scale.
static inline void gemv_q4_0_8x8_decode_block(const block_q4_0x8* blk,
                                              const __m256i& lut, const __m256i& m4b,
                                              __m256i& v_0123_0, __m256i& v_4567_0,
                                              __m256i& v_0123_1, __m256i& v_4567_1,
                                              __m256i& v_0123_2, __m256i& v_4567_2,
                                              __m256i& v_0123_3, __m256i& v_4567_3,
                                              __m256& col_scale) {
    const __m256i raw_0123_0 = _mm256_loadu_si256((const __m256i*)(blk->qs));
    const __m256i raw_4567_0 = _mm256_loadu_si256((const __m256i*)(blk->qs + 32));
    const __m256i raw_0123_1 = _mm256_loadu_si256((const __m256i*)(blk->qs + 64));
    const __m256i raw_4567_1 = _mm256_loadu_si256((const __m256i*)(blk->qs + 96));

    v_0123_0 = _mm256_shuffle_epi8(lut, _mm256_and_si256(raw_0123_0, m4b));
    v_4567_0 = _mm256_shuffle_epi8(lut, _mm256_and_si256(raw_4567_0, m4b));
    v_0123_1 = _mm256_shuffle_epi8(lut, _mm256_and_si256(raw_0123_1, m4b));
    v_4567_1 = _mm256_shuffle_epi8(lut, _mm256_and_si256(raw_4567_1, m4b));
    v_0123_2 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(raw_0123_0, 4), m4b));
    v_4567_2 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(raw_4567_0, 4), m4b));
    v_0123_3 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(raw_0123_1, 4), m4b));
    v_4567_3 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(raw_4567_1, 4), m4b));

    col_scale = load_col_scale_q4_0x8_rearranged(blk->d);
}

// Dot one 8-row group of block_q4_0x8 against the Q8_0 activation; returns
// the accumulator with lanes in [B0,B4,B1,B5,B2,B6,B3,B7] order (callers
// permute via finalperm before storing).
static inline __m256 gemv_q4_0_8x8_dot_group(const uint8_t* w8_group,
                                             const block_q8_0_act* q8_act, int64_t nb,
                                             const __m256i& lut, const __m256i& m4b) {
    const block_q4_0x8* b_ptr = reinterpret_cast<const block_q4_0x8*>(w8_group);
    __m256 acc = _mm256_setzero_ps();

    for (int64_t b = 0; b < nb; ++b) {
        if (b + 4 < nb)
            _mm_prefetch((const char*)(b_ptr + b + 4), _MM_HINT_T0);

        __m256i v_0123_0, v_4567_0, v_0123_1, v_4567_1, v_0123_2, v_4567_2, v_0123_3, v_4567_3;
        __m256 col_scale;
        gemv_q4_0_8x8_decode_block(b_ptr + b, lut, m4b,
                                   v_0123_0, v_4567_0, v_0123_1, v_4567_1,
                                   v_0123_2, v_4567_2, v_0123_3, v_4567_3, col_scale);

        const __m256 row_scale = _mm256_set1_ps(q8_act[b].d);
        const __m256 scale = _mm256_mul_ps(col_scale, row_scale);

        const __m256i iacc = gemv_q4_0_8x8_dots_8lanes(
            v_0123_0, v_4567_0, v_0123_1, v_4567_1,
            v_0123_2, v_4567_2, v_0123_3, v_4567_3, q8_act[b].qs);

        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc), scale, acc);
    }
    return acc;
}

// Q4_0 8x8 batch GEMM (prefill, M > 1) on the block_q4_0x8 layout.
// Each weight block is decoded once and reused across all M activation rows
// (weight DRAM traffic = 1x); accumulators stay lane-ordered in a scratch
// buffer, so no horizontal reductions are needed in the inner loop.
static void gemm_q4_0_8x8_batch_avx2(
    const float* a_data,     // [M, K] FP32 activation
    const uint8_t* w_8x8,    // block_q4_0x8 repacked weights
    float* o_data,           // [M, N] FP32 output
    int64_t M, int64_t K, int64_t N)
{
    constexpr int RM = 8;
    constexpr int BLOCK_SIZE = 32;
    const int64_t nb = K / BLOCK_SIZE;
    const int64_t num_groups = N / RM;

    scratch_vec<block_q8_0_act> q8(M * nb);
    for (int64_t m = 0; m < M; ++m)
        quantize_row_q8_0_act(a_data + m * K, q8.data() + m * nb, (int)K);

    const __m128i lut128 = _mm_set_epi8(-1, -2, -3, -4, -5, -6, -7, -8,
                                        7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i lut = _mm256_permute2f128_si256(
        _mm256_castsi128_si256(lut128), _mm256_castsi128_si256(lut128), 0);
    const __m256i m4b = _mm256_set1_epi8(0x0F);
    const __m256i finalperm = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);

    #pragma omp parallel
    {
        const int64_t ith = omp_get_thread_num();
        const int64_t nth = omp_get_num_threads();
        const int64_t g_begin = (num_groups * ith) / nth;
        const int64_t g_end = (num_groups * (ith + 1)) / nth;
        alignas(32) std::vector<float> acc(M * RM);
        for (int64_t g = g_begin; g < g_end; ++g) {
            const block_q4_0x8* gb = reinterpret_cast<const block_q4_0x8*>(
                w_8x8 + g * nb * sizeof(block_q4_0x8));
            std::fill(acc.begin(), acc.end(), 0.0f);

            for (int64_t b = 0; b < nb; ++b) {
                if (b + 4 < nb)
                    _mm_prefetch((const char*)(gb + b + 4), _MM_HINT_T0);

                __m256i v_0123_0, v_4567_0, v_0123_1, v_4567_1, v_0123_2, v_4567_2, v_0123_3, v_4567_3;
                __m256 col_scale;
                gemv_q4_0_8x8_decode_block(gb + b, lut, m4b,
                                           v_0123_0, v_4567_0, v_0123_1, v_4567_1,
                                           v_0123_2, v_4567_2, v_0123_3, v_4567_3, col_scale);

                for (int64_t m = 0; m < M; ++m) {
                    const block_q8_0_act* ab = q8.data() + m * nb + b;
                    const __m256 scale = _mm256_mul_ps(col_scale, _mm256_set1_ps(ab->d));
                    const __m256i iacc = gemv_q4_0_8x8_dots_8lanes(
                        v_0123_0, v_4567_0, v_0123_1, v_4567_1,
                        v_0123_2, v_4567_2, v_0123_3, v_4567_3, ab->qs);
                    __m256 cur = _mm256_loadu_ps(acc.data() + m * RM);
                    _mm256_storeu_ps(acc.data() + m * RM,
                                     _mm256_add_ps(cur, _mm256_mul_ps(_mm256_cvtepi32_ps(iacc), scale)));
                }
            }

            for (int64_t m = 0; m < M; ++m)
                _mm256_storeu_ps(o_data + m * N + g * RM,
                                 _mm256_permutevar8x32_ps(
                                     _mm256_loadu_ps(acc.data() + m * RM), finalperm));
        }
    }
}

// Q4_0 8x8 decode GEMV (M=1) on the block_q4_0x8 layout, Q8_0 activation.
static void gemv_q4_0_8x8_q8_0_avx2(const float* a_data, const uint8_t* w_repacked,
                                    float* o_data, int64_t K, int64_t N) {
    constexpr int RM = 8;
    constexpr int BLOCK_SIZE = 32;
    const int64_t nb = K / BLOCK_SIZE;
    const int64_t num_groups = N / RM;

    scratch_vec<block_q8_0_act> q8_act(nb);
    quantize_row_q8_0_act(a_data, q8_act.data(), (int)K);

    const __m128i lut128 = _mm_set_epi8(-1, -2, -3, -4, -5, -6, -7, -8,
                                        7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i lut = _mm256_permute2f128_si256(
        _mm256_castsi128_si256(lut128), _mm256_castsi128_si256(lut128), 0);
    const __m256i m4b = _mm256_set1_epi8(0x0F);
    const __m256i finalperm = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);

    #pragma omp parallel for schedule(static)
    for (int64_t g = 0; g < num_groups; ++g) {
        __m256 acc_row = gemv_q4_0_8x8_dot_group(
            w_repacked + g * nb * sizeof(block_q4_0x8), q8_act.data(), nb, lut, m4b);
        _mm256_storeu_ps(o_data + g * RM, _mm256_permutevar8x32_ps(acc_row, finalperm));
    }
}

// Q4_0 8x8 decode GEMV (M=1) with elementwise residual add (used for both the
// FFN down-projection and the attention output projection): out = a@w + r.
static void gemv_q4_0_8x8_residual_avx2(const float* a_data, const uint8_t* w_repacked,
                                        const float* r_data, float* o_data,
                                        int64_t K, int64_t N) {
    constexpr int RM = 8;
    constexpr int BLOCK_SIZE = 32;
    const int64_t nb = K / BLOCK_SIZE;
    const int64_t num_groups = N / RM;

    scratch_vec<block_q8_0_act> q8_act(nb);
    quantize_row_q8_0_act(a_data, q8_act.data(), (int)K);

    const __m128i lut128 = _mm_set_epi8(-1, -2, -3, -4, -5, -6, -7, -8,
                                        7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i lut = _mm256_permute2f128_si256(
        _mm256_castsi128_si256(lut128), _mm256_castsi128_si256(lut128), 0);
    const __m256i m4b = _mm256_set1_epi8(0x0F);
    const __m256i finalperm = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);

    #pragma omp parallel for schedule(static)
    for (int64_t g = 0; g < num_groups; ++g) {
        __m256 acc_row = gemv_q4_0_8x8_dot_group(
            w_repacked + g * nb * sizeof(block_q4_0x8), q8_act.data(), nb, lut, m4b);
        const __m256 res = _mm256_loadu_ps(r_data + g * RM);
        _mm256_storeu_ps(o_data + g * RM,
                         _mm256_add_ps(_mm256_permutevar8x32_ps(acc_row, finalperm), res));
    }
}

// Fused FFN gate+up Q4_0 8x8 decode GEMV (M=1): computes out = silu(gate)*up
// over the same activation in a single pass over both 8x8 repacked matrices.
static void gemv_q4_0_fused_ffn_up_8x8_avx2(const float* a_data,
                                            const uint8_t* w_gate_8x8,
                                            const uint8_t* w_up_8x8,
                                            float* o_data, int64_t K, int64_t N) {
    constexpr int RM = 8;
    constexpr int BLOCK_SIZE = 32;
    const int64_t nb = K / BLOCK_SIZE;
    const int64_t num_groups = N / RM;

    scratch_vec<block_q8_0_act> q8_act(nb);
    quantize_row_q8_0_act(a_data, q8_act.data(), (int)K);

    const __m128i lut128 = _mm_set_epi8(-1, -2, -3, -4, -5, -6, -7, -8,
                                        7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i lut = _mm256_permute2f128_si256(
        _mm256_castsi128_si256(lut128), _mm256_castsi128_si256(lut128), 0);
    const __m256i m4b = _mm256_set1_epi8(0x0F);
    const __m256i finalperm = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);

    auto silu = [](float x) -> float { return x / (1.0f + std::exp(-x)); };

    #pragma omp parallel for schedule(static)
    for (int64_t g = 0; g < num_groups; ++g) {
        const block_q4_0x8* gb = reinterpret_cast<const block_q4_0x8*>(
            w_gate_8x8 + g * nb * sizeof(block_q4_0x8));
        const block_q4_0x8* ub = reinterpret_cast<const block_q4_0x8*>(
            w_up_8x8 + g * nb * sizeof(block_q4_0x8));
        __m256 acc_g = _mm256_setzero_ps();
        __m256 acc_u = _mm256_setzero_ps();

        for (int64_t b = 0; b < nb; ++b) {
            if (b + 4 < nb) {
                _mm_prefetch((const char*)(gb + b + 4), _MM_HINT_T0);
                _mm_prefetch((const char*)(ub + b + 4), _MM_HINT_T0);
            }

            const __m256 row_scale = _mm256_set1_ps(q8_act[b].d);
            const __m128i a0 = _mm_loadu_si128((const __m128i*)q8_act[b].qs);
            const __m128i a1 = _mm_loadu_si128((const __m128i*)(q8_act[b].qs + 16));
            const __m256i lhs_0 = _mm256_permute2f128_si256(
                _mm256_castsi128_si256(a0), _mm256_castsi128_si256(a0), 0);
            const __m256i lhs_1 = _mm256_permute2f128_si256(
                _mm256_castsi128_si256(a1), _mm256_castsi128_si256(a1), 0);

            // ---- gate ----
            {
                const __m256i raw_0 = _mm256_loadu_si256((const __m256i*)(gb[b].qs));
                const __m256i raw_1 = _mm256_loadu_si256((const __m256i*)(gb[b].qs + 32));
                const __m256i raw_2 = _mm256_loadu_si256((const __m256i*)(gb[b].qs + 64));
                const __m256i raw_3 = _mm256_loadu_si256((const __m256i*)(gb[b].qs + 96));
                const __m256i v_0123_0 = _mm256_shuffle_epi8(lut, _mm256_and_si256(raw_0, m4b));
                const __m256i v_4567_0 = _mm256_shuffle_epi8(lut, _mm256_and_si256(raw_1, m4b));
                const __m256i v_0123_1 = _mm256_shuffle_epi8(lut, _mm256_and_si256(raw_2, m4b));
                const __m256i v_4567_1 = _mm256_shuffle_epi8(lut, _mm256_and_si256(raw_3, m4b));
                const __m256i v_0123_2 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(raw_0, 4), m4b));
                const __m256i v_4567_2 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(raw_1, 4), m4b));
                const __m256i v_0123_3 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(raw_2, 4), m4b));
                const __m256i v_4567_3 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(raw_3, 4), m4b));
                const __m256 scale = _mm256_mul_ps(load_col_scale_q4_0x8_rearranged(gb[b].d), row_scale);
                __m256i iacc = _mm256_setzero_si256();
                iacc = mul_sum_i8_pairs_acc_int32x8(iacc, _mm256_blend_epi32(v_0123_0, _mm256_shuffle_epi32(v_4567_0, 177), 170), _mm256_shuffle_epi32(lhs_0, 0));
                iacc = mul_sum_i8_pairs_acc_int32x8(iacc, _mm256_blend_epi32(_mm256_shuffle_epi32(v_0123_0, 177), v_4567_0, 170), _mm256_shuffle_epi32(lhs_0, 85));
                iacc = mul_sum_i8_pairs_acc_int32x8(iacc, _mm256_blend_epi32(v_0123_1, _mm256_shuffle_epi32(v_4567_1, 177), 170), _mm256_shuffle_epi32(lhs_0, 170));
                iacc = mul_sum_i8_pairs_acc_int32x8(iacc, _mm256_blend_epi32(_mm256_shuffle_epi32(v_0123_1, 177), v_4567_1, 170), _mm256_shuffle_epi32(lhs_0, 255));
                iacc = mul_sum_i8_pairs_acc_int32x8(iacc, _mm256_blend_epi32(v_0123_2, _mm256_shuffle_epi32(v_4567_2, 177), 170), _mm256_shuffle_epi32(lhs_1, 0));
                iacc = mul_sum_i8_pairs_acc_int32x8(iacc, _mm256_blend_epi32(_mm256_shuffle_epi32(v_0123_2, 177), v_4567_2, 170), _mm256_shuffle_epi32(lhs_1, 85));
                iacc = mul_sum_i8_pairs_acc_int32x8(iacc, _mm256_blend_epi32(v_0123_3, _mm256_shuffle_epi32(v_4567_3, 177), 170), _mm256_shuffle_epi32(lhs_1, 170));
                iacc = mul_sum_i8_pairs_acc_int32x8(iacc, _mm256_blend_epi32(_mm256_shuffle_epi32(v_0123_3, 177), v_4567_3, 170), _mm256_shuffle_epi32(lhs_1, 255));
                acc_g = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc), scale, acc_g);
            }

            // ---- up ----
            {
                const __m256i raw_0 = _mm256_loadu_si256((const __m256i*)(ub[b].qs));
                const __m256i raw_1 = _mm256_loadu_si256((const __m256i*)(ub[b].qs + 32));
                const __m256i raw_2 = _mm256_loadu_si256((const __m256i*)(ub[b].qs + 64));
                const __m256i raw_3 = _mm256_loadu_si256((const __m256i*)(ub[b].qs + 96));
                const __m256i v_0123_0 = _mm256_shuffle_epi8(lut, _mm256_and_si256(raw_0, m4b));
                const __m256i v_4567_0 = _mm256_shuffle_epi8(lut, _mm256_and_si256(raw_1, m4b));
                const __m256i v_0123_1 = _mm256_shuffle_epi8(lut, _mm256_and_si256(raw_2, m4b));
                const __m256i v_4567_1 = _mm256_shuffle_epi8(lut, _mm256_and_si256(raw_3, m4b));
                const __m256i v_0123_2 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(raw_0, 4), m4b));
                const __m256i v_4567_2 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(raw_1, 4), m4b));
                const __m256i v_0123_3 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(raw_2, 4), m4b));
                const __m256i v_4567_3 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(raw_3, 4), m4b));
                const __m256 scale = _mm256_mul_ps(load_col_scale_q4_0x8_rearranged(ub[b].d), row_scale);
                __m256i iacc = _mm256_setzero_si256();
                iacc = mul_sum_i8_pairs_acc_int32x8(iacc, _mm256_blend_epi32(v_0123_0, _mm256_shuffle_epi32(v_4567_0, 177), 170), _mm256_shuffle_epi32(lhs_0, 0));
                iacc = mul_sum_i8_pairs_acc_int32x8(iacc, _mm256_blend_epi32(_mm256_shuffle_epi32(v_0123_0, 177), v_4567_0, 170), _mm256_shuffle_epi32(lhs_0, 85));
                iacc = mul_sum_i8_pairs_acc_int32x8(iacc, _mm256_blend_epi32(v_0123_1, _mm256_shuffle_epi32(v_4567_1, 177), 170), _mm256_shuffle_epi32(lhs_0, 170));
                iacc = mul_sum_i8_pairs_acc_int32x8(iacc, _mm256_blend_epi32(_mm256_shuffle_epi32(v_0123_1, 177), v_4567_1, 170), _mm256_shuffle_epi32(lhs_0, 255));
                iacc = mul_sum_i8_pairs_acc_int32x8(iacc, _mm256_blend_epi32(v_0123_2, _mm256_shuffle_epi32(v_4567_2, 177), 170), _mm256_shuffle_epi32(lhs_1, 0));
                iacc = mul_sum_i8_pairs_acc_int32x8(iacc, _mm256_blend_epi32(_mm256_shuffle_epi32(v_0123_2, 177), v_4567_2, 170), _mm256_shuffle_epi32(lhs_1, 85));
                iacc = mul_sum_i8_pairs_acc_int32x8(iacc, _mm256_blend_epi32(v_0123_3, _mm256_shuffle_epi32(v_4567_3, 177), 170), _mm256_shuffle_epi32(lhs_1, 170));
                iacc = mul_sum_i8_pairs_acc_int32x8(iacc, _mm256_blend_epi32(_mm256_shuffle_epi32(v_0123_3, 177), v_4567_3, 170), _mm256_shuffle_epi32(lhs_1, 255));
                acc_u = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc), scale, acc_u);
            }
        }

        const __m256 g_row = _mm256_permutevar8x32_ps(acc_g, finalperm);
        const __m256 u_row = _mm256_permutevar8x32_ps(acc_u, finalperm);
        float gr[8], ur[8];
        _mm256_storeu_ps(gr, g_row);
        _mm256_storeu_ps(ur, u_row);
        for (int r = 0; r < RM; ++r)
            o_data[g * RM + r] = silu(gr[r]) * ur[r];
    }
}
// ============================================================================
// Ported from llama.cpp (ggml/src/ggml-cpu/repack.cpp: make_block_q4_Kx8 and
// repack_q4_K_to_q4_K_8_bl). Interleaves 8 rows' same super-block so one
// 256-bit load grabs 8 rows' quants at once, enabling the 8x8 decode kernel
// below. Total size is unchanged (1152 B per 8 rows = 8 * 144 B).

struct block_q4_Kx8 {
    uint16_t d[8];        // super-block scale (fp16) per row
    uint16_t dmin[8];     // super-block min (fp16) per row
    uint8_t  scales[96];  // scales + mins, 6-bit packed (8 rows interleaved)
    uint8_t  qs[1024];    // 4-bit quants, 8-byte interleaved
};
static_assert(sizeof(block_q4_Kx8) == 8 * sizeof(block_q4_K), "block_q4_Kx8 must be 1152 bytes");

static inline block_q4_Kx8 make_block_q4_Kx8(const block_q4_K* in, unsigned blck_size_interleave) {
    constexpr int QK = 256;
    block_q4_Kx8 out;
    for (int i = 0; i < 8; i++) {
        out.d[i] = in[i].d;
    }
    for (int i = 0; i < 8; i++) {
        out.dmin[i] = in[i].dmin;
    }

    const int end = (QK * 4) / static_cast<int>(blck_size_interleave);  // 1024/8 = 128
    for (int i = 0; i < end; ++i) {
        const int src_id = i % 8;
        const int src_offset = (i / 8) * static_cast<int>(blck_size_interleave);
        const int dst_offset = i * static_cast<int>(blck_size_interleave);
        uint64_t elems;
        std::memcpy(&elems, &in[src_id].qs[src_offset], blck_size_interleave);
        std::memcpy(&out.qs[dst_offset], &elems, blck_size_interleave);
    }

    // Unpack and rearrange the 6-bit scales + mins: each output 12-byte group
    // holds the scales/mins of the corresponding sub-block across the 8 rows.
    uint8_t s[8], m[8];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            s[j] = in[j].scales[i] & 63;
            m[j] = in[j].scales[i + 4] & 63;
        }
        out.scales[i * 12 + 0]  = (s[0] & 63) + ((s[4] & 48) << 2);
        out.scales[i * 12 + 1]  = (s[1] & 63) + ((s[5] & 48) << 2);
        out.scales[i * 12 + 2]  = (s[2] & 63) + ((s[6] & 48) << 2);
        out.scales[i * 12 + 3]  = (s[3] & 63) + ((s[7] & 48) << 2);
        out.scales[i * 12 + 4]  = (m[0] & 63) + ((m[4] & 48) << 2);
        out.scales[i * 12 + 5]  = (m[1] & 63) + ((m[5] & 48) << 2);
        out.scales[i * 12 + 6]  = (m[2] & 63) + ((m[6] & 48) << 2);
        out.scales[i * 12 + 7]  = (m[3] & 63) + ((m[7] & 48) << 2);
        out.scales[i * 12 + 8]  = (s[4] & 15) + ((m[4] & 15) << 4);
        out.scales[i * 12 + 9]  = (s[5] & 15) + ((m[5] & 15) << 4);
        out.scales[i * 12 + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
        out.scales[i * 12 + 11] = (s[7] & 15) + ((m[7] & 15) << 4);
    }
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            s[j] = ((in[j].scales[i] & 192) >> 2) | (in[j].scales[i + 8] & 15);
            m[j] = ((in[j].scales[i + 4] & 192) >> 2) | ((in[j].scales[i + 8] & 240) >> 4);
        }
        out.scales[i * 12 + 48] = (s[0] & 63) + ((s[4] & 48) << 2);
        out.scales[i * 12 + 49] = (s[1] & 63) + ((s[5] & 48) << 2);
        out.scales[i * 12 + 50] = (s[2] & 63) + ((s[6] & 48) << 2);
        out.scales[i * 12 + 51] = (s[3] & 63) + ((s[7] & 48) << 2);
        out.scales[i * 12 + 52] = (m[0] & 63) + ((m[4] & 48) << 2);
        out.scales[i * 12 + 53] = (m[1] & 63) + ((m[5] & 48) << 2);
        out.scales[i * 12 + 54] = (m[2] & 63) + ((m[6] & 48) << 2);
        out.scales[i * 12 + 55] = (m[3] & 63) + ((m[7] & 48) << 2);
        out.scales[i * 12 + 56] = (s[4] & 15) + ((m[4] & 15) << 4);
        out.scales[i * 12 + 57] = (s[5] & 15) + ((m[5] & 15) << 4);
        out.scales[i * 12 + 58] = (s[6] & 15) + ((m[6] & 15) << 4);
        out.scales[i * 12 + 59] = (s[7] & 15) + ((m[7] & 15) << 4);
    }
    return out;
}

// Repack a Q4_K weight matrix [N, K] into the 8-row interleaved layout.
// Returns nullptr for shapes that don't fit (N % 8 != 0 or K % 256 != 0).
static inline std::pair<uint8_t*, size_t> repack_q4_K_weights(
    const uint8_t* w_data, int64_t K, int64_t N)
{
    constexpr int QK = 256;
    constexpr int nrows_interleaved = 8;
    const int64_t nblocks = (K + QK - 1) / QK;
    const size_t row_stride = static_cast<size_t>(nblocks) * sizeof(block_q4_K);

    if (N % nrows_interleaved != 0 || K % QK != 0)
        return {nullptr, 0};

    const size_t total_bytes = static_cast<size_t>(N) * row_stride;
    uint8_t* repacked = new uint8_t[total_bytes];

    block_q4_Kx8* dst = reinterpret_cast<block_q4_Kx8*>(repacked);
    const block_q4_K* src = reinterpret_cast<const block_q4_K*>(w_data);
    block_q4_K dst_tmp[8];

    for (int64_t b = 0; b < N; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; ++x) {
            for (int i = 0; i < nrows_interleaved; ++i) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q4_Kx8(dst_tmp, 8);
        }
        src += nrows_interleaved * nblocks;
    }
    return {repacked, total_bytes};
}

// ============================================================================
// Q6_K 8-row interleaved repack (llama.cpp block_q6_Kx8, MIT)
// ============================================================================
// Ported from llama.cpp (ggml/src/ggml-cpu/repack.cpp: make_block_q6_Kx8 and
// repack_q6_K_to_q6_K_8_bl). Interleaves 8 rows' same super-block so one read
// grabs 8 rows' quants at once, enabling the 8x8 decode kernel. Total size is
// unchanged (1680 B per 8 rows = 8 * 210 B).

struct block_q6_Kx8 {
    uint16_t d[8];        // super-block scale (fp16) per row
    int8_t   scales[128]; // 16 scales per block, 8 rows interleaved
    uint8_t  ql[1024];    // 4-bit low quants, 8-byte interleaved
    uint8_t  qh[512];     // 2-bit high quants, 8-byte interleaved
};
static_assert(sizeof(block_q6_Kx8) == 8 * sizeof(block_q6_K), "block_q6_Kx8 must be 1680 bytes");

static inline block_q6_Kx8 make_block_q6_Kx8(const block_q6_K* in, unsigned blck_size_interleave) {
    constexpr int QK = 256;
    block_q6_Kx8 out;
    for (int i = 0; i < 8; i++) {
        out.d[i] = in[i].d;
    }

    const int end_ls = (QK * 4) / static_cast<int>(blck_size_interleave);  // 1024/8 = 128
    for (int i = 0; i < end_ls; ++i) {
        const int src_id = i % 8;
        const int src_offset = (i / 8) * static_cast<int>(blck_size_interleave);
        const int dst_offset = i * static_cast<int>(blck_size_interleave);
        uint64_t elem;
        std::memcpy(&elem, &in[src_id].ql[src_offset], blck_size_interleave);
        std::memcpy(&out.ql[dst_offset], &elem, blck_size_interleave);
    }

    const int end_hs = end_ls / 2;  // 512/8 = 64
    for (int i = 0; i < end_hs; ++i) {
        const int src_id = i % 8;
        const int src_offset = (i / 8) * static_cast<int>(blck_size_interleave);
        const int dst_offset = i * static_cast<int>(blck_size_interleave);
        uint64_t elem;
        std::memcpy(&elem, &in[src_id].qh[src_offset], blck_size_interleave);
        std::memcpy(&out.qh[dst_offset], &elem, blck_size_interleave);
    }

    // scales: [s0 bl0 .. s0 bl7][s1 bl0 .. s1 bl7] ... [s15 bl0 .. s15 bl7]
    constexpr int n_scales = QK / 16;  // 16 scales per super-block
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < n_scales; j++) {
            out.scales[j * 8 + i] = in[i].scales[j];
        }
    }
    return out;
}

// Repack a Q6_K weight matrix [N, K] into the 8-row interleaved layout.
// Returns nullptr for shapes that don't fit (N % 8 != 0 or K % 256 != 0).
static inline std::pair<uint8_t*, size_t> repack_q6_K_weights(
    const uint8_t* w_data, int64_t K, int64_t N)
{
    constexpr int QK = 256;
    constexpr int nrows_interleaved = 8;
    const int64_t nblocks = (K + QK - 1) / QK;
    const size_t row_stride = static_cast<size_t>(nblocks) * sizeof(block_q6_K);

    if (N % nrows_interleaved != 0 || K % QK != 0 || K % 8 != 0)
        return {nullptr, 0};

    const size_t total_bytes = static_cast<size_t>(N) * row_stride;
    uint8_t* repacked = new uint8_t[total_bytes];

    block_q6_Kx8* dst = reinterpret_cast<block_q6_Kx8*>(repacked);
    const block_q6_K* src = reinterpret_cast<const block_q6_K*>(w_data);
    block_q6_K dst_tmp[8];

    for (int64_t b = 0; b < N; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; ++x) {
            for (int i = 0; i < nrows_interleaved; ++i) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q6_Kx8(dst_tmp, 8);
        }
        src += nrows_interleaved * nblocks;
    }
    return {repacked, total_bytes};
}

// ============================================================================
// Q4_K 8x8 Decode GEMV (M=1) — llama.cpp gemv_q4_K_8x8_q8_K, MIT
// ============================================================================
// Processes 8 output rows at a time from the interleaved block_q4_Kx8 layout,
// with 8 lanes of accumulators and a single shared activation quantization.
// Ported from llama.cpp ggml/src/ggml-cpu/arch/x86/repack.cpp:1464.

static void gemv_q4_K_8x8_q8_K_avx2(
    const float* a_data, const uint8_t* w_repacked, float* o_data, int K, int N)
{
    constexpr int QK = 256;
    constexpr int ncols_interleaved = 8;
    const int nb = K / QK;
    if (nb <= 0 || N % ncols_interleaved != 0)
        return;

    scratch_vec<block_q8_K> q8_buf(nb);
    quantize_row_q8_K(a_data, q8_buf.data(), K);

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    const __m256i m4b = _mm256_set1_epi8(0x0F);

    // Shuffle masks to rearrange delta and scale values for 8-row multiply.
    const __m128i deltamask = _mm_set_epi8(15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0);
    const __m128i scalemask = _mm_set_epi8(7, 7, 3, 3, 6, 6, 2, 2, 5, 5, 1, 1, 4, 4, 0, 0);
    const __m256i finalpermutemask = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);

    const int64_t b_nb = K / QK;
    const block_q4_Kx8* b_ptr_start = reinterpret_cast<const block_q4_Kx8*>(w_repacked);
    const block_q8_K* a_ptr_start = q8_buf.data();

    const int nr = 1;  // M = 1 decode
    const int nc = N;

    for (int64_t y = 0; y < nr; y++) {
        const block_q8_K* a_ptr = a_ptr_start + (y * nb);

#    pragma omp parallel for schedule(static)
        for (int64_t x = 0; x < nc / 8; x++) {
            const block_q4_Kx8* b_ptr = b_ptr_start + (x * b_nb);

            __m256 acc_row = _mm256_setzero_ps();
            __m256 acc_min_rows = _mm256_setzero_ps();

            for (int64_t b = 0; b < nb; b++) {
                const __m256 row_scale_f32 = _mm256_set1_ps(a_ptr[b].d);
                const __m256 col_scale_f32 =
                    _mm256_cvtph_ps(_mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(b_ptr[b].d)), deltamask));
                const __m256 col_dmin_f32 =
                    _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(b_ptr[b].dmin)));

                __m256i iacc_b = _mm256_setzero_si256();
                __m256i iacc_min_b = _mm256_setzero_si256();

                const __m256i q8sums = _mm256_loadu_si256((const __m256i*)(a_ptr[b].bsums));
                __m256i q8s = _mm256_castsi128_si256(
                    _mm_hadd_epi16(_mm256_castsi256_si128(q8sums), _mm256_extracti128_si256(q8sums, 1)));
                q8s = _mm256_permute2f128_si256(q8s, q8s, 0);

                for (int sb = 0; sb < QK / 64; sb++) {
                    const __m256i rhs_raw_vec_0123_0 = _mm256_loadu_si256((const __m256i*)(b_ptr[b].qs + sb * 256));
                    const __m256i rhs_raw_vec_4567_0 = _mm256_loadu_si256((const __m256i*)(b_ptr[b].qs + 32 + sb * 256));
                    const __m256i rhs_raw_vec_0123_1 = _mm256_loadu_si256((const __m256i*)(b_ptr[b].qs + 64 + sb * 256));
                    const __m256i rhs_raw_vec_4567_1 = _mm256_loadu_si256((const __m256i*)(b_ptr[b].qs + 96 + sb * 256));
                    const __m256i rhs_raw_vec_0123_2 = _mm256_loadu_si256((const __m256i*)(b_ptr[b].qs + 128 + sb * 256));
                    const __m256i rhs_raw_vec_4567_2 = _mm256_loadu_si256((const __m256i*)(b_ptr[b].qs + 160 + sb * 256));
                    const __m256i rhs_raw_vec_0123_3 = _mm256_loadu_si256((const __m256i*)(b_ptr[b].qs + 192 + sb * 256));
                    const __m256i rhs_raw_vec_4567_3 = _mm256_loadu_si256((const __m256i*)(b_ptr[b].qs + 224 + sb * 256));

                    // 4-bit -> 8-bit (low nibbles, sub-block sb of the 8 rows)
                    const __m256i rhs_vec_0123_00 = _mm256_and_si256(rhs_raw_vec_0123_0, m4b);
                    const __m256i rhs_vec_4567_00 = _mm256_and_si256(rhs_raw_vec_4567_0, m4b);
                    const __m256i rhs_vec_0123_01 = _mm256_and_si256(rhs_raw_vec_0123_1, m4b);
                    const __m256i rhs_vec_4567_01 = _mm256_and_si256(rhs_raw_vec_4567_1, m4b);
                    const __m256i rhs_vec_0123_02 = _mm256_and_si256(rhs_raw_vec_0123_2, m4b);
                    const __m256i rhs_vec_4567_02 = _mm256_and_si256(rhs_raw_vec_4567_2, m4b);
                    const __m256i rhs_vec_0123_03 = _mm256_and_si256(rhs_raw_vec_0123_3, m4b);
                    const __m256i rhs_vec_4567_03 = _mm256_and_si256(rhs_raw_vec_4567_3, m4b);

                    // 4-bit -> 8-bit (high nibbles, sub-block sb of the 8 rows)
                    const __m256i rhs_vec_0123_10 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_0, 4), m4b);
                    const __m256i rhs_vec_4567_10 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_0, 4), m4b);
                    const __m256i rhs_vec_0123_11 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_1, 4), m4b);
                    const __m256i rhs_vec_4567_11 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_1, 4), m4b);
                    const __m256i rhs_vec_0123_12 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_2, 4), m4b);
                    const __m256i rhs_vec_4567_12 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_2, 4), m4b);
                    const __m256i rhs_vec_0123_13 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_3, 4), m4b);
                    const __m256i rhs_vec_4567_13 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_3, 4), m4b);

                    uint32_t utmp_0[4], utmp_1[4];
                    std::memcpy(utmp_0, b_ptr[b].scales + 24 * sb, 12);
                    utmp_0[3] = ((utmp_0[2] >> 4) & kmask2) | (((utmp_0[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_0 = utmp_0[1] & kmask1;
                    utmp_0[1] = (utmp_0[2] & kmask2) | (((utmp_0[0] >> 6) & kmask3) << 4);
                    utmp_0[2] = uaux_0;
                    utmp_0[0] &= kmask1;

                    std::memcpy(utmp_1, b_ptr[b].scales + 12 + sb * 24, 12);
                    utmp_1[3] = ((utmp_1[2] >> 4) & kmask2) | (((utmp_1[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_1 = utmp_1[1] & kmask1;
                    utmp_1[1] = (utmp_1[2] & kmask2) | (((utmp_1[0] >> 6) & kmask3) << 4);
                    utmp_1[2] = uaux_1;
                    utmp_1[0] &= kmask1;

                    // Scales of first / second sub-block in the sb loop
                    const __m128i mins_and_scales_0 = _mm_set_epi32(utmp_0[3], utmp_0[2], utmp_0[1], utmp_0[0]);
                    const __m128i scales_rearrange_0 = _mm_shuffle_epi8(mins_and_scales_0, scalemask);
                    const __m256i scales_0 = _mm256_cvtepu8_epi16(scales_rearrange_0);

                    const __m128i mins_and_scales_1 = _mm_set_epi32(utmp_1[3], utmp_1[2], utmp_1[1], utmp_1[0]);
                    const __m128i scales_rearrange_1 = _mm_shuffle_epi8(mins_and_scales_1, scalemask);
                    const __m256i scales_1 = _mm256_cvtepu8_epi16(scales_rearrange_1);

                    // Mins of the two sub-blocks arranged side by side
                    const __m256i mins_01 = _mm256_cvtepu8_epi16(
                        _mm_unpacklo_epi8(_mm_shuffle_epi32(mins_and_scales_0, 78),
                                          _mm_shuffle_epi32(mins_and_scales_1, 78)));

                    // Load the two sub-block values corresponding to sb in block_q8_K
                    const __m256i lhs_vec_00 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*)(a_ptr[b].qs + sb * 64)));
                    const __m256i lhs_vec_01 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*)(a_ptr[b].qs + 16 + sb * 64)));
                    const __m256i lhs_vec_10 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*)(a_ptr[b].qs + 32 + sb * 64)));
                    const __m256i lhs_vec_11 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*)(a_ptr[b].qs + 48 + sb * 64)));

                    const __m256i lhs_vec_00b = _mm256_permute2f128_si256(lhs_vec_00, lhs_vec_00, 0);
                    const __m256i lhs_vec_01b = _mm256_permute2f128_si256(lhs_vec_01, lhs_vec_01, 0);
                    const __m256i lhs_vec_10b = _mm256_permute2f128_si256(lhs_vec_10, lhs_vec_10, 0);
                    const __m256i lhs_vec_11b = _mm256_permute2f128_si256(lhs_vec_11, lhs_vec_11, 0);

                    __m256i iacc_0 = _mm256_setzero_si256();
                    __m256i iacc_1 = _mm256_setzero_si256();

                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_00, _mm256_shuffle_epi32(rhs_vec_4567_00, 177), 170), _mm256_shuffle_epi32(lhs_vec_00b, 0)));
                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_00, 177), rhs_vec_4567_00, 170), _mm256_shuffle_epi32(lhs_vec_00b, 85)));
                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_01, _mm256_shuffle_epi32(rhs_vec_4567_01, 177), 170), _mm256_shuffle_epi32(lhs_vec_00b, 170)));
                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_01, 177), rhs_vec_4567_01, 170), _mm256_shuffle_epi32(lhs_vec_00b, 255)));
                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_02, _mm256_shuffle_epi32(rhs_vec_4567_02, 177), 170), _mm256_shuffle_epi32(lhs_vec_01b, 0)));
                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_02, 177), rhs_vec_4567_02, 170), _mm256_shuffle_epi32(lhs_vec_01b, 85)));
                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_03, _mm256_shuffle_epi32(rhs_vec_4567_03, 177), 170), _mm256_shuffle_epi32(lhs_vec_01b, 170)));
                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_03, 177), rhs_vec_4567_03, 170), _mm256_shuffle_epi32(lhs_vec_01b, 255)));
                    iacc_0 = _mm256_madd_epi16(iacc_0, scales_0);

                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_10, _mm256_shuffle_epi32(rhs_vec_4567_10, 177), 170), _mm256_shuffle_epi32(lhs_vec_10b, 0)));
                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_10, 177), rhs_vec_4567_10, 170), _mm256_shuffle_epi32(lhs_vec_10b, 85)));
                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_11, _mm256_shuffle_epi32(rhs_vec_4567_11, 177), 170), _mm256_shuffle_epi32(lhs_vec_10b, 170)));
                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_11, 177), rhs_vec_4567_11, 170), _mm256_shuffle_epi32(lhs_vec_10b, 255)));
                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_12, _mm256_shuffle_epi32(rhs_vec_4567_12, 177), 170), _mm256_shuffle_epi32(lhs_vec_11b, 0)));
                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_12, 177), rhs_vec_4567_12, 170), _mm256_shuffle_epi32(lhs_vec_11b, 85)));
                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_13, _mm256_shuffle_epi32(rhs_vec_4567_13, 177), 170), _mm256_shuffle_epi32(lhs_vec_11b, 170)));
                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_13, 177), rhs_vec_4567_13, 170), _mm256_shuffle_epi32(lhs_vec_11b, 255)));
                    iacc_1 = _mm256_madd_epi16(iacc_1, scales_1);

                    const __m256i iacc_sb = _mm256_add_epi32(iacc_0, iacc_1);

                    const __m256i q8s_sb = _mm256_shuffle_epi32(q8s, 0);
                    const __m256i iacc_min_sb = _mm256_madd_epi16(q8s_sb, mins_01);
                    q8s = _mm256_bsrli_epi128(q8s, 4);

                    iacc_b = _mm256_add_epi32(iacc_b, iacc_sb);
                    iacc_min_b = _mm256_add_epi32(iacc_min_b, iacc_min_sb);
                }

                acc_row = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_b), _mm256_mul_ps(col_scale_f32, row_scale_f32), acc_row);
                acc_min_rows = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_min_b), _mm256_mul_ps(col_dmin_f32, row_scale_f32), acc_min_rows);
            }

            acc_row = _mm256_permutevar8x32_ps(acc_row, finalpermutemask);
            _mm256_storeu_ps(o_data + (y * nr + x * 8), _mm256_sub_ps(acc_row, acc_min_rows));
        }
    }
}

// ============================================================================
// Q6_K 8x8 Decode GEMV (M=1) — self-written AVX2 port of llama.cpp ARM
// ggml_gemv_q6_K_8x8_q8_K (arch/arm/repack.cpp:1498) + generic reference
// (repack.cpp:357), MIT. llama.cpp ships no x86 AVX2 Q6_K 8x8, so this is
// written from scratch here using the proven dot_q6_K_q8_K_avx2 idioms.
// ============================================================================
// Processes 8 output rows at a time from the interleaved block_q6_Kx8 layout.
// A single shared activation quantization + shared q8 loads across 8 rows, with
// the -32 offset handled via bsums (bias trick) instead of per-element signed
// reconstruction, exactly like the ARM kernel.

static void gemv_q6_K_8x8_q8_K_avx2(
    const float* a_data, const uint8_t* w_repacked, float* o_data, int K, int N)
{
    constexpr int QK = 256;
    constexpr int ncols_interleaved = 8;
    const int nb = K / QK;
    if (nb <= 0 || N % ncols_interleaved != 0)
        return;

    scratch_vec<block_q8_K> q8_buf(nb);
    quantize_row_q8_K(a_data, q8_buf.data(), K);

    const __m256i m15 = _mm256_set1_epi8(0x0F);
    const __m256i m3  = _mm256_set1_epi8(0x03);
    const __m256i m30 = _mm256_set1_epi8(0x30);
    const __m256i m3f = _mm256_set1_epi8(0x3F);

    // Byte shuffle masks: replicate int16 lanes of the widened scale vector.
    // scales are column-major: bytes scales[i*8..i*8+8) = scale_i for rows 0-7.
    // int16 lane i occupies bytes 2i, 2i+1. _mm256_shuffle_epi8 indexes each
    // 128-bit half independently (0-15), so sc_l16 is broadcast (low 128 ->
    // full 256) first; the two halves use indices 0..7 relative to the same data.
    // SCALE_REP_LO -> [s0 x4, s1 x4, s2 x4, s3 x4] (rows 0-3), SCALE_REP_HI -> rows 4-7.
    // Each 128-bit half is indexed independently by _mm256_shuffle_epi8, so the
    // second half must use indices 4..7 (lo) / 12..15 (hi), not 0..3 / 8..11.
    const __m256i scale_rep_lo = _mm256_setr_epi8(
        0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3,
        4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7);
    const __m256i scale_rep_hi = _mm256_setr_epi8(
        8, 9, 8, 9, 8, 9, 8, 9, 10, 11, 10, 11, 10, 11, 10, 11,
        12, 13, 12, 13, 12, 13, 12, 13, 14, 15, 14, 15, 14, 15, 14, 15);

    const block_q6_Kx8* b_ptr_start = reinterpret_cast<const block_q6_Kx8*>(w_repacked);
    const block_q8_K*   a_ptr       = q8_buf.data();

// Single work-stealing pass over column groups via the persistent pool
    // (replaces an OMP static fork/join; chunk = 4 groups = 32 rows).
    auto process_col_group = [&](int64_t x) {
        const block_q6_Kx8* b_ptr = b_ptr_start + (x * nb);

        __m256 acc_row = _mm256_setzero_ps();

        for (int b = 0; b < nb; b++) {
            const block_q8_K* q8 = a_ptr + b;

            const __m256 row_scale_f32 = _mm256_mul_ps(
                _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(b_ptr[b].d))),
                _mm256_set1_ps(q8->d));

            // Bias per column (the -32 offset): bias[col] = sum_i scales[i*8+col]*bsum[i], <<5.
            // scales are column-major, so each scale index contributes one 8-byte row slice.
            // _mm256_cvtepi8_epi32 sign-extends only the low 4 bytes, so rows 0-3 and 4-7
            // are widened separately into bias_lo / bias_hi.
            alignas(16) int16_t bsums16[16];
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*)q8->bsums);
            _mm256_storeu_si256((__m256i*)bsums16, q8sums);
            __m256i bias_lo = _mm256_setzero_si256();  // rows 0-3
            __m256i bias_hi = _mm256_setzero_si256();  // rows 4-7
            for (int i = 0; i < 16; i++) {
                const __m128i s8 = _mm_loadl_epi64((const __m128i*)(b_ptr[b].scales + i * 8));
                const __m256i s32_lo = _mm256_cvtepi8_epi32(s8);
                const __m256i s32_hi = _mm256_cvtepi8_epi32(_mm_srli_si128(s8, 4));
                const __m256i bs = _mm256_set1_epi32(bsums16[i]);
                bias_lo = _mm256_add_epi32(bias_lo, _mm256_mullo_epi32(s32_lo, bs));
                bias_hi = _mm256_add_epi32(bias_hi, _mm256_mullo_epi32(s32_hi, bs));
            }
            bias_lo = _mm256_slli_epi32(bias_lo, 5);
            bias_hi = _mm256_slli_epi32(bias_hi, 5);

            __m256i iacc_lo = _mm256_setzero_si256();  // cols 0-3 (2 int32 lanes each)
            __m256i iacc_hi = _mm256_setzero_si256();  // cols 4-7

            for (int half = 0; half < 2; half++) {
                const uint8_t* ql_base = b_ptr[b].ql + half * 512;
                const uint8_t* qh_base = b_ptr[b].qh + half * 256;
                const int8_t*  qs_base = q8->qs + half * 128;

                for (int sb = 0; sb < QK / 64; sb++) {
                    const int ql_off = sb * 128;         // 0,128,256,384
                    const int qh_off = ql_off & 255;     // wraps after 256 bytes

                    // q8 sub-block values (16 low + 16 high, shared across 8 rows)
                    const int8_t* q8_l0 = qs_base + sb * 16;
                    const int8_t* q8_h0 = q8_l0 + 64;

                    // Broadcast q8 sub-block to 32 bytes: [8B][8B][8B][8B].
                    // 0x44 duplicates both 32-bit dwords (8 distinct bytes).
                    const __m128i q8l_16 = _mm_shuffle_epi32(
                        _mm_loadl_epi64((const __m128i*)q8_l0), 0x44);
                    const __m128i q8h_16 = _mm_shuffle_epi32(
                        _mm_loadl_epi64((const __m128i*)q8_h0), 0x44);
                    const __m256i q8l_256 = _mm256_broadcastsi128_si256(q8l_16);
                    const __m256i q8h_256 = _mm256_broadcastsi128_si256(q8h_16);
                    const __m128i q8l2_16 = _mm_shuffle_epi32(
                        _mm_loadl_epi64((const __m128i*)(q8_l0 + 8)), 0x44);
                    const __m128i q8h2_16 = _mm_shuffle_epi32(
                        _mm_loadl_epi64((const __m128i*)(q8_h0 + 8)), 0x44);
                    const __m256i q8l2_256 = _mm256_broadcastsi128_si256(q8l2_16);
                    const __m256i q8h2_256 = _mm256_broadcastsi128_si256(q8h2_16);

                    // Scales of this sub-block, widened int8 -> int16, then split
                    // into rows 0-3 and 4-7 replication vectors.
                    const int scale_idx_l = half * 8 + sb;
                    const int scale_idx_h = scale_idx_l + 4;
                    const __m256i sc_l16 = _mm256_cvtepi8_epi16(_mm_loadl_epi64(
                        (const __m128i*)(b_ptr[b].scales + scale_idx_l * 8)));
                    const __m256i sc_h16 = _mm256_cvtepi8_epi16(_mm_loadl_epi64(
                        (const __m128i*)(b_ptr[b].scales + scale_idx_h * 8)));
                    // Duplicate the 8 int16 lanes to both 128-bit halves so the
                    // per-half shuffle masks above resolve s0..s7 everywhere.
                    const __m256i sc_l_full = _mm256_broadcastsi128_si256(
                        _mm256_castsi256_si128(sc_l16));
                    const __m256i sc_h_full = _mm256_broadcastsi128_si256(
                        _mm256_castsi256_si128(sc_h16));
                    const __m256i scales_l_lo = _mm256_shuffle_epi8(sc_l_full, scale_rep_lo);
                    const __m256i scales_l_hi = _mm256_shuffle_epi8(sc_l_full, scale_rep_hi);
                    const __m256i scales_h_lo = _mm256_shuffle_epi8(sc_h_full, scale_rep_lo);
                    const __m256i scales_h_hi = _mm256_shuffle_epi8(sc_h_full, scale_rep_hi);

                    // ql/qh 32-byte blocks for the current (half, sb) group.
                    const __m256i ql_lo_0 = _mm256_loadu_si256((const __m256i*)(ql_base + ql_off));
                    const __m256i ql_lo_1 = _mm256_loadu_si256((const __m256i*)(ql_base + ql_off + 32));
                    const __m256i ql_hi_0 = _mm256_loadu_si256((const __m256i*)(ql_base + ql_off + 64));
                    const __m256i ql_hi_1 = _mm256_loadu_si256((const __m256i*)(ql_base + ql_off + 96));

                    __m256i qh_lo_0 = _mm256_loadu_si256((const __m256i*)(qh_base + qh_off));
                    __m256i qh_lo_1 = _mm256_loadu_si256((const __m256i*)(qh_base + qh_off + 32));
                    __m256i qh_hi_0 = _mm256_loadu_si256((const __m256i*)(qh_base + qh_off + 64));
                    __m256i qh_hi_1 = _mm256_loadu_si256((const __m256i*)(qh_base + qh_off + 96));
                    if (sb > 1) {  // shift qh right by 2 for sub-blocks 2,3
                        qh_lo_0 = _mm256_and_si256(_mm256_srli_epi16(qh_lo_0, 2), m3f);
                        qh_lo_1 = _mm256_and_si256(_mm256_srli_epi16(qh_lo_1, 2), m3f);
                        qh_hi_0 = _mm256_and_si256(_mm256_srli_epi16(qh_hi_0, 2), m3f);
                        qh_hi_1 = _mm256_and_si256(_mm256_srli_epi16(qh_hi_1, 2), m3f);
                    }

                    // Reconstruct q6 (0..63) without the -32 (handled via bsums):
                    //   low part:  (ql & 0x0F) | ((qh & 0x03) << 4)
                    //   high part: (ql >> 4) | (qh & 0x30)
                    const __m256i q6_l0_0 = _mm256_or_si256(
                        _mm256_and_si256(ql_lo_0, m15),
                        _mm256_slli_epi16(_mm256_and_si256(qh_lo_0, m3), 4));
                    const __m256i q6_l0_1 = _mm256_or_si256(
                        _mm256_and_si256(ql_lo_1, m15),
                        _mm256_slli_epi16(_mm256_and_si256(qh_lo_1, m3), 4));
                    const __m256i q6_l1_0 = _mm256_or_si256(
                        _mm256_and_si256(ql_hi_0, m15),
                        _mm256_slli_epi16(_mm256_and_si256(qh_hi_0, m3), 4));
                    const __m256i q6_l1_1 = _mm256_or_si256(
                        _mm256_and_si256(ql_hi_1, m15),
                        _mm256_slli_epi16(_mm256_and_si256(qh_hi_1, m3), 4));
                    const __m256i q6_h0_0 = _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16(ql_lo_0, 4), m15),
                        _mm256_and_si256(qh_lo_0, m30));
                    const __m256i q6_h0_1 = _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16(ql_lo_1, 4), m15),
                        _mm256_and_si256(qh_lo_1, m30));
                    const __m256i q6_h1_0 = _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16(ql_hi_0, 4), m15),
                        _mm256_and_si256(qh_hi_0, m30));
                    const __m256i q6_h1_1 = _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16(ql_hi_1, 4), m15),
                        _mm256_and_si256(qh_hi_1, m30));

                    // Rows 0-3: q6_l0_0/q6_l1_0 are rows 0-3 (elements 16*sb..+7
                    // and +8..+15), q6_h0_0/q6_h1_0 their +64 high counterparts.
                    iacc_lo = _mm256_add_epi32(iacc_lo,
                        _mm256_madd_epi16(_mm256_maddubs_epi16(q6_l0_0, q8l_256), scales_l_lo));
                    iacc_lo = _mm256_add_epi32(iacc_lo,
                        _mm256_madd_epi16(_mm256_maddubs_epi16(q6_l1_0, q8l2_256), scales_l_lo));
                    iacc_lo = _mm256_add_epi32(iacc_lo,
                        _mm256_madd_epi16(_mm256_maddubs_epi16(q6_h0_0, q8h_256), scales_h_lo));
                    iacc_lo = _mm256_add_epi32(iacc_lo,
                        _mm256_madd_epi16(_mm256_maddubs_epi16(q6_h1_0, q8h2_256), scales_h_lo));

                    // Rows 4-7
                    iacc_hi = _mm256_add_epi32(iacc_hi,
                        _mm256_madd_epi16(_mm256_maddubs_epi16(q6_l0_1, q8l_256), scales_l_hi));
                    iacc_hi = _mm256_add_epi32(iacc_hi,
                        _mm256_madd_epi16(_mm256_maddubs_epi16(q6_l1_1, q8l2_256), scales_l_hi));
                    iacc_hi = _mm256_add_epi32(iacc_hi,
                        _mm256_madd_epi16(_mm256_maddubs_epi16(q6_h0_1, q8h_256), scales_h_hi));
                    iacc_hi = _mm256_add_epi32(iacc_hi,
                        _mm256_madd_epi16(_mm256_maddubs_epi16(q6_h1_1, q8h2_256), scales_h_hi));
                }
            }

            // Each col's total = 2 adjacent int32 lanes, minus bias, times scale.
            // hadd produces [c0, c1, c0, c1, c2, c3, c2, c3]; permute4x64_epi64 0xD8
            // reorders the 64-bit pairs so cols 0-3 sit in lanes 0-3.
            const __m256i s_lo = _mm256_permute4x64_epi64(
                _mm256_hadd_epi32(iacc_lo, iacc_lo), 0xD8);  // lanes 0-3 = cols 0-3
            const __m256i s_hi = _mm256_permute4x64_epi64(
                _mm256_hadd_epi32(iacc_hi, iacc_hi), 0xD8);  // lanes 0-3 = cols 4-7
            const __m256i col_lo = _mm256_sub_epi32(s_lo, bias_lo);
            const __m256i col_hi = _mm256_sub_epi32(s_hi, bias_hi);

            const __m256 out_lo = _mm256_cvtepi32_ps(col_lo);
            const __m256 out_hi = _mm256_cvtepi32_ps(col_hi);
            // cols 0-3 then cols 4-7 -> [c0..c7]
            const __m256 res = _mm256_setr_m128(
                _mm256_castps256_ps128(out_lo), _mm256_castps256_ps128(out_hi));

            acc_row = _mm256_fmadd_ps(res, row_scale_f32, acc_row);
        }

        _mm256_storeu_ps(o_data + x * ncols_interleaved, acc_row);
    };

    const int64_t num_col_groups = N / ncols_interleaved;
    parallel_for_chunks((int)num_col_groups, 4, [&](int64_t x0, int64_t x1) {
        for (int64_t x = x0; x < x1; ++x) {
            process_col_group(x);
        }
    });
}

// ============================================================================
// Q4_0 Repack-aware Decode GEMM (M=1, RM=4, F16C)
// ============================================================================
// Uses repacked weight layout where 4 rows' same-block data is contiguous.
// This eliminates cache-line conflicts in the inner loop.

static void gemm_q4_0_decode_repacked_f16c_avx2(
    const float* a_data,
    const uint8_t* w_repacked,
    float* o_data,
    int64_t K,
    int64_t N)
{
    constexpr int RM = 4;
    constexpr int BLOCK_SIZE = 32;
    constexpr int BLOCK_BYTES = 18;
    const int64_t nb = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int64_t num_groups = (N + RM - 1) / RM;

    // Quantize activation once
    scratch_vec<block_q8_0_act> q8_act(nb);
    quantize_row_q8_0_act(a_data, q8_act.data(), (int)K);

    #pragma omp parallel for schedule(static)
    for (int64_t g = 0; g < num_groups; ++g) {
        int64_t n0 = g * RM;
        int64_t rows = std::min(n0 + RM, N) - n0;

        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        // Base pointer for this tile group: blocks are contiguous
        const uint8_t* group_base = w_repacked + (size_t)g * nb * RM * BLOCK_BYTES;

        for (int64_t l = 0; l < nb; ++l) {
            // Prefetch next block (contiguous in repacked layout!)
            if (l + 4 < nb) {
                _mm_prefetch((const char*)(group_base + (l + 4) * RM * BLOCK_BYTES), _MM_HINT_T0);
            }

            // All 4 weight blocks are contiguous — single 72-byte region
            const uint8_t* block_base = group_base + l * RM * BLOCK_BYTES;

            // Load weight blocks (now contiguous in memory)
            __m256i wvec0 = BlockLoader<block_q4_0_tag>::load(block_base + 0 * BLOCK_BYTES);
            __m256i wvec1 = (rows > 1) ? BlockLoader<block_q4_0_tag>::load(block_base + 1 * BLOCK_BYTES) : _mm256_setzero_si256();
            __m256i wvec2 = (rows > 2) ? BlockLoader<block_q4_0_tag>::load(block_base + 2 * BLOCK_BYTES) : _mm256_setzero_si256();
            __m256i wvec3 = (rows > 3) ? BlockLoader<block_q4_0_tag>::load(block_base + 3 * BLOCK_BYTES) : _mm256_setzero_si256();

            // Load activation block
            __m256i avec = BlockLoader<block_q8_0_act>::load(&q8_act[l]);

            // F16C: pack 4 fp16 weight scales — they're contiguous too!
            // scales at block_base[0..1], block_base[18..19], block_base[36..37], block_base[54..55]
            uint64_t packed_scales = 0;
            uint16_t d0, d1, d2, d3;
            memcpy(&d0, block_base + 0 * BLOCK_BYTES, 2);
            memcpy(&d1, block_base + 1 * BLOCK_BYTES, 2);
            memcpy(&d2, block_base + 2 * BLOCK_BYTES, 2);
            memcpy(&d3, block_base + 3 * BLOCK_BYTES, 2);
            packed_scales = (uint64_t)d0 | ((uint64_t)d1 << 16) | ((uint64_t)d2 << 32) | ((uint64_t)d3 << 48);

            __m128 sw_f16 = _mm_cvtph_ps(_mm_set_epi64x(0, packed_scales));
            float act_scale = q8_act[l].d;
            __m128 sw_all = _mm_mul_ps(sw_f16, _mm_set1_ps(act_scale));

            // Broadcast each combined scale to __m256
            __m256 sc0 = _mm256_castps128_ps256(_mm_shuffle_ps(sw_all, sw_all, 0x00));
            sc0 = _mm256_permute2f128_ps(sc0, sc0, 0x00);
            __m256 sc1 = _mm256_castps128_ps256(_mm_shuffle_ps(sw_all, sw_all, 0x55));
            sc1 = _mm256_permute2f128_ps(sc1, sc1, 0x00);
            __m256 sc2 = _mm256_castps128_ps256(_mm_shuffle_ps(sw_all, sw_all, 0xAA));
            sc2 = _mm256_permute2f128_ps(sc2, sc2, 0x00);
            __m256 sc3 = _mm256_castps128_ps256(_mm_shuffle_ps(sw_all, sw_all, 0xFF));
            sc3 = _mm256_permute2f128_ps(sc3, sc3, 0x00);

            // sign_epi8 + updot for each row
            __m256i sa0 = _mm256_sign_epi8(wvec0, wvec0);
            __m256i sa1 = (rows > 1) ? _mm256_sign_epi8(wvec1, wvec1) : _mm256_setzero_si256();
            __m256i sa2 = (rows > 2) ? _mm256_sign_epi8(wvec2, wvec2) : _mm256_setzero_si256();
            __m256i sa3 = (rows > 3) ? _mm256_sign_epi8(wvec3, wvec3) : _mm256_setzero_si256();

            __m256i sb0 = _mm256_sign_epi8(avec, wvec0);
            __m256i sb1 = (rows > 1) ? _mm256_sign_epi8(avec, wvec1) : _mm256_setzero_si256();
            __m256i sb2 = (rows > 2) ? _mm256_sign_epi8(avec, wvec2) : _mm256_setzero_si256();
            __m256i sb3 = (rows > 3) ? _mm256_sign_epi8(avec, wvec3) : _mm256_setzero_si256();

            acc0 = _mm256_fmadd_ps(sc0, updot_avx2(sa0, sb0), acc0);
            acc1 = _mm256_fmadd_ps(sc1, updot_avx2(sa1, sb1), acc1);
            acc2 = _mm256_fmadd_ps(sc2, updot_avx2(sa2, sb2), acc2);
            acc3 = _mm256_fmadd_ps(sc3, updot_avx2(sa3, sb3), acc3);
        }

        o_data[n0 + 0] = vdot::hsum_ps256(acc0);
        if (rows > 1) o_data[n0 + 1] = vdot::hsum_ps256(acc1);
        if (rows > 2) o_data[n0 + 2] = vdot::hsum_ps256(acc2);
        if (rows > 3) o_data[n0 + 3] = vdot::hsum_ps256(acc3);
    }
}

// ============================================================================
// Q4_0 Prefill GEMM Tile: RM=4 activation rows × RN=4 weight cols (F16C)
// ============================================================================
// Processes 4 activation rows × 4 weight columns per tile.
// Uses F16C to batch-convert 4 Q4_0 weight fp16 scales per block.
// Weight blocks and F16C scale conversion are done once per block,
// shared across all activation rows in the tile.
//
// Pattern: gemmMx4 (from llama.cpp sgemm.cpp)
// - F16C: pack 4 weight fp16 scales → _mm_cvtph_ps → 4 fp32
// - Per activation row: broadcast act_scale, multiply with 4 weight scales
// - Shuffle each product to __m256 for FMADD
// - sign_epi8 + updot for each of 4×4 = 16 output elements

static inline void gemm_q4_0_tile_4x4_f16c(
    const block_q8_0_act* q8_act,    // [M * nb] pre-quantized activation
    const uint8_t* w_data,           // Q4_0 weight base pointer
    float* o_data,                    // [M, N] output
    int64_t nb,                       // number of blocks per row
    int64_t m0, int64_t m_rows,      // activation rows [m0, m0+m_rows), ≤ 4
    int64_t n0, int64_t n_cols,      // weight cols [n0, n0+n_cols), ≤ 4
    int64_t N)                        // output row stride
{
    constexpr int BLOCK_BYTES = 18;

    // Accumulators: Cv[j][i] for weight col j, activation row i
    __m256 Cv0_0 = _mm256_setzero_ps(), Cv0_1 = _mm256_setzero_ps(),
           Cv0_2 = _mm256_setzero_ps(), Cv0_3 = _mm256_setzero_ps();
    __m256 Cv1_0 = _mm256_setzero_ps(), Cv1_1 = _mm256_setzero_ps(),
           Cv1_2 = _mm256_setzero_ps(), Cv1_3 = _mm256_setzero_ps();
    __m256 Cv2_0 = _mm256_setzero_ps(), Cv2_1 = _mm256_setzero_ps(),
           Cv2_2 = _mm256_setzero_ps(), Cv2_3 = _mm256_setzero_ps();
    __m256 Cv3_0 = _mm256_setzero_ps(), Cv3_1 = _mm256_setzero_ps(),
           Cv3_2 = _mm256_setzero_ps(), Cv3_3 = _mm256_setzero_ps();

    // Weight row base pointers
    const uint8_t* w0 = w_data + (size_t)(n0 + 0) * nb * BLOCK_BYTES;
    const uint8_t* w1 = (n_cols > 1) ? w_data + (size_t)(n0 + 1) * nb * BLOCK_BYTES : nullptr;
    const uint8_t* w2 = (n_cols > 2) ? w_data + (size_t)(n0 + 2) * nb * BLOCK_BYTES : nullptr;
    const uint8_t* w3 = (n_cols > 3) ? w_data + (size_t)(n0 + 3) * nb * BLOCK_BYTES : nullptr;

    for (int64_t l = 0; l < nb; ++l) {
        // Prefetch next weight blocks
        if (l + 4 < nb) {
            _mm_prefetch((const char*)(w0 + (l + 4) * BLOCK_BYTES), _MM_HINT_T0);
            if (n_cols > 1) _mm_prefetch((const char*)(w1 + (l + 4) * BLOCK_BYTES), _MM_HINT_T0);
            if (n_cols > 2) _mm_prefetch((const char*)(w2 + (l + 4) * BLOCK_BYTES), _MM_HINT_T0);
            if (n_cols > 3) _mm_prefetch((const char*)(w3 + (l + 4) * BLOCK_BYTES), _MM_HINT_T0);
        }

        // Load 4 weight blocks (Q4_0 → 32 signed int8) — shared across act rows
        __m256i wvec0 = BlockLoader<block_q4_0_tag>::load(w0 + l * BLOCK_BYTES);
        __m256i wvec1 = (n_cols > 1) ? BlockLoader<block_q4_0_tag>::load(w1 + l * BLOCK_BYTES) : _mm256_setzero_si256();
        __m256i wvec2 = (n_cols > 2) ? BlockLoader<block_q4_0_tag>::load(w2 + l * BLOCK_BYTES) : _mm256_setzero_si256();
        __m256i wvec3 = (n_cols > 3) ? BlockLoader<block_q4_0_tag>::load(w3 + l * BLOCK_BYTES) : _mm256_setzero_si256();

        // F16C: pack 4 fp16 weight scales, convert to 4 fp32 at once
        uint64_t packed_scales = 0;
        uint16_t d0, d1 = 0, d2 = 0, d3 = 0;
        memcpy(&d0, w0 + l * BLOCK_BYTES, 2);
        if (n_cols > 1) memcpy(&d1, w1 + l * BLOCK_BYTES, 2);
        if (n_cols > 2) memcpy(&d2, w2 + l * BLOCK_BYTES, 2);
        if (n_cols > 3) memcpy(&d3, w3 + l * BLOCK_BYTES, 2);
        packed_scales = (uint64_t)d0 | ((uint64_t)d1 << 16) | ((uint64_t)d2 << 32) | ((uint64_t)d3 << 48);
        __m128 db = _mm_cvtph_ps(_mm_set_epi64x(0, packed_scales));
        // db = [d0_fp32, d1_fp32, d2_fp32, d3_fp32]

        // --- Activation row 0 ---
        if (m_rows > 0) {
            const block_q8_0_act* a0 = q8_act + (m0 + 0) * nb;
            __m256i avec0 = BlockLoader<block_q8_0_act>::load(a0 + l);
            __m256 dvec = _mm256_castps128_ps256(_mm_mul_ps(_mm_set1_ps(a0[l].d), db));
            dvec = _mm256_permute2f128_ps(dvec, dvec, 0x00);
            __m256i sa = _mm256_sign_epi8(avec0, avec0);

            Cv0_0 = _mm256_fmadd_ps(_mm256_shuffle_ps(dvec, dvec, 0x00),
                         updot_avx2(sa, _mm256_sign_epi8(wvec0, avec0)), Cv0_0);
            if (n_cols > 1)
            Cv1_0 = _mm256_fmadd_ps(_mm256_shuffle_ps(dvec, dvec, 0x55),
                         updot_avx2(sa, _mm256_sign_epi8(wvec1, avec0)), Cv1_0);
            if (n_cols > 2)
            Cv2_0 = _mm256_fmadd_ps(_mm256_shuffle_ps(dvec, dvec, 0xAA),
                         updot_avx2(sa, _mm256_sign_epi8(wvec2, avec0)), Cv2_0);
            if (n_cols > 3)
            Cv3_0 = _mm256_fmadd_ps(_mm256_shuffle_ps(dvec, dvec, 0xFF),
                         updot_avx2(sa, _mm256_sign_epi8(wvec3, avec0)), Cv3_0);
        }

        // --- Activation row 1 ---
        if (m_rows > 1) {
            const block_q8_0_act* a1 = q8_act + (m0 + 1) * nb;
            __m256i avec1 = BlockLoader<block_q8_0_act>::load(a1 + l);
            __m256 dvec = _mm256_castps128_ps256(_mm_mul_ps(_mm_set1_ps(a1[l].d), db));
            dvec = _mm256_permute2f128_ps(dvec, dvec, 0x00);
            __m256i sa = _mm256_sign_epi8(avec1, avec1);

            Cv0_1 = _mm256_fmadd_ps(_mm256_shuffle_ps(dvec, dvec, 0x00),
                         updot_avx2(sa, _mm256_sign_epi8(wvec0, avec1)), Cv0_1);
            if (n_cols > 1)
            Cv1_1 = _mm256_fmadd_ps(_mm256_shuffle_ps(dvec, dvec, 0x55),
                         updot_avx2(sa, _mm256_sign_epi8(wvec1, avec1)), Cv1_1);
            if (n_cols > 2)
            Cv2_1 = _mm256_fmadd_ps(_mm256_shuffle_ps(dvec, dvec, 0xAA),
                         updot_avx2(sa, _mm256_sign_epi8(wvec2, avec1)), Cv2_1);
            if (n_cols > 3)
            Cv3_1 = _mm256_fmadd_ps(_mm256_shuffle_ps(dvec, dvec, 0xFF),
                         updot_avx2(sa, _mm256_sign_epi8(wvec3, avec1)), Cv3_1);
        }

        // --- Activation row 2 ---
        if (m_rows > 2) {
            const block_q8_0_act* a2 = q8_act + (m0 + 2) * nb;
            __m256i avec2 = BlockLoader<block_q8_0_act>::load(a2 + l);
            __m256 dvec = _mm256_castps128_ps256(_mm_mul_ps(_mm_set1_ps(a2[l].d), db));
            dvec = _mm256_permute2f128_ps(dvec, dvec, 0x00);
            __m256i sa = _mm256_sign_epi8(avec2, avec2);

            Cv0_2 = _mm256_fmadd_ps(_mm256_shuffle_ps(dvec, dvec, 0x00),
                         updot_avx2(sa, _mm256_sign_epi8(wvec0, avec2)), Cv0_2);
            if (n_cols > 1)
            Cv1_2 = _mm256_fmadd_ps(_mm256_shuffle_ps(dvec, dvec, 0x55),
                         updot_avx2(sa, _mm256_sign_epi8(wvec1, avec2)), Cv1_2);
            if (n_cols > 2)
            Cv2_2 = _mm256_fmadd_ps(_mm256_shuffle_ps(dvec, dvec, 0xAA),
                         updot_avx2(sa, _mm256_sign_epi8(wvec2, avec2)), Cv2_2);
            if (n_cols > 3)
            Cv3_2 = _mm256_fmadd_ps(_mm256_shuffle_ps(dvec, dvec, 0xFF),
                         updot_avx2(sa, _mm256_sign_epi8(wvec3, avec2)), Cv3_2);
        }

        // --- Activation row 3 ---
        if (m_rows > 3) {
            const block_q8_0_act* a3 = q8_act + (m0 + 3) * nb;
            __m256i avec3 = BlockLoader<block_q8_0_act>::load(a3 + l);
            __m256 dvec = _mm256_castps128_ps256(_mm_mul_ps(_mm_set1_ps(a3[l].d), db));
            dvec = _mm256_permute2f128_ps(dvec, dvec, 0x00);
            __m256i sa = _mm256_sign_epi8(avec3, avec3);

            Cv0_3 = _mm256_fmadd_ps(_mm256_shuffle_ps(dvec, dvec, 0x00),
                         updot_avx2(sa, _mm256_sign_epi8(wvec0, avec3)), Cv0_3);
            if (n_cols > 1)
            Cv1_3 = _mm256_fmadd_ps(_mm256_shuffle_ps(dvec, dvec, 0x55),
                         updot_avx2(sa, _mm256_sign_epi8(wvec1, avec3)), Cv1_3);
            if (n_cols > 2)
            Cv2_3 = _mm256_fmadd_ps(_mm256_shuffle_ps(dvec, dvec, 0xAA),
                         updot_avx2(sa, _mm256_sign_epi8(wvec2, avec3)), Cv2_3);
            if (n_cols > 3)
            Cv3_3 = _mm256_fmadd_ps(_mm256_shuffle_ps(dvec, dvec, 0xFF),
                         updot_avx2(sa, _mm256_sign_epi8(wvec3, avec3)), Cv3_3);
        }
    }

    // Horizontal sum and write outputs
    if (m_rows > 0) {
        o_data[(m0 + 0) * N + n0 + 0] = vdot::hsum_ps256(Cv0_0);
        if (n_cols > 1) o_data[(m0 + 0) * N + n0 + 1] = vdot::hsum_ps256(Cv1_0);
        if (n_cols > 2) o_data[(m0 + 0) * N + n0 + 2] = vdot::hsum_ps256(Cv2_0);
        if (n_cols > 3) o_data[(m0 + 0) * N + n0 + 3] = vdot::hsum_ps256(Cv3_0);
    }
    if (m_rows > 1) {
        o_data[(m0 + 1) * N + n0 + 0] = vdot::hsum_ps256(Cv0_1);
        if (n_cols > 1) o_data[(m0 + 1) * N + n0 + 1] = vdot::hsum_ps256(Cv1_1);
        if (n_cols > 2) o_data[(m0 + 1) * N + n0 + 2] = vdot::hsum_ps256(Cv2_1);
        if (n_cols > 3) o_data[(m0 + 1) * N + n0 + 3] = vdot::hsum_ps256(Cv3_1);
    }
    if (m_rows > 2) {
        o_data[(m0 + 2) * N + n0 + 0] = vdot::hsum_ps256(Cv0_2);
        if (n_cols > 1) o_data[(m0 + 2) * N + n0 + 1] = vdot::hsum_ps256(Cv1_2);
        if (n_cols > 2) o_data[(m0 + 2) * N + n0 + 2] = vdot::hsum_ps256(Cv2_2);
        if (n_cols > 3) o_data[(m0 + 2) * N + n0 + 3] = vdot::hsum_ps256(Cv3_2);
    }
    if (m_rows > 3) {
        o_data[(m0 + 3) * N + n0 + 0] = vdot::hsum_ps256(Cv0_3);
        if (n_cols > 1) o_data[(m0 + 3) * N + n0 + 1] = vdot::hsum_ps256(Cv1_3);
        if (n_cols > 2) o_data[(m0 + 3) * N + n0 + 2] = vdot::hsum_ps256(Cv2_3);
        if (n_cols > 3) o_data[(m0 + 3) * N + n0 + 3] = vdot::hsum_ps256(Cv3_3);
    }
}

// ============================================================================
// Q4_0 Prefill GEMM: RM=4, RN=4 with F16C (AVX2)
// ============================================================================
// Computes C = A @ B^T where A is [M, K] FP32 and B is [N, K] Q4_0.
// All M activation rows are quantized to Q8_0_act once.
// Tiles of 4 activation rows × 4 weight columns use F16C for batch
// fp16→fp32 scale conversion and work-stealing for load balance.
//
// Key optimizations over the previous RN=2 version:
// - RN=4: F16C packs 4 weight fp16 scales per block (was 2 + scalar)
// - Each tile produces 4×4=16 outputs (was 4×2=8), better amortizing tile overhead
// - Weight blocks loaded once per tile, shared across 4 activation rows
// - Fully unrolled inner loops for max ILP

static void gemm_q4_0_batch_avx2(
    const float* a_data,     // [M, K] FP32 activation
    const uint8_t* w_data,   // [N, nb*18] Q4_0 weight
    float* o_data,           // [M, N] FP32 output
    int64_t M, int64_t K, int64_t N)
{
    constexpr int RM = 4;
    constexpr int RN = 4;
    constexpr int BLOCK_SIZE = 32;
    constexpr int BLOCK_BYTES = 18;
    const int64_t nb = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Quantize all M activation rows to Q8_0_act (one-time cost)
    scratch_vec<block_q8_0_act> q8_act(M * nb);
    for (int64_t m = 0; m < M; ++m)
        quantize_row_q8_0_act(a_data + m * K, q8_act.data() + m * nb, (int)K);

    // Tile-level work-stealing
    const int64_t m_tiles = (M + RM - 1) / RM;
    const int64_t n_tiles = (N + RN - 1) / RN;
    const int64_t total_tiles = m_tiles * n_tiles;

    std::atomic<int64_t> next_tile{0};

    #pragma omp parallel
    {
        while (true) {
            int64_t tile = next_tile.fetch_add(1, std::memory_order_relaxed);
            if (tile >= total_tiles) break;

            int64_t mt = tile / n_tiles;
            int64_t nt = tile % n_tiles;

            int64_t m0 = mt * RM;
            int64_t n0 = nt * RN;
            int64_t m_rows = std::min(m0 + RM, M) - m0;
            int64_t n_cols = std::min(n0 + RN, N) - n0;

            gemm_q4_0_tile_4x4_f16c(q8_act.data(), w_data, o_data, nb,
                                      m0, m_rows, n0, n_cols, N);
        }
    }
}

// ============================================================================
// Q6_K GEMM: Unified decode (M=1) and prefill (M>1)
// ============================================================================
// Quantizes all M activation rows to Q8_K once.
// For each weight row, shares Q6_K decoding (scales, ql/qh) across all M rows.
// Decode: groups RM=4 weight rows per thread for Q8_K L1 cache reuse.
// Prefill: contiguous per-thread row ranges + fixed-size accumulators.

static void gemm_q6_K_avx2(
    const float* a_data,
    const uint8_t* w_data,
    float* o_data,
    int64_t M, int64_t K, int64_t N)
{
    constexpr int QK_K = 256;
    constexpr int Q6_K_BLOCK_BYTES = 210;
    const int64_t nb = (K + QK_K - 1) / QK_K;

    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i m15 = _mm256_set1_epi8(15);

    // Quantize all M activation rows to Q8_K (one-time cost)
    scratch_vec<block_q8_K> q8_all(M * nb);
    for (int64_t m = 0; m < M; ++m)
        quantize_row_q8_K(a_data + m * K, q8_all.data() + m * nb, (int)K);

    const int Mi = (int)M;
    if (M == 1) {
        // Decode: RM=4 row grouping, Q8_K stays in L1 across 4 weight rows.
        // Dynamic work-stealing over 16 groups (=64 rows/chunk) like llama.cpp
        // mul_mat chunking for better DRAM streaming parallelism.
        parallel_for_chunks(N, 64, [&](int n0, int n1) {
            for (int64_t n = n0; n < n1; n += 4) {
            int64_t rows = std::min(n + 4, (int64_t)n1) - n;
            for (int64_t r = 0; r < rows; ++r) {
                const uint8_t* q6_row = w_data + (size_t)(n + r) * nb * Q6_K_BLOCK_BYTES;
                __m256 acc = _mm256_setzero_ps();
                for (int64_t i = 0; i < nb; ++i) {
                    const block_q6_K* x = reinterpret_cast<const block_q6_K*>(q6_row) + i;
                    // Stream the weight blocks ahead of use: output_proj reads
                    // ~0.4 GB/token, so the load->use latency must be hidden or
                    // the kernel runs at a fraction of DRAM bandwidth.
                    _mm_prefetch((const char*)(x + 2), _MM_HINT_T0);
                    const block_q8_K* y = q8_all.data() + i;
                    const float d = y->d * vdot::fp16_to_fp32(x->d);
                    const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y->bsums);
                    const __m128i scales = _mm_loadu_si128((const __m128i*)x->scales);
                    const __m256i scales_16 = _mm256_cvtepi8_epi16(scales);
                    const __m256i q8sclsub = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales_16), 5);
                    __m256i sumi = _mm256_setzero_si256();
                    const uint8_t* q4 = x->ql;
                    const uint8_t* qh = x->qh;
                    const int8_t* q8d = y->qs;
                    int is = 0;
                    for (int j = 0; j < QK_K / 128; ++j) {
                        const __m256i q4bits1 = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
                        const __m256i q4bits2 = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
                        const __m256i q4bitsH = _mm256_loadu_si256((const __m256i*)qh); qh += 32;
                        const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m3), 4);
                        const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(12)), 2);
                        const __m256i q4h_2 = _mm256_and_si256(q4bitsH, _mm256_set1_epi8(48));
                        const __m256i q4h_3 = _mm256_srli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(-64)), 2);
                        const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
                        const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
                        const __m256i q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), q4h_2);
                        const __m256i q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), q4h_3);
                        const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8d); q8d += 32;
                        const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8d); q8d += 32;
                        const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8d); q8d += 32;
                        const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8d); q8d += 32;
                        __m256i p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
                        __m256i p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
                        __m256i p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
                        __m256i p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);
                        const __m128i scale_0 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 0));
                        const __m128i scale_1 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 1));
                        const __m128i scale_2 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 2));
                        const __m128i scale_3 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 3));
                        is += 4;
                        p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
                        p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);
                        p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2);
                        p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3);
                        sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
                        sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));
                    }
                    sumi = _mm256_sub_epi32(sumi, q8sclsub);
                    acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
                }
                o_data[n + r] = vdot::hsum_ps256(acc);
            }
        }
        });
    } else {
        // Prefill (M > 1): k-blocked GEMM (see gemm_q2_K_avx2 comment)
        constexpr int64_t TM = 16;  // activation rows per tile
        constexpr int64_t TN = 64;  // weight rows per tile
        #pragma omp parallel
        {
            const int64_t ith = omp_get_thread_num();
            const int64_t nth = omp_get_num_threads();
            const int64_t n_begin = (N * ith) / nth;
            const int64_t n_end = (N * (ith + 1)) / nth;
            alignas(32) float acc[TM];
            for (int64_t n0 = n_begin; n0 < n_end; n0 += TN) {
                const int64_t tn = std::min<int64_t>(TN, n_end - n0);
                for (int64_t m0 = 0; m0 < Mi; m0 += TM) {
                    const int nc = (int)std::min<int64_t>(TM, Mi - m0);
                    for (int64_t t = 0; t < tn; ++t) {
                        const uint8_t* q6_row = w_data + (size_t)(n0 + t) * nb * Q6_K_BLOCK_BYTES;
                        for (int c = 0; c < nc; ++c) acc[c] = 0.0f;
                        for (int64_t i = 0; i < nb; ++i) {
                            const block_q6_K* x = reinterpret_cast<const block_q6_K*>(q6_row) + i;
                            _mm_prefetch((const char*)(q6_row + (i + 2) * Q6_K_BLOCK_BYTES), _MM_HINT_T0);

                            // Q6_K shared decoding (once per block per tile)
                            const float d_half = vdot::fp16_to_fp32(x->d);
                            const __m128i scales = _mm_loadu_si128((const __m128i*)x->scales);
                            const __m256i scales_16 = _mm256_cvtepi8_epi16(scales);
                            const uint8_t* ql_base = x->ql;
                            const uint8_t* qh_base = x->qh;

                            for (int c = 0; c < nc; ++c) {
                                const block_q8_K* y = q8_all.data() + (m0 + c) * nb + i;

                                const float d = y->d * d_half;
                                const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y->bsums);
                                const __m256i q8sclsub = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales_16), 5);

                                const int8_t* q8d = y->qs;
                                const uint8_t* q4 = ql_base;
                                const uint8_t* qh = qh_base;
                                __m256i sumi = _mm256_setzero_si256();
                                int is = 0;

                                for (int j = 0; j < QK_K / 128; ++j) {
                                    const __m256i q4bits1 = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
                                    const __m256i q4bits2 = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
                                    const __m256i q4bitsH = _mm256_loadu_si256((const __m256i*)qh); qh += 32;
                                    const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m3), 4);
                                    const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(12)), 2);
                                    const __m256i q4h_2 = _mm256_and_si256(q4bitsH, _mm256_set1_epi8(48));
                                    const __m256i q4h_3 = _mm256_srli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(-64)), 2);
                                    const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
                                    const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
                                    const __m256i q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), q4h_2);
                                    const __m256i q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), q4h_3);
                                    const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8d); q8d += 32;
                                    const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8d); q8d += 32;
                                    const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8d); q8d += 32;
                                    const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8d); q8d += 32;
                                    __m256i p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
                                    __m256i p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
                                    __m256i p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
                                    __m256i p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);
                                    const __m128i scale_0 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 0));
                                    const __m128i scale_1 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 1));
                                    const __m128i scale_2 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 2));
                                    const __m128i scale_3 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 3));
                                    is += 4;
                                    p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
                                    p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);
                                    p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2);
                                    p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3);
                                    sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
                                    sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));
                                }
                                sumi = _mm256_sub_epi32(sumi, q8sclsub);

                            acc[c] += d * vdot::hsum_ps256(_mm256_cvtepi32_ps(sumi));
                        }
                    }

                    for (int c = 0; c < nc; ++c)
                        o_data[(int64_t)(m0 + c) * N + (n0 + t)] = acc[c];
                }
            }
        }
    }
}
}

// ============================================================================
// Q4_K GEMM: decode (M=1) and prefill (M>1)
// ============================================================================
// Q4_K block: 144 bytes per 256 elements
// Layout: d[2](fp16) + dmin[2](fp16) + scales[12] + qs[128]
// 8 sub-blocks of 32 elements each, with per-sub-block scale and min
// Dot product: maddubs(q4_nibbles, q8_values) * scale + min * bsums
//
// Decode (M=1): falls back to existing GEMV (gemv_q4_k_q8k_transB_avx2)
// Prefill (M>1): contiguous per-thread row ranges, Q4_K decoding shared
// across M-chunks of the activation rows per weight row.

static void gemm_q4_K_avx2(
    const float* a_data,
    const uint8_t* w_data,
    float* o_data,
    int64_t M, int64_t K, int64_t N)
{
    constexpr int QK_K = 256;
    constexpr int Q4_K_BLOCK_BYTES = 144;
    const int64_t nb = (K + QK_K - 1) / QK_K;

    const __m256i m4 = _mm256_set1_epi8(0xF);

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    // Quantize all M activation rows to Q8_K
    scratch_vec<block_q8_K> q8_all(M * nb);
    for (int64_t m = 0; m < M; ++m)
        quantize_row_q8_K(a_data + m * K, q8_all.data() + m * nb, (int)K);

    const int Mi = (int)M;

    if (M == 1) {
        // Decode: RM=4 row grouping, Q8_K stays in L1 across 4 weight rows.
        // Dynamic work-stealing over 16 groups (=64 rows/chunk) like llama.cpp
        // mul_mat chunking for better DRAM streaming parallelism.
        parallel_for_chunks(N, 64, [&](int n0, int n1) {
            for (int64_t n = n0; n < n1; n += 4) {
            int64_t rows = std::min(n + 4, (int64_t)n1) - n;
            for (int64_t r = 0; r < rows; ++r) {
                const uint8_t* q4_row = w_data + (size_t)(n + r) * nb * Q4_K_BLOCK_BYTES;
                __m256 acc = _mm256_setzero_ps();
                __m128 acc_m = _mm_setzero_ps();

                for (int64_t i = 0; i < nb; ++i) {
                    const block_q4_K* x = reinterpret_cast<const block_q4_K*>(q4_row) + i;
                    const block_q8_K* y = q8_all.data() + i;

                    _mm_prefetch((const char*)((const block_q4_K*)q4_row + i + 1), _MM_HINT_T0);

                    const float d = y->d * vdot::fp16_to_fp32(x->d);
                    const float dmin = -y->d * vdot::fp16_to_fp32(x->dmin);

                    // Decode 12-byte scales -> 8 scales + 8 mins
                    uint32_t utmp[4];
                    memcpy(utmp, x->scales, 12);
                    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux = utmp[1] & kmask1;
                    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
                    utmp[2] = uaux;
                    utmp[0] &= kmask1;

                    const __m256i mins_and_scales = _mm256_cvtepu8_epi16(
                        _mm_set_epi32((int)utmp[3], (int)utmp[2], (int)utmp[1], (int)utmp[0]));

                    // Min contribution: mins * bsums
                    const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y->bsums);
                    const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0),
                                                         _mm256_extracti128_si256(q8sums, 1));
                    const __m128i mins128 = _mm256_extracti128_si256(mins_and_scales, 1);
                    const __m128i prod = _mm_madd_epi16(mins128, q8s);
                    acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

                    // Scale contribution: dot product
                    const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
                    const __m256i scales = _mm256_set_m128i(sc128, sc128);

                    const uint8_t* q4 = x->qs;
                    const int8_t* q8d = y->qs;
                    __m256i sumi = _mm256_setzero_si256();

                    for (int j = 0; j < QK_K / 64; ++j) {
                        const __m256i scale_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
                        const __m256i scale_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));

                        const __m256i q4bits = _mm256_loadu_si256((const __m256i*)q4);
                        q4 += 32;
                        const __m256i q4l = _mm256_and_si256(q4bits, m4);
                        const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

                        const __m256i q8l = _mm256_loadu_si256((const __m256i*)q8d);
                        q8d += 32;
                        __m256i p16l = _mm256_maddubs_epi16(q4l, q8l);
                        p16l = _mm256_madd_epi16(scale_l, p16l);

                        const __m256i q8h = _mm256_loadu_si256((const __m256i*)q8d);
                        q8d += 32;
                        __m256i p16h = _mm256_maddubs_epi16(q4h, q8h);
                        p16h = _mm256_madd_epi16(scale_h, p16h);

                        sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
                    }

                    acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
                }

                acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
                acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));
                o_data[n + r] = vdot::hsum_ps256(acc) + _mm_cvtss_f32(acc_m);
            }
        }
        });
    } else {
        // Prefill (M > 1): k-blocked GEMM (see gemm_q2_K_avx2 comment)
        constexpr int64_t TM = 16;  // activation rows per tile
        constexpr int64_t TN = 64;  // weight rows per tile
        #pragma omp parallel
        {
            const int64_t ith = omp_get_thread_num();
            const int64_t nth = omp_get_num_threads();
            const int64_t n_begin = (N * ith) / nth;
            const int64_t n_end = (N * (ith + 1)) / nth;
            alignas(32) float acc[TM];
            for (int64_t n0 = n_begin; n0 < n_end; n0 += TN) {
                const int64_t tn = std::min<int64_t>(TN, n_end - n0);
                for (int64_t m0 = 0; m0 < Mi; m0 += TM) {
                    const int nc = (int)std::min<int64_t>(TM, Mi - m0);
                    for (int64_t t = 0; t < tn; ++t) {
                        const uint8_t* q4_row = w_data + (size_t)(n0 + t) * nb * Q4_K_BLOCK_BYTES;
                        for (int c = 0; c < nc; ++c) acc[c] = 0.0f;
                        for (int64_t i = 0; i < nb; ++i) {
                            const block_q4_K* x = reinterpret_cast<const block_q4_K*>(q4_row) + i;
                            _mm_prefetch((const char*)(q4_row + (i + 2) * Q4_K_BLOCK_BYTES), _MM_HINT_T0);

                            // Q4_K shared decoding (once per block per tile)
                            const float d_half = vdot::fp16_to_fp32(x->d);
                            const float dmin_half = vdot::fp16_to_fp32(x->dmin);

                            uint32_t utmp[4];
                            memcpy(utmp, x->scales, 12);
                            utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
                            const uint32_t uaux = utmp[1] & kmask1;
                            utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
                            utmp[2] = uaux;
                            utmp[0] &= kmask1;

                            const __m256i mins_and_scales = _mm256_cvtepu8_epi16(
                                _mm_set_epi32((int)utmp[3], (int)utmp[2], (int)utmp[1], (int)utmp[0]));

                            const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
                            const __m256i scales = _mm256_set_m128i(sc128, sc128);
                            const __m128i mins128 = _mm256_extracti128_si256(mins_and_scales, 1);

                            const uint8_t* q4_base = x->qs;

                            for (int c = 0; c < nc; ++c) {
                                const block_q8_K* y = q8_all.data() + (m0 + c) * nb + i;

                                const float d = y->d * d_half;
                                const float dmin = -y->d * dmin_half;

                                // Min contribution: mins * bsums (per-m)
                                const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y->bsums);
                                const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0),
                                                                    _mm256_extracti128_si256(q8sums, 1));
                                const __m128i prod = _mm_madd_epi16(mins128, q8s);
                                const __m128i hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, _mm_setzero_si128()), _mm_setzero_si128());
                                const float min_term = dmin * (float)_mm_extract_epi32(hsum, 0);

                                // Dot product: Q4_K qs (shared) x Q8_K qs (per-m)
                                const int8_t* q8d = y->qs;
                                const uint8_t* q4 = q4_base;
                                __m256i sumi = _mm256_setzero_si256();

                                for (int j = 0; j < QK_K / 64; ++j) {
                                    const __m256i scale_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
                                    const __m256i scale_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));

                                    const __m256i q4bits = _mm256_loadu_si256((const __m256i*)q4);
                                    q4 += 32;
                                    const __m256i q4l = _mm256_and_si256(q4bits, m4);
                                    const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

                                    const __m256i q8l = _mm256_loadu_si256((const __m256i*)q8d);
                                    q8d += 32;
                                    __m256i p16l = _mm256_maddubs_epi16(q4l, q8l);
                                    p16l = _mm256_madd_epi16(scale_l, p16l);

                                    const __m256i q8h = _mm256_loadu_si256((const __m256i*)q8d);
                                    q8d += 32;
                                    __m256i p16h = _mm256_maddubs_epi16(q4h, q8h);
                                    p16h = _mm256_madd_epi16(scale_h, p16h);

                                    sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
                                }

                                acc[c] += d * vdot::hsum_ps256(_mm256_cvtepi32_ps(sumi)) + min_term;
                            }
                        }

                        for (int c = 0; c < nc; ++c)
                            o_data[(int64_t)(m0 + c) * N + (n0 + t)] = acc[c];
                    }
                }
            }
        }
    }
}
// ============================================================================
// Q5_K GEMM: decode (M=1) and prefill (M>1)
// ============================================================================
// Q5_K block: 176 bytes per 256 elements
// Layout: d[2](fp16) + dmin[2](fp16) + scales[12] + qh[32] + ql[128]
// 5-bit quantization: low 4 bits from ql, high 1 bit from qh
// Dot product: maddubs(q5_values, q8_values) * scale + min * bsums

static void gemm_q5_K_avx2(
    const float* a_data,
    const uint8_t* w_data,
    float* o_data,
    int64_t M, int64_t K, int64_t N)
{
    constexpr int QK_K = 256;
    constexpr int Q5_K_BLOCK_BYTES = 176;
    const int64_t nb = (K + QK_K - 1) / QK_K;

    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m256i mone = _mm256_set1_epi8(1);
    const __m128i mzero = _mm_setzero_si128();

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    // Quantize all M activation rows to Q8_K (one-time cost)
    scratch_vec<block_q8_K> q8_all(M * nb);
    for (int64_t m = 0; m < M; ++m)
        quantize_row_q8_K(a_data + m * K, q8_all.data() + m * nb, (int)K);

    const int Mi = (int)M;

    if (M == 1) {
        // Decode: RM=4 row grouping, Q8_K stays in L1 across 4 weight rows.
        // Dynamic work-stealing over 16 groups (=64 rows/chunk) like llama.cpp
        // mul_mat chunking for better DRAM streaming parallelism.
        parallel_for_chunks(N, 64, [&](int n0, int n1) {
            for (int64_t n = n0; n < n1; n += 4) {
            int64_t rows = std::min(n + 4, (int64_t)n1) - n;
            for (int64_t r = 0; r < rows; ++r) {
                const uint8_t* q5_row = w_data + (size_t)(n + r) * nb * Q5_K_BLOCK_BYTES;
                __m256 acc = _mm256_setzero_ps();
                float summs = 0.0f;

                for (int64_t i = 0; i < nb; ++i) {
                    const block_q5_K* x = reinterpret_cast<const block_q5_K*>(q5_row) + i;
                    const block_q8_K* y = q8_all.data() + i;

                    _mm_prefetch((const char*)((const block_q5_K*)q5_row + i + 1), _MM_HINT_T0);

                    const float d = y->d * vdot::fp16_to_fp32(x->d);
                    const float dmin = -y->d * vdot::fp16_to_fp32(x->dmin);

                    // Decode scales (same encoding as Q4_K)
                    uint32_t utmp[4];
                    memcpy(utmp, x->scales, 12);
                    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux = utmp[1] & kmask1;
                    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
                    utmp[2] = uaux;
                    utmp[0] &= kmask1;

                    const __m256i mins_and_scales = _mm256_cvtepu8_epi16(
                        _mm_set_epi32((int)utmp[3], (int)utmp[2], (int)utmp[1], (int)utmp[0]));

                    // Min contribution
                    const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y->bsums);
                    const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0),
                                                         _mm256_extracti128_si256(q8sums, 1));
                    const __m128i mins128 = _mm256_extracti128_si256(mins_and_scales, 1);
                    const __m128i prod = _mm_madd_epi16(mins128, q8s);
                    const __m128i hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
                    summs += dmin * (float)_mm_extract_epi32(hsum, 0);

                    // Scales
                    const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
                    const __m256i scales = _mm256_set_m128i(sc128, sc128);

                    // Q5_K: 5-bit values = low 4 bits from ql + high 1 bit from qh
                    const __m256i hbits = _mm256_loadu_si256((const __m256i*)x->qh);
                    __m256i hmask = mone;

                    const uint8_t* q5 = x->ql;
                    const int8_t* q8d = y->qs;
                    __m256i sumi = _mm256_setzero_si256();
                    int bit = 0;

                    for (int j = 0; j < QK_K / 64; ++j) {
                        const __m256i scale_0 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
                        const __m256i scale_1 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));

                        const __m256i q5bits = _mm256_loadu_si256((const __m256i*)q5);
                        q5 += 32;

                        const __m256i q5l_0 = _mm256_and_si256(q5bits, m4);
                        const __m256i q5h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
                        const __m256i q5_0  = _mm256_add_epi8(q5l_0, q5h_0);
                        hmask = _mm256_slli_epi16(hmask, 1);

                        const __m256i q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4);
                        const __m256i q5h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
                        const __m256i q5_1  = _mm256_add_epi8(q5l_1, q5h_1);
                        hmask = _mm256_slli_epi16(hmask, 1);

                        const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8d); q8d += 32;
                        const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8d); q8d += 32;

                        __m256i p16_0 = _mm256_maddubs_epi16(q5_0, q8_0);
                        __m256i p16_1 = _mm256_maddubs_epi16(q5_1, q8_1);
                        p16_0 = _mm256_madd_epi16(scale_0, p16_0);
                        p16_1 = _mm256_madd_epi16(scale_1, p16_1);

                        sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
                    }

                    acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
                }

                o_data[n + r] = vdot::hsum_ps256(acc) + summs;
            }
        }
        });
    } else {
        // Prefill (M > 1): k-blocked GEMM (see gemm_q2_K_avx2 comment)
        constexpr int64_t TM = 16;  // activation rows per tile
        constexpr int64_t TN = 64;  // weight rows per tile
        #pragma omp parallel
        {
            const int64_t ith = omp_get_thread_num();
            const int64_t nth = omp_get_num_threads();
            const int64_t n_begin = (N * ith) / nth;
            const int64_t n_end = (N * (ith + 1)) / nth;
            alignas(32) float acc[TM];
            for (int64_t n0 = n_begin; n0 < n_end; n0 += TN) {
                const int64_t tn = std::min<int64_t>(TN, n_end - n0);
                for (int64_t m0 = 0; m0 < Mi; m0 += TM) {
                    const int nc = (int)std::min<int64_t>(TM, Mi - m0);
                    for (int64_t t = 0; t < tn; ++t) {
                        const uint8_t* q5_row = w_data + (size_t)(n0 + t) * nb * Q5_K_BLOCK_BYTES;
                        for (int c = 0; c < nc; ++c) acc[c] = 0.0f;
                        for (int64_t i = 0; i < nb; ++i) {
                            const block_q5_K* x = reinterpret_cast<const block_q5_K*>(q5_row) + i;
                            _mm_prefetch((const char*)(q5_row + (i + 2) * Q5_K_BLOCK_BYTES), _MM_HINT_T0);

                            // Q5_K shared decoding (once per block per tile)
                            const float d_half = vdot::fp16_to_fp32(x->d);
                            const float dmin_half = vdot::fp16_to_fp32(x->dmin);

                            uint32_t utmp[4];
                            memcpy(utmp, x->scales, 12);
                            utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
                            const uint32_t uaux = utmp[1] & kmask1;
                            utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
                            utmp[2] = uaux;
                            utmp[0] &= kmask1;

                            const __m256i mins_and_scales = _mm256_cvtepu8_epi16(
                                _mm_set_epi32((int)utmp[3], (int)utmp[2], (int)utmp[1], (int)utmp[0]));

                            const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
                            const __m256i scales = _mm256_set_m128i(sc128, sc128);
                            const __m128i mins128 = _mm256_extracti128_si256(mins_and_scales, 1);

                            const __m256i hbits = _mm256_loadu_si256((const __m256i*)x->qh);
                            const uint8_t* q5_base = x->ql;

                            for (int c = 0; c < nc; ++c) {
                                const block_q8_K* y = q8_all.data() + (m0 + c) * nb + i;

                                const float d = y->d * d_half;
                                const float dmin = -y->d * dmin_half;

                                // Min contribution: mins * bsums (per-m)
                                const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y->bsums);
                                const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0),
                                                                     _mm256_extracti128_si256(q8sums, 1));
                                const __m128i prod = _mm_madd_epi16(mins128, q8s);
                                const __m128i hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
                                const float min_term = dmin * (float)_mm_extract_epi32(hsum, 0);

                                // Dot product: shared ql + qh, per-m q8
                                const int8_t* q8d = y->qs;
                                const uint8_t* q5 = q5_base;
                                __m256i sumi = _mm256_setzero_si256();
                                __m256i hmask = mone;
                                int bit = 0;

                                for (int j = 0; j < QK_K / 64; ++j) {
                                    const __m256i scale_0 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
                                    const __m256i scale_1 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));

                                    const __m256i q5bits = _mm256_loadu_si256((const __m256i*)q5);
                                    q5 += 32;

                                    const __m256i q5l_0 = _mm256_and_si256(q5bits, m4);
                                    const __m256i q5h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
                                    const __m256i q5_0  = _mm256_add_epi8(q5l_0, q5h_0);
                                    hmask = _mm256_slli_epi16(hmask, 1);

                                    const __m256i q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4);
                                    const __m256i q5h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
                                    const __m256i q5_1  = _mm256_add_epi8(q5l_1, q5h_1);
                                    hmask = _mm256_slli_epi16(hmask, 1);

                                    const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8d); q8d += 32;
                                    const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8d); q8d += 32;

                                    __m256i p16_0 = _mm256_maddubs_epi16(q5_0, q8_0);
                                    __m256i p16_1 = _mm256_maddubs_epi16(q5_1, q8_1);
                                    p16_0 = _mm256_madd_epi16(scale_0, p16_0);
                                    p16_1 = _mm256_madd_epi16(scale_1, p16_1);

sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
                            }

                            acc[c] += d * vdot::hsum_ps256(_mm256_cvtepi32_ps(sumi)) + min_term;
                        }
                    }

                    for (int c = 0; c < nc; ++c)
                        o_data[(int64_t)(m0 + c) * N + (n0 + t)] = acc[c];
                }
            }
        }
    }
}
}

// ============================================================================
// Q2_K GEMM: decode (M=1) and prefill (M>1)
// ============================================================================
// Q2_K block: 84 bytes per 256 elements
// Layout: scales[16] + qs[64] + d[2](fp16) + dmin[2](fp16)
// 2-bit quantization with d + dmin
// Scales are 4-bit packed: lo nibble = scale, hi nibble = min

static void gemm_q2_K_avx2(
    const float* a_data,
    const uint8_t* w_data,
    float* o_data,
    int64_t M, int64_t K, int64_t N)
{
    constexpr int QK_K = 256;
    constexpr int Q2_K_BLOCK_BYTES = 84;
    const int64_t nb = (K + QK_K - 1) / QK_K;

    const __m256i m3 = _mm256_set1_epi8(3);
    const __m128i m4 = _mm_set1_epi8(0xF);

    // Quantize all M activation rows to Q8_K
    scratch_vec<block_q8_K> q8_all(M * nb);
    for (int64_t m = 0; m < M; ++m)
        quantize_row_q8_K(a_data + m * K, q8_all.data() + m * nb, (int)K);

    const int Mi = (int)M;

    if (M == 1) {
        // Decode: RM=4 row grouping, Q8_K stays in L1 across 4 weight rows.
        // Dynamic work-stealing over 16 groups (=64 rows/chunk) like llama.cpp
        // mul_mat chunking for better DRAM streaming parallelism.
        parallel_for_chunks(N, 64, [&](int n0, int n1) {
            for (int64_t n = n0; n < n1; n += 4) {
            int64_t rows = std::min(n + 4, (int64_t)n1) - n;
            for (int64_t r = 0; r < rows; ++r) {
                const uint8_t* q2_row = w_data + (size_t)(n + r) * nb * Q2_K_BLOCK_BYTES;
                __m256 acc = _mm256_setzero_ps();

                for (int64_t i = 0; i < nb; ++i) {
                    const block_q2_K* x = reinterpret_cast<const block_q2_K*>(q2_row) + i;
                    const block_q8_K* y = q8_all.data() + i;

                    _mm_prefetch((const char*)((const block_q2_K*)q2_row + i + 1), _MM_HINT_T0);

                    const float d = y->d * vdot::fp16_to_fp32(x->d);
                    const float dmin = -y->d * vdot::fp16_to_fp32(x->dmin);

                    const uint8_t* q2 = x->qs;
                    const int8_t* q8d = y->qs;

                    const __m128i mins_and_scales = _mm_loadu_si128((const __m128i*)x->scales);
                    const __m128i scales8 = _mm_and_si128(mins_and_scales, m4);
                    const __m128i mins8 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);

                    const __m256i mins = _mm256_cvtepi8_epi16(mins8);
                    const __m256i prod = _mm256_madd_epi16(mins, _mm256_loadu_si256((const __m256i*)y->bsums));
                    acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&dmin), _mm256_cvtepi32_ps(prod), acc);

                    const __m256i all_scales = _mm256_cvtepi8_epi16(scales8);
                    const __m128i l_scales = _mm256_extracti128_si256(all_scales, 0);
                    const __m128i h_scales = _mm256_extracti128_si256(all_scales, 1);
                    const __m256i sc[2] = {_mm256_set_m128i(l_scales, l_scales), _mm256_set_m128i(h_scales, h_scales)};

                    __m256i sumi = _mm256_setzero_si256();
                    for (int j = 0; j < QK_K / 128; ++j) {
                        const __m256i q2bits = _mm256_loadu_si256((const __m256i*)q2); q2 += 32;

                        const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8d); q8d += 32;
                        const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8d); q8d += 32;
                        const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8d); q8d += 32;
                        const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8d); q8d += 32;

                        const __m256i q2_0 = _mm256_and_si256(q2bits, m3);
                        const __m256i q2_1 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 2), m3);
                        const __m256i q2_2 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 4), m3);
                        const __m256i q2_3 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 6), m3);

                        __m256i p0 = _mm256_maddubs_epi16(q2_0, q8_0);
                        __m256i p1 = _mm256_maddubs_epi16(q2_1, q8_1);
                        __m256i p2 = _mm256_maddubs_epi16(q2_2, q8_2);
                        __m256i p3 = _mm256_maddubs_epi16(q2_3, q8_3);

                        p0 = _mm256_madd_epi16(_mm256_shuffle_epi8(sc[j], get_scale_shuffle_q3k(0)), p0);
                        p1 = _mm256_madd_epi16(_mm256_shuffle_epi8(sc[j], get_scale_shuffle_q3k(1)), p1);
                        p2 = _mm256_madd_epi16(_mm256_shuffle_epi8(sc[j], get_scale_shuffle_q3k(2)), p2);
                        p3 = _mm256_madd_epi16(_mm256_shuffle_epi8(sc[j], get_scale_shuffle_q3k(3)), p3);

                        sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(_mm256_add_epi32(p0, p1), _mm256_add_epi32(p2, p3)));
                    }

                    acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
                }

                o_data[n + r] = vdot::hsum_ps256(acc);
            }
        }
        });
    } else {
        // Prefill (M > 1): k-blocked GEMM. Weight rows stream once (i innermost
        // per row) while the M-block of q8 activations for the current k-block
        // stays resident in L1/L2 and is reused across a tile of weight rows.
        // Each output element is written exactly once (register accumulation
        // across k-blocks, then a single store).
        constexpr int64_t TM = 16;  // activation rows per tile
        constexpr int64_t TN = 64;  // weight rows per tile
        #pragma omp parallel
        {
            const int64_t ith = omp_get_thread_num();
            const int64_t nth = omp_get_num_threads();
            const int64_t n_begin = (N * ith) / nth;
            const int64_t n_end = (N * (ith + 1)) / nth;
            alignas(32) float acc[TM];
            for (int64_t n0 = n_begin; n0 < n_end; n0 += TN) {
                const int64_t tn = std::min<int64_t>(TN, n_end - n0);
                for (int64_t m0 = 0; m0 < Mi; m0 += TM) {
                    const int nc = (int)std::min<int64_t>(TM, Mi - m0);
                    for (int64_t t = 0; t < tn; ++t) {
                        const uint8_t* q2_row = w_data + (size_t)(n0 + t) * nb * Q2_K_BLOCK_BYTES;
                        for (int c = 0; c < nc; ++c) acc[c] = 0.0f;
                        for (int64_t i = 0; i < nb; ++i) {
                            const block_q2_K* x = reinterpret_cast<const block_q2_K*>(q2_row) + i;
                            _mm_prefetch((const char*)(q2_row + (i + 2) * Q2_K_BLOCK_BYTES), _MM_HINT_T0);

                            // Q2_K shared decoding (once per block per tile)
                            const float d_half = vdot::fp16_to_fp32(x->d);
                            const float dmin_half = vdot::fp16_to_fp32(x->dmin);

                            const __m128i mins_and_scales = _mm_loadu_si128((const __m128i*)x->scales);
                            const __m128i scales8 = _mm_and_si128(mins_and_scales, m4);
                            const __m128i mins8 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);

                            const __m256i mins = _mm256_cvtepi8_epi16(mins8);
                            const __m256i all_scales = _mm256_cvtepi8_epi16(scales8);
                            const __m128i l_scales = _mm256_extracti128_si256(all_scales, 0);
                            const __m128i h_scales = _mm256_extracti128_si256(all_scales, 1);
                            const __m256i sc[2] = {_mm256_set_m128i(l_scales, l_scales), _mm256_set_m128i(h_scales, h_scales)};

                            const uint8_t* q2_base = x->qs;

                            for (int c = 0; c < nc; ++c) {
                                const block_q8_K* y = q8_all.data() + (m0 + c) * nb + i;

                                const float d = y->d * d_half;
                                const float dmin = -y->d * dmin_half;

                                // Min contribution: mins * bsums
                                const __m256i prod = _mm256_madd_epi16(mins, _mm256_loadu_si256((const __m256i*)y->bsums));
                                const float min_term = dmin * vdot::hsum_ps256(_mm256_cvtepi32_ps(prod));

                                // Dot product: shared q2, per-m q8
                                const int8_t* q8d = y->qs;
                                const uint8_t* q2 = q2_base;
                                __m256i sumi = _mm256_setzero_si256();

                                for (int j = 0; j < QK_K / 128; ++j) {
                                    const __m256i q2bits = _mm256_loadu_si256((const __m256i*)q2); q2 += 32;

                                    const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8d); q8d += 32;
                                    const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8d); q8d += 32;
                                    const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8d); q8d += 32;
                                    const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8d); q8d += 32;

                                    const __m256i q2_0 = _mm256_and_si256(q2bits, m3);
                                    const __m256i q2_1 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 2), m3);
                                    const __m256i q2_2 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 4), m3);
                                    const __m256i q2_3 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 6), m3);

                                    __m256i p0 = _mm256_maddubs_epi16(q2_0, q8_0);
                                    __m256i p1 = _mm256_maddubs_epi16(q2_1, q8_1);
                                    __m256i p2 = _mm256_maddubs_epi16(q2_2, q8_2);
                                    __m256i p3 = _mm256_maddubs_epi16(q2_3, q8_3);

                                    p0 = _mm256_madd_epi16(_mm256_shuffle_epi8(sc[j], get_scale_shuffle_q3k(0)), p0);
                                    p1 = _mm256_madd_epi16(_mm256_shuffle_epi8(sc[j], get_scale_shuffle_q3k(1)), p1);
                                    p2 = _mm256_madd_epi16(_mm256_shuffle_epi8(sc[j], get_scale_shuffle_q3k(2)), p2);
                                    p3 = _mm256_madd_epi16(_mm256_shuffle_epi8(sc[j], get_scale_shuffle_q3k(3)), p3);

                                    sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(_mm256_add_epi32(p0, p1), _mm256_add_epi32(p2, p3)));
                                }

                                acc[c] += d * vdot::hsum_ps256(_mm256_cvtepi32_ps(sumi)) + min_term;
                            }
                        }

                        for (int c = 0; c < nc; ++c)
                            o_data[(int64_t)(m0 + c) * N + (n0 + t)] = acc[c];
                    }
                }
            }
        }
    }
}

// ============================================================================
// Q3_K GEMM: decode (M=1) and prefill (M>1)
// ============================================================================
// Q3_K block: 110 bytes per 256 elements
// Layout: hmask[32] + qs[64] + scales[12] + d[2](fp16)
// 3-bit quantization with sign bit from hmask
// No min contribution (unlike Q2_K/Q4_K/Q5_K)

static void gemm_q3_K_avx2(
    const float* a_data,
    const uint8_t* w_data,
    float* o_data,
    int64_t M, int64_t K, int64_t N)
{
    constexpr int QK_K = 256;
    constexpr int Q3_K_BLOCK_BYTES = 110;
    const int64_t nb = (K + QK_K - 1) / QK_K;

    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i mone = _mm256_set1_epi8(1);
    const __m128i m32 = _mm_set1_epi8(32);

    static const uint32_t kmask1 = 0x03030303;
    static const uint32_t kmask2 = 0x0f0f0f0f;

    // Quantize all M activation rows to Q8_K
    scratch_vec<block_q8_K> q8_all(M * nb);
    for (int64_t m = 0; m < M; ++m)
        quantize_row_q8_K(a_data + m * K, q8_all.data() + m * nb, (int)K);

    const int Mi = (int)M;

    if (M == 1) {
        // Decode: RM=4 row grouping, Q8_K stays in L1 across 4 weight rows.
        // Dynamic work-stealing over 16 groups (=64 rows/chunk) like llama.cpp
        // mul_mat chunking for better DRAM streaming parallelism.
        parallel_for_chunks(N, 64, [&](int n0, int n1) {
            for (int64_t n = n0; n < n1; n += 4) {
            int64_t rows = std::min(n + 4, (int64_t)n1) - n;
            for (int64_t r = 0; r < rows; ++r) {
                const uint8_t* q3_row = w_data + (size_t)(n + r) * nb * Q3_K_BLOCK_BYTES;
                __m256 acc = _mm256_setzero_ps();

                for (int64_t i = 0; i < nb; ++i) {
                    const block_q3_K* x = reinterpret_cast<const block_q3_K*>(q3_row) + i;
                    const block_q8_K* y = q8_all.data() + i;

                    _mm_prefetch((const char*)((const block_q3_K*)q3_row + i + 1), _MM_HINT_T0);

                    const float d = y->d * vdot::fp16_to_fp32(x->d);

                    const uint8_t* q3 = x->qs;
                    const int8_t* q8d = y->qs;

                    uint32_t aux[3];
                    memcpy(aux, x->scales, 12);
                    __m128i scales128 = _mm_set_epi32(
                            ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4),
                            ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4),
                            (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4),
                            (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4));
                    scales128 = _mm_sub_epi8(scales128, m32);

                    const __m256i hbits = _mm256_loadu_si256((const __m256i*)x->hmask);

                    __m256i sumi = _mm256_setzero_si256();
                    int bit = 0;
                    int is = 0;

                    for (int j = 0; j < QK_K / 128; ++j) {
                        const __m256i q3bits = _mm256_loadu_si256((const __m256i*)q3); q3 += 32;

                        const __m256i q3l_0 = _mm256_and_si256(q3bits, m3);
                        const __m256i q3h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
                        ++bit;
                        const __m256i q3l_1 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 2), m3);
                        const __m256i q3h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
                        ++bit;
                        const __m256i q3l_2 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 4), m3);
                        const __m256i q3h_2 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
                        ++bit;
                        const __m256i q3l_3 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 6), m3);
                        const __m256i q3h_3 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
                        ++bit;

                        const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8d); q8d += 32;
                        const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8d); q8d += 32;
                        const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8d); q8d += 32;
                        const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8d); q8d += 32;

                        __m256i q8s_0 = _mm256_maddubs_epi16(q3h_0, q8_0);
                        __m256i q8s_1 = _mm256_maddubs_epi16(q3h_1, q8_1);
                        __m256i q8s_2 = _mm256_maddubs_epi16(q3h_2, q8_2);
                        __m256i q8s_3 = _mm256_maddubs_epi16(q3h_3, q8_3);

                        __m256i p16_0 = _mm256_maddubs_epi16(q3l_0, q8_0);
                        __m256i p16_1 = _mm256_maddubs_epi16(q3l_1, q8_1);
                        __m256i p16_2 = _mm256_maddubs_epi16(q3l_2, q8_2);
                        __m256i p16_3 = _mm256_maddubs_epi16(q3l_3, q8_3);

                        p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
                        p16_1 = _mm256_sub_epi16(p16_1, q8s_1);
                        p16_2 = _mm256_sub_epi16(p16_2, q8s_2);
                        p16_3 = _mm256_sub_epi16(p16_3, q8s_3);

                        const __m128i scale_0 = _mm_shuffle_epi8(scales128, get_scale_shuffle(is + 0));
                        const __m128i scale_1 = _mm_shuffle_epi8(scales128, get_scale_shuffle(is + 1));
                        const __m128i scale_2 = _mm_shuffle_epi8(scales128, get_scale_shuffle(is + 2));
                        const __m128i scale_3 = _mm_shuffle_epi8(scales128, get_scale_shuffle(is + 3));
                        is += 4;

                        p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
                        p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);
                        p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2);
                        p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3);

                        sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(_mm256_add_epi32(p16_0, p16_1), _mm256_add_epi32(p16_2, p16_3)));
                    }

                    acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
                }

                o_data[n + r] = vdot::hsum_ps256(acc);
            }
        }
        });
    } else {
        // Prefill (M > 1): k-blocked GEMM (see gemm_q2_K_avx2 comment)
        constexpr int64_t TM = 16;  // activation rows per tile
        constexpr int64_t TN = 64;  // weight rows per tile
        #pragma omp parallel
        {
            const int64_t ith = omp_get_thread_num();
            const int64_t nth = omp_get_num_threads();
            const int64_t n_begin = (N * ith) / nth;
            const int64_t n_end = (N * (ith + 1)) / nth;
            alignas(32) float acc[TM];
            for (int64_t n0 = n_begin; n0 < n_end; n0 += TN) {
                const int64_t tn = std::min<int64_t>(TN, n_end - n0);
                for (int64_t m0 = 0; m0 < Mi; m0 += TM) {
                    const int nc = (int)std::min<int64_t>(TM, Mi - m0);
                    for (int64_t t = 0; t < tn; ++t) {
                        const uint8_t* q3_row = w_data + (size_t)(n0 + t) * nb * Q3_K_BLOCK_BYTES;
                        for (int c = 0; c < nc; ++c) acc[c] = 0.0f;
                        for (int64_t i = 0; i < nb; ++i) {
                            const block_q3_K* x = reinterpret_cast<const block_q3_K*>(q3_row) + i;
                            _mm_prefetch((const char*)(q3_row + (i + 2) * Q3_K_BLOCK_BYTES), _MM_HINT_T0);

                            // Q3_K shared decoding (once per block per tile)
                            const float d_half = vdot::fp16_to_fp32(x->d);

                            uint32_t aux[3];
                            memcpy(aux, x->scales, 12);
                            __m128i scales128 = _mm_set_epi32(
                                    ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4),
                                    ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4),
                                    (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4),
                                    (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4));
                            scales128 = _mm_sub_epi8(scales128, m32);

                            const __m256i hbits = _mm256_loadu_si256((const __m256i*)x->hmask);
                            const uint8_t* q3_base = x->qs;

                            for (int c = 0; c < nc; ++c) {
                                const block_q8_K* y = q8_all.data() + (m0 + c) * nb + i;

                                const float d = y->d * d_half;

                                const int8_t* q8d = y->qs;
                                const uint8_t* q3 = q3_base;
                                __m256i sumi = _mm256_setzero_si256();
                                int bit = 0;
                                int is = 0;

                                for (int j = 0; j < QK_K / 128; ++j) {
                                    const __m256i q3bits = _mm256_loadu_si256((const __m256i*)q3); q3 += 32;

                                    const __m256i q3l_0 = _mm256_and_si256(q3bits, m3);
                                    const __m256i q3h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
                                    ++bit;
                                    const __m256i q3l_1 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 2), m3);
                                    const __m256i q3h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
                                    ++bit;
                                    const __m256i q3l_2 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 4), m3);
                                    const __m256i q3h_2 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
                                    ++bit;
                                    const __m256i q3l_3 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 6), m3);
                                    const __m256i q3h_3 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
                                    ++bit;

                                    const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8d); q8d += 32;
                                    const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8d); q8d += 32;
                                    const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8d); q8d += 32;
                                    const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8d); q8d += 32;

                                    __m256i q8s_0 = _mm256_maddubs_epi16(q3h_0, q8_0);
                                    __m256i q8s_1 = _mm256_maddubs_epi16(q3h_1, q8_1);
                                    __m256i q8s_2 = _mm256_maddubs_epi16(q3h_2, q8_2);
                                    __m256i q8s_3 = _mm256_maddubs_epi16(q3h_3, q8_3);

                                    __m256i p16_0 = _mm256_maddubs_epi16(q3l_0, q8_0);
                                    __m256i p16_1 = _mm256_maddubs_epi16(q3l_1, q8_1);
                                    __m256i p16_2 = _mm256_maddubs_epi16(q3l_2, q8_2);
                                    __m256i p16_3 = _mm256_maddubs_epi16(q3l_3, q8_3);

                                    p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
                                    p16_1 = _mm256_sub_epi16(p16_1, q8s_1);
                                    p16_2 = _mm256_sub_epi16(p16_2, q8s_2);
                                    p16_3 = _mm256_sub_epi16(p16_3, q8s_3);

                                    const __m128i scale_0 = _mm_shuffle_epi8(scales128, get_scale_shuffle(is + 0));
                                    const __m128i scale_1 = _mm_shuffle_epi8(scales128, get_scale_shuffle(is + 1));
                                    const __m128i scale_2 = _mm_shuffle_epi8(scales128, get_scale_shuffle(is + 2));
                                    const __m128i scale_3 = _mm_shuffle_epi8(scales128, get_scale_shuffle(is + 3));
                                    is += 4;

                                    p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
                                    p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);
                                    p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2);
                                    p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3);

                                    sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(_mm256_add_epi32(p16_0, p16_1), _mm256_add_epi32(p16_2, p16_3)));
                                }

                                acc[c] += d * vdot::hsum_ps256(_mm256_cvtepi32_ps(sumi));
                            }
                        }

                        for (int c = 0; c < nc; ++c)
                            o_data[(int64_t)(m0 + c) * N + (n0 + t)] = acc[c];
                    }
                }
            }
        }
    }
}

#endif  // USE_AVX2

// ============================================================================
// AVX-512 VNNI GEMM Micro-Kernels
// ============================================================================
// Requires: AVX-512F, AVX-512BW, AVX-512VNNI, AVX-512VL, F16C
// Key advantage: _mm512_dpbusd_epi32 replaces sign_epi8 + maddubs + madd + cvt
//                32 ZMM registers allow larger tiles (RN=4 decode, RN=8 prefill)
//                _mm512_cvtph_ps converts 8 fp16 → fp32 at once

#ifdef USE_AVX512_VNNI

#include <immintrin.h>

// All AVX-512 VNNI functions use target attribute to enable AVX-512 codegen
// even when -march=native doesn't include it (e.g., WSL2).
#define FORGE_AVX512_VNNI_ATTR \
    __attribute__((target("avx512f,avx512bw,avx512vnni,avx512vl,avx512cd,f16c")))

// Helper: horizontal sum of __m512 (16 floats → 1 scalar)
static FORGE_AVX512_VNNI_ATTR inline float hsum_ps512(__m512 v) {
    __m256 lo = _mm512_castps512_ps256(v);
    __m256 hi = _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm512_extractf32x4_ps(v, 2)),
        _mm512_extractf32x4_ps(v, 3), 1);
    __m256 sum256 = _mm256_add_ps(lo, hi);
    return vdot::hsum_ps256(sum256);
}

// ============================================================================
// Q4_0 AVX-512 VNNI Decode GEMM (M=1, RN=4 weight cols)
// ============================================================================
// Processes 4 weight rows simultaneously using VNNI dpbusd.
// With 32 ZMM registers: 4×4 = 16 int32 accumulators + 8 temp = 24 ZMM.
// Activation quantized once to Q8_0_act, shared across all weight rows.
//
// VNNI _mm512_dpbusd_epi32(dst, a_uint8, b_int8):
//   for i in 0..63: dst[lane] += (uint8)a[i] * (int8)b[i]
//   Sums 16 products per 32-bit lane, 16 lanes per ZMM.
//
// For Q4_0: we need signed×signed dot product.
// Strategy: sign_epi8(w, w) → |w| (uint8), sign_epi8(act, w) → act*sign(w) (int8)
// Then dpbusd(|w|, act*sign(w)) = |w| * act*sign(w) = w * act ✓

// ---- Q4_0 AVX-512 VNNI Decode (M=1, RN=4) ----
// Per-block: dpbusd → cvtepi32_ps → scale (fp16*cvt + fp32 act) → fmadd accumulate
// This replaces the sign_epi8 + maddubs + madd + cvt chain with dpbusd alone,
// saving ~3 instructions per block per row.

static FORGE_AVX512_VNNI_ATTR void gemm_q4_0_decode_vnni(
    const float* a_data,
    const uint8_t* w_data,
    float* o_data,
    int64_t K,
    int64_t N)
{
    constexpr int RM = 4;
    constexpr int BLOCK_SIZE = 32;
    constexpr int BLOCK_BYTES = 18;
    const int64_t nb = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Quantize activation once — shared by all weight rows
    scratch_vec<block_q8_0_act> q8_act(nb);
    quantize_row_q8_0_act(a_data, q8_act.data(), (int)K);

    #pragma omp parallel for schedule(static)
    for (int64_t n = 0; n < N; n += RM) {
        int64_t rows = (n + RM <= N) ? RM : (N - n);

        // Float accumulators (one per weight row, 16 lanes for hsum)
        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps();
        __m512 acc3 = _mm512_setzero_ps();

        const uint8_t* w0 = w_data + (size_t)(n + 0) * nb * BLOCK_BYTES;
        const uint8_t* w1 = w_data + (size_t)(n + 1) * nb * BLOCK_BYTES;
        const uint8_t* w2 = w_data + (size_t)(n + 2) * nb * BLOCK_BYTES;
        const uint8_t* w3 = w_data + (size_t)(n + 3) * nb * BLOCK_BYTES;

        for (int64_t l = 0; l < nb; ++l) {
            // Prefetch next weight blocks (further ahead than AVX2: 6 vs 4)
            if (l + 6 < nb) {
                _mm_prefetch((const char*)(w0 + (l + 6) * BLOCK_BYTES), _MM_HINT_T1);
                _mm_prefetch((const char*)(w1 + (l + 6) * BLOCK_BYTES), _MM_HINT_T1);
                if (l + 4 < nb) {
                    _mm_prefetch((const char*)(w0 + (l + 4) * BLOCK_BYTES), _MM_HINT_T0);
                    _mm_prefetch((const char*)(w1 + (l + 4) * BLOCK_BYTES), _MM_HINT_T0);
                }
                if (rows > 2) {
                    if (l + 6 < nb) {
                        _mm_prefetch((const char*)(w2 + (l + 6) * BLOCK_BYTES), _MM_HINT_T1);
                        _mm_prefetch((const char*)(w3 + (l + 6) * BLOCK_BYTES), _MM_HINT_T1);
                    }
                    if (l + 4 < nb) {
                        _mm_prefetch((const char*)(w2 + (l + 4) * BLOCK_BYTES), _MM_HINT_T0);
                        _mm_prefetch((const char*)(w3 + (l + 4) * BLOCK_BYTES), _MM_HINT_T0);
                    }
                }
            }

            // Load weight blocks (Q4_0 → 32 signed int8 via AVX2 denibble)
            __m256i wvec0 = BlockLoader<block_q4_0_tag>::load(w0 + l * BLOCK_BYTES);
            __m256i wvec1 = (rows > 1) ? BlockLoader<block_q4_0_tag>::load(w1 + l * BLOCK_BYTES) : _mm256_setzero_si256();
            __m256i wvec2 = (rows > 2) ? BlockLoader<block_q4_0_tag>::load(w2 + l * BLOCK_BYTES) : _mm256_setzero_si256();
            __m256i wvec3 = (rows > 3) ? BlockLoader<block_q4_0_tag>::load(w3 + l * BLOCK_BYTES) : _mm256_setzero_si256();

            // Load activation block
            __m256i avec = BlockLoader<block_q8_0_act>::load(&q8_act[l]);

            // VNNI: sign_epi8 to get |w| (uint8) and a*sign(w) (int8)
            __m256i abs_w0 = _mm256_sign_epi8(wvec0, wvec0);
            __m256i abs_w1 = (rows > 1) ? _mm256_sign_epi8(wvec1, wvec1) : _mm256_setzero_si256();
            __m256i abs_w2 = (rows > 2) ? _mm256_sign_epi8(wvec2, wvec2) : _mm256_setzero_si256();
            __m256i abs_w3 = (rows > 3) ? _mm256_sign_epi8(wvec3, wvec3) : _mm256_setzero_si256();

            __m256i sa0 = _mm256_sign_epi8(avec, wvec0);
            __m256i sa1 = (rows > 1) ? _mm256_sign_epi8(avec, wvec1) : _mm256_setzero_si256();
            __m256i sa2 = (rows > 2) ? _mm256_sign_epi8(avec, wvec2) : _mm256_setzero_si256();
            __m256i sa3 = (rows > 3) ? _mm256_sign_epi8(avec, wvec3) : _mm256_setzero_si256();

            // Zero-extend 256-bit to 512-bit
            __m512i abs_w0_512 = _mm512_castsi256_si512(abs_w0);
            __m512i abs_w1_512 = _mm512_castsi256_si512(abs_w1);
            __m512i abs_w2_512 = _mm512_castsi256_si512(abs_w2);
            __m512i abs_w3_512 = _mm512_castsi256_si512(abs_w3);
            __m512i sa0_512 = _mm512_castsi256_si512(sa0);
            __m512i sa1_512 = _mm512_castsi256_si512(sa1);
            __m512i sa2_512 = _mm512_castsi256_si512(sa2);
            __m512i sa3_512 = _mm512_castsi256_si512(sa3);

            // VNNI dpbusd: 16 int8×uint8 → 16 int32 partial dot products
            __m512i dot0 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), abs_w0_512, sa0_512);
            __m512i dot1 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), abs_w1_512, sa1_512);
            __m512i dot2 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), abs_w2_512, sa2_512);
            __m512i dot3 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), abs_w3_512, sa3_512);

            // Convert int32 → fp32
            __m512 fdot0 = _mm512_cvtepi32_ps(dot0);
            __m512 fdot1 = _mm512_cvtepi32_ps(dot1);
            __m512 fdot2 = _mm512_cvtepi32_ps(dot2);
            __m512 fdot3 = _mm512_cvtepi32_ps(dot3);

            // F16C: pack 4 fp16 weight scales → convert to 4 fp32
            uint64_t packed_scales = 0;
            uint16_t d0, d1 = 0, d2 = 0, d3 = 0;
            memcpy(&d0, w0 + l * BLOCK_BYTES, 2);
            if (rows > 1) memcpy(&d1, w1 + l * BLOCK_BYTES, 2);
            if (rows > 2) memcpy(&d2, w2 + l * BLOCK_BYTES, 2);
            if (rows > 3) memcpy(&d3, w3 + l * BLOCK_BYTES, 2);
            packed_scales = (uint64_t)d0 | ((uint64_t)d1 << 16) | ((uint64_t)d2 << 32) | ((uint64_t)d3 << 48);

            __m128 sw_f16 = _mm_cvtph_ps(_mm_set_epi64x(0, packed_scales));
            float act_scale = q8_act[l].d;
            __m128 sw_all = _mm_mul_ps(sw_f16, _mm_set1_ps(act_scale));

            // Broadcast each combined scale to __m512 for FMADD
            // Use AVX-512 broadcast: much cheaper than AVX2 shuffle+permute
            __m512 sc0 = _mm512_broadcast_f32x4(_mm_shuffle_ps(sw_all, sw_all, 0x00));
            __m512 sc1 = _mm512_broadcast_f32x4(_mm_shuffle_ps(sw_all, sw_all, 0x55));
            __m512 sc2 = _mm512_broadcast_f32x4(_mm_shuffle_ps(sw_all, sw_all, 0xAA));
            __m512 sc3 = _mm512_broadcast_f32x4(_mm_shuffle_ps(sw_all, sw_all, 0xFF));

            // FMADD accumulate
            acc0 = _mm512_fmadd_ps(sc0, fdot0, acc0);
            acc1 = _mm512_fmadd_ps(sc1, fdot1, acc1);
            acc2 = _mm512_fmadd_ps(sc2, fdot2, acc2);
            acc3 = _mm512_fmadd_ps(sc3, fdot3, acc3);
        }

        // Horizontal sum (16 floats → 1 scalar) — AVX-512 has hsum_ps512
        o_data[n + 0] = hsum_ps512(acc0);
        if (rows > 1) o_data[n + 1] = hsum_ps512(acc1);
        if (rows > 2) o_data[n + 2] = hsum_ps512(acc2);
        if (rows > 3) o_data[n + 3] = hsum_ps512(acc3);
    }
}

// ============================================================================
// Q4_0 AVX-512 VNNI Prefill GEMM Tile (RM=4 × RN=4)
// ============================================================================
// 4 activation rows × 4 weight columns.
// Weight decoding and scale conversion done once per block, shared across 4 act rows.
// VNNI dpbusd replaces updot chain for each (act_row, weight_col) pair.

static FORGE_AVX512_VNNI_ATTR inline void gemm_q4_0_tile_4x4_vnni(
    const block_q8_0_act* q8_act,    // [M * nb] pre-quantized activation
    const uint8_t* w_data,           // Q4_0 weight base pointer
    float* o_data,                    // [M, N] output
    int64_t nb,                       // number of blocks per row
    int64_t m0, int64_t m_rows,      // activation rows [m0, m0+m_rows), ≤ 4
    int64_t n0, int64_t n_cols,      // weight cols [n0, n0+n_cols), ≤ 4
    int64_t N)                        // output row stride
{
    constexpr int BLOCK_BYTES = 18;

    // Accumulators: C[i][j] for act row i, weight col j
    __m512 C00 = _mm512_setzero_ps(), C01 = _mm512_setzero_ps(),
           C02 = _mm512_setzero_ps(), C03 = _mm512_setzero_ps();
    __m512 C10 = _mm512_setzero_ps(), C11 = _mm512_setzero_ps(),
           C12 = _mm512_setzero_ps(), C13 = _mm512_setzero_ps();
    __m512 C20 = _mm512_setzero_ps(), C21 = _mm512_setzero_ps(),
           C22 = _mm512_setzero_ps(), C23 = _mm512_setzero_ps();
    __m512 C30 = _mm512_setzero_ps(), C31 = _mm512_setzero_ps(),
           C32 = _mm512_setzero_ps(), C33 = _mm512_setzero_ps();

    // Weight row pointers
    const uint8_t* w0 = w_data + (size_t)(n0 + 0) * nb * BLOCK_BYTES;
    const uint8_t* w1 = (n_cols > 1) ? w_data + (size_t)(n0 + 1) * nb * BLOCK_BYTES : nullptr;
    const uint8_t* w2 = (n_cols > 2) ? w_data + (size_t)(n0 + 2) * nb * BLOCK_BYTES : nullptr;
    const uint8_t* w3 = (n_cols > 3) ? w_data + (size_t)(n0 + 3) * nb * BLOCK_BYTES : nullptr;

    for (int64_t l = 0; l < nb; ++l) {
        // Prefetch weight blocks
        if (l + 6 < nb) {
            _mm_prefetch((const char*)(w0 + (l + 6) * BLOCK_BYTES), _MM_HINT_T1);
            if (n_cols > 1) _mm_prefetch((const char*)(w1 + (l + 6) * BLOCK_BYTES), _MM_HINT_T1);
            if (n_cols > 2) _mm_prefetch((const char*)(w2 + (l + 6) * BLOCK_BYTES), _MM_HINT_T1);
            if (n_cols > 3) _mm_prefetch((const char*)(w3 + (l + 6) * BLOCK_BYTES), _MM_HINT_T1);
        }

        // Load 4 weight blocks (Q4_0 → 32 signed int8)
        __m256i wvec0 = BlockLoader<block_q4_0_tag>::load(w0 + l * BLOCK_BYTES);
        __m256i wvec1 = (n_cols > 1) ? BlockLoader<block_q4_0_tag>::load(w1 + l * BLOCK_BYTES) : _mm256_setzero_si256();
        __m256i wvec2 = (n_cols > 2) ? BlockLoader<block_q4_0_tag>::load(w2 + l * BLOCK_BYTES) : _mm256_setzero_si256();
        __m256i wvec3 = (n_cols > 3) ? BlockLoader<block_q4_0_tag>::load(w3 + l * BLOCK_BYTES) : _mm256_setzero_si256();

        // VNNI: |w| and sign-adjusted activation per weight col
        __m256i abs_w0 = _mm256_sign_epi8(wvec0, wvec0);
        __m256i abs_w1 = (n_cols > 1) ? _mm256_sign_epi8(wvec1, wvec1) : _mm256_setzero_si256();
        __m256i abs_w2 = (n_cols > 2) ? _mm256_sign_epi8(wvec2, wvec2) : _mm256_setzero_si256();
        __m256i abs_w3 = (n_cols > 3) ? _mm256_sign_epi8(wvec3, wvec3) : _mm256_setzero_si256();

        // Zero-extend to 512-bit
        __m512i abs_w0_512 = _mm512_castsi256_si512(abs_w0);
        __m512i abs_w1_512 = _mm512_castsi256_si512(abs_w1);
        __m512i abs_w2_512 = _mm512_castsi256_si512(abs_w2);
        __m512i abs_w3_512 = _mm512_castsi256_si512(abs_w3);

        // F16C: pack 4 fp16 weight scales → 4 fp32
        uint64_t packed_scales = 0;
        uint16_t d0, d1 = 0, d2 = 0, d3 = 0;
        memcpy(&d0, w0 + l * BLOCK_BYTES, 2);
        if (n_cols > 1) memcpy(&d1, w1 + l * BLOCK_BYTES, 2);
        if (n_cols > 2) memcpy(&d2, w2 + l * BLOCK_BYTES, 2);
        if (n_cols > 3) memcpy(&d3, w3 + l * BLOCK_BYTES, 2);
        packed_scales = (uint64_t)d0 | ((uint64_t)d1 << 16) | ((uint64_t)d2 << 32) | ((uint64_t)d3 << 48);
        __m128 sw_f16 = _mm_cvtph_ps(_mm_set_epi64x(0, packed_scales));

        // --- Activation row 0 ---
        if (m_rows > 0) {
            const block_q8_0_act* a0 = q8_act + (m0 + 0) * nb;
            __m256i avec0 = BlockLoader<block_q8_0_act>::load(a0 + l);
            float act_s = a0[l].d;

            // Combined scale = weight_scale * act_scale, broadcast to __m128
            __m128 sw_all = _mm_mul_ps(sw_f16, _mm_set1_ps(act_s));
            // Broadcast to __m512
            __m512 dvec = _mm512_broadcast_f32x4(sw_all);

            // VNNI sign-adjust activation per weight col
            __m256i sa0 = _mm256_sign_epi8(avec0, wvec0);
            __m512i sa0_512 = _mm512_castsi256_si512(sa0);

            __m512i dot0 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), abs_w0_512, sa0_512);
            __m512 fdot0 = _mm512_cvtepi32_ps(dot0);
            C00 = _mm512_fmadd_ps(_mm512_shuffle_ps(dvec, dvec, 0x00), fdot0, C00);

            if (n_cols > 1) {
                __m256i sa1 = _mm256_sign_epi8(avec0, wvec1);
                __m512i sa1_512 = _mm512_castsi256_si512(sa1);
                __m512i dot1 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), abs_w1_512, sa1_512);
                __m512 fdot1 = _mm512_cvtepi32_ps(dot1);
                C01 = _mm512_fmadd_ps(_mm512_shuffle_ps(dvec, dvec, 0x55), fdot1, C01);
            }
            if (n_cols > 2) {
                __m256i sa2 = _mm256_sign_epi8(avec0, wvec2);
                __m512i sa2_512 = _mm512_castsi256_si512(sa2);
                __m512i dot2 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), abs_w2_512, sa2_512);
                __m512 fdot2 = _mm512_cvtepi32_ps(dot2);
                C02 = _mm512_fmadd_ps(_mm512_shuffle_ps(dvec, dvec, 0xAA), fdot2, C02);
            }
            if (n_cols > 3) {
                __m256i sa3 = _mm256_sign_epi8(avec0, wvec3);
                __m512i sa3_512 = _mm512_castsi256_si512(sa3);
                __m512i dot3 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), abs_w3_512, sa3_512);
                __m512 fdot3 = _mm512_cvtepi32_ps(dot3);
                C03 = _mm512_fmadd_ps(_mm512_shuffle_ps(dvec, dvec, 0xFF), fdot3, C03);
            }
        }

        // --- Activation row 1 ---
        if (m_rows > 1) {
            const block_q8_0_act* a1 = q8_act + (m0 + 1) * nb;
            __m256i avec1 = BlockLoader<block_q8_0_act>::load(a1 + l);
            __m128 sw_all1 = _mm_mul_ps(sw_f16, _mm_set1_ps(a1[l].d));
            __m512 dvec1 = _mm512_broadcast_f32x4(sw_all1);

            __m256i sa0 = _mm256_sign_epi8(avec1, wvec0);
            __m512i dot0 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), abs_w0_512,
                                                 _mm512_castsi256_si512(sa0));
            C10 = _mm512_fmadd_ps(_mm512_shuffle_ps(dvec1, dvec1, 0x00),
                                   _mm512_cvtepi32_ps(dot0), C10);

            if (n_cols > 1) {
                __m256i sa1 = _mm256_sign_epi8(avec1, wvec1);
                __m512i dot1 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), abs_w1_512,
                                                     _mm512_castsi256_si512(sa1));
                C11 = _mm512_fmadd_ps(_mm512_shuffle_ps(dvec1, dvec1, 0x55),
                                       _mm512_cvtepi32_ps(dot1), C11);
            }
            if (n_cols > 2) {
                __m256i sa2 = _mm256_sign_epi8(avec1, wvec2);
                __m512i dot2 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), abs_w2_512,
                                                     _mm512_castsi256_si512(sa2));
                C12 = _mm512_fmadd_ps(_mm512_shuffle_ps(dvec1, dvec1, 0xAA),
                                       _mm512_cvtepi32_ps(dot2), C12);
            }
            if (n_cols > 3) {
                __m256i sa3 = _mm256_sign_epi8(avec1, wvec3);
                __m512i dot3 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), abs_w3_512,
                                                     _mm512_castsi256_si512(sa3));
                C13 = _mm512_fmadd_ps(_mm512_shuffle_ps(dvec1, dvec1, 0xFF),
                                       _mm512_cvtepi32_ps(dot3), C13);
            }
        }

        // --- Activation row 2 ---
        if (m_rows > 2) {
            const block_q8_0_act* a2 = q8_act + (m0 + 2) * nb;
            __m256i avec2 = BlockLoader<block_q8_0_act>::load(a2 + l);
            __m128 sw_all2 = _mm_mul_ps(sw_f16, _mm_set1_ps(a2[l].d));
            __m512 dvec2 = _mm512_broadcast_f32x4(sw_all2);

            __m256i sa0 = _mm256_sign_epi8(avec2, wvec0);
            __m512i dot0 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), abs_w0_512,
                                                 _mm512_castsi256_si512(sa0));
            C20 = _mm512_fmadd_ps(_mm512_shuffle_ps(dvec2, dvec2, 0x00),
                                   _mm512_cvtepi32_ps(dot0), C20);

            if (n_cols > 1) {
                __m256i sa1 = _mm256_sign_epi8(avec2, wvec1);
                __m512i dot1 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), abs_w1_512,
                                                     _mm512_castsi256_si512(sa1));
                C21 = _mm512_fmadd_ps(_mm512_shuffle_ps(dvec2, dvec2, 0x55),
                                       _mm512_cvtepi32_ps(dot1), C21);
            }
            if (n_cols > 2) {
                __m256i sa2 = _mm256_sign_epi8(avec2, wvec2);
                __m512i dot2 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), abs_w2_512,
                                                     _mm512_castsi256_si512(sa2));
                C22 = _mm512_fmadd_ps(_mm512_shuffle_ps(dvec2, dvec2, 0xAA),
                                       _mm512_cvtepi32_ps(dot2), C22);
            }
            if (n_cols > 3) {
                __m256i sa3 = _mm256_sign_epi8(avec2, wvec3);
                __m512i dot3 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), abs_w3_512,
                                                     _mm512_castsi256_si512(sa3));
                C23 = _mm512_fmadd_ps(_mm512_shuffle_ps(dvec2, dvec2, 0xFF),
                                       _mm512_cvtepi32_ps(dot3), C23);
            }
        }

        // --- Activation row 3 ---
        if (m_rows > 3) {
            const block_q8_0_act* a3 = q8_act + (m0 + 3) * nb;
            __m256i avec3 = BlockLoader<block_q8_0_act>::load(a3 + l);
            __m128 sw_all3 = _mm_mul_ps(sw_f16, _mm_set1_ps(a3[l].d));
            __m512 dvec3 = _mm512_broadcast_f32x4(sw_all3);

            __m256i sa0 = _mm256_sign_epi8(avec3, wvec0);
            __m512i dot0 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), abs_w0_512,
                                                 _mm512_castsi256_si512(sa0));
            C30 = _mm512_fmadd_ps(_mm512_shuffle_ps(dvec3, dvec3, 0x00),
                                   _mm512_cvtepi32_ps(dot0), C30);

            if (n_cols > 1) {
                __m256i sa1 = _mm256_sign_epi8(avec3, wvec1);
                __m512i dot1 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), abs_w1_512,
                                                     _mm512_castsi256_si512(sa1));
                C31 = _mm512_fmadd_ps(_mm512_shuffle_ps(dvec3, dvec3, 0x55),
                                       _mm512_cvtepi32_ps(dot1), C31);
            }
            if (n_cols > 2) {
                __m256i sa2 = _mm256_sign_epi8(avec3, wvec2);
                __m512i dot2 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), abs_w2_512,
                                                     _mm512_castsi256_si512(sa2));
                C32 = _mm512_fmadd_ps(_mm512_shuffle_ps(dvec3, dvec3, 0xAA),
                                       _mm512_cvtepi32_ps(dot2), C32);
            }
            if (n_cols > 3) {
                __m256i sa3 = _mm256_sign_epi8(avec3, wvec3);
                __m512i dot3 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), abs_w3_512,
                                                     _mm512_castsi256_si512(sa3));
                C33 = _mm512_fmadd_ps(_mm512_shuffle_ps(dvec3, dvec3, 0xFF),
                                       _mm512_cvtepi32_ps(dot3), C33);
            }
        }
    }

    // Horizontal sum and write outputs
    if (m_rows > 0) {
        o_data[(m0 + 0) * N + n0 + 0] = hsum_ps512(C00);
        if (n_cols > 1) o_data[(m0 + 0) * N + n0 + 1] = hsum_ps512(C01);
        if (n_cols > 2) o_data[(m0 + 0) * N + n0 + 2] = hsum_ps512(C02);
        if (n_cols > 3) o_data[(m0 + 0) * N + n0 + 3] = hsum_ps512(C03);
    }
    if (m_rows > 1) {
        o_data[(m0 + 1) * N + n0 + 0] = hsum_ps512(C10);
        if (n_cols > 1) o_data[(m0 + 1) * N + n0 + 1] = hsum_ps512(C11);
        if (n_cols > 2) o_data[(m0 + 1) * N + n0 + 2] = hsum_ps512(C12);
        if (n_cols > 3) o_data[(m0 + 1) * N + n0 + 3] = hsum_ps512(C13);
    }
    if (m_rows > 2) {
        o_data[(m0 + 2) * N + n0 + 0] = hsum_ps512(C20);
        if (n_cols > 1) o_data[(m0 + 2) * N + n0 + 1] = hsum_ps512(C21);
        if (n_cols > 2) o_data[(m0 + 2) * N + n0 + 2] = hsum_ps512(C22);
        if (n_cols > 3) o_data[(m0 + 2) * N + n0 + 3] = hsum_ps512(C23);
    }
    if (m_rows > 3) {
        o_data[(m0 + 3) * N + n0 + 0] = hsum_ps512(C30);
        if (n_cols > 1) o_data[(m0 + 3) * N + n0 + 1] = hsum_ps512(C31);
        if (n_cols > 2) o_data[(m0 + 3) * N + n0 + 2] = hsum_ps512(C32);
        if (n_cols > 3) o_data[(m0 + 3) * N + n0 + 3] = hsum_ps512(C33);
    }
}

// ============================================================================
// Q4_0 AVX-512 VNNI Prefill Batch GEMM (M>1)
// ============================================================================

static FORGE_AVX512_VNNI_ATTR void gemm_q4_0_batch_vnni(
    const float* a_data,
    const uint8_t* w_data,
    float* o_data,
    int64_t M, int64_t K, int64_t N)
{
    constexpr int RM = 4;
    constexpr int RN = 4;
    constexpr int BLOCK_SIZE = 32;
    constexpr int BLOCK_BYTES = 18;
    const int64_t nb = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Quantize all M activation rows to Q8_0_act
    scratch_vec<block_q8_0_act> q8_act(M * nb);
    for (int64_t m = 0; m < M; ++m)
        quantize_row_q8_0_act(a_data + m * K, q8_act.data() + m * nb, (int)K);

    // Tile-level work-stealing
    const int64_t m_tiles = (M + RM - 1) / RM;
    const int64_t n_tiles = (N + RN - 1) / RN;
    const int64_t total_tiles = m_tiles * n_tiles;

    std::atomic<int64_t> next_tile{0};

    #pragma omp parallel
    {
        while (true) {
            int64_t tile = next_tile.fetch_add(1, std::memory_order_relaxed);
            if (tile >= total_tiles) break;

            int64_t mt = tile / n_tiles;
            int64_t nt = tile % n_tiles;

            int64_t m0 = mt * RM;
            int64_t n0 = nt * RN;
            int64_t m_rows = std::min(m0 + RM, M) - m0;
            int64_t n_cols = std::min(n0 + RN, N) - n0;

            gemm_q4_0_tile_4x4_vnni(q8_act.data(), w_data, o_data, nb,
                                      m0, m_rows, n0, n_cols, N);
        }
    }
}

#endif  // USE_AVX512_VNNI

}  // namespace cpu
}  // namespace forge
