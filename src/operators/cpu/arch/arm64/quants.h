#pragma once
// ARM64 NEON I-quant kernel implementations (FORGE_ARCH_ARM64 + USE_NEON).
// NEON (ARMv8-A ASIMD) translations of the x86 IQ quant dot product and GEMV
// kernels from arch/x86/quants.h.
//
// IQ2_S / IQ2_XS / IQ3_S / IQ4_NL fused dot-product and GEMV kernels.
//
// NEON 128-bit registers are half-width of AVX2 256-bit registers.
// All x86 256-bit operations are split into two 128-bit halves.
//
// #ifdef USE_DOTPROD enables vdotq_s32 (ARMv8.2+dotprod, Apple M1+).
// Fallback path uses vmull + vpadd for baseline ARMv8-A NEON.

#ifdef USE_NEON
#include <arm_neon.h>
#endif
#include <cstring>
#include <cmath>
#include <vector>

#include "../../common/quant_tables.h"
#include "../../common/quant_helpers.h"
#include "../../common/scalar.h"
#include "forge/types.h"

namespace forge {
namespace cpu {

#ifdef USE_NEON

// ============================================================================
// Internal helpers — NEON equivalents of x86 intrinsics
// ============================================================================

// Emulate _mm256_maddubs_epi16(unsigned, signed) for 16 elements.
// Takes 16 unsigned grid bytes and 16 signed q8 bytes, produces 8 int16
// pairwise sums: result[i] = u8(grid[2i])*s8(q8[2i]) + u8(grid[2i+1])*s8(q8[2i+1]).
//
// Grid values from I-quant tables are small (0..15 or 0..3), so they fit in
// signed positive range and reinterpret_cast to signed is safe.
static inline int16x8_t maddubs_u8_s8_16_neon(uint8x16_t u, int8x16_t s) {
    // Element-wise signed×signed multiply: 8 products → int16x8 each
    int16x8_t prod0 = vmull_s8(vreinterpret_s8_u8(vget_low_u8(u)),
                                vget_low_s8(s));
    int16x8_t prod1 = vmull_s8(vreinterpret_s8_u8(vget_high_u8(u)),
                                vget_high_s8(s));
    // Pairwise sum: vpaddq_s16(a,b) → [a0+a1,a2+a3,a4+a5,a6+a7, b0+b1,b2+b3,b4+b5,b6+b7]
    int16x8_t psum0 = vpaddq_s16(prod0, prod0);
    int16x8_t psum1 = vpaddq_s16(prod1, prod1);
    // Take lower halves (the valid pairwise sums)
    return vcombine_s16(vget_low_s16(psum0), vget_low_s16(psum1));
}

#ifdef USE_DOTPROD
// Fast path: maddubs_u8_s8 for 16 elements using vdotq_s32.
// vdotq_s32(acc, u8_as_s8, s8) computes 4-element dot products (not pairwise).
// The total sum across all 16 elements is identical to the pairwise approach.
// Returns 4 partial sums in an int32x4_t.
static inline int32x4_t maddubs_dotprod_16_neon(uint8x16_t u, int8x16_t s) {
    int32x4_t acc = vdupq_n_s32(0);
    return vdotq_s32(acc, vreinterpretq_s8_u8(u), s);
}
#endif  // USE_DOTPROD

// ============================================================================
// IQ4_NL x Q8_K fused dot product (NEON)
// IQ4_NL block: 18 bytes per 32 elements = d[2] fp16 + qs[16] nibbles
// 8 sub-blocks per 256-element super-block
// ============================================================================

static inline float dot_iq4_nl_q8_K_neon(const uint8_t* iq4nl_row,
                                         const cpu::block_q8_K* q8, int nb) {
    constexpr int QK_K = 256;
    constexpr int IQ4_NL_BLOCK_SIZE = 18;
    constexpr int IQ4_NL_BLOCK_EL = 32;

    // Load the 16-entry LUT once
    const int8x16_t kvalues = vld1q_s8(forge::ops::kvalues_iq4nl);
    const uint8x16_t nibble_mask = vdupq_n_u8(0x0F);

    float acc = 0.0f;

    for (int bi = 0; bi < nb; ++bi) {
        const cpu::block_q8_K* y = q8 + bi;
        const float d_q8 = y->d;
        const int8_t* q8d = y->qs;

        const uint8_t* block_ptr = iq4nl_row +
            (size_t)bi * (QK_K / IQ4_NL_BLOCK_EL) * IQ4_NL_BLOCK_SIZE;
        float block_acc = 0.0f;

        for (int sb = 0; sb < QK_K / IQ4_NL_BLOCK_EL; ++sb) {
            float d = cpu::fp16_to_float_scalar(
                *reinterpret_cast<const uint16_t*>(block_ptr));
            const uint8_t* qs = block_ptr + 2;

            // Load 16 nibble-packed bytes, extract lo/hi nibbles
            uint8x16_t qs_vec = vld1q_u8(qs);
            uint8x16_t lo_nibbles = vandq_u8(qs_vec, nibble_mask);       // [e0, e2, ..., e30]
            uint8x16_t hi_nibbles = vshrq_n_u8(qs_vec, 4);               // [e1, e3, ..., e31]

            // LUT: 16-way table lookup, each nibble 0..15 → kvalues[nibble]
            int8x16_t vals_low  = vqtbl1q_s8(kvalues, lo_nibbles);      // 16 int8 for even indices
            int8x16_t vals_high = vqtbl1q_s8(kvalues, hi_nibbles);      // 16 int8 for odd indices

            // Load q8: two halves of 16 int8 each
            int8x16_t q8_low  = vld1q_s8(q8d + sb * 32);
            int8x16_t q8_high = vld1q_s8(q8d + sb * 32 + 16);

            int sum = 0;
#ifdef USE_DOTPROD
            {
                int32x4_t acc_dp = vdotq_s32(vdupq_n_s32(0),
                    vals_low, q8_low);
                acc_dp = vdotq_s32(acc_dp,
                    vals_high, q8_high);
                sum = vaddvq_s32(acc_dp);
            }
#else
            {
                // vals_low (16 i8) × q8_low (16 i8) → widen to i16, pairwise madd
                int16x8_t vls0 = vmovl_s8(vget_low_s8(vals_low));
                int16x8_t vls1 = vmovl_s8(vget_high_s8(vals_low));
                int16x8_t qls0 = vmovl_s8(vget_low_s8(q8_low));
                int16x8_t qls1 = vmovl_s8(vget_high_s8(q8_low));

                int32x4_t prod00 = vmull_s16(vget_low_s16(vls0), vget_low_s16(qls0));
                int32x4_t prod01 = vmull_s16(vget_high_s16(vls0), vget_high_s16(qls0));
                int32x4_t prod02 = vmull_s16(vget_low_s16(vls1), vget_low_s16(qls1));
                int32x4_t prod03 = vmull_s16(vget_high_s16(vls1), vget_high_s16(qls1));

                int32x4_t sum0 = vpaddq_s32(prod00, prod01);
                int32x4_t sum1 = vpaddq_s32(prod02, prod03);
                int32x4_t p_lo = vaddq_s32(sum0, sum1);

                // vals_high (16 i8) × q8_high (16 i8)
                int16x8_t vhs0 = vmovl_s8(vget_low_s8(vals_high));
                int16x8_t vhs1 = vmovl_s8(vget_high_s8(vals_high));
                int16x8_t qhs0 = vmovl_s8(vget_low_s8(q8_high));
                int16x8_t qhs1 = vmovl_s8(vget_high_s8(q8_high));

                int32x4_t prod10 = vmull_s16(vget_low_s16(vhs0), vget_low_s16(qhs0));
                int32x4_t prod11 = vmull_s16(vget_high_s16(vhs0), vget_high_s16(qhs0));
                int32x4_t prod12 = vmull_s16(vget_low_s16(vhs1), vget_low_s16(qhs1));
                int32x4_t prod13 = vmull_s16(vget_high_s16(vhs1), vget_high_s16(qhs1));

                int32x4_t sum2 = vpaddq_s32(prod10, prod11);
                int32x4_t sum3 = vpaddq_s32(prod12, prod13);
                int32x4_t p_hi = vaddq_s32(sum2, sum3);

                sum = vaddvq_s32(p_lo) + vaddvq_s32(p_hi);
            }
#endif  // USE_DOTPROD

            block_acc += d * (float)sum;
            block_ptr += IQ4_NL_BLOCK_SIZE;
        }

        acc += d_q8 * block_acc;
    }

    return acc;
}

// ============================================================================
// IQ4_NL GEMV: out = a * w^T  (M×K × N×K → M×N)
// ============================================================================

static void gemv_iq4_nl_q8k_transB_neon(const float* a, const uint8_t* w, float* out,
                                        int M, int K, int N) {
    constexpr int QK_K = 256;
    constexpr int IQ4_NL_BLOCK_SIZE = 18;
    constexpr int IQ4_NL_BLOCK_EL = 32;
    const int nb = (K + QK_K - 1) / QK_K;
    const size_t row_bytes = (size_t)(K + IQ4_NL_BLOCK_EL - 1) / IQ4_NL_BLOCK_EL * IQ4_NL_BLOCK_SIZE;

    if (M == 1) {
        std::vector<cpu::block_q8_K> q8_buf(nb);
        cpu::quantize_row_q8_K(a, q8_buf.data(), K);

#pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const uint8_t* row = w + (size_t)n * row_bytes;
            out[n] = dot_iq4_nl_q8_K_neon(row, q8_buf.data(), nb);
        }
    } else {
        std::vector<cpu::block_q8_K> q8_all((size_t)M * nb);
        for (int m = 0; m < M; ++m)
            cpu::quantize_row_q8_K(a + (size_t)m * K, q8_all.data() + (size_t)m * nb, K);

#pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const uint8_t* row = w + (size_t)n * row_bytes;
            for (int m = 0; m < M; ++m)
                out[(size_t)m * N + n] = dot_iq4_nl_q8_K_neon(row,
                    q8_all.data() + (size_t)m * nb, nb);
        }
    }
}

// ============================================================================
// IQ2_XS x Q8_K fused dot product (NEON)
// IQ2_XS block: 74 bytes = d[2] fp16 + qs[64] uint16 + scales[8]
// 8 ib32 groups, each with 4 lanes × 8 elements
// ============================================================================

static inline float dot_iq2_xs_q8_K_neon(const uint8_t* iq2xs_row,
                                         const cpu::block_q8_K* q8, int nb) {
    constexpr int QK_K = 256;
    constexpr int IQ2_XS_BLOCK_SIZE = 74;

    float acc = 0.0f;

    for (int bi = 0; bi < nb; ++bi) {
        const uint8_t* block_ptr = iq2xs_row + (size_t)bi * IQ2_XS_BLOCK_SIZE;
        float d = cpu::fp16_to_float_scalar(
            *reinterpret_cast<const uint16_t*>(block_ptr));
        const uint16_t* qs = reinterpret_cast<const uint16_t*>(block_ptr + 2);
        const uint8_t* scales = block_ptr + 2 + 64;

        const cpu::block_q8_K* y = q8 + bi;
        const float d_q8 = y->d;
        const int8_t* q8d = y->qs;

        float block_acc = 0.0f;

        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            const float db0_f = (0.5f + (scales[ib32] & 0xf)) * 0.25f;
            const float db1_f = (0.5f + (scales[ib32] >> 4)) * 0.25f;

            // Build grid_vals[32] and sign_mask[32] for this ib32 group
            alignas(16) uint8_t grid_vals[32];
            alignas(16) uint8_t sign_mask[32];
            for (int l = 0; l < 4; ++l) {
                const uint16_t qs_u16 = qs[4 * ib32 + l];
                // Low 9 bits index iq2xs_grid[512] (8 bytes each)
                memcpy(grid_vals + l * 8,
                       &forge::ops::iq2xs_grid[qs_u16 & 511], 8);
                // Upper 7 bits index ksigns_iq2xs[128]
                const uint8_t signs_byte = forge::ops::ksigns_iq2xs[qs_u16 >> 9];
                for (int j = 0; j < 8; ++j)
                    sign_mask[l * 8 + j] =
                        (signs_byte & forge::ops::kmask_iq2xs[j]) ? 0xFF : 0x00;
            }

            // Load two 16-byte halves (NEON = 128-bit)
            uint8x16_t grid_lo = vld1q_u8(grid_vals);
            uint8x16_t grid_hi = vld1q_u8(grid_vals + 16);
            uint8x16_t sign_lo = vld1q_u8(sign_mask);
            uint8x16_t sign_hi = vld1q_u8(sign_mask + 16);

            // Load q8, two 16-byte halves
            int8x16_t q8_lo = vld1q_s8(q8d + ib32 * 32);
            int8x16_t q8_hi = vld1q_s8(q8d + ib32 * 32 + 16);

            // Apply sign: negate q8 where sign_mask byte = 0xFF
            // vbslq_s8(mask, a, b): selects a if mask bit=1, b if mask bit=0
            int8x16_t q8_signed_lo = vbslq_s8(sign_lo, vnegq_s8(q8_lo), q8_lo);
            int8x16_t q8_signed_hi = vbslq_s8(sign_hi, vnegq_s8(q8_hi), q8_hi);

            int s_l01 = 0, s_l23 = 0;
#ifdef USE_DOTPROD
            {
                // Half 0: sum of first 16 elements (lanes 0-1, scale db0_f)
                int32x4_t acc0 = maddubs_dotprod_16_neon(grid_lo, q8_signed_lo);
                s_l01 = vaddvq_s32(acc0);

                // Half 1: sum of next 16 elements (lanes 2-3, scale db1_f)
                int32x4_t acc1 = maddubs_dotprod_16_neon(grid_hi, q8_signed_hi);
                s_l23 = vaddvq_s32(acc1);
            }
#else
            {
                // maddubs: unsigned grid × signed q8 → 8 i16 pairwise sums each half
                int16x8_t p16_lo = maddubs_u8_s8_16_neon(grid_lo, q8_signed_lo);
                int16x8_t p16_hi = maddubs_u8_s8_16_neon(grid_hi, q8_signed_hi);

                // madd with ones: pairwise sum of i16 → i32
                // Each half has 8 i16, producing 4 i32 via pairwise add
                // But we need to separate lo and hi into their 4-element groups
                int32x4_t w0 = vmovl_s16(vget_low_s16(p16_lo));
                int32x4_t w1 = vmovl_s16(vget_high_s16(p16_lo));
                int32x4_t w2 = vmovl_s16(vget_low_s16(p16_hi));
                int32x4_t w3 = vmovl_s16(vget_high_s16(p16_hi));

                // Pairwise sums for lane groups
                int32x4_t ps_lo0 = vpaddq_s32(w0, w1);  // 4 sums from first 16 elements
                // ps_lo0 = [s0+s1, s2+s3, s4+s5, s6+s7]
                // Where each si is a pairwise sum from maddubs
                // Need to further reduce: hadd equivalent
                // x86 hadd(lo, hi) → [lo0+lo1, lo2+lo3, hi0+hi1, hi2+hi3]
                // Then hadd(h,h) → [lo0+lo1+lo2+lo3, hi0+hi1+hi2+hi3, ...]
                int32x4_t ps_hi0 = vpaddq_s32(w2, w3);  // 4 sums from next 16 elements

                // Combine and reduce: h = hadd(ps_lo0, ps_hi0)
                int32x4_t h = vpaddq_s32(ps_lo0, ps_hi0);
                // h = [ps_lo0[0]+ps_lo0[1], ps_lo0[2]+ps_lo0[3],
                //      ps_hi0[0]+ps_hi0[1], ps_hi0[2]+ps_hi0[3]]
                // Second hadd: h = hadd(h, h)
                int32x4_t h2 = vpaddq_s32(h, h);
                // h2 = [h[0]+h[1], h[2]+h[3], h[0]+h[1], h[2]+h[3]]
                s_l01 = vgetq_lane_s32(h2, 0);  // sum of first 16 elements
                s_l23 = vgetq_lane_s32(h2, 1);  // sum of next 16 elements
            }
#endif  // USE_DOTPROD

            block_acc += (float)s_l01 * db0_f + (float)s_l23 * db1_f;
        }

        acc += d * d_q8 * block_acc;
    }

    return acc;
}

// ============================================================================
// IQ2_XS GEMV
// ============================================================================

static void gemv_iq2_xs_q8k_transB_neon(const float* a, const uint8_t* w, float* out,
                                        int M, int K, int N) {
    constexpr int QK_K = 256;
    constexpr int IQ2_XS_BLOCK_SIZE = 74;
    const int nb = (K + QK_K - 1) / QK_K;

    if (M == 1) {
        std::vector<cpu::block_q8_K> q8_buf(nb);
        cpu::quantize_row_q8_K(a, q8_buf.data(), K);

#pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const uint8_t* row = w + (size_t)n * nb * IQ2_XS_BLOCK_SIZE;
            out[n] = dot_iq2_xs_q8_K_neon(row, q8_buf.data(), nb);
        }
    } else {
        std::vector<cpu::block_q8_K> q8_all((size_t)M * nb);
        for (int m = 0; m < M; ++m)
            cpu::quantize_row_q8_K(a + (size_t)m * K, q8_all.data() + (size_t)m * nb, K);

#pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const uint8_t* row = w + (size_t)n * nb * IQ2_XS_BLOCK_SIZE;
            for (int m = 0; m < M; ++m)
                out[(size_t)m * N + n] = dot_iq2_xs_q8_K_neon(row,
                    q8_all.data() + (size_t)m * nb, nb);
        }
    }
}

// ============================================================================
// IQ3_S x Q8_K fused dot product (NEON)
// IQ3_S block: 110 bytes = d[2] + qs[64] + qh[8] + signs[32] + scales[4]
// ============================================================================

static inline float dot_iq3_s_q8_K_neon(const uint8_t* iq3s_row,
                                        const cpu::block_q8_K* q8, int nb) {
    constexpr int QK_K = 256;
    constexpr int IQ3_S_BLOCK_SIZE = 110;

    float acc = 0.0f;

    for (int bi = 0; bi < nb; ++bi) {
        const uint8_t* block_ptr = iq3s_row + (size_t)bi * IQ3_S_BLOCK_SIZE;
        float d = cpu::fp16_to_float_scalar(
            *reinterpret_cast<const uint16_t*>(block_ptr));
        const uint8_t* qs    = block_ptr + 2;
        const uint8_t* qh    = block_ptr + 2 + 64;
        const uint8_t* signs = block_ptr + 2 + 64 + 8;
        const uint8_t* scales = block_ptr + 2 + 64 + 8 + 32;

        const cpu::block_q8_K* y = q8 + bi;
        const float d_q8 = y->d;
        const int8_t* q8d = y->qs;

        float block_acc = 0.0f;

        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const float db1_f = (float)(1 + 2 * (scales[ib32 / 2] & 0xf));
            const float db2_f = (float)(1 + 2 * (scales[ib32 / 2] >> 4));

            for (int half = 0; half < 2; ++half) {
                const float db_f = (half == 0) ? db1_f : db2_f;

                // Build grid_vals[32] and sign_mask[32] for this half
                alignas(16) uint8_t grid_vals[32];
                alignas(16) uint8_t sign_mask_arr[32];
                for (int l = 0; l < 4; ++l) {
                    const int idx1 = qs[2 * l + 0] |
                        ((qh[half] << (8 - 2 * l)) & 256);
                    const int idx2 = qs[2 * l + 1] |
                        ((qh[half] << (7 - 2 * l)) & 256);
                    // iq3s_grid entries are 4 bytes each
                    memcpy(grid_vals + l * 8 + 0,
                           &forge::ops::iq3s_grid[idx1], 4);
                    memcpy(grid_vals + l * 8 + 4,
                           &forge::ops::iq3s_grid[idx2], 4);
                    const uint8_t s = signs[l];
                    for (int j = 0; j < 8; ++j)
                        sign_mask_arr[l * 8 + j] =
                            (s & forge::ops::kmask_iq2xs[j]) ? 0xFF : 0x00;
                }

                // Load two 16-byte halves
                uint8x16_t grid_lo = vld1q_u8(grid_vals);
                uint8x16_t grid_hi = vld1q_u8(grid_vals + 16);
                uint8x16_t sign_lo = vld1q_u8(sign_mask_arr);
                uint8x16_t sign_hi = vld1q_u8(sign_mask_arr + 16);

                // Load q8 for this half
                int8x16_t q8_lo = vld1q_s8(q8d + ib32 * 32 + half * 32);
                int8x16_t q8_hi = vld1q_s8(q8d + ib32 * 32 + half * 32 + 16);

                // Sign adjustment
                int8x16_t q8_signed_lo = vbslq_s8(sign_lo, vnegq_s8(q8_lo), q8_lo);
                int8x16_t q8_signed_hi = vbslq_s8(sign_hi, vnegq_s8(q8_hi), q8_hi);

                int s_all = 0;
#ifdef USE_DOTPROD
                {
                    int32x4_t acc_dp = maddubs_dotprod_16_neon(grid_lo, q8_signed_lo);
                    acc_dp = vdotq_s32(acc_dp,
                        vreinterpretq_s8_u8(grid_hi), q8_signed_hi);
                    s_all = vaddvq_s32(acc_dp);
                }
#else
                {
                    int16x8_t p16_lo = maddubs_u8_s8_16_neon(grid_lo, q8_signed_lo);
                    int16x8_t p16_hi = maddubs_u8_s8_16_neon(grid_hi, q8_signed_hi);

                    // Widen to i32 and pairwise add (madd with ones)
                    int32x4_t w0 = vmovl_s16(vget_low_s16(p16_lo));
                    int32x4_t w1 = vmovl_s16(vget_high_s16(p16_lo));
                    int32x4_t w2 = vmovl_s16(vget_low_s16(p16_hi));
                    int32x4_t w3 = vmovl_s16(vget_high_s16(p16_hi));

                    int32x4_t ps0 = vpaddq_s32(w0, w1);
                    int32x4_t ps1 = vpaddq_s32(w2, w3);
                    // hadd(ps0, ps1) → 4 sums
                    int32x4_t h = vpaddq_s32(ps0, ps1);
                    // hadd(h, h) → 2 sums
                    int32x4_t h2 = vpaddq_s32(h, h);
                    // hadd(h2, h2) → 1 sum
                    int32x4_t h3 = vpaddq_s32(h2, h2);
                    s_all = vgetq_lane_s32(h3, 0);
                }
#endif  // USE_DOTPROD

                block_acc += (float)s_all * db_f;

                qs += 8;
                signs += 4;
            }
            qh += 2;
        }

        acc += d * d_q8 * block_acc;
    }

    return acc;
}

// ============================================================================
// IQ3_S GEMV
// ============================================================================

static void gemv_iq3_s_q8k_transB_neon(const float* a, const uint8_t* w, float* out,
                                       int M, int K, int N) {
    constexpr int QK_K = 256;
    constexpr int IQ3_S_BLOCK_SIZE = 110;
    const int nb = (K + QK_K - 1) / QK_K;

    if (M == 1) {
        std::vector<cpu::block_q8_K> q8_buf(nb);
        cpu::quantize_row_q8_K(a, q8_buf.data(), K);

#pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const uint8_t* row = w + (size_t)n * nb * IQ3_S_BLOCK_SIZE;
            out[n] = dot_iq3_s_q8_K_neon(row, q8_buf.data(), nb);
        }
    } else {
        std::vector<cpu::block_q8_K> q8_all((size_t)M * nb);
        for (int m = 0; m < M; ++m)
            cpu::quantize_row_q8_K(a + (size_t)m * K, q8_all.data() + (size_t)m * nb, K);

#pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const uint8_t* row = w + (size_t)n * nb * IQ3_S_BLOCK_SIZE;
            for (int m = 0; m < M; ++m)
                out[(size_t)m * N + n] = dot_iq3_s_q8_K_neon(row,
                    q8_all.data() + (size_t)m * nb, nb);
        }
    }
}

// ============================================================================
// IQ2_S x Q8_K fused dot product (NEON)
// IQ2_S block: 82 bytes = d[2] + qs[32] + signs[32] + qh[8] + sc[8]
// ============================================================================

static inline float dot_iq2_s_q8_K_neon(const uint8_t* iq2s_row,
                                        const cpu::block_q8_K* q8, int nb) {
    constexpr int QK_K = 256;
    constexpr int IQ2_S_BLOCK_SIZE = 82;

    float acc = 0.0f;

    for (int bi = 0; bi < nb; ++bi) {
        const uint8_t* block_ptr = iq2s_row + (size_t)bi * IQ2_S_BLOCK_SIZE;
        float d = cpu::fp16_to_float_scalar(
            *reinterpret_cast<const uint16_t*>(block_ptr));
        const uint8_t* qs    = block_ptr + 2;
        const uint8_t* signs = qs + 32;
        const uint8_t* qh    = block_ptr + 66;
        const uint8_t* sc    = block_ptr + 74;

        const cpu::block_q8_K* y = q8 + bi;
        const float d_q8 = y->d;
        const int8_t* q8d = y->qs;

        float block_acc = 0.0f;

        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            const float db0_f = (0.5f + (sc[ib32] & 0xf)) * 0.25f;
            const float db1_f = (0.5f + (sc[ib32] >> 4)) * 0.25f;

            // Build grid_vals[32] and sign_mask[32]
            alignas(16) uint8_t grid_vals[32];
            alignas(16) uint8_t sign_mask_arr[32];
            for (int l = 0; l < 4; ++l) {
                const int grid_idx = qs[l] |
                    ((qh[ib32] << (8 - 2 * l)) & 0x300);
                // iq2s_grid entries are 8 bytes each
                memcpy(grid_vals + l * 8,
                       &forge::ops::iq2s_grid[grid_idx], 8);
                const uint8_t s = signs[l];
                for (int j = 0; j < 8; ++j)
                    sign_mask_arr[l * 8 + j] =
                        (s & forge::ops::kmask_iq2xs[j]) ? 0xFF : 0x00;
            }

            uint8x16_t grid_lo = vld1q_u8(grid_vals);
            uint8x16_t grid_hi = vld1q_u8(grid_vals + 16);
            uint8x16_t sign_lo = vld1q_u8(sign_mask_arr);
            uint8x16_t sign_hi = vld1q_u8(sign_mask_arr + 16);

            int8x16_t q8_lo = vld1q_s8(q8d + ib32 * 32);
            int8x16_t q8_hi = vld1q_s8(q8d + ib32 * 32 + 16);

            int8x16_t q8_signed_lo = vbslq_s8(sign_lo, vnegq_s8(q8_lo), q8_lo);
            int8x16_t q8_signed_hi = vbslq_s8(sign_hi, vnegq_s8(q8_hi), q8_hi);

            int s_l01 = 0, s_l23 = 0;
#ifdef USE_DOTPROD
            {
                int32x4_t acc0 = maddubs_dotprod_16_neon(grid_lo, q8_signed_lo);
                s_l01 = vaddvq_s32(acc0);

                int32x4_t acc1 = maddubs_dotprod_16_neon(grid_hi, q8_signed_hi);
                s_l23 = vaddvq_s32(acc1);
            }
#else
            {
                int16x8_t p16_lo = maddubs_u8_s8_16_neon(grid_lo, q8_signed_lo);
                int16x8_t p16_hi = maddubs_u8_s8_16_neon(grid_hi, q8_signed_hi);

                int32x4_t w0 = vmovl_s16(vget_low_s16(p16_lo));
                int32x4_t w1 = vmovl_s16(vget_high_s16(p16_lo));
                int32x4_t w2 = vmovl_s16(vget_low_s16(p16_hi));
                int32x4_t w3 = vmovl_s16(vget_high_s16(p16_hi));

                int32x4_t ps0 = vpaddq_s32(w0, w1);
                int32x4_t ps1 = vpaddq_s32(w2, w3);
                int32x4_t h = vpaddq_s32(ps0, ps1);
                int32x4_t h2 = vpaddq_s32(h, h);
                s_l01 = vgetq_lane_s32(h2, 0);
                s_l23 = vgetq_lane_s32(h2, 1);
            }
#endif  // USE_DOTPROD

            block_acc += (float)s_l01 * db0_f + (float)s_l23 * db1_f;

            qs += 4;
            signs += 4;
        }

        acc += d * d_q8 * block_acc;
    }

    return acc;
}

// ============================================================================
// IQ2_S GEMV
// ============================================================================

static void gemv_iq2_s_q8k_transB_neon(const float* a, const uint8_t* w, float* out,
                                       int M, int K, int N) {
    constexpr int QK_K = 256;
    constexpr int IQ2_S_BLOCK_SIZE = 82;
    const int nb = (K + QK_K - 1) / QK_K;

    if (M == 1) {
        std::vector<cpu::block_q8_K> q8_buf(nb);
        cpu::quantize_row_q8_K(a, q8_buf.data(), K);

#pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const uint8_t* row = w + (size_t)n * nb * IQ2_S_BLOCK_SIZE;
            out[n] = dot_iq2_s_q8_K_neon(row, q8_buf.data(), nb);
        }
    } else {
        std::vector<cpu::block_q8_K> q8_all((size_t)M * nb);
        for (int m = 0; m < M; ++m)
            cpu::quantize_row_q8_K(a + (size_t)m * K, q8_all.data() + (size_t)m * nb, K);

#pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const uint8_t* row = w + (size_t)n * nb * IQ2_S_BLOCK_SIZE;
            for (int m = 0; m < M; ++m)
                out[(size_t)m * N + n] = dot_iq2_s_q8_K_neon(row,
                    q8_all.data() + (size_t)m * nb, nb);
        }
    }
}

// ============================================================================
// DotQ8KFn dispatcher for batched MoE matmul
// ============================================================================

using DotQ8KFn = float (*)(const uint8_t*, const cpu::block_q8_K*, int);

static inline DotQ8KFn get_dot_q8k_fn(DataType dt) {
    switch (dt) {
    case DataType::IQ2_XS: return dot_iq2_xs_q8_K_neon;
    case DataType::IQ3_S:  return dot_iq3_s_q8_K_neon;
    case DataType::IQ4_NL: return dot_iq4_nl_q8_K_neon;
    case DataType::IQ2_S:  return dot_iq2_s_q8_K_neon;
    default: return nullptr;
    }
}

// ============================================================================
// Q4_K × Q8_K single-row dot product (NEON)
// Q4_K block: 144 bytes = d[2](fp16) + dmin[2](fp16) + scales[12] + qs[128]
// 8 sub-blocks of 32 elements each.
// Value = scale_sb × nibble + min_sb  (super-block scale + min applied after).
// ============================================================================

static inline float dot_q4_K_q8_K_row_neon(const uint8_t* q4_row,
                                           const cpu::block_q8_K* q8, int nb) {
    constexpr int QK_K = 256;
    constexpr int Q4_K_BLOCK_BYTES = 144;

    float acc = 0.0f;

    for (int bi = 0; bi < nb; ++bi) {
        const cpu::block_q4_K* x =
            reinterpret_cast<const cpu::block_q4_K*>(q4_row) + bi;
        const cpu::block_q8_K* y = q8 + bi;

        const float d = y->d * fp16_to_float_scalar(x->d);
        const float dmin_ratio =
            -fp16_to_float_scalar(x->dmin) / fp16_to_float_scalar(x->d);

        uint8_t sc[8], mn[8];
        decode_q4_k_scales_neon(x->scales, sc, mn);

        const uint8_t* q4_qs = x->qs;
        const int8_t* q8_qs = y->qs;
        float block_acc = 0.0f;

        for (int sb = 0; sb < 8; ++sb) {
            // Expand 32 nibbles from 16 bytes
            uint8x16_t q4_bytes = vld1q_u8(q4_qs + sb * 16);
            uint8x16_t q4_lo = vandq_u8(q4_bytes, vdupq_n_u8(0x0F));
            uint8x16_t q4_hi = vshrq_n_u8(q4_bytes, 4);

            // Load 32 Q8 values
            int8x16_t q8_0 = vld1q_s8(q8_qs + sb * 32);
            int8x16_t q8_1 = vld1q_s8(q8_qs + sb * 32 + 16);

            // Interleave even/odd nibbles for correct Q8 pairing
            // q4_lo = [n0, n2, ..., n30], q4_hi = [n1, n3, ..., n31]
            // zip → [n0,n1,...,n15] × q8[0..15] + [n16,n17,...,n31] × q8[16..31]
            int8x16_t q4_lo_s8_q = vreinterpretq_s8_u8(q4_lo);
            int8x16_t q4_hi_s8_q = vreinterpretq_s8_u8(q4_hi);
            int8x16_t q4_0 = vzip1q_s8(q4_lo_s8_q, q4_hi_s8_q);
            int8x16_t q4_1 = vzip2q_s8(q4_lo_s8_q, q4_hi_s8_q);

            int dot_sb;
#ifdef USE_DOTPROD
            {
                int32x4_t acc_dp =
                    vdotq_s32(vdupq_n_s32(0), q4_0, q8_0);
                acc_dp = vdotq_s32(acc_dp, q4_1, q8_1);
                dot_sb = vaddvq_s32(acc_dp);
            }
#else
            {
                int16x8_t p0 = vmull_s8(
                    vget_low_s8(q4_0),
                    vget_low_s8(q8_0));
                int16x8_t p1 = vmull_s8(
                    vget_high_s8(q4_0),
                    vget_high_s8(q8_0));
                int16x8_t p2 = vmull_s8(
                    vget_low_s8(q4_1),
                    vget_low_s8(q8_1));
                int16x8_t p3 = vmull_s8(
                    vget_high_s8(q4_1),
                    vget_high_s8(q8_1));
                int32x4_t s0 = vpaddlq_s16(p0);
                int32x4_t s1 = vpaddlq_s16(p1);
                int32x4_t s2 = vpaddlq_s16(p2);
                int32x4_t s3 = vpaddlq_s16(p3);
                dot_sb = vaddvq_s32(
                    vaddq_s32(vaddq_s32(s0, s1),
                              vaddq_s32(s2, s3)));
            }
#endif // USE_DOTPROD

            // Sum of Q8 for min correction
            int16x8_t q8sum_lo = vpaddlq_s8(q8_0);
            int16x8_t q8sum_hi = vpaddlq_s8(q8_1);
            int32x4_t q8sum32 =
                vaddq_s32(vpaddlq_s16(q8sum_lo),
                          vpaddlq_s16(q8sum_hi));
            int q8sum = vaddvq_s32(q8sum32);

            block_acc += (float)dot_sb * (float)sc[sb] +
                         dmin_ratio * (float)q8sum * (float)mn[sb];
        }

        acc += d * block_acc;
    }

    return acc;
}

// ============================================================================
// Q3_K sub-block dot product (NEON)
// Processes one Q3_K super-block (256 elements) with one Q8_K activation block.
// Returns 4 partial sums in float32x4_t (8 sub-groups accumulated into 4 lanes).
//
// Q3_K layout: hmask[32] + qs[64] + scales[12] + d[2]
// Each super-block: 2 halves × 4 sub-groups × 32 elements.
// qs packing: 4×2-bit values per byte, sub-groups interleaved.
// Value = (low_2bit - (hmask_bit ? 0 : 4)) × scale × d
// ============================================================================

static inline float32x4_t q3_k_sb_dot_neon(const uint8_t* q3_sb,
                                           const cpu::block_q8_K* q8) {
    constexpr int QK_K = 256;
    const uint8x16_t m3 = vdupq_n_u8(3);

    const cpu::block_q3_K* x =
        reinterpret_cast<const cpu::block_q3_K*>(q3_sb);
    const float d = q8->d * fp16_to_float_scalar(x->d);

    const uint8_t* q3d = x->qs;
    const int8_t* q8d = q8->qs;

    // Decode 12-byte scales into 16 signed int8, centered around 0.
    // Uses same ordering as x86 _mm_set_epi32.
    constexpr uint32_t kmask1 = 0x03030303;
    constexpr uint32_t kmask2 = 0x0f0f0f0f;

    uint32_t aux[4];
    std::memcpy(aux, x->scales, 12);
    uint32_t aux0 = aux[0];
    uint32_t aux1 = aux[1];
    uint32_t tmp = aux[2];
    aux[0] = (aux0 & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux1 & kmask2) | (((tmp >> 2) & kmask1) << 4);
    aux[2] = ((aux0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);

    int8_t scales_s8[16];
    for (int i = 0; i < 16; ++i)
        scales_s8[i] = (int8_t)(((const uint8_t*)aux)[i] - 32);

    // Accumulate the block total. Each 32-element sub-group is actually
    // two 16-element groups with *distinct* scales:
    //   low 16 elements  → scale[half*8 + sg*2]
    //   high 16 elements → scale[half*8 + sg*2 + 1]
    // Unlike Q4_K, Q3_K uses one int8 scale per 16 elements (16 total).
    int32_t acc_total = 0;
    int bit = 0;

    for (int half = 0; half < QK_K / 128; ++half) {
        // Load 32 bytes of qs (2 NEON loads) for this 128-element half
        uint8x16_t q3_bytes0 = vld1q_u8(q3d);
        uint8x16_t q3_bytes1 = vld1q_u8(q3d + 16);
        q3d += 32;

        // Load hmask bytes (same 32 bytes for both halves; bit position varies)
        uint8x16_t hmask0 = vld1q_u8(x->hmask);
        uint8x16_t hmask1 = vld1q_u8(x->hmask + 16);

        for (int sg = 0; sg < 4; ++sg) {
            uint8_t shift = sg * 2;

            // Extract low 2 bits
            // shift is a runtime value (sg*2). Apple Clang requires constant
            // immediates for vshrq_n_u8 and has no variable-shift vshrq_u8
            // intrinsic, so negate the shift and use vshlq_u8 (USHL shifts
            // right with zero-fill for negative unsigned shift amounts).
            int8x16_t neg_shift = vdupq_n_s8(-static_cast<int8_t>(shift));
            uint8x16_t q3l_0 = vandq_u8(vshlq_u8(q3_bytes0, neg_shift), m3);
            uint8x16_t q3l_1 = vandq_u8(vshlq_u8(q3_bytes1, neg_shift), m3);

            // Extract high bit from hmask:
            // q3h = ((~hmask & (1 << bit)) >> bit) << 2  →  0 or 4
            uint8_t bit_val = 1 << bit;
            uint8x16_t bit_mask = vdupq_n_u8(bit_val);
            int8x16_t neg_bit = vdupq_n_s8(-static_cast<int8_t>(bit));
            uint8x16_t q3h_0 = vbicq_u8(bit_mask, hmask0);
            uint8x16_t q3h_1 = vbicq_u8(bit_mask, hmask1);
            q3h_0 = vshlq_n_u8(vshlq_u8(q3h_0, neg_bit), 2);
            q3h_1 = vshlq_n_u8(vshlq_u8(q3h_1, neg_bit), 2);
            ++bit;

            // Compute q3l = q3_low - q3_high as signed int8
            int8x16_t q3_lo = vsubq_s8(
                vreinterpretq_s8_u8(q3l_0),
                vreinterpretq_s8_u8(q3h_0));
            int8x16_t q3_hi = vsubq_s8(
                vreinterpretq_s8_u8(q3l_1),
                vreinterpretq_s8_u8(q3h_1));

            // Load q8 for this half of the sub-group (16 elements)
            int8x16_t q8_lo = vld1q_s8(q8d);
            int8x16_t q8_hi = vld1q_s8(q8d + 16);
            q8d += 32;

            int32_t dot_lo, dot_hi;
#ifdef USE_DOTPROD
            {
                int32x4_t dp_lo = vdotq_s32(vdupq_n_s32(0), q3_lo, q8_lo);
                int32x4_t dp_hi = vdotq_s32(vdupq_n_s32(0), q3_hi, q8_hi);
                dot_lo = vaddvq_s32(dp_lo);
                dot_hi = vaddvq_s32(dp_hi);
            }
#else
            {
                int16x8_t p0 = vmull_s8(
                    vget_low_s8(q3_lo), vget_low_s8(q8_lo));
                int16x8_t p1 = vmull_s8(
                    vget_high_s8(q3_lo), vget_high_s8(q8_lo));
                int16x8_t p2 = vmull_s8(
                    vget_low_s8(q3_hi), vget_low_s8(q8_hi));
                int16x8_t p3 = vmull_s8(
                    vget_high_s8(q3_hi), vget_high_s8(q8_hi));
                int32x4_t s0 = vpaddlq_s16(p0);
                int32x4_t s1 = vpaddlq_s16(p1);
                int32x4_t s2 = vpaddlq_s16(p2);
                int32x4_t s3 = vpaddlq_s16(p3);
                dot_lo = vaddvq_s32(vaddq_s32(s0, s1));
                dot_hi = vaddvq_s32(vaddq_s32(s2, s3));
            }
#endif // USE_DOTPROD

            // Apply the two distinct scales and accumulate into the total
            const int is = half * 8 + sg * 2;
            acc_total += dot_lo * (int32_t)scales_s8[is] +
                         dot_hi * (int32_t)scales_s8[is + 1];
        }
    }

    // Return the total in every lane so the caller's 4-lane sum is correct.
    float f = d * (float)acc_total;
    return vdupq_n_f32(f);
}

// ============================================================================
// Q3_K / Q4_K fused QKV GEMV (NEON): mixed-precision fused QKV projection
// Quantizes activation to Q8_K once, then:
//   Q and K rows → q3_k_sb_dot_neon
//   V rows      → dot_q4_K_q8_K_row_neon
// Saves 2 quantizations (3→1) plus better cache reuse.
// ============================================================================

static void gemv_q3_k_q4_k_fused_qkv_neon(const float* a, const uint8_t* wq,
                                           const uint8_t* wk, const uint8_t* wv,
                                           float* out_q, float* out_k,
                                           float* out_v, int K, int N_q, int N_k,
                                           int N_v) {
    constexpr int QK_K = 256;
    constexpr int Q3_K_BLOCK_BYTES = 110;
    constexpr int Q4_K_BLOCK_BYTES = 144;
    const int nb = (K + QK_K - 1) / QK_K;

    std::vector<cpu::block_q8_K> q8_buf(nb);
    cpu::quantize_row_q8_K(a, q8_buf.data(), K);

    // Q and K use Q3_K → q3_k_sb_dot
    auto dot_q3_rows = [&](const uint8_t* w, float* out, int N) {
        #pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const uint8_t* q3_row = w + (size_t)n * nb * Q3_K_BLOCK_BYTES;
            float32x4_t acc = vdupq_n_f32(0.0f);
            for (int i = 0; i < nb; ++i) {
                acc = vaddq_f32(acc, q3_k_sb_dot_neon(
                    q3_row + (size_t)i * Q3_K_BLOCK_BYTES, &q8_buf[i]));
            }
            out[n] = vgetq_lane_f32(acc, 0);  // total is broadcast to all lanes
        }
    };

    dot_q3_rows(wq, out_q, N_q);
    dot_q3_rows(wk, out_k, N_k);

    // V uses Q4_K → dot_q4_K_q8_K_row
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N_v; ++n) {
        const uint8_t* q4_row = wv + (size_t)n * nb * Q4_K_BLOCK_BYTES;
        out_v[n] = dot_q4_K_q8_K_row_neon(q4_row, q8_buf.data(), nb);
    }
}

// ============================================================================
// Q3_K fused FFN Up (SiLU(gate) × up) for decode (NEON)
// Quantizes activation to Q8_K once, then for each output row computes
// gate and up dot products in parallel, sharing the activation read.
// ============================================================================

static void gemv_q3_k_fused_ffn_up_neon(const float* a, const uint8_t* w_gate,
                                        const uint8_t* w_up, float* out, int K, int N) {
    constexpr int QK_K = 256;
    constexpr int Q3_K_BLOCK_BYTES = 110;
    const int nb = (K + QK_K - 1) / QK_K;

    std::vector<cpu::block_q8_K> q8_buf(nb);
    cpu::quantize_row_q8_K(a, q8_buf.data(), K);

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        const uint8_t* gate_row = w_gate + (size_t)n * nb * Q3_K_BLOCK_BYTES;
        const uint8_t* up_row = w_up + (size_t)n * nb * Q3_K_BLOCK_BYTES;

        float32x4_t gate_acc = vdupq_n_f32(0.0f);
        float32x4_t up_acc = vdupq_n_f32(0.0f);

        for (int i = 0; i < nb; ++i) {
            const uint8_t* gate_sb = gate_row + (size_t)i * Q3_K_BLOCK_BYTES;
            const uint8_t* up_sb = up_row + (size_t)i * Q3_K_BLOCK_BYTES;
            gate_acc = vaddq_f32(gate_acc, q3_k_sb_dot_neon(gate_sb, &q8_buf[i]));
            up_acc = vaddq_f32(up_acc, q3_k_sb_dot_neon(up_sb, &q8_buf[i]));
        }

        float gate_val = vgetq_lane_f32(gate_acc, 0);  // total is broadcast to all lanes
        float up_val = vgetq_lane_f32(up_acc, 0);
        out[n] = (gate_val / (1.0f + std::exp(-gate_val))) * up_val;
    }
}

// ============================================================================
// Q4_0 fused FFN Up (SiLU(gate) × up) for decode (NEON)
// Quantizes activation to Q8_0, then for each output row computes
// gate and up dot products simultaneously using vec_dot.
// ============================================================================

static void gemv_q4_0_fused_ffn_up_neon(const float* a, const uint8_t* w_gate,
                                        const uint8_t* w_up, float* out, int K, int N) {
    constexpr int BLOCK_SIZE = 32;
    constexpr int BLOCK_BYTES = 18;
    const int nb = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Quantize activation to Q8_0 (shared between gate & up)
    std::vector<cpu::block_q8_0_act> q8_buf(nb);
    cpu::quantize_row_q8_0_act(a, q8_buf.data(), K);

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        const uint8_t* gate_row = w_gate + (size_t)n * nb * BLOCK_BYTES;
        const uint8_t* up_row = w_up + (size_t)n * nb * BLOCK_BYTES;

        float gate_val = forge::cpu::vec_dot_q4_0_q8_0_neon(
            gate_row, q8_buf.data(), nb);
        float up_val = forge::cpu::vec_dot_q4_0_q8_0_neon(
            up_row, q8_buf.data(), nb);

        out[n] = (gate_val / (1.0f + std::exp(-gate_val))) * up_val;
    }
}

// K-quant row dots shared by GEMM and fused projections.
static inline int dot_s8_16_neon(int8x16_t x, int8x16_t y) {
#ifdef USE_DOTPROD
    return vaddvq_s32(vdotq_s32(vdupq_n_s32(0), x, y));
#else
    return vaddvq_s32(vaddq_s32(vpaddlq_s16(vmull_s8(vget_low_s8(x), vget_low_s8(y))),
                                  vpaddlq_s16(vmull_s8(vget_high_s8(x), vget_high_s8(y)))));
#endif
}

static inline float dot_q2_K_q8_K_row_neon(const uint8_t* row,
                                             const block_q8_K* q8, int nb) {
    float acc = 0.0f;
    for (int bi = 0; bi < nb; ++bi) {
        const auto* x = reinterpret_cast<const block_q2_K*>(row) + bi;
        const auto* y = q8 + bi;
        const float d = y->d * fp16_to_float_scalar(x->d);
        const float dmin = -y->d * fp16_to_float_scalar(x->dmin);
        for (int half = 0; half < 2; ++half) for (int j = 0; j < 4; ++j) {
            const int group = half * 8 + j * 2;
            const uint8_t shift = j * 2;
            for (int part = 0; part < 2; ++part) {
                alignas(16) int8_t values[16];
                const uint8_t* src = x->qs + half * 32 + part * 16;
                for (int l = 0; l < 16; ++l) values[l] = (src[l] >> shift) & 3;
                const int gi = group + part;
                const int dot = dot_s8_16_neon(vld1q_s8(values), vld1q_s8(y->qs + gi * 16));
                acc += d * (x->scales[gi] & 0x0f) * dot +
                       dmin * (x->scales[gi] >> 4) * y->bsums[gi];
            }
        }
    }
    return acc;
}

static inline float dot_q3_K_q8_K_row_neon(const uint8_t* row,
                                             const block_q8_K* q8, int nb) {
    float acc = 0.0f;
    for (int bi = 0; bi < nb; ++bi)
        acc += vgetq_lane_f32(q3_k_sb_dot_neon(row + (size_t)bi * sizeof(block_q3_K), q8 + bi), 0);
    return acc;
}

static inline float dot_q5_K_q8_K_row_neon(const uint8_t* row,
                                             const block_q8_K* q8, int nb) {
    float acc = 0.0f;
    for (int bi = 0; bi < nb; ++bi) {
        const auto* x = reinterpret_cast<const block_q5_K*>(row) + bi;
        const auto* y = q8 + bi;
        uint8_t sc[8], mn[8];
        decode_q4_k_scales_neon(x->scales, sc, mn);
        const float d = y->d * fp16_to_float_scalar(x->d);
        const float dmin = -y->d * fp16_to_float_scalar(x->dmin);
        for (int chunk = 0; chunk < 4; ++chunk) {
            alignas(16) int8_t values[16];
            const uint8_t* ql = x->ql + chunk * 32;
            const uint8_t low_bit = 1u << (2 * chunk);
            const uint8_t high_bit = 1u << (2 * chunk + 1);
            int lo_dot = 0, hi_dot = 0;
            for (int part = 0; part < 2; ++part) {
                for (int l = 0; l < 16; ++l)
                    values[l] = (ql[part * 16 + l] & 0x0f) + ((x->qh[part * 16 + l] & low_bit) ? 16 : 0);
                lo_dot += dot_s8_16_neon(vld1q_s8(values), vld1q_s8(y->qs + chunk * 64 + part * 16));
                for (int l = 0; l < 16; ++l)
                    values[l] = (ql[part * 16 + l] >> 4) + ((x->qh[part * 16 + l] & high_bit) ? 16 : 0);
                hi_dot += dot_s8_16_neon(vld1q_s8(values), vld1q_s8(y->qs + chunk * 64 + 32 + part * 16));
            }
            acc += d * (sc[2 * chunk] * lo_dot + sc[2 * chunk + 1] * hi_dot) +
                   dmin * (mn[2 * chunk] * (y->bsums[chunk * 4] + y->bsums[chunk * 4 + 1]) +
                           mn[2 * chunk + 1] * (y->bsums[chunk * 4 + 2] + y->bsums[chunk * 4 + 3]));
        }
    }
    return acc;
}

static inline float dot_q6_K_q8_K_row_neon(const uint8_t* row,
                                             const block_q8_K* q8, int nb) {
    float acc = 0.0f;
    for (int bi = 0; bi < nb; ++bi) {
        const auto* x = reinterpret_cast<const block_q6_K*>(row) + bi;
        const auto* y = q8 + bi;
        const float d = y->d * fp16_to_float_scalar(x->d);
        for (int g = 0; g < 16; ++g) {
            const int half = g / 8, pos = g % 8;
            const int ql_offset = half * 64 + (pos == 2 || pos == 3 || pos == 6 || pos == 7 ? 32 : 0) + (pos & 1) * 16;
            const int qh_offset = half * 32 + (pos & 1) * 16;
            const int shift = (pos / 2) * 2;
            const bool high_nibble = pos >= 4;
            alignas(16) int8_t values[16];
            for (int l = 0; l < 16; ++l) {
                const uint8_t ql = x->ql[ql_offset + l];
                const int low = high_nibble ? (ql >> 4) : (ql & 0x0f);
                values[l] = (low | (((x->qh[qh_offset + l] >> shift) & 3) << 4)) - 32;
            }
            acc += d * x->scales[g] * dot_s8_16_neon(vld1q_s8(values), vld1q_s8(y->qs + g * 16));
        }
    }
    return acc;
}

template <typename DotFn>
static inline void fused_ffn_up_k_neon(const float* a, const uint8_t* gate, const uint8_t* up,
                                       float* out, int K, int N, size_t block_bytes, DotFn dot) {
    const int nb = (K + 255) / 256;
    std::vector<block_q8_K> q8(nb);
    quantize_row_q8_K(a, q8.data(), K);
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        const float g = dot(gate + (size_t)n * nb * block_bytes, q8.data(), nb);
        const float u = dot(up + (size_t)n * nb * block_bytes, q8.data(), nb);
        out[n] = (g / (1.0f + std::exp(-g))) * u;
    }
}

template <typename DotFn>
static inline void fused_qkv_k_neon(const float* a, const uint8_t* wq, const uint8_t* wk,
                                    const uint8_t* wv, float* oq, float* ok, float* ov,
                                    int K, int Nq, int Nk, int Nv, size_t block_bytes, DotFn dot) {
    const int nb = (K + 255) / 256;
    std::vector<block_q8_K> q8(nb);
    quantize_row_q8_K(a, q8.data(), K);
    auto run = [&](const uint8_t* w, float* o, int N) {
        #pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n)
            o[n] = dot(w + (size_t)n * nb * block_bytes, q8.data(), nb);
    };
    run(wq, oq, Nq); run(wk, ok, Nk); run(wv, ov, Nv);
}

template <typename DotFn>
static inline void fused_residual_k_neon(const float* a, const uint8_t* w, const float* residual,
                                         float* out, int K, int N, size_t block_bytes, DotFn dot) {
    const int nb = (K + 255) / 256;
    std::vector<block_q8_K> q8(nb);
    quantize_row_q8_K(a, q8.data(), K);
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n)
        out[n] = residual[n] + dot(w + (size_t)n * nb * block_bytes, q8.data(), nb);
}

static inline void gemv_q3_K_fused_ffn_up_neon(const float* a, const uint8_t* g, const uint8_t* u, float* o, int K, int N) { fused_ffn_up_k_neon(a,g,u,o,K,N,sizeof(block_q3_K),dot_q3_K_q8_K_row_neon); }
static inline void gemv_q4_K_fused_ffn_up_neon(const float* a, const uint8_t* g, const uint8_t* u, float* o, int K, int N) { fused_ffn_up_k_neon(a,g,u,o,K,N,sizeof(block_q4_K),dot_q4_K_q8_K_row_neon); }
static inline void gemv_q5_K_fused_ffn_up_neon(const float* a, const uint8_t* g, const uint8_t* u, float* o, int K, int N) { fused_ffn_up_k_neon(a,g,u,o,K,N,sizeof(block_q5_K),dot_q5_K_q8_K_row_neon); }
static inline void gemv_q2_K_fused_ffn_up_neon(const float* a, const uint8_t* g, const uint8_t* u, float* o, int K, int N) { fused_ffn_up_k_neon(a,g,u,o,K,N,sizeof(block_q2_K),dot_q2_K_q8_K_row_neon); }
static inline void gemv_q3_K_fused_qkv_neon(const float* a, const uint8_t* q, const uint8_t* k, const uint8_t* v, float* oq, float* ok, float* ov, int K, int Nq, int Nk, int Nv) { fused_qkv_k_neon(a,q,k,v,oq,ok,ov,K,Nq,Nk,Nv,sizeof(block_q3_K),dot_q3_K_q8_K_row_neon); }
static inline void gemv_q4_K_fused_qkv_neon(const float* a, const uint8_t* q, const uint8_t* k, const uint8_t* v, float* oq, float* ok, float* ov, int K, int Nq, int Nk, int Nv) { fused_qkv_k_neon(a,q,k,v,oq,ok,ov,K,Nq,Nk,Nv,sizeof(block_q4_K),dot_q4_K_q8_K_row_neon); }
static inline void gemv_q5_K_fused_qkv_neon(const float* a, const uint8_t* q, const uint8_t* k, const uint8_t* v, float* oq, float* ok, float* ov, int K, int Nq, int Nk, int Nv) { fused_qkv_k_neon(a,q,k,v,oq,ok,ov,K,Nq,Nk,Nv,sizeof(block_q5_K),dot_q5_K_q8_K_row_neon); }
static inline void gemv_q2_K_fused_qkv_neon(const float* a, const uint8_t* q, const uint8_t* k, const uint8_t* v, float* oq, float* ok, float* ov, int K, int Nq, int Nk, int Nv) { fused_qkv_k_neon(a,q,k,v,oq,ok,ov,K,Nq,Nk,Nv,sizeof(block_q2_K),dot_q2_K_q8_K_row_neon); }
static inline void gemv_q4_K_attn_proj_residual_neon(const float* a, const uint8_t* w, const float* r, float* o, int K, int N) { fused_residual_k_neon(a,w,r,o,K,N,sizeof(block_q4_K),dot_q4_K_q8_K_row_neon); }
static inline void gemv_q5_K_attn_proj_residual_neon(const float* a, const uint8_t* w, const float* r, float* o, int K, int N) { fused_residual_k_neon(a,w,r,o,K,N,sizeof(block_q5_K),dot_q5_K_q8_K_row_neon); }
static inline void gemv_q6_K_attn_proj_residual_neon(const float* a, const uint8_t* w, const float* r, float* o, int K, int N) { fused_residual_k_neon(a,w,r,o,K,N,sizeof(block_q6_K),dot_q6_K_q8_K_row_neon); }
static inline void gemv_q2_K_attn_proj_residual_neon(const float* a, const uint8_t* w, const float* r, float* o, int K, int N) { fused_residual_k_neon(a,w,r,o,K,N,sizeof(block_q2_K),dot_q2_K_q8_K_row_neon); }
static inline void gemv_q3_K_attn_proj_residual_neon(const float* a, const uint8_t* w, const float* r, float* o, int K, int N) { fused_residual_k_neon(a,w,r,o,K,N,sizeof(block_q3_K),dot_q3_K_q8_K_row_neon); }
static inline void gemv_q4_K_ffn_down_residual_neon(const float* a, const uint8_t* w, const float* r, float* o, int K, int N) { fused_residual_k_neon(a,w,r,o,K,N,sizeof(block_q4_K),dot_q4_K_q8_K_row_neon); }
static inline void gemv_q5_K_ffn_down_residual_neon(const float* a, const uint8_t* w, const float* r, float* o, int K, int N) { fused_residual_k_neon(a,w,r,o,K,N,sizeof(block_q5_K),dot_q5_K_q8_K_row_neon); }
static inline void gemv_q2_K_ffn_down_residual_neon(const float* a, const uint8_t* w, const float* r, float* o, int K, int N) { fused_residual_k_neon(a,w,r,o,K,N,sizeof(block_q2_K),dot_q2_K_q8_K_row_neon); }
static inline void gemv_q3_K_ffn_down_residual_neon(const float* a, const uint8_t* w, const float* r, float* o, int K, int N) { fused_residual_k_neon(a,w,r,o,K,N,sizeof(block_q3_K),dot_q3_K_q8_K_row_neon); }
static inline void gemv_q6_K_ffn_down_residual_neon(const float* a, const uint8_t* w, const float* r, float* o, int K, int N) { fused_residual_k_neon(a,w,r,o,K,N,sizeof(block_q6_K),dot_q6_K_q8_K_row_neon); }

#endif  // USE_NEON

}  // namespace cpu

// The dispatcher lives in forge::cpu (matching this file), but the batched
// MoE drivers in forge::ops call it unqualified; mirror the x86 header by
// re-exporting it into forge::ops.
namespace ops {
using cpu::get_dot_q8k_fn;
}  // namespace ops

}  // namespace forge
