#pragma once
// ARM64 NEON GEMV kernels — decode-phase (M=1) matrix-vector products.
// Provides gemv_fp32_transB, gemv_q4_0_transB, gemv_q8_0_transB.
// All functions compute: out = a * w^T, where a is [1, K] and w is [N, K].

#ifdef USE_NEON
#include <arm_neon.h>
#endif
#include <cstring>
#include <cmath>
#include <vector>

#include "../../common/quant_helpers.h"

namespace forge {
namespace cpu {

#ifdef USE_NEON

// ---- FP32 GEMV: out[n] = sum_k(a[k] * w[n][k]) ----
// NR=4 row-level unrolling. Each iteration processes 4 output rows,
// dotting a[K] with 4 rows of w concurrently via NEON fma.
// Supports both M=1 (decode) and M>1 (prefill / batched) by looping over
// the M dimension; each output row group is written independently.
static inline void gemv_fp32_transB_neon(const float* a, const float* b,
                                          float* out, int M, int K, int N) {
    // M=1 decode fast path
    for (int m = 0; m < M; ++m) {
        const float* a_row = a + (size_t)m * K;
        float* o_row = out + (size_t)m * N;
        const int NR = 4;
        int n = 0;
        for (; n + NR - 1 < N; n += NR) {
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            float32x4_t acc3 = vdupq_n_f32(0.0f);

            int k = 0;
            for (; k + 3 < K; k += 4) {
                float32x4_t av = vld1q_f32(a_row + k);
                acc0 = vmlaq_f32(acc0, av, vld1q_f32(b + (size_t)n * K + k));
                acc1 = vmlaq_f32(acc1, av, vld1q_f32(b + (size_t)(n+1) * K + k));
                acc2 = vmlaq_f32(acc2, av, vld1q_f32(b + (size_t)(n+2) * K + k));
                acc3 = vmlaq_f32(acc3, av, vld1q_f32(b + (size_t)(n+3) * K + k));
            }
            // Scalar tail
            float sum0 = vaddvq_f32(acc0);
            float sum1 = vaddvq_f32(acc1);
            float sum2 = vaddvq_f32(acc2);
            float sum3 = vaddvq_f32(acc3);
            for (; k < K; ++k) {
                float ak = a_row[k];
                sum0 += ak * b[(size_t)n * K + k];
                sum1 += ak * b[(size_t)(n+1) * K + k];
                sum2 += ak * b[(size_t)(n+2) * K + k];
                sum3 += ak * b[(size_t)(n+3) * K + k];
            }
            o_row[n]   = sum0;
            o_row[n+1] = sum1;
            o_row[n+2] = sum2;
            o_row[n+3] = sum3;
        }
        // Remaining rows
        for (; n < N; ++n) {
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            int k = 0;
            for (; k + 7 < K; k += 8) {
                float32x4_t a0 = vld1q_f32(a_row + k);
                float32x4_t a1 = vld1q_f32(a_row + k + 4);
                acc0 = vmlaq_f32(acc0, a0, vld1q_f32(b + (size_t)n * K + k));
                acc1 = vmlaq_f32(acc1, a1, vld1q_f32(b + (size_t)n * K + k + 4));
            }
            for (; k + 3 < K; k += 4) {
                acc0 = vmlaq_f32(acc0, vld1q_f32(a_row + k), vld1q_f32(b + (size_t)n * K + k));
            }
            float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
            for (; k < K; ++k) sum += a_row[k] * b[(size_t)n * K + k];
            o_row[n] = sum;
        }
    }
}

// block_q8_0_act and quantize_row_q8_0_act are defined in vec_dot.h.
// This file provides its own inline GEMV implementations that call them.

// ---- Q4_0 GEMV: quantize a once per row, then dot product with all weight rows ----
// Q4_0 block = 18 bytes: d[2](fp16) + qs[16](32 nibbles).
// K is the total activation dimension, must equal N * 32 * nb.
// Supports both M=1 (decode) and M>1 (prefill) by looping over M.
static inline void gemv_q4_0_transB_neon(const float* a, const uint8_t* w,
                                          float* out, int M, int K, int N) {
    const int QK = 32;
    const int BLOCK_BYTES = 18;
    int nb = K / QK;
    int blocks_per_row = nb;

    for (int m = 0; m < M; ++m) {
        const float* a_row = a + (size_t)m * K;
        float* o_row = out + (size_t)m * N;

    // Quantize activation to Q8_0_act blocks
    // Using stack for small-activation decode case (common: K <= 4096 → 128 blocks)
    // For prefill (M large), caller should use gemm path instead.
    scratch_vec<block_q8_0_act> q8_act(blocks_per_row);
    quantize_row_q8_0_act(a_row, q8_act.data(), K);

    const int NR = 4;
    int n = 0;
    for (; n + NR - 1 < N; n += NR) {
        float32x4_t acc = vdupq_n_f32(0.0f);
        const uint8_t* w_ptrs[4] = {
            w + n * blocks_per_row * BLOCK_BYTES,
            w + (n+1) * blocks_per_row * BLOCK_BYTES,
            w + (n+2) * blocks_per_row * BLOCK_BYTES,
            w + (n+3) * blocks_per_row * BLOCK_BYTES,
        };

        for (int bi = 0; bi < nb; ++bi) {
            float scale_a = q8_act[bi].d;
            int8x16_t q8_lo = vld1q_s8(q8_act[bi].qs);
            int8x16_t q8_hi = vld1q_s8(q8_act[bi].qs + 16);

            // Process 4 weight rows for this block
            float row_sum[4] = {0, 0, 0, 0};
            for (int r = 0; r < 4; ++r) {
                const uint8_t* blk = w_ptrs[r] + bi * BLOCK_BYTES;
                uint16_t ws;
                std::memcpy(&ws, blk, sizeof(uint16_t));
                float scale_w = 0;
                { // fp16 to fp32 inline
                    uint32_t s = (ws >> 15) & 1;
                    uint32_t e = (ws >> 10) & 0x1F;
                    uint32_t mw = ws & 0x3FF;
                    if (e == 0) scale_w = std::ldexp((float)mw / 1024.0f, -14);
                    else scale_w = std::ldexp(1.0f + (float)mw / 1024.0f, (int)e - 15);
                    if (s) scale_w = -scale_w;
                }
                float scale = scale_w * scale_a;

                // Expand and interleave nibbles to correct element order:
                // lo=[e0,e2,...,e30], hi=[e1,e3,...,e31]
                // q_lo=[e0,e1,...,e15], q_hi=[e16,e17,...,e31]
                uint8x16_t q_lo, q_hi;
                {
                    uint8x16_t tmp = vld1q_u8(blk + 2);
                    uint8x16_t lo = vandq_u8(tmp, vdupq_n_u8(0x0F));
                    uint8x16_t hi = vshrq_n_u8(tmp, 4);
                    q_lo = vzip1q_u8(lo, hi);
                    q_hi = vzip2q_u8(lo, hi);
                }

                int32_t dot_i32 = 0;
#ifdef USE_DOTPROD
                int32x4_t da = vdupq_n_s32(0);
                da = vdotq_s32(da, vreinterpretq_s8_u8(q_lo), q8_lo);
                da = vdotq_s32(da, vreinterpretq_s8_u8(q_hi), q8_hi);
                dot_i32 = vaddvq_s32(da);
                // Bias correction: nibbles [0..15] represent (nibble-8)
                int16x8_t sb_lo = vpaddlq_s8(q8_lo);
                int16x8_t sb_hi = vpaddlq_s8(q8_hi);
                int32x4_t sb32 = vaddq_s32(vpaddlq_s16(sb_lo), vpaddlq_s16(sb_hi));
                dot_i32 -= vaddvq_s32(sb32) * 8;
#else
                int8x16_t n_lo = vreinterpretq_s8_u8(vsubq_u8(q_lo, vdupq_n_u8(8)));
                int8x16_t n_hi = vreinterpretq_s8_u8(vsubq_u8(q_hi, vdupq_n_u8(8)));
                int16x8_t p0 = vmull_s8(vget_low_s8(n_lo), vget_low_s8(q8_lo));
                int16x8_t p1 = vmull_s8(vget_high_s8(n_lo), vget_high_s8(q8_lo));
                int16x8_t p2 = vmull_s8(vget_low_s8(n_hi), vget_low_s8(q8_hi));
                int16x8_t p3 = vmull_s8(vget_high_s8(n_hi), vget_high_s8(q8_hi));
                int32x4_t s01 = vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1));
                int32x4_t s23 = vaddq_s32(vpaddlq_s16(p2), vpaddlq_s16(p3));
                dot_i32 = vaddvq_s32(vaddq_s32(s01, s23));
#endif
                row_sum[r] += scale * (float)dot_i32;
            }
            acc = vaddq_f32(acc, vld1q_f32(row_sum));
        }
        vst1q_f32(o_row + n, acc);
    }
    // Remaining rows
    for (; n < N; ++n) {
        float sum = 0.0f;
        const uint8_t* w_row = w + n * blocks_per_row * BLOCK_BYTES;
        for (int bi = 0; bi < nb; ++bi) {
            uint16_t ws;
            std::memcpy(&ws, w_row + bi * BLOCK_BYTES, sizeof(uint16_t));
            float scale_w = 0;
            {
                uint32_t s = (ws >> 15) & 1;
                uint32_t e = (ws >> 10) & 0x1F;
                uint32_t mw = ws & 0x3FF;
                if (e == 0) scale_w = std::ldexp((float)mw / 1024.0f, -14);
                else scale_w = std::ldexp(1.0f + (float)mw / 1024.0f, (int)e - 15);
                if (s) scale_w = -scale_w;
            }
            float scale = scale_w * q8_act[bi].d;

            // Expand and interleave nibbles to correct element order
            uint8x16_t q_lo, q_hi;
            {
                uint8x16_t tmp = vld1q_u8(w_row + bi * BLOCK_BYTES + 2);
                uint8x16_t lo = vandq_u8(tmp, vdupq_n_u8(0x0F));
                uint8x16_t hi = vshrq_n_u8(tmp, 4);
                q_lo = vzip1q_u8(lo, hi);
                q_hi = vzip2q_u8(lo, hi);
            }
            int8x16_t q8_lo = vld1q_s8(q8_act[bi].qs);
            int8x16_t q8_hi = vld1q_s8(q8_act[bi].qs + 16);

            int32_t dot_i32 = 0;
#ifdef USE_DOTPROD
            int32x4_t da = vdupq_n_s32(0);
            da = vdotq_s32(da, vreinterpretq_s8_u8(q_lo), q8_lo);
            da = vdotq_s32(da, vreinterpretq_s8_u8(q_hi), q8_hi);
            dot_i32 = vaddvq_s32(da);
            // Bias correction: nibbles [0..15] represent (nibble-8)
            int16x8_t sb_lo = vpaddlq_s8(q8_lo);
            int16x8_t sb_hi = vpaddlq_s8(q8_hi);
            int32x4_t sb32 = vaddq_s32(vpaddlq_s16(sb_lo), vpaddlq_s16(sb_hi));
            dot_i32 -= vaddvq_s32(sb32) * 8;
#else
            int8x16_t n_lo = vreinterpretq_s8_u8(vsubq_u8(q_lo, vdupq_n_u8(8)));
            int8x16_t n_hi = vreinterpretq_s8_u8(vsubq_u8(q_hi, vdupq_n_u8(8)));
            int16x8_t p0 = vmull_s8(vget_low_s8(n_lo), vget_low_s8(q8_lo));
            int16x8_t p1 = vmull_s8(vget_high_s8(n_lo), vget_high_s8(q8_lo));
            int16x8_t p2 = vmull_s8(vget_low_s8(n_hi), vget_low_s8(q8_hi));
            int16x8_t p3 = vmull_s8(vget_high_s8(n_hi), vget_high_s8(q8_hi));
            int32x4_t s01 = vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1));
            int32x4_t s23 = vaddq_s32(vpaddlq_s16(p2), vpaddlq_s16(p3));
            dot_i32 = vaddvq_s32(vaddq_s32(s01, s23));
#endif
            sum += scale * (float)dot_i32;
        }
        o_row[n] = sum;
    }
    }  // end M loop
}

// ---- Q8_0 GEMV: same pattern as Q4_0 but simpler (no nibble expansion) ----
// Q8_0 block = 34 bytes: d[2](fp16) + qs[32](signed int8)
// Supports both M=1 (decode) and M>1 (prefill) by looping over M.
static inline void gemv_q8_0_transB_neon(const float* a, const uint8_t* w,
                                          float* out, int M, int K, int N) {
    const int QK = 32;
    const int BLOCK_BYTES = 34;
    int nb = K / QK;
    int blocks_per_row = nb;

    for (int m = 0; m < M; ++m) {
        const float* a_row = a + (size_t)m * K;
        float* o_row = out + (size_t)m * N;

    scratch_vec<block_q8_0_act> q8_act(blocks_per_row);
    quantize_row_q8_0_act(a_row, q8_act.data(), K);

    int n = 0;
    for (; n < N; ++n) {
        float sum = 0.0f;
        const uint8_t* w_row = w + n * blocks_per_row * BLOCK_BYTES;
        for (int bi = 0; bi < nb; ++bi) {
            uint16_t ws;
            std::memcpy(&ws, w_row + bi * BLOCK_BYTES, sizeof(uint16_t));
            float scale_w = 0;
            {
                uint32_t sgn = (ws >> 15) & 1;
                uint32_t e = (ws >> 10) & 0x1F;
                uint32_t mw = ws & 0x3FF;
                if (e == 0) scale_w = std::ldexp((float)mw / 1024.0f, -14);
                else scale_w = std::ldexp(1.0f + (float)mw / 1024.0f, (int)e - 15);
                if (sgn) scale_w = -scale_w;
            }
            float scale = scale_w * q8_act[bi].d;

            int8x16_t w_lo = vld1q_s8(reinterpret_cast<const int8_t*>(w_row + bi * BLOCK_BYTES + 2));
            int8x16_t w_hi = vld1q_s8(reinterpret_cast<const int8_t*>(w_row + bi * BLOCK_BYTES + 18));
            int8x16_t a_lo = vld1q_s8(q8_act[bi].qs);
            int8x16_t a_hi = vld1q_s8(q8_act[bi].qs + 16);

            int32_t dot_i32 = 0;
#ifdef USE_DOTPROD
            int32x4_t da = vdupq_n_s32(0);
            da = vdotq_s32(da, w_lo, a_lo);
            da = vdotq_s32(da, w_hi, a_hi);
            dot_i32 = vaddvq_s32(da);
#else
            int16x8_t p0 = vmull_s8(vget_low_s8(w_lo), vget_low_s8(a_lo));
            int16x8_t p1 = vmull_s8(vget_high_s8(w_lo), vget_high_s8(a_lo));
            int16x8_t p2 = vmull_s8(vget_low_s8(w_hi), vget_low_s8(a_hi));
            int16x8_t p3 = vmull_s8(vget_high_s8(w_hi), vget_high_s8(a_hi));
            int32x4_t s01 = vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1));
            int32x4_t s23 = vaddq_s32(vpaddlq_s16(p2), vpaddlq_s16(p3));
            dot_i32 = vaddvq_s32(vaddq_s32(s01, s23));
#endif
            sum += scale * (float)dot_i32;
        }
        o_row[n] = sum;
    }
    }  // end M loop
}

// ---- Q4_1 GEMV: q = d * nibble + m ----
static inline void gemv_q4_1_transB_neon(const float* a, const uint8_t* w,
                                          float* out, int M, int K, int N) {
    constexpr int QK = 32;
    constexpr int BLOCK_BYTES = 20;
    const int nb = K / QK;

    for (int m = 0; m < M; ++m) {
        const float* a_row = a + (size_t)m * K;
        float* o_row = out + (size_t)m * N;
        scratch_vec<block_q8_0_act> q8_act(nb);
        quantize_row_q8_0_act(a_row, q8_act.data(), K);

        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            const uint8_t* w_row = w + (size_t)n * nb * BLOCK_BYTES;
            for (int bi = 0; bi < nb; ++bi) {
                const uint8_t* blk = w_row + (size_t)bi * BLOCK_BYTES;
                uint16_t d_bits, m_bits;
                std::memcpy(&d_bits, blk, sizeof(d_bits));
                std::memcpy(&m_bits, blk + 2, sizeof(m_bits));
                float d = arm_dot::fp16_to_fp32(d_bits) * q8_act[bi].d;
                float min = arm_dot::fp16_to_fp32(m_bits) * q8_act[bi].d;

                uint8x16_t packed = vld1q_u8(blk + 4);
                uint8x16_t lo = vandq_u8(packed, vdupq_n_u8(0x0F));
                uint8x16_t hi = vshrq_n_u8(packed, 4);
                uint8x16_t q_lo = vzip1q_u8(lo, hi);
                uint8x16_t q_hi = vzip2q_u8(lo, hi);
                int8x16_t a_lo = vld1q_s8(q8_act[bi].qs);
                int8x16_t a_hi = vld1q_s8(q8_act[bi].qs + 16);

                int32_t dot;
#ifdef USE_DOTPROD
                int32x4_t acc = vdupq_n_s32(0);
                acc = vdotq_s32(acc, vreinterpretq_s8_u8(q_lo), a_lo);
                acc = vdotq_s32(acc, vreinterpretq_s8_u8(q_hi), a_hi);
                dot = vaddvq_s32(acc);
#else
                int16x8_t p0 = vmull_s8(vget_low_s8(vreinterpretq_s8_u8(q_lo)), vget_low_s8(a_lo));
                int16x8_t p1 = vmull_s8(vget_high_s8(vreinterpretq_s8_u8(q_lo)), vget_high_s8(a_lo));
                int16x8_t p2 = vmull_s8(vget_low_s8(vreinterpretq_s8_u8(q_hi)), vget_low_s8(a_hi));
                int16x8_t p3 = vmull_s8(vget_high_s8(vreinterpretq_s8_u8(q_hi)), vget_high_s8(a_hi));
                dot = vaddvq_s32(vaddq_s32(vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)),
                                           vaddq_s32(vpaddlq_s16(p2), vpaddlq_s16(p3))));
#endif
                int sum_q8 = 0;
                for (int j = 0; j < QK; ++j) sum_q8 += q8_act[bi].qs[j];
                sum += d * dot + min * sum_q8;
            }
            o_row[n] = sum;
        }
    }
}

// ---- Q5_0 GEMV: value = 5bit - 16, 5bit = nibble | (bit5 << 4) ----
// Q5_0 block = 22 bytes: d[2](fp16) + qh[4](bit5) + qs[16](nibbles)
static inline void gemv_q5_0_transB_neon(const float* a, const uint8_t* w,
                                         float* out, int M, int K, int N) {
    constexpr int QK = 32;
    constexpr int BLOCK_BYTES = 22;
    const int nb = K / QK;

    for (int m = 0; m < M; ++m) {
        const float* a_row = a + (size_t)m * K;
        float* o_row = out + (size_t)m * N;
        scratch_vec<block_q8_0_act> q8_act(nb);
        quantize_row_q8_0_act(a_row, q8_act.data(), K);

        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            const uint8_t* w_row = w + (size_t)n * nb * BLOCK_BYTES;
            for (int bi = 0; bi < nb; ++bi) {
                const uint8_t* blk = w_row + (size_t)bi * BLOCK_BYTES;
                uint16_t d_bits;
                std::memcpy(&d_bits, blk, sizeof(d_bits));
                float combined_scale = arm_dot::fp16_to_fp32(d_bits) * q8_act[bi].d;

                alignas(16) int8_t vals[QK];
                const uint8_t* qh = blk + 2;
                const uint8_t* qs = blk + 6;
                for (int j = 0; j < QK; ++j) {
                    const int nib = (qs[j >> 1] >> ((j & 1) << 2)) & 0x0F;
                    const int bit5 = (qh[j >> 3] >> (j & 7)) & 1;
                    vals[j] = (int8_t)(nib | (bit5 << 4));
                }
                int8x16_t w_lo = vld1q_s8(vals);
                int8x16_t w_hi = vld1q_s8(vals + 16);
                int8x16_t a_lo = vld1q_s8(q8_act[bi].qs);
                int8x16_t a_hi = vld1q_s8(q8_act[bi].qs + 16);

                int32_t dot;
#ifdef USE_DOTPROD
                int32x4_t acc = vdupq_n_s32(0);
                acc = vdotq_s32(acc, w_lo, a_lo);
                acc = vdotq_s32(acc, w_hi, a_hi);
                dot = vaddvq_s32(acc);
#else
                int16x8_t p0 = vmull_s8(vget_low_s8(w_lo), vget_low_s8(a_lo));
                int16x8_t p1 = vmull_s8(vget_high_s8(w_lo), vget_high_s8(a_lo));
                int16x8_t p2 = vmull_s8(vget_low_s8(w_hi), vget_low_s8(a_hi));
                int16x8_t p3 = vmull_s8(vget_high_s8(w_hi), vget_high_s8(a_hi));
                int32x4_t s01 = vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1));
                int32x4_t s23 = vaddq_s32(vpaddlq_s16(p2), vpaddlq_s16(p3));
                dot = vaddvq_s32(vaddq_s32(s01, s23));
#endif
                int sum_q8 = 0;
                for (int j = 0; j < QK; ++j) sum_q8 += q8_act[bi].qs[j];
                sum += combined_scale * (float)(dot - 16 * sum_q8);
            }
            o_row[n] = sum;
        }
    }
}

// ---- Q5_1 GEMV: value = 5bit * d + m ----
// Q5_1 block = 24 bytes: d[2](fp16) + m[2](fp16) + qh[4](bit5) + qs[16](nibbles)
static inline void gemv_q5_1_transB_neon(const float* a, const uint8_t* w,
                                         float* out, int M, int K, int N) {
    constexpr int QK = 32;
    constexpr int BLOCK_BYTES = 24;
    const int nb = K / QK;

    for (int m = 0; m < M; ++m) {
        const float* a_row = a + (size_t)m * K;
        float* o_row = out + (size_t)m * N;
        scratch_vec<block_q8_0_act> q8_act(nb);
        quantize_row_q8_0_act(a_row, q8_act.data(), K);

        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            const uint8_t* w_row = w + (size_t)n * nb * BLOCK_BYTES;
            for (int bi = 0; bi < nb; ++bi) {
                const uint8_t* blk = w_row + (size_t)bi * BLOCK_BYTES;
                uint16_t d_bits, m_bits;
                std::memcpy(&d_bits, blk, sizeof(d_bits));
                std::memcpy(&m_bits, blk + 2, sizeof(m_bits));
                float d = arm_dot::fp16_to_fp32(d_bits) * q8_act[bi].d;
                float min = arm_dot::fp16_to_fp32(m_bits) * q8_act[bi].d;

                alignas(16) int8_t vals[QK];
                const uint8_t* qh = blk + 4;
                const uint8_t* qs = blk + 8;
                for (int j = 0; j < QK; ++j) {
                    const int nib = (qs[j >> 1] >> ((j & 1) << 2)) & 0x0F;
                    const int bit5 = (qh[j >> 3] >> (j & 7)) & 1;
                    vals[j] = (int8_t)(nib | (bit5 << 4));
                }
                int8x16_t w_lo = vld1q_s8(vals);
                int8x16_t w_hi = vld1q_s8(vals + 16);
                int8x16_t a_lo = vld1q_s8(q8_act[bi].qs);
                int8x16_t a_hi = vld1q_s8(q8_act[bi].qs + 16);

                int32_t dot;
#ifdef USE_DOTPROD
                int32x4_t acc = vdupq_n_s32(0);
                acc = vdotq_s32(acc, w_lo, a_lo);
                acc = vdotq_s32(acc, w_hi, a_hi);
                dot = vaddvq_s32(acc);
#else
                int16x8_t p0 = vmull_s8(vget_low_s8(w_lo), vget_low_s8(a_lo));
                int16x8_t p1 = vmull_s8(vget_high_s8(w_lo), vget_high_s8(a_lo));
                int16x8_t p2 = vmull_s8(vget_low_s8(w_hi), vget_low_s8(a_hi));
                int16x8_t p3 = vmull_s8(vget_high_s8(w_hi), vget_high_s8(a_hi));
                int32x4_t s01 = vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1));
                int32x4_t s23 = vaddq_s32(vpaddlq_s16(p2), vpaddlq_s16(p3));
                dot = vaddvq_s32(vaddq_s32(s01, s23));
#endif
                int sum_q8 = 0;
                for (int j = 0; j < QK; ++j) sum_q8 += q8_act[bi].qs[j];
                sum += d * (float)dot + min * (float)sum_q8;
            }
            o_row[n] = sum;
        }
    }
}

#endif // USE_NEON

}  // namespace cpu
}  // namespace forge
