#pragma once
// PowerPC64 VSX GEMV kernels — decode-phase (M=1) matrix-vector products.
// Provides gemv_fp32_transB_vsx, gemv_q4_0_transB_vsx, gemv_q8_0_transB_vsx.
// All functions compute: out = a * w^T, where a is [1, K] and w is [N, K].
//
// VSX has NO dotprod instruction, so we always use the vec_mule/vec_mulo
// + vec_sum4s fallback path for nibble×q8 dot products.

#ifdef USE_VSX
#include <altivec.h>
#endif
#include <cstring>
#include <cmath>
#include <vector>

#include "vec.h"
#include "vec_dot.h"

namespace forge {
namespace cpu {

#ifdef USE_VSX

// ---- FP32 GEMV: out[n] = sum_k(a[k] * w[n][k]) ----
// NR=4 row-level unrolling. Each iteration processes 4 output rows,
// dotting a[K] with 4 rows of w concurrently via VSX vec_madd.
static inline void gemv_fp32_transB_vsx(const float* a, const float* b,
                                         float* out, int M, int K, int N) {
    // M=1 decode path only
    const int NR = 4;
    int n = 0;
    for (; n + NR - 1 < N; n += NR) {
        __vector float acc0 = vec_splats(0.0f);
        __vector float acc1 = vec_splats(0.0f);
        __vector float acc2 = vec_splats(0.0f);
        __vector float acc3 = vec_splats(0.0f);

        int k = 0;
        for (; k + 3 < K; k += 4) {
            __vector float av = vec_xl(0, a + k);
            acc0 = vec_madd(av, vec_xl(0, b + (size_t)n * K + k), acc0);
            acc1 = vec_madd(av, vec_xl(0, b + (size_t)(n+1) * K + k), acc1);
            acc2 = vec_madd(av, vec_xl(0, b + (size_t)(n+2) * K + k), acc2);
            acc3 = vec_madd(av, vec_xl(0, b + (size_t)(n+3) * K + k), acc3);
        }
        // Scalar tail
        float sum0 = hsum_f32x4(acc0);
        float sum1 = hsum_f32x4(acc1);
        float sum2 = hsum_f32x4(acc2);
        float sum3 = hsum_f32x4(acc3);
        for (; k < K; ++k) {
            float ak = a[k];
            sum0 += ak * b[(size_t)n * K + k];
            sum1 += ak * b[(size_t)(n+1) * K + k];
            sum2 += ak * b[(size_t)(n+2) * K + k];
            sum3 += ak * b[(size_t)(n+3) * K + k];
        }
        out[n]   = sum0;
        out[n+1] = sum1;
        out[n+2] = sum2;
        out[n+3] = sum3;
    }
    // Remaining rows with dual-accumulator
    for (; n < N; ++n) {
        __vector float acc0 = vec_splats(0.0f);
        __vector float acc1 = vec_splats(0.0f);
        int k = 0;
        for (; k + 7 < K; k += 8) {
            __vector float a0 = vec_xl(0, a + k);
            __vector float a1 = vec_xl(0, a + k + 4);
            acc0 = vec_madd(a0, vec_xl(0, b + (size_t)n * K + k), acc0);
            acc1 = vec_madd(a1, vec_xl(0, b + (size_t)n * K + k + 4), acc1);
        }
        for (; k + 3 < K; k += 4) {
            acc0 = vec_madd(vec_xl(0, a + k), vec_xl(0, b + (size_t)n * K + k), acc0);
        }
        float sum = hsum_f32x4(acc0 + acc1);
        for (; k < K; ++k) sum += a[k] * b[(size_t)n * K + k];
        out[n] = sum;
    }
}

// block_q8_0_act and quantize_row_q8_0_act are defined in vec_dot.h.
// This file provides its own inline GEMV implementations that call them.

// ---- Helper: software fp16_to_fp32 inline (self-contained for GEMV) ----
static inline float gemv_fp16_to_fp32(uint16_t bits) {
    uint32_t sign     = (bits >> 15) & 1;
    uint32_t exponent = (bits >> 10) & 0x1F;
    uint32_t mantissa = bits & 0x3FF;
    float value;
    if (exponent == 0) {
        value = std::ldexp(static_cast<float>(mantissa) / 1024.0f, -14);
    } else {
        value = std::ldexp(1.0f + static_cast<float>(mantissa) / 1024.0f,
                           static_cast<int>(exponent) - 15);
    }
    return sign ? -value : value;
}

// ---- Q4_0 GEMV: quantize a once, then dot product with all weight rows ----
// Q4_0 block = 18 bytes: d[2](fp16) + qs[16](32 nibbles).
// K is the total activation dimension.
static inline void gemv_q4_0_transB_vsx(const float* a, const uint8_t* w,
                                         float* out, int M, int K, int N) {
    const int QK = 32;
    const int BLOCK_BYTES = 18;
    int nb = K / QK;
    int blocks_per_row = nb;

    // Quantize activation to Q8_0_act blocks
    std::vector<block_q8_0_act> q8_act(blocks_per_row);
    quantize_row_q8_0_act(a, q8_act.data(), K);

    const int NR = 4;
    int n = 0;
    for (; n + NR - 1 < N; n += NR) {
        __vector float acc = vec_splats(0.0f);
        const uint8_t* w_ptrs[4] = {
            w + (size_t)n * blocks_per_row * BLOCK_BYTES,
            w + (size_t)(n+1) * blocks_per_row * BLOCK_BYTES,
            w + (size_t)(n+2) * blocks_per_row * BLOCK_BYTES,
            w + (size_t)(n+3) * blocks_per_row * BLOCK_BYTES,
        };

        for (int bi = 0; bi < nb; ++bi) {
            float scale_a = q8_act[bi].d;
            __vector signed char q8_lo = (__vector signed char)vec_xl(0, q8_act[bi].qs);
            __vector signed char q8_hi = (__vector signed char)vec_xl(0, q8_act[bi].qs + 16);

            // Process 4 weight rows for this block
            float row_sum[4] = {0, 0, 0, 0};
            for (int r = 0; r < 4; ++r) {
                const uint8_t* blk = w_ptrs[r] + (size_t)bi * BLOCK_BYTES;
                uint16_t ws;
                std::memcpy(&ws, blk, sizeof(uint16_t));
                float scale_w = gemv_fp16_to_fp32(ws);
                float scale = scale_w * scale_a;

                // Expand and interleave nibbles to correct element order:
                // lo=[e0,e2,...,e30], hi=[e1,e3,...,e31]
                // q_lo=[e0,e1,...,e15], q_hi=[e16,e17,...,e31]
                __vector unsigned char q4_bytes = vec_xl(0, (const uint8_t*)(blk + 2));
                __vector unsigned char lo = q4_bytes & vec_splats((unsigned char)0x0F);
                __vector unsigned char hi = vec_sr(q4_bytes, vec_splats((unsigned char)4));
                // Interleave via vec_perm
                static const __vector unsigned char mz1 = {
                     0, 16,  1, 17,  2, 18,  3, 19,
                     4, 20,  5, 21,  6, 22,  7, 23
                };
                static const __vector unsigned char mz2 = {
                     8, 24,  9, 25, 10, 26, 11, 27,
                    12, 28, 13, 29, 14, 30, 15, 31
                };
                __vector unsigned char q_lo = vec_perm(lo, hi, mz1);
                __vector unsigned char q_hi = vec_perm(lo, hi, mz2);

                // Dot using vec_mule/mulo + vec_sum4s (nibbles [0..15], bias -8*sum(q8))
                __vector signed char q_lo_s = (__vector signed char)q_lo;
                __vector signed char q_hi_s = (__vector signed char)q_hi;

                __vector signed short pe0 = vec_mule(q_lo_s, q8_lo);
                __vector signed short po0 = vec_mulo(q_lo_s, q8_lo);
                __vector signed short pe1 = vec_mule(q_hi_s, q8_hi);
                __vector signed short po1 = vec_mulo(q_hi_s, q8_hi);

                __vector signed int zero = vec_splats((int)0);
                __vector signed int se0 = vec_sum4s(pe0, zero);
                __vector signed int so0 = vec_sum4s(po0, zero);
                __vector signed int se1 = vec_sum4s(pe1, zero);
                __vector signed int so1 = vec_sum4s(po1, zero);

                __vector signed int dot_vec = (se0 + so0) + (se1 + so1);
                union { __vector signed int vi; int i[4]; } u;
                u.vi = dot_vec;
                int dot_i32 = u.i[0] + u.i[1] + u.i[2] + u.i[3];

                // Bias correction: nibbles [0..15] represent (nibble - 8)
                int sum_q8 = 0;
                for (int j = 0; j < 32; ++j) sum_q8 += (int)q8_act[bi].qs[j];
                dot_i32 -= sum_q8 * 8;

                row_sum[r] += scale * (float)dot_i32;
            }
            acc = acc + vec_xl(0, row_sum);
        }
        vec_xst(acc, 0, out + n);
    }
    // Remaining rows
    for (; n < N; ++n) {
        float sum = 0.0f;
        const uint8_t* w_row = w + (size_t)n * blocks_per_row * BLOCK_BYTES;
        for (int bi = 0; bi < nb; ++bi) {
            uint16_t ws;
            std::memcpy(&ws, w_row + (size_t)bi * BLOCK_BYTES, sizeof(uint16_t));
            float scale_w = gemv_fp16_to_fp32(ws);
            float scale = scale_w * q8_act[bi].d;

            __vector unsigned char q4_bytes = vec_xl(0, (const uint8_t*)(w_row + (size_t)bi * BLOCK_BYTES + 2));
            __vector unsigned char lo = q4_bytes & vec_splats((unsigned char)0x0F);
            __vector unsigned char hi = vec_sr(q4_bytes, vec_splats((unsigned char)4));
            static const __vector unsigned char mz1 = {
                 0, 16,  1, 17,  2, 18,  3, 19,
                 4, 20,  5, 21,  6, 22,  7, 23
            };
            static const __vector unsigned char mz2 = {
                 8, 24,  9, 25, 10, 26, 11, 27,
                12, 28, 13, 29, 14, 30, 15, 31
            };
            __vector unsigned char q_lo = vec_perm(lo, hi, mz1);
            __vector unsigned char q_hi = vec_perm(lo, hi, mz2);

            __vector signed char q8_lo = (__vector signed char)vec_xl(0, q8_act[bi].qs);
            __vector signed char q8_hi = (__vector signed char)vec_xl(0, q8_act[bi].qs + 16);

            __vector signed char q_lo_s = (__vector signed char)q_lo;
            __vector signed char q_hi_s = (__vector signed char)q_hi;

            __vector signed short pe0 = vec_mule(q_lo_s, q8_lo);
            __vector signed short po0 = vec_mulo(q_lo_s, q8_lo);
            __vector signed short pe1 = vec_mule(q_hi_s, q8_hi);
            __vector signed short po1 = vec_mulo(q_hi_s, q8_hi);

            __vector signed int zero = vec_splats((int)0);
            __vector signed int dot_vec = vec_sum4s(pe0, zero) + vec_sum4s(po0, zero)
                                      + vec_sum4s(pe1, zero) + vec_sum4s(po1, zero);
            union { __vector signed int vi; int i[4]; } u;
            u.vi = dot_vec;
            int dot_i32 = u.i[0] + u.i[1] + u.i[2] + u.i[3];

            // Bias correction
            int sum_q8 = 0;
            for (int j = 0; j < 32; ++j) sum_q8 += (int)q8_act[bi].qs[j];
            dot_i32 -= sum_q8 * 8;

            sum += scale * (float)dot_i32;
        }
        out[n] = sum;
    }
}

// ---- Q8_0 GEMV: same pattern as Q4_0 but simpler (no nibble expansion) ----
// Q8_0 block = 34 bytes: d[2](fp16) + qs[32](signed int8)
static inline void gemv_q8_0_transB_vsx(const float* a, const uint8_t* w,
                                         float* out, int M, int K, int N) {
    const int QK = 32;
    const int BLOCK_BYTES = 34;
    int nb = K / QK;
    int blocks_per_row = nb;

    std::vector<block_q8_0_act> q8_act(blocks_per_row);
    quantize_row_q8_0_act(a, q8_act.data(), K);

    int n = 0;
    for (; n < N; ++n) {
        float sum = 0.0f;
        const uint8_t* w_row = w + (size_t)n * blocks_per_row * BLOCK_BYTES;
        for (int bi = 0; bi < nb; ++bi) {
            uint16_t ws;
            std::memcpy(&ws, w_row + (size_t)bi * BLOCK_BYTES, sizeof(uint16_t));
            float scale_w = gemv_fp16_to_fp32(ws);
            float scale = scale_w * q8_act[bi].d;

            __vector signed char w_lo = (__vector signed char)vec_xl(0, w_row + (size_t)bi * BLOCK_BYTES + 2);
            __vector signed char w_hi = (__vector signed char)vec_xl(0, w_row + (size_t)bi * BLOCK_BYTES + 18);
            __vector signed char a_lo = (__vector signed char)vec_xl(0, q8_act[bi].qs);
            __vector signed char a_hi = (__vector signed char)vec_xl(0, q8_act[bi].qs + 16);

            // Signed×signed dot via vec_mule/mulo
            __vector signed short pe0 = vec_mule(w_lo, a_lo);
            __vector signed short po0 = vec_mulo(w_lo, a_lo);
            __vector signed short pe1 = vec_mule(w_hi, a_hi);
            __vector signed short po1 = vec_mulo(w_hi, a_hi);

            __vector signed int zero = vec_splats((int)0);
            __vector signed int dot_vec = vec_sum4s(pe0, zero) + vec_sum4s(po0, zero)
                                      + vec_sum4s(pe1, zero) + vec_sum4s(po1, zero);
            union { __vector signed int vi; int i[4]; } u;
            u.vi = dot_vec;
            int dot_i32 = u.i[0] + u.i[1] + u.i[2] + u.i[3];

            sum += scale * (float)dot_i32;
        }
        out[n] = sum;
    }
}

#endif // USE_VSX

}  // namespace cpu
}  // namespace forge
