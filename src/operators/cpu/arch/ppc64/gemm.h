#pragma once
// PowerPC64 VSX GEMM micro-kernels for batch decode (M >= 1, both decode & prefill).
// Provides gemm_q4_K_vsx, gemm_q6_K_vsx, gemm_q2_K_vsx, gemm_q3_K_vsx, gemm_q5_K_vsx.
// All functions compute: out = a @ w^T, where a is [M, K] and w is [N, K] in quantized format.
//
// Decode (M=1): act quantized once, then RM=4 row-grouped dot product against weight rows.
// Prefill (M>1): work-stealing over N, shared weight block decoding per N-column.
//
// VSX has NO dotprod instruction — always uses vec_mule/vec_mulo + vec_sum4s fallback.

#ifdef USE_VSX
#include <altivec.h>
#endif
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <atomic>

#include "forge/types.h"
#include "../../common/quant_helpers.h"
#include "../../common/scalar.h"

namespace forge {
namespace cpu {

#ifdef USE_VSX

// ============================================================================
// Q4_K scale decoding helper — extracts 8 scale + 8 min uint6 values from
// the packed 12-byte scales field. Uses the same bit-twiddling as the x86/arm64 paths.
// ============================================================================

static inline void decode_q4_k_scales_vsx(const uint8_t scales[12],
                                          uint8_t sc[8], uint8_t mn[8]) {
    constexpr uint32_t kmask1 = 0x3f3f3f3f;
    constexpr uint32_t kmask2 = 0x0f0f0f0f;
    constexpr uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4] = {0, 0, 0, 0};
    std::memcpy(utmp, scales, 12);

    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;

    // Extract each uint6 value byte-wise
    for (int i = 0; i < 4; ++i) {
        sc[i]     = (utmp[0] >> (8 * i)) & 0x3F;
        sc[i + 4] = (utmp[1] >> (8 * i)) & 0x3F;
        mn[i]     = (utmp[2] >> (8 * i)) & 0x3F;
        mn[i + 4] = (utmp[3] >> (8 * i)) & 0x3F;
    }
}

// ============================================================================
// Q4_K GEMM: decode (M=1) and prefill (M>1)
// Q4_K block: 144 bytes per 256 elements
// Layout: d[2](fp16) + dmin[2](fp16) + scales[12] + qs[128]
// 8 sub-blocks of 32 elements each.
// Value = scale_sb * nibble + min_sb (super-block scale + min applied after).
// ============================================================================

static void gemm_q4_K_vsx(
    const float* a_data,
    const uint8_t* w_data,
    float* o_data,
    int M, int K, int N)
{
    constexpr int QK_K = 256;
    constexpr int Q4_K_BLOCK_BYTES = 144;
    const int nb = (K + QK_K - 1) / QK_K;

    // Quantize all M activation rows to Q8_K (one-time cost)
    std::vector<block_q8_K> q8_all((size_t)M * nb);
    for (int m = 0; m < M; ++m)
        quantize_row_q8_K(a_data + (size_t)m * K, q8_all.data() + (size_t)m * nb, K);

    // Interleave masks for nibble expansion (VSX vec_perm)
    static const __vector unsigned char mz1 = {
         0, 16,  1, 17,  2, 18,  3, 19,
         4, 20,  5, 21,  6, 22,  7, 23
    };
    static const __vector unsigned char mz2 = {
         8, 24,  9, 25, 10, 26, 11, 27,
        12, 28, 13, 29, 14, 30, 15, 31
    };

    if (M == 1) {
        // Decode: RM=4 row grouping, Q8_K shared across 4 weight rows
        const size_t row_bytes = (size_t)nb * Q4_K_BLOCK_BYTES;
        #pragma omp parallel for schedule(static)
        for (int n_base = 0; n_base < N; n_base += 4) {
            int rows = (n_base + 4 <= N) ? 4 : (N - n_base);
            for (int r = 0; r < rows; ++r) {
                const uint8_t* q4_row = w_data + (size_t)(n_base + r) * row_bytes;
                float sum = 0.0f;

                for (int bi = 0; bi < nb; ++bi) {
                    const block_q4_K* x =
                        reinterpret_cast<const block_q4_K*>(q4_row) + bi;
                    const block_q8_K* y = q8_all.data() + bi;

                    const float d = y->d * fp16_to_float_scalar(x->d);
                    const float dmin_ratio =
                        -fp16_to_float_scalar(x->dmin) / fp16_to_float_scalar(x->d);

                    // Decode 12-byte scales → 8 sc + 8 mn
                    uint8_t sc[8], mn[8];
                    decode_q4_k_scales_vsx(x->scales, sc, mn);

                    const uint8_t* q4_qs = x->qs;
                    const int8_t* q8_qs = y->qs;
                    float block_acc = 0.0f;

                    for (int sb = 0; sb < 8; ++sb) {
                        // Expand 32 nibbles from 16 bytes
                        __vector unsigned char q4_bytes = vec_xl(0, q4_qs + sb * 16);
                        __vector unsigned char q4_lo = q4_bytes & vec_splats((unsigned char)0x0F);
                        __vector unsigned char q4_hi = vec_sr(q4_bytes, vec_splats((unsigned char)4));

                        // Load 32 Q8 values
                        __vector signed char q8_0 = (__vector signed char)vec_xl(0, q8_qs + sb * 32);
                        __vector signed char q8_1 = (__vector signed char)vec_xl(0, q8_qs + sb * 32 + 16);

                        // Interleave even/odd nibbles for correct Q8 pairing
                        // lo=[e0,e2,...,e30], hi=[e1,e3,...,e31]
                        // Interleave → [e0,e1,...,e15], [e16,e17,...,e31]
                        __vector unsigned char q4_0 = vec_perm(q4_lo, q4_hi, mz1);
                        __vector unsigned char q4_1 = vec_perm(q4_lo, q4_hi, mz2);

                        // Dot product via vec_mule/mulo + vec_sum4s
                        __vector signed char q4_0_s = (__vector signed char)q4_0;
                        __vector signed char q4_1_s = (__vector signed char)q4_1;
                        __vector signed char q8_0_s = q8_0;
                        __vector signed char q8_1_s = q8_1;

                        __vector signed short pe0 = vec_mule(q4_0_s, q8_0_s);
                        __vector signed short po0 = vec_mulo(q4_0_s, q8_0_s);
                        __vector signed short pe1 = vec_mule(q4_1_s, q8_1_s);
                        __vector signed short po1 = vec_mulo(q4_1_s, q8_1_s);

                        __vector signed int zero = vec_splats((int)0);
                        __vector signed int dot_vec = vec_sum4s(pe0, zero) + vec_sum4s(po0, zero)
                                                  + vec_sum4s(pe1, zero) + vec_sum4s(po1, zero);

                        union { __vector signed int vi; int i[4]; } u;
                        u.vi = dot_vec;
                        int dot_sb = u.i[0] + u.i[1] + u.i[2] + u.i[3];

                        // Sum of Q8 for min correction
                        int q8sum = 0;
                        for (int j = 0; j < 32; ++j) q8sum += (int)q8_qs[sb * 32 + j];

                        block_acc += (float)dot_sb * (float)sc[sb] +
                                     dmin_ratio * (float)q8sum * (float)mn[sb];
                    }

                    sum += d * block_acc;
                }
                o_data[n_base + r] = sum;
            }
        }
    } else {
        // Prefill: work-stealing over N, shared Q4_K decoding per N-column
        std::atomic<int> next_n{0};
        #pragma omp parallel
        {
            // Stack accumulators (up to M=64)
            float acc_vec[64];
            while (true) {
                int n = next_n.fetch_add(1, std::memory_order_relaxed);
                if (n >= N)
                    break;

                const uint8_t* q4_row = w_data +
                    (size_t)n * (size_t)nb * Q4_K_BLOCK_BYTES;

                for (int m = 0; m < M; ++m)
                    acc_vec[m] = 0.0f;

                for (int bi = 0; bi < nb; ++bi) {
                    const block_q4_K* x =
                        reinterpret_cast<const block_q4_K*>(q4_row) + bi;

                    // Shared Q4_K decode
                    const float d_half = fp16_to_float_scalar(x->d);
                    const float dmin_half = fp16_to_float_scalar(x->dmin);
                    const float inv_d = 1.0f / d_half;

                    uint8_t sc[8], mn[8];
                    decode_q4_k_scales_vsx(x->scales, sc, mn);

                    const uint8_t* q4_qs = x->qs;

                    // Per activation row: load Q8_K and dot
                    for (int m = 0; m < M; ++m) {
                        const block_q8_K* y =
                            q8_all.data() + (size_t)m * nb + bi;

                        const float d = y->d * d_half;
                        const float dmin_ratio = -dmin_half * inv_d;

                        const int8_t* q8_qs = y->qs;
                        float block_acc = 0.0f;

                        for (int sb = 0; sb < 8; ++sb) {
                            __vector unsigned char q4_bytes = vec_xl(0, q4_qs + sb * 16);
                            __vector unsigned char q4_lo = q4_bytes & vec_splats((unsigned char)0x0F);
                            __vector unsigned char q4_hi = vec_sr(q4_bytes, vec_splats((unsigned char)4));

                            __vector signed char q8_0 = (__vector signed char)vec_xl(0, q8_qs + sb * 32);
                            __vector signed char q8_1 = (__vector signed char)vec_xl(0, q8_qs + sb * 32 + 16);

                            __vector unsigned char q4_0 = vec_perm(q4_lo, q4_hi, mz1);
                            __vector unsigned char q4_1 = vec_perm(q4_lo, q4_hi, mz2);

                            __vector signed char q4_0_s = (__vector signed char)q4_0;
                            __vector signed char q4_1_s = (__vector signed char)q4_1;

                            __vector signed short pe0 = vec_mule(q4_0_s, q8_0);
                            __vector signed short po0 = vec_mulo(q4_0_s, q8_0);
                            __vector signed short pe1 = vec_mule(q4_1_s, q8_1);
                            __vector signed short po1 = vec_mulo(q4_1_s, q8_1);

                            __vector signed int zero = vec_splats((int)0);
                            __vector signed int dot_vec = vec_sum4s(pe0, zero) + vec_sum4s(po0, zero)
                                                      + vec_sum4s(pe1, zero) + vec_sum4s(po1, zero);

                            union { __vector signed int vi; int i[4]; } u;
                            u.vi = dot_vec;
                            int dot_sb = u.i[0] + u.i[1] + u.i[2] + u.i[3];

                            int q8sum = 0;
                            for (int j = 0; j < 32; ++j) q8sum += (int)q8_qs[sb * 32 + j];

                            block_acc += (float)dot_sb * (float)sc[sb] +
                                         dmin_ratio * (float)q8sum * (float)mn[sb];
                        }

                        acc_vec[m] += d * block_acc;
                    }
                }

                for (int m = 0; m < M; ++m)
                    o_data[(size_t)m * N + n] = acc_vec[m];
            }
        }
    }
}

// ============================================================================
// Q6_K GEMM: decode (M=1) and prefill (M>1)
// Q6_K block: 210 bytes per 256 elements
// Layout: ql[128] + qh[64] + scales[16] int8 + d[2] fp16
// 16 sub-blocks of 16 elements each.
// Value = (ql_lo4 | (qh_hi2 << 4)) - 32, then × scale_sb × d
// Uses scalar decode loop for each sub-block, then VSX mule/mulo dot with Q8.
// ============================================================================

static void gemm_q6_K_vsx(
    const float* a_data,
    const uint8_t* w_data,
    float* o_data,
    int M, int K, int N)
{
    constexpr int QK_K = 256;
    constexpr int Q6_K_BLOCK_BYTES = 210;
    const int nb = (K + QK_K - 1) / QK_K;

    std::vector<block_q8_K> q8_all((size_t)M * nb);
    for (int m = 0; m < M; ++m)
        quantize_row_q8_K(a_data + (size_t)m * K, q8_all.data() + (size_t)m * nb, K);

    if (M == 1) {
        const size_t row_bytes = (size_t)nb * Q6_K_BLOCK_BYTES;
        #pragma omp parallel for schedule(static)
        for (int n_base = 0; n_base < N; n_base += 4) {
            int rows = (n_base + 4 <= N) ? 4 : (N - n_base);
            for (int r = 0; r < rows; ++r) {
                const uint8_t* q6_row = w_data + (size_t)(n_base + r) * row_bytes;
                float sum = 0.0f;

                for (int bi = 0; bi < nb; ++bi) {
                    const block_q6_K* x =
                        reinterpret_cast<const block_q6_K*>(q6_row) + bi;
                    const block_q8_K* y = q8_all.data() + bi;

                    const float d = y->d * fp16_to_float_scalar(x->d);

                    const uint8_t* ql = x->ql;
                    const uint8_t* qh = x->qh;
                    const int8_t* sc = x->scales;
                    const int8_t* q8_qs = y->qs;
                    float block_acc = 0.0f;

                    for (int sb = 0; sb < 16; ++sb) {
                        // Scalar decode of 16 Q6_K values into int8 array
                        alignas(16) signed char q6_vals[16];
                        for (int j = 0; j < 8; ++j) {
                            uint8_t ql_byte = ql[j];
                            uint8_t qh_bits_lo = qh[j / 2];
                            uint8_t qh_bits_hi = qh[j / 2 + 2];
                            int shift = (j % 2) * 2;
                            q6_vals[2*j]   = (int8_t)((ql_byte & 0x0F) | (((qh_bits_lo >> shift) & 3) << 4)) - 32;
                            q6_vals[2*j+1] = (int8_t)((ql_byte >> 4)   | (((qh_bits_hi >> shift) & 3) << 4)) - 32;
                        }

                        __vector signed char q6_vec = (__vector signed char)vec_xl(0, (const signed char*)q6_vals);
                        __vector signed char q8_vec = (__vector signed char)vec_xl(0, (const signed char*)q8_qs);

                        // Dot via mule/mulo + sum4s
                        __vector signed short pe = vec_mule(q6_vec, q8_vec);
                        __vector signed short po = vec_mulo(q6_vec, q8_vec);
                        __vector signed int zero = vec_splats((int)0);
                        __vector signed int se = vec_sum4s(pe, zero);
                        __vector signed int so = vec_sum4s(po, zero);
                        __vector signed int dot_vec = se + so;

                        union { __vector signed int v; int i[4]; } u;
                        u.v = dot_vec;
                        int dot_sb = u.i[0] + u.i[1] + u.i[2] + u.i[3];

                        block_acc += (float)dot_sb * (float)sc[sb];

                        ql += 8;
                        qh += 4;
                        q8_qs += 16;
                    }

                    sum += d * block_acc;
                }
                o_data[n_base + r] = sum;
            }
        }
    } else {
        // Prefill: work-stealing over N
        std::atomic<int> next_n{0};
        #pragma omp parallel
        {
            float acc_vec[64];
            while (true) {
                int n = next_n.fetch_add(1, std::memory_order_relaxed);
                if (n >= N)
                    break;

                const uint8_t* q6_row = w_data +
                    (size_t)n * (size_t)nb * Q6_K_BLOCK_BYTES;

                for (int m = 0; m < M; ++m)
                    acc_vec[m] = 0.0f;

                for (int bi = 0; bi < nb; ++bi) {
                    const block_q6_K* x =
                        reinterpret_cast<const block_q6_K*>(q6_row) + bi;

                    const float d_half = fp16_to_float_scalar(x->d);
                    const uint8_t* ql_base = x->ql;
                    const uint8_t* qh_base = x->qh;
                    const int8_t* sc_base = x->scales;

                    for (int m = 0; m < M; ++m) {
                        const block_q8_K* y =
                            q8_all.data() + (size_t)m * nb + bi;

                        const float d = y->d * d_half;
                        const uint8_t* ql = ql_base;
                        const uint8_t* qh = qh_base;
                        const int8_t* sc = sc_base;
                        const int8_t* q8_qs = y->qs;
                        float block_acc = 0.0f;

                        for (int sb = 0; sb < 16; ++sb) {
                            alignas(16) signed char q6_vals[16];
                            for (int j = 0; j < 8; ++j) {
                                uint8_t ql_byte = ql[j];
                                uint8_t qh_bits_lo = qh[j / 2];
                                uint8_t qh_bits_hi = qh[j / 2 + 2];
                                int shift = (j % 2) * 2;
                                q6_vals[2*j]   = (int8_t)((ql_byte & 0x0F) | (((qh_bits_lo >> shift) & 3) << 4)) - 32;
                                q6_vals[2*j+1] = (int8_t)((ql_byte >> 4)   | (((qh_bits_hi >> shift) & 3) << 4)) - 32;
                            }

                            __vector signed char q6_vec = (__vector signed char)vec_xl(0, (const signed char*)q6_vals);
                            __vector signed char q8_vec = (__vector signed char)vec_xl(0, (const signed char*)q8_qs);

                            __vector signed short pe = vec_mule(q6_vec, q8_vec);
                            __vector signed short po = vec_mulo(q6_vec, q8_vec);
                            __vector signed int zero = vec_splats((int)0);
                            __vector signed int se = vec_sum4s(pe, zero);
                            __vector signed int so = vec_sum4s(po, zero);
                            __vector signed int dot_vec = se + so;

                            union { __vector signed int v; int i[4]; } u;
                            u.v = dot_vec;
                            int dot_sb = u.i[0] + u.i[1] + u.i[2] + u.i[3];

                            block_acc += (float)dot_sb * (float)sc[sb];

                            ql += 8;
                            qh += 4;
                            q8_qs += 16;
                        }

                        acc_vec[m] += d * block_acc;
                    }
                }

                for (int m = 0; m < M; ++m)
                    o_data[(size_t)m * N + n] = acc_vec[m];
            }
        }
    }
}

// ============================================================================
// Q2_K GEMM fallback: dequantize + fp32 gemv
// ============================================================================

static void gemm_q2_K_vsx(
    const float* a_data,
    const uint8_t* w_data,
    float* o_data,
    int M, int K, int N)
{
    // Dequantize weight rows to fp32, then use fp32 gemv
    std::vector<float> w_fp32((size_t)N * K);
    auto dequant_fn = get_dequant_row_fn(DataType::Q2_K);
    if (!dequant_fn) {
        // Cannot dequantize — zero output
        std::memset(o_data, 0, (size_t)M * N * sizeof(float));
        return;
    }
    #pragma omp parallel for schedule(dynamic)
    for (int n = 0; n < N; ++n)
        dequant_fn(w_data, w_fp32.data() + (size_t)n * K, K, n);

    // Reuse fp32 gemv (defined in gemv.h, included via kernels.h)
    gemv_fp32_transB_vsx(a_data, w_fp32.data(), o_data, M, K, N);
}

// ============================================================================
// Q3_K GEMM fallback: dequantize + fp32 gemv
// ============================================================================

static void gemm_q3_K_vsx(
    const float* a_data,
    const uint8_t* w_data,
    float* o_data,
    int M, int K, int N)
{
    std::vector<float> w_fp32((size_t)N * K);
    auto dequant_fn = get_dequant_row_fn(DataType::Q3_K);
    if (!dequant_fn) {
        std::memset(o_data, 0, (size_t)M * N * sizeof(float));
        return;
    }
    #pragma omp parallel for schedule(dynamic)
    for (int n = 0; n < N; ++n)
        dequant_fn(w_data, w_fp32.data() + (size_t)n * K, K, n);

    gemv_fp32_transB_vsx(a_data, w_fp32.data(), o_data, M, K, N);
}

// ============================================================================
// Q5_K GEMM fallback: dequantize + fp32 gemv
// ============================================================================

static void gemm_q5_K_vsx(
    const float* a_data,
    const uint8_t* w_data,
    float* o_data,
    int M, int K, int N)
{
    std::vector<float> w_fp32((size_t)N * K);
    auto dequant_fn = get_dequant_row_fn(DataType::Q5_K);
    if (!dequant_fn) {
        std::memset(o_data, 0, (size_t)M * N * sizeof(float));
        return;
    }
    #pragma omp parallel for schedule(dynamic)
    for (int n = 0; n < N; ++n)
        dequant_fn(w_data, w_fp32.data() + (size_t)n * K, K, n);

    gemv_fp32_transB_vsx(a_data, w_fp32.data(), o_data, M, K, N);
}

#endif // USE_VSX

}  // namespace cpu
}  // namespace forge
