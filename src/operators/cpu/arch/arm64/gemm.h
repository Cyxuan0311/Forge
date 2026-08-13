#pragma once
// ARM64 NEON GEMM micro-kernels for batch decode (M >= 1, both decode & prefill).
// Provides gemm_q4_K_neon, gemm_q6_K_neon, gemm_q2_K_neon, gemm_q3_K_neon, gemm_q5_K_neon.
// All functions compute: out = a @ w^T, where a is [M, K] and w is [N, K] in quantized format.
//
// Decode (M=1): act quantized once, then RM=4 row-grouped dot product against weight rows.
// Prefill (M>1): work-stealing over N, shared weight block decoding per N-column.
//
// #ifdef USE_DOTPROD enables vdotq_s32 (ARMv8.2+dotprod, Apple M1+).
// Fallback path uses vmull_s8 + vpadal for baseline ARMv8-A NEON.

#ifdef USE_NEON
#include <arm_neon.h>
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

#ifdef USE_NEON

static inline float dot_q2_K_q8_K_row_neon(const uint8_t*, const block_q8_K*, int);
static inline float dot_q3_K_q8_K_row_neon(const uint8_t*, const block_q8_K*, int);
static inline float dot_q5_K_q8_K_row_neon(const uint8_t*, const block_q8_K*, int);
static inline float dot_q6_K_q8_K_row_neon(const uint8_t*, const block_q8_K*, int);

// ============================================================================
// Q4_K scale decoding helper — extracts 8 scale + 8 min uint6 values from
// the packed 12-byte scales field.  Uses the same bit-twiddling as the x86
// get_scale_shuffle path.
// ============================================================================

static inline void decode_q4_k_scales_neon(const uint8_t scales[12],
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

static void gemm_q4_K_neon(
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
                    decode_q4_k_scales_neon(x->scales, sc, mn);

                    const uint8_t* q4_qs = x->qs;
                    const int8_t* q8_qs = y->qs;
                    float block_acc = 0.0f;

                    for (int sb = 0; sb < 8; ++sb) {
                        // Expand 32 nibbles from 16 bytes
                        uint8x16_t q4_bytes =
                            vld1q_u8(q4_qs + sb * 16);
                        uint8x16_t q4_lo =
                            vandq_u8(q4_bytes, vdupq_n_u8(0x0F));
                        uint8x16_t q4_hi =
                            vshrq_n_u8(q4_bytes, 4);

                        // Load 32 Q8 values
                        int8x16_t q8_0 =
                            vld1q_s8(q8_qs + sb * 32);
                        int8x16_t q8_1 =
                            vld1q_s8(q8_qs + sb * 32 + 16);

                        // Dot product: interleave even/odd nibbles for correct pairing with Q8
                        // q4_lo = [n0, n2, ..., n30], q4_hi = [n1, n3, ..., n31]
                        // zip → [n0,n1,...,n15] × q8[0..15] + [n16,n17,...,n31] × q8[16..31]
                        int8x16_t q4_lo_s8 = vreinterpretq_s8_u8(q4_lo);
                        int8x16_t q4_hi_s8 = vreinterpretq_s8_u8(q4_hi);
                        int8x16_t q4_0 = vzip1q_s8(q4_lo_s8, q4_hi_s8);
                        int8x16_t q4_1 = vzip2q_s8(q4_lo_s8, q4_hi_s8);
                        int dot_sb;
#ifdef USE_DOTPROD
                        {
                            int32x4_t acc =
                                vdotq_s32(vdupq_n_s32(0), q4_0, q8_0);
                            acc = vdotq_s32(acc, q4_1, q8_1);
                            dot_sb = vaddvq_s32(acc);
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
                    decode_q4_k_scales_neon(x->scales, sc, mn);

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
                            uint8x16_t q4_bytes =
                                vld1q_u8(q4_qs + sb * 16);
                            uint8x16_t q4_lo =
                                vandq_u8(q4_bytes, vdupq_n_u8(0x0F));
                            uint8x16_t q4_hi =
                                vshrq_n_u8(q4_bytes, 4);

                            int8x16_t q8_0 =
                                vld1q_s8(q8_qs + sb * 32);
                            int8x16_t q8_1 =
                                vld1q_s8(q8_qs + sb * 32 + 16);

                            // Interleave even/odd nibbles for correct Q8 pairing
                            int8x16_t q4_lo_s8 = vreinterpretq_s8_u8(q4_lo);
                            int8x16_t q4_hi_s8 = vreinterpretq_s8_u8(q4_hi);
                            int8x16_t q4_0 = vzip1q_s8(q4_lo_s8, q4_hi_s8);
                            int8x16_t q4_1 = vzip2q_s8(q4_lo_s8, q4_hi_s8);

                            int dot_sb;
#ifdef USE_DOTPROD
                            {
                                int32x4_t acc =
                                    vdotq_s32(vdupq_n_s32(0), q4_0, q8_0);
                                acc = vdotq_s32(acc, q4_1, q8_1);
                                dot_sb = vaddvq_s32(acc);
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
#endif

                            int16x8_t q8sum_lo = vpaddlq_s8(q8_0);
                            int16x8_t q8sum_hi = vpaddlq_s8(q8_1);
                            int32x4_t q8sum32 =
                                vaddq_s32(vpaddlq_s16(q8sum_lo),
                                          vpaddlq_s16(q8sum_hi));
                            int q8sum = vaddvq_s32(q8sum32);

                            block_acc +=
                                (float)dot_sb * (float)sc[sb] +
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
// Dot product correction: subtract 32 * scale_sb * sum(q8) per sub-block.
// ============================================================================

static void gemm_q6_K_neon(
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
                        // Decode 16 Q6_K values:
                        // ql[0..7] → 16 nibbles, qh[0..3] → 16 2-bit high bits
                        alignas(16) int8_t q6_vals[16];

                        for (int j = 0; j < 8; ++j) {
                            uint8_t ql_byte = ql[j];
                            uint8_t qh_bits_lo = qh[j / 2];
                            uint8_t qh_bits_hi = qh[j / 2 + 2];
                            int shift_lo = (j % 2) * 2;
                            int shift_hi = (j % 2) * 2;
                            q6_vals[2 * j] = (int8_t)((ql_byte & 0x0F) |
                                (((qh_bits_lo >> shift_lo) & 3) << 4)) - 32;
                            q6_vals[2 * j + 1] = (int8_t)((ql_byte >> 4) |
                                (((qh_bits_hi >> shift_hi) & 3) << 4)) - 32;
                        }

                        int8x16_t q6_vec = vld1q_s8(q6_vals);
                        int8x16_t q8_vec = vld1q_s8(q8_qs);

                        int dot_sb;
#ifdef USE_DOTPROD
                        {
                            int32x4_t acc =
                                vdotq_s32(vdupq_n_s32(0), q6_vec, q8_vec);
                            dot_sb = vaddvq_s32(acc);
                        }
#else
                        {
                            int16x8_t p0 = vmull_s8(
                                vget_low_s8(q6_vec), vget_low_s8(q8_vec));
                            int16x8_t p1 = vmull_s8(
                                vget_high_s8(q6_vec), vget_high_s8(q8_vec));
                            int32x4_t s0 = vpaddlq_s16(p0);
                            int32x4_t s1 = vpaddlq_s16(p1);
                            dot_sb = vaddvq_s32(vaddq_s32(s0, s1));
                        }
#endif

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
                            alignas(16) int8_t q6_vals[16];
                            for (int j = 0; j < 8; ++j) {
                                uint8_t ql_byte = ql[j];
                                uint8_t qh_bits_lo = qh[j / 2];
                                uint8_t qh_bits_hi = qh[j / 2 + 2];
                                int shift = (j % 2) * 2;
                                q6_vals[2 * j] = (int8_t)((ql_byte & 0x0F) |
                                    (((qh_bits_lo >> shift) & 3) << 4)) - 32;
                                q6_vals[2 * j + 1] = (int8_t)((ql_byte >> 4) |
                                    (((qh_bits_hi >> shift) & 3) << 4)) - 32;
                            }

                            int8x16_t q6_vec = vld1q_s8(q6_vals);
                            int8x16_t q8_vec = vld1q_s8(q8_qs);

                            int dot_sb;
#ifdef USE_DOTPROD
                            {
                                int32x4_t acc =
                                    vdotq_s32(vdupq_n_s32(0), q6_vec, q8_vec);
                                dot_sb = vaddvq_s32(acc);
                            }
#else
                            {
                                int16x8_t p0 = vmull_s8(
                                    vget_low_s8(q6_vec), vget_low_s8(q8_vec));
                                int16x8_t p1 = vmull_s8(
                                    vget_high_s8(q6_vec), vget_high_s8(q8_vec));
                                int32x4_t s0 = vpaddlq_s16(p0);
                                int32x4_t s1 = vpaddlq_s16(p1);
                                dot_sb = vaddvq_s32(vaddq_s32(s0, s1));
                            }
#endif

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

template <typename DotFn>
static void gemm_k_row_dot_neon(const float* a_data, const uint8_t* w_data,
                                float* o_data, int M, int K, int N,
                                size_t block_bytes, DotFn dot_fn) {
    const int nb = (K + 255) / 256;
    std::vector<block_q8_K> q8_all((size_t)M * nb);
    for (int m = 0; m < M; ++m)
        quantize_row_q8_K(a_data + (size_t)m * K, q8_all.data() + (size_t)m * nb, K);
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        const uint8_t* row = w_data + (size_t)n * nb * block_bytes;
        for (int m = 0; m < M; ++m)
            o_data[(size_t)m * N + n] = dot_fn(row, q8_all.data() + (size_t)m * nb, nb);
    }
}

static void gemm_q2_K_neon(const float* a, const uint8_t* w, float* out, int M, int K, int N) {
    gemm_k_row_dot_neon(a, w, out, M, K, N, sizeof(block_q2_K), dot_q2_K_q8_K_row_neon);
}

static void gemm_q3_K_neon(const float* a, const uint8_t* w, float* out, int M, int K, int N) {
    gemm_k_row_dot_neon(a, w, out, M, K, N, sizeof(block_q3_K), dot_q3_K_q8_K_row_neon);
}

static void gemm_q5_K_neon(const float* a, const uint8_t* w, float* out, int M, int K, int N) {
    gemm_k_row_dot_neon(a, w, out, M, K, N, sizeof(block_q5_K), dot_q5_K_q8_K_row_neon);
}

#endif // USE_NEON

}  // namespace cpu
}  // namespace forge
