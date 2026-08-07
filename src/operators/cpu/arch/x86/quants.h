#pragma once
// x86 AVX2 quantized kernel implementations (FORGE_ARCH_X86).
// Moved from matmul.cpp during the arch-split refactor (see cpu_arch_split_plan.md).
// IQ2_S / IQ2_XS / IQ3_S / IQ4_NL fused dot-product and GEMV kernels.

#ifdef USE_AVX2

#include <immintrin.h>
#include <cmath>
#include <vector>

#include "../../common/quant_helpers.h"
#include "../../common/quant_tables.h"
#include "forge/types.h"

namespace forge {
namespace ops {

// ============================================================================
// IQ2_S x Q8_K fused dot product (AVX2)
// IQ2_S block: 82 bytes = d[2] + qs[32B] + signs[32B] + qh[8B] + sc[8B]
// ============================================================================

static inline float dot_iq2_s_q8_K_avx2(const uint8_t* iq2s_row,
                                        const cpu::block_q8_K* q8, int nb) {
    constexpr int QK_K = 256;
    constexpr int IQ2_S_BLOCK_SIZE = 82;

    float acc = 0.0f;

    for (int bi = 0; bi < nb; ++bi) {
        const uint8_t* block_ptr = iq2s_row + (size_t)bi * IQ2_S_BLOCK_SIZE;
        float d = cpu::fp16_to_float_scalar(*reinterpret_cast<const uint16_t*>(block_ptr));
        const uint8_t* qs = block_ptr + 2;      // 32 value bytes (4 per ib32)
        const uint8_t* signs = qs + 32;          // 32 sign bytes  (4 per ib32)
        const uint8_t* qh = block_ptr + 66;      // 8 high-bit bytes (1 per ib32)
        const uint8_t* sc = block_ptr + 74;      // 8 scale bytes   (1 per ib32)

        const cpu::block_q8_K* y = q8 + bi;
        const float d_q8 = y->d;
        const int8_t* q8d = y->qs;

        float block_acc = 0.0f;

        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            const float db0_f = (0.5f + (sc[ib32] & 0xf)) * 0.25f;
            const float db1_f = (0.5f + (sc[ib32] >> 4)) * 0.25f;

            alignas(32) uint8_t grid_vals[32];
            alignas(32) uint8_t sign_mask[32];
            for (int l = 0; l < 4; ++l) {
                const int grid_idx = qs[l] | ((qh[ib32] << (8 - 2 * l)) & 0x300);
                memcpy(grid_vals + l * 8, &iq2s_grid[grid_idx], 8);
                const uint8_t s = signs[l];
                for (int j = 0; j < 8; ++j)
                    sign_mask[l * 8 + j] = (s & kmask_iq2xs[j]) ? 0xFF : 0x00;
            }

            const __m256i grid_vec = _mm256_load_si256((const __m256i*)grid_vals);
            const __m256i q8_vec = _mm256_loadu_si256((const __m256i*)(q8d + ib32 * 32));
            const __m256i sign_vec = _mm256_load_si256((const __m256i*)sign_mask);

            const __m256i zero = _mm256_setzero_si256();
            const __m256i q8_neg = _mm256_sub_epi8(zero, q8_vec);
            const __m256i q8_signed = _mm256_blendv_epi8(q8_vec, q8_neg, sign_vec);

            const __m256i p16 = _mm256_maddubs_epi16(grid_vec, q8_signed);
            const __m256i p32 = _mm256_madd_epi16(_mm256_set1_epi16(1), p16);

            const __m128i lo = _mm256_extracti128_si256(p32, 0);
            const __m128i hi = _mm256_extracti128_si256(p32, 1);
            __m128i h = _mm_hadd_epi32(lo, hi);
            h = _mm_hadd_epi32(h, h);
            const int s_l01 = _mm_cvtsi128_si32(h);
            const int s_l23 = _mm_extract_epi32(h, 1);

            block_acc += (float)s_l01 * db0_f + (float)s_l23 * db1_f;

            qs += 4;
            signs += 4;
        }

        acc += d * d_q8 * block_acc;
    }

    return acc;
}

static void gemv_iq2_s_q8k_transB_avx2(const float* a, const uint8_t* w, float* out,
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
            out[n] = dot_iq2_s_q8_K_avx2(row, q8_buf.data(), nb);
        }
    } else {
        std::vector<cpu::block_q8_K> q8_all((size_t)M * nb);
        for (int m = 0; m < M; ++m)
            cpu::quantize_row_q8_K(a + m * K, q8_all.data() + (size_t)m * nb, K);

#pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const uint8_t* row = w + (size_t)n * nb * IQ2_S_BLOCK_SIZE;
            for (int m = 0; m < M; ++m)
                out[m * N + n] = dot_iq2_s_q8_K_avx2(row, q8_all.data() + (size_t)m * nb, nb);
        }
    }
}

// ============================================================================
// IQ2_XS x Q8_K fused dot product (AVX2)
// IQ2_XS block: 74 bytes = d[2] + qs[32 uint16_t = 64B] + scales[8B]
// ============================================================================

static inline float dot_iq2_xs_q8_K_avx2(const uint8_t* iq2xs_row,
                                         const cpu::block_q8_K* q8, int nb) {
    constexpr int QK_K = 256;
    constexpr int IQ2_XS_BLOCK_SIZE = 74;

    float acc = 0.0f;

    for (int bi = 0; bi < nb; ++bi) {
        const uint8_t* block_ptr = iq2xs_row + (size_t)bi * IQ2_XS_BLOCK_SIZE;
        float d = cpu::fp16_to_float_scalar(*reinterpret_cast<const uint16_t*>(block_ptr));
        const uint16_t* qs = reinterpret_cast<const uint16_t*>(block_ptr + 2);
        const uint8_t* scales = block_ptr + 2 + 64;

        const cpu::block_q8_K* y = q8 + bi;
        const float d_q8 = y->d;
        const int8_t* q8d = y->qs;

        float block_acc = 0.0f;

        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            const float db0_f = (0.5f + (scales[ib32] & 0xf)) * 0.25f;
            const float db1_f = (0.5f + (scales[ib32] >> 4)) * 0.25f;

            alignas(32) uint8_t grid_vals[32];
            alignas(32) uint8_t sign_mask[32];
            for (int l = 0; l < 4; ++l) {
                const uint16_t qs_u16 = qs[4 * ib32 + l];
                memcpy(grid_vals + l * 8, &iq2xs_grid[qs_u16 & 511], 8);
                const uint8_t signs_byte = ksigns_iq2xs[qs_u16 >> 9];
                for (int j = 0; j < 8; ++j)
                    sign_mask[l * 8 + j] = (signs_byte & kmask_iq2xs[j]) ? 0xFF : 0x00;
            }

            const __m256i grid_vec = _mm256_load_si256((const __m256i*)grid_vals);
            const __m256i q8_vec = _mm256_loadu_si256((const __m256i*)(q8d + ib32 * 32));
            const __m256i sign_vec = _mm256_load_si256((const __m256i*)sign_mask);

            const __m256i zero = _mm256_setzero_si256();
            const __m256i q8_neg = _mm256_sub_epi8(zero, q8_vec);
            const __m256i q8_signed = _mm256_blendv_epi8(q8_vec, q8_neg, sign_vec);

            const __m256i p16 = _mm256_maddubs_epi16(grid_vec, q8_signed);
            const __m256i p32 = _mm256_madd_epi16(_mm256_set1_epi16(1), p16);

            const __m128i lo = _mm256_extracti128_si256(p32, 0);
            const __m128i hi = _mm256_extracti128_si256(p32, 1);
            __m128i h = _mm_hadd_epi32(lo, hi);
            h = _mm_hadd_epi32(h, h);
            const int s_l01 = _mm_cvtsi128_si32(h);
            const int s_l23 = _mm_extract_epi32(h, 1);

            block_acc += (float)s_l01 * db0_f + (float)s_l23 * db1_f;
        }

        acc += d * d_q8 * block_acc;
    }

    return acc;
}

static void gemv_iq2_xs_q8k_transB_avx2(const float* a, const uint8_t* w, float* out,
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
            out[n] = dot_iq2_xs_q8_K_avx2(row, q8_buf.data(), nb);
        }
    } else {
        std::vector<cpu::block_q8_K> q8_all((size_t)M * nb);
        for (int m = 0; m < M; ++m)
            cpu::quantize_row_q8_K(a + m * K, q8_all.data() + (size_t)m * nb, K);

#pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const uint8_t* row = w + (size_t)n * nb * IQ2_XS_BLOCK_SIZE;
            for (int m = 0; m < M; ++m)
                out[m * N + n] = dot_iq2_xs_q8_K_avx2(row, q8_all.data() + (size_t)m * nb, nb);
        }
    }
}

// ============================================================================
// IQ3_S x Q8_K fused dot product (AVX2)
// IQ3_S block: 110 bytes = d[2] + qs[64B] + qh[8B] + signs[32B] + scales[4B]
// ============================================================================

static inline float dot_iq3_s_q8_K_avx2(const uint8_t* iq3s_row,
                                        const cpu::block_q8_K* q8, int nb) {
    constexpr int QK_K = 256;
    constexpr int IQ3_S_BLOCK_SIZE = 110;

    float acc = 0.0f;

    for (int bi = 0; bi < nb; ++bi) {
        const uint8_t* block_ptr = iq3s_row + (size_t)bi * IQ3_S_BLOCK_SIZE;
        float d = cpu::fp16_to_float_scalar(*reinterpret_cast<const uint16_t*>(block_ptr));
        const uint8_t* qs = block_ptr + 2;
        const uint8_t* qh = block_ptr + 2 + 64;
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

                alignas(32) uint8_t grid_vals[32];
                alignas(32) uint8_t sign_mask[32];
                for (int l = 0; l < 4; ++l) {
                    const int idx1 = qs[2 * l + 0] | ((qh[half] << (8 - 2 * l)) & 256);
                    const int idx2 = qs[2 * l + 1] | ((qh[half] << (7 - 2 * l)) & 256);
                    memcpy(grid_vals + l * 8 + 0, &iq3s_grid[idx1], 4);
                    memcpy(grid_vals + l * 8 + 4, &iq3s_grid[idx2], 4);
                    const uint8_t s = signs[l];
                    for (int j = 0; j < 8; ++j)
                        sign_mask[l * 8 + j] = (s & kmask_iq2xs[j]) ? 0xFF : 0x00;
                }

                const __m256i grid_vec = _mm256_load_si256((const __m256i*)grid_vals);
                const __m256i q8_vec = _mm256_loadu_si256((const __m256i*)(q8d + ib32 * 32 + half * 32));
                const __m256i sign_vec = _mm256_load_si256((const __m256i*)sign_mask);

                const __m256i zero = _mm256_setzero_si256();
                const __m256i q8_neg = _mm256_sub_epi8(zero, q8_vec);
                const __m256i q8_signed = _mm256_blendv_epi8(q8_vec, q8_neg, sign_vec);

                const __m256i p16 = _mm256_maddubs_epi16(grid_vec, q8_signed);
                const __m256i p32 = _mm256_madd_epi16(_mm256_set1_epi16(1), p16);

                const __m128i lo = _mm256_extracti128_si256(p32, 0);
                const __m128i hi = _mm256_extracti128_si256(p32, 1);
                __m128i h = _mm_hadd_epi32(lo, hi);
                h = _mm_hadd_epi32(h, h);
                h = _mm_hadd_epi32(h, h);
                const int s_all = _mm_cvtsi128_si32(h);

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

static void gemv_iq3_s_q8k_transB_avx2(const float* a, const uint8_t* w, float* out,
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
            out[n] = dot_iq3_s_q8_K_avx2(row, q8_buf.data(), nb);
        }
    } else {
        std::vector<cpu::block_q8_K> q8_all((size_t)M * nb);
        for (int m = 0; m < M; ++m)
            cpu::quantize_row_q8_K(a + m * K, q8_all.data() + (size_t)m * nb, K);

#pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const uint8_t* row = w + (size_t)n * nb * IQ3_S_BLOCK_SIZE;
            for (int m = 0; m < M; ++m)
                out[m * N + n] = dot_iq3_s_q8_K_avx2(row, q8_all.data() + (size_t)m * nb, nb);
        }
    }
}

// ============================================================================
// IQ4_NL x Q8_K fused dot product (AVX2)
// IQ4_NL block: 18 bytes = d[2] + qs[16B] per 32 elements
// ============================================================================

static inline float dot_iq4_nl_q8_K_avx2(const uint8_t* iq4nl_row,
                                         const cpu::block_q8_K* q8, int nb) {
    constexpr int QK_K = 256;
    constexpr int IQ4_NL_BLOCK_SIZE = 18;
    constexpr int IQ4_NL_BLOCK_EL = 32;

    const __m128i m4 = _mm_set1_epi8(0x0F);
    const __m128i kvalues = _mm_loadu_si128((const __m128i*)kvalues_iq4nl);

    float acc = 0.0f;

    for (int bi = 0; bi < nb; ++bi) {
        const cpu::block_q8_K* y = q8 + bi;
        const float d_q8 = y->d;
        const int8_t* q8d = y->qs;

        const uint8_t* block_ptr = iq4nl_row + (size_t)bi * (QK_K / IQ4_NL_BLOCK_EL) * IQ4_NL_BLOCK_SIZE;
        float block_acc = 0.0f;

        for (int sb = 0; sb < QK_K / IQ4_NL_BLOCK_EL; ++sb) {
            float d = cpu::fp16_to_float_scalar(*reinterpret_cast<const uint16_t*>(block_ptr));
            const uint8_t* qs = block_ptr + 2;

            __m128i qs_vec = _mm_loadu_si128((const __m128i*)qs);
            __m128i low_nibbles = _mm_and_si128(qs_vec, m4);
            __m128i high_nibbles = _mm_and_si128(_mm_srli_epi16(qs_vec, 4), m4);

            __m128i vals_low = _mm_shuffle_epi8(kvalues, low_nibbles);
            __m128i vals_high = _mm_shuffle_epi8(kvalues, high_nibbles);

            __m128i q8_low = _mm_loadu_si128((const __m128i*)(q8d + sb * 32));
            __m128i q8_high = _mm_loadu_si128((const __m128i*)(q8d + sb * 32 + 16));

            __m256i vals_lo16 = _mm256_cvtepi8_epi16(vals_low);
            __m256i q8_lo16 = _mm256_cvtepi8_epi16(q8_low);
            __m256i p_lo = _mm256_madd_epi16(vals_lo16, q8_lo16);

            __m256i vals_hi16 = _mm256_cvtepi8_epi16(vals_high);
            __m256i q8_hi16 = _mm256_cvtepi8_epi16(q8_high);
            __m256i p_hi = _mm256_madd_epi16(vals_hi16, q8_hi16);

            __m256i p32 = _mm256_add_epi32(p_lo, p_hi);
            __m128i lo128 = _mm256_extracti128_si256(p32, 0);
            __m128i hi128 = _mm256_extracti128_si256(p32, 1);
            __m128i h = _mm_hadd_epi32(lo128, hi128);
            h = _mm_hadd_epi32(h, h);
            h = _mm_hadd_epi32(h, h);
            const int sum = _mm_cvtsi128_si32(h);

            block_acc += d * (float)sum;
            block_ptr += IQ4_NL_BLOCK_SIZE;
        }

        acc += d_q8 * block_acc;
    }

    return acc;
}

static void gemv_iq4_nl_q8k_transB_avx2(const float* a, const uint8_t* w, float* out,
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
            out[n] = dot_iq4_nl_q8_K_avx2(row, q8_buf.data(), nb);
        }
    } else {
        std::vector<cpu::block_q8_K> q8_all((size_t)M * nb);
        for (int m = 0; m < M; ++m)
            cpu::quantize_row_q8_K(a + m * K, q8_all.data() + (size_t)m * nb, K);

#pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const uint8_t* row = w + (size_t)n * row_bytes;
            for (int m = 0; m < M; ++m)
                out[m * N + n] = dot_iq4_nl_q8_K_avx2(row, q8_all.data() + (size_t)m * nb, nb);
        }
    }
}

// ============================================================================
// DotQ8KFn dispatcher for batched MoE matmul
// ============================================================================

using DotQ8KFn = float (*)(const uint8_t*, const cpu::block_q8_K*, int);

static inline DotQ8KFn get_dot_q8k_fn(DataType dt) {
    switch (dt) {
    case DataType::IQ2_XS: return dot_iq2_xs_q8_K_avx2;
    case DataType::IQ3_S:  return dot_iq3_s_q8_K_avx2;
    case DataType::IQ4_NL: return dot_iq4_nl_q8_K_avx2;
    case DataType::IQ2_S:  return dot_iq2_s_q8_K_avx2;
    default: return nullptr;
    }
}

// ============================================================================
// Q4_K x Q8_K fused dot product (AVX2) — used by mixed-precision QKV fusion
// ============================================================================

static inline float dot_q4_K_q8_K_row_avx2(const uint8_t* q4_row, const cpu::block_q8_K* q8, int nb) {
    constexpr int QK_K = 256;
    const __m256i m4 = _mm256_set1_epi8(0xF);
    __m256 acc = _mm256_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    for (int i = 0; i < nb; ++i) {
        const cpu::block_q4_K* x = reinterpret_cast<const cpu::block_q4_K*>(q4_row) + i;
        const cpu::block_q8_K* y = q8 + i;

        _mm_prefetch((const char*)((const cpu::block_q4_K*)q4_row + i + 1), _MM_HINT_T0);
        _mm_prefetch((const char*)(q8 + i + 1), _MM_HINT_T0);

        const float d = y->d * cpu::fp16_to_float_scalar(x->d);
        const float dmin = -y->d * cpu::fp16_to_float_scalar(x->dmin);

        uint32_t utmp[4];
        memcpy(utmp, x->scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(
            _mm_set_epi32((int)utmp[3], (int)utmp[2], (int)utmp[1], (int)utmp[0]));

        const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y->bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0),
                                           _mm256_extracti128_si256(q8sums, 1));
        const __m128i mins128 = _mm256_extracti128_si256(mins_and_scales, 1);
        const __m128i prod = _mm_madd_epi16(mins128, q8s);
        acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = _mm256_set_m128i(sc128, sc128);

        const uint8_t* q4 = x->qs;
        const int8_t* q8d = y->qs;
        __m256i sumi = _mm256_setzero_si256();

        for (int j = 0; j < QK_K / 64; ++j) {
            const __m256i scale_l = _mm256_shuffle_epi8(scales, cpu::get_scale_shuffle_k4(2 * j + 0));
            const __m256i scale_h = _mm256_shuffle_epi8(scales, cpu::get_scale_shuffle_k4(2 * j + 1));

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

    return cpu::hsum_avx2(acc) + _mm_cvtss_f32(acc_m);
}

static void gemv_q3_k_q4_k_fused_qkv_avx2(const float* a, const uint8_t* wq, const uint8_t* wk,
                                           const uint8_t* wv, float* out_q, float* out_k,
                                           float* out_v, int K, int N_q, int N_k, int N_v) {
    constexpr int QK_K = 256;
    constexpr int Q3_K_BLOCK_BYTES = 110;
    constexpr int Q4_K_BLOCK_BYTES = 144;
    const int nb = (K + QK_K - 1) / QK_K;

    std::vector<cpu::block_q8_K> q8_buf(nb);
    cpu::quantize_row_q8_K(a, q8_buf.data(), K);

    // Q and K use Q3_K Q8_K dot product
    auto dot_q3_rows = [&](const uint8_t* w, float* out, int N) {
#pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const uint8_t* q3_row = w + (size_t)n * nb * Q3_K_BLOCK_BYTES;
            __m256 acc = _mm256_setzero_ps();
            for (int i = 0; i < nb; ++i) {
                _mm_prefetch((const char*)(q3_row + (i + 1) * Q3_K_BLOCK_BYTES), _MM_HINT_T0);
                _mm_prefetch((const char*)(q8_buf.data() + i + 1), _MM_HINT_T0);
                acc = _mm256_add_ps(
                    acc, cpu::q3_k_sb_dot_avx2(q3_row + (size_t)i * Q3_K_BLOCK_BYTES, &q8_buf[i]));
            }
            out[n] = cpu::hsum_avx2(acc);
        }
    };

    dot_q3_rows(wq, out_q, N_q);
    dot_q3_rows(wk, out_k, N_k);

    // V uses Q4_K Q8_K dot product (inline, shares same q8_buf)
#pragma omp parallel for schedule(static)
    for (int n = 0; n < N_v; ++n) {
        const uint8_t* q4_row = wv + (size_t)n * nb * Q4_K_BLOCK_BYTES;
        out_v[n] = dot_q4_K_q8_K_row_avx2(q4_row, q8_buf.data(), nb);
    }
}

// ============================================================================
// Q5_K x Q8_K fused dot product (AVX2)
// ============================================================================

static inline float dot_q5_K_q8_K_row_avx2(const uint8_t* q5_row, const cpu::block_q8_K* q8, int nb) {
    constexpr int QK_K = 256;
    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m128i mzero = _mm_setzero_si128();
    const __m256i mone  = _mm256_set1_epi8(1);

    __m256 acc = _mm256_setzero_ps();
    float summs = 0.f;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

    for (int i = 0; i < nb; ++i) {
        const cpu::block_q5_K* x = reinterpret_cast<const cpu::block_q5_K*>(q5_row) + i;
        const cpu::block_q8_K* y = q8 + i;

        _mm_prefetch((const char*)((const cpu::block_q5_K*)q5_row + i + 1), _MM_HINT_T0);
        _mm_prefetch((const char*)(q8 + i + 1), _MM_HINT_T0);

        const uint8_t* q5 = x->ql;
        const int8_t*  q8d = y->qs;

        const float d = y->d * cpu::fp16_to_float_scalar(x->d);
        const float dmin = -y->d * cpu::fp16_to_float_scalar(x->dmin);

        memcpy(utmp, x->scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(
            _mm_set_epi32((int)utmp[3], (int)utmp[2], (int)utmp[1], (int)utmp[0]));

        const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y->bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0),
                                           _mm256_extracti128_si256(q8sums, 1));
        const __m128i mins128 = _mm256_extracti128_si256(mins_and_scales, 1);
        const __m128i prod = _mm_madd_epi16(mins128, q8s);
        const __m128i hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
        summs += dmin * (float)_mm_extract_epi32(hsum, 0);

        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = _mm256_set_m128i(sc128, sc128);

        const __m256i hbits = _mm256_loadu_si256((const __m256i*)x->qh);
        __m256i hmask = mone;

        __m256i sumi = _mm256_setzero_si256();
        int bit = 0;

        for (int j = 0; j < QK_K / 64; ++j) {
            const __m256i scale_0 = _mm256_shuffle_epi8(scales, cpu::get_scale_shuffle_k4(2 * j + 0));
            const __m256i scale_1 = _mm256_shuffle_epi8(scales, cpu::get_scale_shuffle_k4(2 * j + 1));

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

    return cpu::hsum_avx2(acc) + summs;
}

static void gemv_q5_k_transB_avx2(const float* a, const uint8_t* w, float* out, int M, int K,
                                    int N) {
    constexpr int QK_K = 256;
    constexpr int Q5_K_BLOCK_BYTES = 176;
    constexpr int nrc = 2;
    const int nb = (K + QK_K - 1) / QK_K;

    if (M == 1) {
        std::vector<cpu::block_q8_K> q8_buf(nb);
        cpu::quantize_row_q8_K(a, q8_buf.data(), K);

#pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            _mm_prefetch((const char*)(w + (size_t)(n + 4) * nb * Q5_K_BLOCK_BYTES), _MM_HINT_T1);
            const uint8_t* q5_row = w + (size_t)n * nb * Q5_K_BLOCK_BYTES;
            out[n] = dot_q5_K_q8_K_row_avx2(q5_row, q8_buf.data(), nb);
        }
    } else {
        for (int m_start = 0; m_start < M; m_start += nrc) {
            int m_cur = (m_start + nrc <= M) ? nrc : (M - m_start);
            std::vector<cpu::block_q8_K> q8_tile(m_cur * nb);
            for (int m = 0; m < m_cur; ++m)
                cpu::quantize_row_q8_K(a + (m_start + m) * K, q8_tile.data() + m * nb, K);

#pragma omp parallel for schedule(static)
            for (int n = 0; n < N; ++n) {
                const uint8_t* q5_row = w + (size_t)n * nb * Q5_K_BLOCK_BYTES;
                for (int m = 0; m < m_cur; ++m)
                    out[(m_start + m) * N + n] = dot_q5_K_q8_K_row_avx2(q5_row, q8_tile.data() + m * nb, nb);
            }
        }
    }
}

static void gemv_q5_k_fused_ffn_up_avx2(const float* a, const uint8_t* w_gate,
                                         const uint8_t* w_up, float* out, int K, int N) {
    constexpr int QK_K = 256;
    constexpr int Q5_K_BLOCK_BYTES = 176;
    const int nb = (K + QK_K - 1) / QK_K;

    std::vector<cpu::block_q8_K> q8_buf(nb);
    cpu::quantize_row_q8_K(a, q8_buf.data(), K);

    auto silu = [](float x) -> float { return x / (1.0f + std::exp(-x)); };

#pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        const uint8_t* gate_row = w_gate + (size_t)n * nb * Q5_K_BLOCK_BYTES;
        const uint8_t* up_row = w_up + (size_t)n * nb * Q5_K_BLOCK_BYTES;

        float gate_val = silu(dot_q5_K_q8_K_row_avx2(gate_row, q8_buf.data(), nb));
        float up_val = dot_q5_K_q8_K_row_avx2(up_row, q8_buf.data(), nb);
        out[n] = gate_val * up_val;
    }
}

static void gemv_q2_k_fused_ffn_up_avx2(const float* a, const uint8_t* w_gate,
                                         const uint8_t* w_up, float* out, int K, int N) {
    constexpr int QK_K = 256;
    constexpr int Q2_K_BLOCK_BYTES = 84;
    const int nb = (K + QK_K - 1) / QK_K;

    std::vector<cpu::block_q8_K> q8_buf(nb);
    cpu::quantize_row_q8_K(a, q8_buf.data(), K);

    auto silu = [](float x) -> float { return x / (1.0f + std::exp(-x)); };

#pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        const uint8_t* gate_row = w_gate + (size_t)n * nb * Q2_K_BLOCK_BYTES;
        const uint8_t* up_row = w_up + (size_t)n * nb * Q2_K_BLOCK_BYTES;

        float gate_val = silu(cpu::dot_q2_K_q8_K_avx2(gate_row, q8_buf.data(), nb, nullptr));
        float up_val = cpu::dot_q2_K_q8_K_avx2(up_row, q8_buf.data(), nb, nullptr);
        out[n] = gate_val * up_val;
    }
}

}  // namespace ops
}  // namespace forge

#endif  // USE_AVX2