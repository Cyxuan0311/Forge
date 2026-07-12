#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#include "gemv.h"
#include "vec.h"
#ifdef _OPENMP
#    include <omp.h>
#endif
#ifdef USE_AVX2
#    include <immintrin.h>
#endif

namespace forge {
namespace cpu {

// ---- Fused QKV projection for Q4_0 decode ----
// Reads input vector once, computes Q + K + V outputs simultaneously.

#ifdef USE_AVX2
static void gemv_q4_0_fused_qkv_avx2(const float* a, const uint8_t* wq, const uint8_t* wk,
                                     const uint8_t* wv, float* out_q, float* out_k, float* out_v,
                                     int K, int N_q, int N_k, int N_v) {
    constexpr int BLOCK_SIZE = 32;
    constexpr int BLOCK_BYTES = 18;
    const int blocks_per_row = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const __m128i lo_mask = _mm_set1_epi8(0x0F);
    const __m128i eight = _mm_set1_epi8(8);

    auto process_row = [&](const float* a_row, const uint8_t* w_row, float* out, int N_out) {
#    pragma omp parallel for schedule(static)
        for (int n = 0; n < N_out; ++n) {
            const uint8_t* row = w_row + (size_t)n * blocks_per_row * BLOCK_BYTES;
            __m256 acc = _mm256_setzero_ps();

            for (int bi = 0; bi < blocks_per_row; ++bi) {
                int base = bi * BLOCK_SIZE;
                if (K - base < BLOCK_SIZE) {
                    const uint8_t* block = row + bi * BLOCK_BYTES;
                    uint16_t sb;
                    memcpy(&sb, block, 2);
                    uint32_t sign = (sb >> 15) & 1;
                    uint32_t exponent = (sb >> 10) & 0x1F;
                    uint32_t mantissa = sb & 0x3FF;
                    float scale_f;
                    if (exponent == 0) {
                        scale_f = mantissa == 0
                                      ? 0.0f
                                      : (sign ? -1 : 1) *
                                            std::ldexp(static_cast<float>(mantissa) / 1024.0f, -14);
                    } else {
                        scale_f = (sign ? -1 : 1) *
                                  std::ldexp(1.0f + static_cast<float>(mantissa) / 1024.0f,
                                             static_cast<int>(exponent) - 15);
                    }
                    const uint8_t* qs = block + 2;
                    for (int j = 0; j < 16 && base + j < K; ++j) {
                        float qv = static_cast<float>((qs[j] & 0x0F) - 8) * scale_f;
                        acc = _mm256_fmadd_ps(_mm256_set1_ps(a_row[base + j]), _mm256_set1_ps(qv),
                                              acc);
                    }
                    for (int j = 0; j < 16 && base + 16 + j < K; ++j) {
                        float qv = static_cast<float>(((qs[j] >> 4) & 0x0F) - 8) * scale_f;
                        acc = _mm256_fmadd_ps(_mm256_set1_ps(a_row[base + 16 + j]),
                                              _mm256_set1_ps(qv), acc);
                    }
                    continue;
                }

                const uint8_t* block = row + bi * BLOCK_BYTES;
                uint16_t scale_bits;
                memcpy(&scale_bits, block, 2);
                __m256 scale = fp16_to_fp32_broadcast_avx2(scale_bits);
                const uint8_t* qs = block + 2;

                __m128i q8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs));
                __m128i q_lo = _mm_and_si128(q8, lo_mask);
                __m128i q_lo_signed = _mm_sub_epi8(q_lo, eight);
                __m128i q_hi = _mm_and_si128(_mm_srli_epi16(q8, 4), lo_mask);
                __m128i q_hi_signed = _mm_sub_epi8(q_hi, eight);

                __m256 partial =
                    _mm256_mul_ps(_mm256_loadu_ps(a_row + base),
                                  _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_lo_signed)));
                partial = _mm256_fmadd_ps(
                    _mm256_loadu_ps(a_row + base + 8),
                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_lo_signed, 8))),
                    partial);
                partial =
                    _mm256_fmadd_ps(_mm256_loadu_ps(a_row + base + 16),
                                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_hi_signed)), partial);
                partial = _mm256_fmadd_ps(
                    _mm256_loadu_ps(a_row + base + 24),
                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_hi_signed, 8))),
                    partial);

                acc = _mm256_fmadd_ps(scale, partial, acc);
            }
            out[n] = hsum_avx2(acc);
        }
    };

    process_row(a, wq, out_q, N_q);
    process_row(a, wk, out_k, N_k);
    process_row(a, wv, out_v, N_v);
}
#endif  // USE_AVX2

// ---- Fused FFN gate+up projection for Q4_0 decode ----
// Reads input vector once, computes both gate and up projections simultaneously.
// Then applies SiLU(gate) * up in-place.

#ifdef USE_AVX2
static void gemv_q4_0_fused_ffn_up_avx2(const float* a, const uint8_t* w_gate, const uint8_t* w_up,
                                        float* out, int K, int N) {
    constexpr int BLOCK_SIZE = 32;
    constexpr int BLOCK_BYTES = 18;
    const int blocks_per_row = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const __m128i lo_mask = _mm_set1_epi8(0x0F);
    const __m128i eight = _mm_set1_epi8(8);

    auto silu = [](float x) -> float { return x / (1.0f + std::exp(-x)); };

#    pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        const uint8_t* gate_row = w_gate + (size_t)n * blocks_per_row * BLOCK_BYTES;
        const uint8_t* up_row = w_up + (size_t)n * blocks_per_row * BLOCK_BYTES;

        __m256 acc_gate = _mm256_setzero_ps();
        __m256 acc_up = _mm256_setzero_ps();

        for (int bi = 0; bi < blocks_per_row; ++bi) {
            int base = bi * BLOCK_SIZE;
            if (K - base < BLOCK_SIZE) {
                auto process_partial = [&](const uint8_t* w_row, __m256& acc_ref) {
                    const uint8_t* block = w_row + bi * BLOCK_BYTES;
                    uint16_t sb;
                    memcpy(&sb, block, 2);
                    uint32_t sign = (sb >> 15) & 1;
                    uint32_t exponent = (sb >> 10) & 0x1F;
                    uint32_t mantissa = sb & 0x3FF;
                    float scale_f;
                    if (exponent == 0) {
                        scale_f = mantissa == 0
                                      ? 0.0f
                                      : (sign ? -1 : 1) *
                                            std::ldexp(static_cast<float>(mantissa) / 1024.0f, -14);
                    } else {
                        scale_f = (sign ? -1 : 1) *
                                  std::ldexp(1.0f + static_cast<float>(mantissa) / 1024.0f,
                                             static_cast<int>(exponent) - 15);
                    }
                    const uint8_t* qs = block + 2;
                    for (int j = 0; j < 16 && base + j < K; ++j) {
                        float qv = static_cast<float>((qs[j] & 0x0F) - 8) * scale_f;
                        acc_ref = _mm256_fmadd_ps(_mm256_set1_ps(a[base + j]), _mm256_set1_ps(qv),
                                                  acc_ref);
                    }
                    for (int j = 0; j < 16 && base + 16 + j < K; ++j) {
                        float qv = static_cast<float>(((qs[j] >> 4) & 0x0F) - 8) * scale_f;
                        acc_ref = _mm256_fmadd_ps(_mm256_set1_ps(a[base + 16 + j]),
                                                  _mm256_set1_ps(qv), acc_ref);
                    }
                };
                process_partial(gate_row, acc_gate);
                process_partial(up_row, acc_up);
                continue;
            }

            {
                const uint8_t* block = gate_row + bi * BLOCK_BYTES;
                uint16_t scale_bits;
                memcpy(&scale_bits, block, 2);
                __m256 scale = fp16_to_fp32_broadcast_avx2(scale_bits);
                const uint8_t* qs = block + 2;

                __m128i q8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs));
                __m128i q_lo = _mm_and_si128(q8, lo_mask);
                __m128i q_lo_signed = _mm_sub_epi8(q_lo, eight);
                __m128i q_hi = _mm_and_si128(_mm_srli_epi16(q8, 4), lo_mask);
                __m128i q_hi_signed = _mm_sub_epi8(q_hi, eight);

                __m256 partial =
                    _mm256_mul_ps(_mm256_loadu_ps(a + base),
                                  _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_lo_signed)));
                partial = _mm256_fmadd_ps(
                    _mm256_loadu_ps(a + base + 8),
                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_lo_signed, 8))),
                    partial);
                partial =
                    _mm256_fmadd_ps(_mm256_loadu_ps(a + base + 16),
                                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_hi_signed)), partial);
                partial = _mm256_fmadd_ps(
                    _mm256_loadu_ps(a + base + 24),
                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_hi_signed, 8))),
                    partial);

                acc_gate = _mm256_fmadd_ps(scale, partial, acc_gate);
            }

            {
                const uint8_t* block = up_row + bi * BLOCK_BYTES;
                uint16_t scale_bits;
                memcpy(&scale_bits, block, 2);
                __m256 scale = fp16_to_fp32_broadcast_avx2(scale_bits);
                const uint8_t* qs = block + 2;

                __m128i q8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs));
                __m128i q_lo = _mm_and_si128(q8, lo_mask);
                __m128i q_lo_signed = _mm_sub_epi8(q_lo, eight);
                __m128i q_hi = _mm_and_si128(_mm_srli_epi16(q8, 4), lo_mask);
                __m128i q_hi_signed = _mm_sub_epi8(q_hi, eight);

                __m256 partial =
                    _mm256_mul_ps(_mm256_loadu_ps(a + base),
                                  _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_lo_signed)));
                partial = _mm256_fmadd_ps(
                    _mm256_loadu_ps(a + base + 8),
                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_lo_signed, 8))),
                    partial);
                partial =
                    _mm256_fmadd_ps(_mm256_loadu_ps(a + base + 16),
                                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_hi_signed)), partial);
                partial = _mm256_fmadd_ps(
                    _mm256_loadu_ps(a + base + 24),
                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_hi_signed, 8))),
                    partial);

                acc_up = _mm256_fmadd_ps(scale, partial, acc_up);
            }
        }

        float gate_val = silu(hsum_avx2(acc_gate));
        float up_val = hsum_avx2(acc_up);
        out[n] = gate_val * up_val;
    }
}
#endif  // USE_AVX2

// ---- Fused FFN gate+up projection for Q4_K decode ----
// Reads input vector once, computes both gate and up projections simultaneously.
// Then applies SiLU(gate) * up in-place.

#ifdef USE_AVX2
static void gemv_q4_K_fused_ffn_up_avx2(const float* a, const uint8_t* w_gate, const uint8_t* w_up,
                                        float* out, int K, int N) {
    constexpr int QK_K = 256;
    constexpr int Q4_K_BLOCK_BYTES = 144;
    const int nb = (K + QK_K - 1) / QK_K;

    std::vector<block_q8_K> q8_buf(nb);
    quantize_row_q8_K(a, q8_buf.data(), K);

    auto silu = [](float x) -> float { return x / (1.0f + std::exp(-x)); };

#    pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        const uint8_t* gate_row = w_gate + (size_t)n * nb * Q4_K_BLOCK_BYTES;
        const uint8_t* up_row = w_up + (size_t)n * nb * Q4_K_BLOCK_BYTES;

        float gate_val = silu(dot_q4_K_q8_K_avx2(gate_row, q8_buf.data(), nb));
        float up_val = dot_q4_K_q8_K_avx2(up_row, q8_buf.data(), nb);
        out[n] = gate_val * up_val;
    }
}
#endif  // USE_AVX2

// ---- Fused QKV projection for Q4_K decode ----
// Reads input vector once, computes Q + K + V outputs simultaneously
// with a single activation quantization.

#ifdef USE_AVX2
static void gemv_q4_K_fused_qkv_avx2(const float* a, const uint8_t* wq, const uint8_t* wk,
                                     const uint8_t* wv, float* out_q, float* out_k, float* out_v,
                                     int K, int N_q, int N_k, int N_v) {
    constexpr int QK_K = 256;
    constexpr int Q4_K_BLOCK_BYTES = 144;
    const int nb = (K + QK_K - 1) / QK_K;

    std::vector<block_q8_K> q8_buf(nb);
    quantize_row_q8_K(a, q8_buf.data(), K);

    auto process_row = [&](const uint8_t* w_row, float* out, int N_out) {
#    pragma omp parallel for schedule(static)
        for (int n = 0; n < N_out; ++n) {
            const uint8_t* q4_row = w_row + (size_t)n * nb * Q4_K_BLOCK_BYTES;
            out[n] = dot_q4_K_q8_K_avx2(q4_row, q8_buf.data(), nb);
        }
    };

    process_row(wq, out_q, N_q);
    process_row(wk, out_k, N_k);
    process_row(wv, out_v, N_v);
}
#endif  // USE_AVX2

}  // namespace cpu
}  // namespace forge
