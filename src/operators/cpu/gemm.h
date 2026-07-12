#pragma once

#include "vec.h"
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef USE_AVX2
#include <immintrin.h>
#endif

namespace nanoinfer {
namespace cpu {

// ---- FP32 GEMM (a[M,K] @ b[K,N] -> out[M,N]) ----
// Tiled for better cache utilization

#ifdef USE_AVX2
static void gemm_fp32_avx2(
    const float* a, const float* b, float* out,
    int M, int K, int N)
{
    std::memset(out, 0, (size_t)M * N * sizeof(float));

    constexpr int MR = 6;
    constexpr int NR = 4;

    #pragma omp parallel for schedule(dynamic) if(M * N > 64)
    for (int m = 0; m < M; ++m) {
        const float* a_row = a + m * K;
        float* o_row = out + m * N;
        for (int k = 0; k < K; ++k) {
            float a_val = a_row[k];
            __m256 a_vec = _mm256_set1_ps(a_val);
            int n = 0;
            for (; n + 8 <= N; n += 8) {
                __m256 b_vec = _mm256_loadu_ps(b + k * N + n);
                __m256 o_vec = _mm256_loadu_ps(o_row + n);
                o_vec = _mm256_fmadd_ps(a_vec, b_vec, o_vec);
                _mm256_storeu_ps(o_row + n, o_vec);
            }
            for (; n < N; ++n) {
                o_row[n] += a_val * b[k * N + n];
            }
        }
    }
}
#endif // USE_AVX2

// ---- Q4_0 GEMM (non-transB) ----
// a[M,K] @ dequant(w[K,N]) -> out[M,N], w is Q4_0 quantized

#ifdef USE_AVX2
static void gemm_q4_0_avx2(
    const float* a, const uint8_t* w, float* out,
    int M, int K, int N)
{
    constexpr int BLOCK_SIZE = 32;
    constexpr int BLOCK_BYTES = 18;
    const int blocks_per_row = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    std::memset(out, 0, (size_t)M * N * sizeof(float));

    #pragma omp parallel for schedule(dynamic) if(M * K > 64)
    for (int m = 0; m < M; ++m) {
        const float* a_row = a + m * K;
        float* o_row = out + m * N;
        for (int k = 0; k < K; ++k) {
            float a_val = a_row[k];
            const uint8_t* w_row = w + (size_t)k * blocks_per_row * BLOCK_BYTES;

            for (int bi = 0; bi < blocks_per_row; ++bi) {
                const uint8_t* block = w_row + bi * BLOCK_BYTES;
                uint16_t scale_bits;
                memcpy(&scale_bits, block, 2);
                __m256 scale = fp16_to_fp32_broadcast_avx2(scale_bits);
                const uint8_t* qs = block + 2;
                int base = bi * BLOCK_SIZE;
                int remaining = N - base;

                if (remaining >= BLOCK_SIZE) {
                    __m128i q8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs));
                    __m128i lo_mask = _mm_set1_epi8(0x0F);
                    __m128i eight = _mm_set1_epi8(8);

                    __m128i q_lo = _mm_and_si128(q8, lo_mask);
                    __m128i q_lo_signed = _mm_sub_epi8(q_lo, eight);
                    __m128i q_hi = _mm_and_si128(_mm_srli_epi16(q8, 4), lo_mask);
                    __m128i q_hi_signed = _mm_sub_epi8(q_hi, eight);

                    __m256i q32 = _mm256_cvtepi8_epi32(q_lo_signed);
                    __m256 qf = _mm256_cvtepi32_ps(q32);
                    qf = _mm256_mul_ps(qf, scale);
                    qf = _mm256_mul_ps(qf, _mm256_set1_ps(a_val));
                    __m256 o_v = _mm256_loadu_ps(o_row + base);
                    o_v = _mm256_add_ps(o_v, qf);
                    _mm256_storeu_ps(o_row + base, o_v);

                    q32 = _mm256_cvtepi8_epi32(_mm_srli_si128(q_lo_signed, 8));
                    qf = _mm256_cvtepi32_ps(q32);
                    qf = _mm256_mul_ps(qf, scale);
                    qf = _mm256_mul_ps(qf, _mm256_set1_ps(a_val));
                    o_v = _mm256_loadu_ps(o_row + base + 8);
                    o_v = _mm256_add_ps(o_v, qf);
                    _mm256_storeu_ps(o_row + base + 8, o_v);

                    q32 = _mm256_cvtepi8_epi32(q_hi_signed);
                    qf = _mm256_cvtepi32_ps(q32);
                    qf = _mm256_mul_ps(qf, scale);
                    qf = _mm256_mul_ps(qf, _mm256_set1_ps(a_val));
                    o_v = _mm256_loadu_ps(o_row + base + 16);
                    o_v = _mm256_add_ps(o_v, qf);
                    _mm256_storeu_ps(o_row + base + 16, o_v);

                    q32 = _mm256_cvtepi8_epi32(_mm_srli_si128(q_hi_signed, 8));
                    qf = _mm256_cvtepi32_ps(q32);
                    qf = _mm256_mul_ps(qf, scale);
                    qf = _mm256_mul_ps(qf, _mm256_set1_ps(a_val));
                    o_v = _mm256_loadu_ps(o_row + base + 24);
                    o_v = _mm256_add_ps(o_v, qf);
                    _mm256_storeu_ps(o_row + base + 24, o_v);
                } else {
                    float scale_f;
                    uint16_t scale_bits;
                    memcpy(&scale_bits, block, 2);
                    uint32_t sign = (scale_bits >> 15) & 1;
                    uint32_t exponent = (scale_bits >> 10) & 0x1F;
                    uint32_t mantissa = scale_bits & 0x3FF;
                    if (exponent == 0) {
                        scale_f = mantissa == 0 ? 0.0f :
                            (sign ? -1 : 1) * std::ldexp(static_cast<float>(mantissa) / 1024.0f, -14);
                    } else {
                        scale_f = (sign ? -1 : 1) * std::ldexp(
                            1.0f + static_cast<float>(mantissa) / 1024.0f,
                            static_cast<int>(exponent) - 15);
                    }

                    for (int j = 0; j < 16 && base + j < N; ++j) {
                        o_row[base + j] += a_val * static_cast<float>((qs[j] & 0x0F) - 8) * scale_f;
                    }
                    for (int j = 0; j < 16 && base + 16 + j < N; ++j) {
                        o_row[base + 16 + j] += a_val * static_cast<float>(((qs[j] >> 4) & 0x0F) - 8) * scale_f;
                    }
                }
            }
        }
    }
}
#endif // USE_AVX2

} // namespace cpu
} // namespace nanoinfer
