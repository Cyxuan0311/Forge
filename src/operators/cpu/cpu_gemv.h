#pragma once
// AVX2/FMA optimized GEMV kernels for quantized weights.
// Fused dequantize+GEMV: avoids full dequantization, computes dot products
// with on-the-fly dequantization for better cache utilization.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#ifdef USE_AVX2
#    include <immintrin.h>
#    include "quant_helpers.h"
#    include "vec_dot.h"
#    include "gemm_microkernel.h"
// dot_q4_K_q8_K forward-declared below for the Q8_K-based FFN up kernel
#endif

namespace forge {
namespace cpu {

// ---- Q4_0 fused GEMV (transB layout: weight is [N, K] quantized) ----
// Computes: out[m*N + n] = sum_k(a[m*K + k] * dequant(weight_row_n[k]))
// Q4_0 block: 2 bytes scale (fp16) + 16 bytes quants = 18 bytes per 32 elements

#ifdef USE_AVX2

// Convert a single fp16 value to fp32 and broadcast to all 8 lanes of __m256
static inline __m256 fp16_to_fp32_broadcast_avx2(uint16_t fp16_val) {
    // Broadcast the single fp16 value to all 8 positions, then convert
    return _mm256_cvtph_ps(_mm_set1_epi16(fp16_val));
}

// AVX2 horizontal sum helper
static inline float hsum_avx2(__m256 v) {
    __m128 hi128 = _mm256_extractf128_ps(v, 1);
    __m128 lo128 = _mm256_castps256_ps128(v);
    __m128 sum128 = _mm_add_ps(lo128, hi128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}

// Q4_0 GEMV: a[M,K] @ dequant(w[N,K])^T -> out[M,N]
// w is Q4_0 quantized, shape [N,K], stored row-major with 18-byte blocks per 32 elements
//
// Optimized variant: for M=1 (decode), process NR=4 output rows at a time
// to reuse the input vector load and improve instruction-level parallelism.
//
// Key optimizations:
// - Factor out scale multiplication: accumulate unscaled dot products within each
//   block, then multiply by scale once. Saves 3 mul_ps per row per block.
// - Dual accumulator groups: 2 independent FMA chains per row to break
//   dependency latency (5-cycle FMA on Zen2+, ~3 on Ice Lake).
// - Block-pair loop unrolling: process 2 blocks per iteration to increase
//   instruction-level parallelism and reduce branch overhead.
// - Prefetch: _mm_prefetch next block's weight data to L1 while computing
//   current block, hiding memory latency for large N.
// - Hoist constants (lo_mask, eight) outside the block loop.

// Inner helper: compute unscaled partial for one Q4_0 block (32 elements)
// Returns partial = sum(a[k] * (q[k]-8)) for the block
static inline __m256 q4_0_block_partial_avx2(const float* a_row, int base, const uint8_t* qs,
                                             const __m128i lo_mask, const __m128i eight) {
    __m128i q8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs));
    __m128i q_lo = _mm_and_si128(q8, lo_mask);
    __m128i q_lo_signed = _mm_sub_epi8(q_lo, eight);
    __m128i q_hi = _mm_and_si128(_mm_srli_epi16(q8, 4), lo_mask);
    __m128i q_hi_signed = _mm_sub_epi8(q_hi, eight);

    __m256 partial = _mm256_mul_ps(_mm256_loadu_ps(a_row + base),
                                   _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_lo_signed)));
    partial = _mm256_fmadd_ps(
        _mm256_loadu_ps(a_row + base + 8),
        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_lo_signed, 8))), partial);
    partial = _mm256_fmadd_ps(_mm256_loadu_ps(a_row + base + 16),
                              _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_hi_signed)), partial);
    partial = _mm256_fmadd_ps(
        _mm256_loadu_ps(a_row + base + 24),
        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_hi_signed, 8))), partial);
    return partial;
}

// Forward declaration for Q8_0-accelerated decode path
static void gemv_q4_0_transB_avx2_q8(const float* a, const uint8_t* w, float* out, int M, int K,
                                     int N);

static void gemv_q4_0_transB_avx2(const float* a, const uint8_t* w, float* out, int M, int K,
                                  int N) {
    constexpr int BLOCK_SIZE = 32;
    constexpr int BLOCK_BYTES = 18;
    const int blocks_per_row = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int full_blocks = K / BLOCK_SIZE;  // blocks that are fully aligned

    const __m128i lo_mask = _mm_set1_epi8(0x0F);
    const __m128i eight = _mm_set1_epi8(8);

// Helper: process one full block for one row, accumulate into acc_a or acc_b
// Using inline function to avoid lambda overhead and allow compiler scheduling
#    define Q4_0_PROCESS_BLOCK(a_row, w_row, bi, acc)                                       \
        do {                                                                                \
            const uint8_t* _blk = (w_row) + (size_t)(bi)*18;                                \
            uint16_t _sb;                                                                   \
            memcpy(&_sb, _blk, 2);                                                          \
            __m256 _sc = fp16_to_fp32_broadcast_avx2(_sb);                                  \
            __m256 _pa = q4_0_block_partial_avx2(a_row, (bi)*32, _blk + 2, lo_mask, eight); \
            acc = _mm256_fmadd_ps(_sc, _pa, acc);                                           \
        } while (0)

// Helper: prefetch the weight block for iteration bi
#    define Q4_0_PREFETCH_BLOCK(w_row, bi)                    \
        do {                                                  \
            const uint8_t* _pblk = (w_row) + (size_t)(bi)*18; \
            _mm_prefetch((const char*)_pblk, _MM_HINT_T0);    \
        } while (0)

    if (M == 1) {
        constexpr int NR = 8;
#    pragma omp parallel for schedule(static)
        for (int n = 0; n < N; n += NR) {
            int rows = (n + NR <= N) ? NR : (N - n);

            __m256 acc[8];
            for (int ri = 0; ri < rows; ri++)
                acc[ri] = _mm256_setzero_ps();

            const uint8_t* w_row[8];
            for (int ri = 0; ri < rows; ri++)
                w_row[ri] = w + (size_t)(n + ri) * blocks_per_row * BLOCK_BYTES;

            auto prefetch_all = [&](int bi) {
                for (int ri = 0; ri < rows; ri++)
                    if (w_row[ri])
                        _mm_prefetch((const char*)(w_row[ri] + (size_t)bi * BLOCK_BYTES),
                                     _MM_HINT_T0);
            };

            auto process_block = [&](int bi) {
                for (int ri = 0; ri < rows; ri++) {
                    const uint8_t* blk = w_row[ri] + (size_t)bi * BLOCK_BYTES;
                    uint16_t sb;
                    memcpy(&sb, blk, 2);
                    __m256 scale = fp16_to_fp32_broadcast_avx2(sb);
                    __m256 pa =
                        q4_0_block_partial_avx2(a, bi * BLOCK_SIZE, blk + 2, lo_mask, eight);
                    acc[ri] = _mm256_fmadd_ps(scale, pa, acc[ri]);
                }
            };

            if (full_blocks > 0)
                prefetch_all(0);
            if (full_blocks > 1)
                prefetch_all(1);

            int bi = 0;
            for (; bi + 1 < full_blocks; bi += 2) {
                if (bi + 2 < full_blocks)
                    prefetch_all(bi + 2);
                process_block(bi);
                process_block(bi + 1);
            }
            if (bi < full_blocks)
                process_block(bi);

            if (full_blocks < blocks_per_row) {
                int base = full_blocks * BLOCK_SIZE;
                int remaining = K - base;
                if (remaining > 0) {
                    auto proc_partial = [&](const uint8_t* w_row, __m256& acc_val) {
                        const uint8_t* block = w_row + (size_t)full_blocks * BLOCK_BYTES;
                        uint16_t sb;
                        memcpy(&sb, block, 2);
                        uint32_t sign = (sb >> 15) & 1;
                        uint32_t exponent = (sb >> 10) & 0x1F;
                        uint32_t mantissa = sb & 0x3FF;
                        float scale_f;
                        if (exponent == 0) {
                            scale_f = mantissa == 0
                                          ? 0.0f
                                          : (sign ? -1.0f : 1.0f) *
                                                std::ldexp((float)mantissa / 1024.0f, -14);
                        } else {
                            scale_f = (sign ? -1.0f : 1.0f) *
                                      std::ldexp(1.0f + (float)mantissa / 1024.0f,
                                                 (int)exponent - 15);
                        }
                        const uint8_t* qs = block + 2;
                        for (int j = 0; j < 16 && base + j < K; ++j) {
                            float qv = static_cast<float>((qs[j] & 0x0F) - 8) * scale_f;
                            acc_val = _mm256_fmadd_ps(_mm256_set1_ps(a[base + j]),
                                                       _mm256_set1_ps(qv), acc_val);
                        }
                        for (int j = 0; j < 16 && base + 16 + j < K; ++j) {
                            float qv = static_cast<float>(((qs[j] >> 4) & 0x0F) - 8) * scale_f;
                            acc_val = _mm256_fmadd_ps(_mm256_set1_ps(a[base + 16 + j]),
                                                       _mm256_set1_ps(qv), acc_val);
                        }
                    };
                    for (int ri = 0; ri < rows; ri++)
                        proc_partial(w_row[ri], acc[ri]);
                }
            }

            for (int ri = 0; ri < rows; ri++)
                out[n + ri] = hsum_avx2(acc[ri]);
        }
    } else {
// Prefill path: M > 1, use per-row approach with dual accumulators
#    pragma omp parallel for schedule(dynamic, 4) if (N > 4)
        for (int n = 0; n < N; ++n) {
            const uint8_t* w_row = w + (size_t)n * blocks_per_row * BLOCK_BYTES;
            for (int m = 0; m < M; ++m) {
                const float* a_row = a + m * K;
                __m256 acc_a = _mm256_setzero_ps();
                __m256 acc_b = _mm256_setzero_ps();

                int bi = 0;
                for (; bi + 1 < full_blocks; bi += 2) {
                    Q4_0_PROCESS_BLOCK(a_row, w_row, bi, acc_a);
                    Q4_0_PROCESS_BLOCK(a_row, w_row, bi + 1, acc_b);
                }
                if (bi < full_blocks) {
                    Q4_0_PROCESS_BLOCK(a_row, w_row, bi, acc_a);
                }

                __m256 acc = _mm256_add_ps(acc_a, acc_b);

                // Handle partial last block
                if (full_blocks < blocks_per_row) {
                    int base = full_blocks * BLOCK_SIZE;
                    int remaining = K - base;
                    if (remaining > 0) {
                        const uint8_t* block = w_row + (size_t)full_blocks * BLOCK_BYTES;
                        uint16_t sb;
                        memcpy(&sb, block, 2);
                        uint32_t sign = (sb >> 15) & 1;
                        uint32_t exponent = (sb >> 10) & 0x1F;
                        uint32_t mantissa = sb & 0x3FF;
                        float scale_f;
                        if (exponent == 0) {
                            scale_f =
                                mantissa == 0
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
                            acc = _mm256_fmadd_ps(_mm256_set1_ps(a_row[base + j]),
                                                  _mm256_set1_ps(qv), acc);
                        }
                        for (int j = 0; j < 16 && base + 16 + j < K; ++j) {
                            float qv = static_cast<float>(((qs[j] >> 4) & 0x0F) - 8) * scale_f;
                            acc = _mm256_fmadd_ps(_mm256_set1_ps(a_row[base + 16 + j]),
                                                  _mm256_set1_ps(qv), acc);
                        }
                    }
                }
                out[m * N + n] = hsum_avx2(acc);
            }
        }
    }

#    undef Q4_0_PROCESS_BLOCK
#    undef Q4_0_PREFETCH_BLOCK
}

// FP32 GEMV with AVX2: a[M,K] @ b[N,K]^T -> out[M,N]
// Optimized for both M=1 (decode) and M>1 (prefill) cases
static void gemv_fp32_transB_avx2(const float* a, const float* b, float* out, int M, int K, int N) {
    if (M == 1) {
        // Decode: single query vector against all N rows
        // Use NR=4 row parallelism to reuse input vector loads
        constexpr int NR = 4;
        const float* a_row = a;
#    pragma omp parallel for schedule(static)
        for (int n = 0; n < N; n += NR) {
            int rows = (n + NR <= N) ? NR : (N - n);
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();

            const float* b_row0 = b + (n + 0) * K;
            const float* b_row1 = (rows > 1) ? b + (n + 1) * K : nullptr;
            const float* b_row2 = (rows > 2) ? b + (n + 2) * K : nullptr;
            const float* b_row3 = (rows > 3) ? b + (n + 3) * K : nullptr;

            for (int k = 0; k + 8 <= K; k += 8) {
                __m256 av = _mm256_loadu_ps(a_row + k);

                __m256 bv0 = _mm256_loadu_ps(b_row0 + k);
                acc0 = _mm256_fmadd_ps(av, bv0, acc0);

                if (rows > 1) {
                    __m256 bv1 = _mm256_loadu_ps(b_row1 + k);
                    acc1 = _mm256_fmadd_ps(av, bv1, acc1);
                }
                if (rows > 2) {
                    __m256 bv2 = _mm256_loadu_ps(b_row2 + k);
                    acc2 = _mm256_fmadd_ps(av, bv2, acc2);
                }
                if (rows > 3) {
                    __m256 bv3 = _mm256_loadu_ps(b_row3 + k);
                    acc3 = _mm256_fmadd_ps(av, bv3, acc3);
                }
            }

            // Horizontal sums with tail handling
            auto finalize = [&](const float* a_r, const float* b_r, __m256 acc_val) -> float {
                float sum = hsum_avx2(acc_val);
                for (int k = (K / 8) * 8; k < K; ++k)
                    sum += a_r[k] * b_r[k];
                return sum;
            };
            out[n + 0] = finalize(a_row, b_row0, acc0);
            if (rows > 1)
                out[n + 1] = finalize(a_row, b_row1, acc1);
            if (rows > 2)
                out[n + 2] = finalize(a_row, b_row2, acc2);
            if (rows > 3)
                out[n + 3] = finalize(a_row, b_row3, acc3);
        }
    } else {
// Prefill: multiple query vectors
// Parallelize over M (rows of a), each thread computes one full output row
#    pragma omp parallel for schedule(static)
        for (int m = 0; m < M; ++m) {
            const float* a_row = a + m * K;
            float* o_row = out + m * N;
            for (int n = 0; n < N; ++n) {
                const float* b_row = b + n * K;
                __m256 acc = _mm256_setzero_ps();
                int k = 0;
                for (; k + 8 <= K; k += 8) {
                    __m256 av = _mm256_loadu_ps(a_row + k);
                    __m256 bv = _mm256_loadu_ps(b_row + k);
                    acc = _mm256_fmadd_ps(av, bv, acc);
                }
                __m128 hi128 = _mm256_extractf128_ps(acc, 1);
                __m128 lo128 = _mm256_castps256_ps128(acc);
                __m128 sum128 = _mm_add_ps(lo128, hi128);
                sum128 = _mm_hadd_ps(sum128, sum128);
                sum128 = _mm_hadd_ps(sum128, sum128);
                float sum = _mm_cvtss_f32(sum128);
                for (; k < K; ++k)
                    sum += a_row[k] * b_row[k];
                o_row[n] = sum;
            }
        }
    }
}

// FP32 GEMM with AVX2: a[M,K] @ b[K,N] -> out[M,N]
// Tiled for better cache utilization
static void gemm_fp32_avx2(const float* a, const float* b, float* out, int M, int K, int N) {
    std::memset(out, 0, (size_t)M * N * sizeof(float));

    constexpr int MR = 6;  // micro-tile rows
    constexpr int NR = 4;  // micro-tile cols

#    pragma omp parallel for schedule(dynamic) if (M * N > 64)
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

// Q4_0 GEMM (non-transB): a[M,K] @ dequant(w[K,N]) -> out[M,N]
// w is Q4_0 quantized, shape [K, N], stored row-major
static void gemm_q4_0_avx2(const float* a, const uint8_t* w, float* out, int M, int K, int N) {
    constexpr int BLOCK_SIZE = 32;
    constexpr int BLOCK_BYTES = 18;
    const int blocks_per_row = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    std::memset(out, 0, (size_t)M * N * sizeof(float));

#    pragma omp parallel for schedule(dynamic) if (M * K > 64)
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

                    // Low nibbles first 8
                    __m256i q32 = _mm256_cvtepi8_epi32(q_lo_signed);
                    __m256 qf = _mm256_cvtepi32_ps(q32);
                    qf = _mm256_mul_ps(qf, scale);
                    qf = _mm256_mul_ps(qf, _mm256_set1_ps(a_val));
                    __m256 o_v = _mm256_loadu_ps(o_row + base);
                    o_v = _mm256_add_ps(o_v, qf);
                    _mm256_storeu_ps(o_row + base, o_v);

                    // Low nibbles next 8
                    q32 = _mm256_cvtepi8_epi32(_mm_srli_si128(q_lo_signed, 8));
                    qf = _mm256_cvtepi32_ps(q32);
                    qf = _mm256_mul_ps(qf, scale);
                    qf = _mm256_mul_ps(qf, _mm256_set1_ps(a_val));
                    o_v = _mm256_loadu_ps(o_row + base + 8);
                    o_v = _mm256_add_ps(o_v, qf);
                    _mm256_storeu_ps(o_row + base + 8, o_v);

                    // High nibbles first 8
                    q32 = _mm256_cvtepi8_epi32(q_hi_signed);
                    qf = _mm256_cvtepi32_ps(q32);
                    qf = _mm256_mul_ps(qf, scale);
                    qf = _mm256_mul_ps(qf, _mm256_set1_ps(a_val));
                    o_v = _mm256_loadu_ps(o_row + base + 16);
                    o_v = _mm256_add_ps(o_v, qf);
                    _mm256_storeu_ps(o_row + base + 16, o_v);

                    // High nibbles next 8
                    q32 = _mm256_cvtepi8_epi32(_mm_srli_si128(q_hi_signed, 8));
                    qf = _mm256_cvtepi32_ps(q32);
                    qf = _mm256_mul_ps(qf, scale);
                    qf = _mm256_mul_ps(qf, _mm256_set1_ps(a_val));
                    o_v = _mm256_loadu_ps(o_row + base + 24);
                    o_v = _mm256_add_ps(o_v, qf);
                    _mm256_storeu_ps(o_row + base + 24, o_v);
                } else {
                    // Partial block: scalar
                    float scale_f;
                    uint16_t scale_bits;
                    memcpy(&scale_bits, block, 2);
                    uint32_t sign = (scale_bits >> 15) & 1;
                    uint32_t exponent = (scale_bits >> 10) & 0x1F;
                    uint32_t mantissa = scale_bits & 0x3FF;
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

                    for (int j = 0; j < 16 && base + j < N; ++j) {
                        o_row[base + j] += a_val * static_cast<float>((qs[j] & 0x0F) - 8) * scale_f;
                    }
                    for (int j = 0; j < 16 && base + 16 + j < N; ++j) {
                        o_row[base + 16 + j] +=
                            a_val * static_cast<float>(((qs[j] >> 4) & 0x0F) - 8) * scale_f;
                    }
                }
            }
        }
    }
}

#endif  // USE_AVX2

// ---- Q8_0 fused GEMV ----
// Q8_0 block: 2 bytes scale (fp16) + 32 bytes int8 quants = 34 bytes per 32 elements
//
// Optimized variant: for M=1 (decode), process NR=4 output rows at a time
// to reuse the input vector load and improve instruction-level parallelism.
//
// Key optimizations:
// - Factor out scale multiplication: accumulate unscaled dot products within each
//   block (a * qs), then multiply by scale once. Saves 3 mul_ps per row per block.
// - NR=4 row parallelism for M=1: one input vector load serves 4 output rows.

#ifdef USE_AVX2
static void gemv_q8_0_transB_avx2(const float* a, const uint8_t* w, float* out, int M, int K,
                                  int N) {
    constexpr int BLOCK_SIZE = 32;
    constexpr int BLOCK_BYTES = 34;
    const int blocks_per_row = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int full_blocks = K / BLOCK_SIZE;

    // Process one weight block for a single output row, using dual accumulators.
    // This breaks the 5-cycle FMA dependency chain, allowing 2 independent
    // FMA pipelines to execute in parallel on Zen2+ and Ice Lake.
    auto process_block = [&](const float* a_row, const uint8_t* w_row, int bi,
                             __m256& acc_a, __m256& acc_b) {
        const uint8_t* block = w_row + (size_t)bi * BLOCK_BYTES;
        uint16_t scale_bits;
        memcpy(&scale_bits, block, 2);
        __m256 scale = fp16_to_fp32_broadcast_avx2(scale_bits);
        const int8_t* qs = reinterpret_cast<const int8_t*>(block + 2);
        int base = bi * BLOCK_SIZE;

        // Even sub-blocks → acc_a, odd sub-blocks → acc_b
        __m256 p0 = _mm256_mul_ps(_mm256_loadu_ps(a_row + base),
                                   _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                                       _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qs)))));
        __m256 p1 = _mm256_mul_ps(_mm256_loadu_ps(a_row + base + 8),
                                   _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                                       _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qs + 8)))));
        __m256 p2 = _mm256_mul_ps(_mm256_loadu_ps(a_row + base + 16),
                                   _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                                       _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qs + 16)))));
        __m256 p3 = _mm256_mul_ps(_mm256_loadu_ps(a_row + base + 24),
                                   _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                                       _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qs + 24)))));

        // Accumulate into dual chains
        __m256 partial_a = _mm256_add_ps(p0, p2);
        __m256 partial_b = _mm256_add_ps(p1, p3);
        acc_a = _mm256_fmadd_ps(scale, partial_a, acc_a);
        acc_b = _mm256_fmadd_ps(scale, partial_b, acc_b);
    };

    if (M == 1) {
        // Decode path: single input vector, optimize by processing NR=8 rows at a time
        // with dual accumulators + prefetch for better ILP and memory latency hiding.
        constexpr int NR = 8;
        const float* a_row = a;

#    pragma omp parallel
        {
            // Thread-local max for fused max reduction
            float thread_max = -1e30f;

#    pragma omp for schedule(dynamic, 32) nowait
            for (int n = 0; n < N; n += NR) {
                int rows = (n + NR <= N) ? NR : (N - n);

                __m256 acc0_a = _mm256_setzero_ps(), acc0_b = _mm256_setzero_ps();
                __m256 acc1_a = _mm256_setzero_ps(), acc1_b = _mm256_setzero_ps();
                __m256 acc2_a = _mm256_setzero_ps(), acc2_b = _mm256_setzero_ps();
                __m256 acc3_a = _mm256_setzero_ps(), acc3_b = _mm256_setzero_ps();
                __m256 acc4_a = _mm256_setzero_ps(), acc4_b = _mm256_setzero_ps();
                __m256 acc5_a = _mm256_setzero_ps(), acc5_b = _mm256_setzero_ps();
                __m256 acc6_a = _mm256_setzero_ps(), acc6_b = _mm256_setzero_ps();
                __m256 acc7_a = _mm256_setzero_ps(), acc7_b = _mm256_setzero_ps();

                const uint8_t* w_rows[NR];
                for (int r = 0; r < NR; ++r) {
                    w_rows[r] = (n + r < N) ? w + (size_t)(n + r) * blocks_per_row * BLOCK_BYTES : nullptr;
                }

                // Prefetch first two blocks
                for (int r = 0; r < rows; ++r) {
                    _mm_prefetch((const char*)w_rows[r], _MM_HINT_T0);
                }

                int bi = 0;
                for (; bi + 1 < full_blocks; bi += 2) {
                    // Prefetch next pair of blocks
                    if (bi + 2 < full_blocks) {
                        for (int r = 0; r < rows; ++r) {
                            _mm_prefetch((const char*)(w_rows[r] + (size_t)(bi + 2) * BLOCK_BYTES), _MM_HINT_T0);
                        }
                    }

                    process_block(a_row, w_rows[0], bi, acc0_a, acc0_b);
                    if (rows > 1) process_block(a_row, w_rows[1], bi, acc1_a, acc1_b);
                    if (rows > 2) process_block(a_row, w_rows[2], bi, acc2_a, acc2_b);
                    if (rows > 3) process_block(a_row, w_rows[3], bi, acc3_a, acc3_b);
                    if (rows > 4) process_block(a_row, w_rows[4], bi, acc4_a, acc4_b);
                    if (rows > 5) process_block(a_row, w_rows[5], bi, acc5_a, acc5_b);
                    if (rows > 6) process_block(a_row, w_rows[6], bi, acc6_a, acc6_b);
                    if (rows > 7) process_block(a_row, w_rows[7], bi, acc7_a, acc7_b);

                    process_block(a_row, w_rows[0], bi + 1, acc0_a, acc0_b);
                    if (rows > 1) process_block(a_row, w_rows[1], bi + 1, acc1_a, acc1_b);
                    if (rows > 2) process_block(a_row, w_rows[2], bi + 1, acc2_a, acc2_b);
                    if (rows > 3) process_block(a_row, w_rows[3], bi + 1, acc3_a, acc3_b);
                    if (rows > 4) process_block(a_row, w_rows[4], bi + 1, acc4_a, acc4_b);
                    if (rows > 5) process_block(a_row, w_rows[5], bi + 1, acc5_a, acc5_b);
                    if (rows > 6) process_block(a_row, w_rows[6], bi + 1, acc6_a, acc6_b);
                    if (rows > 7) process_block(a_row, w_rows[7], bi + 1, acc7_a, acc7_b);
                }

                if (bi < full_blocks) {
                    process_block(a_row, w_rows[0], bi, acc0_a, acc0_b);
                    if (rows > 1) process_block(a_row, w_rows[1], bi, acc1_a, acc1_b);
                    if (rows > 2) process_block(a_row, w_rows[2], bi, acc2_a, acc2_b);
                    if (rows > 3) process_block(a_row, w_rows[3], bi, acc3_a, acc3_b);
                    if (rows > 4) process_block(a_row, w_rows[4], bi, acc4_a, acc4_b);
                    if (rows > 5) process_block(a_row, w_rows[5], bi, acc5_a, acc5_b);
                    if (rows > 6) process_block(a_row, w_rows[6], bi, acc6_a, acc6_b);
                    if (rows > 7) process_block(a_row, w_rows[7], bi, acc7_a, acc7_b);
                }

                // Handle partial last block (scalar fallback)
                if (full_blocks < blocks_per_row) {
                    int bi_last = blocks_per_row - 1;
                    int base = bi_last * BLOCK_SIZE;
                    int remaining = K - base;
                    if (remaining > 0 && remaining < BLOCK_SIZE) {
                        auto process_partial = [&](const uint8_t* w_row, __m256& acc_a_ref, __m256& acc_b_ref) {
                            const uint8_t* block = w_row + (size_t)bi_last * BLOCK_BYTES;
                            uint16_t sb;
                            memcpy(&sb, block, 2);
                            __m256 scale = fp16_to_fp32_broadcast_avx2(sb);
                            const int8_t* qs = reinterpret_cast<const int8_t*>(block + 2);
                            __m256 partial = _mm256_setzero_ps();
                            for (int j = 0; j < remaining; ++j) {
                                partial = _mm256_fmadd_ps(
                                    _mm256_set1_ps(a_row[base + j]),
                                    _mm256_set1_ps(static_cast<float>(qs[j])), partial);
                            }
                            // Apply scale and add to both accumulators
                            __m256 scaled = _mm256_mul_ps(scale, partial);
                            acc_a_ref = _mm256_add_ps(acc_a_ref, scaled);
                        };
                        process_partial(w_rows[0], acc0_a, acc0_b);
                        if (rows > 1 && w_rows[1]) process_partial(w_rows[1], acc1_a, acc1_b);
                        if (rows > 2 && w_rows[2]) process_partial(w_rows[2], acc2_a, acc2_b);
                        if (rows > 3 && w_rows[3]) process_partial(w_rows[3], acc3_a, acc3_b);
                        if (rows > 4 && w_rows[4]) process_partial(w_rows[4], acc4_a, acc4_b);
                        if (rows > 5 && w_rows[5]) process_partial(w_rows[5], acc5_a, acc5_b);
                        if (rows > 6 && w_rows[6]) process_partial(w_rows[6], acc6_a, acc6_b);
                        if (rows > 7 && w_rows[7]) process_partial(w_rows[7], acc7_a, acc7_b);
                    }
                }

                // Horizontal sum with dual accumulator merge
                float vals[NR];
                vals[0] = hsum_avx2(_mm256_add_ps(acc0_a, acc0_b));
                if (rows > 1) vals[1] = hsum_avx2(_mm256_add_ps(acc1_a, acc1_b));
                if (rows > 2) vals[2] = hsum_avx2(_mm256_add_ps(acc2_a, acc2_b));
                if (rows > 3) vals[3] = hsum_avx2(_mm256_add_ps(acc3_a, acc3_b));
                if (rows > 4) vals[4] = hsum_avx2(_mm256_add_ps(acc4_a, acc4_b));
                if (rows > 5) vals[5] = hsum_avx2(_mm256_add_ps(acc5_a, acc5_b));
                if (rows > 6) vals[6] = hsum_avx2(_mm256_add_ps(acc6_a, acc6_b));
                if (rows > 7) vals[7] = hsum_avx2(_mm256_add_ps(acc7_a, acc7_b));

                // Write results + track thread-local max
                for (int r = 0; r < rows; ++r) {
                    out[n + r] = vals[r];
                    if (vals[r] > thread_max) thread_max = vals[r];
                }
            }

            // Fused max reduction across threads (optional, saved back to out[0] area if needed)
            // For now the sampler does its own max; this just reduces per-thread overhead
        }
    } else {
// Prefill path: M > 1, use original per-row approach
#    pragma omp parallel for schedule(dynamic, 4) if (N > 4)
        for (int n = 0; n < N; ++n) {
            const uint8_t* w_row = w + (size_t)n * blocks_per_row * BLOCK_BYTES;
            for (int m = 0; m < M; ++m) {
                const float* a_row = a + m * K;
                __m256 acc_a = _mm256_setzero_ps(), acc_b = _mm256_setzero_ps();

                for (int bi = 0; bi < blocks_per_row; ++bi) {
                    int base = bi * BLOCK_SIZE;
                    int remaining = K - base;
                    if (remaining >= BLOCK_SIZE) {
                        process_block(a_row, w_row, bi, acc_a, acc_b);
                    } else if (remaining > 0) {
                        const uint8_t* block = w_row + (size_t)bi * BLOCK_BYTES;
                        uint16_t sb;
                        memcpy(&sb, block, 2);
                        __m256 scale = fp16_to_fp32_broadcast_avx2(sb);
                        const int8_t* qs = reinterpret_cast<const int8_t*>(block + 2);
                        __m256 partial = _mm256_setzero_ps();
                        for (int j = 0; j < remaining; ++j) {
                            partial = _mm256_fmadd_ps(
                                _mm256_set1_ps(a_row[base + j]),
                                _mm256_set1_ps(static_cast<float>(qs[j])), partial);
                        }
                        acc_a = _mm256_add_ps(acc_a, _mm256_mul_ps(scale, partial));
                    }
                }
                out[m * N + n] = hsum_avx2(_mm256_add_ps(acc_a, acc_b));
            }
        }
    }
}
#endif  // USE_AVX2

// ---- Q4_1 fused GEMV ----
// Q4_1 block: 2 bytes scale (fp16) + 2 bytes min (fp16) + 16 bytes quants = 20 bytes per 32
// elements Dequantization: value = q * d + m  (q: 4-bit unsigned 0-15, d: scale, m: min offset)
//
// Optimized: factor out d and m from per-sub-block accumulation.
// Instead of computing (q*d+m) per sub-block (4 FMAs per block for m addition),
// we accumulate unscaled partials:
//   partial = sum(a[k] * q[k])   -- 4 FMAs, no d/m
//   sum_a   = sum(a[k])          -- 3 adds from the same loads
//   acc = fmadd(d, partial, fmadd(m, sum_a, acc))  -- 2 FMAs total
// Net: 4 FMA + 3 ADD + 2 FMA = 9 ops vs original 4*(2 FMA) = 8 FMA = 8 ops
// BUT: we eliminate the d*q dependency chain in the inner loop, allowing
// better instruction-level parallelism. More importantly, the m_scale
// broadcast is used once instead of per-sub-block, reducing port pressure.
// With NR=4 decode path, we also reuse input vector loads across 4 rows.

#ifdef USE_AVX2

// Inner helper: compute unscaled partial sum(a[k]*q[k]) and sum(a[k]) for one Q4_1 block
static inline void q4_1_block_partial_avx2(const float* a_row, int base, const uint8_t* qs,
                                           __m256& partial, __m256& sum_a) {
    const __m128i lo_mask = _mm_set1_epi8(0x0F);
    __m128i q8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs));
    __m128i q_lo = _mm_and_si128(q8, lo_mask);
    __m128i q_hi = _mm_and_si128(_mm_srli_epi16(q8, 4), lo_mask);

    // Sub-block 0: a[base..base+7] * q_lo[0..7]
    __m256 a0 = _mm256_loadu_ps(a_row + base);
    __m256 q0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_lo));
    partial = _mm256_mul_ps(a0, q0);
    sum_a = a0;

    // Sub-block 1: a[base+8..base+15] * q_lo[8..15]
    __m256 a1 = _mm256_loadu_ps(a_row + base + 8);
    __m256 q1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_lo, 8)));
    partial = _mm256_fmadd_ps(a1, q1, partial);
    sum_a = _mm256_add_ps(sum_a, a1);

    // Sub-block 2: a[base+16..base+23] * q_hi[0..7]
    __m256 a2 = _mm256_loadu_ps(a_row + base + 16);
    __m256 q2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_hi));
    partial = _mm256_fmadd_ps(a2, q2, partial);
    sum_a = _mm256_add_ps(sum_a, a2);

    // Sub-block 3: a[base+24..base+31] * q_hi[8..15]
    __m256 a3 = _mm256_loadu_ps(a_row + base + 24);
    __m256 q3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_hi, 8)));
    partial = _mm256_fmadd_ps(a3, q3, partial);
    sum_a = _mm256_add_ps(sum_a, a3);
}

// Helper: fp16 scalar to float (for partial block fallback)
static inline float fp16_to_float_scalar(uint16_t bits) {
    uint32_t sign = (bits >> 15) & 1;
    uint32_t exponent = (bits >> 10) & 0x1F;
    uint32_t mantissa = bits & 0x3FF;
    if (exponent == 0) {
        if (mantissa == 0)
            return 0.0f;
        float v = std::ldexp(static_cast<float>(mantissa) / 1024.0f, -14);
        return sign ? -v : v;
    }
    float v =
        std::ldexp(1.0f + static_cast<float>(mantissa) / 1024.0f, static_cast<int>(exponent) - 15);
    return sign ? -v : v;
}

static void gemv_q4_1_transB_avx2(const float* a, const uint8_t* w, float* out, int M, int K,
                                  int N) {
    constexpr int BLOCK_SIZE = 32;
    constexpr int BLOCK_BYTES = 20;
    const int blocks_per_row = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int full_blocks = K / BLOCK_SIZE;

#    define Q4_1_PROCESS_BLOCK(a_row, w_row, bi, acc_a, acc_b)                   \
        do {                                                                     \
            const uint8_t* _blk = (w_row) + (size_t)(bi)*20;                     \
            uint16_t _db, _mb;                                                   \
            memcpy(&_db, _blk, 2);                                               \
            memcpy(&_mb, _blk + 2, 2);                                           \
            __m256 _d = fp16_to_fp32_broadcast_avx2(_db);                        \
            __m256 _m = fp16_to_fp32_broadcast_avx2(_mb);                        \
            __m256 _partial, _sum_a;                                             \
            q4_1_block_partial_avx2(a_row, (bi)*32, _blk + 4, _partial, _sum_a); \
            /* acc = d * partial + m * sum_a + acc */                            \
            /* Split across two accumulator groups for ILP */                    \
            __m256 _ms = _mm256_mul_ps(_m, _sum_a);                              \
            acc_a = _mm256_fmadd_ps(_d, _partial, acc_a);                        \
            acc_b = _mm256_add_ps(acc_b, _ms);                                   \
        } while (0)

#    define Q4_1_PREFETCH_BLOCK(w_row, bi)                    \
        do {                                                  \
            const uint8_t* _pblk = (w_row) + (size_t)(bi)*20; \
            _mm_prefetch((const char*)_pblk, _MM_HINT_T0);    \
        } while (0)

    if (M == 1) {
        // Decode path: single input vector, NR=4 row parallelism
        constexpr int NR = 4;
        const float* a_row = a;

#    pragma omp parallel for schedule(static)
        for (int n = 0; n < N; n += NR) {
            int rows = (n + NR <= N) ? NR : (N - n);

            // Two accumulator groups: acc_*_p accumulates d*partial terms,
            // acc_*_m accumulates m*sum_a terms. Merged at the end.
            __m256 acc0_p = _mm256_setzero_ps(), acc0_m = _mm256_setzero_ps();
            __m256 acc1_p = _mm256_setzero_ps(), acc1_m = _mm256_setzero_ps();
            __m256 acc2_p = _mm256_setzero_ps(), acc2_m = _mm256_setzero_ps();
            __m256 acc3_p = _mm256_setzero_ps(), acc3_m = _mm256_setzero_ps();

            const uint8_t* w_row0 = w + (size_t)(n + 0) * blocks_per_row * BLOCK_BYTES;
            const uint8_t* w_row1 =
                (rows > 1) ? w + (size_t)(n + 1) * blocks_per_row * BLOCK_BYTES : nullptr;
            const uint8_t* w_row2 =
                (rows > 2) ? w + (size_t)(n + 2) * blocks_per_row * BLOCK_BYTES : nullptr;
            const uint8_t* w_row3 =
                (rows > 3) ? w + (size_t)(n + 3) * blocks_per_row * BLOCK_BYTES : nullptr;

            // Prefetch first block
            Q4_1_PREFETCH_BLOCK(w_row0, 0);
            if (rows > 1)
                Q4_1_PREFETCH_BLOCK(w_row1, 0);
            if (rows > 2)
                Q4_1_PREFETCH_BLOCK(w_row2, 0);
            if (rows > 3)
                Q4_1_PREFETCH_BLOCK(w_row3, 0);

            // Process full blocks with loop unrolling (2 blocks/iter)
            int bi = 0;
            for (; bi + 1 < full_blocks; bi += 2) {
                // Block bi -> acc_p / acc_m
                Q4_1_PROCESS_BLOCK(a_row, w_row0, bi, acc0_p, acc0_m);
                if (rows > 1)
                    Q4_1_PROCESS_BLOCK(a_row, w_row1, bi, acc1_p, acc1_m);
                if (rows > 2)
                    Q4_1_PROCESS_BLOCK(a_row, w_row2, bi, acc2_p, acc2_m);
                if (rows > 3)
                    Q4_1_PROCESS_BLOCK(a_row, w_row3, bi, acc3_p, acc3_m);

                // Prefetch 2 blocks ahead
                if (bi + 2 < full_blocks) {
                    Q4_1_PREFETCH_BLOCK(w_row0, bi + 2);
                    if (rows > 1)
                        Q4_1_PREFETCH_BLOCK(w_row1, bi + 2);
                    if (rows > 2)
                        Q4_1_PREFETCH_BLOCK(w_row2, bi + 2);
                    if (rows > 3)
                        Q4_1_PREFETCH_BLOCK(w_row3, bi + 2);
                }

                // Block bi+1 -> same acc_p / acc_m (dual-chain via FMA latency hiding)
                Q4_1_PROCESS_BLOCK(a_row, w_row0, bi + 1, acc0_p, acc0_m);
                if (rows > 1)
                    Q4_1_PROCESS_BLOCK(a_row, w_row1, bi + 1, acc1_p, acc1_m);
                if (rows > 2)
                    Q4_1_PROCESS_BLOCK(a_row, w_row2, bi + 1, acc2_p, acc2_m);
                if (rows > 3)
                    Q4_1_PROCESS_BLOCK(a_row, w_row3, bi + 1, acc3_p, acc3_m);
            }

            // Handle remaining single full block
            if (bi < full_blocks) {
                Q4_1_PROCESS_BLOCK(a_row, w_row0, bi, acc0_p, acc0_m);
                if (rows > 1)
                    Q4_1_PROCESS_BLOCK(a_row, w_row1, bi, acc1_p, acc1_m);
                if (rows > 2)
                    Q4_1_PROCESS_BLOCK(a_row, w_row2, bi, acc2_p, acc2_m);
                if (rows > 3)
                    Q4_1_PROCESS_BLOCK(a_row, w_row3, bi, acc3_p, acc3_m);
            }

            // Merge: acc = acc_p + acc_m
            __m256 acc0 = _mm256_add_ps(acc0_p, acc0_m);
            __m256 acc1 = _mm256_add_ps(acc1_p, acc1_m);
            __m256 acc2 = _mm256_add_ps(acc2_p, acc2_m);
            __m256 acc3 = _mm256_add_ps(acc3_p, acc3_m);

            // Handle partial last block
            if (full_blocks < blocks_per_row) {
                int base = full_blocks * BLOCK_SIZE;
                int remaining = K - base;
                if (remaining > 0) {
                    auto process_partial = [&](const uint8_t* w_row, __m256& acc_ref) {
                        const uint8_t* block = w_row + (size_t)full_blocks * BLOCK_BYTES;
                        float d_val =
                            fp16_to_float_scalar(*reinterpret_cast<const uint16_t*>(block));
                        float m_val =
                            fp16_to_float_scalar(*reinterpret_cast<const uint16_t*>(block + 2));
                        const uint8_t* qs = block + 4;
                        for (int j = 0; j < 16 && base + j < K; ++j) {
                            float qv = static_cast<float>(qs[j] & 0xF) * d_val + m_val;
                            acc_ref = _mm256_fmadd_ps(_mm256_set1_ps(a_row[base + j]),
                                                      _mm256_set1_ps(qv), acc_ref);
                        }
                        for (int j = 0; j < 16 && base + 16 + j < K; ++j) {
                            float qv = static_cast<float>((qs[j] >> 4) & 0xF) * d_val + m_val;
                            acc_ref = _mm256_fmadd_ps(_mm256_set1_ps(a_row[base + 16 + j]),
                                                      _mm256_set1_ps(qv), acc_ref);
                        }
                    };
                    process_partial(w_row0, acc0);
                    if (rows > 1 && w_row1)
                        process_partial(w_row1, acc1);
                    if (rows > 2 && w_row2)
                        process_partial(w_row2, acc2);
                    if (rows > 3 && w_row3)
                        process_partial(w_row3, acc3);
                }
            }

            out[n + 0] = hsum_avx2(acc0);
            if (rows > 1)
                out[n + 1] = hsum_avx2(acc1);
            if (rows > 2)
                out[n + 2] = hsum_avx2(acc2);
            if (rows > 3)
                out[n + 3] = hsum_avx2(acc3);
        }
    } else {
// Prefill path: M > 1
#    pragma omp parallel for schedule(dynamic, 4) if (N > 4)
        for (int n = 0; n < N; ++n) {
            const uint8_t* w_row = w + (size_t)n * blocks_per_row * BLOCK_BYTES;
            for (int m_idx = 0; m_idx < M; ++m_idx) {
                const float* a_row = a + m_idx * K;
                __m256 acc_p = _mm256_setzero_ps();
                __m256 acc_m = _mm256_setzero_ps();

                for (int bi = 0; bi < full_blocks; ++bi) {
                    Q4_1_PROCESS_BLOCK(a_row, w_row, bi, acc_p, acc_m);
                }

                __m256 acc = _mm256_add_ps(acc_p, acc_m);

                // Handle partial last block
                if (full_blocks < blocks_per_row) {
                    int base = full_blocks * BLOCK_SIZE;
                    int remaining = K - base;
                    if (remaining > 0) {
                        const uint8_t* block = w_row + (size_t)full_blocks * BLOCK_BYTES;
                        float d_val =
                            fp16_to_float_scalar(*reinterpret_cast<const uint16_t*>(block));
                        float m_val =
                            fp16_to_float_scalar(*reinterpret_cast<const uint16_t*>(block + 2));
                        const uint8_t* qs = block + 4;
                        for (int j = 0; j < 16 && base + j < K; ++j) {
                            float qv = static_cast<float>(qs[j] & 0xF) * d_val + m_val;
                            acc = _mm256_fmadd_ps(_mm256_set1_ps(a_row[base + j]),
                                                  _mm256_set1_ps(qv), acc);
                        }
                        for (int j = 0; j < 16 && base + 16 + j < K; ++j) {
                            float qv = static_cast<float>((qs[j] >> 4) & 0xF) * d_val + m_val;
                            acc = _mm256_fmadd_ps(_mm256_set1_ps(a_row[base + 16 + j]),
                                                  _mm256_set1_ps(qv), acc);
                        }
                    }
                }
                out[m_idx * N + n] = hsum_avx2(acc);
            }
        }
    }

#    undef Q4_1_PROCESS_BLOCK
#    undef Q4_1_PREFETCH_BLOCK
}
#endif  // USE_AVX2

// ---- AVX2 dot product helper for dequantized rows ----
// Used by Q4_K and Q6_K which have complex dequant logic:
// dequantize row-by-row (scalar), then AVX2 dot product

#ifdef USE_AVX2
// AVX2 dot product: sum(a[i] * b[i]) for i in [0, n)
static inline float dot_product_avx2(const float* a, const float* b, int n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256 a0 = _mm256_loadu_ps(a + i);
        __m256 b0 = _mm256_loadu_ps(b + i);
        acc0 = _mm256_fmadd_ps(a0, b0, acc0);
        __m256 a1 = _mm256_loadu_ps(a + i + 8);
        __m256 b1 = _mm256_loadu_ps(b + i + 8);
        acc1 = _mm256_fmadd_ps(a1, b1, acc1);
    }
    __m256 acc = _mm256_add_ps(acc0, acc1);
    for (; i + 8 <= n; i += 8) {
        __m256 av = _mm256_loadu_ps(a + i);
        __m256 bv = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(av, bv, acc);
    }
    float sum = hsum_avx2(acc);
    for (; i < n; ++i)
        sum += a[i] * b[i];
    return sum;
}
#endif

// ---- Q8_0 block structure and quantization ----
// Q8_0: 2 bytes fp16 scale + 32 bytes int8 = 34 bytes per 32 elements
struct block_q8_0 {
    uint16_t d;
    int8_t qs[32];
};

#ifdef USE_AVX2
static inline uint16_t fp32_to_fp16_bits(float f) {
    __m128 f32 = _mm_set_ss(f);
    __m128i f16 = _mm_cvtps_ph(f32, _MM_ROUND_NEAREST);
    return (uint16_t)_mm_cvtsi128_si32(f16);
}

// Quantize one FP32 row to Q8_0 format
static void quantize_row_q8_0(const float* src, block_q8_0* dst, int k) {
    constexpr int QK8_0 = 32;
    const int nb = (k + QK8_0 - 1) / QK8_0;
    for (int bi = 0; bi < nb; ++bi) {
        int base = bi * QK8_0;
        int n_el = std::min(QK8_0, k - base);
        float amax = 0.0f;
        for (int j = 0; j < n_el; ++j) {
            float v = std::abs(src[base + j]);
            if (v > amax)
                amax = v;
        }
        float d = amax / 127.0f;
        float id = d > 0.0f ? 1.0f / d : 0.0f;
        dst[bi].d = fp32_to_fp16_bits(d);
        for (int j = 0; j < n_el; ++j) {
            int q = (int)(src[base + j] * id + (src[base + j] >= 0 ? 0.5f : -0.5f));
            if (q < -128)
                q = -128;
            if (q > 127)
                q = 127;
            dst[bi].qs[j] = (int8_t)q;
        }
        for (int j = n_el; j < QK8_0; ++j)
            dst[bi].qs[j] = 0;
    }
}

// Unpack 16 nibbles (16 bytes) into 32 bytes in [0..15]
static inline __m256i bytes_from_nibbles_32(const uint8_t* rsi) {
    const __m128i tmp = _mm_loadu_si128((const __m128i*)rsi);
    const __m128i tmp_hi = _mm_srli_epi16(tmp, 4);
    const __m256i bytes = _mm256_set_m128i(tmp_hi, tmp);
    const __m256i lowMask = _mm256_set1_epi8(0xF);
    return _mm256_and_si256(lowMask, bytes);
}

// Signed i8 dot product: sum(x[i]*y[i]) over 32 elements, result as 8 floats
static inline __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
    const __m256i dot16 = _mm256_maddubs_epi16(ax, sy);
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(dot16, _mm256_set1_epi16(1)));
}

// Dot product: one Q4_0 weight block x one Q8_0 activation block
// Returns 8 partial sums (each sum of 4 element-pairs)
static inline __m256 dot_q4_0_q8_0_block_avx2(const uint8_t* w_block, const block_q8_0& q8) {
    uint16_t ws;
    memcpy(&ws, w_block, 2);
    float ws_f = fp16_to_float_scalar(ws);
    float as_f = fp16_to_float_scalar(*(const uint16_t*)&q8.d);
    __m256 combined_scale = _mm256_set1_ps(ws_f * as_f);

    __m256i qx = bytes_from_nibbles_32(w_block + 2);
    qx = _mm256_sub_epi8(qx, _mm256_set1_epi8(8));
    __m256i qy = _mm256_loadu_si256((const __m256i*)q8.qs);

    __m256 partial = mul_sum_i8_pairs_float(qx, qy);
    return _mm256_mul_ps(combined_scale, partial);
}

// Q4_0 x Q8_0 GEMV: quantizes activations once, then integer dot product
static void gemv_q4_0_transB_avx2_q8(const float* a, const uint8_t* w, float* out, int M, int K,
                                     int N) {
    constexpr int BLOCK_SIZE = 32;
    constexpr int BLOCK_BYTES = 18;
    const int blocks_per_row = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int full_blocks = K / BLOCK_SIZE;

    std::vector<block_q8_0> q8_act(blocks_per_row);
    quantize_row_q8_0(a, q8_act.data(), K);

    constexpr int NR = 4;
#pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n += NR) {
        int rows = (n + NR <= N) ? NR : (N - n);
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        const uint8_t* w_row0 = w + (size_t)(n + 0) * blocks_per_row * BLOCK_BYTES;
        const uint8_t* w_row1 =
            (rows > 1) ? w + (size_t)(n + 1) * blocks_per_row * BLOCK_BYTES : nullptr;
        const uint8_t* w_row2 =
            (rows > 2) ? w + (size_t)(n + 2) * blocks_per_row * BLOCK_BYTES : nullptr;
        const uint8_t* w_row3 =
            (rows > 3) ? w + (size_t)(n + 3) * blocks_per_row * BLOCK_BYTES : nullptr;

        for (int bi = 0; bi < full_blocks; ++bi) {
            acc0 = _mm256_add_ps(acc0,
                                 dot_q4_0_q8_0_block_avx2(w_row0 + bi * BLOCK_BYTES, q8_act[bi]));
            if (rows > 1)
                acc1 = _mm256_add_ps(
                    acc1, dot_q4_0_q8_0_block_avx2(w_row1 + bi * BLOCK_BYTES, q8_act[bi]));
            if (rows > 2)
                acc2 = _mm256_add_ps(
                    acc2, dot_q4_0_q8_0_block_avx2(w_row2 + bi * BLOCK_BYTES, q8_act[bi]));
            if (rows > 3)
                acc3 = _mm256_add_ps(
                    acc3, dot_q4_0_q8_0_block_avx2(w_row3 + bi * BLOCK_BYTES, q8_act[bi]));
        }

        if (full_blocks < blocks_per_row) {
            int base = full_blocks * BLOCK_SIZE;
            int remaining = K - base;
            if (remaining > 0) {
                auto process = [&](const uint8_t* row, __m256& acc) {
                    const uint8_t* block = row + (size_t)full_blocks * BLOCK_BYTES;
                    uint16_t sb;
                    memcpy(&sb, block, 2);
                    float scale_f = fp16_to_float_scalar(sb);
                    const uint8_t* qs = block + 2;
                    for (int j = 0; j < 16 && base + j < K; ++j) {
                        float qv =
                            (float)((int)(qs[j] & 0x0F) - 8) * scale_f * q8_act[full_blocks].qs[j];
                        acc = _mm256_fmadd_ps(_mm256_set1_ps(a[base + j]), _mm256_set1_ps(qv), acc);
                    }
                    for (int j = 0; j < 16 && base + 16 + j < K; ++j) {
                        float qv = (float)((int)((qs[j] >> 4) & 0x0F) - 8) * scale_f *
                                   q8_act[full_blocks].qs[16 + j];
                        acc = _mm256_fmadd_ps(_mm256_set1_ps(a[base + 16 + j]), _mm256_set1_ps(qv),
                                              acc);
                    }
                };
                process(w_row0, acc0);
                if (rows > 1 && w_row1)
                    process(w_row1, acc1);
                if (rows > 2 && w_row2)
                    process(w_row2, acc2);
                if (rows > 3 && w_row3)
                    process(w_row3, acc3);
            }
        }

        out[n + 0] = hsum_avx2(acc0);
        if (rows > 1)
            out[n + 1] = hsum_avx2(acc1);
        if (rows > 2)
            out[n + 2] = hsum_avx2(acc2);
        if (rows > 3)
            out[n + 3] = hsum_avx2(acc3);
    }
}
#endif  // USE_AVX2

// ---- Fused QKV projection for Q4_0 decode ----
// Reads input vector once, computes Q + K + V outputs simultaneously.
// Saves 2x input vector reads compared to 3 separate matmul_transB calls.
// All three weight matrices must be Q4_0 with the same K dimension.

#ifdef USE_AVX2
static void gemv_q4_0_fused_qkv_avx2(const float* a, const uint8_t* wq, const uint8_t* wk,
                                     const uint8_t* wv, float* out_q, float* out_k, float* out_v,
                                     int K, int N_q, int N_k, int N_v) {
    constexpr int BLOCK_SIZE = 32;
    constexpr int BLOCK_BYTES = 18;
    const int64_t blocks_per_row = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Quantize activation once — shared by Q, K, V projections
    std::vector<block_q8_0_act> q8_act(blocks_per_row);
    quantize_row_q8_0_act(a, q8_act.data(), K);

    auto process_weight = [&](const uint8_t* w_row, float* out, int N_out) {
        constexpr int RM = 4;
        constexpr int TILE_NR = 8;
#    pragma omp parallel for schedule(static)
        for (int n = 0; n < N_out; n += TILE_NR) {
            int rows = (n + TILE_NR <= N_out) ? TILE_NR : (N_out - n);
            // Process in tiles of RM=4 rows
            for (int t = 0; t < rows; t += RM) {
                int tile_rows = (t + RM <= rows) ? RM : (rows - t);
                const uint8_t* w_ptrs[4];
                for (int ri = 0; ri < tile_rows; ri++)
                    w_ptrs[ri] = w_row + (size_t)(n + t + ri) * blocks_per_row * BLOCK_BYTES;
                // Zero-fill unused pointers for safety
                for (int ri = tile_rows; ri < 4; ri++)
                    w_ptrs[ri] = w_ptrs[0];  // point to valid memory; rows param prevents use

                float tile_out[4];
                gemm_q4_0_tile_4x1_f16c(w_ptrs, q8_act.data(), tile_out,
                                          blocks_per_row, tile_rows);
                for (int ri = 0; ri < tile_rows; ri++)
                    out[n + t + ri] = tile_out[ri];
            }
        }
    };

    process_weight(wq, out_q, N_q);
    process_weight(wk, out_k, N_k);
    process_weight(wv, out_v, N_v);
}
#endif  // USE_AVX2

// ---- Fused FFN gate+up projection for Q4_0 decode ----
// Reads input vector once, computes both gate and up projections simultaneously.
// Then applies SiLU(gate) * up in-place.
// Saves 1x input vector read compared to 2 separate matmul_transB calls.

#ifdef USE_AVX2
static void gemv_q4_0_fused_ffn_up_avx2(const float* a, const uint8_t* w_gate, const uint8_t* w_up,
                                        float* out, int K, int N) {
    constexpr int BLOCK_SIZE = 32;
    constexpr int BLOCK_BYTES = 18;
    const int64_t blocks_per_row = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    auto silu = [](float x) -> float { return x / (1.0f + std::exp(-x)); };

    // Quantize activation once
    std::vector<block_q8_0_act> q8_act(blocks_per_row);
    quantize_row_q8_0_act(a, q8_act.data(), K);

    constexpr int RM = 4;
    constexpr int TILE_NR = 8;
#    pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n += TILE_NR) {
        int rows = (n + TILE_NR <= N) ? TILE_NR : (N - n);
        // Temporary storage for gate and up results
        float gate_buf[8], up_buf[8];

        // Process gate weights with GEMM tiles
        for (int t = 0; t < rows; t += RM) {
            int tile_rows = (t + RM <= rows) ? RM : (rows - t);
            const uint8_t* w_ptrs[4];
            for (int ri = 0; ri < tile_rows; ri++)
                w_ptrs[ri] = w_gate + (size_t)(n + t + ri) * blocks_per_row * BLOCK_BYTES;
            for (int ri = tile_rows; ri < 4; ri++)
                w_ptrs[ri] = w_ptrs[0];
            gemm_q4_0_tile_4x1_f16c(w_ptrs, q8_act.data(), gate_buf + t,
                                      blocks_per_row, tile_rows);
        }
        // Process up weights with GEMM tiles
        for (int t = 0; t < rows; t += RM) {
            int tile_rows = (t + RM <= rows) ? RM : (rows - t);
            const uint8_t* w_ptrs[4];
            for (int ri = 0; ri < tile_rows; ri++)
                w_ptrs[ri] = w_up + (size_t)(n + t + ri) * blocks_per_row * BLOCK_BYTES;
            for (int ri = tile_rows; ri < 4; ri++)
                w_ptrs[ri] = w_ptrs[0];
            gemm_q4_0_tile_4x1_f16c(w_ptrs, q8_act.data(), up_buf + t,
                                      blocks_per_row, tile_rows);
        }

        // Apply SiLU(gate) * up
        for (int ri = 0; ri < rows; ri++)
            out[n + ri] = silu(gate_buf[ri]) * up_buf[ri];
    }
}

// ---- Fused FFN down-projection + residual for Q4_0 decode ----
// Computes ffn_mid @ w2 + residual in a single pass.
// Saves 1x intermediate read+write (the matmul output tensor) per layer.
static void gemv_q4_0_ffn_down_residual_avx2(const float* a, const uint8_t* w,
                                             const float* residual, float* out, int K, int N) {
    constexpr int BLOCK_SIZE = 32;
    constexpr int BLOCK_BYTES = 18;
    const int64_t blocks_per_row = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Quantize activation once
    std::vector<block_q8_0_act> q8_act(blocks_per_row);
    quantize_row_q8_0_act(a, q8_act.data(), K);

    constexpr int RM = 4;
    constexpr int TILE_NR = 8;
#    pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n += TILE_NR) {
        int rows = (n + TILE_NR <= N) ? TILE_NR : (N - n);

        for (int t = 0; t < rows; t += RM) {
            int tile_rows = (t + RM <= rows) ? RM : (rows - t);
            const uint8_t* w_ptrs[4];
            for (int ri = 0; ri < tile_rows; ri++)
                w_ptrs[ri] = w + (size_t)(n + t + ri) * blocks_per_row * BLOCK_BYTES;
            for (int ri = tile_rows; ri < 4; ri++)
                w_ptrs[ri] = w_ptrs[0];

            float tile_out[4];
            gemm_q4_0_tile_4x1_f16c(w_ptrs, q8_act.data(), tile_out,
                                      blocks_per_row, tile_rows);
            for (int ri = 0; ri < tile_rows; ri++)
                out[n + t + ri] = tile_out[ri] + residual[n + t + ri];
        }
    }
}

// ---- Fused attention output projection + residual for Q4_0 decode ----
// Computes attn_out @ wo + hidden_residual in a single pass.
// Saves 1 intermediate write + 1 add per layer.
static void gemv_q4_0_attn_proj_residual_avx2(const float* a, const uint8_t* w,
                                               const float* residual, float* out, int K, int N) {
    constexpr int BLOCK_SIZE = 32;
    constexpr int BLOCK_BYTES = 18;
    const int64_t blocks_per_row = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Quantize activation once
    std::vector<block_q8_0_act> q8_act(blocks_per_row);
    quantize_row_q8_0_act(a, q8_act.data(), K);

    constexpr int RM = 4;
    constexpr int TILE_NR = 8;
#    pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n += TILE_NR) {
        int rows = (n + TILE_NR <= N) ? TILE_NR : (N - n);

        for (int t = 0; t < rows; t += RM) {
            int tile_rows = (t + RM <= rows) ? RM : (rows - t);
            const uint8_t* w_ptrs[4];
            for (int ri = 0; ri < tile_rows; ri++)
                w_ptrs[ri] = w + (size_t)(n + t + ri) * blocks_per_row * BLOCK_BYTES;
            for (int ri = tile_rows; ri < 4; ri++)
                w_ptrs[ri] = w_ptrs[0];

            float tile_out[4];
            gemm_q4_0_tile_4x1_f16c(w_ptrs, q8_act.data(), tile_out,
                                      blocks_per_row, tile_rows);
            for (int ri = 0; ri < tile_rows; ri++)
                out[n + t + ri] = tile_out[ri] + residual[n + t + ri];
        }
    }
}

// ---- Fused FFN down-projection + residual for Q4_1 decode ----
static void gemv_q4_1_ffn_down_residual_avx2(const float* a, const uint8_t* w,
                                               const float* residual, float* out, int K, int N) {
    constexpr int BLOCK_SIZE = 32;
    constexpr int BLOCK_BYTES = 20;
    constexpr int NR = 8;
    const int blocks_per_row = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int full_blocks = K / BLOCK_SIZE;

#    pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n += NR) {
        int rows = (n + NR <= N) ? NR : (N - n);

        __m256 acc[8];
        for (int ri = 0; ri < rows; ri++)
            acc[ri] = _mm256_setzero_ps();

        const uint8_t* w_row[8];
        for (int ri = 0; ri < rows; ri++)
            w_row[ri] = w + (size_t)(n + ri) * blocks_per_row * BLOCK_BYTES;

        auto prefetch_all = [&](int bi) {
            for (int ri = 0; ri < rows; ri++)
                _mm_prefetch((const char*)(w_row[ri] + (size_t)bi * BLOCK_BYTES),
                             _MM_HINT_T0);
        };

        int bi = 0;
        for (; bi + 1 < full_blocks; bi += 2) {
            if (bi + 2 < full_blocks)
                prefetch_all(bi + 2);
            for (int rip = 0; rip < 2; rip++) {
                int bj = bi + rip;
                for (int ri = 0; ri < rows; ri++) {
                    const uint8_t* blk = w_row[ri] + (size_t)bj * BLOCK_BYTES;
                    uint16_t db, mb;
                    memcpy(&db, blk, 2);
                    memcpy(&mb, blk + 2, 2);
                    __m256 d = fp16_to_fp32_broadcast_avx2(db);
                    __m256 m = fp16_to_fp32_broadcast_avx2(mb);
                    __m256 partial, sum_a;
                    q4_1_block_partial_avx2(a, bj * BLOCK_SIZE, blk + 4, partial, sum_a);
                    acc[ri] = _mm256_fmadd_ps(d, partial, acc[ri]);
                    acc[ri] = _mm256_fmadd_ps(m, sum_a, acc[ri]);
                }
            }
        }
        if (bi < full_blocks) {
            for (int ri = 0; ri < rows; ri++) {
                const uint8_t* blk = w_row[ri] + (size_t)bi * BLOCK_BYTES;
                uint16_t db, mb;
                memcpy(&db, blk, 2);
                memcpy(&mb, blk + 2, 2);
                __m256 d = fp16_to_fp32_broadcast_avx2(db);
                __m256 m = fp16_to_fp32_broadcast_avx2(mb);
                __m256 partial, sum_a;
                q4_1_block_partial_avx2(a, bi * BLOCK_SIZE, blk + 4, partial, sum_a);
                acc[ri] = _mm256_fmadd_ps(d, partial, acc[ri]);
                acc[ri] = _mm256_fmadd_ps(m, sum_a, acc[ri]);
            }
        }

        if (full_blocks < blocks_per_row) {
            int base = full_blocks * BLOCK_SIZE;
            int remaining = K - base;
            if (remaining > 0) {
                auto proc_partial = [&](const uint8_t* w_row, __m256& acc_val) {
                    const uint8_t* block = w_row + (size_t)full_blocks * BLOCK_BYTES;
                    float d_val = fp16_to_float_scalar(*reinterpret_cast<const uint16_t*>(block));
                    float m_val = fp16_to_float_scalar(*reinterpret_cast<const uint16_t*>(block + 2));
                    const uint8_t* qs = block + 4;
                    for (int j = 0; j < 16 && base + j < K; ++j) {
                        float qv = static_cast<float>(qs[j] & 0xF) * d_val + m_val;
                        acc_val = _mm256_fmadd_ps(_mm256_set1_ps(a[base + j]),
                                                   _mm256_set1_ps(qv), acc_val);
                    }
                    for (int j = 0; j < 16 && base + 16 + j < K; ++j) {
                        float qv = static_cast<float>((qs[j] >> 4) & 0xF) * d_val + m_val;
                        acc_val = _mm256_fmadd_ps(_mm256_set1_ps(a[base + 16 + j]),
                                                   _mm256_set1_ps(qv), acc_val);
                    }
                };
                for (int ri = 0; ri < rows; ri++)
                    proc_partial(w_row[ri], acc[ri]);
            }
        }

        for (int ri = 0; ri < rows; ri++)
            out[n + ri] = hsum_avx2(acc[ri]) + residual[n + ri];
    }
}
#endif  // USE_AVX2

// ---- Q4_K fused GEMV ----
// Q4_K super-block: 144 bytes per 256 elements
// Layout: [2B d (fp16)] [2B dmin (fp16)] [12B scales] [128B qs]
// Dequant: value = d * sc * q4 - dmin * m  (q4: unsigned 0-15)
//
// Fused approach:
//   dot = sum(a[k] * (d * sc * q4[k] - dmin * m))
//       = d * sc * sum(a[k] * q4[k]) - dmin * m * sum(a[k])
// Accumulate into acc_sc and acc_min, merge at end.
//
// Key optimizations:
// - NR=4 row parallelism for M=1 decode
// - Split accumulation avoids per-element multiply in the inner loop
// - On-the-fly dequantization: no intermediate FP32 buffer
//   (saves ~63% memory traffic vs dequant+dot fallback)

#ifdef USE_AVX2

// Helper: decode Q4_K packed scale/min pair
static inline void q4_k_get_scale_min(int j, const uint8_t* q, float& sc, float& mn) {
    uint8_t sc_u, mn_u;
    if (j < 4) {
        sc_u = q[j] & 63;
        mn_u = q[j + 4] & 63;
    } else {
        sc_u = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        mn_u = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
    sc = static_cast<float>(sc_u);
    mn = static_cast<float>(mn_u);
}

// Helper: process one Q4_K sub-block (64 elements, 32 bytes of qs)
// Lower nibbles of qs → elements [base..base+31], scale=sc_even, min=m_even
// Upper nibbles of qs → elements [base+32..base+63], scale=sc_odd, min=m_odd
// Accumulates: acc_sc += d * sc * partial, acc_min += dmin * m * sum_a
static inline void q4_k_subblock_dot_avx2(const float* a_row, int base, const uint8_t* qs,
                                          const __m256 d_v, const __m256 dmin_v, float sc_even,
                                          float m_even, float sc_odd, float m_odd, __m256& acc_sc,
                                          __m256& acc_min) {
    const __m128i lo_mask = _mm_set1_epi8(0x0F);

    // --- First 16 bytes of qs ---
    __m128i q8_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs));
    __m128i q_lo_0 = _mm_and_si128(q8_0, lo_mask);
    __m128i q_hi_0 = _mm_and_si128(_mm_srli_epi16(q8_0, 4), lo_mask);

    // Lower nibbles → elements [base..base+15]
    __m256 a0 = _mm256_loadu_ps(a_row + base);
    __m256 pl0 = _mm256_mul_ps(a0, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_lo_0)));
    __m256 a1 = _mm256_loadu_ps(a_row + base + 8);
    pl0 = _mm256_fmadd_ps(a1, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_lo_0, 8))),
                          pl0);
    __m256 sl0 = _mm256_add_ps(a0, a1);

    // Upper nibbles → elements [base+32..base+47]
    __m256 a2 = _mm256_loadu_ps(a_row + base + 32);
    __m256 pu0 = _mm256_mul_ps(a2, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_hi_0)));
    __m256 a3 = _mm256_loadu_ps(a_row + base + 40);
    pu0 = _mm256_fmadd_ps(a3, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_hi_0, 8))),
                          pu0);
    __m256 su0 = _mm256_add_ps(a2, a3);

    // --- Second 16 bytes of qs ---
    __m128i q8_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs + 16));
    __m128i q_lo_1 = _mm_and_si128(q8_1, lo_mask);
    __m128i q_hi_1 = _mm_and_si128(_mm_srli_epi16(q8_1, 4), lo_mask);

    // Lower nibbles → elements [base+16..base+31]
    __m256 a4 = _mm256_loadu_ps(a_row + base + 16);
    __m256 pl1 = _mm256_mul_ps(a4, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_lo_1)));
    __m256 a5 = _mm256_loadu_ps(a_row + base + 24);
    pl1 = _mm256_fmadd_ps(a5, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_lo_1, 8))),
                          pl1);
    __m256 sl1 = _mm256_add_ps(a4, a5);

    // Upper nibbles → elements [base+48..base+63]
    __m256 a6 = _mm256_loadu_ps(a_row + base + 48);
    __m256 pu1 = _mm256_mul_ps(a6, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_hi_1)));
    __m256 a7 = _mm256_loadu_ps(a_row + base + 56);
    pu1 = _mm256_fmadd_ps(a7, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_hi_1, 8))),
                          pu1);
    __m256 su1 = _mm256_add_ps(a6, a7);

    // Merge lower and upper partials
    __m256 partial_lower = _mm256_add_ps(pl0, pl1);
    __m256 sum_a_lower = _mm256_add_ps(sl0, sl1);
    __m256 partial_upper = _mm256_add_ps(pu0, pu1);
    __m256 sum_a_upper = _mm256_add_ps(su0, su1);

    // Accumulate: acc_sc += d*sc*partial, acc_min += dmin*m*sum_a
    __m256 dsc_e = _mm256_mul_ps(d_v, _mm256_set1_ps(sc_even));
    __m256 dsc_o = _mm256_mul_ps(d_v, _mm256_set1_ps(sc_odd));
    __m256 dmm_e = _mm256_mul_ps(dmin_v, _mm256_set1_ps(m_even));
    __m256 dmm_o = _mm256_mul_ps(dmin_v, _mm256_set1_ps(m_odd));

    acc_sc = _mm256_fmadd_ps(dsc_e, partial_lower, acc_sc);
    acc_sc = _mm256_fmadd_ps(dsc_o, partial_upper, acc_sc);
    acc_min = _mm256_fmadd_ps(dmm_e, sum_a_lower, acc_min);
    acc_min = _mm256_fmadd_ps(dmm_o, sum_a_upper, acc_min);
}

// Prefill helper for Q4_K matmul with M > 1.
// Simple (n, m, bi) loop order, separated from decode path to avoid
// affecting M=1 codegen.
static void gemv_q4_k_transB_prefill_avx2(const float* a, const uint8_t* w, float* out, int M,
                                          int K, int N) {
    constexpr int SB_SIZE = 256;
    constexpr int SB_BYTES = 144;
    const int sbs_per_row = (K + SB_SIZE - 1) / SB_SIZE;
    const int full_sbs = K / SB_SIZE;

#    pragma omp parallel for schedule(dynamic, 4) if (N > 4)
    for (int n = 0; n < N; ++n) {
        const uint8_t* w_row = w + (size_t)n * sbs_per_row * SB_BYTES;
        for (int m = 0; m < M; ++m) {
            const float* a_row = a + m * K;
            __m256 acc_sc = _mm256_setzero_ps();
            __m256 acc_min = _mm256_setzero_ps();

            for (int bi = 0; bi < full_sbs; ++bi) {
                const uint8_t* sb = w_row + (size_t)bi * SB_BYTES;
                uint16_t _db, _dmb;
                memcpy(&_db, sb, 2);
                memcpy(&_dmb, sb + 2, 2);
                __m256 _dv = fp16_to_fp32_broadcast_avx2(_db);
                __m256 _dmv = fp16_to_fp32_broadcast_avx2(_dmb);
                const uint8_t* _scales = sb + 4;
                const uint8_t* _qs = sb + 16;
                int _is = 0;
                for (int _j = 0; _j < 4; ++_j) {
                    int _sub_base = bi * 256 + _j * 64;
                    float _sc_e, _m_e, _sc_o, _m_o;
                    q4_k_get_scale_min(_is, _scales, _sc_e, _m_e);
                    q4_k_get_scale_min(_is + 1, _scales, _sc_o, _m_o);
                    q4_k_subblock_dot_avx2(a_row, _sub_base, _qs, _dv, _dmv, _sc_e, _m_e, _sc_o,
                                           _m_o, acc_sc, acc_min);
                    _qs += 32;
                    _is += 2;
                }
            }

            __m256 acc = _mm256_sub_ps(acc_sc, acc_min);

            if (full_sbs < sbs_per_row) {
                int bi = full_sbs;
                int base = bi * SB_SIZE;
                int remaining = K - base;
                if (remaining > 0) {
                    const uint8_t* sb = w_row + (size_t)bi * SB_BYTES;
                    uint16_t d_bits, dmin_bits;
                    memcpy(&d_bits, sb, 2);
                    memcpy(&dmin_bits, sb + 2, 2);
                    float d_val = fp16_to_float_scalar(d_bits);
                    float dmin_val = fp16_to_float_scalar(dmin_bits);
                    const uint8_t* scales = sb + 4;
                    const uint8_t* qs = sb + 16;
                    int is_idx = 0;
                    for (int j = 0; j < 4 && base + j * 64 < K; ++j) {
                        float sc1, m1, sc2, m2;
                        q4_k_get_scale_min(is_idx, scales, sc1, m1);
                        q4_k_get_scale_min(is_idx + 1, scales, sc2, m2);
                        float d1 = d_val * sc1;
                        float m1_val = dmin_val * m1;
                        float d2 = d_val * sc2;
                        float m2_val = dmin_val * m2;
                        int sub_base = base + j * 64;
                        for (int l = 0; l < 16 && sub_base + l < K; ++l) {
                            float qv = d1 * static_cast<float>(qs[l] & 0xF) - m1_val;
                            acc = _mm256_fmadd_ps(_mm256_set1_ps(a_row[sub_base + l]),
                                                  _mm256_set1_ps(qv), acc);
                        }
                        for (int l = 0; l < 16 && sub_base + 16 + l < K; ++l) {
                            float qv = d1 * static_cast<float>(qs[16 + l] & 0xF) - m1_val;
                            acc = _mm256_fmadd_ps(_mm256_set1_ps(a_row[sub_base + 16 + l]),
                                                  _mm256_set1_ps(qv), acc);
                        }
                        for (int l = 0; l < 16 && sub_base + 32 + l < K; ++l) {
                            float qv = d2 * static_cast<float>(qs[l] >> 4) - m2_val;
                            acc = _mm256_fmadd_ps(_mm256_set1_ps(a_row[sub_base + 32 + l]),
                                                  _mm256_set1_ps(qv), acc);
                        }
                        for (int l = 0; l < 16 && sub_base + 48 + l < K; ++l) {
                            float qv = d2 * static_cast<float>(qs[16 + l] >> 4) - m2_val;
                            acc = _mm256_fmadd_ps(_mm256_set1_ps(a_row[sub_base + 48 + l]),
                                                  _mm256_set1_ps(qv), acc);
                        }
                        qs += 32;
                        is_idx += 2;
                    }
                }
            }
            out[m * N + n] = hsum_avx2(acc);
        }
    }
}

static void gemv_q4_k_transB_avx2(const float* a, const uint8_t* w, float* out, int M, int K,
                                  int N) {
    constexpr int SB_SIZE = 256;
    constexpr int SB_BYTES = 144;
    const int sbs_per_row = (K + SB_SIZE - 1) / SB_SIZE;
    const int full_sbs = K / SB_SIZE;

#    define Q4K_PROCESS_SB(a_row, w_row, bi, acc_sc, acc_min)                                      \
        do {                                                                                       \
            const uint8_t* _sb = (w_row) + (size_t)(bi)*144;                                       \
            uint16_t _db, _dmb;                                                                    \
            memcpy(&_db, _sb, 2);                                                                  \
            memcpy(&_dmb, _sb + 2, 2);                                                             \
            __m256 _dv = fp16_to_fp32_broadcast_avx2(_db);                                         \
            __m256 _dmv = fp16_to_fp32_broadcast_avx2(_dmb);                                       \
            const uint8_t* _scales = _sb + 4;                                                      \
            const uint8_t* _qs = _sb + 16;                                                         \
            int _is = 0;                                                                           \
            for (int _j = 0; _j < 4; ++_j) {                                                       \
                int _sub_base = (bi)*256 + _j * 64;                                                \
                float _sc_e, _m_e, _sc_o, _m_o;                                                    \
                q4_k_get_scale_min(_is, _scales, _sc_e, _m_e);                                     \
                q4_k_get_scale_min(_is + 1, _scales, _sc_o, _m_o);                                 \
                q4_k_subblock_dot_avx2(a_row, _sub_base, _qs, _dv, _dmv, _sc_e, _m_e, _sc_o, _m_o, \
                                       acc_sc, acc_min);                                           \
                _qs += 32;                                                                         \
                _is += 2;                                                                          \
            }                                                                                      \
        } while (0)

#    define Q4K_PREFETCH_SB(w_row, bi)                        \
        do {                                                  \
            const uint8_t* _psb = (w_row) + (size_t)(bi)*144; \
            _mm_prefetch((const char*)_psb, _MM_HINT_T0);     \
        } while (0)

    if (M == 1) {
        // Decode path: NR=4 row parallelism
        constexpr int NR = 4;
        const float* a_row = a;

#    pragma omp parallel for schedule(static)
        for (int n = 0; n < N; n += NR) {
            int rows = (n + NR <= N) ? NR : (N - n);

            __m256 acc0_sc = _mm256_setzero_ps(), acc0_min = _mm256_setzero_ps();
            __m256 acc1_sc = _mm256_setzero_ps(), acc1_min = _mm256_setzero_ps();
            __m256 acc2_sc = _mm256_setzero_ps(), acc2_min = _mm256_setzero_ps();
            __m256 acc3_sc = _mm256_setzero_ps(), acc3_min = _mm256_setzero_ps();

            const uint8_t* w_row0 = w + (size_t)(n + 0) * sbs_per_row * SB_BYTES;
            const uint8_t* w_row1 =
                (rows > 1) ? w + (size_t)(n + 1) * sbs_per_row * SB_BYTES : nullptr;
            const uint8_t* w_row2 =
                (rows > 2) ? w + (size_t)(n + 2) * sbs_per_row * SB_BYTES : nullptr;
            const uint8_t* w_row3 =
                (rows > 3) ? w + (size_t)(n + 3) * sbs_per_row * SB_BYTES : nullptr;

            // Prefetch first super-block
            Q4K_PREFETCH_SB(w_row0, 0);
            if (rows > 1)
                Q4K_PREFETCH_SB(w_row1, 0);
            if (rows > 2)
                Q4K_PREFETCH_SB(w_row2, 0);
            if (rows > 3)
                Q4K_PREFETCH_SB(w_row3, 0);

            for (int bi = 0; bi < full_sbs; ++bi) {
                Q4K_PROCESS_SB(a_row, w_row0, bi, acc0_sc, acc0_min);
                if (rows > 1)
                    Q4K_PROCESS_SB(a_row, w_row1, bi, acc1_sc, acc1_min);
                if (rows > 2)
                    Q4K_PROCESS_SB(a_row, w_row2, bi, acc2_sc, acc2_min);
                if (rows > 3)
                    Q4K_PROCESS_SB(a_row, w_row3, bi, acc3_sc, acc3_min);

                // Prefetch next super-block
                if (bi + 1 < full_sbs) {
                    Q4K_PREFETCH_SB(w_row0, bi + 1);
                    if (rows > 1)
                        Q4K_PREFETCH_SB(w_row1, bi + 1);
                    if (rows > 2)
                        Q4K_PREFETCH_SB(w_row2, bi + 1);
                    if (rows > 3)
                        Q4K_PREFETCH_SB(w_row3, bi + 1);
                }
            }

            // Merge: acc = acc_sc - acc_min
            __m256 acc0 = _mm256_sub_ps(acc0_sc, acc0_min);
            __m256 acc1 = _mm256_sub_ps(acc1_sc, acc1_min);
            __m256 acc2 = _mm256_sub_ps(acc2_sc, acc2_min);
            __m256 acc3 = _mm256_sub_ps(acc3_sc, acc3_min);

            // Handle partial last super-block (scalar fallback)
            if (full_sbs < sbs_per_row) {
                int bi = full_sbs;
                int base = bi * SB_SIZE;
                int remaining = K - base;
                if (remaining > 0) {
                    auto process_partial = [&](const uint8_t* w_row, __m256& acc_ref) {
                        const uint8_t* sb = w_row + (size_t)bi * SB_BYTES;
                        uint16_t d_bits, dmin_bits;
                        memcpy(&d_bits, sb, 2);
                        memcpy(&dmin_bits, sb + 2, 2);
                        float d_val = fp16_to_float_scalar(d_bits);
                        float dmin_val = fp16_to_float_scalar(dmin_bits);
                        const uint8_t* scales = sb + 4;
                        const uint8_t* qs = sb + 16;

                        int is_idx = 0;
                        for (int j = 0; j < 4 && base + j * 64 < K; ++j) {
                            float sc1, m1, sc2, m2;
                            q4_k_get_scale_min(is_idx, scales, sc1, m1);
                            q4_k_get_scale_min(is_idx + 1, scales, sc2, m2);
                            float d1 = d_val * sc1;
                            float m1_val = dmin_val * m1;
                            float d2 = d_val * sc2;
                            float m2_val = dmin_val * m2;
                            int sub_base = base + j * 64;
                            for (int l = 0; l < 16 && sub_base + l < K; ++l) {
                                float qv = d1 * static_cast<float>(qs[l] & 0xF) - m1_val;
                                acc_ref = _mm256_fmadd_ps(_mm256_set1_ps(a_row[sub_base + l]),
                                                          _mm256_set1_ps(qv), acc_ref);
                            }
                            for (int l = 0; l < 16 && sub_base + 16 + l < K; ++l) {
                                float qv = d1 * static_cast<float>(qs[16 + l] & 0xF) - m1_val;
                                acc_ref = _mm256_fmadd_ps(_mm256_set1_ps(a_row[sub_base + 16 + l]),
                                                          _mm256_set1_ps(qv), acc_ref);
                            }
                            for (int l = 0; l < 16 && sub_base + 32 + l < K; ++l) {
                                float qv = d2 * static_cast<float>(qs[l] >> 4) - m2_val;
                                acc_ref = _mm256_fmadd_ps(_mm256_set1_ps(a_row[sub_base + 32 + l]),
                                                          _mm256_set1_ps(qv), acc_ref);
                            }
                            for (int l = 0; l < 16 && sub_base + 48 + l < K; ++l) {
                                float qv = d2 * static_cast<float>(qs[16 + l] >> 4) - m2_val;
                                acc_ref = _mm256_fmadd_ps(_mm256_set1_ps(a_row[sub_base + 48 + l]),
                                                          _mm256_set1_ps(qv), acc_ref);
                            }
                            qs += 32;
                            is_idx += 2;
                        }
                    };
                    process_partial(w_row0, acc0);
                    if (rows > 1 && w_row1)
                        process_partial(w_row1, acc1);
                    if (rows > 2 && w_row2)
                        process_partial(w_row2, acc2);
                    if (rows > 3 && w_row3)
                        process_partial(w_row3, acc3);
                }
            }

            out[n + 0] = hsum_avx2(acc0);
            if (rows > 1)
                out[n + 1] = hsum_avx2(acc1);
            if (rows > 2)
                out[n + 2] = hsum_avx2(acc2);
            if (rows > 3)
                out[n + 3] = hsum_avx2(acc3);
        }
    } else {
        gemv_q4_k_transB_prefill_avx2(a, w, out, M, K, N);
    }

#    undef Q4K_PROCESS_SB
#    undef Q4K_PREFETCH_SB
}

// ---- Q3_K GEMV (AVX2) using Q8_K fused dot product ----
// Strategy: quantize input to Q8_K once, then use dot_q3_K_q8_K_avx2 for each row.
// Ported from llama.cpp ggml_vec_dot_q3_K_q8_K but uses proven Q6_K scale shuffle approach.
// Key technique: split Q3_K into low 2-bit and high 1-bit parts,
// use _mm256_maddubs_epi16 separately, then subtract: (low - high) * scale.

// Process one Q3_K super-block (256 elements) with one Q8_K activation block.
// Returns the dot product contribution as an 8-wide float partial sum (before hsum).
static inline __m256 q3_k_sb_dot_avx2(const uint8_t* q3_sb, const block_q8_K* q8) {
    constexpr int QK_K = 256;
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i mone = _mm256_set1_epi8(1);
    const __m128i m32 = _mm_set1_epi8(32);

    static const uint32_t kmask1 = 0x03030303;
    static const uint32_t kmask2 = 0x0f0f0f0f;

    const block_q3_K* x = reinterpret_cast<const block_q3_K*>(q3_sb);
    const float d = q8->d * fp16_to_float_scalar(x->d);

    const uint8_t* q3 = x->qs;
    const int8_t* q8d = q8->qs;

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
        const __m256i q3bits = _mm256_loadu_si256((const __m256i*)q3);
        q3 += 32;

        const __m256i q3l_0 = _mm256_and_si256(q3bits, m3);
        const __m256i q3h_0 = _mm256_slli_epi16(
            _mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
        ++bit;

        const __m256i q3l_1 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 2), m3);
        const __m256i q3h_1 = _mm256_slli_epi16(
            _mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
        ++bit;

        const __m256i q3l_2 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 4), m3);
        const __m256i q3h_2 = _mm256_slli_epi16(
            _mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
        ++bit;

        const __m256i q3l_3 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 6), m3);
        const __m256i q3h_3 = _mm256_slli_epi16(
            _mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
        ++bit;

        const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8d);
        q8d += 32;
        const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8d);
        q8d += 32;
        const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8d);
        q8d += 32;
        const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8d);
        q8d += 32;

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

        p16_0 = _mm256_add_epi32(p16_0, p16_1);
        p16_2 = _mm256_add_epi32(p16_2, p16_3);
        sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_2));
    }

    return _mm256_mul_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi));
}

// Dot product of one Q3_K weight row with one Q8_K activation block row.
// Returns the sum for (nb) Q3_K super-blocks.
static inline float dot_q3_K_q8_K_avx2(const uint8_t* q3_row, const block_q8_K* q8, int nb) {
    constexpr int QK_K = 256;
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i mone = _mm256_set1_epi8(1);
    const __m128i m32 = _mm_set1_epi8(32);

    static const uint32_t kmask1 = 0x03030303;
    static const uint32_t kmask2 = 0x0f0f0f0f;

    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const block_q3_K* x = reinterpret_cast<const block_q3_K*>(q3_row) + i;
        const block_q8_K* y = q8 + i;

        _mm_prefetch((const char*)((const block_q3_K*)q3_row + i + 1), _MM_HINT_T0);
        _mm_prefetch((const char*)(q8 + i + 1), _MM_HINT_T0);

        const float d = y->d * fp16_to_float_scalar(x->d);

        const uint8_t* q3 = x->qs;
        const int8_t* q8d = y->qs;

        // Set up scales: unpack 12-byte packed scales into 16 signed 6-bit values
        // Using same approach as llama.cpp: pack into __m128i then subtract 32
        uint32_t aux[3];
        memcpy(aux, x->scales, 12);
        __m128i scales128 = _mm_set_epi32(
                ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4),
                ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4),
                (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4),
                (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4));
        scales128 = _mm_sub_epi8(scales128, m32);

        // high bit mask
        const __m256i hbits = _mm256_loadu_si256((const __m256i*)x->hmask);

        __m256i sumi = _mm256_setzero_si256();

        int bit = 0;
        int is = 0;

        for (int j = 0; j < QK_K / 128; ++j) {
            // load low 2 bits
            const __m256i q3bits = _mm256_loadu_si256((const __m256i*)q3);
            q3 += 32;

            // prepare low and high bits
            const __m256i q3l_0 = _mm256_and_si256(q3bits, m3);
            const __m256i q3h_0 = _mm256_slli_epi16(
                _mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
            ++bit;

            const __m256i q3l_1 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 2), m3);
            const __m256i q3h_1 = _mm256_slli_epi16(
                _mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
            ++bit;

            const __m256i q3l_2 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 4), m3);
            const __m256i q3h_2 = _mm256_slli_epi16(
                _mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
            ++bit;

            const __m256i q3l_3 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 6), m3);
            const __m256i q3h_3 = _mm256_slli_epi16(
                _mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
            ++bit;

            // load Q8 quants
            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8d);
            q8d += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8d);
            q8d += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8d);
            q8d += 32;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8d);
            q8d += 32;

            // Dot product: multiply low 2 bits and high 1 bit separately,
            // then subtract. High bit part is 0 or 4 (shifted left by 2).
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

            // multiply with scales using Q6_K proven approach:
            // get_scale_shuffle (128-bit, 8 entries) + _mm256_cvtepi8_epi16
            const __m128i scale_0 = _mm_shuffle_epi8(scales128, get_scale_shuffle(is + 0));
            const __m128i scale_1 = _mm_shuffle_epi8(scales128, get_scale_shuffle(is + 1));
            const __m128i scale_2 = _mm_shuffle_epi8(scales128, get_scale_shuffle(is + 2));
            const __m128i scale_3 = _mm_shuffle_epi8(scales128, get_scale_shuffle(is + 3));
            is += 4;

            p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
            p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);
            p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2);
            p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3);

            // accumulate
            p16_0 = _mm256_add_epi32(p16_0, p16_1);
            p16_2 = _mm256_add_epi32(p16_2, p16_3);
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_2));
        }

        // multiply with block scale and accumulate
        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }

    return hsum_avx2(acc);
}

static void gemv_q3_k_transB_avx2(const float* a, const uint8_t* w, float* out, int M, int K,
                                   int N) {
    constexpr int QK_K = 256;
    constexpr int Q3_K_BLOCK_BYTES = 110;
    const int nb = (K + QK_K - 1) / QK_K;

    if (M == 1) {
        std::vector<block_q8_K> q8_buf(nb);
        quantize_row_q8_K(a, q8_buf.data(), K);

#    pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const uint8_t* q3_row = w + (size_t)n * nb * Q3_K_BLOCK_BYTES;
            out[n] = dot_q3_K_q8_K_avx2(q3_row, q8_buf.data(), nb);
        }
    } else {
        std::vector<block_q8_K> q8_all(M * nb);
        for (int m = 0; m < M; ++m) {
            quantize_row_q8_K(a + m * K, q8_all.data() + m * nb, K);
        }

#    pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const uint8_t* q3_row = w + (size_t)n * nb * Q3_K_BLOCK_BYTES;
            for (int m = 0; m < M; ++m) {
                out[m * N + n] = dot_q3_K_q8_K_avx2(q3_row, q8_all.data() + m * nb, nb);
            }
        }
    }
}

// Q3_K Batch-GEMV (transB layout) for M > 1 (prefill).
// Q3_K weight decoding (scales, hmask, qs) is done once per weight block,
// then reused for all M input vectors.
// Uses pre-allocated thread-local accumulators to avoid per-row heap alloc.
static void gemv_q3_k_transB_batch_avx2(const float* a, const uint8_t* w, float* out,
                                          int M, int K, int N) {
    constexpr int QK_K = 256;
    constexpr int Q3_K_BLOCK_BYTES = 110;
    const int nb = (K + QK_K - 1) / QK_K;

    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i mone = _mm256_set1_epi8(1);
    const __m128i m32 = _mm_set1_epi8(32);

    static const uint32_t kmask1 = 0x03030303;
    static const uint32_t kmask2 = 0x0f0f0f0f;

    // Quantize all M input vectors to Q8_K
    std::vector<block_q8_K> q8_src(M * nb);
    for (int m = 0; m < M; ++m) {
        quantize_row_q8_K(a + m * K, q8_src.data() + m * nb, K);
    }

    // Transpose Q8_K data: q8_t[i * M + m] = q8_src[m * nb + i]
    // This makes per-block access sequential for all M rows
    std::vector<block_q8_K> q8_t(nb * M);
    for (int i = 0; i < nb; ++i) {
        for (int m = 0; m < M; ++m) {
            q8_t[i * M + m] = q8_src[m * nb + i];
        }
    }

    // Pre-allocate aligned accumulators per OpenMP thread
    int max_threads = omp_get_max_threads();
    __m256* acc_storage = static_cast<__m256*>(_mm_malloc(max_threads * M * sizeof(__m256), 32));

#pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        int tid = omp_get_thread_num();
        __m256* acc = acc_storage + (size_t)tid * M;
        for (int m = 0; m < M; ++m) acc[m] = _mm256_setzero_ps();

        const uint8_t* q3_row = w + (size_t)n * nb * Q3_K_BLOCK_BYTES;

        for (int i = 0; i < nb; ++i) {
            const uint8_t* sb_ptr = q3_row + (size_t)i * Q3_K_BLOCK_BYTES;
            const block_q3_K* x = reinterpret_cast<const block_q3_K*>(sb_ptr);

            _mm_prefetch((const char*)((const block_q3_K*)q3_row + i + 1), _MM_HINT_T0);

            // === Q3_K shared decoding (done once for all M) ===
            const float d_half = fp16_to_float_scalar(x->d);

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

            for (int m = 0; m < M; ++m) {
                const block_q8_K* y = q8_t.data() + i * M + m;
                const float d = y->d * d_half;

                const int8_t* q8d = y->qs;
                const uint8_t* q3 = q3_base;
                __m256i sumi = _mm256_setzero_si256();
                int bit = 0;
                int is = 0;

                for (int j = 0; j < QK_K / 128; ++j) {
                    const __m256i q3bits = _mm256_loadu_si256((const __m256i*)q3);
                    q3 += 32;

                    const __m256i q3l_0 = _mm256_and_si256(q3bits, m3);
                    const __m256i q3h_0 = _mm256_slli_epi16(
                        _mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
                    ++bit;

                    const __m256i q3l_1 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 2), m3);
                    const __m256i q3h_1 = _mm256_slli_epi16(
                        _mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
                    ++bit;

                    const __m256i q3l_2 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 4), m3);
                    const __m256i q3h_2 = _mm256_slli_epi16(
                        _mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
                    ++bit;

                    const __m256i q3l_3 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 6), m3);
                    const __m256i q3h_3 = _mm256_slli_epi16(
                        _mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
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

                    p16_0 = _mm256_add_epi32(p16_0, p16_1);
                    p16_2 = _mm256_add_epi32(p16_2, p16_3);
                    sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_2));
                }

                acc[m] = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc[m]);
            }
        }

        for (int m = 0; m < M; ++m) {
            out[m * N + n] = hsum_avx2(acc[m]);
        }
    }
    _mm_free(acc_storage);
}

// ---- Q3_K fused FFN Up (SiLU(gate@w1) * (up@w3)) for decode ----
// Quantizes activation to Q8_K once, then for each output row computes
// gate and up dot products in parallel, sharing the activation read.
static void gemv_q3_k_fused_ffn_up_avx2(const float* a, const uint8_t* w_gate,
                                        const uint8_t* w_up, float* out, int K, int N) {
    constexpr int QK_K = 256;
    constexpr int Q3_K_BLOCK_BYTES = 110;
    const int nb = (K + QK_K - 1) / QK_K;

    std::vector<block_q8_K> q8_buf(nb);
    quantize_row_q8_K(a, q8_buf.data(), K);

    auto silu = [](float x) -> float { return x / (1.0f + std::exp(-x)); };

#    pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        const uint8_t* gate_row = w_gate + (size_t)n * nb * Q3_K_BLOCK_BYTES;
        const uint8_t* up_row = w_up + (size_t)n * nb * Q3_K_BLOCK_BYTES;

        __m256 gate_acc = _mm256_setzero_ps();
        __m256 up_acc = _mm256_setzero_ps();

        for (int i = 0; i < nb; ++i) {
            _mm_prefetch((const char*)(gate_row + (i + 1) * Q3_K_BLOCK_BYTES), _MM_HINT_T0);
            _mm_prefetch((const char*)(up_row + (i + 1) * Q3_K_BLOCK_BYTES), _MM_HINT_T0);
            _mm_prefetch((const char*)(q8_buf.data() + i + 1), _MM_HINT_T0);

            const uint8_t* gate_sb = gate_row + (size_t)i * Q3_K_BLOCK_BYTES;
            const uint8_t* up_sb = up_row + (size_t)i * Q3_K_BLOCK_BYTES;
            gate_acc = _mm256_add_ps(gate_acc, q3_k_sb_dot_avx2(gate_sb, &q8_buf[i]));
            up_acc = _mm256_add_ps(up_acc, q3_k_sb_dot_avx2(up_sb, &q8_buf[i]));
        }

        float gate_val = silu(hsum_avx2(gate_acc));
        float up_val = hsum_avx2(up_acc);
        out[n] = gate_val * up_val;
    }
}

// ---- Q3_K fused QK projection (Q and K from single activation read) ----
// Quantizes activation to Q8_K once, shares across Q and K row dot products.
// Saves 1 quantization (2->1) plus better cache reuse.
static void gemv_q3_k_fused_qk_avx2(const float* a, const uint8_t* wq, const uint8_t* wk,
                                     float* out_q, float* out_k, int K, int N_q, int N_k) {
    constexpr int QK_K = 256;
    constexpr int Q3_K_BLOCK_BYTES = 110;
    const int nb = (K + QK_K - 1) / QK_K;

    std::vector<block_q8_K> q8_buf(nb);
    quantize_row_q8_K(a, q8_buf.data(), K);

    auto dot_rows = [&](const uint8_t* w, float* out, int N) {
#    pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const uint8_t* q3_row = w + (size_t)n * nb * Q3_K_BLOCK_BYTES;
            __m256 acc = _mm256_setzero_ps();
            for (int i = 0; i < nb; ++i) {
                _mm_prefetch((const char*)(q3_row + (i + 1) * Q3_K_BLOCK_BYTES), _MM_HINT_T0);
                _mm_prefetch((const char*)(q8_buf.data() + i + 1), _MM_HINT_T0);
                acc = _mm256_add_ps(acc, q3_k_sb_dot_avx2(q3_row + (size_t)i * Q3_K_BLOCK_BYTES, &q8_buf[i]));
            }
            out[n] = hsum_avx2(acc);
        }
    };

    dot_rows(wq, out_q, N_q);
    dot_rows(wk, out_k, N_k);
}

// ---- Q3_K fused QKV projection (Q, K, V from single activation read) ----
// Quantizes activation to Q8_K once, then for each row in Q, K, V computes
// dot product sharing the same Q8_K block. Saves 2 quantizations (3->1).
static void gemv_q3_k_fused_qkv_avx2(const float* a, const uint8_t* wq, const uint8_t* wk,
                                      const uint8_t* wv, float* out_q, float* out_k, float* out_v,
                                      int K, int N_q, int N_k, int N_v) {
    constexpr int QK_K = 256;
    constexpr int Q3_K_BLOCK_BYTES = 110;
    const int nb = (K + QK_K - 1) / QK_K;

    std::vector<block_q8_K> q8_buf(nb);
    quantize_row_q8_K(a, q8_buf.data(), K);

    auto dot_rows = [&](const uint8_t* w, float* out, int N) {
#    pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const uint8_t* q3_row = w + (size_t)n * nb * Q3_K_BLOCK_BYTES;
            __m256 acc = _mm256_setzero_ps();
            for (int i = 0; i < nb; ++i) {
                _mm_prefetch((const char*)(q3_row + (i + 1) * Q3_K_BLOCK_BYTES), _MM_HINT_T0);
                _mm_prefetch((const char*)(q8_buf.data() + i + 1), _MM_HINT_T0);
                acc = _mm256_add_ps(acc, q3_k_sb_dot_avx2(q3_row + (size_t)i * Q3_K_BLOCK_BYTES, &q8_buf[i]));
            }
            out[n] = hsum_avx2(acc);
        }
    };

    dot_rows(wq, out_q, N_q);
    dot_rows(wk, out_k, N_k);
    dot_rows(wv, out_v, N_v);
}

// ---- Q6_K GEMV (AVX2) using Q8_K fused dot product ----
// Strategy: quantize input to Q8_K once, then use dot_q6_K_q8_K_avx2 for each row.
// This uses _mm256_maddubs_epi16 for int8×int8 fused multiply-add, avoiding FP32 intermediate.
// Canonical implementation (gemv.h also had a copy which has been removed to avoid duplication).

static inline float dot_q6_K_q8_K_avx2(const uint8_t* q6_row, const block_q8_K* q8, int nb) {
    constexpr int QK_K = 256;
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i m15 = _mm256_set1_epi8(15);

    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const block_q6_K* x = reinterpret_cast<const block_q6_K*>(q6_row) + i;
        const block_q8_K* y = q8 + i;

        _mm_prefetch((const char*)((const block_q6_K*)q6_row + i + 1), _MM_HINT_T0);
        _mm_prefetch((const char*)(q8 + i + 1), _MM_HINT_T0);

        const float d = y->d * fp16_to_float_scalar(x->d);

        const uint8_t* q4 = x->ql;
        const uint8_t* qh = x->qh;
        const int8_t* q8d = y->qs;

        const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y->bsums);
        const __m128i scales = _mm_loadu_si128((const __m128i*)x->scales);
        const __m256i scales_16 = _mm256_cvtepi8_epi16(scales);
        const __m256i q8sclsub = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales_16), 5);

        __m256i sumi = _mm256_setzero_si256();
        int is = 0;

        for (int j = 0; j < QK_K / 128; ++j) {
            const __m256i q4bits1 = _mm256_loadu_si256((const __m256i*)q4);
            q4 += 32;
            const __m256i q4bits2 = _mm256_loadu_si256((const __m256i*)q4);
            q4 += 32;
            const __m256i q4bitsH = _mm256_loadu_si256((const __m256i*)qh);
            qh += 32;

            const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m3), 4);
            const __m256i q4h_1 =
                _mm256_slli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(12)), 2);
            const __m256i q4h_2 = _mm256_and_si256(q4bitsH, _mm256_set1_epi8(48));
            const __m256i q4h_3 =
                _mm256_srli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(-64)), 2);

            const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
            const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
            const __m256i q4_2 =
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), q4h_2);
            const __m256i q4_3 =
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), q4h_3);

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8d);
            q8d += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8d);
            q8d += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8d);
            q8d += 32;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8d);
            q8d += 32;

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

    return hsum_avx2(acc);
}

static void gemv_q6_K_transB_avx2(const float* a, const uint8_t* w, float* out, int M, int K,
                                   int N) {
    constexpr int QK_K = 256;
    constexpr int Q6_K_BLOCK_BYTES = 210;
    const int nb = (K + QK_K - 1) / QK_K;

    if (M == 1) {
        std::vector<block_q8_K> q8_buf(nb);
        quantize_row_q8_K(a, q8_buf.data(), K);

#    pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const uint8_t* q6_row = w + (size_t)n * nb * Q6_K_BLOCK_BYTES;
            out[n] = dot_q6_K_q8_K_avx2(q6_row, q8_buf.data(), nb);
        }
    } else {
        std::vector<block_q8_K> q8_all(M * nb);
        for (int m = 0; m < M; ++m) {
            quantize_row_q8_K(a + m * K, q8_all.data() + m * nb, K);
        }

#    pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const uint8_t* q6_row = w + (size_t)n * nb * Q6_K_BLOCK_BYTES;
            for (int m = 0; m < M; ++m) {
                out[m * N + n] = dot_q6_K_q8_K_avx2(q6_row, q8_all.data() + m * nb, nb);
            }
        }
    }
}

// Q6_K Batch-GEMV (transB layout) for M > 1 (prefill).
// Same optimization as Q4_K batch: Q6_K weight decoding (scales, ql/qh bit extraction)
// is done once per weight row, then reused for all M input vectors.
static void gemv_q6_K_transB_batch_avx2(const float* a, const uint8_t* w, float* out,
                                          int M, int K, int N) {
    constexpr int QK_K = 256;
    constexpr int Q6_K_BLOCK_BYTES = 210;
    const int nb = (K + QK_K - 1) / QK_K;

    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i m15 = _mm256_set1_epi8(15);

    // Quantize all M input vectors to Q8_K
    std::vector<block_q8_K> q8_all(M * nb);
    for (int m = 0; m < M; ++m) {
        quantize_row_q8_K(a + m * K, q8_all.data() + m * nb, K);
    }

#    pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        const uint8_t* q6_row = w + (size_t)n * nb * Q6_K_BLOCK_BYTES;

        // Per-M accumulators (use _mm_malloc for 32-byte alignment)
        __m256* acc_vec = static_cast<__m256*>(_mm_malloc(M * sizeof(__m256), 32));
        for (int i = 0; i < M; ++i) acc_vec[i] = _mm256_setzero_ps();

        for (int i = 0; i < nb; ++i) {
            const block_q6_K* x = reinterpret_cast<const block_q6_K*>(q6_row) + i;

            _mm_prefetch((const char*)((const block_q6_K*)q6_row + i + 1), _MM_HINT_T0);

            // === Q6_K shared decoding (done once for all M) ===
            const float d_half = fp16_to_float_scalar(x->d);
            const __m128i scales = _mm_loadu_si128((const __m128i*)x->scales);
            const __m256i scales_16 = _mm256_cvtepi8_epi16(scales);
            const uint8_t* ql_base = x->ql;
            const uint8_t* qh_base = x->qh;

            // === Per-M: load Q8_K and compute dot product ===
            for (int m = 0; m < M; ++m) {
                const block_q8_K* y = q8_all.data() + m * nb + i;

                const float d = y->d * d_half;

                const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y->bsums);
                const __m256i q8sclsub = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales_16), 5);

                const int8_t* q8d = y->qs;
                const uint8_t* q4 = ql_base;
                const uint8_t* qh = qh_base;

                __m256i sumi = _mm256_setzero_si256();
                int is = 0;

                for (int j = 0; j < QK_K / 128; ++j) {
                    const __m256i q4bits1 = _mm256_loadu_si256((const __m256i*)q4);
                    q4 += 32;
                    const __m256i q4bits2 = _mm256_loadu_si256((const __m256i*)q4);
                    q4 += 32;
                    const __m256i q4bitsH = _mm256_loadu_si256((const __m256i*)qh);
                    qh += 32;

                    const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m3), 4);
                    const __m256i q4h_1 =
                        _mm256_slli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(12)), 2);
                    const __m256i q4h_2 = _mm256_and_si256(q4bitsH, _mm256_set1_epi8(48));
                    const __m256i q4h_3 =
                        _mm256_srli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(-64)), 2);

                    const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
                    const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
                    const __m256i q4_2 =
                        _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), q4h_2);
                    const __m256i q4_3 =
                        _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), q4h_3);

                    const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8d);
                    q8d += 32;
                    const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8d);
                    q8d += 32;
                    const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8d);
                    q8d += 32;
                    const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8d);
                    q8d += 32;

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
                acc_vec[m] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc_vec[m]);
            }
        }

        for (int m = 0; m < M; ++m) {
            out[m * N + n] = hsum_avx2(acc_vec[m]);
        }
        _mm_free(acc_vec);
    }
}

// Q4_K Batch-GEMV (transB layout) for M > 1 (prefill).
// Q4_K weight decoding (scales, qs bit extraction) is done once per weight row,
// then reused for all M input vectors — saving (M-1)/M of the weight decode cost.
static void gemv_q4_K_transB_batch_avx2(const float* a, const uint8_t* w, float* out,
                                         int M, int K, int N) {
    constexpr int QK_K = 256;
    constexpr int Q4_K_BLOCK_BYTES = 144;
    const int nb = (K + QK_K - 1) / QK_K;

    const __m256i m4 = _mm256_set1_epi8(0xF);
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    // Quantize all M input vectors to Q8_K
    std::vector<block_q8_K> q8_all(M * nb);
    for (int m = 0; m < M; ++m) {
        quantize_row_q8_K(a + m * K, q8_all.data() + m * nb, K);
    }

#    pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        const uint8_t* q4_row = w + (size_t)n * nb * Q4_K_BLOCK_BYTES;

        // Per-M accumulators (use _mm_malloc for proper alignment)
        __m256* acc_vec = static_cast<__m256*>(_mm_malloc(M * sizeof(__m256), 32));
        __m128* acc_m_vec = static_cast<__m128*>(_mm_malloc(M * sizeof(__m128), 16));
        for (int i = 0; i < M; ++i) {
            acc_vec[i] = _mm256_setzero_ps();
            acc_m_vec[i] = _mm_setzero_ps();
        }

        for (int i = 0; i < nb; ++i) {
            const block_q4_K* x = reinterpret_cast<const block_q4_K*>(q4_row) + i;

            _mm_prefetch((const char*)((const block_q4_K*)q4_row + i + 1), _MM_HINT_T0);

            // === Q4_K shared decoding (done once for all M) ===
            const float d_half = fp16_to_float_scalar(x->d);
            const float dmin_half = fp16_to_float_scalar(x->dmin);

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

            // Duplicate scales across both 128-bit lanes
            const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
            const __m256i scales = _mm256_set_m128i(sc128, sc128);
            const __m128i mins128 = _mm256_extracti128_si256(mins_and_scales, 1);

            // Pre-load Q4_K qs bits (shared across all M)
            const uint8_t* q4_base = x->qs;

            // === Per-M: load Q8_K and compute dot product ===
            for (int m = 0; m < M; ++m) {
                const block_q8_K* y = q8_all.data() + m * nb + i;

                const float d = y->d * d_half;
                const float dmin = -y->d * dmin_half;

                // Min contribution: mins * bsums
                const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y->bsums);
                const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0),
                                                      _mm256_extracti128_si256(q8sums, 1));
                const __m128i prod = _mm_madd_epi16(mins128, q8s);
                acc_m_vec[m] = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m_vec[m]);

                // Dot product: Q4_K qs (shared) x Q8_K qs (per-m)
                const int8_t* q8d = y->qs;
                __m256i sumi = _mm256_setzero_si256();
                const uint8_t* q4 = q4_base;

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

                acc_vec[m] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc_vec[m]);
            }
        }

        // Horizontal sum for each m
        for (int m = 0; m < M; ++m) {
            __m128 acc_m = acc_m_vec[m];
            acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
            acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));
            out[m * N + n] = hsum_avx2(acc_vec[m]) + _mm_cvtss_f32(acc_m);
        }
        _mm_free(acc_vec);
        _mm_free(acc_m_vec);
    }
}

#endif  // USE_AVX2

// ---- Q2_K × Q8_K fused dot product (llama.cpp AVX2 impl) ----
// Q2_K block: 84 bytes, 256 elements, 2-bit quants with d+dmin
// scales[16]: lo nibble = d_scale, hi nibble = min
#ifdef USE_AVX2

static inline float dot_q2_K_q8_K_avx2(const uint8_t* q2_row, const block_q8_K* q8, int nb,
                                        const uint8_t* scales_row) {
    constexpr int QK_K = 256;
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m128i m4 = _mm_set1_epi8(0xF);
    (void)scales_row;
    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const block_q2_K* x = reinterpret_cast<const block_q2_K*>(q2_row) + i;
        const block_q8_K* y = q8 + i;

        _mm_prefetch((const char*)((const block_q2_K*)q2_row + i + 1), _MM_HINT_T0);
        _mm_prefetch((const char*)(q8 + i + 1), _MM_HINT_T0);

        const float d = y->d * fp16_to_float_scalar(x->d);
        const float dmin = -y->d * fp16_to_float_scalar(x->dmin);

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
        const __m256i scales[2] = {_mm256_set_m128i(l_scales, l_scales), _mm256_set_m128i(h_scales, h_scales)};

        __m256i sumi = _mm256_setzero_si256();

        for (int j = 0; j < QK_K/128; ++j) {
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

            p0 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(0)), p0);
            p1 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(1)), p1);
            p2 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(2)), p2);
            p3 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(3)), p3);

            p0 = _mm256_add_epi32(p0, p1);
            p2 = _mm256_add_epi32(p2, p3);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p0, p2));
        }

        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }

    return hsum_avx2(acc);
}

// Q2_K GEMV: quantize input to Q8_K once, then dot per row
static void gemv_q2_k_transB_avx2(const float* a, const uint8_t* w, float* out, int M, int K,
                                   int N) {
    constexpr int QK_K = 256;
    constexpr int Q2_K_BLOCK_BYTES = 84;
    const int nb = (K + QK_K - 1) / QK_K;

    if (M == 1) {
        std::vector<block_q8_K> q8_buf(nb);
        quantize_row_q8_K(a, q8_buf.data(), K);

#pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const uint8_t* q2_row = w + (size_t)n * nb * Q2_K_BLOCK_BYTES;
            out[n] = dot_q2_K_q8_K_avx2(q2_row, q8_buf.data(), nb, nullptr);
        }
    } else {
        std::vector<block_q8_K> q8_all(M * nb);
        for (int m = 0; m < M; ++m) {
            quantize_row_q8_K(a + m * K, q8_all.data() + m * nb, K);
        }

#pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const uint8_t* q2_row = w + (size_t)n * nb * Q2_K_BLOCK_BYTES;
            for (int m = 0; m < M; ++m) {
                out[m * N + n] = dot_q2_K_q8_K_avx2(q2_row, q8_all.data() + m * nb, nb, nullptr);
            }
        }
    }
}
// Q2_K Batch-GEMV (transB layout) for M > 1 (prefill).
// Q2_K weight decoding (scales, qs bit extraction) is done once per weight block,
// then reused for all M input vectors.
// Q8_K data is transposed to [block_idx * M + row_idx] for sequential access.
static void gemv_q2_k_transB_batch_avx2(const float* a, const uint8_t* w, float* out,
                                          int M, int K, int N) {
    constexpr int QK_K = 256;
    constexpr int Q2_K_BLOCK_BYTES = 84;
    const int nb = (K + QK_K - 1) / QK_K;

    const __m256i m3 = _mm256_set1_epi8(3);
    const __m128i m4 = _mm_set1_epi8(0xF);

    // Quantize all M input vectors to Q8_K
    // Store as [m][i] layout
    std::vector<block_q8_K> q8_src(M * nb);
    for (int m = 0; m < M; ++m) {
        quantize_row_q8_K(a + m * K, q8_src.data() + m * nb, K);
    }

    // Transpose Q8_K data: q8_t[i * M + m] = q8_src[m * nb + i]
    // This makes per-block access sequential for all M rows
    std::vector<block_q8_K> q8_t(nb * M);
    for (int i = 0; i < nb; ++i) {
        for (int m = 0; m < M; ++m) {
            q8_t[i * M + m] = q8_src[m * nb + i];
        }
    }

    // Pre-allocate aligned accumulators per OpenMP thread
    int max_threads = omp_get_max_threads();
    __m256* acc_storage = static_cast<__m256*>(_mm_malloc(max_threads * M * sizeof(__m256), 32));

#pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        int tid = omp_get_thread_num();
        __m256* acc = acc_storage + (size_t)tid * M;
        for (int m = 0; m < M; ++m) acc[m] = _mm256_setzero_ps();

        const uint8_t* q2_row = w + (size_t)n * nb * Q2_K_BLOCK_BYTES;

        for (int i = 0; i < nb; ++i) {
            const block_q2_K* x = reinterpret_cast<const block_q2_K*>(q2_row) + i;

            _mm_prefetch((const char*)((const block_q2_K*)q2_row + i + 1), _MM_HINT_T0);

            // === Q2_K shared decoding (done once for all M) ===
            const float d_half = fp16_to_float_scalar(x->d);
            const float dmin_half = fp16_to_float_scalar(x->dmin);

            const __m128i mins_and_scales = _mm_loadu_si128((const __m128i*)x->scales);
            const __m128i scales8 = _mm_and_si128(mins_and_scales, m4);
            const __m128i mins8 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);

            const __m256i mins = _mm256_cvtepi8_epi16(mins8);
            const __m256i all_scales = _mm256_cvtepi8_epi16(scales8);
            const __m128i l_scales = _mm256_extracti128_si256(all_scales, 0);
            const __m128i h_scales = _mm256_extracti128_si256(all_scales, 1);
            const __m256i scales[2] = {
                _mm256_set_m128i(l_scales, l_scales),
                _mm256_set_m128i(h_scales, h_scales)
            };

            const uint8_t* q2_base = x->qs;

            // === Per-M: load Q8_K (now sequential) and compute dot product ===
            for (int m = 0; m < M; ++m) {
                const block_q8_K* y = q8_t.data() + i * M + m;

                const float d = y->d * d_half;
                const float dmin = -y->d * dmin_half;

                // Min contribution: mins * bsums
                const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y->bsums);
                const __m256i prod = _mm256_madd_epi16(mins, q8sums);
                acc[m] = _mm256_fmadd_ps(_mm256_broadcast_ss(&dmin), _mm256_cvtepi32_ps(prod), acc[m]);

                // Dot product: Q2_K qs (shared) × Q8_K qs (per-m)
                const int8_t* q8d = y->qs;
                __m256i sumi = _mm256_setzero_si256();
                const uint8_t* q2 = q2_base;

                for (int j = 0; j < QK_K/128; ++j) {
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

                    p0 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(0)), p0);
                    p1 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(1)), p1);
                    p2 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(2)), p2);
                    p3 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(3)), p3);

                    p0 = _mm256_add_epi32(p0, p1);
                    p2 = _mm256_add_epi32(p2, p3);
                    sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p0, p2));
                }

                acc[m] = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc[m]);
            }
        }

        for (int m = 0; m < M; ++m) {
            out[m * N + n] = hsum_avx2(acc[m]);
        }
    }
    _mm_free(acc_storage);
}
#endif  // USE_AVX2

// ---- Q4_K × Q8_K fused dot product ----
// Computes dot(F32(Q4_K row), Q8_K activation) using integer SIMD.
// Ported from gemv.h for use by the Q8_K-based FFN up kernel below.
#ifdef USE_AVX2
static inline float dot_q4_K_q8_K_avx2(const uint8_t* q4_row, const block_q8_K* q8, int nb) {
    constexpr int QK_K = 256;
    const __m256i m4 = _mm256_set1_epi8(0xF);
    __m256 acc = _mm256_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    for (int i = 0; i < nb; ++i) {
        const block_q4_K* x = reinterpret_cast<const block_q4_K*>(q4_row) + i;
        const block_q8_K* y = q8 + i;

        _mm_prefetch((const char*)((const block_q4_K*)q4_row + i + 1), _MM_HINT_T0);
        _mm_prefetch((const char*)(q8 + i + 1), _MM_HINT_T0);

        const float d = y->d * fp16_to_float_scalar(x->d);
        const float dmin = -y->d * fp16_to_float_scalar(x->dmin);

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

    return hsum_avx2(acc) + _mm_cvtss_f32(acc_m);
}

// ---- Q4_K GEMV using Q8_K fused dot product (replaces FP32-domain fused GEMV) ----
// Strategy: quantize input to Q8_K once, then use dot_q4_K_q8_K_avx2 for each row.
// Uses _mm256_maddubs_epi16 for int8×int8 fused multiply-add, avoiding FP32 intermediate.
// For decode (M=1): quantize once, parallel over N rows.
// For prefill (M>1): quantize each row, parallel over N.

static void gemv_q4_k_q8k_transB_avx2(const float* a, const uint8_t* w, float* out, int M, int K,
                                        int N) {
    constexpr int QK_K = 256;
    constexpr int Q4_K_BLOCK_BYTES = 144;
    const int nb = (K + QK_K - 1) / QK_K;

    if (M == 1) {
        std::vector<block_q8_K> q8_buf(nb);
        quantize_row_q8_K(a, q8_buf.data(), K);

#    pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const uint8_t* q4_row = w + (size_t)n * nb * Q4_K_BLOCK_BYTES;
            out[n] = dot_q4_K_q8_K_avx2(q4_row, q8_buf.data(), nb);
        }
    } else {
        std::vector<block_q8_K> q8_all(M * nb);
        for (int m = 0; m < M; ++m) {
            quantize_row_q8_K(a + m * K, q8_all.data() + m * nb, K);
        }

#    pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const uint8_t* q4_row = w + (size_t)n * nb * Q4_K_BLOCK_BYTES;
            for (int m = 0; m < M; ++m) {
                out[m * N + n] = dot_q4_K_q8_K_avx2(q4_row, q8_all.data() + m * nb, nb);
            }
        }
    }
}

// ---- Q4_K fused FFN gate+up ----
// Reads input vector once, computes both gate and up projections simultaneously.
// Then applies SiLU(gate) * up inline.
// Saves 1x input vector read and avoids intermediate gate/up tensors.

static void gemv_q4_k_fused_ffn_up_avx2(const float* a, const uint8_t* w_gate, const uint8_t* w_up,
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

// ---- Q4_K fused QKV ----
// Reads input vector once, computes Q, K, V outputs simultaneously.
// Shares the input activation across all three projections.
// Each output column processes Q4_K super-blocks independently.

#ifdef USE_AVX2
static void gemv_q4_K_fused_qkv_avx2(const float* a, const uint8_t* wq, const uint8_t* wk,
                                     const uint8_t* wv, float* out_q, float* out_k, float* out_v,
                                     int K, int N_q, int N_k, int N_v) {
    constexpr int SB_SIZE = 256;
    constexpr int SB_BYTES = 144;
    const int sbs_per_row = (K + SB_SIZE - 1) / SB_SIZE;
    const int full_sbs = K / SB_SIZE;

    auto process_row = [&](const uint8_t* w_row, float* out, int N_out) {
#    pragma omp parallel for schedule(static)
        for (int n = 0; n < N_out; ++n) {
            const uint8_t* sb_row = w_row + (size_t)n * sbs_per_row * SB_BYTES;

            __m256 acc_sc = _mm256_setzero_ps();
            __m256 acc_mn = _mm256_setzero_ps();

            for (int bi = 0; bi < full_sbs; ++bi) {
                const uint8_t* sb = sb_row + (size_t)bi * SB_BYTES;
                uint16_t d_bits, dm_bits;
                memcpy(&d_bits, sb, 2);
                memcpy(&dm_bits, sb + 2, 2);
                __m256 d_v = fp16_to_fp32_broadcast_avx2(d_bits);
                __m256 dm_v = fp16_to_fp32_broadcast_avx2(dm_bits);
                const uint8_t* scales = sb + 4;
                const uint8_t* qs = sb + 16;
                int is = 0;
                for (int j = 0; j < 4; ++j) {
                    int sub_base = bi * SB_SIZE + j * 64;
                    float sc_e, m_e, sc_o, m_o;
                    q4_k_get_scale_min(is, scales, sc_e, m_e);
                    q4_k_get_scale_min(is + 1, scales, sc_o, m_o);
                    q4_k_subblock_dot_avx2(a, sub_base, qs, d_v, dm_v, sc_e, m_e, sc_o, m_o, acc_sc,
                                           acc_mn);
                    qs += 32;
                    is += 2;
                }
            }

            out[n] = hsum_avx2(_mm256_sub_ps(acc_sc, acc_mn));
        }
    };

    process_row(wq, out_q, N_q);
    process_row(wk, out_k, N_k);
    process_row(wv, out_v, N_v);
}
#endif  // USE_AVX2

// ---- Q6_K fused FFN down + residual ----
// Q6_K super-block: 210 bytes per 256 elements
// Layout: [128B ql] [64B qh] [16B scales(int8)] [2B d(fp16)]
// Dequant: val = d * sc * ((ql_nibble | (qh_bits << 4)) - 32)
//
// Fused approach:
//   dot = sum(a[k] * d * sc * q6[k])
//       = d * sc * sum(a[k] * q6[k])
// Accumulate into acc, multiply by d*scale per 16-element group.
// Saves the intermediate FP32 row buffer (~63% memory traffic vs dequant+dot).

#ifdef USE_AVX2
// Helper: compute dot product of 16 Q6_K weight values with 16 activation values.
static inline void q6_k_dot_16_avx2(const float* a_row, int base, const uint8_t* ql_ptr,
                                    const uint8_t* qh_ptr, int qh_shift, bool use_low_nibble,
                                    float d_scale, __m256& acc) {
    __m128i ql16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ql_ptr));
    __m128i nibbles;
    if (use_low_nibble) {
        nibbles = _mm_and_si128(ql16, _mm_set1_epi8(0x0F));
    } else {
        nibbles = _mm_and_si128(_mm_srli_epi16(ql16, 4), _mm_set1_epi8(0x0F));
    }
    __m128i qh16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qh_ptr));
    __m128i qh_bits = _mm_and_si128(_mm_srli_epi16(qh16, qh_shift), _mm_set1_epi8(0x03));
    __m128i q_vals = _mm_or_si128(nibbles, _mm_slli_epi16(qh_bits, 4));
    q_vals = _mm_sub_epi8(q_vals, _mm_set1_epi8(32));

    __m256 dsc = _mm256_set1_ps(d_scale);
    __m256 a0 = _mm256_loadu_ps(a_row + base);
    __m256 q0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_vals));
    __m256 a0s = _mm256_mul_ps(a0, dsc);
    acc = _mm256_fmadd_ps(a0s, q0, acc);

    __m256 a1 = _mm256_loadu_ps(a_row + base + 8);
    __m256 q1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_vals, 8)));
    __m256 a1s = _mm256_mul_ps(a1, dsc);
    acc = _mm256_fmadd_ps(a1s, q1, acc);
}

// Process one Q6_K super-block: 256 elements, produces dot product contributions.
static inline void q6_k_process_sb_avx2(const float* a_row, int k_offset, const uint8_t* sb,
                                        __m256& acc) {
    uint16_t d_bits;
    memcpy(&d_bits, sb + 208, 2);
    float d_val = fp16_to_float_scalar(d_bits);

    const uint8_t* ql = sb;
    const uint8_t* qh = sb + 128;
    const int8_t* sc = reinterpret_cast<const int8_t*>(sb + 192);

    for (int sub = 0; sub < 2; ++sub) {
        const uint8_t* ql_cur = ql + sub * 64;
        const uint8_t* qh_cur = qh + sub * 32;
        const int8_t* sc_cur = sc + sub * 8;
        int base = k_offset + sub * 128;

        // Pair 1: q1[0..15] (ql low, qh bits[1:0], sc[0]) + q3[0..15] (ql high, qh bits[5:4],
        // sc[4])
        q6_k_dot_16_avx2(a_row, base, ql_cur, qh_cur, 0, true, d_val * sc_cur[0], acc);
        q6_k_dot_16_avx2(a_row, base + 64, ql_cur, qh_cur, 4, false, d_val * sc_cur[4], acc);
        // Pair 2: q2[0..15] (ql+32 low, qh bits[3:2], sc[2]) + q4[0..15] (ql+32 high, qh bits[7:6],
        // sc[6])
        q6_k_dot_16_avx2(a_row, base + 32, ql_cur + 32, qh_cur, 2, true, d_val * sc_cur[2], acc);
        q6_k_dot_16_avx2(a_row, base + 96, ql_cur + 32, qh_cur, 6, false, d_val * sc_cur[6], acc);
        // Pair 3: q1[16..31] (ql+16 low, qh+16 bits[1:0], sc[1])
        q6_k_dot_16_avx2(a_row, base + 16, ql_cur + 16, qh_cur + 16, 0, true, d_val * sc_cur[1],
                         acc);
        // Pair 4: q3[16..31] (ql+16 high, qh+16 bits[5:4], sc[5])
        q6_k_dot_16_avx2(a_row, base + 80, ql_cur + 16, qh_cur + 16, 4, false, d_val * sc_cur[5],
                         acc);
        // Pair 5: q2[16..31] (ql+48 low, qh+16 bits[3:2], sc[3])
        q6_k_dot_16_avx2(a_row, base + 48, ql_cur + 48, qh_cur + 16, 2, true, d_val * sc_cur[3],
                         acc);
        // Pair 6: q4[16..31] (ql+48 high, qh+16 bits[7:6], sc[7])
        q6_k_dot_16_avx2(a_row, base + 112, ql_cur + 48, qh_cur + 16, 6, false, d_val * sc_cur[7],
                         acc);
    }
}

static void gemv_q6_k_ffn_down_residual_avx2(const float* a, const uint8_t* w,
                                             const float* residual, float* out, int K, int N) {
    constexpr int QK_K = 256;
    constexpr int Q6_K_BLOCK_BYTES = 210;
    const int nb = (K + QK_K - 1) / QK_K;

    std::vector<block_q8_K> q8_buf(nb);
    quantize_row_q8_K(a, q8_buf.data(), K);

#    pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        const uint8_t* w_row = w + (size_t)n * nb * Q6_K_BLOCK_BYTES;
        out[n] = dot_q6_K_q8_K_avx2(w_row, q8_buf.data(), nb) + residual[n];
    }
}

// ---- Q4_K fused FFN down + residual (decode, M=1, Q8_K activation) ----
// Quantizes activation to Q8_K once, then uses integer SIMD dot product per row.
static void gemv_q4_k_ffn_down_residual_avx2(const float* a, const uint8_t* w,
                                             const float* residual, float* out, int K, int N) {
    constexpr int QK_K = 256;
    constexpr int Q4_K_BLOCK_BYTES = 144;
    const int nb = (K + QK_K - 1) / QK_K;

    std::vector<block_q8_K> q8_buf(nb);
    quantize_row_q8_K(a, q8_buf.data(), K);

#    pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        const uint8_t* w_row = w + (size_t)n * nb * Q4_K_BLOCK_BYTES;
        out[n] = dot_q4_K_q8_K_avx2(w_row, q8_buf.data(), nb) + residual[n];
    }
}

// ---- Q2_K fused FFN down + residual (decode, M=1, Q8_K activation) ----
static void gemv_q2_k_ffn_down_residual_avx2(const float* a, const uint8_t* w,
                                             const float* residual, float* out, int K, int N) {
    constexpr int QK_K = 256;
    constexpr int Q2_K_BLOCK_BYTES = 84;
    const int nb = (K + QK_K - 1) / QK_K;

    std::vector<block_q8_K> q8_buf(nb);
    quantize_row_q8_K(a, q8_buf.data(), K);

#    pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        const uint8_t* w_row = w + (size_t)n * nb * Q2_K_BLOCK_BYTES;
        out[n] = dot_q2_K_q8_K_avx2(w_row, q8_buf.data(), nb, nullptr) + residual[n];
    }
}

// ---- Q3_K fused FFN down + residual (decode, M=1, Q8_K activation) ----
static void gemv_q3_k_ffn_down_residual_avx2(const float* a, const uint8_t* w,
                                             const float* residual, float* out, int K, int N) {
    constexpr int QK_K = 256;
    constexpr int Q3_K_BLOCK_BYTES = 110;
    const int nb = (K + QK_K - 1) / QK_K;

    std::vector<block_q8_K> q8_buf(nb);
    quantize_row_q8_K(a, q8_buf.data(), K);

#    pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        const uint8_t* w_row = w + (size_t)n * nb * Q3_K_BLOCK_BYTES;
        out[n] = dot_q3_K_q8_K_avx2(w_row, q8_buf.data(), nb) + residual[n];
    }
}

// ---- Q4_K fused attention output projection + residual (decode, M=1, Q8_K activation) ----
// Computes attn_out @ wo + hidden_residual in a single pass.
static void gemv_q4_k_attn_proj_residual_avx2(const float* a, const uint8_t* w,
                                               const float* residual, float* out, int K, int N) {
    constexpr int QK_K = 256;
    constexpr int Q4_K_BLOCK_BYTES = 144;
    const int nb = (K + QK_K - 1) / QK_K;

    std::vector<block_q8_K> q8_buf(nb);
    quantize_row_q8_K(a, q8_buf.data(), K);

#    pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        const uint8_t* w_row = w + (size_t)n * nb * Q4_K_BLOCK_BYTES;
        out[n] = dot_q4_K_q8_K_avx2(w_row, q8_buf.data(), nb) + residual[n];
    }
}

// ---- Q6_K fused attention output projection + residual (decode, M=1, Q8_K activation) ----
static void gemv_q6_k_attn_proj_residual_avx2(const float* a, const uint8_t* w,
                                               const float* residual, float* out, int K, int N) {
    constexpr int QK_K = 256;
    constexpr int Q6_K_BLOCK_BYTES = 210;
    const int nb = (K + QK_K - 1) / QK_K;

    std::vector<block_q8_K> q8_buf(nb);
    quantize_row_q8_K(a, q8_buf.data(), K);

#    pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        const uint8_t* w_row = w + (size_t)n * nb * Q6_K_BLOCK_BYTES;
        out[n] = dot_q6_K_q8_K_avx2(w_row, q8_buf.data(), nb) + residual[n];
    }
}

// ---- Q2_K fused attention output projection + residual (decode, M=1, Q8_K activation) ----
static void gemv_q2_k_attn_proj_residual_avx2(const float* a, const uint8_t* w,
                                               const float* residual, float* out, int K, int N) {
    constexpr int QK_K = 256;
    constexpr int Q2_K_BLOCK_BYTES = 84;
    const int nb = (K + QK_K - 1) / QK_K;

    std::vector<block_q8_K> q8_buf(nb);
    quantize_row_q8_K(a, q8_buf.data(), K);

#    pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        const uint8_t* w_row = w + (size_t)n * nb * Q2_K_BLOCK_BYTES;
        out[n] = dot_q2_K_q8_K_avx2(w_row, q8_buf.data(), nb, nullptr) + residual[n];
    }
}

// ---- Q3_K fused attention output projection + residual (decode, M=1, Q8_K activation) ----
static void gemv_q3_k_attn_proj_residual_avx2(const float* a, const uint8_t* w,
                                               const float* residual, float* out, int K, int N) {
    constexpr int QK_K = 256;
    constexpr int Q3_K_BLOCK_BYTES = 110;
    const int nb = (K + QK_K - 1) / QK_K;

    std::vector<block_q8_K> q8_buf(nb);
    quantize_row_q8_K(a, q8_buf.data(), K);

#    pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        const uint8_t* w_row = w + (size_t)n * nb * Q3_K_BLOCK_BYTES;
        out[n] = dot_q3_K_q8_K_avx2(w_row, q8_buf.data(), nb) + residual[n];
    }
}
#endif  // USE_AVX2

// ============================================================================
// Phase 7: maddubs-based GEMV (quantize activation to Q8_0_act, then int8 dot)
// Replaces the FP32 dequantization path for Q4_0/Q4_1/Q8_0/Q5_0/Q5_1.
// ============================================================================

#ifdef USE_AVX2

// ---- Q4_0 maddubs GEMV ----
// Key insight: quantize activation to Q8_0_act once (O(K)),
// then reuse for all N weight rows with maddubs int8 dot products.
static void gemv_q4_0_maddubs_transB_avx2(const float* a, const uint8_t* w, float* out,
                                            int M, int K, int N) {
    constexpr int BLOCK_BYTES = 18;
    constexpr int nrc = 4;
    const int nb = (K + 31) / 32;

    if (M == 1) {
        std::vector<block_q8_0_act> q8_act(nb);
        quantize_row_q8_0_act(a, q8_act.data(), K);

#    pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            _mm_prefetch((const char*)(w + (size_t)(n + 4) * nb * BLOCK_BYTES), _MM_HINT_T1);
            const uint8_t* w_row = w + (size_t)n * nb * BLOCK_BYTES;
            out[n] = vec_dot_q4_0_q8_0_avx2(w_row, q8_act.data(), nb);
        }
    } else {
        for (int m_start = 0; m_start < M; m_start += nrc) {
            int m_cur = (m_start + nrc <= M) ? nrc : (M - m_start);
            std::vector<block_q8_0_act> q8_tile(m_cur * nb);
            for (int m = 0; m < m_cur; ++m)
                quantize_row_q8_0_act(a + (m_start + m) * K, q8_tile.data() + m * nb, K);

#    pragma omp parallel for schedule(static)
            for (int n = 0; n < N; ++n) {
                const uint8_t* w_row = w + (size_t)n * nb * BLOCK_BYTES;
                for (int m = 0; m < m_cur; ++m)
                    out[(m_start + m) * N + n] = vec_dot_q4_0_q8_0_avx2(w_row, q8_tile.data() + m * nb, nb);
            }
        }
    }
}

// ---- Q4_1 maddubs GEMV ----
static void gemv_q4_1_maddubs_transB_avx2(const float* a, const uint8_t* w, float* out,
                                            int M, int K, int N) {
    constexpr int BLOCK_BYTES = 20;
    constexpr int nrc = 4;
    const int nb = (K + 31) / 32;

    if (M == 1) {
        std::vector<block_q8_0_act> q8_act(nb);
        quantize_row_q8_0_act(a, q8_act.data(), K);

#    pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            _mm_prefetch((const char*)(w + (size_t)(n + 4) * nb * BLOCK_BYTES), _MM_HINT_T1);
            const uint8_t* w_row = w + (size_t)n * nb * BLOCK_BYTES;
            out[n] = vec_dot_q4_1_q8_0_avx2(w_row, q8_act.data(), nb);
        }
    } else {
        for (int m_start = 0; m_start < M; m_start += nrc) {
            int m_cur = (m_start + nrc <= M) ? nrc : (M - m_start);
            std::vector<block_q8_0_act> q8_tile(m_cur * nb);
            for (int m = 0; m < m_cur; ++m)
                quantize_row_q8_0_act(a + (m_start + m) * K, q8_tile.data() + m * nb, K);

#    pragma omp parallel for schedule(static)
            for (int n = 0; n < N; ++n) {
                const uint8_t* w_row = w + (size_t)n * nb * BLOCK_BYTES;
                for (int m = 0; m < m_cur; ++m)
                    out[(m_start + m) * N + n] = vec_dot_q4_1_q8_0_avx2(w_row, q8_tile.data() + m * nb, nb);
            }
        }
    }
}

// ---- Q8_0 maddubs GEMV ----
// Q8_0 × Q8_0 uses offset trick: add 128 to weight bytes for unsigned × signed maddubs
static void gemv_q8_0_maddubs_transB_avx2(const float* a, const uint8_t* w, float* out,
                                            int M, int K, int N) {
    constexpr int BLOCK_BYTES = 34;
    constexpr int nrc = 4;
    const int nb = (K + 31) / 32;

    if (M == 1) {
        std::vector<block_q8_0_act> q8_act(nb);
        quantize_row_q8_0_act(a, q8_act.data(), K);

#    pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            _mm_prefetch((const char*)(w + (size_t)(n + 4) * nb * BLOCK_BYTES), _MM_HINT_T1);
            const uint8_t* w_row = w + (size_t)n * nb * BLOCK_BYTES;
            out[n] = vec_dot_q8_0_q8_0_avx2(w_row, q8_act.data(), nb);
        }
    } else {
        for (int m_start = 0; m_start < M; m_start += nrc) {
            int m_cur = (m_start + nrc <= M) ? nrc : (M - m_start);
            std::vector<block_q8_0_act> q8_tile(m_cur * nb);
            for (int m = 0; m < m_cur; ++m)
                quantize_row_q8_0_act(a + (m_start + m) * K, q8_tile.data() + m * nb, K);

#    pragma omp parallel for schedule(static)
            for (int n = 0; n < N; ++n) {
                const uint8_t* w_row = w + (size_t)n * nb * BLOCK_BYTES;
                for (int m = 0; m < m_cur; ++m)
                    out[(m_start + m) * N + n] = vec_dot_q8_0_q8_0_avx2(w_row, q8_tile.data() + m * nb, nb);
            }
        }
    }
}

// ---- Q5_0 maddubs GEMV ----
static void gemv_q5_0_maddubs_transB_avx2(const float* a, const uint8_t* w, float* out,
                                            int M, int K, int N) {
    constexpr int BLOCK_BYTES = 22;
    constexpr int nrc = 4;
    const int nb = (K + 31) / 32;

    if (M == 1) {
        std::vector<block_q8_0_act> q8_act(nb);
        quantize_row_q8_0_act(a, q8_act.data(), K);

#    pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            _mm_prefetch((const char*)(w + (size_t)(n + 4) * nb * BLOCK_BYTES), _MM_HINT_T1);
            const uint8_t* w_row = w + (size_t)n * nb * BLOCK_BYTES;
            out[n] = vec_dot_q5_0_q8_0_avx2(w_row, q8_act.data(), nb);
        }
    } else {
        for (int m_start = 0; m_start < M; m_start += nrc) {
            int m_cur = (m_start + nrc <= M) ? nrc : (M - m_start);
            std::vector<block_q8_0_act> q8_tile(m_cur * nb);
            for (int m = 0; m < m_cur; ++m)
                quantize_row_q8_0_act(a + (m_start + m) * K, q8_tile.data() + m * nb, K);

#    pragma omp parallel for schedule(static)
            for (int n = 0; n < N; ++n) {
                const uint8_t* w_row = w + (size_t)n * nb * BLOCK_BYTES;
                for (int m = 0; m < m_cur; ++m)
                    out[(m_start + m) * N + n] = vec_dot_q5_0_q8_0_avx2(w_row, q8_tile.data() + m * nb, nb);
            }
        }
    }
}

// ---- Q5_1 maddubs GEMV ----
static void gemv_q5_1_maddubs_transB_avx2(const float* a, const uint8_t* w, float* out,
                                            int M, int K, int N) {
    constexpr int BLOCK_BYTES = 24;
    constexpr int nrc = 4;
    const int nb = (K + 31) / 32;

    if (M == 1) {
        std::vector<block_q8_0_act> q8_act(nb);
        quantize_row_q8_0_act(a, q8_act.data(), K);

#    pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            _mm_prefetch((const char*)(w + (size_t)(n + 4) * nb * BLOCK_BYTES), _MM_HINT_T1);
            const uint8_t* w_row = w + (size_t)n * nb * BLOCK_BYTES;
            out[n] = vec_dot_q5_1_q8_0_avx2(w_row, q8_act.data(), nb);
        }
    } else {
        for (int m_start = 0; m_start < M; m_start += nrc) {
            int m_cur = (m_start + nrc <= M) ? nrc : (M - m_start);
            std::vector<block_q8_0_act> q8_tile(m_cur * nb);
            for (int m = 0; m < m_cur; ++m)
                quantize_row_q8_0_act(a + (m_start + m) * K, q8_tile.data() + m * nb, K);

#    pragma omp parallel for schedule(static)
            for (int n = 0; n < N; ++n) {
                const uint8_t* w_row = w + (size_t)n * nb * BLOCK_BYTES;
                for (int m = 0; m < m_cur; ++m)
                    out[(m_start + m) * N + n] = vec_dot_q5_1_q8_0_avx2(w_row, q8_tile.data() + m * nb, nb);
            }
        }
    }
}

// ---- Q5_K maddubs GEMV ----
// Q5_K uses Q8_K activation (same as Q4_K pattern) with dot_q5_K_q8_K
// Block layout: d[2] + dmin[2] + scales[12] + qh[32] + ql[128] = 176 bytes / 256 elements
static inline float dot_q5_K_q8_K_avx2(const uint8_t* q5_row, const block_q8_K* q8, int nb) {
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
        const block_q5_K* x = reinterpret_cast<const block_q5_K*>(q5_row) + i;
        const block_q8_K* y = q8 + i;

        _mm_prefetch((const char*)((const block_q5_K*)q5_row + i + 1), _MM_HINT_T0);

        const uint8_t* q5 = x->ql;
        const int8_t*  q8d = y->qs;

        const float d = y->d * fp16_to_float_scalar(x->d);
        const float dmin = -y->d * fp16_to_float_scalar(x->dmin);

        // Decode scales (same encoding as Q4_K)
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

        // Q5_K: load all 32 bytes of qh at once, use hmask+bit shifting
        const __m256i hbits = _mm256_loadu_si256((const __m256i*)x->qh);
        __m256i hmask = mone;

        __m256i sumi = _mm256_setzero_si256();
        int bit = 0;

        for (int j = 0; j < QK_K / 64; ++j) {
            const __m256i scale_0 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
            const __m256i scale_1 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));

            // Load 32 bytes of ql
            const __m256i q5bits = _mm256_loadu_si256((const __m256i*)q5);
            q5 += 32;

            // Low nibble + high bit from qh
            const __m256i q5l_0 = _mm256_and_si256(q5bits, m4);
            const __m256i q5h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
            const __m256i q5_0  = _mm256_add_epi8(q5l_0, q5h_0);
            hmask = _mm256_slli_epi16(hmask, 1);

            // High nibble + next high bit from qh
            const __m256i q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4);
            const __m256i q5h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
            const __m256i q5_1  = _mm256_add_epi8(q5l_1, q5h_1);
            hmask = _mm256_slli_epi16(hmask, 1);

            // maddubs with Q8_K activation
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

    return hsum_avx2(acc) + summs;
}

static void gemv_q5_K_maddubs_transB_avx2(const float* a, const uint8_t* w, float* out,
                                            int M, int K, int N) {
    constexpr int QK_K = 256;
    constexpr int Q5_K_BLOCK_BYTES = 176;
    constexpr int nrc = 2;
    const int nb = (K + QK_K - 1) / QK_K;

    if (M == 1) {
        std::vector<block_q8_K> q8_buf(nb);
        quantize_row_q8_K(a, q8_buf.data(), K);

#    pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            _mm_prefetch((const char*)(w + (size_t)(n + 4) * nb * Q5_K_BLOCK_BYTES), _MM_HINT_T1);
            const uint8_t* q5_row = w + (size_t)n * nb * Q5_K_BLOCK_BYTES;
            out[n] = dot_q5_K_q8_K_avx2(q5_row, q8_buf.data(), nb);
        }
    } else {
        for (int m_start = 0; m_start < M; m_start += nrc) {
            int m_cur = (m_start + nrc <= M) ? nrc : (M - m_start);
            std::vector<block_q8_K> q8_tile(m_cur * nb);
            for (int m = 0; m < m_cur; ++m)
                quantize_row_q8_K(a + (m_start + m) * K, q8_tile.data() + m * nb, K);

#    pragma omp parallel for schedule(static)
            for (int n = 0; n < N; ++n) {
                const uint8_t* q5_row = w + (size_t)n * nb * Q5_K_BLOCK_BYTES;
                for (int m = 0; m < m_cur; ++m)
                    out[(m_start + m) * N + n] = dot_q5_K_q8_K_avx2(q5_row, q8_tile.data() + m * nb, nb);
            }
        }
    }
}

// ---- Q5_K fused FFN down + residual (decode, M=1, Q8_K activation) ----
static void gemv_q5_k_ffn_down_residual_avx2(const float* a, const uint8_t* w,
                                             const float* residual, float* out, int K, int N) {
    constexpr int QK_K = 256;
    constexpr int Q5_K_BLOCK_BYTES = 176;
    const int nb = (K + QK_K - 1) / QK_K;

    std::vector<block_q8_K> q8_buf(nb);
    quantize_row_q8_K(a, q8_buf.data(), K);

#    pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        const uint8_t* w_row = w + (size_t)n * nb * Q5_K_BLOCK_BYTES;
        out[n] = dot_q5_K_q8_K_avx2(w_row, q8_buf.data(), nb) + residual[n];
    }
}

// ---- Q5_K fused attention output projection + residual (decode, M=1, Q8_K activation) ----
static void gemv_q5_k_attn_proj_residual_avx2(const float* a, const uint8_t* w,
                                               const float* residual, float* out, int K, int N) {
    constexpr int QK_K = 256;
    constexpr int Q5_K_BLOCK_BYTES = 176;
    const int nb = (K + QK_K - 1) / QK_K;

    std::vector<block_q8_K> q8_buf(nb);
    quantize_row_q8_K(a, q8_buf.data(), K);

#    pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        const uint8_t* w_row = w + (size_t)n * nb * Q5_K_BLOCK_BYTES;
        out[n] = dot_q5_K_q8_K_avx2(w_row, q8_buf.data(), nb) + residual[n];
    }
}

// ---- Q5_K fused QKV projection (decode, M=1) ----
// Quantizes activation to Q8_K once, shares across Q, K, V projections.
static void gemv_q5_K_fused_qkv_avx2(const float* a, const uint8_t* wq, const uint8_t* wk,
                                      const uint8_t* wv, float* out_q, float* out_k,
                                      float* out_v, int K, int N_q, int N_k, int N_v) {
    constexpr int QK_K = 256;
    constexpr int Q5_K_BLOCK_BYTES = 176;
    const int nb = (K + QK_K - 1) / QK_K;

    std::vector<block_q8_K> q8_buf(nb);
    quantize_row_q8_K(a, q8_buf.data(), K);

#    pragma omp parallel for schedule(static)
    for (int n = 0; n < N_q; ++n) {
        const uint8_t* q5_row = wq + (size_t)n * nb * Q5_K_BLOCK_BYTES;
        out_q[n] = dot_q5_K_q8_K_avx2(q5_row, q8_buf.data(), nb);
    }
#    pragma omp parallel for schedule(static)
    for (int n = 0; n < N_k; ++n) {
        const uint8_t* q5_row = wk + (size_t)n * nb * Q5_K_BLOCK_BYTES;
        out_k[n] = dot_q5_K_q8_K_avx2(q5_row, q8_buf.data(), nb);
    }
#    pragma omp parallel for schedule(static)
    for (int n = 0; n < N_v; ++n) {
        const uint8_t* q5_row = wv + (size_t)n * nb * Q5_K_BLOCK_BYTES;
        out_v[n] = dot_q5_K_q8_K_avx2(q5_row, q8_buf.data(), nb);
    }
}

// ---- Q2_K fused QKV projection (decode, M=1) ----
// Quantizes activation to Q8_K once, shares across Q, K, V projections.
static void gemv_q2_K_fused_qkv_avx2(const float* a, const uint8_t* wq, const uint8_t* wk,
                                      const uint8_t* wv, float* out_q, float* out_k,
                                      float* out_v, int K, int N_q, int N_k, int N_v) {
    constexpr int QK_K = 256;
    constexpr int Q2_K_BLOCK_BYTES = 84;
    const int nb = (K + QK_K - 1) / QK_K;

    std::vector<block_q8_K> q8_buf(nb);
    quantize_row_q8_K(a, q8_buf.data(), K);

#    pragma omp parallel for schedule(static)
    for (int n = 0; n < N_q; ++n) {
        const uint8_t* q2_row = wq + (size_t)n * nb * Q2_K_BLOCK_BYTES;
        out_q[n] = dot_q2_K_q8_K_avx2(q2_row, q8_buf.data(), nb, nullptr);
    }
#    pragma omp parallel for schedule(static)
    for (int n = 0; n < N_k; ++n) {
        const uint8_t* q2_row = wk + (size_t)n * nb * Q2_K_BLOCK_BYTES;
        out_k[n] = dot_q2_K_q8_K_avx2(q2_row, q8_buf.data(), nb, nullptr);
    }
#    pragma omp parallel for schedule(static)
    for (int n = 0; n < N_v; ++n) {
        const uint8_t* q2_row = wv + (size_t)n * nb * Q2_K_BLOCK_BYTES;
        out_v[n] = dot_q2_K_q8_K_avx2(q2_row, q8_buf.data(), nb, nullptr);
    }
}
#endif  // USE_AVX2

}  // namespace cpu
}  // namespace forge
