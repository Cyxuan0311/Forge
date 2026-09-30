#pragma once
#include "cpu/common/thread_pool.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "../../common/quant_helpers.h"
#include "scales.h"
#include "vec.h"
#include "forge/host_mem_pool.h"
#ifdef USE_AVX2
#    include <immintrin.h>
#endif

namespace forge {
namespace cpu {

// ---- Q4_0 fused GEMV (transB layout) ----
// Q4_0 block: 2 bytes scale (fp16) + 16 bytes quants = 18 bytes per 32 elements

#ifdef USE_AVX2

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

static void gemv_q4_0_transB_avx2(const float* a, const uint8_t* w, float* out, int M, int K,
                                  int N) {
    constexpr int BLOCK_SIZE = 32;
    constexpr int BLOCK_BYTES = 18;
    const int blocks_per_row = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int full_blocks = K / BLOCK_SIZE;

    const __m128i lo_mask = _mm_set1_epi8(0x0F);
    const __m128i eight = _mm_set1_epi8(8);

#    define Q4_0_PROCESS_BLOCK(a_row, w_row, bi, acc)                                       \
        do {                                                                                \
            const uint8_t* _blk = (w_row) + (size_t)(bi)*18;                                \
            uint16_t _sb;                                                                   \
            memcpy(&_sb, _blk, 2);                                                          \
            __m256 _sc = fp16_to_fp32_broadcast_avx2(_sb);                                  \
            __m256 _pa = q4_0_block_partial_avx2(a_row, (bi)*32, _blk + 2, lo_mask, eight); \
            acc = _mm256_fmadd_ps(_sc, _pa, acc);                                           \
        } while (0)

#    define Q4_0_PREFETCH_BLOCK(w_row, bi)                    \
        do {                                                  \
            const uint8_t* _pblk = (w_row) + (size_t)(bi)*18; \
            _mm_prefetch((const char*)_pblk, _MM_HINT_T0);    \
        } while (0)

    if (M > 1) {
        thread_local static std::vector<float> _dequant_buf;
        if ((int)_dequant_buf.size() < K)
            _dequant_buf.resize(K);
        float* dq = _dequant_buf.data();

        cpu::parallel_for_dynamic(N, 4, [&](int n) {
            const uint8_t* w_row = w + (size_t)n * blocks_per_row * BLOCK_BYTES;

            // Dequantize entire weight row once
            for (int bi = 0; bi < full_blocks; ++bi) {
                const uint8_t* blk = w_row + (size_t)bi * BLOCK_BYTES;
                uint16_t _sb;
                memcpy(&_sb, blk, 2);
                __m256 _sc = fp16_to_fp32_broadcast_avx2(_sb);
                const uint8_t* qs = blk + 2;
                int base = bi * BLOCK_SIZE;

                __m128i q8 = _mm_loadu_si128((const __m128i*)qs);
                __m128i q_lo = _mm_and_si128(q8, lo_mask);
                __m128i q_lo_s = _mm_sub_epi8(q_lo, eight);
                __m128i q_hi = _mm_and_si128(_mm_srli_epi16(q8, 4), lo_mask);
                __m128i q_hi_s = _mm_sub_epi8(q_hi, eight);

                _mm256_storeu_ps(
                    dq + base + 0,
                    _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_lo_s)), _sc));
                _mm256_storeu_ps(
                    dq + base + 8,
                    _mm256_mul_ps(
                        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_lo_s, 8))), _sc));
                _mm256_storeu_ps(
                    dq + base + 16,
                    _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_hi_s)), _sc));
                _mm256_storeu_ps(
                    dq + base + 24,
                    _mm256_mul_ps(
                        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_hi_s, 8))), _sc));
            }

            if (full_blocks < blocks_per_row) {
                int base = full_blocks * BLOCK_SIZE;
                int rem = K - base;
                if (rem > 0) {
                    const uint8_t* blk = w_row + (size_t)full_blocks * BLOCK_BYTES;
                    uint16_t _sb;
                    memcpy(&_sb, blk, 2);
                    const float _sf = fp16_to_float_scalar(_sb);
                    const uint8_t* qs = blk + 2;
                    for (int j = 0; j < 16 && base + j < K; ++j)
                        dq[base + j] = static_cast<float>((qs[j] & 0x0F) - 8) * _sf;
                    for (int j = 0; j < 16 && base + 16 + j < K; ++j)
                        dq[base + 16 + j] = static_cast<float>(((qs[j] >> 4) & 0x0F) - 8) * _sf;
                }
            }

            for (int m = 0; m < M; ++m) {
                out[m * N + n] = dot_product_avx2(a + m * K, dq, K);
            }
            });
    } else if (M == 1) {
        constexpr int NR = 4;
        const float* a_row = a;

        cpu::parallel_for_chunks(N, NR, [&](int n0, int n1) {
            for (int n = n0; n < n1; n += NR) {
            int rows = (n + NR <= N) ? NR : (N - n);

            __m256 acc0_a = _mm256_setzero_ps(), acc0_b = _mm256_setzero_ps();
            __m256 acc1_a = _mm256_setzero_ps(), acc1_b = _mm256_setzero_ps();
            __m256 acc2_a = _mm256_setzero_ps(), acc2_b = _mm256_setzero_ps();
            __m256 acc3_a = _mm256_setzero_ps(), acc3_b = _mm256_setzero_ps();

            const uint8_t* w_row0 = w + (size_t)(n + 0) * blocks_per_row * BLOCK_BYTES;
            const uint8_t* w_row1 =
                (rows > 1) ? w + (size_t)(n + 1) * blocks_per_row * BLOCK_BYTES : nullptr;
            const uint8_t* w_row2 =
                (rows > 2) ? w + (size_t)(n + 2) * blocks_per_row * BLOCK_BYTES : nullptr;
            const uint8_t* w_row3 =
                (rows > 3) ? w + (size_t)(n + 3) * blocks_per_row * BLOCK_BYTES : nullptr;

            Q4_0_PREFETCH_BLOCK(w_row0, 0);
            if (rows > 1)
                Q4_0_PREFETCH_BLOCK(w_row1, 0);
            if (rows > 2)
                Q4_0_PREFETCH_BLOCK(w_row2, 0);
            if (rows > 3)
                Q4_0_PREFETCH_BLOCK(w_row3, 0);

            int bi = 0;
            for (; bi + 1 < full_blocks; bi += 2) {
                Q4_0_PROCESS_BLOCK(a_row, w_row0, bi, acc0_a);
                if (rows > 1)
                    Q4_0_PROCESS_BLOCK(a_row, w_row1, bi, acc1_a);
                if (rows > 2)
                    Q4_0_PROCESS_BLOCK(a_row, w_row2, bi, acc2_a);
                if (rows > 3)
                    Q4_0_PROCESS_BLOCK(a_row, w_row3, bi, acc3_a);

                if (bi + 2 < full_blocks) {
                    Q4_0_PREFETCH_BLOCK(w_row0, bi + 2);
                    if (rows > 1)
                        Q4_0_PREFETCH_BLOCK(w_row1, bi + 2);
                    if (rows > 2)
                        Q4_0_PREFETCH_BLOCK(w_row2, bi + 2);
                    if (rows > 3)
                        Q4_0_PREFETCH_BLOCK(w_row3, bi + 2);
                }

                Q4_0_PROCESS_BLOCK(a_row, w_row0, bi + 1, acc0_b);
                if (rows > 1)
                    Q4_0_PROCESS_BLOCK(a_row, w_row1, bi + 1, acc1_b);
                if (rows > 2)
                    Q4_0_PROCESS_BLOCK(a_row, w_row2, bi + 1, acc2_b);
                if (rows > 3)
                    Q4_0_PROCESS_BLOCK(a_row, w_row3, bi + 1, acc3_b);
            }

            if (bi < full_blocks) {
                Q4_0_PROCESS_BLOCK(a_row, w_row0, bi, acc0_a);
                if (rows > 1)
                    Q4_0_PROCESS_BLOCK(a_row, w_row1, bi, acc1_a);
                if (rows > 2)
                    Q4_0_PROCESS_BLOCK(a_row, w_row2, bi, acc2_a);
                if (rows > 3)
                    Q4_0_PROCESS_BLOCK(a_row, w_row3, bi, acc3_a);
                ++bi;
            }

            __m256 acc0 = _mm256_add_ps(acc0_a, acc0_b);
            __m256 acc1 = _mm256_add_ps(acc1_a, acc1_b);
            __m256 acc2 = _mm256_add_ps(acc2_a, acc2_b);
            __m256 acc3 = _mm256_add_ps(acc3_a, acc3_b);

            if (full_blocks < blocks_per_row) {
                int base = full_blocks * BLOCK_SIZE;
                int remaining = K - base;
                if (remaining > 0) {
                    auto process_partial = [&](const uint8_t* w_row, __m256& acc_ref) {
                        const uint8_t* block = w_row + (size_t)full_blocks * BLOCK_BYTES;
                        uint16_t sb;
                        memcpy(&sb, block, 2);
                        const float scale_f = fp16_to_float_scalar(sb);
                        const uint8_t* qs = block + 2;
                        for (int j = 0; j < 16 && base + j < K; ++j) {
                            float qv = static_cast<float>((qs[j] & 0x0F) - 8) * scale_f;
                            acc_ref = _mm256_fmadd_ps(_mm256_set1_ps(a_row[base + j]),
                                                      _mm256_set1_ps(qv), acc_ref);
                        }
                        for (int j = 0; j < 16 && base + 16 + j < K; ++j) {
                            float qv = static_cast<float>(((qs[j] >> 4) & 0x0F) - 8) * scale_f;
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
            });
    }

#    undef Q4_0_PROCESS_BLOCK
#    undef Q4_0_PREFETCH_BLOCK
}

#endif  // USE_AVX2

// ---- FP32 GEMV (transB layout) ----

#ifdef USE_AVX2
// ---------------------------------------------------------------------------
// Native FP16 / BF16 weight GEMV (transB layout), AVX2 + F16C.
//
// Why this exists. An F16 or BF16 GGUF used to be expanded to FP32 at load time
// (gguf_model.cpp), which doubled both the resident footprint and the per-token
// memory traffic. On a 2.6B model that is 5.4 GB of weights read as 10.8 GB per
// generated token, and on a 15 GB host that expansion is close to fatal.
//
// The fix is the one llama.cpp uses (ggml/src/ggml-cpu/vec.h, simd-mappings.h:
// GGML_F16_STEP 32 / GGML_F16_EPR 8 under AVX2+F16C): keep the weights at 2
// bytes and widen them in registers, 8 elements at a time, accumulating in FP32.
// The F16C conversion is one instruction per 8 elements and is not the
// bottleneck -- this kernel is memory-bound, so halving the bytes is the whole
// win.
//
//   FP16: _mm256_cvtph_ps  (vcvtph2ps, 8 x fp16 -> 8 x fp32)
//   BF16: _mm256_cvtepu16_epi32 then <<16, because bf16 IS the top half of a
//         float32, so the widening is a shift rather than a conversion. There is
//         no bf16 F16C instruction, and the shift is exact.
//
// Accumulation is FP32 in both cases, matching the FP32 path this replaces, so
// the only numerical difference from the old behaviour is the width of the
// weights themselves -- which is the same difference the model already had.
// ---------------------------------------------------------------------------

#ifdef USE_AVX2

// A 2-byte weight format, viewed as: how to widen 8 of them, and how to widen one
// (the K tail, where a vector load would overrun). Both members are needed
// because the tail is dtype-specific too -- widening a bf16 with the fp16 rule
// would silently produce garbage in the last few elements of every row.
struct WgtF16 {
    static inline __m256 load8(const uint16_t* p) {
        return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
    }
    static inline float scalar(uint16_t v) {
        return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<int>(v))));
    }
};

struct WgtBF16 {
    // No bf16 F16C instruction exists; bf16 is literally the top half of a
    // float32, so widening is an exact zero-extend plus a 16-bit shift.
    static inline __m256 load8(const uint16_t* p) {
        const __m256i u16 =
            _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
        return _mm256_castsi256_ps(_mm256_slli_epi32(u16, 16));
    }
    static inline float scalar(uint16_t v) {
        const uint32_t bits = static_cast<uint32_t>(v) << 16;
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        return f;
    }
};

#endif  // USE_AVX2

// Generic native-16-bit-weight GEMV (transB). W supplies the widening.
template <typename W>
static inline void gemv_f16like_transB_avx2_impl(const float* a, const uint16_t* b, float* out,
                                                 int M, int K, int N) {
    if (M == 1) {
        // Same 4-row unroll and chunking as gemv_fp32_transB_avx2, so the two
        // paths schedule identically and a dtype switch does not change
        // threading behaviour.
        constexpr int NR = 4;
        const float* a_row = a;
        cpu::parallel_for_chunks(N, NR, [&](int n0, int n1) {
            for (int n = n0; n < n1; n += NR) {
                const int rows = (n + NR <= N) ? NR : (N - n);
                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                __m256 acc2 = _mm256_setzero_ps();
                __m256 acc3 = _mm256_setzero_ps();

                const uint16_t* b_row0 = b + static_cast<size_t>(n + 0) * K;
                const uint16_t* b_row1 = (rows > 1) ? b + static_cast<size_t>(n + 1) * K : nullptr;
                const uint16_t* b_row2 = (rows > 2) ? b + static_cast<size_t>(n + 2) * K : nullptr;
                const uint16_t* b_row3 = (rows > 3) ? b + static_cast<size_t>(n + 3) * K : nullptr;

                for (int k = 0; k + 8 <= K; k += 8) {
                    const __m256 av = _mm256_loadu_ps(a_row + k);
                    acc0 = _mm256_fmadd_ps(av, W::load8(b_row0 + k), acc0);
                    if (rows > 1) acc1 = _mm256_fmadd_ps(av, W::load8(b_row1 + k), acc1);
                    if (rows > 2) acc2 = _mm256_fmadd_ps(av, W::load8(b_row2 + k), acc2);
                    if (rows > 3) acc3 = _mm256_fmadd_ps(av, W::load8(b_row3 + k), acc3);
                }

                const int ktail = (K / 8) * 8;
                const auto finalize = [&](const uint16_t* b_r, __m256 acc_val) -> float {
                    float sum = hsum_avx2(acc_val);
                    for (int k = ktail; k < K; ++k)
                        sum += a_row[k] * W::scalar(b_r[k]);
                    return sum;
                };
                out[n + 0] = finalize(b_row0, acc0);
                if (rows > 1) out[n + 1] = finalize(b_row1, acc1);
                if (rows > 2) out[n + 2] = finalize(b_row2, acc2);
                if (rows > 3) out[n + 3] = finalize(b_row3, acc3);
            }
        });
    } else {
        cpu::parallel_for(M, [&](int m) {
            const float* a_row = a + static_cast<size_t>(m) * K;
            float* o_row = out + static_cast<size_t>(m) * N;
            for (int n = 0; n < N; ++n) {
                const uint16_t* b_row = b + static_cast<size_t>(n) * K;
                __m256 acc = _mm256_setzero_ps();
                int k = 0;
                for (; k + 8 <= K; k += 8)
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(a_row + k), W::load8(b_row + k), acc);
                float sum = hsum_avx2(acc);
                for (; k < K; ++k)
                    sum += a_row[k] * W::scalar(b_row[k]);
                o_row[n] = sum;
            }
        });
    }
}

static inline void gemv_fp16_transB_avx2(const float* a, const uint16_t* b, float* out, int M,
                                        int K, int N) {
    gemv_f16like_transB_avx2_impl<WgtF16>(a, b, out, M, K, N);
}

static inline void gemv_bf16_transB_avx2(const float* a, const uint16_t* b, float* out, int M,
                                         int K, int N) {
    gemv_f16like_transB_avx2_impl<WgtBF16>(a, b, out, M, K, N);
}

static void gemv_fp32_transB_avx2(const float* a, const float* b, float* out, int M, int K, int N) {
    if (M == 1) {
        constexpr int NR = 4;
        const float* a_row = a;
        cpu::parallel_for_chunks(N, NR, [&](int n0, int n1) {
            for (int n = n0; n < n1; n += NR) {
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
            });
    } else {
        cpu::parallel_for(M, [&](int m) {
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
            });
    }
}
#endif  // USE_AVX2

// ---- Q8_0 fused GEMV (transB layout) ----
// Q8_0 block: 2 bytes scale (fp16) + 32 bytes int8 quants = 34 bytes per 32 elements

#ifdef USE_AVX2
static void gemv_q8_0_transB_avx2(const float* a, const uint8_t* w, float* out, int M, int K,
                                  int N) {
    constexpr int BLOCK_SIZE = 32;
    constexpr int BLOCK_BYTES = 34;
    const int blocks_per_row = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    auto process_block = [&](const float* a_row, const uint8_t* w_row, int bi, __m256& acc) {
        const uint8_t* block = w_row + bi * BLOCK_BYTES;
        uint16_t scale_bits;
        memcpy(&scale_bits, block, 2);
        __m256 scale = fp16_to_fp32_broadcast_avx2(scale_bits);
        const int8_t* qs = reinterpret_cast<const int8_t*>(block + 2);
        int base = bi * BLOCK_SIZE;

        __m256 partial = _mm256_mul_ps(_mm256_loadu_ps(a_row + base),
                                       _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                                           _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qs)))));
        partial = _mm256_fmadd_ps(_mm256_loadu_ps(a_row + base + 8),
                                  _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                                      _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qs + 8)))),
                                  partial);
        partial = _mm256_fmadd_ps(_mm256_loadu_ps(a_row + base + 16),
                                  _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                                      _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qs + 16)))),
                                  partial);
        partial = _mm256_fmadd_ps(_mm256_loadu_ps(a_row + base + 24),
                                  _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                                      _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qs + 24)))),
                                  partial);

        acc = _mm256_fmadd_ps(scale, partial, acc);
    };

    if (M > 1) {
        thread_local static std::vector<float> _dq;
        if ((int)_dq.size() < K)
            _dq.resize(K);
        float* dq = _dq.data();

        cpu::parallel_for_dynamic(N, 4, [&](int n) {
            const uint8_t* w_row = w + (size_t)n * blocks_per_row * BLOCK_BYTES;

            for (int bi = 0; bi < blocks_per_row; ++bi) {
                int base = bi * BLOCK_SIZE;
                int remaining = K - base;
                if (remaining >= BLOCK_SIZE) {
                    const uint8_t* block = w_row + bi * BLOCK_BYTES;
                    uint16_t _sb;
                    memcpy(&_sb, block, 2);
                    __m256 _sc = fp16_to_fp32_broadcast_avx2(_sb);
                    const int8_t* qs = reinterpret_cast<const int8_t*>(block + 2);

                    _mm256_storeu_ps(dq + base + 0,
                                     _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                                                       _mm_loadl_epi64((const __m128i*)qs))),
                                                   _sc));
                    _mm256_storeu_ps(dq + base + 8,
                                     _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                                                       _mm_loadl_epi64((const __m128i*)(qs + 8)))),
                                                   _sc));
                    _mm256_storeu_ps(dq + base + 16,
                                     _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                                                       _mm_loadl_epi64((const __m128i*)(qs + 16)))),
                                                   _sc));
                    _mm256_storeu_ps(dq + base + 24,
                                     _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                                                       _mm_loadl_epi64((const __m128i*)(qs + 24)))),
                                                   _sc));
                } else if (remaining > 0) {
                    const uint8_t* block = w_row + bi * BLOCK_BYTES;
                    uint16_t _sb;
                    memcpy(&_sb, block, 2);
                    const float _sf = fp16_to_float_scalar(_sb);
                    const int8_t* qs = reinterpret_cast<const int8_t*>(block + 2);
                    for (int j = 0; j < remaining; ++j)
                        dq[base + j] = static_cast<float>(qs[j]) * _sf;
                }
            }

            for (int m = 0; m < M; ++m) {
                out[m * N + n] = dot_product_avx2(a + m * K, dq, K);
            }
            });
    } else {
        constexpr int NR = 4;
        const float* a_row = a;

        cpu::parallel_for_chunks(N, NR, [&](int n0, int n1) {
            for (int n = n0; n < n1; n += NR) {
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

            for (int bi = 0; bi < blocks_per_row; ++bi) {
                int base = bi * BLOCK_SIZE;
                if (K - base < BLOCK_SIZE)
                    break;

                process_block(a_row, w_row0, bi, acc0);
                if (rows > 1)
                    process_block(a_row, w_row1, bi, acc1);
                if (rows > 2)
                    process_block(a_row, w_row2, bi, acc2);
                if (rows > 3)
                    process_block(a_row, w_row3, bi, acc3);
            }

            {
                int bi = blocks_per_row - 1;
                int base = bi * BLOCK_SIZE;
                int remaining = K - base;
                if (remaining > 0 && remaining < BLOCK_SIZE) {
                    auto process_partial = [&](const uint8_t* w_row, __m256& acc_ref) {
                        const uint8_t* block = w_row + bi * BLOCK_BYTES;
                        uint16_t sb;
                        memcpy(&sb, block, 2);
                        const float scale_f = fp16_to_float_scalar(sb);
                        const int8_t* qs = reinterpret_cast<const int8_t*>(block + 2);
                        for (int j = 0; j < remaining; ++j) {
                            acc_ref = _mm256_fmadd_ps(
                                _mm256_set1_ps(a_row[base + j]),
                                _mm256_set1_ps(static_cast<float>(qs[j]) * scale_f), acc_ref);
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
            });
    }
}
#endif  // USE_AVX2

// ---- Q4_1 fused GEMV (transB layout) ----
// Q4_1 block: 2 bytes scale (fp16) + 2 bytes min (fp16) + 16 bytes quants = 20 bytes per 32
// elements

#ifdef USE_AVX2

static inline void q4_1_block_partial_avx2(const float* a_row, int base, const uint8_t* qs,
                                           __m256& partial, __m256& sum_a) {
    const __m128i lo_mask = _mm_set1_epi8(0x0F);
    __m128i q8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs));
    __m128i q_lo = _mm_and_si128(q8, lo_mask);
    __m128i q_hi = _mm_and_si128(_mm_srli_epi16(q8, 4), lo_mask);

    __m256 a0 = _mm256_loadu_ps(a_row + base);
    __m256 q0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_lo));
    partial = _mm256_mul_ps(a0, q0);
    sum_a = a0;

    __m256 a1 = _mm256_loadu_ps(a_row + base + 8);
    __m256 q1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_lo, 8)));
    partial = _mm256_fmadd_ps(a1, q1, partial);
    sum_a = _mm256_add_ps(sum_a, a1);

    __m256 a2 = _mm256_loadu_ps(a_row + base + 16);
    __m256 q2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_hi));
    partial = _mm256_fmadd_ps(a2, q2, partial);
    sum_a = _mm256_add_ps(sum_a, a2);

    __m256 a3 = _mm256_loadu_ps(a_row + base + 24);
    __m256 q3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_hi, 8)));
    partial = _mm256_fmadd_ps(a3, q3, partial);
    sum_a = _mm256_add_ps(sum_a, a3);
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
            __m256 _ms = _mm256_mul_ps(_m, _sum_a);                              \
            acc_a = _mm256_fmadd_ps(_d, _partial, acc_a);                        \
            acc_b = _mm256_add_ps(acc_b, _ms);                                   \
        } while (0)

#    define Q4_1_PREFETCH_BLOCK(w_row, bi)                    \
        do {                                                  \
            const uint8_t* _pblk = (w_row) + (size_t)(bi)*20; \
            _mm_prefetch((const char*)_pblk, _MM_HINT_T0);    \
        } while (0)

    if (M > 1) {
        thread_local static std::vector<float> _dq;
        if ((int)_dq.size() < K)
            _dq.resize(K);
        float* dq = _dq.data();

        cpu::parallel_for_dynamic(N, 4, [&](int n) {
            const uint8_t* w_row = w + (size_t)n * blocks_per_row * BLOCK_BYTES;

            for (int bi = 0; bi < full_blocks; ++bi) {
                const uint8_t* blk = w_row + (size_t)bi * BLOCK_BYTES;
                uint16_t _db, _mb;
                memcpy(&_db, blk, 2);
                memcpy(&_mb, blk + 2, 2);
                float _d = fp16_to_float_scalar(_db);
                float _m = fp16_to_float_scalar(_mb);
                const uint8_t* qs = blk + 4;
                int base = bi * BLOCK_SIZE;
                for (int j = 0; j < 16; ++j) {
                    dq[base + j] = static_cast<float>(qs[j] & 0x0F) * _d + _m;
                    dq[base + 16 + j] = static_cast<float>((qs[j] >> 4) & 0x0F) * _d + _m;
                }
            }
            if (full_blocks < blocks_per_row) {
                int base = full_blocks * BLOCK_SIZE;
                int rem = K - base;
                if (rem > 0) {
                    const uint8_t* blk = w_row + (size_t)full_blocks * BLOCK_BYTES;
                    float _d = fp16_to_float_scalar(*((const uint16_t*)blk));
                    float _m = fp16_to_float_scalar(*((const uint16_t*)(blk + 2)));
                    const uint8_t* qs = blk + 4;
                    for (int j = 0; j < 16 && base + j < K; ++j)
                        dq[base + j] = static_cast<float>(qs[j] & 0x0F) * _d + _m;
                    for (int j = 0; j < 16 && base + 16 + j < K; ++j)
                        dq[base + 16 + j] = static_cast<float>((qs[j] >> 4) & 0x0F) * _d + _m;
                }
            }

            for (int m = 0; m < M; ++m) {
                out[m * N + n] = dot_product_avx2(a + m * K, dq, K);
            }
            });
    } else {
        constexpr int NR = 4;
        const float* a_row = a;

        cpu::parallel_for_chunks(N, NR, [&](int n0, int n1) {
            for (int n = n0; n < n1; n += NR) {
            int rows = (n + NR <= N) ? NR : (N - n);

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

            Q4_1_PREFETCH_BLOCK(w_row0, 0);
            if (rows > 1)
                Q4_1_PREFETCH_BLOCK(w_row1, 0);
            if (rows > 2)
                Q4_1_PREFETCH_BLOCK(w_row2, 0);
            if (rows > 3)
                Q4_1_PREFETCH_BLOCK(w_row3, 0);

            int bi = 0;
            for (; bi + 1 < full_blocks; bi += 2) {
                Q4_1_PROCESS_BLOCK(a_row, w_row0, bi, acc0_p, acc0_m);
                if (rows > 1)
                    Q4_1_PROCESS_BLOCK(a_row, w_row1, bi, acc1_p, acc1_m);
                if (rows > 2)
                    Q4_1_PROCESS_BLOCK(a_row, w_row2, bi, acc2_p, acc2_m);
                if (rows > 3)
                    Q4_1_PROCESS_BLOCK(a_row, w_row3, bi, acc3_p, acc3_m);

                if (bi + 2 < full_blocks) {
                    Q4_1_PREFETCH_BLOCK(w_row0, bi + 2);
                    if (rows > 1)
                        Q4_1_PREFETCH_BLOCK(w_row1, bi + 2);
                    if (rows > 2)
                        Q4_1_PREFETCH_BLOCK(w_row2, bi + 2);
                    if (rows > 3)
                        Q4_1_PREFETCH_BLOCK(w_row3, bi + 2);
                }

                Q4_1_PROCESS_BLOCK(a_row, w_row0, bi + 1, acc0_p, acc0_m);
                if (rows > 1)
                    Q4_1_PROCESS_BLOCK(a_row, w_row1, bi + 1, acc1_p, acc1_m);
                if (rows > 2)
                    Q4_1_PROCESS_BLOCK(a_row, w_row2, bi + 1, acc2_p, acc2_m);
                if (rows > 3)
                    Q4_1_PROCESS_BLOCK(a_row, w_row3, bi + 1, acc3_p, acc3_m);
            }

            if (bi < full_blocks) {
                Q4_1_PROCESS_BLOCK(a_row, w_row0, bi, acc0_p, acc0_m);
                if (rows > 1)
                    Q4_1_PROCESS_BLOCK(a_row, w_row1, bi, acc1_p, acc1_m);
                if (rows > 2)
                    Q4_1_PROCESS_BLOCK(a_row, w_row2, bi, acc2_p, acc2_m);
                if (rows > 3)
                    Q4_1_PROCESS_BLOCK(a_row, w_row3, bi, acc3_p, acc3_m);
            }

            __m256 acc0 = _mm256_add_ps(acc0_p, acc0_m);
            __m256 acc1 = _mm256_add_ps(acc1_p, acc1_m);
            __m256 acc2 = _mm256_add_ps(acc2_p, acc2_m);
            __m256 acc3 = _mm256_add_ps(acc3_p, acc3_m);

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
            });
    }

#    undef Q4_1_PROCESS_BLOCK
#    undef Q4_1_PREFETCH_BLOCK
}
#endif  // USE_AVX2

// ---- Q4_K fused GEMV (transB layout) ----
// Q4_K block: 2+2+12+128 = 144 bytes per 256 elements
// Dot is computed as Q4_K x Q8_K (activation quantized to Q8_K on the fly)

#ifdef USE_AVX2

// Dot product of one Q4_K weight row with one Q8_K activation block row.
// Returns the sum for (nb) Q4_K super-blocks.
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

        // Duplicate scales across both 128-bit lanes
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

// Q4_K GEMV (transB layout) using fused Q4_K x Q8_K dot product.
// For M=1 (common decode case): quantizes activation once, then does N dot products.
static void gemv_q4_K_transB_avx2(const float* a, const uint8_t* w, float* out, int M, int K,
                                  int N) {
    constexpr int QK_K = 256;
    constexpr int Q4_K_BLOCK_BYTES = 144;
    const int nb = (K + QK_K - 1) / QK_K;

    if (M == 1) {
        scratch_vec<block_q8_K> q8_buf(nb);
        quantize_row_q8_K(a, q8_buf.data(), K);

        cpu::parallel_for(N, [&](int n) {
            const uint8_t* q4_row = w + (size_t)n * nb * Q4_K_BLOCK_BYTES;
            out[n] = dot_q4_K_q8_K_avx2(q4_row, q8_buf.data(), nb);
            });
    } else {
        scratch_vec<block_q8_K> q8_all(M * nb);
        for (int m = 0; m < M; ++m) {
            quantize_row_q8_K(a + m * K, q8_all.data() + m * nb, K);
        }

        cpu::parallel_for(N, [&](int n) {
            const uint8_t* q4_row = w + (size_t)n * nb * Q4_K_BLOCK_BYTES;
            for (int m = 0; m < M; ++m) {
                out[m * N + n] = dot_q4_K_q8_K_avx2(q4_row, q8_all.data() + m * nb, nb);
            }
            });
    }
}

// Q4_K Batch-GEMV (transB layout) for M > 1 (prefill).
// Key optimization: Q4_K weight decoding (scales, qs bit extraction) is done once
// per weight row, then reused for all M input vectors. This avoids re-reading and
// re-decoding the same Q4_K blocks M times, saving (M-1)/M of the Q4_K bandwidth.
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
    scratch_vec<block_q8_K> q8_all(M * nb);
    for (int m = 0; m < M; ++m) {
        quantize_row_q8_K(a + m * K, q8_all.data() + m * nb, K);
    }

// Accumulators per OpenMP thread, allocated once per call (was per-iteration
    // _mm_malloc inside the parallel loop: N heap alloc/frees per GEMV).
    const int max_threads = cpu::get_num_threads();
    __m256* acc_storage = static_cast<__m256*>(forge::host_mem::allocate(
        static_cast<size_t>(max_threads) * M * sizeof(__m256)));
    __m128* acc_m_storage = static_cast<__m128*>(forge::host_mem::allocate(
        static_cast<size_t>(max_threads) * M * sizeof(__m128)));

    cpu::parallel_for_chunks_ex(N, 1, [&](int ith, int nth, int n0, int n1) {
        for (int n = n0; n < n1; ++n) {
        const uint8_t* q4_row = w + (size_t)n * nb * Q4_K_BLOCK_BYTES;

        // Per-M accumulators (FP32 for the dot, __m128 for min contribution)
        __m256* acc_vec = acc_storage + (size_t)ith * M;
        __m128* acc_m_vec = acc_m_storage + (size_t)ith * M;
        for (int m = 0; m < M; ++m) {
            acc_vec[m] = _mm256_setzero_ps();
            acc_m_vec[m] = _mm_setzero_ps();
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
    }
        });

    forge::host_mem::deallocate(acc_storage);
    forge::host_mem::deallocate(acc_m_storage);
}

// Q6_K GEMV has been consolidated into cpu_gemv.h to avoid duplication with matmul.cpp.
// The canonical implementation is cpu::dot_q6_K_q8_K_avx2 and cpu::gemv_q6_K_transB_avx2.

#endif  // USE_AVX2

}  // namespace cpu
}  // namespace forge
