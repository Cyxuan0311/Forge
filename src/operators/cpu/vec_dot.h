#pragma once
// AVX2 maddubs-based quantized dot product primitives.
// Phase 7: quantize activation to Q8_0_act, then use _mm256_maddubs_epi16
// for int8 dot products instead of full dequantization to FP32.
//
// This header is self-contained and does NOT include vec.h or quant_helpers.h
// to avoid redefinition issues when included from cpu_gemv.h.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#ifdef USE_AVX2
#    include <immintrin.h>
#endif

namespace forge {
namespace cpu {

// ---- Q8_0 activation block (for dot products, not for storage) ----
// Uses FP32 scale for simplicity (only exists in working memory).
struct block_q8_0_act {
    float   d;       // scale (FP32)
    int8_t  qs[32];  // quantized values
};

#ifdef USE_AVX2

// ---- Internal helpers (namespaced to avoid conflicts with vec.h) ----

namespace vdot {

static inline float hsum_ps256(__m256 v) {
    __m128 hi128 = _mm256_extractf128_ps(v, 1);
    __m128 lo128 = _mm256_castps256_ps128(v);
    __m128 sum128 = _mm_add_ps(lo128, hi128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}

static inline float fp16_to_fp32(uint16_t bits) {
    uint32_t sign = (bits >> 15) & 1;
    uint32_t exponent = (bits >> 10) & 0x1F;
    uint32_t mantissa = bits & 0x3FF;
    if (exponent == 0) {
        if (mantissa == 0) return 0.0f;
        float v = std::ldexp(static_cast<float>(mantissa) / 1024.0f, -14);
        return sign ? -v : v;
    }
    float v = std::ldexp(1.0f + static_cast<float>(mantissa) / 1024.0f,
                         static_cast<int>(exponent) - 15);
    return sign ? -v : v;
}

static inline int32_t hsum_i32(__m256i v) {
    __m128i lo = _mm256_extracti128_si256(v, 0);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i sum = _mm_add_epi32(lo, hi);
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_extract_epi32(sum, 0);
}

// Unpack 16 bytes of 4-bit nibbles into 32 unsigned bytes [0..15]
static inline __m256i nibbles_to_bytes_32(const uint8_t* rsi) {
    const __m128i tmp = _mm_loadu_si128((const __m128i*)rsi);
    const __m128i tmp_hi = _mm_srli_epi16(tmp, 4);
    const __m256i bytes = _mm256_set_m128i(tmp_hi, tmp);
    const __m256i lowMask = _mm256_set1_epi8(0xF);
    return _mm256_and_si256(lowMask, bytes);
}

// Unpack 16 bytes of 4-bit nibbles + 4 bytes of high bits into 32 unsigned bytes for Q5_x.
static inline __m256i nibbles5_to_bytes_32(const uint8_t* qs, const uint8_t* qh) {
    __m256i q4 = nibbles_to_bytes_32(qs);
    uint32_t qh_bits;
    memcpy(&qh_bits, qh, 4);
    alignas(32) uint8_t q5_mask[32];
    for (int i = 0; i < 32; ++i)
        q5_mask[i] = ((qh_bits >> i) & 1) << 4;
    __m256i hi_bits = _mm256_load_si256((const __m256i*)q5_mask);
    return _mm256_or_si256(q4, hi_bits);
}

}  // namespace vdot

// ---- Quantize FP32 row to Q8_0_act activation format ----
static inline void quantize_row_q8_0_act(const float* src, block_q8_0_act* dst, int k) {
    constexpr int QK = 32;
    const int nb = (k + QK - 1) / QK;
    for (int i = 0; i < nb; ++i) {
        int base = i * QK;
        int n_el = (base + QK <= k) ? QK : (k - base);
        float amax = 0.0f;
        for (int j = 0; j < n_el; ++j) {
            float v = std::abs(src[base + j]);
            if (v > amax) amax = v;
        }
        float d = amax / 127.0f;
        float id = d > 0.0f ? 1.0f / d : 0.0f;
        dst[i].d = d;
        for (int j = 0; j < n_el; ++j) {
            int q = (int)(src[base + j] * id + (src[base + j] >= 0 ? 0.5f : -0.5f));
            if (q < -128) q = -128;
            if (q > 127) q = 127;
            dst[i].qs[j] = (int8_t)q;
        }
        for (int j = n_el; j < QK; ++j)
            dst[i].qs[j] = 0;
    }
}

// ---- vec_dot primitives ----

// Q4_0 weight × Q8_0_act activation dot product.
// Q4_0 value = (nibble - 8) * scale_w
// dot = scale_w * scale_a * (maddubs(nibbles, q8) - 8 * sum(q8))
static inline float vec_dot_q4_0_q8_0_avx2(const uint8_t* w_row,
                                             const block_q8_0_act* act,
                                             int nb) {
    constexpr int BLOCK_BYTES = 18;
    __m256 acc = _mm256_setzero_ps();
    const __m256i ones_epi8 = _mm256_set1_epi8(1);
    const __m256i ones_epi16 = _mm256_set1_epi16(1);
    const __m256i eight_epi32 = _mm256_set1_epi32(8);

    for (int i = 0; i < nb; ++i) {
        const uint8_t* blk = w_row + (size_t)i * BLOCK_BYTES;
        _mm_prefetch((const char*)(w_row + (size_t)(i + 4) * BLOCK_BYTES), _MM_HINT_T0);
        uint16_t scale_bits;
        memcpy(&scale_bits, blk, 2);
        float combined_scale = vdot::fp16_to_fp32(scale_bits) * act[i].d;

        __m256i q4 = vdot::nibbles_to_bytes_32(blk + 2);  // unsigned [0..15]
        __m256i q8 = _mm256_loadu_si256((const __m256i*)act[i].qs);  // signed

        // maddubs: unsigned × signed → 16 int16 pair-sums
        __m256i p16 = _mm256_maddubs_epi16(q4, q8);
        __m256i p32 = _mm256_madd_epi16(p16, ones_epi16);

        // sum(q8) for -8 correction
        __m256i q8_sum16 = _mm256_maddubs_epi16(ones_epi8, q8);
        __m256i q8_sum32 = _mm256_madd_epi16(q8_sum16, ones_epi16);
        __m256i correction = _mm256_mullo_epi32(q8_sum32, eight_epi32);
        __m256i corrected = _mm256_sub_epi32(p32, correction);

        __m256 sc = _mm256_set1_ps(combined_scale);
        acc = _mm256_fmadd_ps(sc, _mm256_cvtepi32_ps(corrected), acc);
    }

    return vdot::hsum_ps256(acc);
}

// Q4_1 weight × Q8_0_act activation dot product.
// Q4_1 value = nibble * d + m
// dot = d * maddubs(nibbles, q8) * scale_a + m * sum(q8) * scale_a
static inline float vec_dot_q4_1_q8_0_avx2(const uint8_t* w_row,
                                             const block_q8_0_act* act,
                                             int nb) {
    constexpr int BLOCK_BYTES = 20;
    __m256 acc_d = _mm256_setzero_ps();
    float acc_m = 0.0f;
    const __m256i ones_epi8 = _mm256_set1_epi8(1);
    const __m256i ones_epi16 = _mm256_set1_epi16(1);

    for (int i = 0; i < nb; ++i) {
        const uint8_t* blk = w_row + (size_t)i * BLOCK_BYTES;
        _mm_prefetch((const char*)(w_row + (size_t)(i + 4) * BLOCK_BYTES), _MM_HINT_T0);
        uint16_t d_bits, m_bits;
        memcpy(&d_bits, blk, 2);
        memcpy(&m_bits, blk + 2, 2);
        float d_val = vdot::fp16_to_fp32(d_bits);
        float m_val = vdot::fp16_to_fp32(m_bits);
        float scale_a = act[i].d;

        __m256i q4 = vdot::nibbles_to_bytes_32(blk + 4);
        __m256i q8 = _mm256_loadu_si256((const __m256i*)act[i].qs);

        __m256i p16 = _mm256_maddubs_epi16(q4, q8);
        __m256i p32 = _mm256_madd_epi16(p16, ones_epi16);

        __m256i q8_sum16 = _mm256_maddubs_epi16(ones_epi8, q8);
        __m256i q8_sum32 = _mm256_madd_epi16(q8_sum16, ones_epi16);
        int32_t q8_total = vdot::hsum_i32(q8_sum32);

        __m256 sc = _mm256_set1_ps(d_val * scale_a);
        acc_d = _mm256_fmadd_ps(sc, _mm256_cvtepi32_ps(p32), acc_d);
        acc_m += m_val * scale_a * (float)q8_total;
    }

    return vdot::hsum_ps256(acc_d) + acc_m;
}

// Q8_0 weight × Q8_0_act activation dot product.
// Both signed int8. maddubs requires unsigned × signed.
// Add 128 to weight bytes (XOR with sign bit), then correct.
static inline float vec_dot_q8_0_q8_0_avx2(const uint8_t* w_row,
                                             const block_q8_0_act* act,
                                             int nb) {
    constexpr int BLOCK_BYTES = 34;
    __m256 acc = _mm256_setzero_ps();
    const __m256i ones_epi8 = _mm256_set1_epi8(1);
    const __m256i ones_epi16 = _mm256_set1_epi16(1);
    const __m256i xor_128 = _mm256_set1_epi8(static_cast<char>(0x80));
    const __m256i mul_128 = _mm256_set1_epi32(128);

    for (int i = 0; i < nb; ++i) {
        const uint8_t* blk = w_row + (size_t)i * BLOCK_BYTES;
        _mm_prefetch((const char*)(w_row + (size_t)(i + 4) * BLOCK_BYTES), _MM_HINT_T0);
        uint16_t scale_bits;
        memcpy(&scale_bits, blk, 2);
        float combined_scale = vdot::fp16_to_fp32(scale_bits) * act[i].d;

        __m256i q8_w = _mm256_loadu_si256((const __m256i*)(blk + 2));
        __m256i q8_w_unsigned = _mm256_xor_si256(q8_w, xor_128);
        __m256i q8_act = _mm256_loadu_si256((const __m256i*)act[i].qs);

        __m256i p16 = _mm256_maddubs_epi16(q8_w_unsigned, q8_act);
        __m256i p32 = _mm256_madd_epi16(p16, ones_epi16);

        __m256i q8_sum16 = _mm256_maddubs_epi16(ones_epi8, q8_act);
        __m256i q8_sum32 = _mm256_madd_epi16(q8_sum16, ones_epi16);
        __m256i correction = _mm256_mullo_epi32(q8_sum32, mul_128);
        __m256i corrected = _mm256_sub_epi32(p32, correction);

        __m256 sc = _mm256_set1_ps(combined_scale);
        acc = _mm256_fmadd_ps(sc, _mm256_cvtepi32_ps(corrected), acc);
    }

    return vdot::hsum_ps256(acc);
}

// Q5_0 weight × Q8_0_act activation dot product.
// Q5_0 value = (5bit_value - 16) * scale, 5bit = nibble | (bit5 << 4)
// Correction: -16 * sum(q8_act) per block
static inline float vec_dot_q5_0_q8_0_avx2(const uint8_t* w_row,
                                             const block_q8_0_act* act,
                                             int nb) {
    constexpr int BLOCK_BYTES = 22;  // d[2] + qh[4] + qs[16]
    __m256 acc = _mm256_setzero_ps();
    const __m256i ones_epi8 = _mm256_set1_epi8(1);
    const __m256i ones_epi16 = _mm256_set1_epi16(1);
    const __m256i sixteen_epi32 = _mm256_set1_epi32(16);

    for (int i = 0; i < nb; ++i) {
        const uint8_t* blk = w_row + (size_t)i * BLOCK_BYTES;
        _mm_prefetch((const char*)(w_row + (size_t)(i + 4) * BLOCK_BYTES), _MM_HINT_T0);
        uint16_t scale_bits;
        memcpy(&scale_bits, blk, 2);
        float combined_scale = vdot::fp16_to_fp32(scale_bits) * act[i].d;

        // blk layout: d[2], qh[4], qs[16]
        __m256i q5 = vdot::nibbles5_to_bytes_32(blk + 6, blk + 2);  // unsigned [0..31]
        __m256i q8 = _mm256_loadu_si256((const __m256i*)act[i].qs);

        __m256i p16 = _mm256_maddubs_epi16(q5, q8);
        __m256i p32 = _mm256_madd_epi16(p16, ones_epi16);

        __m256i q8_sum16 = _mm256_maddubs_epi16(ones_epi8, q8);
        __m256i q8_sum32 = _mm256_madd_epi16(q8_sum16, ones_epi16);
        __m256i correction = _mm256_mullo_epi32(q8_sum32, sixteen_epi32);
        __m256i corrected = _mm256_sub_epi32(p32, correction);

        __m256 sc = _mm256_set1_ps(combined_scale);
        acc = _mm256_fmadd_ps(sc, _mm256_cvtepi32_ps(corrected), acc);
    }

    return vdot::hsum_ps256(acc);
}

// Q5_1 weight × Q8_0_act activation dot product.
// Q5_1 value = 5bit_value * d + m
static inline float vec_dot_q5_1_q8_0_avx2(const uint8_t* w_row,
                                             const block_q8_0_act* act,
                                             int nb) {
    constexpr int BLOCK_BYTES = 24;  // d[2] + m[2] + qh[4] + qs[16]
    __m256 acc_d = _mm256_setzero_ps();
    float acc_m = 0.0f;
    const __m256i ones_epi8 = _mm256_set1_epi8(1);
    const __m256i ones_epi16 = _mm256_set1_epi16(1);

    for (int i = 0; i < nb; ++i) {
        const uint8_t* blk = w_row + (size_t)i * BLOCK_BYTES;
        _mm_prefetch((const char*)(w_row + (size_t)(i + 4) * BLOCK_BYTES), _MM_HINT_T0);
        uint16_t d_bits, m_bits;
        memcpy(&d_bits, blk, 2);
        memcpy(&m_bits, blk + 2, 2);
        float d_val = vdot::fp16_to_fp32(d_bits);
        float m_val = vdot::fp16_to_fp32(m_bits);
        float scale_a = act[i].d;

        // blk layout: d[2], m[2], qh[4], qs[16]
        __m256i q5 = vdot::nibbles5_to_bytes_32(blk + 8, blk + 4);
        __m256i q8 = _mm256_loadu_si256((const __m256i*)act[i].qs);

        __m256i p16 = _mm256_maddubs_epi16(q5, q8);
        __m256i p32 = _mm256_madd_epi16(p16, ones_epi16);

        __m256i q8_sum16 = _mm256_maddubs_epi16(ones_epi8, q8);
        __m256i q8_sum32 = _mm256_madd_epi16(q8_sum16, ones_epi16);
        int32_t q8_total = vdot::hsum_i32(q8_sum32);

        __m256 sc = _mm256_set1_ps(d_val * scale_a);
        acc_d = _mm256_fmadd_ps(sc, _mm256_cvtepi32_ps(p32), acc_d);
        acc_m += m_val * scale_a * (float)q8_total;
    }

    return vdot::hsum_ps256(acc_d) + acc_m;
}

#endif  // USE_AVX2

}  // namespace cpu
}  // namespace forge
