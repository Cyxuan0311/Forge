#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#ifdef USE_AVX2
#    include <immintrin.h>
#endif

namespace forge {
namespace cpu {

#ifdef USE_AVX2

static inline float hsum_avx2(__m256 v) {
    __m128 hi128 = _mm256_extractf128_ps(v, 1);
    __m128 lo128 = _mm256_castps256_ps128(v);
    __m128 sum128 = _mm_add_ps(lo128, hi128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}

static inline __m256 fp16_to_fp32_broadcast_avx2(uint16_t fp16_val) {
    return _mm256_cvtph_ps(_mm_set1_epi16(fp16_val));
}

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

#endif  // USE_AVX2

static inline float fp16_to_float_scalar(uint16_t h) {
    const uint32_t w = (uint32_t)h << 16;
    const uint32_t sign = w & 0x80000000u;
    const uint32_t two_w = w + w;
    const uint32_t exp_offset = 0xE0u << 23;
    const float exp_scale = 0x1.0p-112f;
    const float normalized_value = std::bit_cast<float>((two_w >> 4) + exp_offset) * exp_scale;
    const uint32_t magic_mask = 126u << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = std::bit_cast<float>((two_w >> 17) | magic_mask) - magic_bias;
    const uint32_t denormalized_cutoff = 1u << 27;
    const uint32_t result =
        sign | (two_w < denormalized_cutoff
                    ? std::bit_cast<uint32_t>(denormalized_value)
                    : std::bit_cast<uint32_t>(normalized_value));
    return std::bit_cast<float>(result);
}

}  // namespace cpu
}  // namespace forge
