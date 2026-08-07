#pragma once
// x86 AVX2 attention kernels (FORGE_ARCH_X86 + USE_AVX2).
// hsum_f32 / dot_f32 — extracted from attention.cpp.
// scale_f32 / fmadd_f32 — building blocks for online-softmax accumulate.

#ifdef USE_AVX2
#    include <immintrin.h>
#endif

namespace forge {
namespace cpu {

#ifdef USE_AVX2

// ---- hsum_f32: horizontal sum of 8 floats ----
static inline float hsum_f32(__m256 v) {
    __m128 hi128 = _mm256_extractf128_ps(v, 1);
    __m128 lo128 = _mm256_castps256_ps128(v);
    __m128 sum128 = _mm_add_ps(lo128, hi128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}

// ---- dot_f32: dot product of two float vectors ----
static inline float dot_f32(const float* a, const float* b, int n) {
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
    float sum = hsum_f32(acc);
    for (; i < n; ++i)
        sum += a[i] * b[i];
    return sum;
}

// ---- scale_f32: out[i] *= scale for i in [0, n) ----
static inline void scale_f32(float* data, int n, float scale) {
    __m256 r = _mm256_set1_ps(scale);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a = _mm256_loadu_ps(data + i);
        _mm256_storeu_ps(data + i, _mm256_mul_ps(a, r));
    }
    for (; i < n; ++i)
        data[i] *= scale;
}

// ---- fmadd_f32: acc[i] += src[i] * weight for i in [0, n) ----
static inline void fmadd_f32(float* acc, const float* src, int n, float weight) {
    __m256 w_vec = _mm256_set1_ps(weight);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a = _mm256_loadu_ps(acc + i);
        __m256 vr = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(acc + i, _mm256_fmadd_ps(w_vec, vr, a));
    }
    for (; i < n; ++i)
        acc[i] += weight * src[i];
}

#endif  // USE_AVX2

}  // namespace cpu
}  // namespace forge