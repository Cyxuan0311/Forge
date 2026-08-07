#pragma once
// x86 AVX2 RMS norm kernel (FORGE_ARCH_X86 + USE_AVX2).
// Processes one row: rms = 1/sqrt(mean(x^2) + eps), then out = x * rms * weight.

#ifdef USE_AVX2
#    include <immintrin.h>
#endif

namespace forge {
namespace cpu {

#ifdef USE_AVX2
// rms_norm_row_f32: Normalize one row of x with optional weight.
// x_row, w_row, o_row are all length cols floats.
// If w_row is nullptr, weight is treated as 1.0.
static inline void rms_norm_row_f32(const float* x_row, const float* w_row, float* o_row,
                                    int cols, float eps) {
    __m256 sum_sq_v = _mm256_setzero_ps();
    int c = 0;
    for (; c + 8 <= cols; c += 8) {
        __m256 xv = _mm256_loadu_ps(x_row + c);
        sum_sq_v = _mm256_fmadd_ps(xv, xv, sum_sq_v);
    }
    // Horizontal sum
    __m128 hi128 = _mm256_extractf128_ps(sum_sq_v, 1);
    __m128 lo128 = _mm256_castps256_ps128(sum_sq_v);
    __m128 sum128 = _mm_add_ps(lo128, hi128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float sum_sq = _mm_cvtss_f32(sum128);
    for (; c < cols; ++c) {
        float v = x_row[c];
        sum_sq += v * v;
    }
    float rms = 1.0f / std::sqrt(sum_sq / cols + eps);
    __m256 rms_v = _mm256_set1_ps(rms);
    c = 0;
    if (w_row) {
        for (; c + 8 <= cols; c += 8) {
            __m256 xv = _mm256_loadu_ps(x_row + c);
            __m256 wv = _mm256_loadu_ps(w_row + c);
            __m256 ov = _mm256_mul_ps(_mm256_mul_ps(xv, rms_v), wv);
            _mm256_storeu_ps(o_row + c, ov);
        }
        for (; c < cols; ++c) {
            o_row[c] = x_row[c] * rms * w_row[c];
        }
    } else {
        for (; c + 8 <= cols; c += 8) {
            __m256 xv = _mm256_loadu_ps(x_row + c);
            _mm256_storeu_ps(o_row + c, _mm256_mul_ps(xv, rms_v));
        }
        for (; c < cols; ++c) {
            o_row[c] = x_row[c] * rms;
        }
    }
}
#endif  // USE_AVX2

}  // namespace cpu
}  // namespace forge