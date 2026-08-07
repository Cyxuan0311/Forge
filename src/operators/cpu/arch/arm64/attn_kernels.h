#pragma once
// ARM64 NEON attention building-block kernels.
// Provides dot product, scale, and fused-multiply-add for online-softmax
// attention (both decode and prefill paths).

#ifdef USE_NEON
#include <arm_neon.h>
#endif

namespace forge {
namespace cpu {

#ifdef USE_NEON

// Horizontal sum of a float32x4_t (ARMv8+ vaddvq_f32).
static inline float hsum_f32(float32x4_t v) {
    return vaddvq_f32(v);
}

// Dot product of two float32 vectors.
// Dual-accumulator pattern for ILP (instruction-level parallelism).
static inline float dot_f32(const float* a, const float* b, int n) {
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);

    int i = 0;
    for (; i + 15 < n; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t a3 = vld1q_f32(a + i + 12);
        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        float32x4_t b2 = vld1q_f32(b + i + 8);
        float32x4_t b3 = vld1q_f32(b + i + 12);
        sum0 = vmlaq_f32(sum0, a0, b0);
        sum1 = vmlaq_f32(sum1, a1, b1);
        sum2 = vmlaq_f32(sum2, a2, b2);
        sum3 = vmlaq_f32(sum3, a3, b3);
    }
    for (; i + 3 < n; i += 4) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t b0 = vld1q_f32(b + i);
        sum0 = vmlaq_f32(sum0, a0, b0);
    }

    sum0 = vaddq_f32(sum0, sum1);
    sum2 = vaddq_f32(sum2, sum3);
    sum0 = vaddq_f32(sum0, sum2);
    float total = vaddvq_f32(sum0);

    // Scalar tail
    for (; i < n; ++i) {
        total += a[i] * b[i];
    }
    return total;
}

// Scale a float32 vector in-place: data[i] *= scale.
static inline void scale_f32(float* data, int n, float scale) {
    float32x4_t s = vdupq_n_f32(scale);
    int i = 0;
    for (; i + 15 < n; i += 16) {
        float32x4_t d0 = vld1q_f32(data + i);
        float32x4_t d1 = vld1q_f32(data + i + 4);
        float32x4_t d2 = vld1q_f32(data + i + 8);
        float32x4_t d3 = vld1q_f32(data + i + 12);
        vst1q_f32(data + i,      vmulq_f32(d0, s));
        vst1q_f32(data + i + 4,  vmulq_f32(d1, s));
        vst1q_f32(data + i + 8,  vmulq_f32(d2, s));
        vst1q_f32(data + i + 12, vmulq_f32(d3, s));
    }
    for (; i + 3 < n; i += 4) {
        float32x4_t d = vld1q_f32(data + i);
        vst1q_f32(data + i, vmulq_f32(d, s));
    }
    for (; i < n; ++i) {
        data[i] *= scale;
    }
}

// Fused multiply-add: acc[i] += weight * src[i].
static inline void fmadd_f32(float* acc, const float* src, int n, float weight) {
    float32x4_t w = vdupq_n_f32(weight);
    int i = 0;
    for (; i + 15 < n; i += 16) {
        float32x4_t s0 = vld1q_f32(src + i);
        float32x4_t s1 = vld1q_f32(src + i + 4);
        float32x4_t s2 = vld1q_f32(src + i + 8);
        float32x4_t s3 = vld1q_f32(src + i + 12);
        float32x4_t a0 = vld1q_f32(acc + i);
        float32x4_t a1 = vld1q_f32(acc + i + 4);
        float32x4_t a2 = vld1q_f32(acc + i + 8);
        float32x4_t a3 = vld1q_f32(acc + i + 12);
        vst1q_f32(acc + i,      vmlaq_f32(a0, s0, w));
        vst1q_f32(acc + i + 4,  vmlaq_f32(a1, s1, w));
        vst1q_f32(acc + i + 8,  vmlaq_f32(a2, s2, w));
        vst1q_f32(acc + i + 12, vmlaq_f32(a3, s3, w));
    }
    for (; i + 3 < n; i += 4) {
        float32x4_t s = vld1q_f32(src + i);
        float32x4_t a = vld1q_f32(acc + i);
        vst1q_f32(acc + i, vmlaq_f32(a, s, w));
    }
    for (; i < n; ++i) {
        acc[i] += weight * src[i];
    }
}

#endif // USE_NEON

}  // namespace cpu
}  // namespace forge
