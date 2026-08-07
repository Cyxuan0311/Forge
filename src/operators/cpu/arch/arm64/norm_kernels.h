#pragma once
// ARM64 NEON RMS normalization kernel.
// Processes one row: o_row = x_row * rms * w_row (optional weight).
// rms = 1/sqrt(sum(x^2)/cols + eps).

#ifdef USE_NEON
#include <arm_neon.h>
#endif
#include <cmath>

namespace forge {
namespace cpu {

#ifdef USE_NEON

static inline void rms_norm_row_f32(const float* x_row, const float* w_row, float* o_row,
                                    int cols, float eps) {
    // Compute sum of squares using NEON
    float32x4_t sum_sq0 = vdupq_n_f32(0.0f);
    float32x4_t sum_sq1 = vdupq_n_f32(0.0f);
    float32x4_t sum_sq2 = vdupq_n_f32(0.0f);
    float32x4_t sum_sq3 = vdupq_n_f32(0.0f);

    int c = 0;
    for (; c + 15 < cols; c += 16) {
        float32x4_t v0 = vld1q_f32(x_row + c);
        float32x4_t v1 = vld1q_f32(x_row + c + 4);
        float32x4_t v2 = vld1q_f32(x_row + c + 8);
        float32x4_t v3 = vld1q_f32(x_row + c + 12);
        sum_sq0 = vmlaq_f32(sum_sq0, v0, v0);
        sum_sq1 = vmlaq_f32(sum_sq1, v1, v1);
        sum_sq2 = vmlaq_f32(sum_sq2, v2, v2);
        sum_sq3 = vmlaq_f32(sum_sq3, v3, v3);
    }
    for (; c + 3 < cols; c += 4) {
        float32x4_t v = vld1q_f32(x_row + c);
        sum_sq0 = vmlaq_f32(sum_sq0, v, v);
    }

    // Horizontal reduction of 4 accumulators
    sum_sq0 = vaddq_f32(sum_sq0, sum_sq1);
    sum_sq2 = vaddq_f32(sum_sq2, sum_sq3);
    sum_sq0 = vaddq_f32(sum_sq0, sum_sq2);
    float sum_sq = vaddvq_f32(sum_sq0);

    // Scalar tail
    for (; c < cols; ++c) {
        float v = x_row[c];
        sum_sq += v * v;
    }

    float rms = 1.0f / std::sqrt(sum_sq / cols + eps);

    // Apply norm with optional weight
    if (w_row) {
        int c2 = 0;
        for (; c2 + 15 < cols; c2 += 16) {
            float32x4_t x0 = vld1q_f32(x_row + c2);
            float32x4_t x1 = vld1q_f32(x_row + c2 + 4);
            float32x4_t x2 = vld1q_f32(x_row + c2 + 8);
            float32x4_t x3 = vld1q_f32(x_row + c2 + 12);
            float32x4_t w0 = vld1q_f32(w_row + c2);
            float32x4_t w1 = vld1q_f32(w_row + c2 + 4);
            float32x4_t w2 = vld1q_f32(w_row + c2 + 8);
            float32x4_t w3 = vld1q_f32(w_row + c2 + 12);
            float32x4_t s = vdupq_n_f32(rms);
            vst1q_f32(o_row + c2,      vmulq_f32(vmulq_f32(x0, s), w0));
            vst1q_f32(o_row + c2 + 4,  vmulq_f32(vmulq_f32(x1, s), w1));
            vst1q_f32(o_row + c2 + 8,  vmulq_f32(vmulq_f32(x2, s), w2));
            vst1q_f32(o_row + c2 + 12, vmulq_f32(vmulq_f32(x3, s), w3));
        }
        for (; c2 + 3 < cols; c2 += 4) {
            float32x4_t x = vld1q_f32(x_row + c2);
            float32x4_t w = vld1q_f32(w_row + c2);
            float32x4_t s = vdupq_n_f32(rms);
            vst1q_f32(o_row + c2, vmulq_f32(vmulq_f32(x, s), w));
        }
        for (; c2 < cols; ++c2) {
            o_row[c2] = x_row[c2] * rms * w_row[c2];
        }
    } else {
        int c2 = 0;
        float32x4_t s = vdupq_n_f32(rms);
        for (; c2 + 15 < cols; c2 += 16) {
            float32x4_t x0 = vld1q_f32(x_row + c2);
            float32x4_t x1 = vld1q_f32(x_row + c2 + 4);
            float32x4_t x2 = vld1q_f32(x_row + c2 + 8);
            float32x4_t x3 = vld1q_f32(x_row + c2 + 12);
            vst1q_f32(o_row + c2,      vmulq_f32(x0, s));
            vst1q_f32(o_row + c2 + 4,  vmulq_f32(x1, s));
            vst1q_f32(o_row + c2 + 8,  vmulq_f32(x2, s));
            vst1q_f32(o_row + c2 + 12, vmulq_f32(x3, s));
        }
        for (; c2 + 3 < cols; c2 += 4) {
            float32x4_t x = vld1q_f32(x_row + c2);
            vst1q_f32(o_row + c2, vmulq_f32(x, s));
        }
        for (; c2 < cols; ++c2) {
            o_row[c2] = x_row[c2] * rms;
        }
    }
}

#endif // USE_NEON

}  // namespace cpu
}  // namespace forge
