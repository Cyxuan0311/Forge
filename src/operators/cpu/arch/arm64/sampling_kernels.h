#pragma once
// ARM64 NEON sampling kernels.
// Provides vectorised argmax, max, softcap, exp-and-sum, scale-normalize
// for the greedy sampling pipeline.
//
// Uses fast-exp polynomial approximation identical to the x86 sampler path
// (max error ~1.5%), and Cephes-style exp-based tanh for softcap.
// Numerically distinct from elementwise kernels — must NOT be merged.

#ifdef USE_NEON
#include <arm_neon.h>
#endif
#include <cmath>
#include <algorithm>

namespace forge {
namespace cpu {

#ifdef USE_NEON

// ---- vector_tanh: tanh(z) = (exp(2z)-1)/(exp(2z)+1) via cephes_exp_f32_neon ----
// (cephes_exp_f32_neon is defined in elementwise_kernels.h, included via kernels.h)
static inline float32x4_t vector_tanh_neon(float32x4_t z) {
    float32x4_t two_z = vaddq_f32(z, z);
    float32x4_t e2z = cephes_exp_f32_neon(two_z);
    float32x4_t one = vdupq_n_f32(1.0f);
    return vdivq_f32(vsubq_f32(e2z, one), vaddq_f32(e2z, one));
}

// ---- softcap_and_argmax_f32 ----
// Applies tanh softcap in-place, returns index of max.
static inline int softcap_and_argmax_f32(float* logits, int n, float cap) {
    float inv_cap = 1.0f / cap;
    float best_val = -__builtin_huge_valf();
    int best = 0;

    int i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(logits + i);
        float32x4_t z = vmulq_f32(v, vdupq_n_f32(inv_cap));

        // tanh(z) = (exp(2z)-1)/(exp(2z)+1) via Cephes exp
        float32x4_t tanh_z = vector_tanh_neon(z);

        float32x4_t result = vmulq_f32(tanh_z, vdupq_n_f32(cap));
        vst1q_f32(logits + i, result);

        // Find max within this vector (scalar compare)
        float vals[4];
        vst1q_f32(vals, result);
        for (int j = 0; j < 4; ++j) {
            if (vals[j] > best_val) { best_val = vals[j]; best = i + j; }
        }
    }
    for (; i < n; ++i) {
        logits[i] = std::tanh(logits[i] / cap) * cap;
        if (logits[i] > best_val) { best_val = logits[i]; best = i; }
    }
    return best;
}

// ---- argmax_f32 ----
static inline int argmax_f32(const float* data, int n) {
    int best = 0;
    float best_val = data[0];

    int i = 1;
    // Track max index across NEON vectors
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        float max_v = vmaxvq_f32(v);
        if (max_v > best_val) {
            // Find the exact index within this vector
            float vals[4];
            vst1q_f32(vals, v);
            for (int j = 0; j < 4; ++j) {
                if (vals[j] > best_val) { best_val = vals[j]; best = i + j; }
            }
        }
    }
    for (; i < n; ++i) {
        if (data[i] > best_val) { best_val = data[i]; best = i; }
    }
    return best;
}

// ---- max_f32 ----
static inline float max_f32(const float* data, int n) {
    float32x4_t max_v = vdupq_n_f32(data[0]);
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        max_v = vmaxq_f32(max_v, v);
    }
    float result = vmaxvq_f32(max_v);
    for (; i < n; ++i) {
        if (data[i] > result) result = data[i];
    }
    return result;
}

// ---- softcap_and_max_f32 ----
static inline float softcap_and_max_f32(float* data, int n, float cap) {
    float inv_cap = 1.0f / cap;
    float best_val = data[0];
    // Apply softcap to first element
    data[0] = std::tanh(data[0] / cap) * cap;
    best_val = data[0];

    int i = 1;
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        float32x4_t z = vmulq_f32(v, vdupq_n_f32(inv_cap));
        float32x4_t tanh_z = vector_tanh_neon(z);
        float32x4_t result = vmulq_f32(tanh_z, vdupq_n_f32(cap));
        vst1q_f32(data + i, result);

        float vals[4];
        vst1q_f32(vals, result);
        for (int j = 0; j < 4; ++j) {
            if (vals[j] > best_val) best_val = vals[j];
        }
    }
    for (; i < n; ++i) {
        data[i] = std::tanh(data[i] / cap) * cap;
        if (data[i] > best_val) best_val = data[i];
    }
    return best_val;
}

// ---- exp_and_sum_f32 (fast-exp approximation, ~1.5% error) ----
// exp(x) approximated as 2^f where integer part is separated:
//   2^f ≈ 1 + f*(c1 + f*(c2 + f*c3))  with c1=0.693147, c2=0.240227, c3=0.055504
// Matches the x86 AVX2 fast-exp sampler path exactly.
static inline float exp_and_sum_f32(const float* data, float* out, int n, float max_val, float inv_temp) {
    const float log2e = 1.4426950408889634f;
    const float c1 = 0.6931471805599453f;
    const float c2 = 0.2402265069591006f;
    const float c3 = 0.05550410866482158f;

    float32x4_t sum_all = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t d = vld1q_f32(data + i);
        // x = (d - max_val) * inv_temp * log2(e)
        float32x4_t x = vmulq_f32(vsubq_f32(d, vdupq_n_f32(max_val)),
                                  vdupq_n_f32(inv_temp * log2e));
        // Floor to get integer exponent
        int32x4_t x_int = vcvtq_s32_f32(x);
        // Fractional part f = x - floor(x) in [0, 1)
        float32x4_t f = vsubq_f32(x, vcvtq_f32_s32(x_int));
        // Polynomial: 1 + f*(c1 + f*(c2 + f*c3))
        float32x4_t poly = vdupq_n_f32(c3);
        poly = vmlaq_f32(vdupq_n_f32(c2), poly, f);
        poly = vmlaq_f32(vdupq_n_f32(c1), poly, f);
        poly = vmlaq_f32(vdupq_n_f32(1.0f), poly, f);
        // Scale by 2^int: ldexpf via integer add to exponent
        int32x4_t exp_bias = vshlq_n_s32(x_int, 23);
        float32x4_t result = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(poly), exp_bias));

        vst1q_f32(out + i, result);
        sum_all = vaddq_f32(sum_all, result);
    }

    float sum = vaddvq_f32(sum_all);
    for (; i < n; ++i) {
        float x = (data[i] - max_val) * inv_temp * log2e;
        int x_int = (int)x;
        float f = x - (float)x_int;
        float poly = 1.0f + f * (c1 + f * (c2 + f * c3));
        // Scale by 2^x_int using ldexpf
        int exp_bits = x_int << 23;
        float poly_scaled;
        std::memcpy(&poly_scaled, &exp_bits, sizeof(float));
        // TRICK: poly * 2^x_int via float addition to exponent
        // The poly is already ~1.0, so we do: reinterpret cast + integer add
        int poly_bits;
        std::memcpy(&poly_bits, &poly, sizeof(int));
        poly_bits += exp_bits;
        std::memcpy(&poly_scaled, &poly_bits, sizeof(float));
        out[i] = poly_scaled;
        sum += poly_scaled;
    }
    return sum;
}

// ---- scale_normalize_f32 ----
static inline void scale_normalize_f32(float* data, int n, float inv_sum) {
    float32x4_t s = vdupq_n_f32(inv_sum);
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
        data[i] *= inv_sum;
    }
}

#endif // USE_NEON

}  // namespace cpu
}  // namespace forge
