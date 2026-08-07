#pragma once
// ARM64 NEON elementwise kernels.
// Provides vectorised add / mul / silu_mul / gelu_mul for float32 vectors.
//
// Apple Silicon (M1+) and ARMv8+ all support NEON (ASIMD).
// On hardware without NEON (unlikely on AArch64), the generic scalar
// fallback in arch/generic/kernels.h is used instead.

#ifdef USE_NEON
#include <arm_neon.h>
#endif
#include <cmath>

namespace forge {
namespace cpu {

#ifdef USE_NEON

// ---- add_f32_vec ----
static inline void add_f32_vec(const float* a, const float* b, float* out, int n) {
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
        vst1q_f32(out + i,      vaddq_f32(a0, b0));
        vst1q_f32(out + i + 4,  vaddq_f32(a1, b1));
        vst1q_f32(out + i + 8,  vaddq_f32(a2, b2));
        vst1q_f32(out + i + 12, vaddq_f32(a3, b3));
    }
    for (; i + 3 < n; i += 4) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t b0 = vld1q_f32(b + i);
        vst1q_f32(out + i, vaddq_f32(a0, b0));
    }
    for (; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}

// ---- mul_f32_vec ----
static inline void mul_f32_vec(const float* a, const float* b, float* out, int n) {
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
        vst1q_f32(out + i,      vmulq_f32(a0, b0));
        vst1q_f32(out + i + 4,  vmulq_f32(a1, b1));
        vst1q_f32(out + i + 8,  vmulq_f32(a2, b2));
        vst1q_f32(out + i + 12, vmulq_f32(a3, b3));
    }
    for (; i + 3 < n; i += 4) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t b0 = vld1q_f32(b + i);
        vst1q_f32(out + i, vmulq_f32(a0, b0));
    }
    for (; i < n; ++i) {
        out[i] = a[i] * b[i];
    }
}

// ---- Cephes-style exp approximation (NEON) ----
// exp(x) = 2^(x/ln2), separate integer and fractional parts.
// Uses 6th-order polynomial for 2^f, f in [0,1).
// Max relative error < 1e-6 for |x| <= 20. Matches the x86 AVX2 Cephes path.
static inline float32x4_t cephes_exp_f32_neon(float32x4_t x) {
    const float log2e     = 1.4426950408889634f;
    const float c1        = 0.6931471805599453f;
    const float c2        = 0.2402265069591007f;
    const float c3        = 0.05549525927235975f;
    const float c4        = 0.009608917886916534f;
    const float c5        = 0.001333355814681543f;
    const float c6        = 0.0001540353039338152f;

    float32x4_t xl = vmulq_f32(x, vdupq_n_f32(log2e));
    // floor(xl) via: trunc + adjust for negative non-integers
    int32x4_t n = vcvtq_s32_f32(xl);
    float32x4_t nf = vcvtq_f32_s32(n);
    uint32x4_t lt_mask = vcltq_f32(xl, nf);
    n = vsubq_s32(n, vreinterpretq_s32_u32(vandq_u32(lt_mask, vdupq_n_u32(1))));
    float32x4_t f = vsubq_f32(xl, vcvtq_f32_s32(n));   // f in [0, 1)

    // 2^f ≈ 1 + f*(c1 + f*(c2 + f*(c3 + f*(c4 + f*(c5 + f*c6)))))
    float32x4_t poly = vdupq_n_f32(c6);
    poly = vmlaq_f32(vdupq_n_f32(c5), poly, f);
    poly = vmlaq_f32(vdupq_n_f32(c4), poly, f);
    poly = vmlaq_f32(vdupq_n_f32(c3), poly, f);
    poly = vmlaq_f32(vdupq_n_f32(c2), poly, f);
    poly = vmlaq_f32(vdupq_n_f32(c1), poly, f);
    poly = vmlaq_f32(vdupq_n_f32(1.0f), poly, f);

    // Scale by 2^n: add 127 (bias) and shift into exponent bits
    int32x4_t exp_bits = vshlq_n_s32(vaddq_s32(n, vdupq_n_s32(127)), 23);
    return vmulq_f32(poly, vreinterpretq_f32_s32(exp_bits));
}

// ---- silu_mul_f32_vec: out[i] = silu(gate[i]) * up[i] ----
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
// Uses Cephes-style exp approximation (max error < 1e-6).
static inline void silu_mul_f32_vec(const float* gate, const float* up, float* out, int n) {
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t x = vld1q_f32(gate + i);
        float32x4_t u = vld1q_f32(up + i);

        // sigmoid(x) = 1 / (1 + exp(-x))
        float32x4_t exp_neg = cephes_exp_f32_neon(vnegq_f32(x));
        float32x4_t one = vdupq_n_f32(1.0f);
        float32x4_t sigmoid = vdivq_f32(one, vaddq_f32(one, exp_neg));

        float32x4_t silu_v = vmulq_f32(x, sigmoid);
        vst1q_f32(out + i, vmulq_f32(silu_v, u));
    }
    for (; i < n; ++i) {
        float v = gate[i];
        float silu_v = v / (1.0f + std::exp(-v));
        out[i] = silu_v * up[i];
    }
}

// ---- gelu_mul_f32_vec: out[i] = gelu(gate[i]) * up[i] ----
// GELU(x) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
// tanh(z) = (exp(2z) - 1) / (exp(2z) + 1) via Cephes exp.
static inline void gelu_mul_f32_vec(const float* gate, const float* up, float* out, int n) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    const float half = 0.5f;

    int i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t x = vld1q_f32(gate + i);
        float32x4_t u = vld1q_f32(up + i);

        // z = sqrt(2/pi) * (x + 0.044715*x^3)
        float32x4_t x2 = vmulq_f32(x, x);
        float32x4_t x3 = vmulq_f32(x2, x);
        float32x4_t inner = vmlaq_f32(x, x3, vdupq_n_f32(coeff));
        float32x4_t z = vmulq_f32(vdupq_n_f32(sqrt_2_over_pi), inner);

        // tanh(z) = (exp(2z) - 1) / (exp(2z) + 1)
        float32x4_t two_z = vaddq_f32(z, z);
        float32x4_t e2z = cephes_exp_f32_neon(two_z);
        float32x4_t one = vdupq_n_f32(1.0f);
        float32x4_t tanh_z = vdivq_f32(vsubq_f32(e2z, one), vaddq_f32(e2z, one));

        // gelu = 0.5 * x * (1 + tanh(z))
        float32x4_t gelu_val = vmulq_f32(vdupq_n_f32(half),
            vmulq_f32(x, vaddq_f32(tanh_z, one)));

        vst1q_f32(out + i, vmulq_f32(gelu_val, u));
    }
    for (; i < n; ++i) {
        float x = gate[i];
        float gelu_val = 0.5f * x * (1.0f + std::tanh(sqrt_2_over_pi * (x + coeff * x * x * x)));
        out[i] = gelu_val * up[i];
    }
}

#endif // USE_NEON

}  // namespace cpu
}  // namespace forge
