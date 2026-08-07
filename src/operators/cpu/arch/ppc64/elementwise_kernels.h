#pragma once
// PowerPC64 VSX elementwise kernels.
// Provides vectorised add / mul / silu_mul / gelu_mul for float32 vectors.
// VSX uses C operator overloading (+, -, *, &, |, ^) directly on vector types.

#ifdef USE_VSX
#include <altivec.h>
#endif
#include <cmath>

namespace forge {
namespace cpu {

#ifdef USE_VSX

// ---- add_f32_vec ----
static inline void add_f32_vec(const float* a, const float* b, float* out, int n) {
    int i = 0;
    for (; i + 15 < n; i += 16) {
        __vector float a0 = vec_xl(0, (const float*)(a + i));
        __vector float a1 = vec_xl(0, (const float*)(a + i + 4));
        __vector float a2 = vec_xl(0, (const float*)(a + i + 8));
        __vector float a3 = vec_xl(0, (const float*)(a + i + 12));
        __vector float b0 = vec_xl(0, (const float*)(b + i));
        __vector float b1 = vec_xl(0, (const float*)(b + i + 4));
        __vector float b2 = vec_xl(0, (const float*)(b + i + 8));
        __vector float b3 = vec_xl(0, (const float*)(b + i + 12));
        vec_xst(a0 + b0, 0, (float*)(out + i));
        vec_xst(a1 + b1, 0, (float*)(out + i + 4));
        vec_xst(a2 + b2, 0, (float*)(out + i + 8));
        vec_xst(a3 + b3, 0, (float*)(out + i + 12));
    }
    for (; i + 3 < n; i += 4) {
        __vector float a0 = vec_xl(0, (const float*)(a + i));
        __vector float b0 = vec_xl(0, (const float*)(b + i));
        vec_xst(a0 + b0, 0, (float*)(out + i));
    }
    for (; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}

// ---- mul_f32_vec ----
static inline void mul_f32_vec(const float* a, const float* b, float* out, int n) {
    int i = 0;
    for (; i + 15 < n; i += 16) {
        __vector float a0 = vec_xl(0, (const float*)(a + i));
        __vector float a1 = vec_xl(0, (const float*)(a + i + 4));
        __vector float a2 = vec_xl(0, (const float*)(a + i + 8));
        __vector float a3 = vec_xl(0, (const float*)(a + i + 12));
        __vector float b0 = vec_xl(0, (const float*)(b + i));
        __vector float b1 = vec_xl(0, (const float*)(b + i + 4));
        __vector float b2 = vec_xl(0, (const float*)(b + i + 8));
        __vector float b3 = vec_xl(0, (const float*)(b + i + 12));
        vec_xst(a0 * b0, 0, (float*)(out + i));
        vec_xst(a1 * b1, 0, (float*)(out + i + 4));
        vec_xst(a2 * b2, 0, (float*)(out + i + 8));
        vec_xst(a3 * b3, 0, (float*)(out + i + 12));
    }
    for (; i + 3 < n; i += 4) {
        __vector float a0 = vec_xl(0, (const float*)(a + i));
        __vector float b0 = vec_xl(0, (const float*)(b + i));
        vec_xst(a0 * b0, 0, (float*)(out + i));
    }
    for (; i < n; ++i) {
        out[i] = a[i] * b[i];
    }
}

// ---- Cephes-style exp approximation (VSX) ----
// exp(x) = 2^(x/ln2), separate integer and fractional parts.
// Uses 6th-order polynomial for 2^f, f in [0,1).
// Max relative error < 1e-6 for |x| <= 20. Matches the x86 AVX2 / ARM64 NEON Cephes path.
static inline __vector float cephes_exp_f32_vsx(__vector float x) {
    const float log2e = 1.4426950408889634f;
    const float c1    = 0.6931471805599453f;
    const float c2    = 0.2402265069591007f;
    const float c3    = 0.05549525927235975f;
    const float c4    = 0.009608917886916534f;
    const float c5    = 0.001333355814681543f;
    const float c6    = 0.0001540353039338152f;

    __vector float xl = x * vec_splats(log2e);
    // floor(xl) via trunc + adjust for negative non-integers
    __vector signed int n = vec_cts(xl, 0);
    __vector float nf = vec_ctf(n, 0);
    __vector unsigned int lt_mask = (__vector unsigned int)vec_cmplt(xl, nf);
    n = n - (__vector signed int)(lt_mask & vec_splats((unsigned int)1));
    __vector float f = xl - vec_ctf(n, 0);   // f in [0, 1)

    // 2^f ≈ 1 + f*(c1 + f*(c2 + f*(c3 + f*(c4 + f*(c5 + f*c6)))))
    __vector float poly = vec_splats(c6);
    poly = vec_madd(poly, f, vec_splats(c5));
    poly = vec_madd(poly, f, vec_splats(c4));
    poly = vec_madd(poly, f, vec_splats(c3));
    poly = vec_madd(poly, f, vec_splats(c2));
    poly = vec_madd(poly, f, vec_splats(c1));
    poly = vec_madd(poly, f, vec_splats(1.0f));

    // Scale by 2^n: add 127 (bias) and shift into exponent bits
    __vector unsigned int exp_bits = vec_sl(
        (__vector unsigned int)(n + vec_splats((signed int)127)),
        vec_splats((unsigned int)23));
    return poly * (__vector float)exp_bits;
}

// ---- silu_mul_f32_vec: out[i] = silu(gate[i]) * up[i] ----
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
// Uses Cephes-style exp approximation (max error < 1e-6).
static inline void silu_mul_f32_vec(const float* gate, const float* up, float* out, int n) {
    __vector float one = vec_splats(1.0f);
    int i = 0;
    for (; i + 3 < n; i += 4) {
        __vector float x = vec_xl(0, (const float*)(gate + i));
        __vector float u = vec_xl(0, (const float*)(up + i));

        // sigmoid(x) = 1 / (1 + exp(-x))
        __vector float exp_neg = cephes_exp_f32_vsx(-x);
        __vector float sigmoid = one / (one + exp_neg);

        __vector float silu_v = x * sigmoid;
        vec_xst(silu_v * u, 0, (float*)(out + i));
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

    __vector float v_sqrt_2_over_pi = vec_splats(sqrt_2_over_pi);
    __vector float v_coeff = vec_splats(coeff);
    __vector float v_half = vec_splats(half);
    __vector float one = vec_splats(1.0f);

    int i = 0;
    for (; i + 3 < n; i += 4) {
        __vector float x = vec_xl(0, (const float*)(gate + i));
        __vector float u = vec_xl(0, (const float*)(up + i));

        // z = sqrt(2/pi) * (x + 0.044715*x^3)
        __vector float x2 = x * x;
        __vector float x3 = x2 * x;
        __vector float inner = vec_madd(x3, v_coeff, x);
        __vector float z = v_sqrt_2_over_pi * inner;

        // tanh(z) = (exp(2z) - 1) / (exp(2z) + 1)
        __vector float two_z = z + z;
        __vector float e2z = cephes_exp_f32_vsx(two_z);
        __vector float tanh_z = (e2z - one) / (e2z + one);

        // gelu = 0.5 * x * (1 + tanh(z))
        __vector float gelu_val = v_half * (x * (tanh_z + one));

        vec_xst(gelu_val * u, 0, (float*)(out + i));
    }
    for (; i < n; ++i) {
        float x = gate[i];
        float gelu_val = 0.5f * x * (1.0f + std::tanh(sqrt_2_over_pi * (x + coeff * x * x * x)));
        out[i] = gelu_val * up[i];
    }
}

#endif // USE_VSX

}  // namespace cpu
}  // namespace forge
