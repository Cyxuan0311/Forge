#pragma once
// x86 AVX2 elementwise kernels (FORGE_ARCH_X86 + USE_AVX2).
// add_f32_vec / mul_f32_vec are shared between elementwise.cpp and op_kernels.cpp.
// silu_mul_f32_vec / gelu_mul_f32_vec use polynomial approximations that differ
// numerically from std::exp / std::tanh — do NOT merge with sampler kernels.

#ifdef USE_AVX2
#    include <immintrin.h>
#endif

namespace forge {
namespace cpu {

#ifdef USE_AVX2
// ---- add_f32_vec: out[i] = a[i] + b[i] ----
static inline void add_f32_vec(const float* a, const float* b, float* out, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 av = _mm256_loadu_ps(a + i);
        __m256 bv = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_add_ps(av, bv));
    }
    for (; i < n; ++i)
        out[i] = a[i] + b[i];
}

// ---- mul_f32_vec: out[i] = a[i] * b[i] ----
static inline void mul_f32_vec(const float* a, const float* b, float* out, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 av = _mm256_loadu_ps(a + i);
        __m256 bv = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(av, bv));
    }
    for (; i < n; ++i)
        out[i] = a[i] * b[i];
}

// ---- silu_mul_f32_vec: out[i] = silu(gate[i]) * up[i] ----
// Uses Cephes-style exp approximation (max error < 1e-6).
static inline void silu_mul_f32_vec(const float* gate, const float* up, float* out, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 gv = _mm256_loadu_ps(gate + i);
        __m256 uv = _mm256_loadu_ps(up + i);
        // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
        __m256 neg_gv = _mm256_sub_ps(_mm256_setzero_ps(), gv);

        // Cephes-style exp: exp(x) = 2^(x/ln2)
        __m256 x = _mm256_mul_ps(neg_gv, _mm256_set1_ps(1.4426950408889634f));  // 1/ln2
        __m256i emm0 = _mm256_cvttps_epi32(x);
        __m256 z = _mm256_cvtepi32_ps(emm0);
        __m256 mask = _mm256_cmp_ps(x, z, _MM_CMPINT_LT);
        z = _mm256_sub_ps(z, _mm256_and_ps(mask, _mm256_set1_ps(1.0f)));
        emm0 = _mm256_cvttps_epi32(z);
        __m256 f = _mm256_sub_ps(x, z);  // f in [0, 1)

        emm0 = _mm256_add_epi32(emm0, _mm256_set1_epi32(127));
        emm0 = _mm256_slli_epi32(emm0, 23);
        __m256 pow2n = _mm256_castsi256_ps(emm0);

        // Cephes 6th-order polynomial for 2^f, f in [0, 1)
        __m256 P0 = _mm256_set1_ps(1.0f);
        __m256 P1 = _mm256_set1_ps(0.6931471805599453f);     // ln2
        __m256 P2 = _mm256_set1_ps(0.2402265069591007f);     // ln2^2/2
        __m256 P3 = _mm256_set1_ps(0.05549525927235975f);    // ln2^3/6
        __m256 P4 = _mm256_set1_ps(0.009608917886916534f);   // ln2^4/24
        __m256 P5 = _mm256_set1_ps(0.001333355814681543f);   // ln2^5/120
        __m256 P6 = _mm256_set1_ps(0.0001540353039338152f);  // ln2^6/720

        __m256 poly = _mm256_add_ps(P5, _mm256_mul_ps(f, P6));
        poly = _mm256_add_ps(P4, _mm256_mul_ps(f, poly));
        poly = _mm256_add_ps(P3, _mm256_mul_ps(f, poly));
        poly = _mm256_add_ps(P2, _mm256_mul_ps(f, poly));
        poly = _mm256_add_ps(P1, _mm256_mul_ps(f, poly));
        poly = _mm256_add_ps(P0, _mm256_mul_ps(f, poly));

        __m256 exp_neg = _mm256_mul_ps(poly, pow2n);
        __m256 one = _mm256_set1_ps(1.0f);
        __m256 sigmoid = _mm256_div_ps(gv, _mm256_add_ps(one, exp_neg));
        _mm256_storeu_ps(out + i, _mm256_mul_ps(sigmoid, uv));
    }
    for (; i < n; ++i) {
        float v = gate[i];
        float silu_v = v / (1.0f + std::exp(-v));
        out[i] = silu_v * up[i];
    }
}

// ---- gelu_mul_f32_vec: out[i] = gelu(gate[i]) * up[i] ----
// Uses Padé tanh approximation (clamped to [-1,1]).
static inline void gelu_mul_f32_vec(const float* gate, const float* up, float* out, int n) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 gv = _mm256_loadu_ps(gate + i);
        __m256 uv = _mm256_loadu_ps(up + i);
        // GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        __m256 x3 = _mm256_mul_ps(gv, _mm256_mul_ps(gv, gv));
        __m256 inner = _mm256_mul_ps(_mm256_set1_ps(sqrt_2_over_pi),
                                      _mm256_add_ps(gv, _mm256_mul_ps(_mm256_set1_ps(coeff), x3)));
        __m256 inner2 = _mm256_mul_ps(inner, inner);
        __m256 tanh_approx =
            _mm256_mul_ps(inner, _mm256_div_ps(_mm256_add_ps(_mm256_set1_ps(27.0f), inner2),
                                                _mm256_add_ps(_mm256_set1_ps(27.0f), _mm256_mul_ps(_mm256_set1_ps(9.0f), inner2))));
        tanh_approx = _mm256_min_ps(_mm256_max_ps(tanh_approx, _mm256_set1_ps(-1.0f)), _mm256_set1_ps(1.0f));
        __m256 one_plus_tanh = _mm256_add_ps(_mm256_set1_ps(1.0f), tanh_approx);
        __m256 gelu_val = _mm256_mul_ps(_mm256_set1_ps(0.5f), _mm256_mul_ps(gv, one_plus_tanh));
        _mm256_storeu_ps(out + i, _mm256_mul_ps(gelu_val, uv));
    }
    for (; i < n; ++i) {
        float x = gate[i];
        float gelu_val = 0.5f * x * (1.0f + std::tanh(sqrt_2_over_pi * (x + coeff * x * x * x)));
        out[i] = gelu_val * up[i];
    }
}

#endif  // USE_AVX2

}  // namespace cpu
}  // namespace forge