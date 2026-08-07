#pragma once
// x86 AVX2 sampling kernels (FORGE_ARCH_X86 + USE_AVX2).
// Extracted verbatim from sampler.cpp. The fast-exp and tanh approximations
// produce numerically different results from std::exp/std::tanh — do NOT
// merge with elementwise kernels.

#ifdef USE_AVX2
#    include <immintrin.h>
#endif

namespace forge {
namespace cpu {

#ifdef USE_AVX2

// ---- softcap_and_argmax_f32: apply tanh-softcap in-place, return argmax ----
// logits:  [in/out] modified in-place with softcapped values
// n:       number of elements
// cap:     softcapping threshold
// returns: index of maximum value
static inline int softcap_and_argmax_f32(float* logits, int n, float cap) {
    __m256 vcap = _mm256_set1_ps(cap);
    __m256 vdiv = _mm256_set1_ps(1.0f / cap);
    __m256 c27 = _mm256_set1_ps(27.0f);
    __m256 c9 = _mm256_set1_ps(9.0f);
    __m256 vminus_one = _mm256_set1_ps(-1.0f);
    __m256 vone = _mm256_set1_ps(1.0f);
    __m256 vmax = _mm256_set1_ps(-1e30f);
    __m256i vidx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(&logits[i]);
        __m256 x = _mm256_mul_ps(v, vdiv);
        __m256 x2 = _mm256_mul_ps(x, x);
        __m256 num = _mm256_add_ps(c27, x2);
        __m256 den = _mm256_add_ps(c27, _mm256_mul_ps(c9, x2));
        __m256 th = _mm256_mul_ps(x, _mm256_div_ps(num, den));
        th = _mm256_min_ps(_mm256_max_ps(th, vminus_one), vone);
        __m256 sc = _mm256_mul_ps(th, vcap);
        _mm256_storeu_ps(&logits[i], sc);
        __m256 cmp = _mm256_cmp_ps(sc, vmax, _CMP_GT_OS);
        vmax = _mm256_blendv_ps(vmax, sc, cmp);
        __m256i new_idx = _mm256_add_epi32(_mm256_set1_epi32(i),
                                           _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
        vidx = _mm256_blendv_epi8(vidx, new_idx, _mm256_castps_si256(cmp));
    }
    float vals[8];
    int idxs[8];
    _mm256_storeu_ps(vals, vmax);
    _mm256_storeu_si256((__m256i*)idxs, vidx);
    int best = idxs[0];
    float best_val = vals[0];
    for (int j = 1; j < 8; ++j) {
        if (vals[j] > best_val) { best_val = vals[j]; best = idxs[j]; }
    }
    for (; i < n; ++i) {
        float xx = logits[i] / cap;
        float x2 = xx * xx;
        float t = xx * (27.0f + x2) / (27.0f + 9.0f * x2);
        if (t > 1.0f) t = 1.0f; if (t < -1.0f) t = -1.0f;
        logits[i] = t * cap;
        if (logits[i] > best_val) { best_val = logits[i]; best = i; }
    }
    return best;
}

// ---- argmax_f32: find index of maximum value (read-only) ----
static inline int argmax_f32(const float* data, int n) {
    float best_val = data[0];
    __m256 vmax = _mm256_set1_ps(-1e30f);
    __m256i vidx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(&data[i]);
        __m256 cmp = _mm256_cmp_ps(v, vmax, _CMP_GT_OS);
        vmax = _mm256_blendv_ps(vmax, v, cmp);
        __m256i new_idx = _mm256_add_epi32(_mm256_set1_epi32(i),
                                           _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
        vidx = _mm256_blendv_epi8(vidx, new_idx, _mm256_castps_si256(cmp));
    }
    float vals[8];
    int idxs[8];
    _mm256_storeu_ps(vals, vmax);
    _mm256_storeu_si256((__m256i*)idxs, vidx);
    int best = idxs[0];
    for (int j = 0; j < 8; ++j) {
        if (vals[j] > best_val) { best_val = vals[j]; best = idxs[j]; }
    }
    for (; i < n; ++i) {
        if (data[i] > best_val) { best_val = data[i]; best = i; }
    }
    return best;
}

// ---- max_f32: find maximum value ----
static inline float max_f32(const float* data, int n) {
    __m256 vmax = _mm256_set1_ps(-1e30f);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(&data[i]);
        vmax = _mm256_max_ps(vmax, v);
    }
    __m128 m = _mm256_castps256_ps128(vmax);
    m = _mm_max_ps(m, _mm256_extractf128_ps(vmax, 1));
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 1));
    float max_val = _mm_cvtss_f32(m);
    for (; i < n; ++i) {
        if (data[i] > max_val)
            max_val = data[i];
    }
    return max_val;
}

// ---- softcap_and_max_f32: apply tanh-softcap in-place, return max ----
static inline float softcap_and_max_f32(float* data, int n, float cap) {
    __m256 vcap = _mm256_set1_ps(cap);
    __m256 vdiv = _mm256_set1_ps(1.0f / cap);
    __m256 c27 = _mm256_set1_ps(27.0f);
    __m256 c9 = _mm256_set1_ps(9.0f);
    __m256 vminus_one = _mm256_set1_ps(-1.0f);
    __m256 vone = _mm256_set1_ps(1.0f);
    __m256 vmax = _mm256_set1_ps(-1e30f);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(&data[i]);
        __m256 x = _mm256_mul_ps(v, vdiv);
        __m256 x2 = _mm256_mul_ps(x, x);
        __m256 num = _mm256_add_ps(c27, x2);
        __m256 den = _mm256_add_ps(c27, _mm256_mul_ps(c9, x2));
        __m256 th = _mm256_mul_ps(x, _mm256_div_ps(num, den));
        th = _mm256_min_ps(_mm256_max_ps(th, vminus_one), vone);
        __m256 sc = _mm256_mul_ps(th, vcap);
        _mm256_storeu_ps(&data[i], sc);
        vmax = _mm256_max_ps(vmax, sc);
    }
    __m128 m = _mm256_castps256_ps128(vmax);
    m = _mm_max_ps(m, _mm256_extractf128_ps(vmax, 1));
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 1));
    float max_val = _mm_cvtss_f32(m);
    for (; i < n; ++i) {
        float xx = data[i] / cap;
        float x2 = xx * xx;
        float t = xx * (27.0f + x2) / (27.0f + 9.0f * x2);
        if (t > 1.0f) t = 1.0f; if (t < -1.0f) t = -1.0f;
        data[i] = t * cap;
        if (data[i] > max_val) max_val = data[i];
    }
    return max_val;
}

// ---- exp_and_sum_f32: compute exp((data[i]-max_val)*inv_temp), store in out, return sum ----
// Uses fast-exp approximation (max error ~1.5%) — numerically different from std::exp.
static inline float exp_and_sum_f32(const float* data, float* out, int n, float max_val, float inv_temp) {
    __m256 vsum = _mm256_setzero_ps();
    __m256 vshift = _mm256_set1_ps(max_val);
    __m256 vscale = _mm256_set1_ps(inv_temp);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(&data[i]);
        v = _mm256_sub_ps(v, vshift);
        v = _mm256_mul_ps(v, vscale);
        // Fast exp approximation using AVX2 (max error ~1.5%)
        // exp(x) = 2^(x/ln2), use polynomial for fractional part
        __m256 exp_v;
        // Clamp to [-88, 88] to avoid overflow
        v = _mm256_min_ps(v, _mm256_set1_ps(88.0f));
        v = _mm256_max_ps(v, _mm256_set1_ps(-88.0f));
        // exp(x) = 2^(x * 1.44269504) = 2^(n + f) where n = floor(x*1.44269504)
        __m256 x = _mm256_mul_ps(v, _mm256_set1_ps(1.44269504f));
        __m256i n = _mm256_cvttps_epi32(x);
        __m256 f = _mm256_sub_ps(x, _mm256_cvtepi32_ps(n));
        // Polynomial: 2^f ≈ 1 + f*(0.693147 + f*(0.240227 + f*0.055504))
        __m256 p = _mm256_fmadd_ps(f, _mm256_set1_ps(0.055504f), _mm256_set1_ps(0.240227f));
        p = _mm256_fmadd_ps(f, p, _mm256_set1_ps(0.693147f));
        p = _mm256_fmadd_ps(f, p, _mm256_set1_ps(1.0f));
        // 2^n via bit manipulation
        __m256i n_shifted =
            _mm256_slli_epi32(_mm256_add_epi32(n, _mm256_set1_epi32(127)), 23);
        exp_v = _mm256_mul_ps(p, _mm256_castsi256_ps(n_shifted));
        _mm256_storeu_ps(&out[i], exp_v);
        vsum = _mm256_add_ps(vsum, exp_v);
    }
    // Horizontal sum
    __m128 hi = _mm256_extractf128_ps(vsum, 1);
    __m128 lo = _mm256_castps256_ps128(vsum);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    float sum = _mm_cvtss_f32(s);
    for (; i < n; ++i) {
        out[i] = std::exp((data[i] - max_val) * inv_temp);
        sum += out[i];
    }
    return sum;
}

// ---- scale_normalize_f32: data[i] *= inv_sum for i in [0, n) ----
static inline void scale_normalize_f32(float* data, int n, float inv_sum) {
    __m256 vinv = _mm256_set1_ps(inv_sum);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(&data[i]);
        _mm256_storeu_ps(&data[i], _mm256_mul_ps(v, vinv));
    }
    for (; i < n; ++i) {
        data[i] *= inv_sum;
    }
}

#endif  // USE_AVX2

}  // namespace cpu
}  // namespace forge