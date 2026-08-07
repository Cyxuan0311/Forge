#pragma once
// Generic (scalar) kernels aggregate header (FORGE_ARCH_GENERIC).
// Fallback for non-x86 / non-AVX2 builds: provides scalar implementations
// of all kernel functions that the x86 arch provides via SIMD.
// Selected by simd.h when no arch-specific SIMD is available.

#include <algorithm>
#include <cmath>
#include <cstring>

namespace forge {
namespace cpu {

// ---- cpuinfo (non-x86 stub) ----
inline bool cached_has_avx512_vnni() { return false; }

// ---- elementwise kernels ----

inline void add_f32_vec(const float* a, const float* b, float* out, int n) {
    for (int i = 0; i < n; ++i)
        out[i] = a[i] + b[i];
}

inline void mul_f32_vec(const float* a, const float* b, float* out, int n) {
    for (int i = 0; i < n; ++i)
        out[i] = a[i] * b[i];
}

inline void silu_mul_f32_vec(const float* gate, const float* up, float* out, int n) {
    for (int i = 0; i < n; ++i) {
        float v = gate[i];
        float silu_v = v / (1.0f + std::exp(-v));
        out[i] = silu_v * up[i];
    }
}

inline void gelu_mul_f32_vec(const float* gate, const float* up, float* out, int n) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    for (int i = 0; i < n; ++i) {
        float x = gate[i];
        float gelu_val = 0.5f * x * (1.0f + std::tanh(sqrt_2_over_pi * (x + coeff * x * x * x)));
        out[i] = gelu_val * up[i];
    }
}

// ---- norm kernel ----

inline void rms_norm_row_f32(const float* x_row, const float* w_row, float* o_row,
                             int cols, float eps) {
    float sum_sq = 0.0f;
    for (int c = 0; c < cols; ++c) {
        float v = x_row[c];
        sum_sq += v * v;
    }
    float rms = 1.0f / std::sqrt(sum_sq / cols + eps);
    if (w_row) {
        for (int c = 0; c < cols; ++c) {
            o_row[c] = x_row[c] * rms * w_row[c];
        }
    } else {
        for (int c = 0; c < cols; ++c) {
            o_row[c] = x_row[c] * rms;
        }
    }
}

// ---- attention kernels ----

inline float hsum_f32(float v0, float v1, float v2, float v3,
                      float v4, float v5, float v6, float v7) {
    return v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7;
}

inline float dot_f32(const float* a, const float* b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i)
        sum += a[i] * b[i];
    return sum;
}

inline void scale_f32(float* data, int n, float scale) {
    for (int i = 0; i < n; ++i)
        data[i] *= scale;
}

inline void fmadd_f32(float* acc, const float* src, int n, float weight) {
    for (int i = 0; i < n; ++i)
        acc[i] += weight * src[i];
}

// ---- sampling kernels ----

inline int softcap_and_argmax_f32(float* logits, int n, float cap) {
    int best = 0;
    float best_val = logits[0];
    for (int i = 0; i < n; ++i) {
        logits[i] = std::tanh(logits[i] / cap) * cap;
        if (logits[i] > best_val) { best_val = logits[i]; best = i; }
    }
    return best;
}

inline int argmax_f32(const float* data, int n) {
    int best = 0;
    float best_val = data[0];
    for (int i = 1; i < n; ++i) {
        if (data[i] > best_val) { best_val = data[i]; best = i; }
    }
    return best;
}

inline float max_f32(const float* data, int n) {
    float max_val = data[0];
    for (int i = 1; i < n; ++i) {
        if (data[i] > max_val) max_val = data[i];
    }
    return max_val;
}

inline float softcap_and_max_f32(float* data, int n, float cap) {
    float max_val = data[0];
    for (int i = 0; i < n; ++i) {
        data[i] = std::tanh(data[i] / cap) * cap;
        if (data[i] > max_val) max_val = data[i];
    }
    return max_val;
}

inline float exp_and_sum_f32(const float* data, float* out, int n, float max_val, float inv_temp) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        out[i] = std::exp((data[i] - max_val) * inv_temp);
        sum += out[i];
    }
    return sum;
}

inline void scale_normalize_f32(float* data, int n, float inv_sum) {
    for (int i = 0; i < n; ++i) {
        data[i] *= inv_sum;
    }
}

// ---- kv kernels ----

inline void expand_kv_heads_f32(const float* kv_data, float* out_data,
                                int seq_len, int num_heads, int num_kv_heads,
                                int head_dim) {
    int kv_groups = num_heads / num_kv_heads;
    for (int s = 0; s < seq_len; ++s) {
        for (int h = 0; h < num_heads; ++h) {
            int kv_h = h / kv_groups;
            for (int d = 0; d < head_dim; ++d) {
                out_data[s * num_heads * head_dim + h * head_dim + d] =
                    kv_data[s * num_kv_heads * head_dim + kv_h * head_dim + d];
            }
        }
    }
}

}  // namespace cpu
}  // namespace forge