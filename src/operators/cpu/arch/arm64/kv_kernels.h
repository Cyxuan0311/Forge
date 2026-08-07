#pragma once
// ARM64 NEON KV head expansion kernel.
// Expands GQA KV cache heads: copies KV head data to each Q head in its group.
// kv_groups = num_heads / num_kv_heads.

#ifdef USE_NEON
#include <arm_neon.h>
#endif
#include <cstring>

namespace forge {
namespace cpu {

#ifdef USE_NEON

static inline void expand_kv_heads_f32(const float* kv_data, float* out_data,
                                       int seq_len, int num_heads, int num_kv_heads,
                                       int head_dim) {
    int kv_groups = num_heads / num_kv_heads;
    int kv_stride = num_kv_heads * head_dim;
    int out_stride = num_heads * head_dim;

    for (int s = 0; s < seq_len; ++s) {
        for (int h = 0; h < num_heads; ++h) {
            int kv_h = h / kv_groups;
            const float* src = kv_data + s * kv_stride + kv_h * head_dim;
            float* dst = out_data + s * out_stride + h * head_dim;

            // Vectorised copy of head_dim floats
            int d = 0;
            for (; d + 15 < head_dim; d += 16) {
                float32x4_t v0 = vld1q_f32(src + d);
                float32x4_t v1 = vld1q_f32(src + d + 4);
                float32x4_t v2 = vld1q_f32(src + d + 8);
                float32x4_t v3 = vld1q_f32(src + d + 12);
                vst1q_f32(dst + d,      v0);
                vst1q_f32(dst + d + 4,  v1);
                vst1q_f32(dst + d + 8,  v2);
                vst1q_f32(dst + d + 12, v3);
            }
            for (; d + 3 < head_dim; d += 4) {
                float32x4_t v = vld1q_f32(src + d);
                vst1q_f32(dst + d, v);
            }
            for (; d < head_dim; ++d) {
                dst[d] = src[d];
            }
        }
    }
}

#endif // USE_NEON

}  // namespace cpu
}  // namespace forge
