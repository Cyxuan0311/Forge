#pragma once
// PowerPC64 VSX KV head expansion kernel.
// Expands GQA KV cache heads: copies KV head data to each Q head in its group.
// kv_groups = num_heads / num_kv_heads.
// Uses 16-wide vectorised copy via vec_xl / vec_xst.

#ifdef USE_VSX
#include <altivec.h>
#endif

namespace forge {
namespace cpu {

#ifdef USE_VSX

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

            // Vectorised copy of head_dim floats, 16-wide
            int d = 0;
            for (; d + 15 < head_dim; d += 16) {
                __vector float v0 = vec_xl(0, (const float*)(src + d));
                __vector float v1 = vec_xl(0, (const float*)(src + d + 4));
                __vector float v2 = vec_xl(0, (const float*)(src + d + 8));
                __vector float v3 = vec_xl(0, (const float*)(src + d + 12));
                vec_xst(v0, 0, (float*)(dst + d));
                vec_xst(v1, 0, (float*)(dst + d + 4));
                vec_xst(v2, 0, (float*)(dst + d + 8));
                vec_xst(v3, 0, (float*)(dst + d + 12));
            }
            for (; d + 3 < head_dim; d += 4) {
                __vector float v = vec_xl(0, (const float*)(src + d));
                vec_xst(v, 0, (float*)(dst + d));
            }
            for (; d < head_dim; ++d) {
                dst[d] = src[d];
            }
        }
    }
}

#endif // USE_VSX

}  // namespace cpu
}  // namespace forge
