#pragma once
// x86 AVX2 KV-head expansion kernel (FORGE_ARCH_X86 + USE_AVX2).
// Duplicated identically in transformer_engine.cpp and attention_executor.cpp —
// extracted to a single shared header.

#ifdef USE_AVX2
#    include <immintrin.h>
#endif

namespace forge {
namespace cpu {

#ifdef USE_AVX2
// expand_kv_heads_f32: replicate each KV head to all query heads in its group.
// kv_data:  [seq_len, num_kv_heads, head_dim]  (row-major)
// out_data: [seq_len, num_heads,    head_dim]  (row-major)
// kv_groups = num_heads / num_kv_heads
static inline void expand_kv_heads_f32(const float* kv_data, float* out_data,
                                       int seq_len, int num_heads, int num_kv_heads,
                                       int head_dim) {
    int kv_groups = num_heads / num_kv_heads;
    for (int s = 0; s < seq_len; ++s) {
        const float* kv_row = kv_data + s * num_kv_heads * head_dim;
        float* out_row = out_data + s * num_heads * head_dim;
        for (int kv_h = 0; kv_h < num_kv_heads; ++kv_h) {
            const float* src = kv_row + kv_h * head_dim;
            // Replicate this KV head to all query heads in its group
            for (int g = 0; g < kv_groups; ++g) {
                int dst_h = kv_h * kv_groups + g;
                float* dst = out_row + dst_h * head_dim;
                // head_dim is typically 64 = 8 * 8 floats, use AVX2
                for (int d = 0; d + 8 <= head_dim; d += 8) {
                    __m256 v = _mm256_loadu_ps(src + d);
                    _mm256_storeu_ps(dst + d, v);
                }
                // Handle remaining elements (unlikely for head_dim=64)
                for (int d = (head_dim / 8) * 8; d < head_dim; ++d) {
                    dst[d] = src[d];
                }
            }
        }
    }
}
#endif  // USE_AVX2

}  // namespace cpu
}  // namespace forge