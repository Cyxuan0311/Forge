#include "forge/inference/layers/qwen35_rope.h"

#include <cmath>

namespace forge {

int Qwen35Rope::rot_dim(const ModelConfig& cfg) {
    int n_rot = cfg.use_mrope ? cfg.rope_dimension_count : cfg.head_dim;
    if (n_rot <= 0) n_rot = cfg.head_dim;
    return n_rot;
}

void Qwen35Rope::apply_cpu(const float* q_data, const float* k_data, float* q_out, float* k_out,
                           int seq_len, int num_heads, int num_kv_heads, int head_dim, int n_rot,
                           int64_t start_pos, float theta) {
    int half_rot = n_rot / 2;
    float theta_scale = 1.0f / theta;

    int q_stride = num_heads * head_dim;
    int k_stride = num_kv_heads * head_dim;

#pragma omp parallel for schedule(static) if (seq_len > 1)
    for (int s = 0; s < seq_len; ++s) {
        int64_t pos = start_pos + s;

        for (int h = 0; h < num_heads; ++h) {
            const float* q_src = q_data + s * q_stride + h * head_dim;
            float* q_dst = q_out + s * q_stride + h * head_dim;

            for (int d = 0; d < half_rot; ++d) {
                float freq = std::pow(theta_scale, 2.0f * d / n_rot);
                float angle = pos * freq;
                float cos_a = std::cos(angle);
                float sin_a = std::sin(angle);

                float x0 = q_src[d];
                float x1 = q_src[d + half_rot];

                q_dst[d] = x0 * cos_a - x1 * sin_a;
                q_dst[d + half_rot] = x0 * sin_a + x1 * cos_a;
            }

            for (int d = n_rot; d < head_dim; ++d) {
                q_dst[d] = q_src[d];
            }
        }

        for (int h = 0; h < num_kv_heads; ++h) {
            const float* k_src = k_data + s * k_stride + h * head_dim;
            float* k_dst = k_out + s * k_stride + h * head_dim;

            for (int d = 0; d < half_rot; ++d) {
                float freq = std::pow(theta_scale, 2.0f * d / n_rot);
                float angle = pos * freq;
                float cos_a = std::cos(angle);
                float sin_a = std::sin(angle);

                float x0 = k_src[d];
                float x1 = k_src[d + half_rot];

                k_dst[d] = x0 * cos_a - x1 * sin_a;
                k_dst[d + half_rot] = x0 * sin_a + x1 * cos_a;
            }

            for (int d = n_rot; d < head_dim; ++d) {
                k_dst[d] = k_src[d];
            }
        }
    }
}

}  // namespace forge
