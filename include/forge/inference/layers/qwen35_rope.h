#pragma once

// Qwen35Rope: Qwen3.5 Full Attention 层的 MRoPE。
//
// 从 Qwen35Engine::apply_rope_mrope 抽出。CUDA 路径复用通用 launch_rope_gqa,
// 这里只提供 CPU 实现和 n_rot 的推导规则, 保证两处使用同一套维度判断。

#include <cstdint>

#include "forge/model.h"

namespace forge {

struct Qwen35Rope {
    // 旋转维度: 启用 MRoPE 时取 rope_dimension_count, 否则整个 head_dim。
    static int rot_dim(const ModelConfig& cfg);

    static void apply_cpu(const float* q_data, const float* k_data, float* q_out, float* k_out,
                          int seq_len, int num_heads, int num_kv_heads, int head_dim, int n_rot,
                          int64_t start_pos, float theta);
};

}  // namespace forge
