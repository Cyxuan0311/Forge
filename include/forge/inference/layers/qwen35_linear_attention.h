#pragma once

// Qwen35LinearAttention: Qwen3.5 Linear Attention (Gated Delta Net) 层。
//
// 从 Qwen35Engine::forward_linear_attn_layer_{cpu,cuda} 抽出。输入为已过 attn_norm
// 的 hidden, 返回已过 ssm_out 投影的输出 (不含残差)。
// 循环状态由 Qwen35RecurrentMemory 提供, 本组件不持有状态所有权。

#include "forge/inference/layer_execution_context.h"
#include "forge/inference/layers/qwen35_recurrent_memory.h"
#include "forge/model.h"

namespace forge {

class Qwen35LinearAttention {
public:
    explicit Qwen35LinearAttention(Qwen35RecurrentMemory& memory) : memory_(memory) {}

    // 权重缺失时返回 nullptr, 由 engine 决定如何降级。
    TensorPtr apply(const TensorPtr& normed, const LayerExecutionContext& lctx);

    // Gated Delta Net 单 token 递推 (CPU)。state 原地更新。
    static void gated_delta_net_step(const float* q, const float* k, const float* v,
                                     const float* gate, const float* beta, float* state,
                                     float* output, int head_k_dim, int head_v_dim,
                                     int num_k_heads, int num_v_heads);

    // 带持久状态的 causal conv1d (CPU)。conv_state 原地更新。
    static void conv1d_cpu(const float* x_data, const float* weight_data, float* y_data,
                           float* conv_state, int seq_len, int conv_channels, int d_conv);

private:
    TensorPtr apply_cpu(const TensorPtr& normed, const LayerExecutionContext& lctx);
#ifdef USE_CUDA
    TensorPtr apply_cuda(const TensorPtr& normed, const LayerExecutionContext& lctx);
#endif

    Qwen35RecurrentMemory& memory_;
};

}  // namespace forge
