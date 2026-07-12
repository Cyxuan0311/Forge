#pragma once

#include "forge/engines/transformer_engine.h"

namespace forge {

class Qwen35Engine : public TransformerEngine {
public:
    explicit Qwen35Engine(Model& model, InferenceContext& ctx);

    std::string name() const override { return "qwen35"; }

    void reset() override;

protected:
    TensorPtr forward_layer(const TensorPtr& hidden, int layer_idx,
                            int seq_len, int64_t start_pos, DeviceType dev) override;
    bool init_weights() override;

private:
    // Forward pass for Linear Attention (Gated Delta Net) layers
    TensorPtr forward_linear_attn_layer(const TensorPtr& hidden, int layer_idx,
                                         int seq_len, int64_t start_pos, DeviceType dev);
    // Forward pass for Full Attention layers (with gated Q)
    TensorPtr forward_full_attn_layer(const TensorPtr& hidden, int layer_idx,
                                       int seq_len, int64_t start_pos, DeviceType dev);

    // Gated Delta Net autoregressive step (single token)
    void gated_delta_net_ar_cpu(
        const float* q, const float* k, const float* v,
        const float* gate, const float* beta,
        float* state, float* output,
        int head_k_dim, int head_v_dim, int num_k_heads, int num_v_heads);

    // Causal conv1d for linear attention layers
    void ssm_conv1d_cpu(
        const float* x_data, const float* weight_data, float* y_data,
        float* conv_state, int seq_len, int conv_channels, int d_conv);

    // MRoPE (Multi-dimensional RoPE) for Qwen3.5 Full Attention layers
    void apply_rope_mrope(
        const float* q_data, const float* k_data,
        float* q_out, float* k_out,
        int seq_len, int num_heads, int num_kv_heads,
        int head_dim, int n_rot, int64_t start_pos, float theta);

    // Persistent states for linear attention layers
    struct RecurrentLayerState {
        std::vector<float> conv_state;   // [d_conv-1, conv_channels]
        std::vector<float> ssm_state;    // [head_v_dim, head_v_dim, num_v_heads]
    };
    std::vector<RecurrentLayerState> recurrent_states_;
    bool states_initialized_ = false;

    // SSM dimensions (parsed from model config and weight shapes)
    int ssm_d_inner_ = 0;
    int ssm_d_state_ = 0;
    int ssm_n_group_ = 0;
    int ssm_dt_rank_ = 0;
    int ssm_d_conv_ = 0;
    int ssm_head_v_dim_ = 0;
    int ssm_conv_channels_ = 0;

    void init_recurrent_states();
};

} // namespace forge
