#include "forge/inference/graph/qwen35_graph_builder.h"

#include <cstring>

#include "forge/inference/forward_request.h"
#include "forge/inference/layer_execution_context.h"
#include "forge/inference/layers/qwen35_full_attention.h"
#include "forge/inference/layers/qwen35_linear_attention.h"
#include "forge/inference/layers/qwen35_recurrent_memory.h"
#include "forge/op_enum.h"
#include "forge/operators.h"

namespace forge {

namespace {

inline int ref(int node_idx) {
    return -(node_idx + 1);
}

}  // namespace

int Qwen35GraphBuilder::build_layer(const GraphBuildContext& bctx) {
    if (bctx.weights.layer_type == LayerType::LinearAttention) {
        return build_linear_attn_layer(bctx);
    }
    return build_full_attn_layer(bctx);
}

int Qwen35GraphBuilder::build_full_attn_layer(const GraphBuildContext& bctx) {
    ComputeGraph& graph = bctx.graph;
    const auto& cfg = bctx.config;
    const auto& lw = bctx.weights;
    KVCache& kv_cache = bctx.kv_cache;
    const int hidden_idx = bctx.hidden_idx;
    const int layer_idx = bctx.layer_idx;
    const int seq_len = bctx.seq_len;
    const DeviceType dev = bctx.device;
    auto runtime = bctx.runtime;

    const float rms_eps = cfg.rms_norm_eps;

    int32_t eps_params[OP_PARAMS_MAX_SIZE / sizeof(int32_t)] = {};
    std::memcpy(eps_params, &rms_eps, sizeof(float));

    int attn_w_idx = graph.add_input(lw.attn_norm());
    int normed_idx =
        graph.add_node("attn_norm", OpType::RMS_NORM, {hidden_idx, attn_w_idx}, eps_params, dev);

    // Full attention: encapsulate Qwen35FullAttention::attend as a single node.
    // This handles split_q_gate, per-head QK-Norm, MRoPE, KV cache update, SDPA,
    // sigmoid(gate)*attn_out, and attn_output projection internally.
    Qwen35FullAttention full_attn(kv_cache);
    int attn_out_idx = graph.add_node(
        "full_attn", "full_attn", {ref(normed_idx)},
        [full_attn = std::move(full_attn), &cfg, &lw, runtime, layer_idx, seq_len,
         dev](const std::vector<TensorPtr>& inputs) mutable -> TensorPtr {
            ForwardRequest req = ForwardRequest::from_hidden(seq_len, runtime->start_pos,
                                                             runtime->seq_id);
            LayerExecutionContext lctx{cfg, lw, req, layer_idx, dev};
            return full_attn.attend(inputs[0], lctx);
        },
        dev);

    int after_attn_idx =
        graph.add_node("residual_add", OpType::ADD, {hidden_idx, ref(attn_out_idx)}, nullptr, dev);

    // FFN: follows Qwen35Engine::apply_ffn exactly.
    // Uses post_attention_norm (not ffn_norm) as the FFN input normalization.
    int ffn_in_idx = after_attn_idx;
    if (lw.post_attention_norm()) {
        int post_norm_w_idx = graph.add_input(lw.post_attention_norm());
        ffn_in_idx = graph.add_node("post_attn_norm", OpType::RMS_NORM,
                                    {ref(after_attn_idx), post_norm_w_idx}, eps_params, dev);
    }

    int w1_idx = graph.add_input(lw.w1());
    int gate_idx = graph.add_node("gate_proj", OpType::MUL_MAT_TRANSB,
                                  {ref(ffn_in_idx), w1_idx}, nullptr, dev);

    int w3_idx = graph.add_input(lw.w3());
    int up_idx = graph.add_node("up_proj", OpType::MUL_MAT_TRANSB,
                                {ref(ffn_in_idx), w3_idx}, nullptr, dev);

    int ffn_mid_idx = graph.add_node(
        "silu_multiply", "silu_multiply", {ref(gate_idx), ref(up_idx)},
        [](const std::vector<TensorPtr>& inputs) -> TensorPtr {
            return ops::silu_multiply(inputs[0], inputs[1]);
        },
        dev);

    int w2_idx = graph.add_input(lw.w2());
    int down_idx = graph.add_node("down_proj", OpType::MUL_MAT_TRANSB,
                                  {ref(ffn_mid_idx), w2_idx}, nullptr, dev);

    return graph.add_node("ffn_residual_add", OpType::ADD,
                          {ref(after_attn_idx), ref(down_idx)}, nullptr, dev);
}

int Qwen35GraphBuilder::build_linear_attn_layer(const GraphBuildContext& bctx) {
    ComputeGraph& graph = bctx.graph;
    const auto& cfg = bctx.config;
    const auto& lw = bctx.weights;
    KVCache& kv_cache = bctx.kv_cache;
    const int hidden_idx = bctx.hidden_idx;
    const int layer_idx = bctx.layer_idx;
    const int seq_len = bctx.seq_len;
    const DeviceType dev = bctx.device;
    auto runtime = bctx.runtime;
    auto* recurrent_memory = bctx.recurrent_memory;

    const float rms_eps = cfg.rms_norm_eps;

    int32_t eps_params[OP_PARAMS_MAX_SIZE / sizeof(int32_t)] = {};
    std::memcpy(eps_params, &rms_eps, sizeof(float));

    int attn_w_idx = graph.add_input(lw.attn_norm());
    int normed_idx =
        graph.add_node("attn_norm", OpType::RMS_NORM, {hidden_idx, attn_w_idx}, eps_params, dev);

    // Linear attention: encapsulate Qwen35LinearAttention::apply as a single node.
    // This handles all SSM operations internally, including conv_state and ssm_state updates.
    if (!recurrent_memory) {
        return hidden_idx;
    }

    Qwen35LinearAttention linear_attn(*recurrent_memory);
    int attn_out_idx = graph.add_node(
        "linear_attn", "linear_attn", {ref(normed_idx)},
        [linear_attn = std::move(linear_attn), &cfg, &lw, runtime, layer_idx, seq_len,
         dev](const std::vector<TensorPtr>& inputs) mutable -> TensorPtr {
            ForwardRequest req = ForwardRequest::from_hidden(seq_len, runtime->start_pos,
                                                             runtime->seq_id);
            LayerExecutionContext lctx{cfg, lw, req, layer_idx, dev};
            return linear_attn.apply(inputs[0], lctx);
        },
        dev);

    int after_attn_idx =
        graph.add_node("residual_add", OpType::ADD, {hidden_idx, ref(attn_out_idx)}, nullptr, dev);

    // FFN (same as FullAttention layer)
    int ffn_in_idx = after_attn_idx;
    if (lw.post_attention_norm()) {
        int post_norm_w_idx = graph.add_input(lw.post_attention_norm());
        ffn_in_idx = graph.add_node("post_attn_norm", OpType::RMS_NORM,
                                    {ref(after_attn_idx), post_norm_w_idx}, eps_params, dev);
    }

    int w1_idx = graph.add_input(lw.w1());
    int gate_idx = graph.add_node("gate_proj", OpType::MUL_MAT_TRANSB,
                                  {ref(ffn_in_idx), w1_idx}, nullptr, dev);

    int w3_idx = graph.add_input(lw.w3());
    int up_idx = graph.add_node("up_proj", OpType::MUL_MAT_TRANSB,
                                {ref(ffn_in_idx), w3_idx}, nullptr, dev);

    int ffn_mid_idx = graph.add_node(
        "silu_multiply", "silu_multiply", {ref(gate_idx), ref(up_idx)},
        [](const std::vector<TensorPtr>& inputs) -> TensorPtr {
            return ops::silu_multiply(inputs[0], inputs[1]);
        },
        dev);

    int w2_idx = graph.add_input(lw.w2());
    int down_idx = graph.add_node("down_proj", OpType::MUL_MAT_TRANSB,
                                  {ref(ffn_mid_idx), w2_idx}, nullptr, dev);

    return graph.add_node("ffn_residual_add", OpType::ADD,
                          {ref(after_attn_idx), ref(down_idx)}, nullptr, dev);
}

int Qwen35GraphBuilder::build_output(ComputeGraph& graph, int hidden_idx,
                                     const ModelWeights& weights, const ModelConfig& cfg) {
    int out_norm_w_idx = graph.add_input(weights.output_norm);
    float out_eps = cfg.rms_norm_eps;
    int32_t out_eps_params[OP_PARAMS_MAX_SIZE / sizeof(int32_t)] = {};
    std::memcpy(out_eps_params, &out_eps, sizeof(float));
    int norm_idx = graph.add_node("output_norm", OpType::RMS_NORM, {ref(hidden_idx), out_norm_w_idx},
                                  out_eps_params);

    auto output_weight = weights.output_weight;
    if (!output_weight && cfg.tie_embeddings) {
        output_weight = weights.token_embedding;
    }
    int out_w_idx = graph.add_input(output_weight);
    return graph.add_node("output_proj", OpType::MUL_MAT_TRANSB, {ref(norm_idx), out_w_idx},
                          nullptr);
}

}  // namespace forge
