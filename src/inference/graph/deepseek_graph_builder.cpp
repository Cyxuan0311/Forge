#include "forge/inference/graph/deepseek_graph_builder.h"

#include <cstring>

#include "forge/op_enum.h"
#include "forge/operators.h"

namespace forge {

namespace {

inline int ref(int node_idx) {
    return -(node_idx + 1);
}

constexpr int kRopeStartPosOffset = 4;

void pack_rope_params(int32_t* p, int is_q, int num_h, int head_dim, int seq_len, float theta,
                      DeviceType dev) {
    p[0] = is_q;
    p[1] = num_h;
    p[2] = head_dim;
    p[3] = seq_len;
    int64_t zero_pos = 0;
    std::memcpy(p + kRopeStartPosOffset, &zero_pos, sizeof(int64_t));
    std::memcpy(p + 6, &theta, sizeof(float));
    p[8] = 0;
    p[9] = (dev == DeviceType::CUDA) ? 1 : 0;
}

// CPU 上取到的 KV 在 CUDA 层需要搬运。抽出来避免 GQA/MLA 两处重复。
int add_to_cuda_node(ComputeGraph& graph, const char* name, int src_idx, DeviceType dev) {
    if (dev != DeviceType::CUDA) return src_idx;
    return graph.add_node(
        name, "cpy", {ref(src_idx)},
        [](const std::vector<TensorPtr>& inputs) -> TensorPtr {
            auto t = inputs[0];
            if (t->device() == DeviceType::CPU) {
                auto t_cuda =
                    std::make_shared<Tensor>(DataType::FP32, t->shape(), DeviceType::CUDA);
                t_cuda->copy_from(*t);
                return t_cuda;
            }
            return t;
        },
        dev);
}

// GQA 与 MLA 的 FFN 完全相同。
int build_ffn(ComputeGraph& graph, int after_attn_idx, const LayerWeights& lw, const int32_t* eps,
              DeviceType dev) {
    int ffn_w_idx = graph.add_input(lw.ffn_norm());
    int ffn_normed_idx =
        graph.add_node("ffn_norm", OpType::RMS_NORM, {ref(after_attn_idx), ffn_w_idx}, eps, dev);

    int w1_idx = graph.add_input(lw.w1());
    int gate_idx = graph.add_node("gate_proj", OpType::MUL_MAT_TRANSB,
                                  {ref(ffn_normed_idx), w1_idx}, nullptr, dev);

    int w3_idx = graph.add_input(lw.w3());
    int up_idx = graph.add_node("up_proj", OpType::MUL_MAT_TRANSB, {ref(ffn_normed_idx), w3_idx},
                                nullptr, dev);

    int silu_idx = graph.add_node("silu_gate", OpType::SILU, {ref(gate_idx)}, nullptr, dev);
    int ffn_mid_idx =
        graph.add_node("multiply_gate_up", OpType::MUL, {ref(silu_idx), ref(up_idx)}, nullptr, dev);

    int w2_idx = graph.add_input(lw.w2());
    int down_idx = graph.add_node("down_proj", OpType::MUL_MAT_TRANSB, {ref(ffn_mid_idx), w2_idx},
                                  nullptr, dev);

    return graph.add_node("ffn_residual_add", OpType::ADD, {ref(after_attn_idx), ref(down_idx)},
                          nullptr, dev);
}

}  // namespace

int DeepSeekGraphBuilder::build_layer(const GraphBuildContext& bctx) {
    if (bctx.weights.layer_type == LayerType::MLA) {
        return build_mla_layer(bctx);
    }
    return build_gqa_layer(bctx);
}

int DeepSeekGraphBuilder::build_gqa_layer(const GraphBuildContext& bctx) {
    ComputeGraph& graph = bctx.graph;
    const auto& cfg = bctx.config;
    const auto& lw = bctx.weights;
    KVCache& kv_cache = bctx.kv_cache;
    const int hidden_idx = bctx.hidden_idx;
    const int layer_idx = bctx.layer_idx;
    const int seq_len = bctx.seq_len;
    const DeviceType dev = bctx.device;
    auto runtime = bctx.runtime;

    const int num_heads = cfg.num_heads;
    const int num_kv_heads = cfg.num_kv_heads;
    const int head_dim = cfg.head_dim;

    int32_t eps_params[OP_PARAMS_MAX_SIZE / sizeof(int32_t)] = {};
    float rms_eps = cfg.rms_norm_eps;
    std::memcpy(eps_params, &rms_eps, sizeof(float));

    int attn_w_idx = graph.add_input(lw.attn_norm());
    int normed_idx =
        graph.add_node("attn_norm", OpType::RMS_NORM, {hidden_idx, attn_w_idx}, eps_params, dev);

    int wq_idx = graph.add_input(lw.wq());
    std::vector<int> q_inputs = {ref(normed_idx), wq_idx};
    if (lw.bq()) q_inputs.push_back(graph.add_input(lw.bq()));
    int q_idx = graph.add_node("q_proj", OpType::MUL_MAT_TRANSB, q_inputs, nullptr, dev);

    int wk_idx = graph.add_input(lw.wk());
    std::vector<int> k_inputs = {ref(normed_idx), wk_idx};
    if (lw.bk()) k_inputs.push_back(graph.add_input(lw.bk()));
    int k_idx = graph.add_node("k_proj", OpType::MUL_MAT_TRANSB, k_inputs, nullptr, dev);

    int wv_idx = graph.add_input(lw.wv());
    std::vector<int> v_inputs = {ref(normed_idx), wv_idx};
    if (lw.bv()) v_inputs.push_back(graph.add_input(lw.bv()));
    int v_idx = graph.add_node("v_proj", OpType::MUL_MAT_TRANSB, v_inputs, nullptr, dev);

    int32_t rope_params[OP_PARAMS_MAX_SIZE / sizeof(int32_t)] = {};
    pack_rope_params(rope_params, 1, num_heads, head_dim, seq_len, cfg.rope_theta, dev);
    int q_rope_idx = graph.add_node("q_rope", OpType::ROPE, {ref(q_idx)}, rope_params, dev);
    bctx.record_start_pos_slot(q_rope_idx, kRopeStartPosOffset);

    pack_rope_params(rope_params, 0, num_kv_heads, head_dim, seq_len, cfg.rope_theta, dev);
    int k_rope_idx = graph.add_node("k_rope", OpType::ROPE, {ref(k_idx)}, rope_params, dev);
    bctx.record_start_pos_slot(k_rope_idx, kRopeStartPosOffset);

    int cache_idx = graph.add_node(
        "cache_cpy", "cpy_k", {ref(k_rope_idx), ref(v_idx)},
        [&kv_cache, layer_idx, seq_len, dev, runtime](
            const std::vector<TensorPtr>& inputs) -> TensorPtr {
            kv_cache.update(layer_idx, runtime->seq_id, runtime->start_pos, inputs[0], inputs[1],
                            seq_len);
            if (kv_cache.kv_dtype() == KVCacheDType::Q4_0) {
                kv_cache.dequantize_layer(layer_idx);
            }
            return std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1}, dev);
        },
        dev);

    int k_cache_idx = graph.add_node(
        "get_k", "get_k", {ref(cache_idx)},
        [&kv_cache, layer_idx, dev](const std::vector<TensorPtr>&) -> TensorPtr {
            TensorPtr k_sliced = kv_cache.get_key_filled(layer_idx);
            if (dev == DeviceType::CUDA && k_sliced->device() == DeviceType::CPU) {
                auto k_cuda =
                    std::make_shared<Tensor>(DataType::FP32, k_sliced->shape(), DeviceType::CUDA);
                k_cuda->copy_from(*k_sliced);
                k_sliced = k_cuda;
            }
            return k_sliced;
        },
        dev);

    int v_cache_idx = graph.add_node(
        "get_v", "get_v", {ref(cache_idx)},
        [&kv_cache, layer_idx, dev](const std::vector<TensorPtr>&) -> TensorPtr {
            TensorPtr v_sliced = kv_cache.get_value_filled(layer_idx);
            if (dev == DeviceType::CUDA && v_sliced->device() == DeviceType::CPU) {
                auto v_cuda =
                    std::make_shared<Tensor>(DataType::FP32, v_sliced->shape(), DeviceType::CUDA);
                v_cuda->copy_from(*v_sliced);
                v_sliced = v_cuda;
            }
            return v_sliced;
        },
        dev);

    int32_t fa_params[OP_PARAMS_MAX_SIZE / sizeof(int32_t)] = {};
    fa_params[0] = num_heads;
    fa_params[1] = num_kv_heads;
    fa_params[2] = head_dim;
    fa_params[3] = 1;  // causal
    fa_params[4] = (dev == DeviceType::CUDA) ? 1 : 0;
    int attn_idx =
        graph.add_node("attention", OpType::FLASH_ATTN_GQA,
                       {ref(q_rope_idx), ref(k_cache_idx), ref(v_cache_idx)}, fa_params, dev);

    int wo_idx = graph.add_input(lw.wo());
    int proj_idx =
        graph.add_node("out_proj", OpType::MUL_MAT_TRANSB, {ref(attn_idx), wo_idx}, nullptr, dev);

    int after_attn_idx =
        graph.add_node("residual_add", OpType::ADD, {hidden_idx, ref(proj_idx)}, nullptr, dev);

    return build_ffn(graph, after_attn_idx, lw, eps_params, dev);
}

int DeepSeekGraphBuilder::build_mla_layer(const GraphBuildContext& bctx) {
    ComputeGraph& graph = bctx.graph;
    const auto& cfg = bctx.config;
    const auto& lw = bctx.weights;
    KVCache& kv_cache = bctx.kv_cache;
    const int hidden_idx = bctx.hidden_idx;
    const int layer_idx = bctx.layer_idx;
    const int seq_len = bctx.seq_len;
    const DeviceType dev = bctx.device;
    auto runtime = bctx.runtime;

    const int num_heads = cfg.num_heads;
    const int head_dim = cfg.head_dim;
    const int kv_lora_rank = cfg.kv_lora_rank;

    int32_t eps_params[OP_PARAMS_MAX_SIZE / sizeof(int32_t)] = {};
    float rms_eps = cfg.rms_norm_eps;
    std::memcpy(eps_params, &rms_eps, sizeof(float));

    int attn_w_idx = graph.add_input(lw.attn_norm());
    int normed_idx =
        graph.add_node("attn_norm", OpType::RMS_NORM, {hidden_idx, attn_w_idx}, eps_params, dev);

    // Q 的 low-rank 投影
    int q_latent_idx;
    if (lw.wq_a() && lw.wq_b()) {
        int wq_a_idx = graph.add_input(lw.wq_a());
        int q_a_idx = graph.add_node("q_a_proj", OpType::MUL_MAT_TRANSB,
                                     {ref(normed_idx), wq_a_idx}, nullptr, dev);
        int wq_b_idx = graph.add_input(lw.wq_b());
        q_latent_idx = graph.add_node("q_b_proj", OpType::MUL_MAT_TRANSB, {ref(q_a_idx), wq_b_idx},
                                      nullptr, dev);
    } else {
        int wq_idx = graph.add_input(lw.wq_a() ? lw.wq_a() : lw.wq_b());
        q_latent_idx = graph.add_node("q_proj", OpType::MUL_MAT_TRANSB, {ref(normed_idx), wq_idx},
                                      nullptr, dev);
    }

    // KV 压缩投影
    int kv_a_w_idx = graph.add_input(lw.kv_a_proj());
    int compressed_kv_idx = graph.add_node("kv_a_proj", OpType::MUL_MAT_TRANSB,
                                           {ref(normed_idx), kv_a_w_idx}, nullptr, dev);

    int kv_b_w_idx = graph.add_input(lw.kv_b_proj());
    int v_latent_idx = graph.add_node("kv_b_proj", OpType::MUL_MAT_TRANSB,
                                      {ref(compressed_kv_idx), kv_b_w_idx}, nullptr, dev);

    int32_t rope_params[OP_PARAMS_MAX_SIZE / sizeof(int32_t)] = {};
    pack_rope_params(rope_params, 1, num_heads, head_dim, seq_len, cfg.rope_theta, dev);
    int q_rope_idx = graph.add_node("q_rope", OpType::ROPE, {ref(q_latent_idx)}, rope_params, dev);
    bctx.record_start_pos_slot(q_rope_idx, kRopeStartPosOffset);

    pack_rope_params(rope_params, 0, 1, head_dim, seq_len, cfg.rope_theta, dev);
    int k_rope_idx =
        graph.add_node("k_rope", OpType::ROPE, {ref(compressed_kv_idx)}, rope_params, dev);
    bctx.record_start_pos_slot(k_rope_idx, kRopeStartPosOffset);

    int cache_idx = graph.add_node(
        "cache_cpy", "cpy_k", {ref(k_rope_idx), ref(v_latent_idx)},
        [&kv_cache, layer_idx, seq_len, dev, runtime](
            const std::vector<TensorPtr>& inputs) -> TensorPtr {
            kv_cache.update(layer_idx, runtime->seq_id, runtime->start_pos, inputs[0], inputs[1],
                            seq_len);
            if (kv_cache.kv_dtype() == KVCacheDType::Q4_0) {
                kv_cache.dequantize_layer(layer_idx);
            }
            return std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1}, dev);
        },
        dev);

    // latent 空间的 K/V 直接取 filled 视图, 不需要额外的 total_len 节点。
    int k_sliced_idx = graph.add_node(
        "get_k", "get_k", {ref(cache_idx)},
        [&kv_cache, layer_idx](const std::vector<TensorPtr>&) -> TensorPtr {
            return kv_cache.get_key_filled(layer_idx);
        },
        dev);

    int v_sliced_idx = graph.add_node(
        "get_v", "get_v", {ref(cache_idx)},
        [&kv_cache, layer_idx](const std::vector<TensorPtr>&) -> TensorPtr {
            return kv_cache.get_value_filled(layer_idx);
        },
        dev);

    int k_final_idx = add_to_cuda_node(graph, "k_to_cuda", k_sliced_idx, dev);
    int v_final_idx = add_to_cuda_node(graph, "v_to_cuda", v_sliced_idx, dev);

    // MLA 的 attention 在 latent 空间进行, head_dim 用 kv_lora_rank。
    int attn_idx = graph.add_node(
        "attention", "sdpa_mla", {ref(q_rope_idx), ref(k_final_idx), ref(v_final_idx)},
        [num_heads, kv_lora_rank](const std::vector<TensorPtr>& inputs) -> TensorPtr {
            int seq_len_q = static_cast<int>(inputs[0]->shape()[0]);
            int total_len = static_cast<int>(inputs[1]->shape()[0]);
            return ops::scaled_dot_product_attention_2d(inputs[0], inputs[1], inputs[2], seq_len_q,
                                                        total_len, num_heads, kv_lora_rank, nullptr,
                                                        true);
        },
        dev);

    int wo_idx = graph.add_input(lw.wo());
    int proj_idx =
        graph.add_node("out_proj", OpType::MUL_MAT_TRANSB, {ref(attn_idx), wo_idx}, nullptr, dev);

    int after_attn_idx =
        graph.add_node("residual_add", OpType::ADD, {hidden_idx, ref(proj_idx)}, nullptr, dev);

    return build_ffn(graph, after_attn_idx, lw, eps_params, dev);
}

int DeepSeekGraphBuilder::build_output(ComputeGraph& graph, int hidden_idx,
                                       const ModelWeights& weights, const ModelConfig& cfg) {
    int32_t eps_params[OP_PARAMS_MAX_SIZE / sizeof(int32_t)] = {};
    float out_eps = cfg.rms_norm_eps;
    std::memcpy(eps_params, &out_eps, sizeof(float));

    int out_norm_w_idx = graph.add_input(weights.output_norm);
    int norm_idx =
        graph.add_node("output_norm", OpType::RMS_NORM, {ref(hidden_idx), out_norm_w_idx}, eps_params);

    auto output_weight = weights.output_weight;
    if (!output_weight && cfg.tie_embeddings) {
        output_weight = weights.token_embedding;
    }
    int out_w_idx = graph.add_input(output_weight);
    return graph.add_node("output_proj", OpType::MUL_MAT_TRANSB, {ref(norm_idx), out_w_idx},
                          nullptr);
}

}  // namespace forge
