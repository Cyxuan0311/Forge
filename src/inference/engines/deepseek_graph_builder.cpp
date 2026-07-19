#include "forge/engines/deepseek_graph_builder.h"

#include <cmath>
#include <cstring>

#include "forge/cuda_kernels.h"
#include "forge/logger.h"
#include "forge/op_dispatch.h"
#include "forge/op_enum.h"
#include "forge/operators.h"

#ifdef USE_CUDA
#    include <cuda_runtime.h>
#endif

namespace forge {

static void rope_q_only_cpu(const float* q_data, float* q_out, int seq_len, int num_heads,
                            int head_dim, int64_t start_pos, float theta) {
    int half_dim = head_dim / 2;
    int q_stride = num_heads * head_dim;
    for (int s = 0; s < seq_len; ++s) {
        for (int h = 0; h < num_heads; ++h) {
            for (int d = 0; d < half_dim; ++d) {
                float freq = 1.0f / std::pow(theta, 2.0f * d / head_dim);
                float angle = (start_pos + s) * freq;
                float cos_a = std::cos(angle);
                float sin_a = std::sin(angle);

                int idx0 = s * q_stride + h * head_dim + d;
                int idx1 = idx0 + half_dim;

                q_out[idx0] = q_data[idx0] * cos_a - q_data[idx1] * sin_a;
                q_out[idx1] = q_data[idx0] * sin_a + q_data[idx1] * cos_a;
            }
        }
    }
}

static void rope_k_only_cpu(const float* k_data, float* k_out, int seq_len, int num_kv_heads,
                            int head_dim, int64_t start_pos, float theta) {
    int half_dim = head_dim / 2;
    int k_stride = num_kv_heads * head_dim;
    for (int s = 0; s < seq_len; ++s) {
        for (int h = 0; h < num_kv_heads; ++h) {
            for (int d = 0; d < half_dim; ++d) {
                float freq = 1.0f / std::pow(theta, 2.0f * d / head_dim);
                float angle = (start_pos + s) * freq;
                float cos_a = std::cos(angle);
                float sin_a = std::sin(angle);

                int idx0 = s * k_stride + h * head_dim + d;
                int idx1 = idx0 + half_dim;

                k_out[idx0] = k_data[idx0] * cos_a - k_data[idx1] * sin_a;
                k_out[idx1] = k_data[idx0] * sin_a + k_data[idx1] * cos_a;
            }
        }
    }
}

static inline int ref(int node_idx) {
    return -(node_idx + 1);
}

int DeepSeekGraphBuilder::build_layer_graph(ComputeGraph& graph, int hidden_idx,
                                            const LayerWeights& lw, const ModelConfig& cfg,
                                            int layer_idx, int seq_len, int64_t start_pos,
                                            DeviceType dev, KVCache& kv_cache) {
    if (lw.layer_type == LayerType::MLA) {
        return build_mla_layer(graph, hidden_idx, lw, cfg, layer_idx, seq_len, start_pos, dev,
                               kv_cache);
    } else {
        return build_gqa_layer(graph, hidden_idx, lw, cfg, layer_idx, seq_len, start_pos, dev,
                               kv_cache);
    }
}

int DeepSeekGraphBuilder::build_gqa_layer(ComputeGraph& graph, int hidden_idx,
                                          const LayerWeights& lw, const ModelConfig& cfg,
                                          int layer_idx, int seq_len, int64_t start_pos,
                                          DeviceType dev, KVCache& kv_cache) {
    int num_heads = cfg.num_heads;
    int num_kv_heads = cfg.num_kv_heads;
    int head_dim = cfg.head_dim;
    float rms_eps = cfg.rms_norm_eps;
    float rope_theta = cfg.rope_theta;

    // === Attention block (fine-grained nodes) ===

    int attn_w_idx = graph.add_input(lw.attn_norm());
    int32_t d_eps_params[OP_PARAMS_MAX_SIZE / sizeof(int32_t)] = {};
    std::memcpy(d_eps_params, &rms_eps, sizeof(float));
    int normed_idx =
        graph.add_node("attn_norm", OpType::RMS_NORM, {hidden_idx, attn_w_idx}, d_eps_params, dev);

    int wq_ds_idx = graph.add_input(lw.wq());
    std::vector<int> q_ds_inputs = {ref(normed_idx), wq_ds_idx};
    if (lw.bq())
        q_ds_inputs.push_back(graph.add_input(lw.bq()));
    int q_idx = graph.add_node("q_proj", OpType::MUL_MAT_TRANSB, q_ds_inputs, nullptr, dev);

    int wk_ds_idx = graph.add_input(lw.wk());
    std::vector<int> k_ds_inputs = {ref(normed_idx), wk_ds_idx};
    if (lw.bk())
        k_ds_inputs.push_back(graph.add_input(lw.bk()));
    int k_idx = graph.add_node("k_proj", OpType::MUL_MAT_TRANSB, k_ds_inputs, nullptr, dev);

    int wv_ds_idx = graph.add_input(lw.wv());
    std::vector<int> v_ds_inputs = {ref(normed_idx), wv_ds_idx};
    if (lw.bv())
        v_ds_inputs.push_back(graph.add_input(lw.bv()));
    int v_idx = graph.add_node("v_proj", OpType::MUL_MAT_TRANSB, v_ds_inputs, nullptr, dev);

    auto pack_rope_ds = [&](int is_q, int num_h) {
        int32_t* p = new int32_t[OP_PARAMS_MAX_SIZE / sizeof(int32_t)]();
        p[0] = is_q;
        p[1] = num_h;
        p[2] = head_dim;
        p[3] = seq_len;
        std::memcpy(p + 4, &start_pos, sizeof(int64_t));
        std::memcpy(p + 6, &rope_theta, sizeof(float));
        p[8] = 0;
        p[9] = (dev == DeviceType::CUDA) ? 1 : 0;
        return p;
    };

    int32_t* q_rope_ds_p = pack_rope_ds(1, num_heads);
    int q_rope_idx = graph.add_node("q_rope", OpType::ROPE, {ref(q_idx)}, q_rope_ds_p, dev);
    delete[] q_rope_ds_p;

    int32_t* k_rope_ds_p = pack_rope_ds(0, num_kv_heads);
    int k_rope_idx = graph.add_node("k_rope", OpType::ROPE, {ref(k_idx)}, k_rope_ds_p, dev);
    delete[] k_rope_ds_p;

    int cache_idx = graph.add_node(
        "cache_cpy", "cpy_k", {ref(k_rope_idx), ref(v_idx)},
        [&kv_cache, layer_idx, seq_len, start_pos, dev](const std::vector<TensorPtr>& inputs) -> TensorPtr {
            kv_cache.update(layer_idx, /*seq_id=*/0, start_pos, inputs[0], inputs[1], seq_len);
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

    int32_t fa_ds_params[OP_PARAMS_MAX_SIZE / sizeof(int32_t)] = {};
    fa_ds_params[0] = num_heads;
    fa_ds_params[1] = num_kv_heads;
    fa_ds_params[2] = head_dim;
    fa_ds_params[3] = 1;  // causal
    fa_ds_params[4] = (dev == DeviceType::CUDA) ? 1 : 0;

    int attn_idx =
        graph.add_node("attention", OpType::FLASH_ATTN_GQA,
                       {ref(q_rope_idx), ref(k_cache_idx), ref(v_cache_idx)}, fa_ds_params, dev);

    int wo_ds_idx = graph.add_input(lw.wo());
    int proj_idx = graph.add_node("out_proj", OpType::MUL_MAT_TRANSB, {ref(attn_idx), wo_ds_idx},
                                  nullptr, dev);

    int after_attn_idx =
        graph.add_node("residual_add", OpType::ADD, {hidden_idx, ref(proj_idx)}, nullptr, dev);

    // === FFN block (fine-grained nodes) ===

    int ffn_w_idx = graph.add_input(lw.ffn_norm());
    int ffn_normed_idx = graph.add_node("ffn_norm", OpType::RMS_NORM,
                                        {ref(after_attn_idx), ffn_w_idx}, d_eps_params, dev);

    int w1_ds_idx = graph.add_input(lw.w1());
    int gate_idx = graph.add_node("gate_proj", OpType::MUL_MAT_TRANSB,
                                  {ref(ffn_normed_idx), w1_ds_idx}, nullptr, dev);

    int w3_ds_idx = graph.add_input(lw.w3());
    int up_idx = graph.add_node("up_proj", OpType::MUL_MAT_TRANSB, {ref(ffn_normed_idx), w3_ds_idx},
                                nullptr, dev);

    int silu_idx = graph.add_node("silu_gate", OpType::SILU, {ref(gate_idx)}, nullptr, dev);

    int ffn_mid_idx =
        graph.add_node("multiply_gate_up", OpType::MUL, {ref(silu_idx), ref(up_idx)}, nullptr, dev);

    int w2_ds_idx = graph.add_input(lw.w2());
    int down_idx = graph.add_node("down_proj", OpType::MUL_MAT_TRANSB,
                                  {ref(ffn_mid_idx), w2_ds_idx}, nullptr, dev);

    int last_idx = graph.add_node("ffn_residual_add", OpType::ADD,
                                  {ref(after_attn_idx), ref(down_idx)}, nullptr, dev);

    return last_idx;
}

int DeepSeekGraphBuilder::build_mla_layer(ComputeGraph& graph, int hidden_idx,
                                          const LayerWeights& lw, const ModelConfig& cfg,
                                          int layer_idx, int seq_len, int64_t start_pos,
                                          DeviceType dev, KVCache& kv_cache) {
    int num_heads = cfg.num_heads;
    int head_dim = cfg.head_dim;
    int kv_lora_rank = cfg.kv_lora_rank;
    float rms_eps = cfg.rms_norm_eps;
    float rope_theta = cfg.rope_theta;

    // === MLA attention block (fine-grained nodes) ===

    int attn_w_idx = graph.add_input(lw.attn_norm());
    int32_t d_eps_params[OP_PARAMS_MAX_SIZE / sizeof(int32_t)] = {};
    std::memcpy(d_eps_params, &rms_eps, sizeof(float));
    int normed_idx =
        graph.add_node("attn_norm", OpType::RMS_NORM, {hidden_idx, attn_w_idx}, d_eps_params, dev);

    // Compressed query via LoRA
    int q_latent_idx;
    if (lw.wq_a() && lw.wq_b()) {
        int wq_a_idx = graph.add_input(lw.wq_a());
        int q_a_idx = graph.add_node("q_a_proj", OpType::MUL_MAT_TRANSB,
                                     {ref(normed_idx), wq_a_idx}, nullptr, dev);
        int wq_b_idx = graph.add_input(lw.wq_b());
        q_latent_idx = graph.add_node("q_b_proj", OpType::MUL_MAT_TRANSB, {ref(q_a_idx), wq_b_idx},
                                      nullptr, dev);
    } else {
        auto w = lw.wq_a() ? lw.wq_a() : lw.wq_b();
        int wq_mla_idx = graph.add_input(w);
        q_latent_idx = graph.add_node("q_proj", OpType::MUL_MAT_TRANSB,
                                      {ref(normed_idx), wq_mla_idx}, nullptr, dev);
    }

    // Compressed KV
    int kv_a_w_idx = graph.add_input(lw.kv_a_proj());
    int compressed_kv_idx = graph.add_node("kv_a_proj", OpType::MUL_MAT_TRANSB,
                                           {ref(normed_idx), kv_a_w_idx}, nullptr, dev);

    int kv_b_w_idx = graph.add_input(lw.kv_b_proj());
    int v_latent_idx = graph.add_node("kv_b_proj", OpType::MUL_MAT_TRANSB,
                                      {ref(compressed_kv_idx), kv_b_w_idx}, nullptr, dev);

    // RoPE on q and k_latent
    auto pack_rope_mla = [&](int is_q, int num_h) {
        int32_t* p = new int32_t[OP_PARAMS_MAX_SIZE / sizeof(int32_t)]();
        p[0] = is_q;
        p[1] = num_h;
        p[2] = head_dim;
        p[3] = seq_len;
        std::memcpy(p + 4, &start_pos, sizeof(int64_t));
        std::memcpy(p + 6, &rope_theta, sizeof(float));
        p[8] = 0;
        p[9] = (dev == DeviceType::CUDA) ? 1 : 0;
        return p;
    };

    int32_t* q_rope_mla_p = pack_rope_mla(1, num_heads);
    int q_rope_idx = graph.add_node("q_rope", OpType::ROPE, {ref(q_latent_idx)}, q_rope_mla_p, dev);
    delete[] q_rope_mla_p;

    int32_t* k_rope_mla_p = pack_rope_mla(0, 1);
    int k_rope_idx =
        graph.add_node("k_rope", OpType::ROPE, {ref(compressed_kv_idx)}, k_rope_mla_p, dev);
    delete[] k_rope_mla_p;

    // KV cache update
    int cache_idx = graph.add_node(
        "cache_cpy", "cpy_k", {ref(k_rope_idx), ref(v_latent_idx)},
        [&kv_cache, layer_idx, seq_len, start_pos, dev](const std::vector<TensorPtr>& inputs) -> TensorPtr {
            kv_cache.update(layer_idx, /*seq_id=*/0, start_pos, inputs[0], inputs[1], seq_len);
            if (kv_cache.kv_dtype() == KVCacheDType::Q4_0) {
                kv_cache.dequantize_layer(layer_idx);
            }
            return std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1}, dev);
        },
        dev);

    // Read full K/V from cache and slice to filled portion
    int k_all_idx = graph.add_node(
        "get_k", "get_k", {ref(cache_idx)},
        [&kv_cache, layer_idx, dev](const std::vector<TensorPtr>&) -> TensorPtr {
            return kv_cache.get_key(layer_idx);
        },
        dev);

    int v_all_idx = graph.add_node(
        "get_v", "get_v", {ref(cache_idx)},
        [&kv_cache, layer_idx, dev](const std::vector<TensorPtr>&) -> TensorPtr {
            return kv_cache.get_value(layer_idx);
        },
        dev);

    int total_len_idx = graph.add_node(
        "total_len", "custom", {ref(cache_idx)},
        [&kv_cache, layer_idx](const std::vector<TensorPtr>&) -> TensorPtr {
            int total = kv_cache.filled(layer_idx);
            return std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1});
        },
        dev);

    // Slice K and V to the filled portion
    int k_sliced_idx = graph.add_node(
        "k_slice", "view", {ref(k_all_idx), ref(total_len_idx)},
        [&kv_cache, layer_idx](const std::vector<TensorPtr>& inputs) -> TensorPtr {
            auto k_all = inputs[0];
            int total = kv_cache.filled(layer_idx);
            int max_seq = kv_cache.max_seq_len();
            if (total < max_seq) {
                return std::make_shared<Tensor>(k_all->slice(0, 0, total));
            }
            return k_all;
        },
        dev);

    int v_sliced_idx = graph.add_node(
        "v_slice", "view", {ref(v_all_idx), ref(total_len_idx)},
        [&kv_cache, layer_idx](const std::vector<TensorPtr>& inputs) -> TensorPtr {
            auto v_all = inputs[0];
            int total = kv_cache.filled(layer_idx);
            int max_seq = kv_cache.max_seq_len();
            if (total < max_seq) {
                return std::make_shared<Tensor>(v_all->slice(0, 0, total));
            }
            return v_all;
        },
        dev);

    // Optional CPU → CUDA copy
    int k_final_idx = (dev == DeviceType::CUDA)
                          ? graph.add_node(
                                "k_to_cuda", "cpy", {ref(k_sliced_idx)},
                                [dev](const std::vector<TensorPtr>& inputs) -> TensorPtr {
                                    auto t = inputs[0];
                                    if (t->device() == DeviceType::CPU) {
                                        auto t_cuda = std::make_shared<Tensor>(
                                            DataType::FP32, t->shape(), DeviceType::CUDA);
                                        t_cuda->copy_from(*t);
                                        return t_cuda;
                                    }
                                    return t;
                                },
                                dev)
                          : k_sliced_idx;

    int v_final_idx = (dev == DeviceType::CUDA)
                          ? graph.add_node(
                                "v_to_cuda", "cpy", {ref(v_sliced_idx)},
                                [dev](const std::vector<TensorPtr>& inputs) -> TensorPtr {
                                    auto t = inputs[0];
                                    if (t->device() == DeviceType::CPU) {
                                        auto t_cuda = std::make_shared<Tensor>(
                                            DataType::FP32, t->shape(), DeviceType::CUDA);
                                        t_cuda->copy_from(*t);
                                        return t_cuda;
                                    }
                                    return t;
                                },
                                dev)
                          : v_sliced_idx;

    // Scaled dot-product attention with kv_lora_rank head_dim
    int attn_idx = graph.add_node(
        "attention", "flash_attn_gqa", {ref(q_rope_idx), ref(k_final_idx), ref(v_final_idx)},
        [num_heads, kv_lora_rank, dev](const std::vector<TensorPtr>& inputs) -> TensorPtr {
            auto q_rope = inputs[0];
            auto k_sliced = inputs[1];
            auto v_sliced = inputs[2];
            int seq_len_q = static_cast<int>(q_rope->shape()[0]);
            int total_len = static_cast<int>(k_sliced->shape()[0]);
            return ops::scaled_dot_product_attention_2d(q_rope, k_sliced, v_sliced, seq_len_q,
                                                        total_len, num_heads, kv_lora_rank, nullptr, true);
        },
        dev);

    int wo_ds_idx = graph.add_input(lw.wo());
    int proj_idx = graph.add_node("out_proj", OpType::MUL_MAT_TRANSB, {ref(attn_idx), wo_ds_idx},
                                  nullptr, dev);

    int after_attn_idx =
        graph.add_node("residual_add", OpType::ADD, {hidden_idx, ref(proj_idx)}, nullptr, dev);

    // === FFN block (fine-grained nodes) ===

    int ffn_w_idx = graph.add_input(lw.ffn_norm());
    int ffn_normed_idx = graph.add_node("ffn_norm", OpType::RMS_NORM,
                                        {ref(after_attn_idx), ffn_w_idx}, d_eps_params, dev);

    int w1_ds_idx = graph.add_input(lw.w1());
    int gate_idx = graph.add_node("gate_proj", OpType::MUL_MAT_TRANSB,
                                  {ref(ffn_normed_idx), w1_ds_idx}, nullptr, dev);

    int w3_ds_idx = graph.add_input(lw.w3());
    int up_idx = graph.add_node("up_proj", OpType::MUL_MAT_TRANSB, {ref(ffn_normed_idx), w3_ds_idx},
                                nullptr, dev);

    int silu_idx = graph.add_node("silu_gate", OpType::SILU, {ref(gate_idx)}, nullptr, dev);

    int ffn_mid_idx =
        graph.add_node("multiply_gate_up", OpType::MUL, {ref(silu_idx), ref(up_idx)}, nullptr, dev);

    int w2_ds_idx = graph.add_input(lw.w2());
    int down_idx = graph.add_node("down_proj", OpType::MUL_MAT_TRANSB,
                                  {ref(ffn_mid_idx), w2_ds_idx}, nullptr, dev);

    int last_idx = graph.add_node("ffn_residual_add", OpType::ADD,
                                  {ref(after_attn_idx), ref(down_idx)}, nullptr, dev);

    return last_idx;
}

int DeepSeekGraphBuilder::build_output_graph(ComputeGraph& graph, int hidden_idx,
                                             const ModelWeights& weights, const ModelConfig& cfg) {
    float out_eps_ds = cfg.rms_norm_eps;
    int32_t out_eps_ds_params[OP_PARAMS_MAX_SIZE / sizeof(int32_t)] = {};
    std::memcpy(out_eps_ds_params, &out_eps_ds, sizeof(float));
    int out_norm_w_idx = graph.add_input(weights.output_norm);
    int norm_idx = graph.add_node("output_norm", OpType::RMS_NORM, {hidden_idx, out_norm_w_idx},
                                  out_eps_ds_params);

    auto output_weight_ds = weights.output_weight;
    if (!output_weight_ds && cfg.tie_embeddings) {
        output_weight_ds = weights.token_embedding;
    }
    int out_w_idx = graph.add_input(output_weight_ds);
    int logits_idx =
        graph.add_node("output_proj", OpType::MUL_MAT_TRANSB, {ref(norm_idx), out_w_idx}, nullptr);

    return logits_idx;
}

// Register DeepSeek graph builders
static GraphBuilderAutoRegister _reg_deepseek_gb(
    "deepseek", []() -> std::unique_ptr<LayerGraphBuilder> {
        return std::make_unique<DeepSeekGraphBuilder>();
    });
static GraphBuilderAutoRegister _reg_deepseek_v2_gb(
    "deepseek_v2", []() -> std::unique_ptr<LayerGraphBuilder> {
        return std::make_unique<DeepSeekGraphBuilder>();
    });
static GraphBuilderAutoRegister _reg_deepseek_v3_gb(
    "deepseek_v3", []() -> std::unique_ptr<LayerGraphBuilder> {
        return std::make_unique<DeepSeekGraphBuilder>();
    });

}  // namespace forge
