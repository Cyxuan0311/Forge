#include "forge/engines/llama_graph_builder.h"

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

// Static helper for rope computation
static void apply_rope_standard_cpu(const float* q_data, const float* k_data, float* q_out,
                                    float* k_out, int seq_len, int num_heads, int num_kv_heads,
                                    int head_dim, int64_t start_pos, float theta) {
    int half_dim = head_dim / 2;
    int q_stride = num_heads * head_dim;
    int k_stride = num_kv_heads * head_dim;
    for (int s = 0; s < seq_len; ++s) {
        for (int h = 0; h < num_heads; ++h) {
            for (int d = 0; d < half_dim; ++d) {
                float freq = 1.0f / std::pow(theta, 2.0f * d / head_dim);
                float angle = (start_pos + s) * freq;
                float cos_a = std::cos(angle);
                float sin_a = std::sin(angle);

                int q_idx0 = s * q_stride + h * head_dim + d;
                int q_idx1 = q_idx0 + half_dim;

                q_out[q_idx0] = q_data[q_idx0] * cos_a - q_data[q_idx1] * sin_a;
                q_out[q_idx1] = q_data[q_idx0] * sin_a + q_data[q_idx1] * cos_a;

                if (h < num_kv_heads) {
                    int k_idx0 = s * k_stride + h * head_dim + d;
                    int k_idx1 = k_idx0 + half_dim;

                    k_out[k_idx0] = k_data[k_idx0] * cos_a - k_data[k_idx1] * sin_a;
                    k_out[k_idx1] = k_data[k_idx0] * sin_a + k_data[k_idx1] * cos_a;
                }
            }
        }
    }
}

static void apply_rope_neox_cpu(const float* q_data, const float* k_data, float* q_out,
                                float* k_out, int seq_len, int num_heads, int num_kv_heads,
                                int head_dim, int64_t start_pos, float theta) {
    int half_dim = head_dim / 2;
    int q_stride = num_heads * head_dim;
    int k_stride = num_kv_heads * head_dim;
    for (int s = 0; s < seq_len; ++s) {
        for (int h = 0; h < num_heads; ++h) {
            for (int d = 0; d < half_dim; ++d) {
                float freq = 1.0f / std::pow(theta, 2.0f * d / head_dim);
                float angle = (start_pos + s) * freq;
                float cos_a = std::cos(angle);
                float sin_a = std::sin(angle);

                int q_idx0 = s * q_stride + h * head_dim + d;
                int q_idx1 = q_idx0 + half_dim;

                q_out[q_idx0] = q_data[q_idx0] * cos_a - q_data[q_idx1] * sin_a;
                q_out[q_idx1] = q_data[q_idx0] * sin_a + q_data[q_idx1] * cos_a;

                if (h < num_kv_heads) {
                    int k_idx0 = s * k_stride + h * head_dim + d;
                    int k_idx1 = k_idx0 + half_dim;

                    k_out[k_idx0] = k_data[k_idx0] * cos_a - k_data[k_idx1] * sin_a;
                    k_out[k_idx1] = k_data[k_idx0] * sin_a + k_data[k_idx1] * cos_a;
                }
            }
        }
    }
}

static void rope_q_only_cpu(const float* q_data, float* q_out, int seq_len, int num_heads,
                            int head_dim, int64_t start_pos, float theta, bool use_neox) {
    (void)use_neox;
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
                            int head_dim, int64_t start_pos, float theta, bool use_neox) {
    (void)use_neox;
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

int LlamaGraphBuilder::build_layer_graph(ComputeGraph& graph, int hidden_idx,
                                         const LayerWeights& lw, const ModelConfig& cfg,
                                         int layer_idx, int seq_len, int64_t start_pos,
                                         DeviceType dev, KVCache& kv_cache) {
    int num_heads = cfg.num_heads;
    int num_kv_heads = cfg.num_kv_heads;
    int head_dim = cfg.head_dim;
    float rms_eps = cfg.rms_norm_eps;
    float rope_theta = cfg.rope_theta;
    bool use_neox = cfg.use_neox_rope;

    // === Attention block (fine-grained nodes) ===

    // Node: attn_rms_norm
    int attn_w_idx = graph.add_input(lw.attn_norm());
    int32_t eps_params[OP_PARAMS_MAX_SIZE / sizeof(int32_t)] = {};
    std::memcpy(eps_params, &rms_eps, sizeof(float));
    int normed_idx =
        graph.add_node("attn_norm", OpType::RMS_NORM, {hidden_idx, attn_w_idx}, eps_params, dev);

    // Node: q_proj
    int wq_idx = graph.add_input(lw.wq());
    std::vector<int> q_inputs = {ref(normed_idx), wq_idx};
    if (lw.bq())
        q_inputs.push_back(graph.add_input(lw.bq()));
    int q_idx = graph.add_node("q_proj", OpType::MUL_MAT_TRANSB, q_inputs, nullptr, dev);

    // Node: k_proj
    int wk_idx = graph.add_input(lw.wk());
    std::vector<int> k_inputs = {ref(normed_idx), wk_idx};
    if (lw.bk())
        k_inputs.push_back(graph.add_input(lw.bk()));
    int k_idx = graph.add_node("k_proj", OpType::MUL_MAT_TRANSB, k_inputs, nullptr, dev);

    // Node: v_proj
    int wv_idx = graph.add_input(lw.wv());
    std::vector<int> v_inputs = {ref(normed_idx), wv_idx};
    if (lw.bv())
        v_inputs.push_back(graph.add_input(lw.bv()));
    int v_idx = graph.add_node("v_proj", OpType::MUL_MAT_TRANSB, v_inputs, nullptr, dev);

    // Helper to pack ROPE params
    auto pack_rope_params = [&](int is_q, int num_h) {
        int32_t* p = new int32_t[OP_PARAMS_MAX_SIZE / sizeof(int32_t)]();
        p[0] = is_q;
        p[1] = num_h;
        p[2] = head_dim;
        p[3] = seq_len;
        std::memcpy(p + 4, &start_pos, sizeof(int64_t));
        std::memcpy(p + 6, &rope_theta, sizeof(float));
        p[8] = use_neox ? 1 : 0;
        p[9] = (dev == DeviceType::CUDA) ? 1 : 0;
        return p;
    };

    // Node: q_rope
    int32_t* q_rope_params = pack_rope_params(1, num_heads);
    int q_rope_idx = graph.add_node("q_rope", OpType::ROPE, {ref(q_idx)}, q_rope_params, dev);
    delete[] q_rope_params;

    // Node: k_rope
    int32_t* k_rope_params = pack_rope_params(0, num_kv_heads);
    int k_rope_idx = graph.add_node("k_rope", OpType::ROPE, {ref(k_idx)}, k_rope_params, dev);
    delete[] k_rope_params;

    // Node: cache_cpy (update KV cache: store k_rope + v, optionally dequantize)
    int cache_idx = graph.add_node(
        "cache_cpy", "cpy_k", {ref(k_rope_idx), ref(v_idx)},
        [&kv_cache, layer_idx, seq_len, dev](const std::vector<TensorPtr>& inputs) -> TensorPtr {
            auto k_rope = inputs[0];
            auto v = inputs[1];
            kv_cache.update(layer_idx, k_rope, v, seq_len);
            if (kv_cache.kv_dtype() == KVCacheDType::Q4_0) {
                kv_cache.dequantize_layer(layer_idx);
            }
            return std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1}, dev);
        },
        dev);

    // Node: get_k (read filled K from cache, optionally copy to CUDA)
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

    // Node: get_v (read filled V from cache, optionally copy to CUDA)
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

    // Node: attention (flash attention or expand + sdpa)
    int32_t fa_params[OP_PARAMS_MAX_SIZE / sizeof(int32_t)] = {};
    fa_params[0] = num_heads;
    fa_params[1] = num_kv_heads;
    fa_params[2] = head_dim;
    fa_params[3] = 1;  // causal
    fa_params[4] = (dev == DeviceType::CUDA) ? 1 : 0;

    int attn_idx =
        graph.add_node("attention", OpType::FLASH_ATTN_GQA,
                       {ref(q_rope_idx), ref(k_cache_idx), ref(v_cache_idx)}, fa_params, dev);

    // Node: out_proj
    int wo_idx = graph.add_input(lw.wo());
    int proj_idx =
        graph.add_node("out_proj", OpType::MUL_MAT_TRANSB, {ref(attn_idx), wo_idx}, nullptr, dev);

    // Node: residual_add (hidden + attn_proj)
    int after_attn_idx =
        graph.add_node("residual_add", OpType::ADD, {hidden_idx, ref(proj_idx)}, nullptr, dev);

    // === FFN block (fine-grained nodes) ===

    // Node: ffn_rms_norm
    int ffn_w_idx = graph.add_input(lw.ffn_norm());
    int ffn_normed_idx = graph.add_node("ffn_norm", OpType::RMS_NORM,
                                        {ref(after_attn_idx), ffn_w_idx}, eps_params, dev);

    // FFN core: fused or split gate/up/silu path
    int ffn_mid_idx;
    bool use_ffn_up_fused = (dev == DeviceType::CUDA && lw.w1()->dtype() == DataType::Q4_0 &&
                             lw.w3()->dtype() == DataType::Q4_0);
    if (use_ffn_up_fused) {
        ffn_mid_idx = graph.add_node(
            "ffn_up_fused", "ffn_up_fused", {ref(ffn_normed_idx)},
            [&lw](const std::vector<TensorPtr>& inputs) -> TensorPtr {
                return ops::ffn_up_fused(inputs[0], lw.w1(), lw.w3(),
                                         static_cast<int>(lw.w1()->shape()[0]));
            },
            dev);
    } else {
        int w1_idx = graph.add_input(lw.w1());
        int gate_idx = graph.add_node("gate_proj", OpType::MUL_MAT_TRANSB,
                                      {ref(ffn_normed_idx), w1_idx}, nullptr, dev);
        int w3_idx = graph.add_input(lw.w3());
        int up_idx = graph.add_node("up_proj", OpType::MUL_MAT_TRANSB,
                                    {ref(ffn_normed_idx), w3_idx}, nullptr, dev);
        ffn_mid_idx = graph.add_node(
            "silu_multiply", "silu_multiply", {ref(gate_idx), ref(up_idx)},
            [](const std::vector<TensorPtr>& inputs) -> TensorPtr {
                return ops::silu_multiply(inputs[0], inputs[1]);
            },
            dev);
    }

    // FFN down: fused or regular matmul + residual
    bool use_ffn_down_fused = (dev == DeviceType::CUDA && lw.w2()->dtype() == DataType::Q4_0);
    int last_idx;
    if (use_ffn_down_fused) {
        last_idx = graph.add_node(
            "ffn_down_fused", "ffn_down_fused", {ref(ffn_mid_idx), ref(after_attn_idx)},
            [&lw](const std::vector<TensorPtr>& inputs) -> TensorPtr {
                int K_down = static_cast<int>(lw.w2()->shape()[1]);
                int N_down = static_cast<int>(lw.w2()->shape()[0]);
                auto ffn_out = std::make_shared<Tensor>(
                    DataType::FP32, std::vector<int64_t>{1, N_down}, DeviceType::CUDA);
                cuda::launch_ffn_down_fused_q4_0(
                    static_cast<const float*>(inputs[0]->data()), lw.w2()->data(),
                    static_cast<const float*>(inputs[1]->data()),
                    static_cast<float*>(ffn_out->data()), K_down, N_down);
                return ffn_out;
            },
            dev);
    } else {
        int w2_idx = graph.add_input(lw.w2());
        int down_idx = graph.add_node("down_proj", OpType::MUL_MAT_TRANSB,
                                      {ref(ffn_mid_idx), w2_idx}, nullptr, dev);
        last_idx = graph.add_node("ffn_residual_add", OpType::ADD,
                                  {ref(after_attn_idx), ref(down_idx)}, nullptr, dev);
    }

    return last_idx;
}

int LlamaGraphBuilder::build_output_graph(ComputeGraph& graph, int hidden_idx,
                                          const ModelWeights& weights, const ModelConfig& cfg) {
    // Node: Output norm
    int out_norm_w_idx = graph.add_input(weights.output_norm);
    float out_eps = cfg.rms_norm_eps;
    int32_t out_eps_params[OP_PARAMS_MAX_SIZE / sizeof(int32_t)] = {};
    std::memcpy(out_eps_params, &out_eps, sizeof(float));
    int norm_idx = graph.add_node("output_norm", OpType::RMS_NORM, {hidden_idx, out_norm_w_idx},
                                  out_eps_params);

    // Node: Output projection
    int logits_idx = graph.add_node(
        "output_proj", "mul_mat_transB", {ref(norm_idx)},
        [&weights, tie = cfg.tie_embeddings](const std::vector<TensorPtr>& inputs) -> TensorPtr {
            auto output_weight = weights.output_weight;
            if (!output_weight && tie) {
                output_weight = weights.token_embedding;
            }
            if (output_weight && output_weight->device() == DeviceType::CUDA &&
                output_weight->dtype() == DataType::Q4_0 && inputs[0]->shape()[0] == 1) {
                int K = static_cast<int>(output_weight->shape()[1]);
                int N = static_cast<int>(output_weight->shape()[0]);
                auto logits = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, N},
                                                       DeviceType::CUDA);
                cuda::launch_output_proj_q4_0(static_cast<const float*>(inputs[0]->data()),
                                              output_weight->data(),
                                              static_cast<float*>(logits->data()), K, N);
                return logits;
            }
            return ops::matmul_transB(inputs[0], output_weight);
        });

    return logits_idx;
}

// Register the LlamaGraphBuilder for all GQA-based architectures
static GraphBuilderAutoRegister _reg_llama_gb("llama", []() -> std::unique_ptr<LayerGraphBuilder> {
    return std::make_unique<LlamaGraphBuilder>();
});
static GraphBuilderAutoRegister _reg_mistral_gb("mistral",
                                                []() -> std::unique_ptr<LayerGraphBuilder> {
                                                    return std::make_unique<LlamaGraphBuilder>();
                                                });
static GraphBuilderAutoRegister _reg_qwen_gb("qwen", []() -> std::unique_ptr<LayerGraphBuilder> {
    return std::make_unique<LlamaGraphBuilder>();
});
static GraphBuilderAutoRegister _reg_qwen2_gb("qwen2", []() -> std::unique_ptr<LayerGraphBuilder> {
    return std::make_unique<LlamaGraphBuilder>();
});
static GraphBuilderAutoRegister _reg_yi_gb("yi", []() -> std::unique_ptr<LayerGraphBuilder> {
    return std::make_unique<LlamaGraphBuilder>();
});

}  // namespace forge
