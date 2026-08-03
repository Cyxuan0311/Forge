#include "forge/inference/graph/llama_graph_builder.h"

#include <cstring>

#include "forge/cuda_kernels.h"
#include "forge/op_enum.h"
#include "forge/operators.h"

#ifdef USE_CUDA
#    include <cuda_runtime.h>
#endif

namespace forge {

namespace {

// 图内节点引用使用负数编码, 与 GraphNode::input_indices 约定一致。
inline int ref(int node_idx) {
    return -(node_idx + 1);
}

// ROPE op_params 布局见 op_kernels.cpp rope_kernel。
// start_pos 占 [4..5], 构建时写 0, 由 GraphRuntime 每次执行前回填。
constexpr int kRopeStartPosOffset = 4;

void pack_rope_params(int32_t* p, int is_q, int num_h, int head_dim, int seq_len, float theta,
                      bool use_neox, DeviceType dev) {
    p[0] = is_q;
    p[1] = num_h;
    p[2] = head_dim;
    p[3] = seq_len;
    int64_t zero_pos = 0;
    std::memcpy(p + kRopeStartPosOffset, &zero_pos, sizeof(int64_t));
    std::memcpy(p + 6, &theta, sizeof(float));
    p[8] = use_neox ? 1 : 0;
    p[9] = (dev == DeviceType::CUDA) ? 1 : 0;
}

}  // namespace

int LlamaGraphBuilder::build_layer(const GraphBuildContext& bctx) {
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
    const float rms_eps = cfg.rms_norm_eps;

    // === Attention ===

    int attn_w_idx = graph.add_input(lw.attn_norm());
    int32_t eps_params[OP_PARAMS_MAX_SIZE / sizeof(int32_t)] = {};
    std::memcpy(eps_params, &rms_eps, sizeof(float));
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
    pack_rope_params(rope_params, 1, num_heads, head_dim, seq_len, cfg.rope_theta,
                     cfg.use_neox_rope, dev);
    int q_rope_idx = graph.add_node("q_rope", OpType::ROPE, {ref(q_idx)}, rope_params, dev);
    bctx.record_start_pos_slot(q_rope_idx, kRopeStartPosOffset);

    pack_rope_params(rope_params, 0, num_kv_heads, head_dim, seq_len, cfg.rope_theta,
                     cfg.use_neox_rope, dev);
    int k_rope_idx = graph.add_node("k_rope", OpType::ROPE, {ref(k_idx)}, rope_params, dev);
    bctx.record_start_pos_slot(k_rope_idx, kRopeStartPosOffset);

    // KV cache 写入。start_pos / seq_id 从 runtime 读取而非捕获, 使缓存图可跨位置复用。
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

    // Fused attn output projection + residual for CPU decode
    bool use_attn_proj_fused =
        (dev == DeviceType::CPU && seq_len == 1 &&
         (lw.wo()->dtype() == DataType::Q4_0 || lw.wo()->dtype() == DataType::Q4_K ||
          lw.wo()->dtype() == DataType::Q5_K || lw.wo()->dtype() == DataType::Q6_K ||
          lw.wo()->dtype() == DataType::Q2_K || lw.wo()->dtype() == DataType::Q3_K));

    int after_attn_idx;
    if (use_attn_proj_fused) {
        after_attn_idx = graph.add_node(
            "attn_proj_fused", "attn_proj_fused", {ref(attn_idx), hidden_idx},
            [&lw](const std::vector<TensorPtr>& inputs) -> TensorPtr {
                auto wo_dtype = lw.wo()->dtype();
                if (wo_dtype == DataType::Q4_0) {
                    return ops::matmul_transB_fused_attn_proj_residual_q4_0(inputs[0], lw.wo(),
                                                                             inputs[1]);
                } else if (wo_dtype == DataType::Q4_K) {
                    return ops::matmul_transB_fused_attn_proj_residual_q4_k(inputs[0], lw.wo(),
                                                                             inputs[1]);
                } else if (wo_dtype == DataType::Q5_K) {
                    return ops::matmul_transB_fused_attn_proj_residual_q5_k(inputs[0], lw.wo(),
                                                                             inputs[1]);
                } else if (wo_dtype == DataType::Q6_K) {
                    return ops::matmul_transB_fused_attn_proj_residual_q6_k(inputs[0], lw.wo(),
                                                                             inputs[1]);
                } else if (wo_dtype == DataType::Q2_K) {
                    return ops::matmul_transB_fused_attn_proj_residual_q2_k(inputs[0], lw.wo(),
                                                                             inputs[1]);
                } else if (wo_dtype == DataType::Q3_K) {
                    return ops::matmul_transB_fused_attn_proj_residual_q3_k(inputs[0], lw.wo(),
                                                                             inputs[1]);
                }
                return ops::add(inputs[1], ops::matmul_transB(inputs[0], lw.wo()));
            },
            dev);
    } else {
        int proj_idx = graph.add_node("out_proj", OpType::MUL_MAT_TRANSB,
                                      {ref(attn_idx), wo_idx}, nullptr, dev);
        after_attn_idx =
            graph.add_node("residual_add", OpType::ADD, {hidden_idx, ref(proj_idx)}, nullptr, dev);
    }

    // === FFN ===

    int ffn_w_idx = graph.add_input(lw.ffn_norm());
    int ffn_normed_idx = graph.add_node("ffn_norm", OpType::RMS_NORM,
                                        {ref(after_attn_idx), ffn_w_idx}, eps_params, dev);

    int ffn_mid_idx;
    bool use_ffn_up_fused =
        (dev == DeviceType::CUDA &&
         ((lw.w1()->dtype() == DataType::Q4_0 && lw.w3()->dtype() == DataType::Q4_0) ||
          (lw.w1()->dtype() == DataType::Q4_K && lw.w3()->dtype() == DataType::Q4_K)));
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

    bool use_ffn_down_fused =
        (dev == DeviceType::CUDA &&
         (lw.w2()->dtype() == DataType::Q4_0 || lw.w2()->dtype() == DataType::Q4_K ||
          lw.w2()->dtype() == DataType::Q5_K || lw.w2()->dtype() == DataType::Q6_K)) ||
        (dev == DeviceType::CPU && seq_len == 1 &&
         (lw.w2()->dtype() == DataType::Q4_0 || lw.w2()->dtype() == DataType::Q4_1 ||
          lw.w2()->dtype() == DataType::Q4_K || lw.w2()->dtype() == DataType::Q5_K ||
          lw.w2()->dtype() == DataType::Q6_K || lw.w2()->dtype() == DataType::Q2_K ||
          lw.w2()->dtype() == DataType::Q3_K));
    if (!use_ffn_down_fused) {
        int w2_idx = graph.add_input(lw.w2());
        int down_idx = graph.add_node("down_proj", OpType::MUL_MAT_TRANSB,
                                      {ref(ffn_mid_idx), w2_idx}, nullptr, dev);
        return graph.add_node("ffn_residual_add", OpType::ADD,
                              {ref(after_attn_idx), ref(down_idx)}, nullptr, dev);
    }

    if (dev == DeviceType::CUDA) {
        return graph.add_node(
            "ffn_down_fused", "ffn_down_fused", {ref(ffn_mid_idx), ref(after_attn_idx)},
            [&lw](const std::vector<TensorPtr>& inputs) -> TensorPtr {
                int K_down = static_cast<int>(lw.w2()->shape()[1]);
                int N_down = static_cast<int>(lw.w2()->shape()[0]);
                auto ffn_out = std::make_shared<Tensor>(DataType::FP32,
                                                        std::vector<int64_t>{1, N_down},
                                                        DeviceType::CUDA);
#ifdef USE_CUDA
                auto w2_dtype = lw.w2()->dtype();
                const float* mid = static_cast<const float*>(inputs[0]->data());
                const float* residual = static_cast<const float*>(inputs[1]->data());
                float* out = static_cast<float*>(ffn_out->data());
                if (w2_dtype == DataType::Q4_0) {
                    cuda::launch_ffn_down_fused_q4_0_q8_1(mid, lw.w2()->data(), residual, out, K_down,
                                                          N_down);
                } else if (w2_dtype == DataType::Q4_K) {
                    cuda::launch_ffn_down_fused_q4_k_q8_1(mid, lw.w2()->data(), residual, out, K_down,
                                                          N_down);
                } else if (w2_dtype == DataType::Q5_K) {
                    cuda::launch_ffn_down_fused_q5_k(mid, lw.w2()->data(), residual, out, K_down,
                                                     N_down);
                } else if (w2_dtype == DataType::Q6_K) {
                    cuda::launch_ffn_down_fused_q6_k(mid, lw.w2()->data(), residual, out, K_down,
                                                     N_down);
                }
#endif
                return ffn_out;
            },
            dev);
    }

    // CPU fused FFN down + residual
    return graph.add_node(
        "ffn_down_fused", "ffn_down_fused", {ref(ffn_mid_idx), ref(after_attn_idx)},
        [&lw](const std::vector<TensorPtr>& inputs) -> TensorPtr {
            auto w2_dtype = lw.w2()->dtype();
            if (w2_dtype == DataType::Q4_0) {
                return ops::matmul_transB_fused_ffn_down_residual_q4_0(inputs[0], lw.w2(), inputs[1]);
            } else if (w2_dtype == DataType::Q4_1) {
                return ops::matmul_transB_fused_ffn_down_residual_q4_1(inputs[0], lw.w2(), inputs[1]);
            } else if (w2_dtype == DataType::Q4_K) {
                return ops::matmul_transB_fused_ffn_down_residual_q4_k(inputs[0], lw.w2(), inputs[1]);
            } else if (w2_dtype == DataType::Q5_K) {
                return ops::matmul_transB_fused_ffn_down_residual_q5_k(inputs[0], lw.w2(), inputs[1]);
            } else if (w2_dtype == DataType::Q6_K) {
                return ops::matmul_transB_fused_ffn_down_residual_q6_k(inputs[0], lw.w2(), inputs[1]);
            } else if (w2_dtype == DataType::Q2_K) {
                return ops::matmul_transB_fused_ffn_down_residual_q2_k(inputs[0], lw.w2(), inputs[1]);
            } else if (w2_dtype == DataType::Q3_K) {
                return ops::matmul_transB_fused_ffn_down_residual_q3_k(inputs[0], lw.w2(), inputs[1]);
            }
            return ops::add(inputs[1], ops::matmul_transB(inputs[0], lw.w2()));
        },
        dev);
}

int LlamaGraphBuilder::build_output(ComputeGraph& graph, int hidden_idx,
                                    const ModelWeights& weights, const ModelConfig& cfg) {
    int out_norm_w_idx = graph.add_input(weights.output_norm);
    float out_eps = cfg.rms_norm_eps;
    int32_t out_eps_params[OP_PARAMS_MAX_SIZE / sizeof(int32_t)] = {};
    std::memcpy(out_eps_params, &out_eps, sizeof(float));
    int norm_idx = graph.add_node("output_norm", OpType::RMS_NORM, {ref(hidden_idx), out_norm_w_idx},
                                  out_eps_params);

    return graph.add_node(
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
#ifdef USE_CUDA
                cuda::launch_output_proj_q4_0_q8_1(static_cast<const float*>(inputs[0]->data()),
                                                   output_weight->data(),
                                                   static_cast<float*>(logits->data()), K, N);
#endif
                return logits;
            }
            return ops::matmul_transB(inputs[0], output_weight);
        });
}

}  // namespace forge
