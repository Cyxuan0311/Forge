#include "forge/inference/graph/qwen3vl_graph_builder.h"

#include <cstring>

#include "forge/cuda_kernels.h"
#include "forge/inference/forward_request.h"
#include "forge/inference/layer_execution_context.h"
#include "forge/inference/layers/ffn_executor.h"
#include "forge/inference/layers/norm_executor.h"
#include "forge/inference/layers/qwen3vl_full_attention.h"
#include "forge/op_enum.h"
#include "forge/operators.h"

namespace forge {

namespace {

inline int ref(int node_idx) {
    return -(node_idx + 1);
}

}  // namespace

int Qwen3vlGraphBuilder::build_layer(const GraphBuildContext& bctx) {
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

    // Full attention: encapsulate Qwen3vlFullAttention::attend as a single node.
    // This handles per-head QK-Norm, MRoPE, KV cache update, SDPA and the
    // attn_output projection internally, reusing GenericEngine's exact executors.
    Qwen3vlFullAttention full_attn(kv_cache);
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

    // Pre-FFN norm (ffn_norm / post_attention_layernorm), exactly like
    // GenericEngine::forward_layer's sequential residual path.
    int ffn_norm_w_idx = graph.add_input(lw.ffn_norm());
    int ffn_in_idx = graph.add_node("ffn_norm", OpType::RMS_NORM,
                                    {ref(after_attn_idx), ffn_norm_w_idx}, eps_params, dev);

    // FFN: encapsulate FfnExecutor::apply as a single node, mirroring
    // GenericEngine::forward_layer's sequential FFN branch call-for-call.
    // The executor dispatches to fused gate+up and fused down+residual kernels
    // for decode (M=1, quantized weights); those kernels share the input
    // quantization across w1/w3 and fold the residual add into the down_proj
    // accumulator, so a graph built from separate MUL_MAT_TRANSB + silu + ADD
    // primitives rounds differently at the ULP level and diverges from the
    // imperative baseline within a few greedy decode steps.
    //   inputs[0] = hidden_after_attn (residual), inputs[1] = ffn_normed.
    int ffn_out_idx = graph.add_node(
        "ffn", "ffn", {ref(after_attn_idx), ref(ffn_in_idx)},
        [cfg, lw, seq_len, dev](const std::vector<TensorPtr>& inputs) mutable -> TensorPtr {
            TensorPtr ffn_out =
                FfnExecutor::apply(inputs[1], inputs[0], cfg, lw, seq_len, dev);
            // Post-FFN norm (Gemma2 only; absent for qwen3vl, kept for parity
            // with GenericEngine's sequential path).
            if (cfg.has_post_ffn_norm) {
                auto post_ffn_norm = lw.post_ffn_norm();
                if (post_ffn_norm) {
                    ffn_out = NormExecutor::apply(ffn_out, post_ffn_norm, nullptr,
                                                  NormType::RMSNorm, cfg.rms_norm_eps);
                }
            }
            // Only skip the residual add when a fused down_proj kernel already
            // added it (same guard as GenericEngine).
            if (FfnExecutor::residual_fused(cfg, lw, seq_len, dev))
                return ffn_out;
            return ops::add(inputs[0], ffn_out);
        },
        dev);

    return ffn_out_idx;
}

int Qwen3vlGraphBuilder::build_output(ComputeGraph& graph, int hidden_idx,
                                     const ModelWeights& weights, const ModelConfig& cfg,
                                     DeviceType dev) {
    int out_norm_w_idx = graph.add_input(weights.output_norm);
    float out_eps = cfg.rms_norm_eps;
    int32_t out_eps_params[OP_PARAMS_MAX_SIZE / sizeof(int32_t)] = {};
    std::memcpy(out_eps_params, &out_eps, sizeof(float));
    int norm_idx = graph.add_node("output_norm", OpType::RMS_NORM,
                                  {ref(hidden_idx), out_norm_w_idx}, out_eps_params, dev);

    auto output_weight = weights.output_weight;
    if (!output_weight && cfg.tie_embeddings) {
        output_weight = weights.token_embedding;
    }
    int out_w_idx = graph.add_input(output_weight);
    return graph.add_node(
        "output_proj", "output_proj", {ref(norm_idx), out_w_idx},
        [output_weight](const std::vector<TensorPtr>& inputs) mutable -> TensorPtr {
            auto x = inputs[0];
            const int n_tokens = static_cast<int>(x->shape()[0]);
            // Mirror TransformerEngine::forward_layers' decode dispatch: CUDA
            // M=1 output projection runs specialized GEMV kernels (q4_0/q4_k/
            // q5_k/q6_k) whose reduction order differs from the generic MMQ
            // path ops::matmul_transB picks, so a plain MUL_MAT_TRANSB node
            // shifts every decode step's logits by ULPs and flips greedy
            // top-2 edges. Must keep the same kernel selection as imperative.
            if (output_weight->device() == DeviceType::CUDA &&
                x->device() == DeviceType::CUDA && n_tokens == 1) {
#ifdef USE_CUDA
                const int K = static_cast<int>(output_weight->shape()[1]);
                const int N = static_cast<int>(output_weight->shape()[0]);
                auto logits = std::make_shared<Tensor>(DataType::FP32,
                                                       std::vector<int64_t>{1, N},
                                                       DeviceType::CUDA);
                const float* in_ptr = static_cast<const float*>(x->data());
                float* out_ptr = static_cast<float*>(logits->data());
                switch (output_weight->dtype()) {
                    case DataType::Q4_0:
                        cuda::launch_output_proj_q4_0(in_ptr, output_weight->data(), out_ptr, K,
                                                      N);
                        return logits;
                    case DataType::Q4_K:
                        cuda::launch_output_proj_q4_k(in_ptr, output_weight->data(), out_ptr, K,
                                                      N);
                        return logits;
                    case DataType::Q5_K:
                        cuda::launch_output_proj_q5_k(in_ptr, output_weight->data(), out_ptr, K,
                                                      N);
                        return logits;
                    case DataType::Q6_K:
                        cuda::launch_output_proj_q6_k(in_ptr, output_weight->data(), out_ptr, K,
                                                      N);
                        return logits;
                    default:
                        break;
                }
#endif
            }
            return ops::matmul_transB(x, output_weight);
        },
        dev);
}

}  // namespace forge
