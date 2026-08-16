#pragma once

// PhimoeEngine: dedicated engine for Phi-mini-MoE-instruct (arch "phimoe").
//
// Architecture characteristics that prevent reuse of GenericEngine:
//   - MoE FFN (SiLU-gated, top-2 of 16 experts, no shared expert)
//   - RMSNorm followed by additive norm bias (attn_norm.bias, ffn_norm.bias, output_norm.bias)
//   - Attention output bias (attn_output.bias) and output projection bias (output.bias)
//   - NeoX RoPE (weights pre-permuted to half-split format in model.cpp)
//
// The base TransformerEngine head (rms_norm + matmul_transB) has no bias support,
// so forward_request is overridden to apply output_norm_bias and output_bias.

#include "forge/engines/transformer_engine.h"
#include "forge/inference/layers/attention_executor.h"
#include "forge/inference/layers/rope_executor.h"

namespace forge {

class PhimoeEngine : public TransformerEngine {
public:
    PhimoeEngine(Model& model, InferenceContext& ctx);

    std::string name() const override { return "phimoe"; }

    TensorPtr forward_request(const ForwardRequest& req) override;
    TensorPtr forward_batch(const InferenceBatch& batch) override;

protected:
    bool init_weights() override;
    TensorPtr forward_layer(const TensorPtr& hidden, const LayerExecutionContext& lctx) override;

private:
    // Apply RMSNorm then add bias (broadcast 1D bias over rows). bias may be null.
    static TensorPtr rms_norm_with_bias(const TensorPtr& x, const TensorPtr& weight,
                                        const TensorPtr& bias, float eps);

    // MoE FFN (CPU router path). SiLU-gated, top-k, no shared expert.
    TensorPtr moe_ffn_cpu(const TensorPtr& ffn_normed, const LayerWeights& lw,
                          const ModelConfig& cfg, int seq_len);

#ifdef USE_CUDA
    // MoE FFN fully on GPU: device router (softmax+top-k) + per-expert CUDA
    // GEMV dispatch. Reused scratch buffers avoid per-layer pool allocations.
    TensorPtr moe_ffn_cuda(const TensorPtr& ffn_normed, const LayerWeights& lw,
                           const ModelConfig& cfg, int seq_len);

    // Reused across moe_ffn_cuda invocations (device router outputs, softmax
    // buffer, host mirror of routing, and the per-layer expert accumulator).
    TensorPtr moe_router_indices_;
    TensorPtr moe_router_weights_;
    TensorPtr moe_router_softmax_;
    TensorPtr moe_indices_h_;
    TensorPtr moe_weights_h_;
    TensorPtr moe_expert_out_;
    TensorPtr moe_gate_out_;
    TensorPtr moe_up_out_;
    TensorPtr moe_gated_out_;
#endif

    RopeExecutor rope_executor_;
    AttentionExecutor attention_executor_;
};

}  // namespace forge
