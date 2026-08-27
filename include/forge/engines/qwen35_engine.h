#pragma once

#include "forge/engines/transformer_engine.h"
#include "forge/inference/layers/qwen35_full_attention.h"
#include "forge/inference/layers/qwen35_linear_attention.h"

namespace forge {
class Qwen35Engine : public TransformerEngine {
public:
    explicit Qwen35Engine(Model& model, InferenceContext& ctx);

    std::string name() const override { return "qwen35"; }

    void reset() override;

    // ------------------------------------------------------------------
    // DeepSeek-MTP style nextn head (single trained MTP layer, qwen35).
    // ------------------------------------------------------------------
    // Pull "mtp.*" glue tensors and the trailing decoder layer's weights
    // (canonical "layers.{trunk}.*" keys loaded by model.cpp). Returns false
    // when the checkpoint carries no MTP head.
    bool init_mtp();
    bool mtp_valid() const { return mtp_valid_; }

    // One autoregressive MTP step at KV position `pos`:
    //   x = eh_proj([enorm(embed(token)), hnorm(h_prev)])
    //   x = decoder_layer(x)              (KV slot = cfg.num_layers)
    //   h_next = shared_head_norm(x)      (falls back to output_norm)
    //   logits = lm_head(h_next)          (shared_head_head or output_weight)
    // `h_prev` must be [1, hidden_dim] FP32; outputs land on its device.
    TensorPtr mtp_step(int32_t token, const TensorPtr& h_prev, int64_t pos,
                       int seq_id, TensorPtr* logits_out);

protected:
    TensorPtr forward_layer(const TensorPtr& hidden, const LayerExecutionContext& lctx) override;
    bool init_weights() override;

private:
    // FFN 对 FullAttention 与 LinearAttention 两种层型完全相同。
    TensorPtr apply_ffn(const TensorPtr& hidden_after_attn,
                        const LayerExecutionContext& lctx);

    // recurrent state 由 InferenceContext 的 HybridMemory 持有, engine 只引用。
    Qwen35RecurrentMemory& recurrent_memory_;
    Qwen35FullAttention full_attention_;
    Qwen35LinearAttention linear_attention_;

    // MTP module state.
    bool mtp_valid_ = false;
    int mtp_kv_layer_ = -1;  // KV slot index == cfg.num_layers (reserved by init_kv_cache)
    LayerWeights mtp_lw_;
    TensorPtr mtp_eh_proj_, mtp_enorm_, mtp_hnorm_;
    TensorPtr mtp_shared_head_norm_, mtp_shared_head_head_;
};

}  // namespace forge
