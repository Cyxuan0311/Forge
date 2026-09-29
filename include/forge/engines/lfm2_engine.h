#pragma once

// Lfm2Engine: LiquidAI LFM2 / LFM2.5 hybrid architecture.
//
// Block topology (per-layer attention.head_count_kv array):
//   == 0 -> ShortConv operator layer: y = out_proj(c ⊙ conv(b ⊙ x)), where
//           [b|c|x] = split(in_proj(hidden), 3) and conv is a causal depthwise
//           conv of width l_cache whose recurrent state keeps the last
//           (l_cache - 1) rows of bx (llama.cpp lfm2.cpp build_shortconv_block).
//   >  0 -> GQA full attention with per-head QK RMS-norm and NeoX RoPE over
//           the full head dim (theta 1e7). No output gate (unlike qwen35).
//
// Both block kinds share the same pre-norm residual + SwiGLU FFN skeleton.

#include <unordered_map>
#include <vector>

#include "forge/engines/transformer_engine.h"
#include "forge/inference/layers/attention_executor.h"

namespace forge {

class Lfm2Engine : public TransformerEngine {
public:
    explicit Lfm2Engine(Model& model, InferenceContext& ctx);

    std::string name() const override { return "lfm2"; }

    void reset() override;

protected:
    TensorPtr forward_layer(const TensorPtr& hidden, const LayerExecutionContext& lctx) override;
    bool init_weights() override;

private:
    TensorPtr apply_ffn(const TensorPtr& hidden_after_attn, const LayerExecutionContext& lctx);

    TensorPtr attention_cpu(const TensorPtr& normed, const LayerExecutionContext& lctx);
    TensorPtr shortconv_cpu(const TensorPtr& normed, const LayerExecutionContext& lctx);
#ifdef USE_CUDA
    TensorPtr attention_cuda(const TensorPtr& normed, const LayerExecutionContext& lctx);
    TensorPtr shortconv_cuda(const TensorPtr& normed, const LayerExecutionContext& lctx);
#endif

    // Effective conv kernel width (GGUF shortconv.l_cache; falls back to the
    // conv weight's trailing dim = [channels, kernel]).
    int conv_kernel(const ModelConfig& cfg, const LayerWeights& lw) const;

    // ---- per-sequence shortconv recurrent state ----
    // conv_states_[seq][layer]: [kernel-1, hidden_dim] FP32 tensor on the
    // layer's device, lazily zero-initialized on first touch.
    struct ConvStateEntry {
        TensorPtr state;
        int64_t next_pos = 0;
        bool valid = false;
    };
    float* conv_state(int seq_id, int layer_idx, int64_t start_pos, int seq_len, DeviceType dev,
                      int state_rows, int hidden_dim);

    AttentionExecutor attn_;
    std::unordered_map<int, std::vector<ConvStateEntry>> conv_states_;
};

}  // namespace forge
