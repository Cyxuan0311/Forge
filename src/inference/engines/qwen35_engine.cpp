#include "forge/engines/qwen35_engine.h"

#include <stdexcept>

#include "forge/logger.h"
#include "forge/operators.h"

namespace forge {

Qwen35Engine::Qwen35Engine(Model& model, InferenceContext& ctx)
    : TransformerEngine(model, ctx),
      recurrent_memory_(*memory_.recurrent()),
      full_attention_(kv_cache_),
      linear_attention_(recurrent_memory_) {
    if (!init_weights()) {
        throw std::runtime_error("Qwen35Engine: failed to initialize weights");
    }
    recurrent_memory_.init(model_.config(), weights_);
}

bool Qwen35Engine::init_weights() {
    return weights_.init(model_.weights(), model_.config());
}

void Qwen35Engine::reset() {
    kv_cache_.reset();
    set_kv_cache_initialized(false);
    // recurrent state 由 InferenceContext 的 memory 持有, 但 engine 仍需触发 reset。
    // memory_->reset() 会在 context 层调用, 这里不需要重复。
}

TensorPtr Qwen35Engine::forward_layer(const TensorPtr& hidden,
                                      const LayerExecutionContext& lctx) {
    const auto& cfg = lctx.config;
    const auto& lw = lctx.weights;

    if (!lw.attn_norm()) {
        LOG_ERROR("Qwen35Engine: layer " + std::to_string(lctx.layer_idx) + " missing attn_norm");
        return hidden;
    }

    auto normed = ops::rms_norm(hidden, lw.attn_norm(), cfg.rms_norm_eps);

    TensorPtr attn_out;
    if (lw.layer_type == LayerType::FullAttention) {
        attn_out = full_attention_.attend(normed, lctx);
    } else {
        attn_out = linear_attention_.apply(normed, lctx);
    }
    if (!attn_out) {
        return hidden;
    }

    auto hidden_after_attn = ops::add(hidden, attn_out);
    return apply_ffn(hidden_after_attn, lctx);
}

TensorPtr Qwen35Engine::apply_ffn(const TensorPtr& hidden_after_attn,
                                  const LayerExecutionContext& lctx) {
    const auto& cfg = lctx.config;
    const auto& lw = lctx.weights;

    if (!lw.w1() || !lw.w3() || !lw.w2()) {
        return hidden_after_attn;
    }

    TensorPtr ffn_input = hidden_after_attn;
    if (lw.post_attention_norm()) {
        ffn_input = ops::rms_norm(hidden_after_attn, lw.post_attention_norm(), cfg.rms_norm_eps);
    }

    auto gate = ops::matmul_transB(ffn_input, lw.w1());
    auto up = ops::matmul_transB(ffn_input, lw.w3());
    auto ffn_mid = ops::silu_multiply(gate, up);
    auto ffn_out = ops::matmul_transB(ffn_mid, lw.w2());
    return ops::add(hidden_after_attn, ffn_out);
}

}  // namespace forge
