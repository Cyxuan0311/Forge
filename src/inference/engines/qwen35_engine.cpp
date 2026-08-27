#include "forge/engines/qwen35_engine.h"

#include <cstring>
#include <stdexcept>

#include "forge/inference/layers/ffn_executor.h"
#include "forge/logger.h"
#include "forge/operators.h"
#include "forge/perf_profiler.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

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
    graph_runtime_.set_recurrent_memory(&recurrent_memory_);
    init_mtp();
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

    // Route through FfnExecutor so decode (M=1) uses the fused gate+up and
    // down+residual kernels (single OpenMP region each, x quantized once)
    // instead of four separate matmul/silu/add calls.
    int seq_len = static_cast<int>(ffn_input->shape()[0]);
    TensorPtr ffn_out;
    {
        PERF_SCOPE("layer/ffn");
        ffn_out = FfnExecutor::apply(ffn_input, hidden_after_attn, cfg, lw, seq_len,
                                     lctx.device.type);
    }
    if (FfnExecutor::residual_fused(cfg, lw, seq_len, lctx.device.type)) {
        return ffn_out;
    }
    return ops::add(hidden_after_attn, ffn_out);
}

// =========================================================================
// DeepSeek-MTP style nextn head
// =========================================================================

bool Qwen35Engine::init_mtp() {
    mtp_valid_ = false;
    const auto& cfg = model_.config();
    if (cfg.n_nextn_layers <= 0) {
        return false;
    }

    auto ws = [&](const std::string& k) { return model_.weights().get(k); };

    mtp_eh_proj_ = ws("mtp.eh_proj");
    mtp_enorm_ = ws("mtp.enorm");
    mtp_hnorm_ = ws("mtp.hnorm");
    if (!mtp_eh_proj_ || !mtp_enorm_ || !mtp_hnorm_) {
        LOG_WARN("qwen35: nextn_predict_layers>0 but MTP glue tensors missing; "
                 "MTP drafting disabled");
        return false;
    }
    // Optional explicit copies fall back to the shared trunk weights at use.
    mtp_shared_head_norm_ = ws("mtp.shared_head_norm");
    mtp_shared_head_head_ = ws("mtp.shared_head_head");

    // Decoder layer weights live under the canonical trunk-style keys
    // "layers.{trunk}.*" written by Model::load.
    const int layer_idx = cfg.num_layers;  // first trailing block
    if (!ws("layers." + std::to_string(layer_idx) + ".attn_q")) {
        LOG_WARN("qwen35: MTP decoder layer weights missing for blk." +
                 std::to_string(layer_idx));
        return false;
    }
    LayerWeightInitContext wctx{model_.weights(), cfg, layer_idx, mtp_lw_};
    WeightInitRegistry::instance().init_layer(cfg.arch_type, wctx);

    mtp_kv_layer_ = layer_idx;  // KV slot reserved by init_kv_cache
    mtp_valid_ = true;
    LOG_INFO("qwen35: MTP module initialized (kv layer " + std::to_string(mtp_kv_layer_) + ")");
    return true;
}

TensorPtr Qwen35Engine::mtp_step(int32_t token, const TensorPtr& h_prev, int64_t pos,
                                 int seq_id, TensorPtr* logits_out) {
    if (!mtp_valid_) {
        return nullptr;
    }
    const auto& cfg = model_.config();
    const DeviceType dev = h_prev->device();

    // token -> embedding ([1, H])
    auto ids = std::make_shared<Tensor>(DataType::INT32, std::vector<int64_t>{1},
                                        DeviceType::CPU);
    *static_cast<int32_t*>(ids->data()) = token;
    auto emb_w = model_.weights().get("token_embedding");
    auto h = ops::embedding(emb_w, ids);
    if (!h) return nullptr;
    if (h->device() != dev) {
        h->to_device(dev);
    }

    // x = eh_proj([enorm(embed), hnorm(h_prev)])   [1, 2H] -> [1, H]
    auto e = ops::rms_norm(h, mtp_enorm_, cfg.rms_norm_eps);
    auto hn_in = h_prev;
    if (hn_in->device() != dev) hn_in->to_device(dev);
    auto hn = ops::rms_norm(hn_in, mtp_hnorm_, cfg.rms_norm_eps);

    const int64_t H = h->shape()[1];
    auto cat = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, 2 * H}, dev);
    float* dst = static_cast<float*>(cat->data());
#ifdef USE_CUDA
    if (dev == DeviceType::CUDA) {
        cudaMemcpyAsync(dst, e->data(), sizeof(float) * H, cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(dst + H, hn->data(), sizeof(float) * H, cudaMemcpyDeviceToDevice);
    } else {
        std::memcpy(dst, e->data(), sizeof(float) * H);
        std::memcpy(dst + H, hn->data(), sizeof(float) * H);
    }
#else
    std::memcpy(dst, e->data(), sizeof(float) * H);
    std::memcpy(dst + H, hn->data(), sizeof(float) * H);
#endif

    auto x = ops::matmul_transB(cat, mtp_eh_proj_);
    if (!x) return nullptr;
    if (x->device() != dev) x->to_device(dev);

    // Decoder layer with its own KV slot.
    ForwardRequest req = ForwardRequest::from_hidden(/*n_tokens=*/1, pos, seq_id);
    LayerExecutionContext lctx{cfg, mtp_lw_, req, mtp_kv_layer_, DeviceTarget(dev)};
    auto cur = forward_layer(x, lctx);
    if (!cur) return nullptr;

    // Head norm + LM head.
    auto norm_w = mtp_shared_head_norm_ ? mtp_shared_head_norm_
                                        : model_.weights().get("output_norm");
    auto h_next = ops::rms_norm(cur, norm_w, cfg.rms_norm_eps);

    auto lm_head = mtp_shared_head_head_ ? mtp_shared_head_head_
                                         : model_.weights().get("output_weight");
    if (lm_head && lm_head->device() != h_next->device()) {
        h_next->to_device(lm_head->device());
    }
    auto logits = ops::matmul_transB(h_next, lm_head);

    // Keep h_next on the caller's device for chaining.
    if (logits && logits->device() != dev) logits->to_device(dev);
    if (h_next->device() != dev) h_next->to_device(dev);

    if (logits_out) *logits_out = logits;
    return h_next;
}

}  // namespace forge
