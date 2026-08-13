#include "forge/engines/gemma4_engine.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

#include "forge/cuda_kernels.h"
#include "forge/inference/forward_request.h"
#include "forge/inference/layers/gemma4_moe.h"
#include "forge/inference/tensor_device_utils.h"
#include "forge/inference_batch.h"
#include "forge/logger.h"
#include "forge/operators.h"
#include "forge/perf_profiler.h"

namespace forge {

Gemma4Engine::Gemma4Engine(Model& model, InferenceContext& ctx)
    : TransformerEngine(model, ctx), attention_(kv_cache_) {
    if (!init_weights()) {
        throw std::runtime_error("Gemma4Engine: failed to initialize weights");
    }
}

bool Gemma4Engine::init_weights() {
    if (!weights_.init(model_.weights(), model_.config())) {
        return false;
    }
    // Load proportional RoPE frequency factors for full-attention layers.
    // The GGUF stores rope_freqs per full-attention layer (e.g., blk.4.rope_freqs.weight),
    // NOT as a global tensor. Layer 0 is SWA and has no rope_freqs, so we must search
    // for the first full-attention layer that has it.
    TensorPtr rope_freqs = model_.weights().get("rope_freqs");
    if (!rope_freqs) {
        const auto& cfg = model_.config();
        for (int i = 0; i < cfg.num_layers; ++i) {
            bool is_swa = (i < (int)cfg.swa_layers.size() && cfg.swa_layers[i] == 1);
            if (!is_swa) {
                rope_freqs = weights_.layers[i].get("rope_freqs");
                if (rope_freqs) break;
            }
        }
    }
    if (rope_freqs) {
        if (is_quantized_type(rope_freqs->dtype())) {
            rope_freqs = ops::dequantize_weight(rope_freqs);
        }
        LOG_INFO("rope_freqs loaded: shape=" + std::to_string(rope_freqs->shape()[0]));
    } else {
        LOG_WARN("rope_freqs NOT found in weights for Gemma4");
    }
    attention_.set_rope_freqs(rope_freqs);

    return true;
}

void Gemma4Engine::init_kv_cache(const ModelConfig& cfg) {
    if (kv_cache_initialized())
        return;

    int kv_max_seq = cfg.max_seq_len;
    const int KV_MAX_SEQ_CAP = 4096;
    if (kv_max_seq > KV_MAX_SEQ_CAP) {
        LOG_INFO("Capping KV cache max_seq_len from " + std::to_string(kv_max_seq) + " to " +
                 std::to_string(KV_MAX_SEQ_CAP) + " to avoid OOM");
        kv_max_seq = KV_MAX_SEQ_CAP;
    }

    DeviceType kv_dev;
    if (ctx_.params().offload_kqv) {
        kv_dev = (gpu_layers_ >= cfg.num_layers) ? DeviceType::CUDA : DeviceType::CPU;
    } else {
        kv_dev = DeviceType::CPU;
    }

    // Gemma4: per-layer KV cache with different head dimensions
    // Full-attention layers use head_dim, SWA layers use head_dim_swa
    // SWA layers without own KV reuse the nearest full-attention layer's KV
    std::vector<int> kv_dims(cfg.num_layers, 0);
    for (int i = 0; i < cfg.num_layers; ++i) {
        if (i < cfg.n_layer_kv_from_start) {
            bool is_swa = (i < (int)cfg.swa_layers.size() && cfg.swa_layers[i] == 1);
            int hd = is_swa ? cfg.head_dim_swa : cfg.head_dim;
            int n_kv = is_swa ? cfg.num_kv_heads_swa : cfg.num_kv_heads;
            kv_dims[i] = n_kv * hd;
        }
        // else: no own KV, will reuse from another layer; kv_dims[i] stays 0
    }

    LOG_INFO("KV cache init (Gemma4 per-layer): layers=" + std::to_string(cfg.num_layers) +
             ", max_seq_len=" + std::to_string(kv_max_seq) +
             ", dev=" + (kv_dev == DeviceType::CUDA ? "CUDA" : "CPU") +
             ", head_dim=" + std::to_string(cfg.head_dim) +
             ", head_dim_swa=" + std::to_string(cfg.head_dim_swa));

    kv_cache_.init_per_layer(cfg.num_layers, kv_dims, kv_max_seq, kv_dev);
    // Phase 6: set per-layer policies (replaces set_ring_buffer calls).
    // SlidingWindow layers (swa_layers[i] == 1) use ring buffer eviction;
    // full-attention layers grow linearly without windowing.
    if (cfg.n_swa > 0) {
        std::vector<KVLayerPolicy> policies(cfg.num_layers, KVLayerPolicy::Full);
        for (int i = 0; i < cfg.num_layers; ++i) {
            if (i < (int)cfg.swa_layers.size() && cfg.swa_layers[i] == 1)
                policies[i] = KVLayerPolicy::SlidingWindow;
        }
        kv_cache_.set_layer_policies(policies, cfg.n_swa);
    }
    // Place each layer's KV cache on the corresponding device
    if (!layer_devices_.empty()) {
        kv_cache_.set_layer_devices(layer_devices_);
    }
    set_kv_cache_initialized(true);

    LOG_INFO("KV cache initialized successfully, actual size: " +
             std::to_string(kv_cache_.nbytes() / (1024 * 1024)) + " MB");
}

TensorPtr Gemma4Engine::forward_request(const ForwardRequest& req) {
    const auto& cfg = model_.config();

    init_kv_cache(cfg);

    DeviceType first_dev = layer_device(0);
    auto ids_on_dev = transfer_hidden(req.input_ids, first_dev);
    auto token_emb = model_.weights().get("token_embedding");

    auto hidden = embedding_.embed(ids_on_dev, token_emb, weights_, cfg, first_dev, req.n_tokens);

    auto logits = forward_layers(hidden, req);
    apply_logit_postprocess(logits, cfg);
    return logits;
}

void Gemma4Engine::apply_logit_postprocess(TensorPtr& logits, const ModelConfig& cfg) {
    if (!logits) return;

    bool has_softcap = (cfg.f_final_logit_softcapping > 0.0f);
    bool has_suppress = !cfg.suppress_tokens.empty();
    if (!has_softcap && !has_suppress) return;

    PERF_SCOPE("forward/logit_softcap");
    int vocab_size = logits->shape().back();

#ifdef USE_CUDA
    if (logits->device() == DeviceType::CUDA) {
        // GPU path: softcap (optional) + suppress in one kernel
        float cap = has_softcap ? cfg.f_final_logit_softcapping : 1.0f;
        const int* d_suppress = nullptr;
        int n_suppress = 0;
        if (has_suppress) {
            // Lazy init: cache suppress tokens on GPU (first call only)
            if (!suppress_tokens_gpu_) {
                int n_sup = static_cast<int>(cfg.suppress_tokens.size());
                auto cpu_tokens = std::make_shared<Tensor>(
                    DataType::INT32, std::vector<int64_t>{n_sup}, DeviceType::CPU);
                int* cpu_data = static_cast<int*>(cpu_tokens->data());
                for (int i = 0; i < n_sup; ++i) cpu_data[i] = cfg.suppress_tokens[i];
                suppress_tokens_gpu_ = std::make_shared<Tensor>(
                    DataType::INT32, std::vector<int64_t>{n_sup}, DeviceType::CUDA);
                suppress_tokens_gpu_->copy_from(*cpu_tokens);
            }
            n_suppress = static_cast<int>(cfg.suppress_tokens.size());
            d_suppress = static_cast<const int*>(suppress_tokens_gpu_->data());
        }
        cuda::launch_logit_softcap(static_cast<float*>(logits->data()), cap, has_softcap, d_suppress,
                                   n_suppress, vocab_size);
        return;
    }
#endif
    // CPU path: softcap is fused into the sampler (softcap + max/argmax in one
    // traversal), so only suppress tokens are handled here.
    if (!has_suppress) return;
    logits = ensure_cpu(logits);
    float* data = static_cast<float*>(logits->data());
    int num_rows = static_cast<int>(logits->numel()) / vocab_size;
    for (int tok_id : cfg.suppress_tokens) {
        if (tok_id >= 0 && tok_id < vocab_size) {
            for (int r = 0; r < num_rows; ++r) {
                data[r * vocab_size + tok_id] = -INFINITY;
            }
        }
    }
}

TensorPtr Gemma4Engine::forward_layer(const TensorPtr& hidden,
                                      const LayerExecutionContext& lctx) {
    const auto& cfg = lctx.config;
    const auto& lw = lctx.weights;

    // ---- Attention ----
    TensorPtr normed;
    {
        PERF_SCOPE("layer/attn_norm");
        normed = ops::rms_norm(hidden, lw.attn_norm(), cfg.rms_norm_eps);
    }

    auto attn_out = attention_.attend(normed, lctx);

    {
        PERF_SCOPE("layer/attn_proj");
        attn_out = ops::matmul_transB(attn_out, lw.wo());
    }
    if (lw.attn_post_norm()) {
        PERF_SCOPE("layer/attn_post_norm");
        attn_out = ops::rms_norm(attn_out, lw.attn_post_norm(), cfg.rms_norm_eps);
    }

    auto attn_residual = ops::add(hidden, attn_out);

    // ---- FFN ----
    TensorPtr ffn_out;
    if (lw.ffn_gate_inp()) {
        ffn_out = Gemma4Moe::apply(attn_residual, lctx);
    } else {
        // Standard GeGLU FFN
        PERF_SCOPE("layer/ffn");
        auto ffn_normed = ops::rms_norm(attn_residual, lw.ffn_norm(), cfg.rms_norm_eps);

#ifdef USE_CUDA
        // Fused gate+up+GeGLU kernel for Q4_K weights on GPU
        const int seq_len = lctx.seq_len();
        auto w1_tensor = lw.w1();
        auto w3_tensor = lw.w3();
        if (seq_len == 1 && lctx.device == DeviceType::CUDA && w1_tensor && w3_tensor &&
            w1_tensor->device() == DeviceType::CUDA && w3_tensor->device() == DeviceType::CUDA &&
            w1_tensor->dtype() == DataType::Q4_K && w3_tensor->dtype() == DataType::Q4_K) {
            // matmul_transB convention: b shape is [N, K] = [intermediate_dim, hidden_dim]
            int K = cfg.hidden_dim;
            int N = static_cast<int>(w1_tensor->shape()[0]);
            auto gated = std::make_shared<Tensor>(DataType::FP32,
                                                  std::vector<int64_t>{seq_len, N},
                                                  DeviceType::CUDA);
            cuda::launch_ffn_up_fused_q4_k_geglu(static_cast<const float*>(ffn_normed->data()),
                                                 w1_tensor->data(), w3_tensor->data(),
                                                 static_cast<float*>(gated->data()), K, N);
            ffn_out = ops::matmul_transB(gated, lw.w2());
        } else
#endif
        {
            auto gate = ops::matmul_transB(ffn_normed, lw.w1());
            auto up = ops::matmul_transB(ffn_normed, lw.w3());
            auto gated = ops::gelu_multiply(gate, up);
            ffn_out = ops::matmul_transB(gated, lw.w2());
        }
    }

    if (lw.post_ffn_norm()) {
        PERF_SCOPE("layer/post_ffn_norm");
        ffn_out = ops::rms_norm(ffn_out, lw.post_ffn_norm(), cfg.rms_norm_eps);
    }

    auto output = ops::add(attn_residual, ffn_out);

    // ---- Per-layer embeddings ----
    output = embedding_.apply_per_layer(output, lctx);

    // ---- Layer output scale ----
    if (lw.layer_out_scale()) {
        auto scale_cpu = ensure_cpu(lw.layer_out_scale());
        float s = static_cast<const float*>(scale_cpu->data())[0];
#ifdef USE_CUDA
        if (output->device() == DeviceType::CUDA) {
            cuda::launch_scale(static_cast<float*>(output->data()), s,
                               static_cast<int>(output->numel()));
        } else
#endif
        {
            output = ensure_cpu(output);
            float* out_data = static_cast<float*>(output->data());
            int n = static_cast<int>(output->numel());
            for (int i = 0; i < n; ++i) {
                out_data[i] *= s;
            }
        }
    }

    return output;
}

TensorPtr Gemma4Engine::forward_batch(const InferenceBatch& batch) {
    // Gemma4 has custom embedding scaling + per-layer projection that
    // the flat hidden state path doesn't handle. Fall back to per-sequence forward().
    if (batch.empty())
        return nullptr;

    const auto& cfg = model_.config();
    init_kv_cache(cfg);

    int n_seq = batch.size();
    int vocab_size = -1;

    // Process all sequences and collect logits
    struct SeqLogits { int idx; std::vector<float> data; };
    std::vector<SeqLogits> all_logits;

    for (int i = 0; i < n_seq; i++) {
        const auto& item = batch.items[i];
        int seq_len = static_cast<int>(item.tokens.size());
        auto input_ids = std::make_shared<Tensor>(DataType::INT32,
                                                   std::vector<int64_t>{seq_len}, DeviceType::CPU);
        std::memcpy(input_ids->data(), item.tokens.data(), seq_len * sizeof(int32_t));

        auto logits = forward_request(ForwardRequest::from_ids(input_ids, item.start_pos, item.seq_id));
        if (!logits)
            continue;

        TensorPtr logits_cpu = ensure_cpu(logits);

        vocab_size = static_cast<int>(logits_cpu->shape().back());
        int seq_len_out = static_cast<int>(logits_cpu->shape()[0]);
        const float* src = static_cast<const float*>(logits_cpu->data()) +
                           (seq_len_out - 1) * vocab_size;

        SeqLogits sl;
        sl.idx = i;
        sl.data.assign(src, src + vocab_size);
        all_logits.push_back(std::move(sl));
    }

    if (all_logits.empty() || vocab_size <= 0)
        return nullptr;

    auto result = std::make_shared<Tensor>(DataType::FP32,
                                            std::vector<int64_t>{n_seq, vocab_size},
                                            DeviceType::CPU);
    float* dst = static_cast<float*>(result->data());
    std::memset(dst, 0, n_seq * vocab_size * sizeof(float));

    for (auto& sl : all_logits) {
        std::memcpy(dst + sl.idx * vocab_size, sl.data.data(), vocab_size * sizeof(float));
    }

    return result;
}

}  // namespace forge
