#include "forge/engines/dflash_engine.h"

#include <cstring>

#include "forge/inference/layers/ffn_executor.h"
#include "forge/inference/layers/norm_executor.h"
#include "forge/logger.h"
#include "forge/operators.h"

namespace forge {

DflashEngine::DflashEngine(Model& model, InferenceContext& ctx)
    : GenericEngine(model, ctx),
      dflash_attn_(kv_cache_, kv_memory_ ? &kv_memory_->storage() : nullptr) {}

TensorPtr DflashEngine::forward_request(const ForwardRequest& req) {
    init_kv_cache(model_.config());
    DeviceType first_dev = layer_device(0);
    TensorPtr hidden;
    if (req.input_embeddings) {
        hidden = transfer_hidden(req.input_embeddings, first_dev);
    } else {
        if (!target_) {
            throw std::runtime_error(
                "DflashEngine::forward_request: target not set, "
                "cannot embed tokens");
        }
        auto ids = transfer_hidden(req.input_ids, first_dev);
        hidden = embed_via_target(ids, first_dev);
    }
    // Reuse the base forward_layers: it runs the draft layers (our forward_layer
    // override) and the final norm + lm_head projection (weights_.output_weight
    // is borrowed from the target via set_target()).
    return TransformerEngine::forward_layers(hidden, req);
}

TensorPtr DflashEngine::embed_via_target(const TensorPtr& ids, DeviceType dev) {
    auto ids_dev = transfer_hidden(ids, dev);
    auto& tw = static_cast<TransformerEngine*>(target_)->weights();
    return ops::embedding(tw.token_embedding, ids_dev, tw.token_embedding_fp32);
}

TensorPtr DflashEngine::encode(const TensorPtr& target_layer_hiddens) {
    const auto& cfg = model_.config();
    TensorPtr h = ops::matmul_transB(target_layer_hiddens, weights_.dflash_fc);
    if (weights_.dflash_output_norm_enc) {
        h = NormExecutor::apply(h, weights_.dflash_output_norm_enc, nullptr, NormType::RMSNorm,
                                cfg.rms_norm_eps);
    }
    context_feature_ = h;
    return h;
}

void DflashEngine::precompute_context_kv(const TensorPtr& prefix_embd, int64_t start_pos) {
    kv_cache_.reset();
    const auto& cfg = model_.config();
    const int prefix_len = static_cast<int>(prefix_embd->shape()[0]);
    const int num_heads = cfg.num_heads;
    const int num_kv_heads = cfg.num_kv_heads;
    const int head_dim = cfg.head_dim;
    DeviceType dev = prefix_embd->device();
    for (int layer = 0; layer < cfg.num_layers; ++layer) {
        const auto& lw = weights_.layers[layer];
        TensorPtr K = ops::matmul_transB(prefix_embd, lw.wk());
        TensorPtr V = ops::matmul_transB(prefix_embd, lw.wv());
        // RoPE applies to K (and Q); build a dummy Q so we can reuse apply() and
        // take the rotated K. V is intentionally left un-rotated.
        TensorPtr q_dummy = std::make_shared<Tensor>(
            K->dtype(), std::vector<int64_t>{prefix_len, num_heads * head_dim}, dev);
        q_dummy->zero_();
        auto rope = dflash_rope_.apply(q_dummy, K, cfg, start_pos, prefix_len, dev);
        kv_cache_.update(layer, 0, start_pos, rope.k_rope, V, prefix_len);
    }
}

void DflashEngine::precompute_context_kv(const std::vector<int32_t>& prefix_ids,
                                         int64_t start_pos) {
    if (!target_) {
        throw std::runtime_error(
            "DflashEngine::precompute_context_kv(ids): target not set, cannot embed prefix");
    }
    DeviceType dev = layer_device(0);
    const int64_t n = static_cast<int64_t>(prefix_ids.size());
    auto ids =
        std::make_shared<Tensor>(DataType::INT32, std::vector<int64_t>{1, n}, DeviceType::CPU);
    std::memcpy(ids->data(), prefix_ids.data(), static_cast<size_t>(n) * sizeof(int32_t));
    TensorPtr emb = embed_via_target(ids, dev);
    precompute_context_kv(emb, start_pos);
}

TensorPtr DflashEngine::decode(const std::vector<int32_t>& query_ids, int64_t start_pos) {
    DeviceType dev = layer_device(0);
    const int64_t n = static_cast<int64_t>(query_ids.size());
    TensorPtr ids =
        std::make_shared<Tensor>(DataType::INT32, std::vector<int64_t>{1, n}, DeviceType::CPU);
    std::memcpy(ids->data(), query_ids.data(), static_cast<size_t>(n) * sizeof(int32_t));
    TensorPtr emb = embed_via_target(ids, dev);
    // Query-block decode is non-causal: every query attends to the full prefix
    // (injected via precompute_context_kv) plus the entire block.
    decode_noncausal_ = true;
    auto req = ForwardRequest::from_embedding(emb, static_cast<int>(n), start_pos, 0);
    TensorPtr logits = forward_request(req);
    decode_noncausal_ = false;
    return logits;
}

TensorPtr DflashEngine::forward_layer(const TensorPtr& hidden, const LayerExecutionContext& lctx) {
    const auto& cfg = lctx.config;
    const int layer_idx = lctx.layer_idx;
    const int seq_len = lctx.seq_len();
    const int64_t start_pos = lctx.start_pos();
    const DeviceType dev = lctx.device.type;
    const int num_heads = cfg.num_heads;
    const int num_kv_heads = cfg.num_kv_heads;
    const int head_dim = cfg.head_dim;
    const auto& lw = lctx.weights;

    const NormType norm_type = cfg.norm_type;
    const float eps = (norm_type == NormType::LayerNorm) ? cfg.layer_norm_eps : cfg.rms_norm_eps;
    const bool has_qkv_bias = (lw.bq() && lw.bq()->numel() > 0);

    // 1. Pre-attention norm
    TensorPtr pre_attn_norm = NormExecutor::apply(hidden, lw.attn_norm(), nullptr, norm_type, eps);

    // 2. QKV projection
    auto qkv = AttentionExecutor::project_qkv(pre_attn_norm, lw, has_qkv_bias, dev, seq_len);
    TensorPtr q = qkv.q, k = qkv.k, v = qkv.v;

    // 3. QK-Norm (per-head RMSNorm)
    if (cfg.use_qk_norm) {
        if (lw.attn_q_norm()) {
            q = NormExecutor::apply_qk_norm(q, lw.attn_q_norm(), num_heads, head_dim,
                                            cfg.rms_norm_eps, dev);
        }
        if (lw.attn_k_norm()) {
            k = NormExecutor::apply_qk_norm(k, lw.attn_k_norm(), num_kv_heads, head_dim,
                                            cfg.rms_norm_eps, dev);
        }
    }

    // 4. RoPE
    auto rope = dflash_rope_.apply(q, k, cfg, start_pos, seq_len, dev);
    TensorPtr q_rope = rope.q_rope, k_rope = rope.k_rope;

    // 5. KV cache update (contiguous; the drafter owns a non-paged cache)
    kv_cache_.update(layer_idx, lctx.seq_id(), start_pos, k_rope, v, seq_len);

    // 6. Attention. Non-causal during the query-block decode.
    const bool causal = !decode_noncausal_;
    TensorPtr attn_out =
        dflash_attn_.attend(q_rope, cfg, layer_idx, seq_len, dev, nullptr, lctx.seq_id(), causal);

    // 7. Attention output projection
    TensorPtr attn_proj = ops::matmul_transB(attn_out, lw.wo());

    // 8/9. Sequential residual: norm -> attn -> residual -> ffn -> residual
    TensorPtr hidden_after_attn = ops::add(hidden, attn_proj);
    TensorPtr ffn_normed =
        NormExecutor::apply(hidden_after_attn, lw.ffn_norm(), nullptr, norm_type, eps);
    TensorPtr ffn_out = FfnExecutor::apply(ffn_normed, hidden_after_attn, cfg, lw, seq_len, dev);
    return ops::add(hidden_after_attn, ffn_out);
}

bool DflashEngine::has_markov_head() const {
    return weights_.dflash_markov_embed && weights_.dflash_markov_bias;
}

static TensorPtr ensure_cpu(const TensorPtr& t) {
    if (!t)
        return nullptr;
    if (t->device() != DeviceType::CPU)
        t->to_device(DeviceType::CPU);  // in-place
    return t;
}

TensorPtr DflashEngine::markov_embed() const {
    return ensure_cpu(weights_.dflash_markov_embed);
}
TensorPtr DflashEngine::markov_bias() const {
    return ensure_cpu(weights_.dflash_markov_bias);
}
TensorPtr DflashEngine::draft_id_map() const {
    return ensure_cpu(weights_.dflash_draft_id_to_target_id);
}

std::vector<int32_t> dspark_sequential_sample(const std::vector<const float*>& rows, int vocab,
                                              const float* markov_embed, int markov_d,
                                              const float* markov_bias, const int32_t* draft_id_map,
                                              int32_t anchor_token, int n) {
    std::vector<int32_t> out;
    out.reserve(n);
    int32_t prev = anchor_token;  // target vocab id (markov_embed is keyed by it)
    for (int j = 0; j < n; ++j) {
        const float* row = (j < static_cast<int>(rows.size())) ? rows[j] : rows.back();
        int best = 0;
        float best_score = row[0];
        for (int v = 0; v < vocab; ++v) {
            float score = row[v];
            if (markov_embed && markov_bias && markov_d > 0) {
                const float* eb = markov_embed + static_cast<int64_t>(prev) * markov_d;
                const float* wb = markov_bias + static_cast<int64_t>(v) * markov_d;
                float b = 0.0f;
                for (int k = 0; k < markov_d; ++k)
                    b += wb[k] * eb[k];
                score += b;
            }
            if (score > best_score) {
                best_score = score;
                best = v;
            }
        }
        if (draft_id_map)
            best = draft_id_map[best];  // reduced draft vocab -> target
        out.push_back(best);
        prev = best;  // next markov step keys on the (mapped) target vocab id
    }
    return out;
}

}  // namespace forge
