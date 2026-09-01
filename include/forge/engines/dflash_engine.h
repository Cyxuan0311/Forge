#pragma once

// DFlash / DSPark standalone drafter for speculative decoding.
//
// A lightweight GQA transformer that shares the target model's token embedding
// and lm_head. Its encoder fuses target hidden states (taken from target_layers)
// into a context feature; its decoder consumes a shared token-embedding batch
// (with an [anchor, MASK*N] query block) and produces draft logits via the
// target lm_head.
//
// The draft engine must be paired with a target engine via set_target(); without
// it the token embedding / lm_head are unavailable and embedding-based paths fail.
// The drafter is intentionally lightweight: it reuses GenericEngine's layer
// operators but drives a non-causal query-block decode (DFlash decoder core).

#include "forge/engines/generic_engine.h"
#include "forge/inference/layers/attention_executor.h"
#include "forge/inference/layers/rope_executor.h"

namespace forge {

class DflashEngine : public GenericEngine {
public:
    explicit DflashEngine(Model& model, InferenceContext& ctx);

    std::string name() const override { return "dflash"; }

    // Pair with the target engine (shares token_embedding + lm_head). Borrowed
    // pointers are copied into weights_ so the base forward_layers / output path
    // works unchanged.
    void set_target(InferenceEngine* target) {
        target_ = target;
        if (target_) {
            auto& tw = static_cast<TransformerEngine*>(target_)->weights();
            weights_.token_embedding = tw.token_embedding;
            weights_.token_embedding_fp32 = tw.token_embedding_fp32;
            weights_.output_weight = tw.output_weight;
            weights_.output_norm = tw.output_norm;
        }
    }
    InferenceEngine* target() const { return target_; }

    // Embedding entry: skip lookup when embeddings are supplied, else use the
    // target's shared token embedding.
    TensorPtr forward_request(const ForwardRequest& req) override;

    // Encoder: target multi-layer hiddens -> context feature [seq, hidden].
    TensorPtr encode(const TensorPtr& target_layer_hiddens);
    // Context-KV precompute: project target prefix embedding into the draft KV cache.
    void precompute_context_kv(const TensorPtr& prefix_embd, int64_t start_pos);
    // Convenience: embed the prefix token ids via the shared target embedding,
    // then inject (start_pos is the committed-prefix length / KV offset).
    void precompute_context_kv(const std::vector<int32_t>& prefix_ids, int64_t start_pos);
    // Decoder: [anchor, MASK*N] query block -> [N+1, vocab] logits (non-causal).
    TensorPtr decode(const std::vector<int32_t>& query_ids, int64_t start_pos);

    bool is_dspark() const { return is_dspark_; }
    void set_dspark(bool v) { is_dspark_ = v; }

    // DSPark sequential Markov head. Returns nullptr when the drafter has no
    // markov weights (plain DFlash). The returned tensors are on the CPU.
    bool has_markov_head() const;
    TensorPtr markov_embed() const;  // [target_vocab, d] or nullptr
    TensorPtr markov_bias() const;   // [target_vocab, d] or nullptr
    TensorPtr draft_id_map() const;  // [draft_vocab] int32 mapping or nullptr

protected:
    // Draft layer: standard GQA block, but attend() is non-causal during the
    // query-block decode (decode_noncausal_). Uses the engine-local executors so
    // we can inject the causal flag without disturbing GenericEngine.
    TensorPtr forward_layer(const TensorPtr& hidden, const LayerExecutionContext& lctx) override;

private:
    TensorPtr embed_via_target(const TensorPtr& ids, DeviceType dev);

    InferenceEngine* target_ = nullptr;
    AttentionExecutor dflash_attn_;
    RopeExecutor dflash_rope_;
    TensorPtr context_feature_;      // cached encoder output [seq, hidden]
    bool decode_noncausal_ = false;  // query-block decode uses non-causal attention
    bool is_dspark_ = false;
};

// DSPark sequential Markov sampling (pure, deterministic, testable).
//
// Given `n` logit rows (each [vocab] float, row-major in `rows`), samples one
// token per row left-to-right. For position j the previous token (target vocab
// id) is fed through markov_embed and its contribution is added to every vocab
// score via markov_bias, so the block acquires an intra-dependence:
//     score_j[v] = rows[j][v] + markov_bias[v] . markov_embed[prev_j]
//     prev_0 = anchor_token ; prev_{j+1} = sampled_j
// When `markov_embed` / `markov_bias` are null the previous token contributes
// nothing and each row is sampled greedily and independently (a DFlash-style
// fallback). When `draft_id_map` is non-null the draft logits are over a reduced
// draft vocabulary; the sampled draft id is mapped back to the target id before
// being emitted and before becoming `prev` for the next position.
std::vector<int32_t> dspark_sequential_sample(const std::vector<const float*>& rows, int vocab,
                                              const float* markov_embed, int markov_d,
                                              const float* markov_bias, const int32_t* draft_id_map,
                                              int32_t anchor_token, int n);

}  // namespace forge
