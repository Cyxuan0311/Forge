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

#include <unordered_map>

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
    // Incremental context-KV append (vLLM DFlash style): write only the newly
    // confirmed tokens' K/V at [start_pos, start_pos + ids.size()) WITHOUT
    // resetting the cache. The provider keeps the drafter KV persistent across
    // speculation rounds so per-round cost is O(delta) instead of O(full prefix).
    // Rows left behind by the previous round's query block are overwritten by
    // the append (accepted drafts occupy the same positions) or by the next
    // query block before attention reads them.
    void append_context_kv(const std::vector<int32_t>& delta_ids, int64_t start_pos);
    // Decoder: [anchor, MASK*N] query block -> [N+1, vocab] logits (non-causal).
    TensorPtr decode(const std::vector<int32_t>& query_ids, int64_t start_pos);

    bool is_dspark() const { return is_dspark_; }
    void set_dspark(bool v) { is_dspark_ = v; }

    // DFlash2 drafter (selector weights present). The decoder additionally
    // applies the block-causal dynamic conv and packs a top-k lattice for the
    // CPU greedy walk instead of backend sampling.
    bool is_dflash2() const;
    // DFlash2 selector lattice from the last decode() (nullptr otherwise). Row i
    // (query-block position i) packs the greedy-walk state:
    //   [0, top_k)                  candidate token ids (float-cast, score-descending)
    //   [top_k, top_k + top_k^2)    transition scores, indexed [pred][succ]
    // Position 0 (the anchor) is zero-filled; position 1 is scored against the
    // anchor token id, later positions against the previous position's
    // candidate list (see speculative.cpp's greedy lattice walk).
    TensorPtr dflash2_lattice() const;

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
    // 将"已编码 context feature" (append: encode() 的 fc 输出; precompute: raw
    // prefix) 统一应用 encoder 输出 norm (output_norm_enc, 若存在) 后, 逐层投影
    // 为 K/V, 再逐层 k_norm (attn_k_norm, 若 use_qk_norm) + RoPE, 写入
    // [start_pos, ...) (不含 reset)
    void write_context_kv(const TensorPtr& prefix_embd, int64_t start_pos);
    // DFlash2 conv/selector 权重归一化: matmul_transB 的 CPU 路径没有 FP16
    // kernel, conv/selector 的标量读取也需要原始浮点, 因此把 FP16/量化权重一次性
    // 转成 FP32 并按源张量缓存 (FP32 权重原样透传, 任意设备)。
    TensorPtr dflash2_weight_f32(const TensorPtr& w);
    // DFlash2 selector 专用的主机 FP32 权重访问器。selector lattice 是 CPU-only
    // 算法 (裸读主机指针 + CPU GEMV gate), 而源权重落在 last_layer_dev (全 GPU
    // drafter 时为 CUDA)。dflash2_weight_f32 的结果保留在源设备上供 CUDA
    // conv-proj matmul 使用, 这里按源权重另存一份主机 FP32 副本并缓存 —— 跨设备
    // 搬运每个权重只发生一次, 而不是每个 decode 步一次。
    TensorPtr dflash2_selector_weight_cpu(const TensorPtr& w);

    InferenceEngine* target_ = nullptr;
    AttentionExecutor dflash_attn_;
    RopeExecutor dflash_rope_;
    TensorPtr context_feature_;      // cached encoder output [seq, hidden]
    bool decode_noncausal_ = false;  // query-block decode uses non-causal attention
    bool is_dspark_ = false;
    std::unordered_map<const Tensor*, TensorPtr> dflash2_f32_;  // conv/selector weights → FP32
    std::unordered_map<const Tensor*, TensorPtr> dflash2_f32_cpu_;  // selector weights → host FP32
    TensorPtr dflash2_lattice_;  // last decode()'s selector lattice (CPU FP32)
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

// DFlash2 block-causal dynamic depthwise conv (llama.cpp build_dflash2_conv,
// row-major port; exposed for unit tests). Applied twice per attn/ffn sublayer:
// side 0 transforms the block input, side 1 the sublayer output before the
// residual add. Activations/weights may be FP32 or FP16 on any device; the
// result matches the input dtype/device.
TensorPtr build_dflash2_conv(const TensorPtr& hidden, const TensorPtr& dynamic,
                             const TensorPtr& base_f32, int side, const ModelConfig& cfg);

// DFlash2 selector lattice builder (llama.cpp build_dflash2_selector CPU port;
// exposed for unit tests). Packs per-position top-k candidate ids and
// [pred][succ] transition scores consumed by dflash2_lattice_walk. CPU-only
// algorithm: all tensor inputs are staged to host FP32 at entry (FP32/FP16 on
// any device accepted; quantized weights must be pre-converted via
// dflash2_weight_f32), so a fully-GPU drafter cannot trigger cross-device
// dereferences here. Hot-path callers pass cached host tensors and pay nothing.
TensorPtr build_dflash2_selector(const TensorPtr& logits, const TensorPtr& t_embd,
                                 const ModelConfig& cfg, const TensorPtr& sel_prev,
                                 const TensorPtr& sel_next, const TensorPtr& sel_hidden,
                                 int32_t anchor_id);

// DFlash2 CPU greedy lattice walk (pure, deterministic, testable; mirrors
// speculative.cpp's is_dflash2 branch). `lattice` is the packed selector output
// from the last decode() — row-major [n_block_tokens, row_width], row_width >=
// top_k + top_k*top_k. Position 0 is the anchor: the walk starts with
// predecessor index 0 and for each position i in [1, n_block_tokens) reads
//   scores = row(i) + top_k + predecessor * top_k
// picks the argmax successor, applies the p_min softmax-confidence gate
// (1 / sum(exp(s_k - s_max)) < p_min → stop), and emits the candidate token id
// (int32) row(i)[predecessor] before advancing. Returns up to n_block_tokens-1
// draft ids (empty when the first position already fails the p_min gate).
std::vector<int32_t> dflash2_lattice_walk(const float* lattice, int64_t row_width, int top_k,
                                          int n_block_tokens, float p_min);

}  // namespace forge
