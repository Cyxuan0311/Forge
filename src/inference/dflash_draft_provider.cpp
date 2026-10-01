#include "forge/dflash_draft_provider.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>

#include "forge/arch_registry.h"
#include "forge/context.h"
#include "forge/engine.h"
#include "forge/engines/dflash_engine.h"
#include "forge/engines/transformer_engine.h"
#include "forge/gguf_model.h"
#include "forge/inference/forward_request.h"
#include "forge/logger.h"
#include "forge/model.h"
#include "forge/model_loader.h"
#include "forge/ninf_model.h"
#include "forge/perf_profiler.h"
#include "forge/sampler.h"

namespace forge {

namespace {

// Default DSpark Markov-bias candidate width when neither the CLI nor the
// drafter checkpoint requests one. See the resolution site in the Impl
// constructor for why this is equivalent to the full-vocabulary path rather
// than a quality/speed trade.
constexpr int kDefaultDSParkTopK = 64;

void register_runtime() {
    force_link_arch_registrations();
    auto& registry = ModelLoaderRegistry::instance();
    registry.register_loader(
        "gguf", []() -> std::unique_ptr<ModelLoader> { return std::make_unique<GgufModel>(); });
    registry.register_loader(
        "ninf", []() -> std::unique_ptr<ModelLoader> { return std::make_unique<NinfModel>(); });
}

std::unique_ptr<InferenceContext> make_context(Model& model, const ContextParams& target,
                                               int gpu_layers) {
    ContextParams params = target;
    params.gpu_layers = gpu_layers;
    params.speculative_config = SpeculativeConfig{};  // drafter must not recurse

    auto ctx = std::make_unique<InferenceContext>(model, params);
    auto engine = EngineRegistry::instance().create(model.config().arch_type, model, *ctx);
    if (!engine)
        return nullptr;
    if (auto* transformer = dynamic_cast<TransformerEngine*>(engine.get())) {
        transformer->set_gpu_layers(gpu_layers);
        transformer->set_kv_cache_dtype(params.kv_cache_dtype);
        // drafter 只需要 [已确认前缀 + query block] 的 KV, 按模型原生 262144
        // 全量分配毫无意义: 27B 目标场景实测 3GB 匿名内存挤占权重页缓存,
        // 整机 swap 颠簸。上限 8192 (约 328MB @fp32) 覆盖绝大多数 CLI 场景;
        // 超长上下文场景需另行放宽。
        transformer->set_kv_cache_max_seq_len(
            std::min(8192, model.config().max_seq_len));
    }
    ctx->set_engine(std::move(engine));
    return ctx;
}

float top_probability(const float* logits, int vocab_size) {
    float max_logit = logits[0];
    for (int i = 1; i < vocab_size; ++i)
        max_logit = std::max(max_logit, logits[i]);
    double sum = 0.0;
    for (int i = 0; i < vocab_size; ++i)
        sum += std::exp(static_cast<double>(logits[i] - max_logit));
    return sum > 0.0 ? static_cast<float>(1.0 / sum) : 0.0f;
}

// Leading `rows` rows of a 2D tensor, as an independent contiguous copy.
// A Tensor::view() would inherit the parent's nbytes() (a reshape, not a slice),
// which makes the downstream device transfer over-read, so materialize instead.
// The block is a handful of rows, so the copy is negligible.
std::shared_ptr<Tensor> leading_rows(const TensorPtr& t, int64_t rows) {
    const int64_t cols = t->shape()[1];
    auto out = std::make_shared<Tensor>(t->dtype(), std::vector<int64_t>{rows, cols}, t->device());
    const size_t row_bytes = static_cast<size_t>(cols) * dtype_size(t->dtype());
    const auto* src = static_cast<const uint8_t*>(t->data());
    auto* dst = static_cast<uint8_t*>(out->data());
    for (int64_t r = 0; r < rows; ++r)
        std::memcpy(dst + r * row_bytes, src + r * row_bytes, row_bytes);
    return out;
}

}  // namespace

struct DFlashDraftProvider::Impl {
    Model model;
    std::unique_ptr<InferenceContext> ctx;
    DflashEngine* engine = nullptr;     // raw pointer into ctx->engine()
    InferenceEngine* target = nullptr;  // borrowed target engine
    Sampler sampler;
    ContextParams target_params;
    std::string model_path;
    int gpu_layers = 0;
    int vocab_size = 0;
    float p_min = 0.0f;
    int n_spec = 5;
    int mask_token_id_ = 0;
    bool is_dspark = false;
    bool valid = false;

    // DSpark tuning (vLLM apply_bias_gathered + adaptive verification). Defaults
    // come from the drafter GGUF (ModelConfig); the CLI overrides when >= 0.
    int dspark_top_k_ = 0;                 // 0 = full-vocab markov bias
    float dspark_conf_threshold_ = 0.5f;   // confidence truncation threshold

    int64_t committed_len = 0;
    std::vector<int32_t> committed_ids;
    // drafter KV 中已写入上下文 K/V 的位置数 (持久化, 逐轮增量追加)
    int64_t kv_filled = 0;

    // ---- Target hidden handoff (P0: skip the redundant target replay) ----
    // `verify_hiddens_` is the encoder input for the tokens the target's most
    // recent verify forward already processed, [rows, n_embd_inp_enc] on CPU.
    // `verify_hiddens_pos_` is the kv_filled value the stash is anchored to: the
    // verify forward ran at exactly that position, so stash row j is the encoder
    // input for the token at kv_filled + j. Any advance of kv_filled (an append)
    // or an uncaptured confirmation invalidates it.
    std::shared_ptr<Tensor> verify_hiddens_;
    int64_t verify_hiddens_pos_ = -1;
    int64_t verify_hiddens_rows_ = 0;
    std::vector<int> capture_layers;
    // Set once the replay path has been used at least once, so the log can tell
    // "handoff never landed" apart from "handoff is simply not available yet".
    bool saw_handoff_ = false;
    bool warned_replay_ = false;

    bool handoff_covers(int64_t ctx_end) const {
        return verify_hiddens_ && verify_hiddens_pos_ == kv_filled &&
               ctx_end - kv_filled <= verify_hiddens_rows_;
    }

    Impl(const SpeculativeConfig& spec, InferenceContext& tctx, InferenceEngine& t,
         int target_vocab)
        : sampler([] {
              SamplerConfig config;
              config.do_sample = false;  // greedy proposals (verify is lossless)
              config.temperature = 0.0f;
              config.top_k = 0;
              config.top_p = 1.0f;
              config.repeat_penalty = 1.0f;
              return config;
          }()),
          target_params(tctx.params()),
          model_path(spec.draft_model_path),
          gpu_layers(spec.draft_gpu_layers >= 0
                         ? spec.draft_gpu_layers
                         : -1),  // drafter is tiny: default to full GPU so it
                                 // never inherits the target's partial -ngl and
                                 // avoids its own CPU/GPU boundary sync. Only
                                 // an explicit --spec-draft-ngl overrides this.
          vocab_size(target_vocab),
          p_min(spec.p_min),
          n_spec(std::max(1, spec.draft_n_spec)),
          mask_token_id_(spec.draft_mask_token_id >= 0 ? spec.draft_mask_token_id : -1),
          is_dspark(spec.draft_arch == "dspark"),
          dspark_top_k_(spec.dspark_top_k),
          dspark_conf_threshold_(spec.dspark_conf_threshold),
          target(&t) {
        register_runtime();

        if (model_path.empty() || !model.load(model_path, gpu_layers)) {
            LOG_WARN("DFlashDraftProvider: failed to load drafter model '" + model_path + "'");
            return;
        }

        // A drafter tensor sitting in host memory is not by itself a problem: the
        // handful of scalar tensors (norm epsilons, quantization scales) routinely
        // land on the host and cost nothing, since they are read once per forward
        // rather than streamed. Only report it when real weight volume is stranded.
        //
        // This is deliberately not the "your drafter is slow, check VRAM" warning. A
        // drafter can be 3-13x slower purely as a function of the configured context
        // while every one of its weights sits on the GPU, so blaming placement here
        // would send people chasing the wrong thing.
        {
            const size_t kNegligibleBytes = 1024 * 1024;  // 1 MiB
            size_t on_host = 0, on_device = 0;
            size_t host_bytes = 0;
            for (const auto& [name, t] : model.weights().all()) {
                if (!t)
                    continue;
                if (t->device() == DeviceType::CPU) {
                    ++on_host;
                    host_bytes += static_cast<size_t>(t->nbytes());
                } else {
                    ++on_device;
                }
            }
            if (host_bytes > kNegligibleBytes) {
                LOG_WARN("DFlashDraftProvider: drafter '" + model_path + "' has weight volume "
                         "in host memory: " + std::to_string(host_bytes / (1024 * 1024)) +
                         " MB across " + std::to_string(on_host) + " of " +
                         std::to_string(on_host + on_device) +
                         " tensors. Those are re-read every draft round, so this costs PCIe "
                         "bandwidth. Pass --spec-draft-ngl to place them on the GPU.");
            }
        }

        // DSpark tuning: the CLI flags (SpeculativeConfig) override the GGUF-derived
        // defaults parsed into ModelConfig. "<0 = not set" is what makes an
        // explicit 0 meaningful (0 top-k = full vocab, 0 conf = truncation off).
        dspark_top_k_ = spec.dspark_top_k >= 0 ? spec.dspark_top_k
                                              : model.config().dspark_markov_top_k;
        // Neither the CLI nor the checkpoint asked for a top-k, so default to a
        // truncated one instead of the full vocabulary.
        //
        // This is not an approximation. The Markov bias is a bounded logit shift
        // (see dspark_sequential_sample: the head emits roughly +/-3 logits), so it
        // can only reorder candidates whose base logits are within that band of the
        // leader. Any candidate below rank kDefaultDSParkTopK sits further than that
        // from the top token unless several hundred candidates are mutually tied, so
        // the argmax over the truncated set equals the argmax over the full set.
        //
        // The full-vocab path costs a 256-long dot product per vocabulary entry per
        // drafted position -- 130560 x 256 scalar MACs on the CPU. Measured on
        // MiniCPM5-2.6B-DSpark (vocab 130560, markov_d 256): 7.2 -> 11.7 tok/s, a
        // 1.62x speedup, with the acceptance rate bit-identical at 26.8% and the
        // same per-position breakdown (34.4% / 15.0%), which is the confirmation
        // that the two paths pick the same tokens.
        //
        // Pass --spec-dspark-topk 0 to force the full-vocabulary path.
        if (spec.dspark_top_k < 0 && dspark_top_k_ <= 0)
            dspark_top_k_ = kDefaultDSParkTopK;
        dspark_conf_threshold_ =
            spec.dspark_conf_threshold >= 0.0f ? spec.dspark_conf_threshold
                                              : model.config().dspark_confidence_threshold;

        // Encoder input layers: the drafter's own target_layers, overridden by
        // --spec-draft-target-layers. The executor reads this list to know which
        // target layers to capture during the verify forward.
        capture_layers = model.config().target_layers;
        if (!spec.draft_target_layers.empty()) {
            capture_layers.assign(spec.draft_target_layers.begin(),
                                  spec.draft_target_layers.end());
        }

        // MASK token id resolution chain (mirrors vLLM get_parallel_drafting_token_id):
        //   1. explicit --spec-draft-mask-id (kept from the member initializer);
        //   2. GGUF "dflash.parallel_drafting_token_id" (parsed into ModelConfig);
        //   3. GGUF "tokenizer.ggml.mask_token_id" (common in Qwen3 DFlash2 checkpoints).
        // Falling back to 0 produces a wrong parallel block (id 0 is an ordinary
        // token), so we bail out loudly instead of silently degrading acceptance.
        if (mask_token_id_ < 0) {
            if (model.config().parallel_drafting_token_id >= 0) {
                mask_token_id_ = model.config().parallel_drafting_token_id;
            } else {
                mask_token_id_ =
                    static_cast<int>(model.metadata_int("tokenizer.ggml.mask_token_id", -1));
            }
        }
        if (mask_token_id_ < 0) {
            LOG_WARN(
                "DFlashDraftProvider: no MASK token id in the drafter GGUF "
                "(dflash.parallel_drafting_token_id / tokenizer.ggml.mask_token_id). "
                "Pass --spec-draft-mask-id <id> explicitly.");
            return;
        }

        // Target must share token embeddings (embed_via_target static_casts).
        auto* t_transformer = dynamic_cast<TransformerEngine*>(target);
        if (!t_transformer) {
            LOG_WARN("DFlashDraftProvider: target engine is not a TransformerEngine");
            return;
        }
        // DFlash/DSPark drafter 借用 target 的 token embedding 与 lm_head
        // (见 DflashEngine::embed_via_target / set_target), 从不使用自身词表,
        // 因此 GGUF 头里的 vocab_size 元数据不可作为配对依据 (转换器常写入
        // 默认值 32000, 与 248320 的 Qwen 词表不符)。真正需要一致的是 hidden
        // 维度: target embedding 直接流入 drafter 各层。
        const int64_t target_hidden =
            t_transformer->weights().token_embedding
                ? static_cast<int64_t>(t_transformer->weights().token_embedding->shape()[1])
                : 0;
        if (target_hidden > 0 && model.config().hidden_dim != static_cast<int>(target_hidden)) {
            LOG_WARN("DFlashDraftProvider: drafter/target hidden size mismatch (" +
                     std::to_string(model.config().hidden_dim) + " vs " +
                     std::to_string(target_hidden) + ")");
            return;
        }
        if (target_hidden > 0 && model.config().target_hidden_size > 0 &&
            model.config().target_hidden_size != static_cast<int>(target_hidden)) {
            LOG_WARN("DFlashDraftProvider: drafter metadata target_hidden_size (" +
                     std::to_string(model.config().target_hidden_size) +
                     ") != actual target hidden (" + std::to_string(target_hidden) + ")");
            return;
        }

        ctx = make_context(model, target_params, gpu_layers);
        if (!ctx || !ctx->engine()) {
            LOG_WARN("DFlashDraftProvider: failed to create drafter inference context");
            return;
        }
        engine = dynamic_cast<DflashEngine*>(ctx->engine());
        if (!engine) {
            LOG_WARN("DFlashDraftProvider: engine is not a DflashEngine");
            return;
        }
        engine->set_target(target);
        // The GGUF decides the block layout: sample_from_anchor means the anchor
        // row is a sampled prediction, which is what the DSpark path assumes when
        // it indexes rows. A DSpark checkpoint declares general.architecture =
        // "dflash", so --spec-draft-arch dflash on one used to leave is_dspark
        // false while sample_from_anchor stayed true -- the two disagreed, the
        // row math was sized for the wrong layout, and draft() read past the
        // returned logits. Follow the checkpoint and warn on the disagreement.
        if (model.config().sample_from_anchor && !is_dspark) {
            LOG_WARN("DFlashDraftProvider: drafter sets sample_from_anchor but "
                     "--spec-draft-arch said dflash; using the checkpoint's DSpark "
                     "layout (pass --spec-draft-arch dspark to silence this)");
            is_dspark = true;
        }
        if (is_dspark) {
            engine->set_dspark(true);
            if (!engine->has_markov_head()) {
                // DSPark requires the sequential Markov head; a plain DFlash GGUF
                // does not carry markov_embed/markov_bias. Skip gracefully so the
                // run falls back to plain (non-speculative) decoding.
                LOG_WARN(
                    "DFlashDraftProvider: --spec-draft-arch dspark requested but the "
                    "drafter has no markov head; skipping DSPark");
                return;
            }
        }

        valid = true;
        LOG_INFO(std::string("DFlashDraftProvider: ready (") + (is_dspark ? "dspark" : "dflash") +
                 ") n_spec=" + std::to_string(n_spec) + " (incremental context KV)");
        if (capture_layers.empty()) {
            LOG_WARN("DFlashDraftProvider: no encoder target_layers resolved from the drafter "
                     "GGUF; the target will be re-forwarded every round to extract hiddens. "
                     "Pass --spec-draft-target-layers LIST to avoid it.");
        } else {
            std::string joined;
            for (int l : capture_layers) {
                if (!joined.empty())
                    joined += ",";
                joined += std::to_string(l);
            }
            LOG_INFO("DFlashDraftProvider: reusing target hidden states from the verify "
                     "forward (target_layers=" + joined + ")");
        }
    }

    void reset() {
        if (ctx)
            ctx->reset_kv_cache();
        committed_len = 0;
        committed_ids.clear();
        kv_filled = 0;
        clear_verify_hiddens();
    }

    void clear_verify_hiddens() {
        verify_hiddens_.reset();
        verify_hiddens_pos_ = -1;
        verify_hiddens_rows_ = 0;
    }
};

DFlashDraftProvider::DFlashDraftProvider(const SpeculativeConfig& spec, InferenceContext& ctx,
                                         InferenceEngine& target, int target_vocab)
    : impl_(nullptr),
      is_dspark_(spec.draft_arch == "dspark") {
    impl_ = std::make_unique<Impl>(spec, ctx, target, target_vocab);
}

DFlashDraftProvider::~DFlashDraftProvider() = default;

bool DFlashDraftProvider::valid() const {
    return impl_ && impl_->valid;
}

void DFlashDraftProvider::begin(const std::vector<int32_t>& prompt) {
    if (!valid())
        return;
    impl_->reset();
    impl_->committed_ids.assign(prompt.begin(), prompt.end());
    impl_->committed_len = static_cast<int64_t>(prompt.size());
}

const std::vector<int>* DFlashDraftProvider::capture_layers() const {
    if (!valid() || impl_->capture_layers.empty())
        return nullptr;
    return &impl_->capture_layers;
}

void DFlashDraftProvider::set_verify_hiddens(std::shared_ptr<Tensor> hiddens) {
    if (!valid())
        return;
    if (!hiddens) {
        clear_verify_hiddens();
        return;
    }
    // The verify forward started at the position this provider had just anchored
    // kv_filled to (the query block's own start position), so stash row j is the
    // encoder input for the token at kv_filled + j.
    impl_->verify_hiddens_ = std::move(hiddens);
    impl_->verify_hiddens_pos_ = impl_->kv_filled;
    impl_->verify_hiddens_rows_ = impl_->verify_hiddens_->shape()[0];
}

void DFlashDraftProvider::clear_verify_hiddens() {
    if (impl_)
        impl_->clear_verify_hiddens();
}

std::vector<int32_t> DFlashDraftProvider::draft(int32_t last_token, int n_draft) {
    std::vector<int32_t> result;
    if (!valid() || !impl_->engine || n_draft <= 0)
        return result;

    const int n = std::min(n_draft, impl_->n_spec);

    // anchor 吸收: prefill 产出的首个 token (t_0) 不经过 accept()/notify_confirmed
    // (begin() 只登记了 prompt), committed_ids 必须在此把它并入, 否则 drafter
    // 上下文从第二轮起整体错一位。executor 的 plain 步骤会逐 token notify, 此后
    // back()==last_token 恒成立, 本分支自然退化为何等判断, 幂等。
    if (impl_->committed_ids.empty() || impl_->committed_ids.back() != last_token) {
        impl_->committed_ids.push_back(last_token);
        impl_->committed_len += 1;
    }

    // query block 宽度固定 (上游 n_max 语义): 块是非因果 forward, 宽度收缩会让
    // 上一轮更宽 block 留下的 MASK 残行逃过本轮覆盖, 注意力读到垃圾 —— 因此宽度
    // 只由配置上限与训练块大小决定, 不随 adaptive n_draft 变化。返回值仍截断到
    // 本轮请求的 n, executor 的 adaptive/budget 语义不变。
    // DFlash2 的 conv/selector 还要求块宽不超过训练时的 dflash.block_size。
    const bool sfa = impl_->model.config().sample_from_anchor;
    // DSpark sample_from_anchor (vLLM): the query block has n_spec positions with
    // the anchor as position 0 and ALL n_spec rows are sampled (row i predicts
    // draft token i); the markov walk is seeded by the anchor token. Plain DFlash
    // (1+N) uses an anchor + n_block MASK rows, skipping row 0 (a bonus prediction
    // of last_token itself).
    int n_block = impl_->n_spec;
    if (impl_->model.config().dflash_block_size > 1)
        n_block = std::min(impl_->model.config().dflash_block_size - 1, impl_->n_spec);
    const int block = impl_->model.config().dflash_block_size;
    int query_len = sfa ? impl_->n_spec : n_block + 1;
    if (sfa && block > 1)
        query_len = std::min(query_len, block);  // drafter trained on blocks of `block`
    const int n_rows = sfa ? query_len : n_block;

    // The GGUF is the authority on the block layout (sample_from_anchor), and
    // everything below indexes rows against it. Guard the two places that can
    // otherwise read past the returned logits:
    //   - query block width must be >= 2 (an anchor plus at least one row),
    //   - the drafter is trained for exactly block_size; a narrower block leaves
    //     stale MASK rows visible (see the rollback note in DflashEngine::decode),
    //     and a wider one is not what the weights saw.
    if (query_len < 2) {
        LOG_WARN("DFlashDraftProvider: computed query block width " +
                 std::to_string(query_len) + " < 2; skipping draft");
        return result;
    }

    // 1. 增量注入上下文 K/V: delta = committed_ids[kv_filled, committed_len-1)。
    //    末位的 anchor 不经 target 重放镜像, 而是作为 query block 首行在它的真实
    //    位置 (committed_len-1) 写入 drafter KV。旧实现把 anchor 也算进 delta,
    //    导致 block 整体后移一位 (anchor 重复 + MASK 行位置 +1)。上一轮 query
    //    block 的残行 (旧 MASK 行) 已由 decode() 末尾的 KV 回滚清除, 本 delta
    //    从 anchor 位置起重写, 每轮 O(delta), 不再整前缀重建。
    const int64_t ctx_end = impl_->committed_len - 1;
    if (impl_->kv_filled < ctx_end) {
        const int64_t delta_n = ctx_end - impl_->kv_filled;
        // Fast path: the target's verify forward already produced the encoder
        // input for exactly these tokens (stash row j == position kv_filled + j),
        // so the context K/V is built without touching the target at all. This
        // removes a full 42-layer target forward per round -- measured at ~200ms
        // on a 2B/42-layer target, i.e. ~35% of the whole decode wall clock.
        if (impl_->handoff_covers(ctx_end)) {
            TensorPtr rows = leading_rows(impl_->verify_hiddens_, delta_n);
            {
                PERF_SCOPE("dspark/append_kv");
                impl_->engine->append_context_kv_from_hiddens(rows, impl_->kv_filled);
            }
            impl_->saw_handoff_ = true;
            impl_->clear_verify_hiddens();
        } else {
            // Fallback: replay the delta through the target. Costs one target
            // forward, but only on rounds where no captured hiddens are available
            // (first round, or after a plain-decode confirmation invalidated them).
            std::vector<int32_t> delta(impl_->committed_ids.begin() + impl_->kv_filled,
                                       impl_->committed_ids.begin() + ctx_end);
            {
                PERF_SCOPE("dspark/append_kv");
                impl_->engine->append_context_kv(delta, impl_->kv_filled);
            }
            if (!impl_->saw_handoff_ && !impl_->warned_replay_) {
                impl_->warned_replay_ = true;
                LOG_WARN("DFlashDraftProvider: target hidden handoff unavailable -- falling back "
                         "to re-forwarding the delta through the target (much slower).");
            }
            impl_->clear_verify_hiddens();
        }
        impl_->kv_filled = ctx_end;
    }

    // 2. query block: [anchor=last_token, MASK*(query_len-1)] (non-causal forward),
    //    anchor 置于它的真实位置 committed_len-1 (上游 dp.pos0). For sample_from_anchor
    //    (DSpark) query_len == n_spec and row 0 IS the first draft prediction; for
    //    plain DFlash (1+N) query_len == n_block+1 and row 0 is a bonus.
    std::vector<int32_t> query(static_cast<size_t>(query_len));
    query[0] = last_token;
    for (int i = 1; i < query_len; ++i)
        query[static_cast<size_t>(i)] = impl_->mask_token_id_;

    TensorPtr logits;
    {
        PERF_SCOPE("dspark/decode");
        logits = impl_->engine->decode(query, impl_->committed_len - 1);
    }
    if (!logits || logits->ndim() < 2) {
        LOG_WARN("DFlashDraftProvider: decode failed; skipping draft");
        return result;
    }
    if (logits->device() != DeviceType::CPU) {
        PERF_SCOPE("dspark/d2h");
        logits->to_device(DeviceType::CPU);
    }

    // 3. sample n_block tokens. DFlash takes greedy argmax of each row (parallel,
    //    position-independent drafts); DSPark walks the rows left-to-right and
    //    injects the previous sampled token through the markov head, so the block
    //    acquires an intra-dependence (vLLM dspark/_sample_sequential). Both feed
    //    the same [anchor, MASK*N] decode output; resample-consistency in the
    //    verify step keeps generation lossless regardless of the sampling policy.
    //    DFlash2 skips the logits entirely: the decode() call packs a top-k
    //    transition lattice (backend sampling disabled upstream for DFlash2) and
    //    the CPU greedy walk reads it.
    const int vocab = static_cast<int>(logits->shape()[1]);
    const float* data = static_cast<const float*>(logits->data());

    if (impl_->engine->is_dflash2()) {
        TensorPtr lattice = impl_->engine->dflash2_lattice();
        if (!lattice || lattice->ndim() < 2)
            return result;  // block too short / lattice unavailable → no draft
        result = dflash2_lattice_walk(static_cast<const float*>(lattice->data()),
                                      lattice->shape()[1],
                                      impl_->model.config().dflash_selector_top_k,
                                      static_cast<int>(lattice->shape()[0]), impl_->p_min);
        if (static_cast<int>(result.size()) > n)
            result.resize(n);
        return result;
    }

    if (impl_->is_dspark && impl_->engine->has_markov_head()) {
        std::vector<const float*> rows;
        rows.reserve(n_rows);
        if (sfa) {
            // sample_from_anchor (vLLM DSpark): row 0 is the anchor position and IS
            // the first draft prediction; sample every row 0..n_rows-1.
            for (int j = 0; j < n_rows; ++j)
                rows.push_back(data + static_cast<int64_t>(j) * vocab);
        } else {
            // Plain DFlash (1+N): row 0 is a bonus prediction of last_token itself
            // and is skipped; only the MASK rows (1..n_block) feed the draft.
            for (int j = 0; j < n_block; ++j)
                rows.push_back(data + static_cast<int64_t>(j + 1) * vocab);
        }
        TensorPtr me = impl_->engine->markov_embed();
        TensorPtr mb = impl_->engine->markov_bias();
        TensorPtr idmap = impl_->engine->draft_id_map();
        const float* mep = me ? static_cast<const float*>(me->data()) : nullptr;
        const float* mbp = mb ? static_cast<const float*>(mb->data()) : nullptr;
        const int32_t* idp = idmap ? static_cast<const int32_t*>(idmap->data()) : nullptr;
        const int md = me ? static_cast<int>(me->shape()[1]) : 0;

        // DSpark confidence head: build per-position hidden rows aligned 1:1 with the
        // logits rows, and pull the confidence projection weights.
        const float* hidden_data = nullptr;
        int hidden_dim = 0;
        TensorPtr hidden = impl_->engine->last_hidden_states();
        if (hidden && hidden->ndim() >= 2) {
            hidden_data = static_cast<const float*>(hidden->data());
            hidden_dim = static_cast<int>(hidden->shape()[1]);
        }
        std::vector<const float*> hidden_rows;
        hidden_rows.reserve(rows.size());
        if (hidden_data) {
            if (sfa) {
                for (int j = 0; j < n_rows; ++j)
                    hidden_rows.push_back(hidden_data + static_cast<int64_t>(j) * hidden_dim);
            } else {
                for (int j = 0; j < n_block; ++j)
                    hidden_rows.push_back(hidden_data + static_cast<int64_t>(j + 1) * hidden_dim);
            }
        }
        TensorPtr cw = impl_->engine->has_confidence_head() ? impl_->engine->conf_proj_weight() : nullptr;
        TensorPtr cb = impl_->engine->has_confidence_head() ? impl_->engine->conf_proj_bias() : nullptr;
        const float* cwp = cw ? static_cast<const float*>(cw->data()) : nullptr;
        const float* cbp = cb ? static_cast<const float*>(cb->data()) : nullptr;
        {
            PERF_SCOPE("dspark/markov_sample");
            result = dspark_sequential_sample(
                rows, vocab, hidden_rows, hidden_dim, mep, md, mbp, idp, last_token,
                static_cast<int>(rows.size()), impl_->dspark_top_k_, cwp, cbp,
                impl_->dspark_conf_threshold_);
        }
        // p_min confidence gate still applies: drop the tail if the first markov
        // position is below threshold (rare for DSPark, kept for parity).
        if (impl_->p_min > 0.0f && !result.empty()) {
            // For sample_from_anchor the first draft prediction IS row 0 (anchor);
            // otherwise it is the first MASK row (row 1).
            const float* row0 = sfa ? data : (data + vocab);
            if (top_probability(row0, vocab) < impl_->p_min)
                result.clear();
        }
        if (static_cast<int>(result.size()) > n)
            result.resize(n);
        return result;
    }

    for (int j = 0; j < n_block; ++j) {
        // Skip the anchor/bonus row (index 0): only the MASK rows (1..n_block) are
        // sampled, matching vLLM's _prepare_dflash_inputs_kernel.
        const float* row = data + static_cast<int64_t>(j + 1) * vocab;
        if (impl_->p_min > 0.0f && top_probability(row, vocab) < impl_->p_min)
            break;
        int best = 0;
        for (int t = 1; t < vocab; ++t)
            if (row[t] > row[best])
                best = t;
        result.push_back(best);
    }
    if (static_cast<int>(result.size()) > n)
        result.resize(n);
    return result;
}

void DFlashDraftProvider::accept(const std::vector<int32_t>& tokens) {
    if (!valid() || tokens.empty())
        return;
    // drafter KV 现在持久化并在 draft() 里增量追加, 这里只推进已确认前缀的
    // 记账。SpeculativeExecutor 传入完整确认序列 (accepted + resampled/bonus),
    // 与 target 的确认边界一致; 部分接受时 executor 的 KV rollback 只作用于
    // target 侧缓存, drafter 缓存中的旧 query 残行会在下轮被覆盖。
    impl_->committed_ids.insert(impl_->committed_ids.end(), tokens.begin(), tokens.end());
    impl_->committed_len += static_cast<int64_t>(tokens.size());
}

void DFlashDraftProvider::reset() {
    if (impl_)
        impl_->reset();
}

}  // namespace forge
