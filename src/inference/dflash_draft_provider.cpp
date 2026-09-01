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
#include "forge/sampler.h"

namespace forge {

namespace {

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
    std::vector<int> target_layers_;
    int mask_token_id_ = 0;
    bool is_dspark = false;
    bool valid = false;

    int64_t committed_len = 0;
    std::vector<int32_t> committed_ids;

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
          gpu_layers(spec.draft_gpu_layers >= 0 ? spec.draft_gpu_layers : tctx.params().gpu_layers),
          vocab_size(target_vocab),
          p_min(spec.p_min),
          n_spec(std::max(1, spec.draft_n_spec)),
          mask_token_id_(spec.draft_mask_token_id >= 0 ? spec.draft_mask_token_id : 0),
          is_dspark(spec.draft_arch == "dspark"),
          target(&t) {
        register_runtime();

        if (model_path.empty() || !model.load(model_path, target_params.device)) {
            LOG_WARN("DFlashDraftProvider: failed to load drafter model '" + model_path + "'");
            return;
        }
        if (model.config().vocab_size != target_vocab) {
            LOG_WARN("DFlashDraftProvider: drafter/target vocabulary mismatch (" +
                     std::to_string(model.config().vocab_size) + " vs " +
                     std::to_string(target_vocab) + ")");
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

        // Target must capture multi-layer hiddens so we can fuse the encoder.
        if (auto* tt = dynamic_cast<TransformerEngine*>(target)) {
            tt->set_capture_layer_hiddens(true);
        } else {
            LOG_WARN("DFlashDraftProvider: target engine cannot capture layer hiddens");
            return;
        }

        // Encoder input layers: CLI override or GGUF metadata.
        target_layers_ = spec.draft_target_layers;
        if (target_layers_.empty())
            target_layers_ = model.config().target_layers;
        if (target_layers_.empty()) {
            LOG_WARN("DFlashDraftProvider: no target_layers (CLI or GGUF), cannot encode");
            return;
        }

        valid = true;
        LOG_INFO(std::string("DFlashDraftProvider: ready (") + (is_dspark ? "dspark" : "dflash") +
                 ") n_spec=" + std::to_string(n_spec) +
                 " target_layers=" + std::to_string(target_layers_.size()));
    }

    void reset() {
        if (ctx)
            ctx->reset_kv_cache();
        committed_len = 0;
        committed_ids.clear();
    }

    // Re-run the committed prefix through the target so its layer_hiddens_
    // cache holds exactly the prefix features the encoder needs. This is
    // idempotent on the target KV (writes the same positions) and avoids
    // depending on the SpeculativeExecutor's verify forward, which only covers
    // the draft window. (A future optimization could stitch hiddens from the
    // verify forward instead of re-forwarding the prefix.)
    TensorPtr refresh_prefix_hiddens() {
        if (!target || committed_len <= 0)
            return nullptr;
        auto ids = std::make_shared<Tensor>(DataType::INT32, std::vector<int64_t>{1, committed_len},
                                            DeviceType::CPU);
        std::memcpy(ids->data(), committed_ids.data(),
                    static_cast<size_t>(committed_len) * sizeof(int32_t));
        target->forward_request(ForwardRequest::from_ids(ids, 0));
        return target->take_layer_hiddens(target_layers_);
    }
};

DFlashDraftProvider::DFlashDraftProvider(const SpeculativeConfig& spec, InferenceContext& ctx,
                                         InferenceEngine& target, int target_vocab)
    : impl_(std::make_unique<Impl>(spec, ctx, target, target_vocab)),
      is_dspark_(spec.draft_arch == "dspark") {}

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

std::vector<int32_t> DFlashDraftProvider::draft(int32_t last_token, int n_draft) {
    std::vector<int32_t> result;
    if (!valid() || !impl_->engine || n_draft <= 0)
        return result;

    const int n = std::min(n_draft, impl_->n_spec);

    // 1. target multi-layer hiddens for the committed prefix.
    TensorPtr h = impl_->refresh_prefix_hiddens();
    if (!h || h->ndim() < 2) {
        LOG_WARN("DFlashDraftProvider: target layer hiddens unavailable; skipping draft");
        return result;
    }

    // 2. encoder -> context feature (cached in the engine).
    impl_->engine->encode(h);

    // 3. inject target prefix K/V into the drafter KV cache.
    impl_->engine->precompute_context_kv(impl_->committed_ids, impl_->committed_len);

    // 4. query block: [anchor=last_token, MASK*n] (non-causal forward).
    std::vector<int32_t> query(static_cast<size_t>(n) + 1);
    query[0] = last_token;
    for (int i = 1; i <= n; ++i)
        query[static_cast<size_t>(i)] = impl_->mask_token_id_;

    TensorPtr logits = impl_->engine->decode(query, impl_->committed_len);
    if (!logits || logits->ndim() < 2) {
        LOG_WARN("DFlashDraftProvider: decode failed; skipping draft");
        return result;
    }
    if (logits->device() != DeviceType::CPU)
        logits->to_device(DeviceType::CPU);

    // 5. sample n tokens. DFlash takes greedy argmax of each row (parallel,
    //    position-independent drafts); DSPark walks the rows left-to-right and
    //    injects the previous sampled token through the markov head, so the block
    //    acquires an intra-dependence (vLLM dspark/_sample_sequential). Both feed
    //    the same [anchor, MASK*N] decode output; resample-consistency in the
    //    verify step keeps generation lossless regardless of the sampling policy.
    const int vocab = static_cast<int>(logits->shape()[1]);
    const float* data = static_cast<const float*>(logits->data());

    if (impl_->is_dspark && impl_->engine->has_markov_head()) {
        std::vector<const float*> rows;
        rows.reserve(n);
        for (int j = 0; j < n; ++j)
            rows.push_back(data + static_cast<int64_t>(j) * vocab);
        TensorPtr me = impl_->engine->markov_embed();
        TensorPtr mb = impl_->engine->markov_bias();
        TensorPtr idmap = impl_->engine->draft_id_map();
        const float* mep = me ? static_cast<const float*>(me->data()) : nullptr;
        const float* mbp = mb ? static_cast<const float*>(mb->data()) : nullptr;
        const int32_t* idp = idmap ? static_cast<const int32_t*>(idmap->data()) : nullptr;
        const int md = me ? static_cast<int>(me->shape()[1]) : 0;
        result = dspark_sequential_sample(rows, vocab, mep, md, mbp, idp, last_token, n);
        // p_min confidence gate still applies: drop the tail if the first markov
        // position is below threshold (rare for DSPark, kept for parity).
        if (impl_->p_min > 0.0f && !result.empty()) {
            const float* row0 = data;
            if (top_probability(row0, vocab) < impl_->p_min)
                result.clear();
        }
        return result;
    }

    for (int j = 0; j < n; ++j) {
        const float* row = data + static_cast<int64_t>(j) * vocab;
        if (impl_->p_min > 0.0f && top_probability(row, vocab) < impl_->p_min)
            break;
        int best = 0;
        for (int t = 1; t < vocab; ++t)
            if (row[t] > row[best])
                best = t;
        result.push_back(best);
    }
    return result;
}

void DFlashDraftProvider::accept(const std::vector<int32_t>& tokens) {
    if (!valid() || tokens.empty())
        return;
    // The drafter KV is fully rebuilt each draft() (precompute_context_kv resets
    // it), so we only need to advance the committed-prefix bookkeeping. The
    // SpeculativeExecutor passes the full confirmed run (accepted + resampled/
    // bonus), which extends the prefix exactly as the target does.
    impl_->committed_ids.insert(impl_->committed_ids.end(), tokens.begin(), tokens.end());
    impl_->committed_len += static_cast<int64_t>(tokens.size());
}

void DFlashDraftProvider::reset() {
    if (impl_)
        impl_->reset();
}

}  // namespace forge
