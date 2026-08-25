#include "forge/speculative.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>

#include "forge/arch_registry.h"
#include "forge/context.h"
#include "forge/engine.h"
#include "forge/gguf_model.h"
#include "forge/inference/forward_request.h"
#include "forge/kv_cache.h"
#include "forge/logger.h"
#include "forge/model.h"
#include "forge/model_loader.h"
#include "forge/ninf_model.h"
#include "forge/engines/transformer_engine.h"
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
    params.speculative_config = SpeculativeConfig{};

    auto ctx = std::make_unique<InferenceContext>(model, params);
    auto engine = EngineRegistry::instance().create(model.config().arch_type, model, *ctx);
    if (!engine) return nullptr;

    if (auto* transformer = dynamic_cast<TransformerEngine*>(engine.get())) {
        transformer->set_gpu_layers(gpu_layers);
        transformer->set_kv_cache_dtype(params.kv_cache_dtype);
    }
    ctx->set_engine(std::move(engine));
    return ctx;
}

float top_probability(const float* logits, int vocab_size) {
    float max_logit = logits[0];
    for (int i = 1; i < vocab_size; ++i) max_logit = std::max(max_logit, logits[i]);

    double sum = 0.0;
    for (int i = 0; i < vocab_size; ++i) {
        sum += std::exp(static_cast<double>(logits[i] - max_logit));
    }
    return sum > 0.0 ? static_cast<float>(1.0 / sum) : 0.0f;
}

}  // namespace

struct ModelDraftProvider::Impl {
    Model model;
    std::unique_ptr<InferenceContext> ctx;
    Sampler sampler;
    ContextParams target_params;
    std::string model_path;
    int gpu_layers = 0;
    int vocab_size = 0;
    float p_min = 0.0f;
    int64_t committed_len = 0;
    bool draft_active = false;
    bool valid = false;
    // Logits predicting the token right after the confirmed prefix
    // (last row of the most recent confirmed feed). Lets draft() propose the
    // first candidate without re-feeding the seed token, keeping the draft
    // KV strictly equal to the confirmed prefix.
    std::vector<float> pending_logits;

    Impl(const SpeculativeConfig& spec, const ContextParams& target, int target_vocab)
        : sampler([] {
              SamplerConfig config;
              // Greedy proposals: maximise agreement with greedy/low-temperature
              // target verification. Losslessness is guaranteed by the verify
              // step regardless of how candidates were proposed.
              config.do_sample = false;
              config.temperature = 0.0f;
              config.top_k = 0;
              config.top_p = 1.0f;
              config.repeat_penalty = 1.0f;
              return config;
          }()),
          target_params(target),
          model_path(spec.draft_model_path),
          gpu_layers(spec.draft_gpu_layers >= 0 ? spec.draft_gpu_layers : target.gpu_layers),
          vocab_size(target_vocab),
          p_min(spec.p_min) {
        register_runtime();

        if (model_path.empty() || !model.load(model_path, target.device)) {
            LOG_WARN("ModelDraftProvider: failed to load draft model");
            return;
        }
        if (model.config().vocab_size != target_vocab) {
            LOG_WARN("ModelDraftProvider: draft/target vocabulary sizes differ");
            return;
        }

        ctx = make_context(model, target_params, gpu_layers);
        if (!ctx || !ctx->engine()) {
            LOG_WARN("ModelDraftProvider: failed to create draft inference context");
            return;
        }
        valid = true;
    }

    void reset() {
        if (ctx) ctx->reset_kv_cache();
        sampler.clear_history();
        committed_len = 0;
        draft_active = false;
        pending_logits.clear();
    }

    // Feed tokens and cache the last row's logits (prediction after the
    // final token of the confirmed prefix).
    bool feed(const std::vector<int32_t>& tokens, int64_t start_pos) {
        if (tokens.empty()) return true;
        auto ids = std::make_shared<Tensor>(DataType::INT32,
                                            std::vector<int64_t>{static_cast<int64_t>(tokens.size())},
                                            DeviceType::CPU);
        std::memcpy(ids->data(), tokens.data(), tokens.size() * sizeof(int32_t));
        auto logits = ctx->engine()->forward_request(
            ForwardRequest::from_ids(ids, start_pos));
        if (!logits || logits->ndim() < 2) return false;
        if (logits->device() != DeviceType::CPU) logits->to_device(DeviceType::CPU);
        const int64_t vocab = logits->shape()[1];
        const float* last_row =
            static_cast<const float*>(logits->data()) + (logits->shape()[0] - 1) * vocab;
        pending_logits.assign(last_row, last_row + vocab);
        return true;
    }

    TensorPtr feed_one(int32_t token, int64_t pos) {
        auto ids = std::make_shared<Tensor>(DataType::INT32,
                                            std::vector<int64_t>{1}, DeviceType::CPU);
        *static_cast<int32_t*>(ids->data()) = token;
        auto logits = ctx->engine()->forward_request(ForwardRequest::from_ids(ids, pos));
        if (!logits || logits->ndim() < 2) return nullptr;
        if (logits->device() != DeviceType::CPU) logits->to_device(DeviceType::CPU);
        return logits;
    }

    void abort_draft() {
        if (draft_active && ctx->engine()->kv_cache()) {
            ctx->engine()->kv_cache()->rollback(committed_len);
        }
        draft_active = false;
    }
};

ModelDraftProvider::ModelDraftProvider(const SpeculativeConfig& spec,
                                       const ContextParams& target, int target_vocab)
    : impl_(std::make_unique<Impl>(spec, target, target_vocab)) {}

ModelDraftProvider::~ModelDraftProvider() = default;

bool ModelDraftProvider::valid() const { return impl_ && impl_->valid; }

void ModelDraftProvider::begin(const std::vector<int32_t>& prompt) {
    if (!valid()) return;
    impl_->reset();
    if (!impl_->feed(prompt, 0)) {
        impl_->valid = false;
        return;
    }
    impl_->committed_len = static_cast<int64_t>(prompt.size());
}

std::vector<int32_t> ModelDraftProvider::draft(int32_t last_token, int n_draft) {
    (void)last_token;  // seed logits are cached from the previous confirmed feed
    std::vector<int32_t> result;
    if (!valid() || n_draft <= 0) return result;

    impl_->abort_draft();
    impl_->draft_active = true;

    // Invariant: draft KV == confirmed prefix [0, committed_len), and
    // pending_logits predicts the token at committed_len. The first candidate
    // comes straight from those logits — the seed token is NOT re-fed (it is
    // already the last row of the confirmed prefix).
    constexpr int kVocabRow = 1;
    const float* row = impl_->pending_logits.data();
    for (int i = 0; i < n_draft; ++i) {
        if (!row) {
            impl_->abort_draft();
            return {};
        }
        if (impl_->p_min > 0.0f &&
            top_probability(row, impl_->vocab_size) < impl_->p_min) {
            break;
        }

        int32_t next =
            static_cast<int32_t>(impl_->sampler.sample_ptr(row, impl_->vocab_size));
        result.push_back(next);

        // Feed the candidate to get logits for the following position.
        auto logits = impl_->feed_one(next, impl_->committed_len + i);
        if (!logits || logits->shape()[0] < kVocabRow) {
            impl_->abort_draft();
            return {};
        }
        row = static_cast<const float*>(logits->data());
    }

    if (result.empty()) impl_->abort_draft();
    return result;
}

void ModelDraftProvider::accept(const std::vector<int32_t>& tokens) {
    if (!valid() || tokens.empty()) return;

    // Discard every speculative row so the draft KV returns to the confirmed
    // prefix before replaying target-confirmed tokens.
    impl_->abort_draft();

    if (!impl_->feed(tokens, impl_->committed_len)) {
        impl_->valid = false;
        return;
    }
    impl_->committed_len += static_cast<int64_t>(tokens.size());
}

void ModelDraftProvider::reset() {
    if (impl_) impl_->reset();
}

}  // namespace forge
