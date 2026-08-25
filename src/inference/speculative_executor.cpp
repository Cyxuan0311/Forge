#include "forge/speculative_executor.h"

#include <algorithm>
#include <cstring>

#include "forge/context.h"
#include "forge/engine.h"
#include "forge/inference/forward_request.h"
#include "forge/kv_cache.h"
#include "forge/logger.h"
#include "forge/perf_profiler.h"
#include "forge/tensor.h"

namespace forge {

SpeculativeExecutor::SpeculativeExecutor(InferenceContext& ctx, Sampler& sampler,
                                         const SpeculativeConfig& cfg)
    : ctx_(ctx), sampler_(sampler), cfg_(cfg) {
    if (!cfg_.enabled) return;

    if (!cfg_.draft_model_path.empty()) {
        auto draft = std::make_unique<ModelDraftProvider>(cfg_, ctx_.params(),
                                                           ctx_.model().config().vocab_size);
        if (draft->valid()) {
            provider_ = std::move(draft);
        } else {
            LOG_WARN("SpeculativeExecutor: draft model unavailable, falling back to n-gram");
        }
    }
    if (!provider_ && cfg_.use_ngram) {
        provider_ = std::make_unique<NgramDraftProvider>(cfg_.ngram_n, cfg_.ngram_min);
        LOG_INFO(std::string("SpeculativeExecutor: draft provider '") + provider_->name() +
                 "' enabled, n_draft=" + std::to_string(cfg_.n_draft));
    }
}

SpeculativeExecutor::~SpeculativeExecutor() = default;

void SpeculativeExecutor::begin_generation(const std::vector<int32_t>& prompt) {
    stats_.reset();
    if (provider_) {
        provider_->reset();
        provider_->begin(prompt);
    }
}

SpeculativeExecutor::StepOutput SpeculativeExecutor::step(const StepInput& in) {
    StepOutput out;

    auto* engine = ctx_.engine();
    if (!provider_ || !engine || in.max_tokens <= 1) {
        ++stats_.n_fallback_steps;
        return out;
    }

    const int n_draft = std::min(cfg_.n_draft, in.max_tokens - 1);
    if (n_draft <= 0) {
        ++stats_.n_fallback_steps;
        return out;
    }

    auto drafts = provider_->draft(in.last_token, n_draft);
    if (static_cast<int>(drafts.size()) < std::max(cfg_.n_min, 1)) {
        ++stats_.n_fallback_steps;
        return out;
    }

    const int n_rows = static_cast<int>(drafts.size()) + 1;
    auto input_ids = std::make_shared<Tensor>(DataType::INT32,
                                              std::vector<int64_t>{n_rows}, DeviceType::CPU);
    int32_t* ids_ptr = static_cast<int32_t*>(input_ids->data());
    ids_ptr[0] = in.last_token;
    std::memcpy(ids_ptr + 1, drafts.data(), drafts.size() * sizeof(int32_t));
    if (ctx_.device() == DeviceType::CUDA) {
        input_ids->to_device(DeviceType::CUDA);
    }

    TensorPtr logits_all;
    {
        PERF_SCOPE("speculative/verify_forward");
        logits_all = engine->forward_request(
            ForwardRequest::from_ids(input_ids, in.pos));
    }

    if (!logits_all || logits_all->ndim() < 2 ||
        logits_all->shape()[0] < n_rows) {
        LOG_WARN("SpeculativeExecutor: verify forward failed, falling back to plain decode");
        ++stats_.n_fallback_steps;
        return out;
    }
    if (logits_all->device() != DeviceType::CPU) {
        logits_all->to_device(DeviceType::CPU);
    }

    const int vocab_size = static_cast<int>(logits_all->shape()[1]);
    const float* logits_data = static_cast<const float*>(logits_all->data());

    auto vr = verify_draft_tokens(logits_data, vocab_size, drafts, sampler_, cfg_);

    out.tokens.insert(out.tokens.end(), vr.accepted_tokens.begin(), vr.accepted_tokens.end());
    if (vr.resampled >= 0) {
        out.tokens.push_back(vr.resampled);
    } else if (vr.bonus >= 0) {
        out.tokens.push_back(vr.bonus);
    }
    if (out.tokens.empty()) {
        ++stats_.n_fallback_steps;
        return out;
    }

    const bool full_accept = (vr.n_accepted == static_cast<int>(drafts.size()));
    if (!full_accept) {
        KVCache* kv = engine->kv_cache();
        if (kv) kv->rollback(in.pos + vr.n_accepted + 1);
    }

    out.speculated = true;

    ++stats_.n_spec_steps;
    stats_.n_draft_tokens += static_cast<int64_t>(drafts.size());
    stats_.n_accepted_tokens += vr.n_accepted;
    stats_.n_output_tokens += static_cast<int64_t>(out.tokens.size());
    provider_->accept(out.tokens);

    return out;
}

void SpeculativeExecutor::rollback_kv(int64_t valid_end_pos) {
    auto* engine = ctx_.engine();
    KVCache* kv = engine ? engine->kv_cache() : nullptr;
    if (kv) kv->rollback(valid_end_pos);
}

void SpeculativeExecutor::notify_confirmed(int32_t token) {
    if (provider_) provider_->accept({token});
}

}  // namespace forge
