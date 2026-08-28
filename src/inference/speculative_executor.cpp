#include "forge/speculative_executor.h"

#include <algorithm>
#include <cstring>

#include "forge/context.h"
#include "forge/engine.h"
#include "forge/inference/forward_request.h"
#include "forge/mtp_draft_provider.h"
#include "forge/kv_cache.h"
#include "forge/logger.h"
#include "forge/perf_profiler.h"
#include "forge/tensor.h"

namespace forge {

SpeculativeExecutor::SpeculativeExecutor(InferenceContext& ctx, Sampler& sampler,
                                         const SpeculativeConfig& cfg)
    : ctx_(ctx), sampler_(sampler), cfg_(cfg) {
    if (!cfg_.enabled) return;

    // Build chain in priority order: MTP > Model > NgramMod > Ngram
    // Each provider is tried in order at draft time; first that yields
    // >= n_min candidates wins. Global providers (ngram) still receive
    // accept() for history sync even when they were not the drafter.
    if (cfg_.use_mtp) {
        auto mtp = std::make_unique<MtpDraftProvider>(ctx_, cfg_.p_min);
        if (mtp->valid()) {
            providers_.push_back(std::move(mtp));
        } else {
            LOG_WARN("SpeculativeExecutor: MTP provider unavailable, skipping");
        }
    }
    if (!cfg_.draft_model_path.empty()) {
        auto draft = std::make_unique<ModelDraftProvider>(cfg_, ctx_.params(),
                                                           ctx_.model().config().vocab_size);
        if (draft->valid()) {
            providers_.push_back(std::move(draft));
        } else {
            LOG_WARN("SpeculativeExecutor: draft model unavailable, skipping");
        }
    }
    if (cfg_.use_ngram_mod) {
        auto mod = std::make_unique<NgramModProvider>(
            cfg_.ngram_mod_n, cfg_.ngram_mod_n_min, cfg_.ngram_mod_n_max,
            cfg_.ngram_mod_pool_size);
        providers_.push_back(std::move(mod));
        LOG_INFO(std::string("SpeculativeExecutor: draft provider 'ngram-mod' enabled, n_match=") +
                 std::to_string(cfg_.ngram_mod_n) + " pool=" + std::to_string(cfg_.ngram_mod_pool_size));
    }
    if (cfg_.use_ngram && !cfg_.no_ngram) {
        auto ngram = std::make_unique<NgramDraftProvider>(cfg_.ngram_n, cfg_.ngram_min);
        providers_.push_back(std::move(ngram));
    } else if (cfg_.use_ngram_mod && cfg_.no_ngram) {
        LOG_INFO("SpeculativeExecutor: plain n-gram fallback disabled (--no-ngram); "
                 "isolating ngram-mod provider only");
    }
    if (!providers_.empty()) {
        provider_ = nullptr; // legacy single pointer not used when chain present
        // For backwards compat keep provider_ pointing to primary
        // but valid() now checks providers_.empty()
        std::string names;
        for (auto& p : providers_) {
            if (!names.empty()) names += ",";
            names += p->name();
        }
        LOG_INFO(std::string("SpeculativeExecutor: chain [") + names + "] n_draft=" + std::to_string(cfg_.n_draft));
    }
}

SpeculativeExecutor::~SpeculativeExecutor() = default;

void SpeculativeExecutor::begin_generation(const std::vector<int32_t>& prompt) {
    stats_.reset();
    last_drafter_idx_ = -1;
    adaptive_draft_ = 0;
    accept_ewma_ = -1.0;
    for (auto& p : providers_) {
        p->reset();
        p->begin(prompt);
    }
    if (provider_) {
        provider_->reset();
        provider_->begin(prompt);
    }
}

SpeculativeExecutor::StepOutput SpeculativeExecutor::step(const StepInput& in) {
    StepOutput out;

    auto* engine = ctx_.engine();
    bool has_chain = !providers_.empty();
    bool has_legacy = provider_ != nullptr;
    if ((!has_chain && !has_legacy) || !engine || in.max_tokens <= 1) {
        ++stats_.n_fallback_steps;
        return out;
    }

    // Adaptive draft: scale effective n_draft by recent acceptance rate.
    // When acceptance is low, a large draft wastes a full multi-row forward
    // for little gain (net negative). Shrink toward 1; grow back when the
    // provider is reliably accepted. Starts at the user-requested ceiling.
    if (adaptive_draft_ <= 0) {
        adaptive_draft_ = cfg_.n_draft;  // initialize to configured ceiling
    }
    if (accept_ewma_ >= 0.0) {
        // target draft proportional to acceptance, clamped to [1, cfg_.n_draft]
        int target = static_cast<int>(std::max(1.0, std::min<double>(cfg_.n_draft,
                                                                      adaptive_draft_ * (0.5 + accept_ewma_))));
        adaptive_draft_ = target;
    }
    const int n_draft = std::min({adaptive_draft_, cfg_.n_draft, in.max_tokens - 1});
    if (n_draft <= 0) {
        ++stats_.n_fallback_steps;
        return out;
    }

    std::vector<int32_t> drafts;
    int drafter_idx = -1;
    if (has_chain) {
        for (size_t i = 0; i < providers_.size(); ++i) {
            auto cand = providers_[i]->draft(in.last_token, n_draft);
            if (static_cast<int>(cand.size()) >= std::max(cfg_.n_min, 1)) {
                drafts = std::move(cand);
                drafter_idx = static_cast<int>(i);
                break;
            }
        }
        if (drafts.empty()) {
            ++stats_.n_fallback_steps;
            return out;
        }
    } else {
        drafts = provider_->draft(in.last_token, n_draft);
        if (static_cast<int>(drafts.size()) < std::max(cfg_.n_min, 1)) {
            ++stats_.n_fallback_steps;
            return out;
        }
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
    last_drafter_idx_ = drafter_idx;

    // Update acceptance EWMA (used by adaptive draft sizing next round).
    {
        const double round_acc = drafts.empty() ? 0.0
                                                 : static_cast<double>(vr.n_accepted) / drafts.size();
        if (accept_ewma_ < 0.0) {
            accept_ewma_ = round_acc;
        } else {
            accept_ewma_ = 0.7 * accept_ewma_ + 0.3 * round_acc;
        }
    }

    ++stats_.n_spec_steps;
    stats_.n_draft_tokens += static_cast<int64_t>(drafts.size());
    stats_.n_accepted_tokens += vr.n_accepted;
    stats_.n_output_tokens += static_cast<int64_t>(out.tokens.size());
    if (has_chain) {
        for (size_t i = 0; i < providers_.size(); ++i) {
            if (static_cast<int>(i) == drafter_idx || providers_[i]->is_global()) {
                providers_[i]->accept(out.tokens);
            }
        }
    } else {
        provider_->accept(out.tokens);
    }

    return out;
}

void SpeculativeExecutor::rollback_kv(int64_t valid_end_pos) {
    auto* engine = ctx_.engine();
    KVCache* kv = engine ? engine->kv_cache() : nullptr;
    if (kv) kv->rollback(valid_end_pos);
}

void SpeculativeExecutor::notify_confirmed(int32_t token) {
    for (auto& p : providers_) {
        p->accept({token});
    }
    if (provider_) provider_->accept({token});
}

}  // namespace forge
