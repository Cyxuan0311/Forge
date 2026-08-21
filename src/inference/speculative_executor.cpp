#include "forge/speculative_executor.h"

#include <algorithm>

#include "forge/context.h"
#include "forge/engine.h"
#include "forge/inference_batch.h"
#include "forge/kv_cache.h"
#include "forge/logger.h"
#include "forge/perf_profiler.h"

namespace forge {

SpeculativeExecutor::SpeculativeExecutor(InferenceContext& ctx, Sampler& sampler,
                                         const SpeculativeConfig& cfg)
    : ctx_(ctx), sampler_(sampler), cfg_(cfg) {
    if (!cfg_.enabled) return;

    // Phase 3: when cfg_.draft_model_path is set, create a ModelDraftProvider here
    // (takes priority over n-gram).
    if (cfg_.use_ngram) {
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

    // Each round produces at least one token (resampled or bonus), so a round
    // consumes up to n_draft + 1 from the budget.
    const int n_draft = std::min(cfg_.n_draft, in.max_tokens - 1);
    if (n_draft <= 0) {
        ++stats_.n_fallback_steps;
        return out;
    }

    auto drafts = provider_->draft(n_draft);
    if (static_cast<int>(drafts.size()) < std::max(cfg_.n_min, 1)) {
        ++stats_.n_fallback_steps;
        return out;
    }

    // ---- Verification batch: [last_token @ pos, d0 @ pos+1, ..., dN-1 @ pos+N] ----
    // Row i of the returned logits verifies drafts[i-1].
    InferenceBatch batch;
    batch.all_logits = true;
    InferenceBatchItem item;
    item.seq_id = 0;
    item.logits = true;
    item.tokens.reserve(drafts.size() + 1);
    item.tokens.push_back(in.last_token);
    item.tokens.insert(item.tokens.end(), drafts.begin(), drafts.end());
    item.start_pos = in.pos;
    const int n_rows = static_cast<int>(item.tokens.size());
    item.positions.resize(n_rows);
    for (int j = 0; j < n_rows; ++j) item.positions[j] = in.pos + j;
    batch.items.push_back(std::move(item));

    TensorPtr logits_all;
    {
        PERF_SCOPE("speculative/forward_batch");
        logits_all = engine->forward_batch(batch);
    }

    if (!logits_all || logits_all->ndim() < 2) {
        LOG_WARN("SpeculativeExecutor: forward_batch failed, falling back to plain decode");
        ++stats_.n_fallback_steps;
        return out;
    }
    if (logits_all->device() != DeviceType::CPU) {
        logits_all->to_device(DeviceType::CPU);
    }

    const int vocab_size = static_cast<int>(logits_all->shape()[1]);
    const float* logits_data = static_cast<const float*>(logits_all->data());

    // ---- Verify ----
    auto vr = verify_draft_tokens(logits_data, vocab_size, drafts, sampler_, cfg_);

    // ---- Assemble produced tokens: accepted... then resampled|bonus ----
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

    // ---- KV rollback on partial rejection ----
    // Batch rows wrote KV rows [pos, pos+n_draft]; accepted rows cover
    // [pos+1, pos+n_accepted]. On rejection, drop the rejected tail so the
    // valid region becomes [0, pos + tokens.size()).
    const bool full_accept = (vr.n_accepted == static_cast<int>(drafts.size()));
    if (!full_accept) {
        KVCache* kv = engine->kv_cache();
        if (kv) kv->rollback(in.pos + vr.n_accepted + 1);
    }

    out.speculated = true;

    // ---- Stats & provider bookkeeping ----
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

}  // namespace forge
