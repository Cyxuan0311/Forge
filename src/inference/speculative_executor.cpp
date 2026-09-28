#include "forge/speculative_executor.h"

#include <algorithm>
#include <cstring>

#include "forge/context.h"
#include "forge/dflash_draft_provider.h"
#include "forge/engine.h"
#include "forge/engines/transformer_engine.h"
#include "forge/inference/forward_request.h"
#include "forge/inference/layers/qwen35_recurrent_memory.h"
#include "forge/kv_cache.h"
#include "forge/logger.h"
#include "forge/mtp_draft_provider.h"
#include "forge/perf_profiler.h"
#include "forge/tensor.h"

namespace forge {

namespace {

// 2.3: spec 部分拒绝后, recurrent (SSM) 状态已回滚到 draft 前的快照; 已接受的
// token (last_token + 前 n_accepted 个 draft) 需要重新前向一次, 把 conv/ssm 状态
// 推进到"接受点之后", 与 KVCache::rollback(in.pos + n_accepted + 1) 后的位置对齐。
// 重放对 KV 是幂等的: 覆盖写相同位置, 前向是确定性的。
void replay_recurrent(InferenceEngine* engine, InferenceContext& ctx, int32_t last_token,
                      const std::vector<int32_t>& drafts, int n_accepted, int64_t start_pos) {
    if (!engine)
        return;
    const int n = n_accepted + 1;
    auto ids = std::make_shared<Tensor>(DataType::INT32, std::vector<int64_t>{n}, DeviceType::CPU);
    int32_t* p = static_cast<int32_t*>(ids->data());
    p[0] = last_token;
    if (n_accepted > 0) {
        std::memcpy(p + 1, drafts.data(), static_cast<size_t>(n_accepted) * sizeof(int32_t));
    }
    if (ctx.device() == DeviceType::CUDA) {
        ids->to_device(DeviceType::CUDA);
    }
    engine->forward_request(ForwardRequest::from_ids(ids, start_pos));
}

}  // namespace

SpeculativeExecutor::SpeculativeExecutor(InferenceContext& ctx, Sampler& sampler,
                                         const SpeculativeConfig& cfg)
    : ctx_(ctx), sampler_(sampler), cfg_(cfg), spec_min_acc_(cfg.min_accept_rate) {
    if (!cfg_.enabled)
        return;

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
    // DFlash / DSPark standalone drafter: shares the target's token_embd/lm_head
    // and fuses target multi-layer hiddens as the encoder input. Tried right
    // after MTP so a matched drafter takes priority over n-gram self-draft.
    if (!cfg_.draft_arch.empty() && (cfg_.draft_arch == "dflash" || cfg_.draft_arch == "dspark")) {
        if (cfg_.draft_model_path.empty()) {
            LOG_WARN(
                "SpeculativeExecutor: --spec-draft-arch set but no drafter GGUF "
                "(--spec-model); skipping DFlash");
        } else {
            auto dflash = std::make_unique<DFlashDraftProvider>(cfg_, ctx_, *ctx_.engine(),
                                                                ctx_.model().config().vocab_size);
            if (dflash->valid()) {
                providers_.push_back(std::move(dflash));
            } else {
                LOG_WARN("SpeculativeExecutor: DFlash provider unavailable, skipping");
            }
        }
    }
    if (!cfg_.draft_model_path.empty() &&
        cfg_.draft_arch != "dflash" && cfg_.draft_arch != "dspark") {
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
            cfg_.ngram_mod_n, cfg_.ngram_mod_n_min, cfg_.ngram_mod_n_max, cfg_.ngram_mod_pool_size);
        providers_.push_back(std::move(mod));
        LOG_INFO(std::string("SpeculativeExecutor: draft provider 'ngram-mod' enabled, n_match=") +
                 std::to_string(cfg_.ngram_mod_n) +
                 " pool=" + std::to_string(cfg_.ngram_mod_pool_size));
    }
    if (cfg_.use_ngram && !cfg_.no_ngram) {
        auto ngram = std::make_unique<NgramDraftProvider>(cfg_.ngram_n, cfg_.ngram_min);
        providers_.push_back(std::move(ngram));
    } else if (cfg_.use_ngram_mod && cfg_.no_ngram) {
        LOG_INFO(
            "SpeculativeExecutor: plain n-gram fallback disabled (--no-ngram); "
            "isolating ngram-mod provider only");
    }
    if (!providers_.empty()) {
        provider_ = nullptr;  // legacy single pointer not used when chain present
        // For backwards compat keep provider_ pointing to primary
        // but valid() now checks providers_.empty()
        std::string names;
        for (auto& p : providers_) {
            if (!names.empty())
                names += ",";
            names += p->name();
        }
        LOG_INFO(std::string("SpeculativeExecutor: chain [") + names +
                 "] n_draft=" + std::to_string(cfg_.n_draft));
        // The verify forward's target hiddens are captured on behalf of every
        // provider that declares an encoder layer list (DFlash/DSPark). N-gram /
        // MTP providers return nullptr here, so the capture stays off and the
        // target keeps its CUDA graph.
        for (auto& p : providers_) {
            if (auto* layers = p->capture_layers()) {
                if (capture_layers_.empty())
                    capture_layers_ = *layers;
            }
        }
    }
}

SpeculativeExecutor::~SpeculativeExecutor() = default;

int adaptive_draft_target(double accept_len_ewma, int cur, int max_draft, double gain,
                          double accepted, double proposed) {
    if (max_draft < 1) return max_draft;
    if (cur < 1) return std::min(max_draft, std::max(1, max_draft));
    // Negative means "no measurement yet" -> hold the current value rather than
    // reacting to an uninitialised signal.
    if (accept_len_ewma < 0.0) return cur;

    int target;
    if (accept_len_ewma < 1.0) {
        // Under 1 token accepted per round, a round cannot break even: the drafter
        // forward and the verify pass both scale with n_draft, so proposing 2 rows
        // to yield 0.2 is strictly worse than proposing 1. --spec-min-acc is the
        // lever for abandoning speculation altogether; this just stops paying for
        // rows that cannot pay for themselves.
        target = 1;
    } else {
        // Aim above the recent accepted length: a good run can reach past the mean,
        // and an unused draft costs nothing extra because the drafter always runs
        // the full trained block width regardless of n_draft. The payoff of a row
        // that goes unused is zero; the payoff of one that lands is ~1 token.
        target = static_cast<int>(std::ceil(accept_len_ewma * gain)) + 1;
    }
    target = std::max(1, std::min(target, max_draft));

    // Slew limit of one row per round. Two reasons: a single noisy round cannot
    // swing the draft across its whole range, and the response is gradual enough
    // that a transient dip (one hard token) does not collapse a drafter that is
    // otherwise performing -- which is exactly what the old rate-based rule did.
    if (target > cur) return std::min(target, cur + 1);
    if (target < cur) return std::max(target, cur - 1);
    return cur;
}

void SpeculativeExecutor::begin_generation(const std::vector<int32_t>& prompt) {
    stats_.reset();
    last_drafter_idx_ = -1;
    adaptive_draft_ = 0;
    accept_ewma_ = -1.0;
    accept_len_ewma_ = -1.0;
    spec_off_streak_ = 0;
    spec_off_ = false;
    invalidate_verify_hiddens();
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
        static bool warned_top = false;
        if (!warned_top) {
            warned_top = true;
            LOG_WARN("SpecExecutor: top-guard fallback (chain=" + std::to_string(has_chain) +
                     " legacy=" + std::to_string(has_legacy) + " engine=" +
                     std::to_string(engine != nullptr) + " max_tokens=" +
                     std::to_string(in.max_tokens) + ")");
        }
        ++stats_.n_fallback_steps;
        ++stats_.n_fb_guard;
        return out;
    }

    // Adaptive draft sizing.
    //
    // This used to scale n_draft by (0.5 + acceptance_rate), which is a one-way
    // ratchet: any rate below 0.5 gives a multiplier < 1, so the draft shrank
    // every round and could never grow back. At the measured 41.7% acceptance it
    // walked 7 -> 1 within six rounds and then stayed at 1 for the rest of the
    // generation, which is why "draft tokens" came out ~= "spec steps" no matter
    // what --spec-draft or --spec-draft-n were set to. It was also
    // self-reinforcing: once n_draft reached 1 a round could only accept 0 or 1,
    // so the measured rate kept decaying and pinned it there for good.
    //
    // Acceptance *rate* is the wrong signal on two counts:
    //   - it is confounded with the draft length. Shrinking the draft raises the
    //     rate at constant drafter quality, so the controller rewards its own
    //     mistake and converges to 1.
    //   - the saving from a shorter draft is tiny. The drafter forward always
    //     runs the full trained block width; n_draft only truncates how many of
    //     the tokens it hands back. A discarded draft token is already paid for,
    //     so dropping it wastes work that was done.
    //
    // Size on accepted LENGTH instead -- that is the actual payoff -- and let it
    // move in both directions under a slew limit so it tracks without
    // oscillating. --no-adaptive-draft pins n_draft to the configured value.
    if (!cfg_.adaptive_draft) {
        adaptive_draft_ = cfg_.n_draft;
    } else {
        if (adaptive_draft_ <= 0) {
            adaptive_draft_ = cfg_.n_draft;  // start at the ceiling to gather signal fast
        } else {
            adaptive_draft_ =
                adaptive_draft_target(accept_len_ewma_, adaptive_draft_, cfg_.n_draft,
                                      kDraftLenGain, kEwmaAccepted, kEwmaProposed);
        }
    }
    const int n_draft = std::min({adaptive_draft_, cfg_.n_draft, in.max_tokens - 1});
    if (n_draft <= 0) {
        LOG_WARN("SpecExecutor: fallback n_draft<=0 (adaptive=" +
                 std::to_string(adaptive_draft_) + " cfg=" + std::to_string(cfg_.n_draft) +
                 " max_tokens=" + std::to_string(in.max_tokens) + ")");
        ++stats_.n_fallback_steps;
        ++stats_.n_fb_guard;
        return out;
    }

    // Adaptive early-exit: a persistently low acceptance rate means the drafter
    // costs more than it returns (e.g. a draft model larger than the target, so
    // the extra draft forward can never pay off). Require kSpecOffStreak
    // consecutive low rounds, then disable speculation for the REST of this
    // generation and let Generator fall back to plain decode at the target's full
    // speed. One-way: it does not re-enable mid-generation (acceptance is stable
    // within one prompt), so we don't oscillate back into the slow path.
    if (spec_min_acc_ > 0.0) {
        if (spec_off_) {
            ++stats_.n_fallback_steps;
            ++stats_.n_fb_min_acc;
            return out;
        }
        if (accept_ewma_ >= 0.0 && accept_ewma_ < spec_min_acc_) {
            if (++spec_off_streak_ >= kSpecOffStreak) {
                spec_off_ = true;
                LOG_WARN("SpecExecutor: acceptance EWMA=" +
                         std::to_string(accept_ewma_ * 100.0) +
                         "% < floor=" + std::to_string(spec_min_acc_ * 100.0) +
                         "% -> drafter net-negative, disabling speculation for this "
                         "generation (falling back to plain decode)");
                ++stats_.n_fallback_steps;
                ++stats_.n_fb_min_acc;
                return out;
            }
        } else {
            spec_off_streak_ = 0;
        }
    }

    std::vector<int32_t> drafts;
    int drafter_idx = -1;
    if (has_chain) {
        for (size_t i = 0; i < providers_.size(); ++i) {
            std::vector<int32_t> cand;
            {
                PERF_SCOPE("speculative/draft");
                cand = providers_[i]->draft(in.last_token, n_draft);
            }
            if (static_cast<int>(cand.size()) >= std::max(cfg_.n_min, 1)) {
                drafts = std::move(cand);
                drafter_idx = static_cast<int>(i);
                break;
            }
        }
        if (drafts.empty()) {
            LOG_WARN(std::string("SpecExecutor: fallback no-draft (provider0=") +
                     providers_[0]->name() + ")");
            ++stats_.n_fallback_steps;
            ++stats_.n_fb_no_draft;
            return out;
        }
    } else {
        {
            PERF_SCOPE("speculative/draft");
            drafts = provider_->draft(in.last_token, n_draft);
        }
        if (static_cast<int>(drafts.size()) < std::max(cfg_.n_min, 1)) {
            ++stats_.n_fallback_steps;
            ++stats_.n_fb_no_draft;
            return out;
        }
    }

    const int n_rows = static_cast<int>(drafts.size()) + 1;
    auto input_ids =
        std::make_shared<Tensor>(DataType::INT32, std::vector<int64_t>{n_rows}, DeviceType::CPU);
    int32_t* ids_ptr = static_cast<int32_t*>(input_ids->data());
    ids_ptr[0] = in.last_token;
    std::memcpy(ids_ptr + 1, drafts.data(), drafts.size() * sizeof(int32_t));
    if (ctx_.device() == DeviceType::CUDA) {
        input_ids->to_device(DeviceType::CUDA);
    }

    // 2.3: speculative 部分拒绝 draft 时回滚 recurrent (SSM) 状态, 与 KV
    // rollback 语义对齐。快照保存在 draft 验证前, spec 路径 seq_id 恒为 0。
    auto* rmem = engine->recurrent_memory();
    if (rmem)
        rmem->snapshot(0);

    // The verify forward already produces the DFlash/DSPark encoder input (the
    // target's multi-layer hidden states). Capture it here so the next round can
    // build its context K/V from it instead of re-forwarding the same tokens
    // through the target. Capture is what disables the graph path
    // (use_graph = use_graph_ && !captures_layer_hiddens()), so it is only
    // enabled when a provider actually consumes the hiddens.
    auto* tf = dynamic_cast<TransformerEngine*>(engine);
    const bool was_capturing = tf && tf->captures_layer_hiddens();
    if (tf && !was_capturing)
        tf->set_capture_layer_hiddens(true);

    TensorPtr logits_all;
    TensorPtr verify_hiddens;
    {
        PERF_SCOPE("speculative/verify_forward");
        auto verify_req = ForwardRequest::from_ids(input_ids, in.pos);
        verify_req.return_all_logits = true;
        logits_all = engine->forward_request(verify_req);
        if (logits_all->device() != DeviceType::CPU)
            logits_all->to_device(DeviceType::CPU);
        if (tf)
            verify_hiddens = tf->take_layer_hiddens(capture_layers_);
    }
    if (tf && !was_capturing)
        tf->set_capture_layer_hiddens(false);

    if (!capture_layers_.empty() && !verify_hiddens)
        LOG_WARN("SpecExecutor: verify forward produced no target hiddens for the capture "
                 "layers; the drafter will fall back to a target re-forward this round");
    // Hand the hiddens over: replaces any previous stash. Callers that add tokens
    // the target did not just compute (plain-decode confirmations) must call
    // invalidate_verify_hiddens() instead, since a stale stash would silently
    // encode the wrong token into the drafter's context K/V.
    for (auto& p : providers_) {
        if (p->capture_layers())
            p->set_verify_hiddens(verify_hiddens);
    }
    if (provider_ && provider_->capture_layers())
        provider_->set_verify_hiddens(verify_hiddens);

    if (!logits_all || logits_all->ndim() < 2 || logits_all->shape()[0] < n_rows) {
        LOG_WARN("SpeculativeExecutor: verify forward failed, falling back to plain decode");
        ++stats_.n_fallback_steps;
        ++stats_.n_fb_verify_fail;
        if (rmem)
            rmem->rollback(0);
        return out;
    }

    const int vocab_size = static_cast<int>(logits_all->shape()[1]);
    const float* logits_data = static_cast<const float*>(logits_all->data());

    SpecVerifyResult vr;
    {
        PERF_SCOPE("speculative/verify_accept");
        vr = verify_draft_tokens(logits_data, vocab_size, drafts, sampler_, cfg_);
    }

    out.tokens.insert(out.tokens.end(), vr.accepted_tokens.begin(), vr.accepted_tokens.end());
    if (vr.resampled >= 0) {
        out.tokens.push_back(vr.resampled);
    } else if (vr.bonus >= 0) {
        out.tokens.push_back(vr.bonus);
    }
    if (out.tokens.empty()) {
        static bool warned_empty = false;
        if (!warned_empty) {
            warned_empty = true;
            ++stats_.n_fb_empty;
            LOG_WARN("SpecExecutor: fallback tokens-empty (accepted=" +
                     std::to_string(vr.accepted_tokens.size()) + " resampled=" +
                     std::to_string(vr.resampled) + " bonus=" + std::to_string(vr.bonus) + ")");
        }
        ++stats_.n_fallback_steps;
        return out;
    }

    const bool full_accept = (vr.n_accepted == static_cast<int>(drafts.size()));
    if (!full_accept) {
        KVCache* kv = engine->kv_cache();
        if (kv)
            kv->rollback(in.pos + vr.n_accepted + 1);
        // recurrent 状态回到 draft 前快照, 再重放已接受 token 推进到接受点。
        if (rmem) {
            rmem->rollback(0);
            {
                PERF_SCOPE("speculative/replay");
                replay_recurrent(engine, ctx_, in.last_token, drafts, vr.n_accepted, in.pos);
            }
        }
    }

    out.speculated = true;
    last_drafter_idx_ = drafter_idx;

    // Update the acceptance EWMAs. Kept separate on purpose: the rate drives the
    // --spec-min-acc early exit (a quality measure) while the length drives
    // adaptive sizing (a payoff measure). They can disagree -- a round that
    // accepts 1 of 1 is 100% but yields nothing.
    {
        const double round_acc =
            drafts.empty() ? 0.0 : static_cast<double>(vr.n_accepted) / drafts.size();
        if (accept_ewma_ < 0.0) {
            accept_ewma_ = round_acc;
        } else {
            accept_ewma_ = kEwmaAccepted * accept_ewma_ + kEwmaProposed * round_acc;
        }
        const double round_len = static_cast<double>(vr.n_accepted);
        if (accept_len_ewma_ < 0.0) {
            accept_len_ewma_ = round_len;
        } else {
            accept_len_ewma_ =
                kEwmaAccepted * accept_len_ewma_ + kEwmaProposed * round_len;
        }
    }

    ++stats_.n_spec_steps;
    stats_.n_draft_tokens += static_cast<int64_t>(drafts.size());
    stats_.n_accepted_tokens += vr.n_accepted;
    stats_.n_output_tokens += static_cast<int64_t>(out.tokens.size());
    // Per-position acceptance and the per-round proposed length. See the struct
    // comment: position 0 vs the tail is the diagnostic that separates "the
    // drafter is inaccurate" from "the tail was never worth proposing".
    {
        const int nd = static_cast<int>(drafts.size());
        for (int j = 0; j < nd && j < SpeculativeStats::kMaxTrackedPos; ++j)
            ++stats_.n_proposed[j];
        for (int j = 0; j < vr.n_accepted && j < SpeculativeStats::kMaxTrackedPos; ++j)
            ++stats_.n_accepted_at[j];
        // buckets: 0, 1, 2, 3, 4-5, 6-7, 8+  ->  idx 0..6
        int bucket = 6;
        if (nd <= 3) {
            bucket = nd;
        } else if (nd <= 5) {
            bucket = 4;
        } else if (nd <= 7) {
            bucket = 5;
        }
        ++stats_.n_rounds_by_len[bucket];
    }
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
    if (kv)
        kv->rollback(valid_end_pos);
    // Truncating the committed stream can pull an uncaptured token into the next
    // drafter delta, so the captured-hiddens stash is no longer trustworthy.
    invalidate_verify_hiddens();
}

void SpeculativeExecutor::invalidate_verify_hiddens() {
    for (auto& p : providers_) {
        if (p->capture_layers())
            p->set_verify_hiddens(nullptr);
    }
    if (provider_ && provider_->capture_layers())
        provider_->set_verify_hiddens(nullptr);
}

void SpeculativeExecutor::notify_confirmed(int32_t token) {
    // This token came from the plain-decode path, so the target never produced
    // hiddens for it. The next drafter delta may include it, which would make a
    // stale stash encode the WRONG token into the drafter context K/V (the row
    // that happens to sit at that index belongs to a rejected draft instead).
    // Drop the stash so the next round falls back to the replay path.
    invalidate_verify_hiddens();
    for (auto& p : providers_) {
        p->accept({token});
    }
    if (provider_)
        provider_->accept({token});
}

}  // namespace forge
