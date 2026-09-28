#pragma once

#include <memory>
#include <vector>

#include "speculative.h"

namespace forge {

class InferenceContext;
class Sampler;

// =========================================================================
// SpeculativeExecutor -- speculation orchestrator.
//
// Encapsulates one draft -> verify -> KV-rollback cycle per step() call.
// Generator's main loop calls step(); on a non-speculated round it falls
// back to its plain single-token decode path.
//
// NOT handled here (stays in Generator): EOS/stop-token decisions, streaming
// callbacks, the plain decode path itself.
//
// KV invariant after a successful step(): valid KV region is
// [0, in.pos + out.tokens.size()); the next token must be written at that
// position. If Generator stops early mid-output (e.g. EOS inside accepted
// tokens) it must call rollback_kv() to truncate leftover rows.
// =========================================================================

// Sizing law for adaptive draft, as a free function so it can be unit-tested
// without a model (the control loop lives in SpeculativeExecutor::step).
//
// `accept_len_ewma` is the recent EWMA of accepted tokens per round and `cur` the
// current effective n_draft. Returns the next n_draft, clamped to [1, max_draft]
// and slewed by at most one row per call.
//
// Note it sizes on accepted LENGTH, never on acceptance rate. Scaling by the rate
// is a one-way ratchet: any rate below 0.5 gives a multiplier under 1, so the
// draft shrinks every round and cannot recover. It is also self-reinforcing,
// because a shorter draft scores a higher rate at unchanged drafter quality, so
// the controller converges to 1 no matter how good the drafter is.
int adaptive_draft_target(double accept_len_ewma, int cur, int max_draft,
                          double gain = 1.3, double accepted = 0.7, double proposed = 0.3);

class SpeculativeExecutor {
public:
    struct StepInput {
        int32_t last_token;   // last confirmed token, NOT yet written to KV
        int64_t pos;          // KV position where last_token will be written
        int max_tokens;       // max tokens this round may produce (budget left)
    };

    struct StepOutput {
        bool speculated = false;      // false = no candidates / failure -> plain decode
        std::vector<int32_t> tokens;  // produced: accepted... (+ resampled|bonus)
    };

    SpeculativeExecutor(InferenceContext& ctx, Sampler& sampler,
                        const SpeculativeConfig& cfg);
    ~SpeculativeExecutor();

    // Whether a draft provider is available and usable.
    bool valid() const { return !providers_.empty() || provider_ != nullptr; }
    bool has_providers() const { return !providers_.empty(); }

    // Call once per generation after prompt prefill. Resets stats and provider state.
    void begin_generation(const std::vector<int32_t>& prompt);

    // Run one speculation cycle. Returns speculated=false to request fallback.
    StepOutput step(const StepInput& in);

    // Truncate KV so that only positions [0, valid_end_pos) remain. Used when
    // generation stops early inside a speculated batch output.
    void rollback_kv(int64_t valid_end_pos);

    // Notify the draft provider about a token produced by the plain fallback
    // path, keeping a model draft context synchronized between rounds.
    void notify_confirmed(int32_t token);

    // Drop the captured target-hiddens stash held by the providers. Required
    // whenever a token enters the committed stream that the capturing verify
    // forward did not compute (plain-decode confirmations, KV truncation).
    void invalidate_verify_hiddens();

    const SpeculativeStats& stats() const { return stats_; }
    SpeculativeStats& stats_mut() { return stats_; }

private:
    InferenceContext& ctx_;
    Sampler& sampler_;
    SpeculativeConfig cfg_;
    std::vector<DraftProviderPtr> providers_;
    DraftProviderPtr provider_; // legacy single-provider accessor (points to primary)
    int last_drafter_idx_ = -1;
    SpeculativeStats stats_;

    // Adaptive draft control: n_draft tracks the recent accepted LENGTH, not the
    // acceptance rate. The rate is confounded with the draft length (a shorter
    // draft scores a higher rate at unchanged drafter quality), so using it here
    // made the controller shrink toward 1 and never recover. See step().
    int adaptive_draft_ = 0;       // current effective n_draft (0 = uninitialized)
    double accept_ewma_ = -1.0;     // EWMA acceptance rate over recent rounds
    double accept_len_ewma_ = -1.0; // EWMA accepted tokens per round

    // Aim this multiple of the recent accepted length when sizing n_draft. >1 so
    // a strong run can reach past the mean; unused drafts are free because the
    // drafter forward always runs the full trained block width.
    static constexpr double kDraftLenGain = 1.3;

    // EWMA weights, shared by both signals so the rate (--spec-min-acc) and the
    // length (adaptive sizing) react to the same history at the same rate.
    static constexpr double kEwmaAccepted = 0.7;
    static constexpr double kEwmaProposed = 0.3;

    // Adaptive early-exit: when the acceptance EWMA stays below `spec_min_acc_`
    // for kSpecOffStreak consecutive rounds, speculation is judged net-negative
    // (e.g. a drafter larger than the target, so the extra draft forward can
    // never pay off) and is disabled for the REST of this generation -- a plain
    // decode fallback runs at the target's full speed. One-way: it does not
    // re-enable mid-generation, since acceptance is stable within one prompt.
    // Only engages once accept_ewma_ is initialized (>0 rounds seen).
    double spec_min_acc_ = 0.0;     // floor from cfg_.min_accept_rate (<=0 disables)
    int spec_off_streak_ = 0;       // consecutive rounds below floor
    bool spec_off_ = false;         // speculation disabled for this generation
    static constexpr int kSpecOffStreak = 5;

    // Target layers to capture out of the verify forward for the DFlash/DSPark
    // encoder. Empty means no provider consumes them, so the target keeps its
    // CUDA graph (capture forces the imperative path).
    std::vector<int> capture_layers_;
};

}  // namespace forge
