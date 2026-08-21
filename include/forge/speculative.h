#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace forge {

class Sampler;

// =========================================================================
// Speculative Decoding public types.
//
// Layered architecture (mirrors llama.cpp common/speculative.*):
//   IDraftProvider      -- candidate source abstraction (n-gram / draft model / MTP ...)
//   verify_draft_tokens -- verification algorithm (greedy fast path /
//                          resample-consistency in Phase 1)
//   SpeculativeExecutor -- orchestrator (draft -> verify -> KV rollback),
//                          see forge/speculative_executor.h
// =========================================================================

// ---- Per-round verification result ----
struct SpecVerifyResult {
    int n_accepted = 0;                      // number of accepted draft tokens
    std::vector<int32_t> accepted_tokens;    // accepted draft tokens
    int32_t resampled = -1;                  // token re-sampled by target at first rejection
    int32_t bonus = -1;                      // bonus token when all drafts accepted
};

// ---- Runtime statistics ----
struct SpeculativeStats {
    int64_t n_spec_steps = 0;       // speculation rounds (one draft->verify cycle each)
    int64_t n_fallback_steps = 0;   // rounds degraded to plain decode (no candidates / forward failure)
    int64_t n_draft_tokens = 0;     // total proposed candidate tokens
    int64_t n_accepted_tokens = 0;  // accepted candidates (excludes resampled/bonus)
    int64_t n_output_tokens = 0;    // tokens produced via spec path (accepted + resampled/bonus)

    void reset();
    double acceptance_rate() const;   // candidate acceptance rate [0,1]
    double tokens_per_step() const;   // avg tokens produced per speculation round (>1 means speedup)
};

// ---- Configuration ----
struct SpeculativeConfig {
    bool enabled = false;       // master switch, off by default
    int n_draft = 5;            // max drafted tokens per round (--spec-draft)
    int n_min = 0;              // drop the whole round if fewer candidates than this (--spec-n-min)
    float p_min = 0.0f;         // draft confidence early-stop threshold (Phase 3 model draft)

    // Standalone small draft model (Phase 3); takes priority over n-gram when set
    std::string draft_model_path;
    int draft_gpu_layers = -1;  // GPU offload layers for draft model (-1 = follow target)

    // n-gram self-speculative parameters
    bool use_ngram = true;      // enable n-gram candidate source
    int ngram_n = 5;            // longest suffix match length
    int ngram_min = 2;          // shortest suffix match length
};

// =========================================================================
// IDraftProvider -- candidate source abstraction.
//
// Self-managed state: callers (SpeculativeExecutor) never pass token history.
//   begin(prompt)                     once per generation start
//   [draft(n) -> (target verify) -> accept(confirmed)]*  main loop
//   reset()                           clear internal state
// =========================================================================

class IDraftProvider {
public:
    virtual ~IDraftProvider() = default;

    // Called once per generation, after prompt prefill. Implementations should
    // initialize their state from `prompt`.
    virtual void begin(const std::vector<int32_t>& /*prompt*/) {}

    // Propose at most n_draft candidate tokens; return empty if none available.
    virtual std::vector<int32_t> draft(int n_draft) = 0;

    // Called after each verification round with the tokens finally confirmed
    // by the target (accepted + resampled/bonus).
    virtual void accept(const std::vector<int32_t>& tokens) = 0;

    virtual void reset() {}

    virtual const char* name() const = 0;
};

using DraftProviderPtr = std::unique_ptr<IDraftProvider>;

// ---- N-gram self-speculative draft provider ----
// Suffix-matches against its own confirmed token history and returns the
// continuation found at the match position.

class NgramDraftProvider : public IDraftProvider {
public:
    explicit NgramDraftProvider(int ngram_n = 5, int ngram_min = 2);

    void begin(const std::vector<int32_t>& prompt) override;
    std::vector<int32_t> draft(int n_draft) override;
    void accept(const std::vector<int32_t>& tokens) override;
    void reset() override;
    const char* name() const override { return "ngram"; }

private:
    int ngram_n_;
    int ngram_min_;
    std::vector<int32_t> history_;  // full confirmed token sequence (prompt + generated)
};

// =========================================================================
// Verification entry.
// logits_all: [1+n_draft, vocab_size] row-major; row i predicts the token at
// position start_pos+i+1, i.e. row i verifies draft_tokens[i]. When all drafts
// are accepted, the last row yields the bonus token.
// NOTE: currently the greedy-match fast path only; Phase 1 adds
//       resample-consistency for do_sample (aligned with llama.cpp
//       common_sampler_sample_and_accept_n).
// =========================================================================

SpecVerifyResult verify_draft_tokens(
    const float* logits_all, int vocab_size,
    const std::vector<int32_t>& draft_tokens,
    Sampler& sampler, const SpeculativeConfig& config);

}  // namespace forge
