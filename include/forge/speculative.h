#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
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
    bool print_stats = false;   // CLI: print SpeculativeStats after generation (--spec-stats)

    // Standalone small draft model (Phase 3); takes priority over n-gram when set
    std::string draft_model_path;
    int draft_gpu_layers = -1;  // GPU offload layers for draft model (-1 = follow target)

    // n-gram self-speculative parameters
    bool use_mtp = false;       // DeepSeek-MTP style nextn draft head (qwen35)
    bool use_ngram = true;      // enable n-gram candidate source
    int ngram_n = 5;            // longest suffix match length
    int ngram_min = 2;          // shortest suffix match length
};

// =========================================================================
// IDraftProvider -- candidate source abstraction.
//
// Self-managed state: callers (SpeculativeExecutor) never pass token history.
//   begin(prompt)                     once per generation start
//   [draft(last, n) -> (target verify) -> accept(confirmed)]*   main loop
//   reset()                           clear internal state
//
// `last` in draft() is the latest confirmed token (already reflected in
// accept() calls EXCEPT the prefill-sampled first token, which arrives only
// here). Providers that track the full history internally (n-gram) may
// ignore it.
// =========================================================================

class IDraftProvider {
public:
    virtual ~IDraftProvider() = default;

    // Called once per generation, after prompt prefill. Implementations should
    // initialize their state from `prompt`.
    virtual void begin(const std::vector<int32_t>& /*prompt*/) {}

    // Propose at most n_draft candidate tokens continuing after `last_token`;
    // return empty if none available.
    virtual std::vector<int32_t> draft(int32_t last_token, int n_draft) = 0;

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
//
// Maintains a hash index (n-gram key -> ascending occurrence positions) over
// all window lengths [ngram_min_, ngram_n_] so draft() is O(match length)
// instead of an O(len * n) scan. The index is built incrementally as tokens
// are confirmed. Among all occurrences of a matched suffix the MOST RECENT
// one with at least one continuation token available wins (recency beats
// distance for prediction quality).

class NgramDraftProvider : public IDraftProvider {
public:
    explicit NgramDraftProvider(int ngram_n = 5, int ngram_min = 2);

    void begin(const std::vector<int32_t>& prompt) override;
    std::vector<int32_t> draft(int32_t last_token, int n_draft) override;
    void accept(const std::vector<int32_t>& tokens) override;
    void reset() override;
    const char* name() const override { return "ngram"; }

private:
    // Index every window completed as the history grows from old_len to
    // new_len (windows straddling the old boundary included).
    void index_range(int old_len, int new_len);

    int ngram_n_;
    int ngram_min_;
    std::vector<int32_t> history_;  // full confirmed token sequence (prompt + generated)

    struct VecHash {
        size_t operator()(const std::vector<int32_t>& v) const {
            size_t h = 1469598103934665603ull;  // FNV-1a 64-bit offset basis
            for (int32_t t : v) {
                h ^= static_cast<size_t>(static_cast<uint32_t>(t));
                h *= 1099511628211ull;
            }
            return h;
        }
    };
    // key -> ascending start positions of occurrences inside history_
    std::unordered_map<std::vector<int32_t>, std::vector<int32_t>, VecHash> index_;
};

// ---- Standalone small-model draft provider ----
// Runs a second GGUF model in its own InferenceContext and proposes sampled
// continuations from a top-k restricted chain (llama.cpp DRAFT_SIMPLE style).
// The draft context keeps its own KV cache; accept() trims it back to the
// last confirmed boundary and replays the confirmed tokens so both sides stay
// aligned for the next round.

struct ContextParams;  // defined in forge/context.h (circular-include guard)

class ModelDraftProvider : public IDraftProvider {
public:
    // Loads the draft model described by spec.draft_model_path. `target`
    // supplies device/threading defaults (spec.draft_gpu_layers >= 0
    // overrides the layer offload count); target_vocab must match the draft
    // model's vocabulary or valid() returns false.
    ModelDraftProvider(const SpeculativeConfig& spec, const ContextParams& target,
                       int target_vocab);
    ~ModelDraftProvider() override;

    bool valid() const;

    void begin(const std::vector<int32_t>& prompt) override;
    std::vector<int32_t> draft(int32_t last_token, int n_draft) override;
    void accept(const std::vector<int32_t>& tokens) override;
    void reset() override;
    const char* name() const override { return "model"; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// =========================================================================
// Verification entry.
// logits_all: [1+n_draft, vocab_size] row-major; row i predicts the token at
// position start_pos+i+1, i.e. row i verifies draft_tokens[i]. When all drafts
// are accepted, the last row yields the bonus token.
//
// Resample-consistency (lossless): with do_sample=true every position runs
// the sampler's full chain; a draft token is accepted iff it equals the
// target's own sample, otherwise that sample becomes the continuation.
// Output distribution == plain decode. Greedy mode uses argmax matching.
// =========================================================================

SpecVerifyResult verify_draft_tokens(
    const float* logits_all, int vocab_size,
    const std::vector<int32_t>& draft_tokens,
    Sampler& sampler, const SpeculativeConfig& config);

}  // namespace forge
