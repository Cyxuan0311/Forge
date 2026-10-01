#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace forge {

class Sampler;
class Tensor;

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

    // Where the DRAFTED token sat in the TARGET's distribution at the first
    // rejected position, and the target's probability mass on it. Diagnostic: it
    // separates "the drafter nearly had it" (low rank) from "the drafter was
    // elsewhere" (high rank), which need opposite fixes. -1 when not applicable.
    int draft_rank_of_truth = -1;
    double draft_prob_of_truth = -1.0;
};

// ---- Runtime statistics ----
struct SpeculativeStats {
    int64_t n_spec_steps = 0;       // speculation rounds (one draft->verify cycle each)
    int64_t n_fallback_steps = 0;   // rounds degraded to plain decode (no candidates / forward failure)
    int64_t n_draft_tokens = 0;     // total proposed candidate tokens
    int64_t n_accepted_tokens = 0;  // accepted candidates (excludes resampled/bonus)
    int64_t n_output_tokens = 0;    // tokens produced via spec path (accepted + resampled/bonus)

    // Fallback rounds broken down by cause. A single n_fallback_steps counter hid
    // the two failures that actually cost the most here: rounds where the
    // provider returned nothing (n_fb_no_draft, typically confidence truncation
    // emptying the batch) and rounds abandoned by the --spec-min-acc floor. A
    // large no-draft count with a healthy acceptance rate means the drafter is
    // working but is being asked for nothing, not that it is inaccurate.
    int64_t n_fb_no_draft = 0;      // provider returned fewer than n_min candidates
    int64_t n_fb_min_acc = 0;       // --spec-min-acc floor tripped, speculation disabled
    int64_t n_fb_verify_fail = 0;   // verify forward returned no usable logits
    int64_t n_fb_empty = 0;         // verification produced no tokens at all
    int64_t n_fb_guard = 0;         // no provider / no engine / max_tokens<=1

    // Acceptance by draft position: the first proposed token is by far the most
    // likely to be accepted and the last the least, so a flat curve means the
    // tail is dead weight while a cliff at position 0 means the drafter's very
    // first guess is wrong (a prompt-encoding or conditioning problem, not a
    // depth problem). Indexed by position; entries at or beyond the largest
    // round size requested stay 0.
    static constexpr int kMaxTrackedPos = 16;
    int64_t n_proposed[kMaxTrackedPos] = {0};
    int64_t n_accepted_at[kMaxTrackedPos] = {0};

    // How many tokens each round actually proposed, bucketed 0 / 1 / 2 / 3 /
    // 4-5 / 6-7 / 8+. Reveals a draft that has collapsed to a constant width
    // (adaptive sizing pinned by its own control signal) as opposed to one that
    // is genuinely varying with confidence.
    int64_t n_rounds_by_len[7] = {0};

    // Where the DRAFTED token sat in the TARGET's own distribution, at the first
    // rejected position. This is the diagnostic that says whether the drafter was
    // "nearly right and mis-ranked" (low rank) or "elsewhere" (rank 5+), which are
    // opposite problems. Bucketed: 0 = rank 1, 1-4 = close, 5 aggregates 5+.
    // It must rank the draft inside the target's row -- ranking the target's own
    // sampled token reports a meaningless permanent "#1".
    // Populated only when probe_ranks is set (env-gated): it costs a vocab scan
    // per rejection.
    int64_t n_rank_hist[6] = {0};   // idx 5 aggregates rank >= 5
    double rank_prob_sum = 0.0;
    int64_t n_rank_samples = 0;
    bool probe_ranks = false;

    void reset();
    double acceptance_rate() const;   // candidate acceptance rate [0,1]
    double tokens_per_step() const;   // avg tokens produced per speculation round (>1 means speedup)
    double acceptance_at(int pos) const;  // -1.0 when that position was never proposed
    void record_truth_rank(int rank, double prob);
};

// ---- Configuration ----
struct SpeculativeConfig {
    bool enabled = false;       // master switch, off by default
    int n_draft = 5;            // max drafted tokens per round (--spec-draft)
    int n_min = 0;              // drop the whole round if fewer candidates than this (--spec-n-min)
    float p_min = 0.0f;         // draft confidence early-stop threshold (Phase 3 model draft)
    bool print_stats = false;   // CLI: print SpeculativeStats after generation (--spec-stats)

    // Size n_draft per round to track the recent accepted length. OFF by default,
    // matching llama.cpp, which uses a static draft length every round
    // (common/speculative.cpp:1202, `const int32_t n_draft = params.n_max;`)
    // and only clamps it once to the drafter's trained block size (line 1001).
    //
    // It used to be on by default, and it was a trap. The controller shrinks the
    // draft when acceptance is low, but a shorter draft raises the acceptance rate
    // at constant drafter quality, so the controller rewards its own mistake and
    // walks the length down to 1. At 1 token per round the payoff is capped at
    // ~1.4 tokens/step, and since the drafter forward still runs the full trained
    // block width, the discarded tokens are work already paid for and thrown away.
    //
    // Measured on MiniCPM5-2.6B-DSpark over a 2B target: adaptive sizing collapsed
    // to 1 token within a few rounds and held there for the rest of the generation,
    // hiding the real per-position acceptance data (34% / 15% / 3%) behind a single
    // average. Turn it on with --adaptive-draft; --no-adaptive-draft is kept as an
    // accepted no-op so existing scripts keep working.
    bool adaptive_draft = false;

    // Standalone small draft model (Phase 3); takes priority over n-gram when set
    std::string draft_model_path;
    int draft_gpu_layers = -1;  // GPU offload layers for draft model (-1 = follow target)

    // ---- DFlash / DSPark draft models (Phase 0 skeleton, engine in Phase 2) ----
    std::string draft_arch;                // "" (off) / "dflash" / "dspark"
    std::vector<int> draft_target_layers;  // override for the encoder's target layers
    int draft_mask_token_id = -1;          // MASK token for the query block (<0 = read from GGUF)
    int draft_n_spec = 5;                  // speculative tokens proposed per round

    // DSpark (MiniCPM5-2.6B-DSpark) tuning knobs, override the GGUF-derived defaults.
    // <0 means "not set on the command line", so the GGUF default is used; 0 is a
    // real value meaning "off" and is distinguishable from unset. Using 0 as the
    // unset marker made `--spec-dspark-conf 0` silently fall back to 0.5, i.e.
    // the opposite of what it reads as.
    int dspark_top_k = -1;              // truncated candidate re-ranking top-k (0 = full vocab)
    float dspark_conf_threshold = -1.0f;  // confidence truncation threshold (<0 = from GGUF, 0 = off)

    // Acceptance floor below which the drafter is judged net-negative and
    // speculation is disabled for the rest of the generation (<=0 = never).
    // Speculation only pays off when cost(draft) + cost(verify) <
    // (accepted + 1) * cost(plain); with a drafter as large as the target that
    // needs a high acceptance rate, so a persistently low EWMA is a reliable
    // signal that the round is costing more than it returns.
    float min_accept_rate = 0.0f;

    // n-gram self-speculative parameters
    bool use_mtp = false;       // DeepSeek-MTP style nextn draft head (qwen35)
    bool use_ngram = true;      // enable n-gram candidate source
    bool no_ngram = false;      // --no-ngram: disable ngram fallback (used with --spec-ngram-mod to isolate mod)
    int ngram_n = 5;            // longest suffix match length
    int ngram_min = 2;          // shortest suffix match length

    // n-gram-mod global hash pool (llama.cpp ngram-mod style)
    bool use_ngram_mod = false;     // enable global hash-pool ngram-mod
    int ngram_mod_n = 24;           // hash key length (llama n_match)
    int ngram_mod_n_min = 1;        // min draft tokens to keep when mod truncates (lowered from 48)
    int ngram_mod_n_max = 64;       // max draft tokens to pull from mod pool
    size_t ngram_mod_pool_size = 4 * 1024 * 1024; // entries, ~16 MB
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

    // Whether this provider tracks global history and should receive
    // accept() even when it was not the drafter (e.g. n-gram pools).
    virtual bool is_global() const { return false; }

    // ---- Target hidden-state handoff (DFlash / DSPark) ----
    //
    // Cross-attention drafters (DFlash/DSPark) build their incremental context
    // KV by projecting the TARGET's multi-layer hidden states. Those states are
    // already produced by the target's own verify forward, so re-running the
    // target over the same tokens is pure duplicated work (a full 42-layer
    // forward per round, and it also forces the target off its CUDA graph).
    // The executor therefore captures them during the verify forward and hands
    // them over here; the provider slices the rows it needs.
    //
    // This mirrors vLLM's `pass_hidden_states_to_model=True` proposer contract
    // (vllm/v1/spec_decode/dflash.py:35): the drafter consumes the target's
    // hidden states straight out of the target forward, never a replay.
    //
    // capture_layers(): which target layers to concatenate into the encoder
    // input. nullptr (default) means the provider does not consume target
    // hiddens, in which case the executor skips the capture entirely and
    // leaves the target on its graph.
    virtual const std::vector<int>* capture_layers() const { return nullptr; }

    // Receive the captured encoder input for the most recent verify forward:
    // [n_verify_rows, sum(hidden over capture_layers())] on the CPU, row j being
    // the encoder input for the token at that forward's `start_pos + j`.
    // Providers must treat it as a single-shot stash: it is replaced after every
    // verify and cleared whenever an uncaptured token can enter the delta range
    // (plain-decode confirmations, generation restarts, KV truncation).
    virtual void set_verify_hiddens(std::shared_ptr<Tensor> /*hiddens*/) {}
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
    bool is_global() const override { return true; }

private:
    // Rolling FNV-1a hash for a token window [ptr, ptr+len).
    static uint64_t hash_window(const int32_t* ptr, int len);
    // Index every window completed as the history grows from old_len to
    // new_len (windows straddling the old boundary included).
    void index_range(int old_len, int new_len);

    int ngram_n_;
    int ngram_min_;
    std::vector<int32_t> history_;  // full confirmed token sequence (prompt + generated)

    // key hash -> ascending start positions of occurrences inside history_
    std::unordered_map<uint64_t, std::vector<int32_t>> index_;
};

// ---- Global hash-pool n-gram provider (llama.cpp ngram-mod style) ----
// Constant-memory O(1) lookup via a shared 4M-entry table. The pool is
// process-global so concurrent sequences benefit from shared history
// (cross-request prefix reuse for code/reasoning workloads).
class NgramModProvider : public IDraftProvider {
public:
    explicit NgramModProvider(int n_match = 24, int n_min = 48, int n_max = 64,
                              size_t pool_size = 4 * 1024 * 1024);
    void begin(const std::vector<int32_t>& prompt) override;
    std::vector<int32_t> draft(int32_t last_token, int n_draft) override;
    void accept(const std::vector<int32_t>& tokens) override;
    void reset() override;
    const char* name() const override { return "ngram-mod"; }
    bool is_global() const override { return true; }

private:
    // LCG hash matching llama.cpp common_ngram_mod::idx
    static uint64_t lcg_hash(const int32_t* tokens, int n);

    int n_match_;
    int n_min_;
    int n_max_;
    std::vector<int32_t> history_;

    // Per-provider bookkeeping for lazy indexing and reset heuristics
    int i_last_ = 0;              // next history position to index into pool
    int n_draft_last_ = 0;
    int n_low_streak_ = 0;        // consecutive low-accept rounds

    // ---- Global shared pool (singleton) ----
    struct SharedPool {
        std::vector<int32_t> table;
        size_t used = 0;
        std::mutex mutex;
    };
    static SharedPool& shared_pool();
    static size_t pool_size_;
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
