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
    bool valid() const { return provider_ != nullptr; }

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

    const SpeculativeStats& stats() const { return stats_; }
    SpeculativeStats& stats_mut() { return stats_; }

private:
    InferenceContext& ctx_;
    Sampler& sampler_;
    SpeculativeConfig cfg_;
    DraftProviderPtr provider_;
    SpeculativeStats stats_;
};

}  // namespace forge
