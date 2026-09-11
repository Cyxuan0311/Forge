#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "context.h"
#include "model.h"
#include "sampler.h"

namespace forge {

class SpeculativeExecutor;

struct GenerationConfig {
    int max_new_tokens = 256;
    float temperature = 1.0f;
    int top_k = 0;
    float top_p = 1.0f;
    float repeat_penalty = 1.0f;
    int repeat_last_n = 64;  // Window size for repeat penalty (<=0 means use full history)
    bool do_sample = true;
    uint64_t seed = 0;
    int eos_token_id = -1;
    std::vector<int32_t> stop_token_ids;  // Additional stop tokens beyond eos
    bool reset_kv_cache = true;           // If false, append to existing KV (multi-turn)
    // Number of leading prompt tokens whose KV rows are already resident from a
    // previous request (session prefix reuse). Only prompt_tokens[prefix_len:]
    // is forwarded, starting at KV position prefix_len; the caller must
    // guarantee the cached positions really hold those token ids. Ignored when
    // 0. Implies reset_kv_cache == false.
    int prefix_len = 0;
    // When non-null and the pointee becomes true, generate() stops at the next
    // decode boundary and reports finish_reason "aborted". Owned by the caller
    // and must outlive the generate() call.
    const std::atomic<bool>* abort_flag = nullptr;
};

struct GenerationResult {
    std::vector<int32_t> token_ids;
    std::string text;
    int num_prompt_tokens = 0;
    int num_generated_tokens = 0;
    bool finished = false;
    std::string finish_reason;
    const SpeculativeStats* spec_stats = nullptr;  // set by Generator after generate()

    // ---- Per-request timing (always populated, independent of FORGE_PROFILING) ----
    double prefill_ms = 0.0;  // prompt forward + first-token sample
    double decode_ms = 0.0;   // decode loop (excludes prefill; includes sampling)
    double prefill_ms_per_1k() const {
        return num_prompt_tokens > 0 ? prefill_ms * 1000.0 / num_prompt_tokens : 0.0;
    }
    double tokens_per_sec() const {
        // (generated - 1): the first generated token comes out of prefill.
        const int n = num_generated_tokens - 1;
        return n > 0 ? n / (decode_ms / 1000.0) : 0.0;
    }
};

class Generator {
public:
    Generator(InferenceContext& ctx, const SamplerConfig& sampler_cfg = SamplerConfig{});
    ~Generator();

    GenerationResult generate(const std::vector<int32_t>& prompt_tokens,
                              const GenerationConfig& config);

    using TokenCallback = std::function<void(int32_t token_id, int step)>;

    GenerationResult generate(const std::vector<int32_t>& prompt_tokens,
                              const GenerationConfig& config, const TokenCallback& callback);

    // Speculation statistics of the underlying executor (nullptr when the
    // speculative config is disabled).
    const SpeculativeStats* spec_stats() const;

private:
    InferenceContext& ctx_;
    Sampler sampler_;
    TensorPtr decode_input_gpu_;
    void* decode_token_buf_ = nullptr;
    std::unique_ptr<SpeculativeExecutor> spec_executor_;  // null when speculation disabled
};

}  // namespace forge
