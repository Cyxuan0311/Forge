#include "forge/speculative.h"

#include <algorithm>

#include "forge/sampler.h"

namespace forge {

// =========================================================================
// SpeculativeStats
// =========================================================================

void SpeculativeStats::reset() { *this = SpeculativeStats{}; }

double SpeculativeStats::acceptance_rate() const {
    return n_draft_tokens > 0
               ? static_cast<double>(n_accepted_tokens) / static_cast<double>(n_draft_tokens)
               : 0.0;
}

double SpeculativeStats::tokens_per_step() const {
    return n_spec_steps > 0
               ? static_cast<double>(n_output_tokens) / static_cast<double>(n_spec_steps)
               : 0.0;
}

// =========================================================================
// NgramDraftProvider -- hash-indexed suffix matching
// =========================================================================

NgramDraftProvider::NgramDraftProvider(int ngram_n, int ngram_min)
    : ngram_n_(ngram_n), ngram_min_(ngram_min) {}

void NgramDraftProvider::index_range(int old_len, int new_len) {
    // Every window [s, s+L) that became complete as the history grew from
    // old_len to new_len tokens. Windows straddling the old boundary are
    // included: they start at s >= old_len-L+1.
    for (int L = ngram_min_; L <= ngram_n_; ++L) {
        const int s_begin = std::max(0, old_len - L + 1);
        const int s_end = new_len - L;  // inclusive
        for (int s = s_begin; s <= s_end; ++s) {
            std::vector<int32_t> key(history_.begin() + s, history_.begin() + s + L);
            index_[std::move(key)].push_back(s);
        }
    }
}

void NgramDraftProvider::begin(const std::vector<int32_t>& prompt) {
    history_ = prompt;
    index_.clear();
    index_range(0, static_cast<int>(history_.size()));
}

std::vector<int32_t> NgramDraftProvider::draft(int32_t last_token, int n_draft) {
    (void)last_token;
    std::vector<int32_t> result;
    const int len = static_cast<int>(history_.size());
    if (len < ngram_min_ || n_draft <= 0) return result;

    // Longest suffix first; among occurrences of a matched suffix prefer the
    // most recent one that still has >=1 continuation token available.
    const int max_L = std::min(ngram_n_, len);
    for (int L = max_L; L >= ngram_min_; --L) {
        std::vector<int32_t> key(history_.end() - L, history_.end());
        auto it = index_.find(key);
        if (it == index_.end()) continue;

        const auto& positions = it->second;
        for (auto p = positions.rbegin(); p != positions.rend(); ++p) {
            const int s = *p;
            if (s + L > len - 1) continue;  // tail self-occurrence / no continuation
            const int avail = std::min(n_draft, len - s - L);
            result.assign(history_.begin() + s + L,
                          history_.begin() + s + L + avail);
            return result;
        }
    }

    return result;
}

void NgramDraftProvider::accept(const std::vector<int32_t>& tokens) {
    if (tokens.empty()) return;
    const int old_len = static_cast<int>(history_.size());
    history_.insert(history_.end(), tokens.begin(), tokens.end());
    index_range(old_len, static_cast<int>(history_.size()));
}

void NgramDraftProvider::reset() {
    history_.clear();
    index_.clear();
}

// =========================================================================
// verify_draft_tokens -- resample-consistency verification
//
// Two modes, selected from the sampler's live config:
// - Greedy (do_sample=false or temperature<=0): argmax at each position must
//   equal the draft token.
// - Sampling (do_sample=true): the FULL sampling chain (repeat penalty ->
//   temperature -> top-k -> top-p -> multinomial) runs at every position and
//   the draft token is accepted iff it equals the target's own sample. On
//   mismatch the target sample becomes the confirmed continuation. Since
//   every sampled token (match or not) is emitted as output, the output
//   distribution is EXACTLY the target model's -- provably lossless, and
//   identical to plain decode. Mirrors llama.cpp's
//   common_sampler_sample_and_accept_n.
//
// Each row's sample is appended to the sampler token history immediately so
// subsequent rows see the same repeat-penalty context as plain decode would.
// =========================================================================

SpecVerifyResult verify_draft_tokens(
    const float* logits_all, int vocab_size,
    const std::vector<int32_t>& draft_tokens,
    Sampler& sampler, const SpeculativeConfig& config) {
    (void)config;  // p_min is a draft-side knob consumed by model draft providers

    SpecVerifyResult result;
    const int n_draft = static_cast<int>(draft_tokens.size());
    const SamplerConfig& scfg = sampler.config();
    const bool greedy_mode = (!scfg.do_sample || scfg.temperature <= 0.0f);

    auto sample_row = [&](int row) -> int32_t {
        const float* row_logits =
            logits_all + static_cast<size_t>(row) * static_cast<size_t>(vocab_size);
        int32_t tok = greedy_mode ? sampler.sample_greedy_ptr(row_logits, vocab_size)
                                  : sampler.sample_temperature_ptr(row_logits, vocab_size,
                                                                   scfg.temperature);
        sampler.add_token_to_history(tok);
        return tok;
    };

    for (int i = 0; i < n_draft; ++i) {
        // Row i predicts the token at position start_pos+i+1.
        const int32_t tok = sample_row(i);
        if (tok == draft_tokens[i]) {
            result.accepted_tokens.push_back(draft_tokens[i]);
            result.n_accepted++;
        } else {
            // Rejected: the target's own sample takes over from here.
            result.resampled = tok;
            return result;
        }
    }

    // All drafts accepted -> sample one bonus token from the last row.
    if (n_draft > 0) {
        result.bonus = sample_row(n_draft);
    }

    return result;
}

}  // namespace forge
