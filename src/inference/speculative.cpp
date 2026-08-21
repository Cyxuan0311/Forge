#include "forge/speculative.h"

#include <algorithm>
#include <cstring>

#include "forge/logger.h"
#include "forge/sampler.h"
#include "forge/tensor.h"

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
// NgramDraftProvider
// =========================================================================

NgramDraftProvider::NgramDraftProvider(int ngram_n, int ngram_min)
    : ngram_n_(ngram_n), ngram_min_(ngram_min) {}

void NgramDraftProvider::begin(const std::vector<int32_t>& prompt) {
    history_ = prompt;
}

std::vector<int32_t> NgramDraftProvider::draft(int n_draft) {
    std::vector<int32_t> result;
    const int search_len = static_cast<int>(history_.size());
    if (search_len < ngram_min_ || n_draft <= 0) return result;

    // Longest-suffix-first matching against the confirmed history itself.
    // A match must leave at least one continuation token available.
    for (int n = std::min(ngram_n_, search_len); n >= ngram_min_; --n) {
        const int32_t* pattern = history_.data() + search_len - n;

        for (int i = 0; i <= search_len - n - 1; ++i) {
            bool match = true;
            for (int k = 0; k < n; ++k) {
                if (history_[i + k] != pattern[k]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                int avail = std::min(n_draft, search_len - i - n);
                for (int j = 0; j < avail; ++j) result.push_back(history_[i + n + j]);
                if (!result.empty()) return result;
            }
        }
    }

    return result;
}

void NgramDraftProvider::accept(const std::vector<int32_t>& tokens) {
    history_.insert(history_.end(), tokens.begin(), tokens.end());
}

void NgramDraftProvider::reset() {
    history_.clear();
}

// =========================================================================
// verify_draft_tokens
// =========================================================================

SpecVerifyResult verify_draft_tokens(
    const float* logits_all, int vocab_size,
    const std::vector<int32_t>& draft_tokens,
    Sampler& sampler, const SpeculativeConfig& config) {
    (void)config;  // Phase 1: branch on do_sample / p_min here

    SpecVerifyResult result;
    const int n_draft = static_cast<int>(draft_tokens.size());

    for (int i = 0; i < n_draft; ++i) {
        // Row i predicts the token at position start_pos+i+1.
        const float* pos_logits = logits_all + static_cast<size_t>(i) * vocab_size;

        auto logits_tensor =
            std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, vocab_size},
                                     DeviceType::CPU);
        std::memcpy(logits_tensor->data(), pos_logits, vocab_size * sizeof(float));

        const int32_t target_token = sampler.sample_greedy(logits_tensor);

        if (target_token == draft_tokens[i]) {
            result.accepted_tokens.push_back(draft_tokens[i]);
            result.n_accepted++;
        } else {
            // Rejected: target's own sample takes over from here.
            result.resampled = target_token;
            return result;
        }
    }

    // All drafts accepted -> sample one bonus token from the last row.
    if (n_draft > 0) {
        const float* bonus_logits =
            logits_all + static_cast<size_t>(n_draft) * vocab_size;
        auto logits_tensor =
            std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, vocab_size},
                                     DeviceType::CPU);
        std::memcpy(logits_tensor->data(), bonus_logits, vocab_size * sizeof(float));
        result.bonus = sampler.sample_greedy(logits_tensor);
    }

    return result;
}

}  // namespace forge
