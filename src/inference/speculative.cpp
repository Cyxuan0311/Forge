#include "forge/speculative.h"

#include <algorithm>
#include <cstring>

#include "forge/logger.h"
#include "forge/sampler.h"

namespace forge {

// =========================================================================
// NgramDraftModel
// =========================================================================

NgramDraftModel::NgramDraftModel(int ngram_n, int ngram_min)
    : ngram_n_(ngram_n), ngram_min_(ngram_min) {}

std::vector<int32_t> NgramDraftModel::draft(const std::vector<int32_t>& prefix, int n_draft) {
    std::vector<int32_t> result;
    if (static_cast<int>(prefix.size()) < ngram_min_ || n_draft <= 0)
        return result;

    // 在 history_ + prefix 中搜索最长后缀匹配
    // 将 prefix 中的新 token 加入临时搜索范围
    const auto& search_space = history_.empty() ? prefix : history_;
    int search_len = static_cast<int>(search_space.size());

    // 从最长 n-gram 开始尝试匹配
    for (int n = std::min(ngram_n_, search_len); n >= ngram_min_; --n) {
        // prefix 的最后 n 个 token 作为 pattern
        const int32_t* pattern = prefix.data() + prefix.size() - n;

        // 在 search_space 中查找 pattern 出现的所有位置
        for (int i = 0; i <= search_len - n - 1; ++i) {
            bool match = true;
            for (int k = 0; k < n; ++k) {
                if (search_space[i + k] != pattern[k]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                // 找到匹配，返回匹配位置之后的 token
                int avail = std::min(n_draft, search_len - i - n);
                for (int j = 0; j < avail; ++j) {
                    result.push_back(search_space[i + n + j]);
                }
                if (!result.empty()) return result;
            }
        }
    }

    return result;
}

void NgramDraftModel::accept(const std::vector<int32_t>& tokens) {
    history_.insert(history_.end(), tokens.begin(), tokens.end());
}

// =========================================================================
// verify_draft_tokens
// =========================================================================

SpeculativeConfig::VerifyResult verify_draft_tokens(
    const float* logits_all, int vocab_size,
    const std::vector<int32_t>& draft_tokens,
    Sampler& sampler, const SpeculativeConfig& config) {

    SpeculativeConfig::VerifyResult result;
    int n_draft = static_cast<int>(draft_tokens.size());

    for (int i = 0; i < n_draft; ++i) {
        // logits_all[i] 是位置 i 的 logits（预测位置 i+1 的 token）
        const float* pos_logits = logits_all + i * vocab_size;

        // 用 target model 的 logits 采样/取 argmax
        auto logits_tensor = std::make_shared<Tensor>(DataType::FP32,
                                                       std::vector<int64_t>{1, vocab_size},
                                                       DeviceType::CPU);
        std::memcpy(logits_tensor->data(), pos_logits, vocab_size * sizeof(float));

        int32_t target_token = sampler.sample_greedy(logits_tensor);

        if (target_token == draft_tokens[i]) {
            // 接受
            result.accepted_tokens.push_back(draft_tokens[i]);
            result.n_accepted++;
        } else {
            // 拒绝：重采样
            result.resampled = target_token;
            return result;
        }
    }

    // 所有 draft tokens 都被接受 → bonus token
    if (n_draft > 0) {
        const float* bonus_logits = logits_all + n_draft * vocab_size;
        auto logits_tensor = std::make_shared<Tensor>(DataType::FP32,
                                                       std::vector<int64_t>{1, vocab_size},
                                                       DeviceType::CPU);
        std::memcpy(logits_tensor->data(), bonus_logits, vocab_size * sizeof(float));
        result.bonus = sampler.sample_greedy(logits_tensor);
    }

    return result;
}

}  // namespace forge
