#pragma once

#include <memory>
#include <vector>

namespace forge {

class InferenceEngine;
class Sampler;
class KVCache;

// ---- Speculative Decoding 配置 ----

struct SpeculativeConfig {
    int n_draft = 5;            // 每次猜想 token 数
    float p_min = 0.0f;         // 接受阈值（0 = greedy match）
    bool use_ngram = true;      // 启用 n-gram self-speculative
    int ngram_n = 5;            // n-gram 匹配长度
    int ngram_min = 2;          // 最短 n-gram 后缀匹配长度
    bool enabled = false;       // 默认关闭，需显式启用

    // 验证结果
    struct VerifyResult {
        int n_accepted = 0;                      // 接受的 draft token 数
        std::vector<int32_t> accepted_tokens;     // 被接受的 draft tokens
        int32_t resampled = -1;                   // 首个拒绝位置重采样的 token（或 bonus token）
        int32_t bonus = -1;                       // 全部接受时的 bonus token
    };
};

// ---- Draft Model 抽象接口 ----

class DraftModel {
public:
    virtual ~DraftModel() = default;
    // 基于 prefix 生成 n_draft 个猜想 token
    virtual std::vector<int32_t> draft(const std::vector<int32_t>& prefix, int n_draft) = 0;
    // 将已接受的 token 追加到内部状态（如 n-gram 索引）
    virtual void accept(const std::vector<int32_t>& tokens) = 0;
};

// ---- N-gram Self-Speculative Draft Model ----
// 在已生成的 token 序列中做 n-gram 后缀匹配，返回匹配后的续写

class NgramDraftModel : public DraftModel {
public:
    explicit NgramDraftModel(int ngram_n = 5, int ngram_min = 2);

    std::vector<int32_t> draft(const std::vector<int32_t>& prefix, int n_draft) override;
    void accept(const std::vector<int32_t>& tokens) override;

private:
    int ngram_n_;
    int ngram_min_;
    std::vector<int32_t> history_;  // 所有已生成的 token 历史
};

// ---- 验证函数 ----
// 用 target model 的 logits 验证 draft tokens，返回接受数和重采样结果
// logits_all: [n_tokens, vocab_size]，每行对应一个位置的 logits
// draft_tokens: 待验证的 draft token 列表
// start_pos: 第一个 draft token 对应的 KV 位置
// kv: 用于在拒绝时回滚

SpeculativeConfig::VerifyResult verify_draft_tokens(
    const float* logits_all, int vocab_size,
    const std::vector<int32_t>& draft_tokens,
    Sampler& sampler, const SpeculativeConfig& config);

}  // namespace forge
