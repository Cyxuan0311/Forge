#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "forge/speculative.h"
#include "forge/tensor.h"
#include "forge/sampler.h"

namespace forge {

class InferenceContext;
class Qwen35Engine;

// Single-head DeepSeek-MTP provider backed by the target qwen35 engine.
class MtpDraftProvider final : public IDraftProvider {
public:
    explicit MtpDraftProvider(InferenceContext& ctx, float p_min = 0.0f);

    bool valid() const;
    void begin(const std::vector<int32_t>& prompt) override;
    std::vector<int32_t> draft(int32_t last_token, int n_draft) override;
    void accept(const std::vector<int32_t>& tokens) override;
    void reset() override;
    const char* name() const override { return "mtp"; }

private:
    InferenceContext& ctx_;
    Qwen35Engine* eng_ = nullptr;
    Sampler sampler_;
    float p_min_ = 0.0f;
    int64_t committed_len_ = 0;
    int32_t last_committed_tok_ = -1;
    size_t drafted_tail_ = 0;
    bool draft_active_ = false;
    TensorPtr pending_h_;
    // 本轮草稿链各步的输出隐藏态 h_0..h_{k-1} (h_i = 第 i 步 mtp_step 的输出)。
    // 部分接受时, 已接受前缀对应的链步骤与重放逐位一致 (确定性同输入),
    // accept() 直接复用, 最多补 1 步 —— 省掉整段 O(k) 重放。
    std::vector<TensorPtr> chain_h_;
};

}  // namespace forge
