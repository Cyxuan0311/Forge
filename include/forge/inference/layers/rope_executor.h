#pragma once

// RopeExecutor: RoPE 策略。
//
// 从 GenericEngine::rope_forward 抽出。唯一的持久状态是 Proportional RoPE 的
// frequency factors (rope_freqs), 由 engine 在 init_weights() 之后注入。

#include "forge/inference/layer_execution_context.h"
#include "forge/model.h"

namespace forge {

class RopeExecutor {
public:
    struct Result {
        TensorPtr q_rope;
        TensorPtr k_rope;
    };

    // Proportional RoPE 频率因子。未设置时 Proportional 退化为标准 NeoX 频率。
    void set_rope_freqs(TensorPtr freqs) { rope_freqs_ = std::move(freqs); }
    const TensorPtr& rope_freqs() const { return rope_freqs_; }

    Result apply(const TensorPtr& q, const TensorPtr& k, const ModelConfig& cfg, int64_t start_pos,
                 int seq_len, DeviceType dev);

    // 标准 (split-half) RoPE, 不依赖 ModelConfig.rope_type。
    // DeepSeek 的 GQA 与 MLA 层直接按此语义计算, 因此单独暴露。
    static Result apply_standard(const TensorPtr& q, const TensorPtr& k, int num_heads,
                                 int num_kv_heads, int head_dim, int seq_len, int64_t start_pos,
                                 float theta, DeviceType dev);

private:
    // 返回 CPU 上的 freq_factors 指针, 需要时做一次 device->host 拷贝并缓存。
    const float* freq_factors_cpu();

    TensorPtr rope_freqs_;
    TensorPtr rope_freqs_cpu_;
};

}  // namespace forge
