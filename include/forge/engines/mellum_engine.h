#pragma once

// MellumEngine: llama.cpp arch "mellum" (Mellum2.1 12B-A2.5B)。
//
// 每层: rms_norm → QKV (无 bias) → per-head Q/K RMSNorm → RoPE
//       (SWA 层 plain / 全注意力层 yarn) → attention → wo → 残差
//       → rms_norm(ffn_norm) → 路由-only SiLU MoE → 残差。
// KV cache / SWA ring buffer 用 TransformerEngine 基类的 per-layer 实现。

#include "forge/inference/layers/attention_executor.h"
#include "forge/engines/transformer_engine.h"

namespace forge {

class MellumEngine : public TransformerEngine {
public:
    MellumEngine(Model& model, InferenceContext& ctx);

    std::string name() const override { return "MellumEngine"; }

protected:
    TensorPtr forward_layer(const TensorPtr& hidden, const LayerExecutionContext& lctx) override;
    bool init_weights() override;

private:
    // ggml rope_yarn 的 per-dim 频率 scale 表 + mscale (照抄 xing4 build_yarn_table)。
    void build_yarn_table();
    // yarn pair_scale 表的 CUDA 镜像 (惰性创建)。
    const float* yarn_pair_s_dev_ptr();

    AttentionExecutor attn_;  // KVCache 引用自基类 kv_cache_

    // yarn (host): 全 dim 表 (xing4 兼容) + 每 pair 表 (rope kernel 消费)。
    std::vector<float> yarn_s_;
    std::vector<float> yarn_pair_s_;
    float yarn_mscale_ = 1.0f;
    TensorPtr yarn_pair_s_dev_;
};

}  // namespace forge
