#pragma once

#include <unordered_map>
#include <vector>

#include "forge/engines/transformer_engine.h"
#include "forge/inference/layers/bailingmoe3_recurrent_memory.h"

namespace forge {

// BailingMoE3 (Ling 3.0) 推理引擎。
//
// 混合架构: 每个层是以下三类分支之一 ——
//   1. GDN/KDA 循环注意力层  (head_count_kv[layer] == 0)
//   2. absorbed MLA 层        (head_count_kv[layer] > 0, kv heads = 1)
//   3. FFN: leading_dense_block_count 层之前为稠密 FFN, 之后为 MoE
//      (sigmoid 路由 + 共享专家)
//
// 当前版本为 CPU-first 实现: 自定义 GDN 与 MLA 数学在 CPU 上执行, 权重在构造时
// 整体搬到 CPU。GPU 路径 (CUDA 上的 GDN/MLA 专用 kernel) 是后续工作。
class Bailingmoe3Engine : public TransformerEngine {
public:
    explicit Bailingmoe3Engine(Model& model, InferenceContext& ctx);

    std::string name() const override { return "bailingmoe3"; }

    void reset() override;

    // 本引擎自定义 kernel 仅 CPU, 强制全 CPU 执行。
    void set_gpu_layers(int gpu_layers) override;

    TensorPtr forward_request(const ForwardRequest& req) override;

    Bailingmoe3RecurrentMemory* bailingmoe3_recurrent() { return &recurrent_memory_; }
    const Bailingmoe3RecurrentMemory* bailingmoe3_recurrent() const { return &recurrent_memory_; }

protected:
    TensorPtr forward_layer(const TensorPtr& hidden, const LayerExecutionContext& lctx) override;
    bool init_weights() override;

    void init_kv_cache(const ModelConfig& cfg) override;

private:
    // 注意力分支
    TensorPtr apply_attention(const TensorPtr& normed, const LayerExecutionContext& lctx);
    TensorPtr apply_mla(const TensorPtr& normed, const LayerExecutionContext& lctx);
    TensorPtr apply_gdn(const TensorPtr& normed, const LayerExecutionContext& lctx);

    // FFN 分支
    TensorPtr apply_ffn(const TensorPtr& attn_residual, const LayerExecutionContext& lctx);
    TensorPtr apply_dense_ffn(const TensorPtr& x, const LayerExecutionContext& lctx);
    TensorPtr apply_moe_ffn(const TensorPtr& x, const LayerExecutionContext& lctx);

    Bailingmoe3RecurrentMemory& recurrent_memory_;
    int n_layers_ = 0;

    // MLA latent (kv_a_proj 输出) 历史, 按层、按 seq 缓存, 用于 decode 时重建 K/V。
    std::vector<std::unordered_map<int, TensorPtr>> latent_hist_;
};

}  // namespace forge
