#pragma once

// Spark25Engine: iFlytek Spark-X2.5 (arch spark2_5)。
//
// 层结构 (参照 llama.cpp spark2-5.cpp):
//   normed = rms_norm(inp)
//   Q/K/V = fused qkv 投影; 每层 RoPE 参数不同:
//     SWA 层:   n_rot = 256 (全旋转), theta = 10k
//     全注意力层: n_rot = 64 (部分旋转), theta = 5M
//   attn = GQA(Q, K, V, scale = 1/sqrt(head_dim))   (无 QK/V norm)
//   attn *= sigmoid(attn_gate @ normed)              ← per-head 输出门控,
//                                                      门控输入是归一化后的输入
//   out = wo @ attn; 残差 → ffn_norm → gelu(gate)*up → down → 残差
//   末尾 rms_norm + tied embedding 输出头 (基类 forward_layers 处理)。
//
// SWA (window=512, pattern [1,1,1,0]) 由 KVCache 的 ring buffer 处理
// (SlidingWindow layer policy, 与 Gemma4Engine 相同)。

#include "forge/engines/transformer_engine.h"

namespace forge {

class Spark25Engine : public TransformerEngine {
public:
    Spark25Engine(Model& model, InferenceContext& ctx);

    std::string name() const override { return "Spark25Engine"; }

protected:
    bool init_weights() override;
    void init_kv_cache(const ModelConfig& cfg) override;
    TensorPtr forward_layer(const TensorPtr& hidden, const LayerExecutionContext& lctx) override;
};

}  // namespace forge
