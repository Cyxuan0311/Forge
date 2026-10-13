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
// 自定义 GDN/MLA orchestration 仍在 CPU 上执行；矩阵乘法可按 layer device
// 使用 CUDA 权重和 GEMV，结果在需要 CPU recurrence/attention 时显式回传。
class Bailingmoe3Engine : public TransformerEngine {
public:
    explicit Bailingmoe3Engine(Model& model, InferenceContext& ctx);

    std::string name() const override { return "bailingmoe3"; }

    void reset() override;

    // GDN/KDA 层状态在 Bailingmoe3RecurrentMemory, MLA latent 历史由引擎自管,
    // 都不在 paged KV 里; prefix hit 会以零状态续算后缀。禁用 radix 前缀缓存。
    bool prefix_cache_supported() const override { return false; }

    // gpu_layers == 0: all CPU. gpu_layers > 0: all CUDA (rejected until the
    // GDN/MLA CUDA path lands). Partial offload is intentionally rejected
    // because the recurrent/MLA paths have whole-model device state and cannot
    // safely mix CPU pointers with CUDA weights.
    void set_gpu_layers(int gpu_layers) override;
    // bailingmoe3 currently cannot offload, so uses_gpu() is false even when
    // -ngl -1 was requested (the base would have reported "on GPU").
    bool uses_gpu() const override { return cuda_mode_; }

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

    // CUDA 后端分支 (设备感知)。输入/输出均为 CUDA 张量, 状态缓冲由
    // recurrent_memory_ 的设备后端持有。仅在 USE_CUDA 构建中可用; CPU-only
    // 构建走 apply_mla/apply_gdn, 故此处声明一并守卫, 避免链接器为
    // 未编译的 cuda::* 符号报 undefined reference。
#ifdef USE_CUDA
    TensorPtr apply_mla_cuda(const TensorPtr& normed, const LayerExecutionContext& lctx);
    TensorPtr apply_gdn_cuda(const TensorPtr& normed, const LayerExecutionContext& lctx);
#endif

    // FFN 分支
    TensorPtr apply_ffn(const TensorPtr& attn_residual, const LayerExecutionContext& lctx);
    TensorPtr apply_dense_ffn(const TensorPtr& x, const LayerExecutionContext& lctx);
    TensorPtr apply_moe_ffn(const TensorPtr& x, const LayerExecutionContext& lctx);

    Bailingmoe3RecurrentMemory& recurrent_memory_;
    int n_layers_ = 0;
    bool cuda_mode_ = false;

    struct MlaCache {
        TensorPtr kb_fp32;
        TensorPtr vb_fp32;
        TensorPtr kb_fp32_dev;  // CUDA 后端复用的设备副本
        TensorPtr vb_fp32_dev;
        // wq_b 的展平视图 [n_head*qk_head_dim, q_lora]: 视图共享权重存储
        // (无拷贝、无 host 往返), 按层缓存以避免每 token 重建 Tensor。
        TensorPtr wq_b_flat;
    };
    std::vector<MlaCache> mla_cache_;

    // GDN 层逐层缓存: 卷积权重与极小的 bias/a/nw 常量全部驻留设备端,
    // init 时一次性上传。若逐次 H2D (cudaMemcpy), 每步 18 层 × 3 次阻塞拷贝
    // 会把默认流流水线整条排空。
    struct GdnCache {
        TensorPtr conv_w_q;   // 设备端 [d_inner, d_conv]
        TensorPtr conv_w_k;
        TensorPtr conv_w_v;
        TensorPtr bias_d;     // 设备端 [d_inner]
        TensorPtr a_d;        // 设备端 [num_heads]
        TensorPtr nw_d;       // 设备端 [head_dim]
        bool init = false;
    };
    std::vector<GdnCache> gdn_cache_;

    // MoE grouped CUDA scratch (按需扩容, 跨层/跨步复用)。分组 top-k 路由是
    // host 控制流: 路由结果先写入 host 数组再整块上传设备; gate/up/gated
    // [seq*k, n_ff] 与 expert_out [seq, hidden] 全程驻留设备, 消除逐专家的
    // D2D 拷贝与临时张量分配 (每层 launch 数 ~55 -> ~5)。
    struct MoeCudaScratch {
        TensorPtr idx_h, w_h;                   // host   [seq*k]  int32 / fp32
        TensorPtr idx_d, w_d;                   // device [seq*k]
        TensorPtr gate_out, up_out, gated_out;  // device [seq*k, n_ff]
        TensorPtr expert_out;                   // device [seq, hidden]
    };
    MoeCudaScratch moe_;

    // MLA 派生状态历史 (按层、按 seq 缓存): c_nope_norm / k_pe_rot / v_all 在绝对
    // 位置入库后即固定, 后续 token 仅计算 cur_n 个新行并追加, 实现增量 O(1) ——
    // 消除逐 token 全量重算 RoPE 与收缩值 (见 bailingmoe3_mla_attention)。
    struct MlaDerived {
        TensorPtr c_nope_norm;  // [total, kv_lora]
        TensorPtr k_pe_rot;     // [total, rope_dim]  (已按绝对位置 j 旋转)
        TensorPtr v_all;        // [total, n_head*v_dim]
    };
    std::vector<std::unordered_map<int, MlaDerived>> mla_derived_hist_;
};

}  // namespace forge
