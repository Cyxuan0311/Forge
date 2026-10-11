#pragma once

#include <unordered_map>
#include <vector>

#include "forge/engines/transformer_engine.h"
#include "forge/inference/layers/xing4_mhc.h"

namespace forge {

// Xing4.0 (China Telecom) inference engine.
//
// Architecture (llama.cpp PR #29012): every trunk layer is a DeepSeek-V3 style
// absorbed MLA attention + sigmoid-routed MoE (4-of-64 + 1 shared expert;
// 2 leading dense layers), wrapped in Manifold-Constrained Hyper-Connections
// (mHC): a 4-way residual stream x4 [nt, 4*hidden] is mixed per layer by
//
//   hc_pre(fn, scale, base)  -> x_in [nt, hidden] + post [nt,4] + comb [nt,16]
//   norm -> body (MLA or FFN)
//   hc_post: x4[dst] = body_out * post[dst] + sum_src x4[src] * comb[src,dst]
//
// hc_pre/hc_post run twice per layer (attn + ffn); the 4 streams are averaged
// before the final norm / lm_head.
//
// Partial offload is supported: layer placement (prefix on CUDA, rest on CPU)
// is driven by the base class layer_devices_; every helper dispatches on
// lctx.device.type. The MLA latent history (c_nope_norm / k_pe_rot / v_all)
// lives on the owning layer's device, so no cross-device state exists.
class Xing4_0Engine : public TransformerEngine {
public:
    explicit Xing4_0Engine(Model& model, InferenceContext& ctx);

    std::string name() const override { return "xing4_0"; }

    void reset() override;

    // MLA latent history (mla_derived_hist_) is engine-managed per sequence,
    // outside the paged KV storage; a prefix hit would resume the suffix with
    // an empty derived-history window. Radix caching is off until state
    // snapshots at prefix boundaries exist.
    bool prefix_cache_supported() const override { return false; }

    TensorPtr forward_request(const ForwardRequest& req) override;
    TensorPtr forward_batch(const InferenceBatch& batch) override;

protected:
    TensorPtr forward_layer(const TensorPtr& hidden, const LayerExecutionContext& lctx) override;
    bool init_weights() override;
    void init_kv_cache(const ModelConfig& cfg) override;

private:
    // 4-way residual stream (mHC): expand/pre/post/mean live in layers/xing4_mhc.
    MhcMixer mhc_;

    // ---- YaRN rope ----
    // Per-element frequency scale table s_i (device copy for the CUDA MLA
    // kernels, host vector for the CPU path) + rotation magnitude mscale.
    // Built once in init_weights(); empty when the model has no YaRN metadata.
    void build_yarn_table();
    // Lazily uploads `yarn_s_` to the device; returns nullptr when disabled.
    const float* yarn_s_dev_ptr();
    std::vector<float> yarn_s_;   // [rope_dim], 1.0 entries when disabled
    TensorPtr yarn_s_dev_;        // device copy (CUDA path)
    float yarn_mscale_ = 1.0f;

    // ---- layer body ----
    TensorPtr apply_mla(const TensorPtr& normed, const LayerExecutionContext& lctx);
    TensorPtr apply_dense_ffn(const TensorPtr& x, const LayerExecutionContext& lctx);
    TensorPtr apply_moe_ffn(const TensorPtr& x, const LayerExecutionContext& lctx);

    int n_layers_ = 0;

    // Per-layer caches. CPU layers keep host FP32 copies; CUDA layers keep
    // device copies (dequantized once) plus tiny HC affine constants uploaded
    // once per layer instead of per step.
    struct LayerCache {
        TensorPtr kb_fp32;       // attn_k_b dequantized, host (CPU layers)
        TensorPtr vb_fp32;       // attn_v_b dequantized, host (CPU layers)
        TensorPtr kb_fp32_dev;   // device copies (CUDA layers)
        TensorPtr vb_fp32_dev;
        // wq_b flattened view [n_head*qk_head_dim, q_lora]: view shares weight
        // storage (no copy, no host round-trip); cached per layer to avoid
        // rebuilding the Tensor every token.
        TensorPtr wq_b_flat;
        TensorPtr hc_scale_a_d;  // [3] device (CUDA layers)
        TensorPtr hc_base_a_d;   // [24] device
        TensorPtr hc_scale_f_d;  // ffn HC constants
        TensorPtr hc_base_f_d;
    };
    std::vector<LayerCache> cache_;

    // MLA derived-state history (per layer, per seq): rows are fixed once
    // written at their absolute position, later tokens only append.
    struct MlaDerived {
        TensorPtr c_nope_norm;  // [total, kv_lora]
        TensorPtr k_pe_rot;     // [total, rope_dim] (rotated at absolute pos j)
        TensorPtr v_all;        // [total, n_head*v_dim]
    };
    std::vector<std::unordered_map<int, MlaDerived>> mla_derived_hist_;

    // MoE grouped CUDA scratch (cross-layer/cross-step reuse; see bailingmoe3).
    struct MoeCudaScratch {
        TensorPtr idx_d, w_d;                   // device [seq*k]
        TensorPtr gate_out, up_out, gated_out;  // device [seq*k, n_ff]
        TensorPtr expert_out;                   // device [seq, hidden]
    };
    MoeCudaScratch moe_;
    TensorPtr moe_idx_h_;  // host routing arrays (all paths)
    TensorPtr moe_w_h_;
};

}  // namespace forge
