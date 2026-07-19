#include "forge/arch_weight_inits.h"
#include "forge/arch_registry.h"
#include "forge/logger.h"
#include "forge/weight_store.h"

namespace forge {

// ============================================================================
// WeightInitRegistry implementation
// ============================================================================

WeightInitRegistry& WeightInitRegistry::instance() {
    static WeightInitRegistry registry;
    return registry;
}

void WeightInitRegistry::register_init(const std::string& arch, LayerWeightInitFn fn) {
    inits_[arch] = std::move(fn);
}

bool WeightInitRegistry::has(const std::string& arch) const {
    return inits_.find(arch) != inits_.end();
}

void WeightInitRegistry::init_layer(const std::string& arch, LayerWeightInitContext& ctx) const {
    auto it = inits_.find(arch);
    if (it != inits_.end()) {
        it->second(ctx);
    }
}

WeightInitAutoRegister::WeightInitAutoRegister(const std::string& arch, LayerWeightInitFn fn) {
    WeightInitRegistry::instance().register_init(arch, std::move(fn));
}

// ============================================================================
// Architecture-specific weight init functions (named, non-static)
// ============================================================================

static void load_if_present(const WeightStore& store, LayerWeights& lw,
                            const std::string& canonical, const std::string& store_name) {
    auto t = store.get(store_name);
    if (t)
        lw.set(canonical, t);
}

void init_gqa_layer_weights(LayerWeightInitContext& ctx) {
    const auto& store = ctx.store;
    auto& lw = ctx.lw;
    std::string base = "layers." + std::to_string(ctx.layer_idx);

    load_if_present(store, lw, "attn_norm", base + ".attn_norm");
    load_if_present(store, lw, "ffn_norm", base + ".ffn_norm");
    load_if_present(store, lw, "w1", base + ".gate_proj");
    load_if_present(store, lw, "w2", base + ".down_proj");
    load_if_present(store, lw, "w3", base + ".up_proj");
    load_if_present(store, lw, "wq", base + ".wq");
    load_if_present(store, lw, "wk", base + ".wk");
    load_if_present(store, lw, "wv", base + ".wv");
    load_if_present(store, lw, "wo", base + ".wo");
    load_if_present(store, lw, "bq", base + ".bq");
    load_if_present(store, lw, "bk", base + ".bk");
    load_if_present(store, lw, "bv", base + ".bv");
    load_if_present(store, lw, "attn_q_norm", base + ".attn_q_norm");
    load_if_present(store, lw, "attn_k_norm", base + ".attn_k_norm");
}

void init_mla_layer_weights(LayerWeightInitContext& ctx) {
    const auto& store = ctx.store;
    auto& lw = ctx.lw;
    std::string base = "layers." + std::to_string(ctx.layer_idx);

    load_if_present(store, lw, "attn_norm", base + ".attn_norm");
    load_if_present(store, lw, "ffn_norm", base + ".ffn_norm");
    load_if_present(store, lw, "w1", base + ".gate_proj");
    load_if_present(store, lw, "w2", base + ".down_proj");
    load_if_present(store, lw, "w3", base + ".up_proj");
    load_if_present(store, lw, "wq_a", base + ".wq_a");
    load_if_present(store, lw, "wq_b", base + ".wq_b");
    load_if_present(store, lw, "kv_a_proj", base + ".kv_a_proj");
    load_if_present(store, lw, "kv_b_proj", base + ".kv_b_proj");
    load_if_present(store, lw, "wo", base + ".wo");
}

void init_qwen35_layer_weights(LayerWeightInitContext& ctx) {
    const auto& store = ctx.store;
    auto& lw = ctx.lw;
    std::string base = "layers." + std::to_string(ctx.layer_idx);

    load_if_present(store, lw, "attn_norm", base + ".attn_norm");
    load_if_present(store, lw, "ffn_norm", base + ".ffn_norm");
    load_if_present(store, lw, "w1", base + ".gate_proj");
    load_if_present(store, lw, "w2", base + ".down_proj");
    load_if_present(store, lw, "w3", base + ".up_proj");

    load_if_present(store, lw, "attn_q", base + ".attn_q");
    load_if_present(store, lw, "attn_k", base + ".attn_k");
    load_if_present(store, lw, "attn_v", base + ".attn_v");
    load_if_present(store, lw, "attn_output", base + ".attn_output");
    load_if_present(store, lw, "attn_q_norm", base + ".attn_q_norm");
    load_if_present(store, lw, "attn_k_norm", base + ".attn_k_norm");
    load_if_present(store, lw, "post_attention_norm", base + ".post_attention_norm");

    load_if_present(store, lw, "attn_qkv", base + ".attn_qkv");
    load_if_present(store, lw, "attn_gate", base + ".attn_gate");
    load_if_present(store, lw, "ssm_conv1d", base + ".ssm_conv1d");
    load_if_present(store, lw, "ssm_dt", base + ".ssm_dt");
    load_if_present(store, lw, "ssm_a", base + ".ssm_a");
    load_if_present(store, lw, "ssm_alpha", base + ".ssm_alpha");
    load_if_present(store, lw, "ssm_beta", base + ".ssm_beta");
    load_if_present(store, lw, "ssm_norm", base + ".ssm_norm");
    load_if_present(store, lw, "ssm_out", base + ".ssm_out");
}

void init_falcon_layer_weights(LayerWeightInitContext& ctx) {
    const auto& store = ctx.store;
    auto& lw = ctx.lw;
    std::string base = "layers." + std::to_string(ctx.layer_idx);

    load_if_present(store, lw, "attn_norm", base + ".attn_norm");
    load_if_present(store, lw, "ffn_norm", base + ".ffn_norm");
    load_if_present(store, lw, "ffn_norm_bias", base + ".ffn_norm_bias");
    load_if_present(store, lw, "wq", base + ".wq");
    load_if_present(store, lw, "wk", base + ".wk");
    load_if_present(store, lw, "wv", base + ".wv");
    load_if_present(store, lw, "wo", base + ".wo");
    // Falcon FFN: up + down only (no gate/w1)
    load_if_present(store, lw, "w3", base + ".up_proj");
    load_if_present(store, lw, "w2", base + ".down_proj");
    // Optional attn_norm_2 (Falcon-40B)
    load_if_present(store, lw, "attn_norm_2", base + ".attn_norm_2");
    load_if_present(store, lw, "attn_norm_2_bias", base + ".attn_norm_2_bias");
}

void init_gemma4_layer_weights(LayerWeightInitContext& ctx) {
    const auto& store = ctx.store;
    auto& lw = ctx.lw;
    std::string base = "layers." + std::to_string(ctx.layer_idx);

    // Attention weights
    load_if_present(store, lw, "attn_norm", base + ".attn_norm");
    load_if_present(store, lw, "wq", base + ".wq");
    load_if_present(store, lw, "wk", base + ".wk");
    load_if_present(store, lw, "wv", base + ".wv");
    load_if_present(store, lw, "wo", base + ".wo");
    load_if_present(store, lw, "attn_q_norm", base + ".attn_q_norm");
    load_if_present(store, lw, "attn_k_norm", base + ".attn_k_norm");
    load_if_present(store, lw, "attn_post_norm", base + ".attn_post_norm");

    // FFN weights (shared expert / dense FFN)
    load_if_present(store, lw, "ffn_norm", base + ".ffn_norm");
    load_if_present(store, lw, "w1", base + ".gate_proj");
    load_if_present(store, lw, "w2", base + ".down_proj");
    load_if_present(store, lw, "w3", base + ".up_proj");
    load_if_present(store, lw, "post_ffn_norm", base + ".ffn_post_norm");

    // MoE weights (optional - only for MoE layers)
    load_if_present(store, lw, "ffn_gate_inp", base + ".ffn_gate_inp");
    load_if_present(store, lw, "ffn_gate_inp_s", base + ".ffn_gate_inp_s");
    load_if_present(store, lw, "ffn_gate_exps", base + ".ffn_gate_exps");
    load_if_present(store, lw, "ffn_up_exps", base + ".ffn_up_exps");
    load_if_present(store, lw, "ffn_down_exps", base + ".ffn_down_exps");
    load_if_present(store, lw, "ffn_gate_up_exps", base + ".ffn_gate_up_exps");
    load_if_present(store, lw, "ffn_pre_norm_2", base + ".ffn_pre_norm_2");
    load_if_present(store, lw, "ffn_post_norm_1", base + ".ffn_post_norm_1");
    load_if_present(store, lw, "ffn_post_norm_2", base + ".ffn_post_norm_2");

    // Per-layer embeddings (optional)
    load_if_present(store, lw, "per_layer_inp_gate", base + ".per_layer_inp_gate");
    load_if_present(store, lw, "per_layer_proj", base + ".per_layer_proj");
    load_if_present(store, lw, "per_layer_post_norm", base + ".per_layer_post_norm");

    // Layer output scale (optional)
    load_if_present(store, lw, "layer_out_scale", base + ".layer_out_scale");

    // Proportional RoPE frequency factors (full-attention layers only, optional)
    load_if_present(store, lw, "rope_freqs", base + ".rope_freqs");
}

}  // namespace forge
