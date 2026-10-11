#include <cmath>

#include "forge/arch_config_parsers.h"
#include "forge/arch_registry.h"
#include "forge/logger.h"
#include "forge/model_loader.h"

namespace forge {

// ============================================================================
// ConfigParserRegistry implementation
// ============================================================================

ConfigParserRegistry& ConfigParserRegistry::instance() {
    static ConfigParserRegistry registry;
    return registry;
}

void ConfigParserRegistry::register_parser(const std::string& arch, ConfigParseFn fn) {
    parsers_[arch] = std::move(fn);
}

bool ConfigParserRegistry::has(const std::string& arch) const {
    return parsers_.find(arch) != parsers_.end();
}

ModelConfig ConfigParserRegistry::parse(ModelLoader& loader, const std::string& arch) const {
    auto it = parsers_.find(arch);
    if (it != parsers_.end())
        return it->second(loader, arch);
    return ModelConfig{};
}

ConfigParserAutoRegister::ConfigParserAutoRegister(const std::string& arch, ConfigParseFn fn) {
    ConfigParserRegistry::instance().register_parser(arch, std::move(fn));
}

// ============================================================================
// Default config parsing: common fields shared by all architectures
// ============================================================================

static ModelConfig parse_common_gguf_config(ModelLoader& loader, const std::string& arch) {
    ModelConfig cfg;
    cfg.vocab_size = static_cast<int>(loader.get_metadata_int(arch + ".vocab_size", 0));
    cfg.hidden_dim = static_cast<int>(loader.get_metadata_int(arch + ".embedding_length", 4096));
    cfg.intermediate_dim =
        static_cast<int>(loader.get_metadata_int(arch + ".feed_forward_length", 11008));
    cfg.num_layers = static_cast<int>(loader.get_metadata_int(arch + ".block_count", 32));
    cfg.num_heads = static_cast<int>(loader.get_metadata_int(arch + ".attention.head_count", 32));
    cfg.num_kv_heads =
        static_cast<int>(loader.get_metadata_int(arch + ".attention.head_count_kv", cfg.num_heads));
    cfg.head_dim = static_cast<int>(loader.get_metadata_int(arch + ".attention.key_length", 0));
    if (cfg.head_dim == 0)
        cfg.head_dim = cfg.hidden_dim / cfg.num_heads;
    cfg.rope_theta =
        static_cast<float>(loader.get_metadata_float(arch + ".rope.freq_base", 10000.0));
    cfg.rms_norm_eps = static_cast<float>(
        loader.get_metadata_float(arch + ".attention.layer_norm_rms_epsilon", 1e-6));
    cfg.max_seq_len = static_cast<int>(loader.get_metadata_int(arch + ".context_length", 4096));
    cfg.arch_type = arch;
    cfg.norm_type = NormType::RMSNorm;
    cfg.ffn_activation = ActivationType::SiLU_GELU;
    cfg.use_gqa = (cfg.num_kv_heads != cfg.num_heads);

    if (cfg.vocab_size == 0) {
        auto embd_shape = loader.get_tensor_shape("token_embd.weight");
        if (!embd_shape.empty() && embd_shape.size() >= 2) {
            cfg.vocab_size = static_cast<int>(embd_shape[0]);
        } else if (loader.has_tensor("token_embd.weight")) {
            cfg.vocab_size = 32000;
        }
        if (cfg.vocab_size == 0)
            cfg.vocab_size = 32000;
    }

    return cfg;
}

// ============================================================================
// Architecture-specific config parsers (named functions)
// ============================================================================

ModelConfig parse_llama_config(ModelLoader& loader, const std::string& arch) {
    auto cfg = parse_common_gguf_config(loader, arch);
    cfg.use_neox_rope = true;
    cfg.rope_type = RopeType::NeoX;
    cfg.ffn_type = FFNType::SiLUGated;
    return cfg;
}

// DFlash / DSPark: lightweight standalone drafter for speculative decoding.
// Decoder layers are plain GQA transformer blocks; the target model's token
// embedding and lm_head are shared, and the encoder fuses hidden states taken
// from the target layers listed in <arch>.target_layers.
ModelConfig parse_dflash_config(ModelLoader& loader, const std::string& arch) {
    auto cfg = parse_common_gguf_config(loader, arch);
    cfg.ffn_type = FFNType::SiLUGated;
    cfg.rope_type = RopeType::Standard;  // Qwen-style RoPE (no NeoX interleave)
    cfg.use_qk_norm = true;              // dflash carries attn_q_norm / attn_k_norm

    cfg.target_layers = loader.get_metadata_int_array(arch + ".target_layers", {});
    cfg.target_hidden_size =
        static_cast<int>(loader.get_metadata_int(arch + ".target_hidden_size", 0));
    cfg.parallel_drafting_token_id =
        static_cast<int>(loader.get_metadata_int(arch + ".parallel_drafting_token_id", -1));

    // Encoder input width = concatenated target features, one per extracted
    // layer (mirrors llama.cpp: n_embd_inp_enc = target_layers.size() * n_embd).
    if (!cfg.target_layers.empty()) {
        cfg.n_embd_inp_enc = static_cast<int>(cfg.target_layers.size()) * cfg.hidden_dim;
    } else {
        LOG_WARN("dflash: metadata '" + arch +
                 ".target_layers' missing; draft encoder input cannot be sized");
    }

    // iSWA (interleaved sliding window attention): a BOOL pattern marks which
    // draft layers use a sliding window of size `attention.sliding_window`
    // (the rest use full attention). Forge routes this via the KV-cache ring
    // buffer (see TransformerEngine::init_kv_cache), exactly like Gemma4.
    // Without populating these fields, n_swa stays 0 and every layer runs full
    // attention, diverging from the reference drafter (Qwen3.6-35B-A3B-DFlash:
    // window=4096, pattern=[1,1,1,1,1,0] -> 5 SWA + 1 full layer).
    cfg.n_swa = static_cast<int>(loader.get_metadata_int(arch + ".attention.sliding_window", 0));
    auto swa_pattern =
        loader.get_metadata_int_array(arch + ".attention.sliding_window_pattern", {});
    cfg.swa_layers.resize(cfg.num_layers, 0);
    for (size_t i = 0; i < swa_pattern.size() && i < (size_t)cfg.num_layers; ++i) {
        cfg.swa_layers[i] = static_cast<int>(swa_pattern[i]);
    }

    // ---- DFlash2 metadata (llama.cpp PR #27342) ----
    // All five keys are optional; the model is treated as DFlash2 only when the
    // GGUF actually carries the selector weights (dflash_selector_top_k > 0).
    cfg.dflash_block_size =
        static_cast<int>(loader.get_metadata_int(arch + ".block_size", 0));
    cfg.dflash_conv_kernel_size =
        static_cast<int>(loader.get_metadata_int(arch + ".conv_kernel_size", 0));
    cfg.dflash_conv_group_size =
        static_cast<int>(loader.get_metadata_int(arch + ".conv_group_size", 0));
    cfg.dflash_selector_rank =
        static_cast<int>(loader.get_metadata_int(arch + ".selector_rank", 0));
    cfg.dflash_selector_top_k =
        static_cast<int>(loader.get_metadata_int(arch + ".selector_top_k", 0));
    cfg.sample_from_anchor =
        loader.get_metadata_int(arch + ".sample_from_anchor", 0) != 0;

    // DSpark truncated candidate re-ranking (vLLM dspark_draft_topk / apply_bias_gathered).
    // When unset the checkpoint uses the full-vocab markov bias (topk == 0).
    cfg.dspark_markov_top_k =
        static_cast<int>(loader.get_metadata_int("dspark_draft_topk", 0));
    if (cfg.dspark_markov_top_k == 0)
        cfg.dspark_markov_top_k =
            static_cast<int>(loader.get_metadata_int(arch + ".draft_topk", 0));
    // Confidence-head adaptive draft truncation threshold (forge-side knob; the
    // checkpoint does not carry it, vLLM defaults to a fixed sigmoid + budget).
    cfg.dspark_confidence_threshold = static_cast<float>(
        loader.get_metadata_float(arch + ".confidence_threshold", 0.5));

    // 部分 RoPE (partial rope): Qwen3 系 DFlash 转换器把旋转维度写成
    // rope.dimension_sections (mrope 四段布局, [64,0,0,0] = 仅旋转前 64 维,
    // 第 4 段为不旋转的透传维)。漏解析时 rot_dim 回退 head_dim=128, 位置编码
    // 与训练不符, 直接摧毁 drafter 接受率。
    auto rope_sections =
        loader.get_metadata_int_array(arch + ".rope.dimension_sections", {});
    if (!rope_sections.empty()) {
        for (size_t i = 0; i < rope_sections.size() && i < 4; ++i) {
            cfg.rope_dimension_sections[i] = rope_sections[i];
        }
        if (cfg.rope_dimension_count <= 0)
            cfg.rope_dimension_count = rope_sections[0] + rope_sections[1] + rope_sections[2];
    }
    if (cfg.rope_dimension_count <= 0) {
        cfg.rope_dimension_count =
            static_cast<int>(loader.get_metadata_int(arch + ".rope.dimension_count", 0));
    }

    // Target logits fed to the DFlash2 selector go through the target's output
    // transforms: logit_scale scaling + final_logit_softcapping tanh cap.
    cfg.f_logit_scale =
        static_cast<float>(loader.get_metadata_float(arch + ".logit_scale", 0.0));
    cfg.f_final_logit_softcapping =
        static_cast<float>(loader.get_metadata_float(arch + ".final_logit_softcapping", 0.0));

    return cfg;
}

ModelConfig parse_mistral_config(ModelLoader& loader, const std::string& arch) {
    auto cfg = parse_common_gguf_config(loader, arch);
    cfg.use_neox_rope = true;
    cfg.rope_type = RopeType::NeoX;
    cfg.ffn_type = FFNType::SiLUGated;
    return cfg;
}

ModelConfig parse_qwen_config(ModelLoader& loader, const std::string& arch) {
    auto cfg = parse_common_gguf_config(loader, arch);
    cfg.tie_embeddings = true;
    cfg.ffn_type = FFNType::SiLUGated;
    return cfg;
}

ModelConfig parse_qwen2_config(ModelLoader& loader, const std::string& arch) {
    auto cfg = parse_common_gguf_config(loader, arch);
    cfg.tie_embeddings = true;
    cfg.ffn_type = FFNType::SiLUGated;
    return cfg;
}

ModelConfig parse_qwen3vl_config(ModelLoader& loader, const std::string& arch) {
    auto cfg = parse_common_gguf_config(loader, arch);
    cfg.tie_embeddings = true;
    cfg.use_mrope = true;
    cfg.rope_type = RopeType::MRoPE;
    cfg.use_qk_norm = true;
    cfg.ffn_type = FFNType::SiLUGated;

    cfg.rope_dimension_count =
        static_cast<int>(loader.get_metadata_int(arch + ".rope.dimension_count", 0));
    if (cfg.rope_dimension_count <= 0) {
        cfg.rope_dimension_count = cfg.head_dim;
    }
    auto sections = loader.get_metadata_int_array(arch + ".rope.dimension_sections", {});
    if (!sections.empty()) {
        for (size_t i = 0; i < sections.size() && i < 4; ++i) {
            cfg.rope_dimension_sections[i] = sections[i];
        }
    }
    return cfg;
}

ModelConfig parse_yi_config(ModelLoader& loader, const std::string& arch) {
    auto cfg = parse_common_gguf_config(loader, arch);
    cfg.use_neox_rope = true;
    cfg.rope_type = RopeType::NeoX;
    cfg.ffn_type = FFNType::SiLUGated;
    return cfg;
}

ModelConfig parse_deepseek_config(ModelLoader& loader, const std::string& arch) {
    auto cfg = parse_common_gguf_config(loader, arch);
    cfg.ffn_type = FFNType::SiLUGated;
    return cfg;
}

ModelConfig parse_deepseek_v2_config(ModelLoader& loader, const std::string& arch) {
    auto cfg = parse_common_gguf_config(loader, arch);
    cfg.kv_lora_rank = static_cast<int>(loader.get_metadata_int(arch + ".kv_lora_rank", 0));
    cfg.q_lora_rank = static_cast<int>(loader.get_metadata_int(arch + ".q_lora_rank", 0));
    cfg.use_mla = (cfg.kv_lora_rank > 0);
    cfg.n_routed_experts = static_cast<int>(loader.get_metadata_int(arch + ".n_routed_experts", 0));
    cfg.n_shared_experts = static_cast<int>(loader.get_metadata_int(arch + ".n_shared_experts", 0));
    cfg.num_expert_per_tok =
        static_cast<int>(loader.get_metadata_int(arch + ".num_expert_per_tok", 0));
    if (cfg.kv_lora_rank > 0) {
        cfg.num_kv_heads = 1;
        cfg.head_dim = cfg.kv_lora_rank;
    }
    return cfg;
}

ModelConfig parse_deepseek_v3_config(ModelLoader& loader, const std::string& arch) {
    auto cfg = parse_common_gguf_config(loader, arch);
    cfg.kv_lora_rank = static_cast<int>(loader.get_metadata_int(arch + ".kv_lora_rank", 0));
    cfg.q_lora_rank = static_cast<int>(loader.get_metadata_int(arch + ".q_lora_rank", 0));
    cfg.use_mla = (cfg.kv_lora_rank > 0);
    cfg.n_routed_experts = static_cast<int>(loader.get_metadata_int(arch + ".n_routed_experts", 0));
    cfg.n_shared_experts = static_cast<int>(loader.get_metadata_int(arch + ".n_shared_experts", 0));
    cfg.num_expert_per_tok =
        static_cast<int>(loader.get_metadata_int(arch + ".num_expert_per_tok", 0));
    if (cfg.kv_lora_rank > 0) {
        cfg.num_kv_heads = 1;
        cfg.head_dim = cfg.kv_lora_rank;
    }
    return cfg;
}

ModelConfig parse_qwen35_config(ModelLoader& loader, const std::string& arch) {
    auto cfg = parse_common_gguf_config(loader, arch);
    cfg.tie_embeddings = true;
    cfg.use_ssm = true;
    // DeepSeek-MTP style trailing nextn block(s): blk.{N}.* carries a full
    // decoder layer plus nextn.{eh_proj,enorm,hnorm,...}. Exclude from trunk.
    cfg.n_nextn_layers =
        static_cast<int>(loader.get_metadata_int(arch + ".nextn_predict_layers", 0));
    if (cfg.n_nextn_layers > 0 && cfg.num_layers > cfg.n_nextn_layers) {
        LOG_INFO("qwen35: " + std::to_string(cfg.n_nextn_layers) +
                 " nextn MTP layer(s) present; trunk layers = " +
                 std::to_string(cfg.num_layers - cfg.n_nextn_layers));
        cfg.num_layers -= cfg.n_nextn_layers;
    } else {
        cfg.n_nextn_layers = 0;
    }
    cfg.ssm_group_count = static_cast<int>(loader.get_metadata_int(arch + ".ssm.group_count", 0));
    cfg.ssm_time_step_rank =
        static_cast<int>(loader.get_metadata_int(arch + ".ssm.time_step_rank", 0));
    cfg.ssm_inner_size = static_cast<int>(loader.get_metadata_int(arch + ".ssm.inner_size", 0));
    cfg.ssm_state_size = static_cast<int>(loader.get_metadata_int(arch + ".ssm.state_size", 0));
    cfg.ssm_conv_kernel = static_cast<int>(loader.get_metadata_int(arch + ".ssm.conv_kernel", 0));
    cfg.full_attention_interval =
        static_cast<int>(loader.get_metadata_int(arch + ".full_attention_interval", 0));

    cfg.rope_dimension_count =
        static_cast<int>(loader.get_metadata_int(arch + ".rope.dimension_count", 0));
    auto sections = loader.get_metadata_int_array(arch + ".rope.dimension_sections", {});
    if (!sections.empty() && cfg.rope_dimension_count > 0) {
        cfg.use_mrope = true;
        for (size_t i = 0; i < sections.size() && i < 4; ++i) {
            cfg.rope_dimension_sections[i] = sections[i];
        }
    }
    return cfg;
}

// ============================================================================
// LFM2 / LFM2.5 (LiquidAI): hybrid shortconv + full attention, tied embeddings.
// Per-layer `attention.head_count_kv`: 0 => shortconv operator layer,
// >0 => GQA full attention (QK-norm, NeoX RoPE). No GDN/SSM layers.
// ============================================================================
ModelConfig parse_lfm2_config(ModelLoader& loader, const std::string& arch) {
    auto cfg = parse_common_gguf_config(loader, arch);
    cfg.tie_embeddings = true;
    // LFM2 GGUF stores Q/K in the HF rotate-half layout (llama.cpp converts it
    // with NEOX rope kernels and no load-time permute), so keep the weights
    // as-is: use_neox_rope=false skips inverse_neox_permute_rows (same reason
    // as phimoe/spark2_5). rope_type=NeoX still selects half-split RoPE pairing.
    cfg.use_neox_rope = false;
    cfg.rope_type = RopeType::NeoX;
    cfg.use_qk_norm = true;

    // Hybrid layer topology from the per-layer head_count_kv array. The scalar
    // read in parse_common_gguf_config sees an array and falls back to
    // num_heads, so re-derive it here.
    auto kv_heads = loader.get_metadata_int_array(arch + ".attention.head_count_kv", {});
    cfg.head_count_kv.assign(kv_heads.begin(), kv_heads.end());
    if (!cfg.head_count_kv.empty()) {
        int max_kv = 0;
        for (int h : cfg.head_count_kv) max_kv = std::max(max_kv, h);
        cfg.num_kv_heads = max_kv > 0 ? max_kv : cfg.num_kv_heads;
    } else {
        // Scalar (or missing) KV heads: uniform model, treat all layers as
        // full attention unless shortconv tensors say otherwise (they do).
        cfg.head_count_kv.assign(cfg.num_layers, cfg.num_kv_heads);
    }
    cfg.use_gqa = (cfg.num_kv_heads != cfg.num_heads);

    cfg.shortconv_l_cache =
        static_cast<int>(loader.get_metadata_int(arch + ".shortconv.l_cache", 0));
    cfg.rope_dimension_count =
        static_cast<int>(loader.get_metadata_int(arch + ".rope.dimension_count", 0));
    return cfg;
}

// ============================================================================
// BailingMoE3 (Ling 3.0): hybrid MoE + MLA + SSM/KDA recurrent attention.
// Per-layer `attention.head_count_kv`: 0 => SSM/KDA recurrent branch,
// >0 => MLA branch (kv heads = 1). Rope is NORM-style (interleaved pairs).
// ============================================================================
ModelConfig parse_bailingmoe3_config(ModelLoader& loader, const std::string& arch) {
    auto cfg = parse_common_gguf_config(loader, arch);
    cfg.tie_embeddings = true;
    cfg.ffn_type = FFNType::MoE;
    cfg.ffn_activation = ActivationType::SiLU_GELU;
    cfg.use_neox_rope = false;
    cfg.rope_type = RopeType::Standard;

    // Hybrid layer topology from the per-layer head_count_kv array.
    auto kv_heads = loader.get_metadata_int_array(arch + ".attention.head_count_kv", {});
    cfg.head_count_kv.assign(kv_heads.begin(), kv_heads.end());
    if (cfg.head_count_kv.empty()) {
        cfg.head_count_kv.assign(cfg.num_layers, 1);  // scalar fallback: all MLA
    }
    cfg.num_kv_heads = 1;  // MLA layers use a single compressed-KV "head"
    cfg.use_mla = true;
    cfg.use_ssm = true;

    // MLA low-rank dims
    cfg.kv_lora_rank = static_cast<int>(loader.get_metadata_int(arch + ".attention.kv_lora_rank", 0));
    cfg.q_lora_rank = static_cast<int>(loader.get_metadata_int(arch + ".attention.q_lora_rank", 0));
    cfg.mla_key_dim = static_cast<int>(loader.get_metadata_int(arch + ".attention.key_length_mla", 0));
    cfg.mla_value_dim = static_cast<int>(loader.get_metadata_int(arch + ".attention.value_length_mla", 0));
    // RoPE dim of the MLA pe part (n_rot in llama.cpp terms)
    cfg.rope_dimension_count =
        static_cast<int>(loader.get_metadata_int(arch + ".rope.dimension_count", 0));

    // SSM / KDA recurrent branch
    cfg.ssm_conv_kernel = static_cast<int>(loader.get_metadata_int(arch + ".ssm.conv_kernel", 4));
    cfg.kda_head_dim = static_cast<int>(loader.get_metadata_int(arch + ".kda.head_dim", 0));
    // kda.safe_gate 在 GGUF 中是 BOOL（加载器归入 int 表），默认 true。
    cfg.kda_safe_gate = static_cast<float>(loader.get_metadata_int(arch + ".kda.safe_gate", 1));
    cfg.kda_gate_lower_bound =
        static_cast<float>(loader.get_metadata_float(arch + ".kda.gate_lower_bound", 0.0f));

    // MoE routing / expert dims
    cfg.n_expert = static_cast<int>(loader.get_metadata_int(arch + ".expert_count", 0));
    cfg.n_expert_used = static_cast<int>(loader.get_metadata_int(arch + ".expert_used_count", 0));
    cfg.moe_group_count =
        static_cast<int>(loader.get_metadata_int(arch + ".expert_group_count", 0));
    cfg.moe_group_used_count =
        static_cast<int>(loader.get_metadata_int(arch + ".expert_group_used_count", 0));
    cfg.moe_gating_func =
        static_cast<int>(loader.get_metadata_int(arch + ".expert_gating_func", 1));
    cfg.moe_weights_scale =
        static_cast<float>(loader.get_metadata_float(arch + ".expert_weights_scale", 0.0f));
    cfg.moe_weights_norm = loader.get_metadata_int(arch + ".expert_weights_norm", 0) != 0;
    // expert_feed_forward_length may be scalar or per-layer array (uniform in practice)
    cfg.expert_feed_forward_length =
        static_cast<int>(loader.get_metadata_int(arch + ".expert_feed_forward_length", 0));
    if (cfg.expert_feed_forward_length == 0) {
        auto arr = loader.get_metadata_int_array(arch + ".expert_feed_forward_length", {});
        if (!arr.empty())
            cfg.expert_feed_forward_length = static_cast<int>(arr[0]);
    }
    cfg.expert_shared_feed_forward_length = static_cast<int>(
        loader.get_metadata_int(arch + ".expert_shared_feed_forward_length", 0));
    if (cfg.expert_shared_feed_forward_length == 0) {
        cfg.expert_shared_feed_forward_length =
            cfg.expert_feed_forward_length * std::max(1, static_cast<int>(
                loader.get_metadata_int(arch + ".expert_shared_count", 1)));
    }
    cfg.expert_shared_count =
        static_cast<int>(loader.get_metadata_int(arch + ".expert_shared_count", 1));
    cfg.leading_dense_block_count =
        static_cast<int>(loader.get_metadata_int(arch + ".leading_dense_block_count", 0));

    // DeepSeek-MTP style trailing NextN block(s), if present
    cfg.n_nextn_layers =
        static_cast<int>(loader.get_metadata_int(arch + ".nextn_predict_layers", 0));
    if (cfg.n_nextn_layers > 0 && cfg.num_layers > cfg.n_nextn_layers) {
        LOG_INFO("bailingmoe3: " + std::to_string(cfg.n_nextn_layers) +
                 " nextn MTP layer(s) present; trunk layers = " +
                 std::to_string(cfg.num_layers - cfg.n_nextn_layers));
        cfg.num_layers -= cfg.n_nextn_layers;
    } else {
        cfg.n_nextn_layers = 0;
    }
    return cfg;
}

// Xing4.0 (China Telecom): DeepSeek-V3 style absorbed MLA + sigmoid-routed MoE,
// wrapped in Manifold-Constrained Hyper-Connections (mHC): a 4-way residual
// stream mixed per layer by hc_{attn,ffn}_{fn,scale,base} tensors (Sinkhorn-
// normalized 4x4 combiner). Mirrors llama.cpp PR #29012 xing4_0.cpp.
ModelConfig parse_xing4_0_config(ModelLoader& loader, const std::string& arch) {
    auto cfg = parse_common_gguf_config(loader, arch);
    cfg.tie_embeddings = false;  // separate output.weight (Q5_K)
    cfg.ffn_type = FFNType::MoE;
    cfg.ffn_activation = ActivationType::SiLU_GELU;
    cfg.use_neox_rope = false;
    cfg.rope_type = RopeType::Standard;  // NORM interleaved rope on the MLA pe part
    cfg.use_mla = true;
    cfg.use_ssm = false;

    // MLA low-rank dims (deepseek2 conventions)
    cfg.kv_lora_rank = static_cast<int>(loader.get_metadata_int(arch + ".attention.kv_lora_rank", 0));
    cfg.q_lora_rank = static_cast<int>(loader.get_metadata_int(arch + ".attention.q_lora_rank", 0));
    cfg.mla_key_dim = static_cast<int>(loader.get_metadata_int(arch + ".attention.key_length_mla", 0));
    cfg.mla_value_dim = static_cast<int>(loader.get_metadata_int(arch + ".attention.value_length_mla", 0));
    cfg.rope_dimension_count =
        static_cast<int>(loader.get_metadata_int(arch + ".rope.dimension_count", 0));

    // MoE routing / expert dims (sigmoid gating + shared expert)
    cfg.n_expert = static_cast<int>(loader.get_metadata_int(arch + ".expert_count", 0));
    cfg.n_expert_used = static_cast<int>(loader.get_metadata_int(arch + ".expert_used_count", 0));
    cfg.moe_group_count = static_cast<int>(loader.get_metadata_int(arch + ".expert_group_count", 0));
    cfg.moe_group_used_count =
        static_cast<int>(loader.get_metadata_int(arch + ".expert_group_used_count", 0));
    cfg.moe_gating_func = static_cast<int>(loader.get_metadata_int(arch + ".expert_gating_func", 2));
    cfg.moe_weights_scale =
        static_cast<float>(loader.get_metadata_float(arch + ".expert_weights_scale", 0.0f));
    cfg.moe_weights_norm = loader.get_metadata_int(arch + ".expert_weights_norm", 0) != 0;
    cfg.expert_feed_forward_length =
        static_cast<int>(loader.get_metadata_int(arch + ".expert_feed_forward_length", 0));
    cfg.expert_shared_count =
        static_cast<int>(loader.get_metadata_int(arch + ".expert_shared_count", 1));
    cfg.expert_shared_feed_forward_length = static_cast<int>(
        loader.get_metadata_int(arch + ".expert_shared_feed_forward_length", 0));
    if (cfg.expert_shared_feed_forward_length == 0)
        cfg.expert_shared_feed_forward_length = cfg.expert_feed_forward_length;
    cfg.leading_dense_block_count =
        static_cast<int>(loader.get_metadata_int(arch + ".leading_dense_block_count", 0));

    // mHC hyper-connections
    cfg.hc_count = static_cast<int>(loader.get_metadata_int(arch + ".hyper_connection.count", 0));
    cfg.hc_sinkhorn_iters = static_cast<int>(
        loader.get_metadata_int(arch + ".hyper_connection.sinkhorn_iterations", 20));
    cfg.hc_epsilon = static_cast<float>(
        loader.get_metadata_float(arch + ".hyper_connection.epsilon", 1e-6f));
    if (cfg.hc_count != 0 && cfg.hc_count != 4)
        throw std::runtime_error("xing4_0: only hyper_connection.count == 4 is supported");

    // nextn_predict_layers = 1 in the metadata but this GGUF carries NO MTP
    // tensors (no blk.40.* / nextn.* weights), so trunk = block_count. Force 0:
    // subtracting would silently drop a real layer.
    cfg.n_nextn_layers = 0;

    // YaRN rope scaling. llama.cpp applies the interpolated frequencies at
    // EVERY position (rope_yarn(): theta = freq_scale*theta_extrap blended by
    // the ext_factor ramp) — it is NOT a no-op below the original context.
    // Parameters follow llama-context.cpp's yarn_attn_factor_adjust.
    const double yarn_factor = loader.get_metadata_float(arch + ".rope.scaling.factor", 0.0);
    if (yarn_factor > 0.0) {
        const float factor = static_cast<float>(yarn_factor);
        cfg.rope_yarn_enabled = true;
        cfg.rope_yarn_freq_scale = 1.0f / factor;
        cfg.rope_yarn_orig_ctx = static_cast<int>(
            loader.get_metadata_int(arch + ".rope.scaling.original_context_length", 0));
        cfg.rope_yarn_beta_fast =
            static_cast<float>(loader.get_metadata_float(arch + ".rope.scaling.yarn_beta_fast", 32.0));
        cfg.rope_yarn_beta_slow =
            static_cast<float>(loader.get_metadata_float(arch + ".rope.scaling.yarn_beta_slow", 1.0));
        // The converter writes mscale_all_dim pre-multiplied by 0.1; llama.cpp
        // cancels that in the xing4_0 loader (rope_yarn_log_mul /= 0.1f).
        cfg.rope_yarn_log_mul = static_cast<float>(
                                    loader.get_metadata_float(arch + ".rope.scaling.yarn_log_multiplier", 0.0)) /
                                0.1f;
        // llama-context.cpp: yarn_ext_factor defaults to 1.0 for LLAMA_ROPE_SCALING_TYPE_YARN
        cfg.rope_yarn_ext_factor = 1.0f;

        auto get_mscale = [](float s, float m) {
            return s <= 1.0f ? 1.0f : (0.1f * m * std::log(s) + 1.0f);
        };
        float attn = 1.0f;
        if (cfg.rope_yarn_ext_factor != 0.0f) {
            if (cfg.rope_yarn_log_mul != 0.0f) {
                // note: llama.cpp assumes mscale == 1.0f unless arch == DEEPSEEK2
                attn = get_mscale(factor, 1.0f) / get_mscale(factor, cfg.rope_yarn_log_mul);
            } else {
                attn = get_mscale(factor, 1.0f);
            }
            // cancel the factor ggml's rope_yarn() re-applies
            attn *= 1.0f / (1.0f + 0.1f * std::log(factor));
        }
        // rope.scaling.attn_factor is absent in this GGUF -> hparams value 1.0
        cfg.rope_yarn_attn_factor = attn;

        LOG_INFO("xing4_0: YaRN enabled (factor " + std::to_string(factor) +
                 ", freq_scale " + std::to_string(cfg.rope_yarn_freq_scale) +
                 ", orig_ctx " + std::to_string(cfg.rope_yarn_orig_ctx) +
                 ", ext_factor " + std::to_string(cfg.rope_yarn_ext_factor) +
                 ", attn_factor " + std::to_string(cfg.rope_yarn_attn_factor) +
                 ", log_mul " + std::to_string(cfg.rope_yarn_log_mul) + ")");
    }
    return cfg;
}

ModelConfig parse_phi3_config(ModelLoader& loader, const std::string& arch) {
    auto cfg = parse_common_gguf_config(loader, arch);
    cfg.use_neox_rope = true;
    cfg.rope_type = RopeType::NeoX;
    cfg.ffn_type = FFNType::SiLUGated;
    return cfg;
}

ModelConfig parse_phimoe_config(ModelLoader& loader, const std::string& arch) {
    auto cfg = parse_common_gguf_config(loader, arch);
    // PhiMoE (Phi-3 based, like LLaMA) stores Q/K weights in standard format and
    // expects half-split RoPE (rotating pairs (d, d+half_dim)) applied directly.
    // Keep rope_type=NeoX so the NeoX kernel (half-split rotation) is used, but
    // set use_neox_rope=false to skip inverse_neox_permute_rows, which would
    // wrongly convert the RoPE convention to interleaved.
    cfg.use_neox_rope = false;
    cfg.rope_type = RopeType::NeoX;
    cfg.ffn_type = FFNType::MoE;
    cfg.ffn_activation = ActivationType::SiLU_GELU;
    cfg.n_expert = static_cast<int>(loader.get_metadata_int(arch + ".expert_count", 0));
    cfg.n_expert_used = static_cast<int>(loader.get_metadata_int(arch + ".expert_used_count", 0));
    cfg.n_ff_exp = static_cast<int>(loader.get_metadata_int(arch + ".feed_forward_length", 0));
    // phimoe uses full attention (sliding_window metadata exists but is NOT used).
    // Do NOT read attention.sliding_window.
    return cfg;
}

ModelConfig parse_gemma_config(ModelLoader& loader, const std::string& arch) {
    auto cfg = parse_common_gguf_config(loader, arch);
    cfg.use_neox_rope = true;
    cfg.rope_type = RopeType::NeoX;
    cfg.tie_embeddings = true;
    cfg.ffn_activation = ActivationType::GeGLU;
    cfg.ffn_type = FFNType::GeGLU;
    cfg.rope_q_scale = 1.0f / std::sqrt(static_cast<float>(cfg.head_dim));
    return cfg;
}

ModelConfig parse_gemma2_config(ModelLoader& loader, const std::string& arch) {
    auto cfg = parse_common_gguf_config(loader, arch);
    cfg.use_neox_rope = true;
    cfg.rope_type = RopeType::NeoX;
    cfg.tie_embeddings = true;
    cfg.ffn_activation = ActivationType::GeGLU;
    cfg.ffn_type = FFNType::GeGLU;
    cfg.rope_q_scale = 1.0f / std::sqrt(static_cast<float>(cfg.head_dim));
    cfg.f_attn_logit_softcapping =
        static_cast<float>(loader.get_metadata_float(arch + ".attn_logit_softcapping", 50.0f));
    cfg.f_final_logit_softcapping =
        static_cast<float>(loader.get_metadata_float(arch + ".final_logit_softcapping", 30.0f));
    cfg.has_post_attention_norm = true;
    cfg.has_post_ffn_norm = true;
    return cfg;
}

ModelConfig parse_falcon_config(ModelLoader& loader, const std::string& arch) {
    auto cfg = parse_common_gguf_config(loader, arch);
    cfg.norm_type = NormType::LayerNorm;
    cfg.layer_norm_eps = static_cast<float>(
        loader.get_metadata_float(arch + ".attention.layer_norm_epsilon", 1e-5f));
    cfg.ffn_activation = ActivationType::GELU;
    cfg.ffn_type = FFNType::SimpleGELU;
    cfg.use_parallel_residual = true;
    return cfg;
}

ModelConfig parse_gemma4_config(ModelLoader& loader, const std::string& arch) {
    auto cfg = parse_common_gguf_config(loader, arch);
    cfg.use_neox_rope = false;           // Gemma4 GGUF already stores weights in half-split format
    cfg.rope_type = RopeType::Standard;  // Half-split format → standard RoPE pairing
    cfg.tie_embeddings = true;
    cfg.ffn_activation = ActivationType::GeGLU;
    cfg.ffn_type = FFNType::MoE;
    cfg.use_qk_norm = true;
    cfg.rope_q_scale = 1.0f / std::sqrt(static_cast<float>(cfg.head_dim));

    // Per-layer embeddings
    cfg.n_embd_per_layer =
        static_cast<int>(loader.get_metadata_int(arch + ".embedding_length_per_layer_input", 0));

    // Expert FFN intermediate dim
    cfg.n_ff_exp =
        static_cast<int>(loader.get_metadata_int(arch + ".expert_feed_forward_length", 0));

    // MoE configuration
    cfg.n_expert = static_cast<int>(loader.get_metadata_int(arch + ".expert_count", 0));
    cfg.n_expert_used = static_cast<int>(loader.get_metadata_int(arch + ".expert_used_count", 0));

    // Sliding Window Attention
    cfg.n_swa = static_cast<int>(loader.get_metadata_int(arch + ".attention.sliding_window", 0));

    // SWA layer pattern (BOOL array, now supported by gguf_model.cpp)
    auto swa_pattern =
        loader.get_metadata_int_array(arch + ".attention.sliding_window_pattern", {});
    cfg.swa_layers.resize(cfg.num_layers, 0);
    for (size_t i = 0; i < swa_pattern.size() && i < (size_t)cfg.num_layers; ++i) {
        cfg.swa_layers[i] = static_cast<int>(swa_pattern[i]);
    }

    // KV shared layers
    int n_kv_shared =
        static_cast<int>(loader.get_metadata_int(arch + ".attention.shared_kv_layers", 0));
    cfg.n_layer_kv_from_start = cfg.num_layers - n_kv_shared;

    // Final logit softcapping
    cfg.f_final_logit_softcapping =
        static_cast<float>(loader.get_metadata_float(arch + ".final_logit_softcapping", 0.0f));

    // SWA-specific dimensions from GGUF metadata
    int key_length_swa =
        static_cast<int>(loader.get_metadata_int(arch + ".attention.key_length_swa", 0));
    int value_length_swa =
        static_cast<int>(loader.get_metadata_int(arch + ".attention.value_length_swa", 0));

    if (key_length_swa > 0) {
        // Use metadata values directly
        cfg.head_dim_swa = key_length_swa;
        cfg.num_kv_heads_swa = cfg.num_kv_heads;  // Same KV head count
    } else if (!cfg.swa_layers.empty()) {
        // Fallback: infer from tensor shapes
        for (int i = 0; i < cfg.num_layers; ++i) {
            if (cfg.swa_layers[i] == 1) {
                auto k_norm_shape =
                    loader.get_tensor_shape("blk." + std::to_string(i) + ".attn_k_norm.weight");
                if (k_norm_shape.size() >= 1 && k_norm_shape[0] > 0) {
                    cfg.head_dim_swa = static_cast<int>(k_norm_shape[0]);
                    cfg.num_kv_heads_swa = cfg.num_kv_heads;
                }
                break;
            }
        }
    }

    // Gemma4 SWA layers use the same number of Q heads as full-attention layers
    if (cfg.num_heads_swa == 0) {
        cfg.num_heads_swa = cfg.num_heads;
    }

    // Fallback: if no SWA layers detected, default to same as full-attention
    if (cfg.head_dim_swa == 0) {
        cfg.head_dim_swa = cfg.head_dim;
        cfg.num_heads_swa = cfg.num_heads;
        cfg.num_kv_heads_swa = cfg.num_kv_heads;
    }

    // SWA RoPE frequency base
    cfg.rope_theta_swa =
        static_cast<float>(loader.get_metadata_float(arch + ".rope.freq_base_swa", cfg.rope_theta));

    // RoPE dimension counts
    cfg.rope_dim_count =
        static_cast<int>(loader.get_metadata_int(arch + ".rope.dimension_count", 0));
    cfg.rope_dim_count_swa =
        static_cast<int>(loader.get_metadata_int(arch + ".rope.dimension_count_swa", 0));

    // Suppress tokens: read from GGUF metadata, fallback to known Gemma4 tokens
    cfg.suppress_tokens = loader.get_metadata_int_array("tokenizer.ggml.suppress_tokens", {});
    if (cfg.suppress_tokens.empty()) {
        // Gemma4 models should suppress <image|> and <audio|> tokens to avoid
        // degenerate repetition loops in generated output
        cfg.suppress_tokens = {258882, 258883};
    }

    return cfg;
}

// ============================================================================
// Mellum (llama.cpp arch "mellum": GQA + per-head Q/K norm + SWA/full 交错
// + 路由-only SiLU MoE + 全注意力层 yarn)。计算图对齐 src/models/mellum.cpp。
// ============================================================================

ModelConfig parse_mellum_config(ModelLoader& loader, const std::string& arch) {
    auto cfg = parse_common_gguf_config(loader, arch);
    cfg.use_neox_rope = false;
    cfg.rope_type = RopeType::Standard;  // GGUF 权重已是 half-split 排布
    cfg.ffn_type = FFNType::MoE;
    cfg.ffn_activation = ActivationType::SiLU_GELU;  // 路由专家用 SiLU(gate)*up
    cfg.use_qk_norm = true;                          // attn_q_norm / attn_k_norm (per-head)

    // MoE 路由 (softmax-all + top-k + topk 归一, 无共享专家)
    cfg.n_expert = static_cast<int>(loader.get_metadata_int(arch + ".expert_count", 0));
    cfg.n_expert_used = static_cast<int>(loader.get_metadata_int(arch + ".expert_used_count", 0));
    cfg.n_ff_exp =
        static_cast<int>(loader.get_metadata_int(arch + ".expert_feed_forward_length", 0));
    // mellum 的 feed_forward_length 是 8 专家激活总宽 (8*896=7168), 每专家宽度以
    // expert_feed_forward_length 为准, 用它修正 activation 估算用的 intermediate_dim。
    if (cfg.n_ff_exp > 0)
        cfg.intermediate_dim = cfg.n_ff_exp;

    // SWA: llama.cpp load_swa_pattern(ml, 4) — 周期 4, i%4!=3 为 SWA 层。
    // GGUF 不带显式 pattern 数组时按该周期生成; 带了则以数组为准。
    cfg.n_swa = static_cast<int>(loader.get_metadata_int(arch + ".attention.sliding_window", 0));
    auto swa_pattern =
        loader.get_metadata_int_array(arch + ".attention.sliding_window_pattern", {});
    cfg.swa_layers.assign(cfg.num_layers, 0);
    if (!swa_pattern.empty()) {
        for (size_t i = 0; i < swa_pattern.size() && i < (size_t)cfg.num_layers; ++i)
            cfg.swa_layers[i] = static_cast<int>(swa_pattern[i]);
    } else if (cfg.n_swa > 0) {
        for (int i = 0; i < cfg.num_layers; ++i)
            cfg.swa_layers[i] = (i % 4 != 3) ? 1 : 0;
    }

    // 每层都有独立 KV (mellum 无 KV 共享), 与 gemma4 的 shared_kv_layers=0 语义一致
    cfg.n_layer_kv_from_start = cfg.num_layers;

    // SWA 层与全注意力层同 head 几何 (key_length_swa 缺省 → 同 head_dim)
    cfg.head_dim_swa = cfg.head_dim;
    cfg.num_heads_swa = cfg.num_heads;
    cfg.num_kv_heads_swa = cfg.num_kv_heads;
    cfg.rope_theta_swa =
        static_cast<float>(loader.get_metadata_float(arch + ".rope.freq_base_swa", cfg.rope_theta));
    cfg.rope_dim_count = cfg.head_dim;
    cfg.rope_dim_count_swa = cfg.head_dim;

    // YaRN 只作用于全注意力层 (i%4==3); SWA 层用 plain rope。字段按 xing4 的
    // build_yarn_table 消费约定填: attn_factor=1.0 (GGUF 的旧式 yarn_attn_factor
    // key llama.cpp 读不到, ctx 侧调整后 net mscale = 1+0.1*ln(factor), 由
    // build_yarn_table 的 mscale 公式复原), log_mul=0。
    const double yarn_factor = loader.get_metadata_float(arch + ".rope.scaling.factor", 0.0);
    if (yarn_factor > 0.0) {
        const float factor = static_cast<float>(yarn_factor);
        cfg.rope_yarn_enabled = true;
        cfg.rope_yarn_freq_scale = 1.0f / factor;
        cfg.rope_yarn_orig_ctx = static_cast<int>(
            loader.get_metadata_int(arch + ".rope.scaling.original_context_length", 0));
        cfg.rope_yarn_beta_fast = static_cast<float>(
            loader.get_metadata_float(arch + ".rope.scaling.yarn_beta_fast", 32.0));
        cfg.rope_yarn_beta_slow = static_cast<float>(
            loader.get_metadata_float(arch + ".rope.scaling.yarn_beta_slow", 1.0));
        cfg.rope_yarn_ext_factor = 1.0f;  // llama.cpp: yarn 默认 ext_factor=1
        cfg.rope_yarn_attn_factor =
            static_cast<float>(loader.get_metadata_float(arch + ".rope.scaling.attn_factor", 1.0f));
        cfg.rope_yarn_log_mul = static_cast<float>(
            loader.get_metadata_float(arch + ".rope.scaling.yarn_log_multiplier", 0.0));
        // build_yarn_table 需要 rope_dimension_count 作为 corr-dims 的 dim
        cfg.rope_dimension_count = cfg.head_dim;
        LOG_INFO("mellum: YaRN enabled (factor " + std::to_string(factor) + ", orig_ctx " +
                 std::to_string(cfg.rope_yarn_orig_ctx) + ") — full-attention layers only");
    }

    LOG_INFO("mellum: MoE n_expert=" + std::to_string(cfg.n_expert) +
             " n_expert_used=" + std::to_string(cfg.n_expert_used) +
             " n_ff_exp=" + std::to_string(cfg.n_ff_exp) + " n_swa=" + std::to_string(cfg.n_swa));

    return cfg;
}

// ============================================================================
// gemma-embedding2 (EmbeddingGemma 2): embedding-only Gemma4 block stack
// ============================================================================
// Gemma4 blocks (QK-norm + SWA/global mix + per-layer embeddings + out_scale)
// with embedding-model differences: bidirectional attention (no causal mask),
// symmetric SWA window, no KV cache (whole prompt attends at once), exact-erf
// GELU everywhere, and a final output projection to n_embd_out (768).

ModelConfig parse_gemma_embedding2_config(ModelLoader& loader, const std::string& arch) {
    auto cfg = parse_common_gguf_config(loader, arch);
    cfg.use_neox_rope = false;           // GGUF already stores weights in half-split format
    cfg.rope_type = RopeType::Standard;  // Half-split format → standard RoPE pairing
    cfg.tie_embeddings = false;          // real output projection to n_embd_out
    cfg.ffn_activation = ActivationType::GeGLU;
    cfg.ffn_type = FFNType::GeGLU;       // exact-erf gelu(gate)*up
    cfg.use_qk_norm = true;
    cfg.non_causal_attention = true;
    cfg.pooling_type =
        static_cast<int>(loader.get_metadata_int(arch + ".pooling_type", 1));

    // Per-layer embeddings (projection only; no per-layer token embd in this arch)
    cfg.n_embd_per_layer =
        static_cast<int>(loader.get_metadata_int(arch + ".embedding_length_per_layer_input", 0));
    cfg.n_embd_out =
        static_cast<int>(loader.get_metadata_int(arch + ".embedding_length_out", 0));

    // Sliding Window Attention (symmetric: mask when |i-j| > n_swa/2)
    cfg.n_swa = static_cast<int>(loader.get_metadata_int(arch + ".attention.sliding_window", 0));

    // SWA layer pattern (BOOL array)
    auto swa_pattern =
        loader.get_metadata_int_array(arch + ".attention.sliding_window_pattern", {});
    cfg.swa_layers.resize(cfg.num_layers, 0);
    for (size_t i = 0; i < swa_pattern.size() && i < (size_t)cfg.num_layers; ++i) {
        cfg.swa_layers[i] = static_cast<int>(swa_pattern[i]);
    }

    // Every layer has its own KV (unused at runtime — no KV cache for embeddings)
    int n_kv_shared =
        static_cast<int>(loader.get_metadata_int(arch + ".attention.shared_kv_layers", 0));
    cfg.n_layer_kv_from_start = cfg.num_layers - n_kv_shared;

    // SWA-specific head geometry
    int key_length_swa =
        static_cast<int>(loader.get_metadata_int(arch + ".attention.key_length_swa", 0));
    if (key_length_swa > 0) {
        cfg.head_dim_swa = key_length_swa;
    } else {
        cfg.head_dim_swa = cfg.head_dim;
    }
    cfg.num_heads_swa = cfg.num_heads;

    // Per-layer KV head counts: attention.head_count_kv is an ARRAY here
    // (e.g. [2,2,2,2,2,1] pattern) — SWA layers carry 2, global layers 1.
    auto kv_heads_arr = loader.get_metadata_int_array(arch + ".attention.head_count_kv", {});
    if (!kv_heads_arr.empty()) {
        int global_kv = -1, swa_kv = -1;
        for (size_t i = 0;
             i < kv_heads_arr.size() && i < (size_t)cfg.num_layers; ++i) {
            if (cfg.swa_layers[i] == 1) {
                if (swa_kv < 0) swa_kv = static_cast<int>(kv_heads_arr[i]);
            } else if (global_kv < 0) {
                global_kv = static_cast<int>(kv_heads_arr[i]);
            }
        }
        if (global_kv > 0) cfg.num_kv_heads = global_kv;
        if (swa_kv > 0) {
            cfg.num_kv_heads_swa = swa_kv;
        } else {
            cfg.num_kv_heads_swa = cfg.num_kv_heads;
        }
    } else {
        cfg.num_kv_heads_swa = cfg.num_kv_heads;
    }
    cfg.use_gqa = (cfg.num_kv_heads != cfg.num_heads) || (cfg.num_kv_heads_swa != cfg.num_heads);

    // Dual RoPE frequency bases (global vs SWA layers)
    cfg.rope_theta_swa =
        static_cast<float>(loader.get_metadata_float(arch + ".rope.freq_base_swa", cfg.rope_theta));

    // RoPE dimension counts (both equal the full head dims here)
    cfg.rope_dim_count =
        static_cast<int>(loader.get_metadata_int(arch + ".rope.dimension_count", 0));
    cfg.rope_dim_count_swa =
        static_cast<int>(loader.get_metadata_int(arch + ".rope.dimension_count_swa", 0));

    return cfg;
}

// ============================================================================
// Spark-X2.5 (iFlytek SparkLLM, arch spark2_5)
// ============================================================================
// Mixed attention (3 SWA + 1 full), dual RoPE (SWA: full 256-dim @ 10k;
// full-attention: partial 64-dim @ 5M), per-head sigmoid output gate,
// GELU-gated FFN, tied embeddings.

ModelConfig parse_spark25_config(ModelLoader& loader, const std::string& arch) {
    auto cfg = parse_common_gguf_config(loader, arch);
    // GGUF 权重已是半拆 (split-half) 布局, 不可再做 neox 权重重排,
    // 否则会把半拆权重翻成交错布局, 与引擎的半拆 RoPE 配对冲突 → Q/K 错乱。
    // (同 Gemma4: "GGUF already stores weights in half-split format")
    cfg.use_neox_rope = false;
    cfg.rope_type = RopeType::NeoX;
    cfg.tie_embeddings = true;
    cfg.ffn_activation = ActivationType::GeGLU;
    cfg.ffn_type = FFNType::GeGLU;       // gelu(gate)*up (exact GELU, not tanh)

    // Sliding window attention
    cfg.n_swa = static_cast<int>(loader.get_metadata_int(arch + ".attention.sliding_window", 0));

    // SWA layer pattern (bool array: 1=SWA, 0=full attention)
    auto swa_pattern =
        loader.get_metadata_int_array(arch + ".attention.sliding_window_pattern", {});
    cfg.swa_layers.resize(cfg.num_layers, 0);
    for (size_t i = 0; i < swa_pattern.size() && i < (size_t)cfg.num_layers; ++i) {
        cfg.swa_layers[i] = static_cast<int>(swa_pattern[i]);
    }
    cfg.n_layer_kv_from_start = cfg.num_layers;  // every layer has its own KV

    // Dual RoPE: full-attention layers use rope.freq_base + rope.dimension_count
    // (partial rotary, 64 of 256 dims); SWA layers use the *_swa variants.
    cfg.rope_theta_swa =
        static_cast<float>(loader.get_metadata_float(arch + ".rope.freq_base_swa", cfg.rope_theta));
    cfg.rope_dim_count =
        static_cast<int>(loader.get_metadata_int(arch + ".rope.dimension_count", 0));
    cfg.rope_dim_count_swa =
        static_cast<int>(loader.get_metadata_int(arch + ".rope.dimension_count_swa", 0));

    // All layers share the same head geometry (attention.key_length = 256)
    cfg.head_dim_swa = cfg.head_dim;
    cfg.num_heads_swa = cfg.num_heads;
    cfg.num_kv_heads_swa = cfg.num_kv_heads;

    return cfg;
}

}  // namespace forge
