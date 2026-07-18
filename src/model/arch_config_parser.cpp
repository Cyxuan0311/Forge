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
// Architecture-specific config parsers
// ============================================================================

namespace {
static ConfigParserAutoRegister _reg_cfg_llama(
    "llama", [](ModelLoader& loader, const std::string& arch) -> ModelConfig {
        auto cfg = parse_common_gguf_config(loader, arch);
        cfg.use_neox_rope = true;
        return cfg;
    });

static ConfigParserAutoRegister _reg_cfg_mistral(
    "mistral", [](ModelLoader& loader, const std::string& arch) -> ModelConfig {
        auto cfg = parse_common_gguf_config(loader, arch);
        cfg.use_neox_rope = true;
        return cfg;
    });

static ConfigParserAutoRegister _reg_cfg_qwen("qwen",
                                              [](ModelLoader& loader,
                                                 const std::string& arch) -> ModelConfig {
                                                  auto cfg = parse_common_gguf_config(loader, arch);
                                                  cfg.tie_embeddings = true;
                                                  return cfg;
                                              });

static ConfigParserAutoRegister _reg_cfg_qwen2(
    "qwen2", [](ModelLoader& loader, const std::string& arch) -> ModelConfig {
        auto cfg = parse_common_gguf_config(loader, arch);
        cfg.tie_embeddings = true;
        return cfg;
    });

static ConfigParserAutoRegister _reg_cfg_qwen3vl(
    "qwen3vl", [](ModelLoader& loader, const std::string& arch) -> ModelConfig {
        auto cfg = parse_common_gguf_config(loader, arch);
        cfg.tie_embeddings = true;
        cfg.use_mrope = true;
        cfg.use_qk_norm = true;

        // MRoPE: rope_dimension_count defaults to head_dim (same as llama.cpp).
        // The sections array defines how the n_rot/2 pairs are distributed among
        // position counters (t/h/w), but does NOT reduce the number of rotary dims.
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
    });

static ConfigParserAutoRegister _reg_cfg_yi("yi",
                                            [](ModelLoader& loader,
                                               const std::string& arch) -> ModelConfig {
                                                auto cfg = parse_common_gguf_config(loader, arch);
                                                cfg.use_neox_rope = true;
                                                return cfg;
                                            });

static ConfigParserAutoRegister _reg_cfg_deepseek("deepseek",
                                                  [](ModelLoader& loader,
                                                     const std::string& arch) -> ModelConfig {
                                                      return parse_common_gguf_config(loader, arch);
                                                  });

static ConfigParserAutoRegister _reg_cfg_deepseek_v2(
    "deepseek_v2", [](ModelLoader& loader, const std::string& arch) -> ModelConfig {
        auto cfg = parse_common_gguf_config(loader, arch);
        cfg.kv_lora_rank = static_cast<int>(loader.get_metadata_int(arch + ".kv_lora_rank", 0));
        cfg.q_lora_rank = static_cast<int>(loader.get_metadata_int(arch + ".q_lora_rank", 0));
        cfg.use_mla = (cfg.kv_lora_rank > 0);
        cfg.n_routed_experts =
            static_cast<int>(loader.get_metadata_int(arch + ".n_routed_experts", 0));
        cfg.n_shared_experts =
            static_cast<int>(loader.get_metadata_int(arch + ".n_shared_experts", 0));
        cfg.num_expert_per_tok =
            static_cast<int>(loader.get_metadata_int(arch + ".num_expert_per_tok", 0));
        if (cfg.kv_lora_rank > 0) {
            cfg.num_kv_heads = 1;
            cfg.head_dim = cfg.kv_lora_rank;
        }
        return cfg;
    });

static ConfigParserAutoRegister _reg_cfg_deepseek_v3(
    "deepseek_v3", [](ModelLoader& loader, const std::string& arch) -> ModelConfig {
        auto cfg = parse_common_gguf_config(loader, arch);
        cfg.kv_lora_rank = static_cast<int>(loader.get_metadata_int(arch + ".kv_lora_rank", 0));
        cfg.q_lora_rank = static_cast<int>(loader.get_metadata_int(arch + ".q_lora_rank", 0));
        cfg.use_mla = (cfg.kv_lora_rank > 0);
        cfg.n_routed_experts =
            static_cast<int>(loader.get_metadata_int(arch + ".n_routed_experts", 0));
        cfg.n_shared_experts =
            static_cast<int>(loader.get_metadata_int(arch + ".n_shared_experts", 0));
        cfg.num_expert_per_tok =
            static_cast<int>(loader.get_metadata_int(arch + ".num_expert_per_tok", 0));
        if (cfg.kv_lora_rank > 0) {
            cfg.num_kv_heads = 1;
            cfg.head_dim = cfg.kv_lora_rank;
        }
        return cfg;
    });

static ConfigParserAutoRegister _reg_cfg_qwen35(
    "qwen35", [](ModelLoader& loader, const std::string& arch) -> ModelConfig {
        auto cfg = parse_common_gguf_config(loader, arch);
        cfg.tie_embeddings = true;
        cfg.use_ssm = true;
        cfg.ssm_group_count =
            static_cast<int>(loader.get_metadata_int(arch + ".ssm.group_count", 0));
        cfg.ssm_time_step_rank =
            static_cast<int>(loader.get_metadata_int(arch + ".ssm.time_step_rank", 0));
        cfg.ssm_inner_size = static_cast<int>(loader.get_metadata_int(arch + ".ssm.inner_size", 0));
        cfg.ssm_state_size = static_cast<int>(loader.get_metadata_int(arch + ".ssm.state_size", 0));
        cfg.ssm_conv_kernel =
            static_cast<int>(loader.get_metadata_int(arch + ".ssm.conv_kernel", 0));
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
    });

// Phi-3
static ConfigParserAutoRegister _reg_cfg_phi3("phi3",
                                              [](ModelLoader& loader,
                                                 const std::string& arch) -> ModelConfig {
                                                  auto cfg = parse_common_gguf_config(loader, arch);
                                                  cfg.use_neox_rope = true;
                                                  return cfg;
                                              });

// Gemma
static ConfigParserAutoRegister _reg_cfg_gemma(
    "gemma", [](ModelLoader& loader, const std::string& arch) -> ModelConfig {
        auto cfg = parse_common_gguf_config(loader, arch);
        cfg.use_neox_rope = true;
        cfg.tie_embeddings = true;
        cfg.ffn_activation = ActivationType::GeGLU;
        return cfg;
    });

// Gemma2
static ConfigParserAutoRegister _reg_cfg_gemma2(
    "gemma2", [](ModelLoader& loader, const std::string& arch) -> ModelConfig {
        auto cfg = parse_common_gguf_config(loader, arch);
        cfg.use_neox_rope = true;
        cfg.tie_embeddings = true;
        cfg.ffn_activation = ActivationType::GeGLU;
        cfg.f_attn_logit_softcapping =
            static_cast<float>(loader.get_metadata_float(arch + ".attn_logit_softcapping", 50.0f));
        cfg.f_final_logit_softcapping =
            static_cast<float>(loader.get_metadata_float(arch + ".final_logit_softcapping", 30.0f));
        return cfg;
    });

// Falcon
static ConfigParserAutoRegister _reg_cfg_falcon(
    "falcon", [](ModelLoader& loader, const std::string& arch) -> ModelConfig {
        auto cfg = parse_common_gguf_config(loader, arch);
        cfg.norm_type = NormType::LayerNorm;
        cfg.layer_norm_eps = static_cast<float>(
            loader.get_metadata_float(arch + ".attention.layer_norm_epsilon", 1e-5f));
        cfg.ffn_activation = ActivationType::GELU;
        cfg.use_parallel_residual = true;
        return cfg;
    });

// Gemma4
static ConfigParserAutoRegister _reg_cfg_gemma4(
    "gemma4", [](ModelLoader& loader, const std::string& arch) -> ModelConfig {
        auto cfg = parse_common_gguf_config(loader, arch);
        cfg.use_neox_rope = false;  // Gemma4 GGUF already stores weights in half-split format
        cfg.tie_embeddings = true;
        cfg.ffn_activation = ActivationType::GeGLU;
        cfg.use_qk_norm = true;

        // Per-layer embeddings
        cfg.n_embd_per_layer =
            static_cast<int>(loader.get_metadata_int(arch + ".embedding_length_per_layer_input", 0));

        // Expert FFN intermediate dim
        cfg.n_ff_exp =
            static_cast<int>(loader.get_metadata_int(arch + ".expert_feed_forward_length", 0));

        // MoE configuration
        cfg.n_expert =
            static_cast<int>(loader.get_metadata_int(arch + ".expert_count", 0));
        cfg.n_expert_used =
            static_cast<int>(loader.get_metadata_int(arch + ".expert_used_count", 0));

        // Sliding Window Attention
        cfg.n_swa =
            static_cast<int>(loader.get_metadata_int(arch + ".attention.sliding_window", 0));

        // SWA layer pattern (BOOL array, now supported by gguf_model.cpp)
        auto swa_pattern = loader.get_metadata_int_array(
            arch + ".attention.sliding_window_pattern", {});
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
                    auto k_norm_shape = loader.get_tensor_shape(
                        "blk." + std::to_string(i) + ".attn_k_norm.weight");
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
    });

}  // anonymous namespace
}  // namespace forge
