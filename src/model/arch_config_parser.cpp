#include "forge/arch_registry.h"
#include "forge/model_loader.h"
#include "forge/logger.h"

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
    if (it != parsers_.end()) return it->second(loader, arch);
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
    cfg.intermediate_dim = static_cast<int>(loader.get_metadata_int(arch + ".feed_forward_length", 11008));
    cfg.num_layers = static_cast<int>(loader.get_metadata_int(arch + ".block_count", 32));
    cfg.num_heads = static_cast<int>(loader.get_metadata_int(arch + ".attention.head_count", 32));
    cfg.num_kv_heads = static_cast<int>(loader.get_metadata_int(arch + ".attention.head_count_kv", cfg.num_heads));
    cfg.head_dim = static_cast<int>(loader.get_metadata_int(arch + ".attention.key_length", 0));
    if (cfg.head_dim == 0) cfg.head_dim = cfg.hidden_dim / cfg.num_heads;
    cfg.rope_theta = static_cast<float>(loader.get_metadata_float(arch + ".rope.freq_base", 10000.0));
    cfg.rms_norm_eps = static_cast<float>(loader.get_metadata_float(arch + ".attention.layer_norm_rms_epsilon", 1e-6));
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
        if (cfg.vocab_size == 0) cfg.vocab_size = 32000;
    }

    return cfg;
}

// ============================================================================
// Architecture-specific config parsers
// ============================================================================

namespace {
static ConfigParserAutoRegister _reg_cfg_llama("llama", [](ModelLoader& loader, const std::string& arch) -> ModelConfig {
    auto cfg = parse_common_gguf_config(loader, arch);
    cfg.use_neox_rope = true;
    return cfg;
});

static ConfigParserAutoRegister _reg_cfg_mistral("mistral", [](ModelLoader& loader, const std::string& arch) -> ModelConfig {
    auto cfg = parse_common_gguf_config(loader, arch);
    cfg.use_neox_rope = true;
    return cfg;
});

static ConfigParserAutoRegister _reg_cfg_qwen("qwen", [](ModelLoader& loader, const std::string& arch) -> ModelConfig {
    auto cfg = parse_common_gguf_config(loader, arch);
    if (cfg.rope_theta == 10000.0f) cfg.rope_theta = 1000000.0f;
    cfg.tie_embeddings = true;
    return cfg;
});

static ConfigParserAutoRegister _reg_cfg_qwen2("qwen2", [](ModelLoader& loader, const std::string& arch) -> ModelConfig {
    auto cfg = parse_common_gguf_config(loader, arch);
    if (cfg.rope_theta == 10000.0f) cfg.rope_theta = 1000000.0f;
    cfg.tie_embeddings = true;
    return cfg;
});

static ConfigParserAutoRegister _reg_cfg_yi("yi", [](ModelLoader& loader, const std::string& arch) -> ModelConfig {
    auto cfg = parse_common_gguf_config(loader, arch);
    cfg.use_neox_rope = true;
    return cfg;
});

static ConfigParserAutoRegister _reg_cfg_deepseek("deepseek", [](ModelLoader& loader, const std::string& arch) -> ModelConfig {
    return parse_common_gguf_config(loader, arch);
});

static ConfigParserAutoRegister _reg_cfg_deepseek_v2("deepseek_v2", [](ModelLoader& loader, const std::string& arch) -> ModelConfig {
    auto cfg = parse_common_gguf_config(loader, arch);
    cfg.kv_lora_rank = static_cast<int>(loader.get_metadata_int(arch + ".kv_lora_rank", 0));
    cfg.q_lora_rank = static_cast<int>(loader.get_metadata_int(arch + ".q_lora_rank", 0));
    cfg.use_mla = (cfg.kv_lora_rank > 0);
    cfg.n_routed_experts = static_cast<int>(loader.get_metadata_int(arch + ".n_routed_experts", 0));
    cfg.n_shared_experts = static_cast<int>(loader.get_metadata_int(arch + ".n_shared_experts", 0));
    cfg.num_expert_per_tok = static_cast<int>(loader.get_metadata_int(arch + ".num_expert_per_tok", 0));
    if (cfg.kv_lora_rank > 0) {
        cfg.num_kv_heads = 1;
        cfg.head_dim = cfg.kv_lora_rank;
    }
    return cfg;
});

static ConfigParserAutoRegister _reg_cfg_deepseek_v3("deepseek_v3", [](ModelLoader& loader, const std::string& arch) -> ModelConfig {
    auto cfg = parse_common_gguf_config(loader, arch);
    cfg.kv_lora_rank = static_cast<int>(loader.get_metadata_int(arch + ".kv_lora_rank", 0));
    cfg.q_lora_rank = static_cast<int>(loader.get_metadata_int(arch + ".q_lora_rank", 0));
    cfg.use_mla = (cfg.kv_lora_rank > 0);
    cfg.n_routed_experts = static_cast<int>(loader.get_metadata_int(arch + ".n_routed_experts", 0));
    cfg.n_shared_experts = static_cast<int>(loader.get_metadata_int(arch + ".n_shared_experts", 0));
    cfg.num_expert_per_tok = static_cast<int>(loader.get_metadata_int(arch + ".num_expert_per_tok", 0));
    if (cfg.kv_lora_rank > 0) {
        cfg.num_kv_heads = 1;
        cfg.head_dim = cfg.kv_lora_rank;
    }
    return cfg;
});

static ConfigParserAutoRegister _reg_cfg_qwen35("qwen35", [](ModelLoader& loader, const std::string& arch) -> ModelConfig {
    auto cfg = parse_common_gguf_config(loader, arch);
    cfg.tie_embeddings = true;
    cfg.use_ssm = true;
    cfg.ssm_group_count = static_cast<int>(loader.get_metadata_int(arch + ".ssm.group_count", 0));
    cfg.ssm_time_step_rank = static_cast<int>(loader.get_metadata_int(arch + ".ssm.time_step_rank", 0));
    cfg.ssm_inner_size = static_cast<int>(loader.get_metadata_int(arch + ".ssm.inner_size", 0));
    cfg.ssm_state_size = static_cast<int>(loader.get_metadata_int(arch + ".ssm.state_size", 0));
    cfg.ssm_conv_kernel = static_cast<int>(loader.get_metadata_int(arch + ".ssm.conv_kernel", 0));
    cfg.full_attention_interval = static_cast<int>(loader.get_metadata_int(arch + ".full_attention_interval", 0));

    cfg.rope_dimension_count = static_cast<int>(loader.get_metadata_int(arch + ".rope.dimension_count", 0));
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
static ConfigParserAutoRegister _reg_cfg_phi3("phi3", [](ModelLoader& loader, const std::string& arch) -> ModelConfig {
    auto cfg = parse_common_gguf_config(loader, arch);
    cfg.use_neox_rope = true;
    return cfg;
});

// Gemma
static ConfigParserAutoRegister _reg_cfg_gemma("gemma", [](ModelLoader& loader, const std::string& arch) -> ModelConfig {
    auto cfg = parse_common_gguf_config(loader, arch);
    cfg.use_neox_rope = true;
    cfg.tie_embeddings = true;
    cfg.ffn_activation = ActivationType::GeGLU;
    return cfg;
});

// Gemma2
static ConfigParserAutoRegister _reg_cfg_gemma2("gemma2", [](ModelLoader& loader, const std::string& arch) -> ModelConfig {
    auto cfg = parse_common_gguf_config(loader, arch);
    cfg.use_neox_rope = true;
    cfg.tie_embeddings = true;
    cfg.ffn_activation = ActivationType::GeGLU;
    cfg.f_attn_logit_softcapping = static_cast<float>(loader.get_metadata_float(arch + ".attn_logit_softcapping", 50.0f));
    cfg.f_final_logit_softcapping = static_cast<float>(loader.get_metadata_float(arch + ".final_logit_softcapping", 30.0f));
    return cfg;
});

// Falcon
static ConfigParserAutoRegister _reg_cfg_falcon("falcon", [](ModelLoader& loader, const std::string& arch) -> ModelConfig {
    auto cfg = parse_common_gguf_config(loader, arch);
    cfg.norm_type = NormType::LayerNorm;
    cfg.layer_norm_eps = static_cast<float>(loader.get_metadata_float(arch + ".attention.layer_norm_epsilon", 1e-5f));
    cfg.ffn_activation = ActivationType::GELU;
    cfg.use_parallel_residual = true;
    return cfg;
});

} // anonymous namespace
} // namespace forge
