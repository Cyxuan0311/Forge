#include "forge/engine.h"
#include "forge/logger.h"
#include "forge/model.h"

namespace forge {

// Static table of engine capabilities for fallback compatibility checking.
// Only used when an architecture has no exact engine match and falls back
// to a known engine based on ArchCapability flags.
static const std::unordered_map<std::string, EngineCapability>& engine_capabilities() {
    static const std::unordered_map<std::string, EngineCapability> caps = {
        {"llama", EngineCapability{
            .supported_norm = NormType::RMSNorm,
            .supported_activation = ActivationType::SiLU_GELU,
            .supports_qk_norm = true,
            .supports_mrope = true,
            .supports_neox_rope = true,
        }},
        {"deepseek_v2", EngineCapability{
            .supported_norm = NormType::RMSNorm,
            .supported_activation = ActivationType::SiLU_GELU,
            .supports_neox_rope = true,
        }},
        {"qwen35", EngineCapability{
            .supported_norm = NormType::RMSNorm,
            .supported_activation = ActivationType::SiLU_GELU,
            .supports_mrope = true,
        }},
        {"gemma", EngineCapability{
            .supported_norm = NormType::RMSNorm,
            .supported_activation = ActivationType::GeGLU,
            .supports_embedding_scale = true,
            .supports_post_attention_norm = true,
            .supports_post_ffn_norm = true,
            .supports_neox_rope = true,
        }},
        {"gemma4", EngineCapability{
            .supported_norm = NormType::RMSNorm,
            .supported_activation = ActivationType::GeGLU,
            .supports_qk_norm = true,
            .supports_embedding_scale = true,
            .supports_neox_rope = true,
        }},
        {"falcon", EngineCapability{
            .supported_norm = NormType::LayerNorm,
            .supports_norm_bias = true,
            .supported_activation = ActivationType::SiLU_GELU,
            .supports_qkv_bias = true,
            .supports_parallel_residual = true,
            .supports_neox_rope = true,
        }},
    };
    return caps;
}

// Helper: check fallback compatibility and throw if incompatible
static void check_fallback_compatibility(const std::string& engine_name,
                                          const std::string& arch,
                                          const ArchCapability& cap) {
    auto& caps = engine_capabilities();
    auto cap_it = caps.find(engine_name);
    if (cap_it == caps.end())
        return;  // No capability info registered, skip check
    auto reasons = cap_it->second.check_compatibility(cap);
    if (!reasons.empty()) {
        throw std::runtime_error(
            "Engine '" + engine_name + "' cannot handle architecture '" + arch +
            "': " + reasons +
            "Please register a dedicated engine for this architecture.");
    }
}

EngineRegistry& EngineRegistry::instance() {
    static EngineRegistry registry;
    return registry;
}

void EngineRegistry::register_engine(const std::string& arch, EngineCreator creator) {
    creators_[arch] = std::move(creator);
}

std::unique_ptr<InferenceEngine> EngineRegistry::create(const std::string& arch, Model& model,
                                                        InferenceContext& ctx) {
    // Try exact match first
    auto it = creators_.find(arch);
    if (it != creators_.end())
        return it->second(model, ctx);

    // Fallback: use architecture capability to auto-select engine
    auto& cap_registry = ArchCapabilityRegistry::instance();
    if (cap_registry.has(arch)) {
        auto cap = cap_registry.get(arch);
        // SSM → Qwen35Engine
        if (cap.use_ssm) {
            auto ssm_it = creators_.find("qwen35");
            if (ssm_it != creators_.end()) {
                check_fallback_compatibility("qwen35", arch, cap);
                return ssm_it->second(model, ctx);
            }
        }
        // MLA → DeepSeekEngine
        if (cap.use_mla) {
            auto mla_it = creators_.find("deepseek_v2");
            if (mla_it != creators_.end()) {
                check_fallback_compatibility("deepseek_v2", arch, cap);
                return mla_it->second(model, ctx);
            }
        }
        // GQA (default) → LlamaEngine
        if (cap.use_gqa) {
            auto gqa_it = creators_.find("llama");
            if (gqa_it != creators_.end()) {
                check_fallback_compatibility("llama", arch, cap);
                return gqa_it->second(model, ctx);
            }
        }
    }

    // No matching engine found — return nullptr (caller will raise error)
    return nullptr;
}

std::vector<std::string> EngineRegistry::registered_archs() const {
    std::vector<std::string> result;
    result.reserve(creators_.size());
    for (const auto& [name, _] : creators_) {
        result.push_back(name);
    }
    return result;
}

bool EngineRegistry::has(const std::string& arch) const {
    return creators_.find(arch) != creators_.end();
}

EngineAutoRegister::EngineAutoRegister(const std::string& arch, EngineCreator creator) {
    EngineRegistry::instance().register_engine(arch, std::move(creator));
}

}  // namespace forge
