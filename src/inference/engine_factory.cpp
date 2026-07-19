#include "forge/engine.h"
#include "forge/logger.h"
#include "forge/model.h"

#include <mutex>

namespace forge {

// Static table of engine capabilities for compatibility checking.
// Keyed by engine registration name. GenericEngine handles all architectures
// through strategy sub-functions, so its entries are comprehensive.
static const std::unordered_map<std::string, EngineCapability>& engine_capabilities() {
    static const std::unordered_map<std::string, EngineCapability> caps = {
        // GenericEngine (registered under many architecture names)
        {"llama", EngineCapability{
            .supported_norm = NormType::RMSNorm,
            .supported_activation = ActivationType::SiLU_GELU,
            .supports_qk_norm = true,
            .supports_embedding_scale = true,
            .supports_post_attention_norm = true,
            .supports_post_ffn_norm = true,
            .supports_mrope = true,
            .supports_neox_rope = true,
        }},
        {"mistral", EngineCapability{
            .supported_norm = NormType::RMSNorm,
            .supported_activation = ActivationType::SiLU_GELU,
            .supports_neox_rope = true,
        }},
        {"qwen", EngineCapability{
            .supported_norm = NormType::RMSNorm,
            .supported_activation = ActivationType::SiLU_GELU,
        }},
        {"qwen2", EngineCapability{
            .supported_norm = NormType::RMSNorm,
            .supported_activation = ActivationType::SiLU_GELU,
        }},
        {"qwen3vl", EngineCapability{
            .supported_norm = NormType::RMSNorm,
            .supported_activation = ActivationType::SiLU_GELU,
            .supports_qk_norm = true,
            .supports_mrope = true,
        }},
        {"yi", EngineCapability{
            .supported_norm = NormType::RMSNorm,
            .supported_activation = ActivationType::SiLU_GELU,
            .supports_neox_rope = true,
        }},
        {"deepseek", EngineCapability{
            .supported_norm = NormType::RMSNorm,
            .supported_activation = ActivationType::SiLU_GELU,
        }},
        // Gemma family (GeGLU + embedding scale)
        {"gemma", EngineCapability{
            .supported_norm = NormType::RMSNorm,
            .supported_activation = ActivationType::GeGLU,
            .supports_embedding_scale = true,
            .supports_neox_rope = true,
        }},
        {"gemma2", EngineCapability{
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
        // Falcon (LayerNorm + bias + parallel residual)
        {"falcon", EngineCapability{
            .supported_norm = NormType::LayerNorm,
            .supports_norm_bias = true,
            .supported_activation = ActivationType::GELU,
            .supports_qkv_bias = true,
            .supports_parallel_residual = true,
        }},
        // Specialized engines
        {"deepseek_v2", EngineCapability{
            .supported_norm = NormType::RMSNorm,
            .supported_activation = ActivationType::SiLU_GELU,
            .supports_neox_rope = true,
        }},
        {"deepseek_v3", EngineCapability{
            .supported_norm = NormType::RMSNorm,
            .supported_activation = ActivationType::SiLU_GELU,
            .supports_neox_rope = true,
        }},
        {"qwen35", EngineCapability{
            .supported_norm = NormType::RMSNorm,
            .supported_activation = ActivationType::SiLU_GELU,
            .supports_mrope = true,
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

// ============================================================================
// Startup compatibility validation
// ============================================================================
// Runs once (via std::call_once) on first EngineRegistry::create() call.
// Verifies that all registered architectures have compatible engine backends.

static void ensure_arch_compatibility() {
    auto& cap_reg = ArchCapabilityRegistry::instance();
    auto& eng_reg = EngineRegistry::instance();
    auto& caps = engine_capabilities();

    std::string errors;

    for (const auto& [arch, cap] : cap_reg.all()) {
        // Check 1: if the architecture has a dedicated engine, verify that the
        // engine's declared capability covers the architecture's requirements.
        if (eng_reg.has(arch)) {
            auto cap_it = caps.find(arch);
            if (cap_it != caps.end()) {
                auto reasons = cap_it->second.check_compatibility(cap);
                if (!reasons.empty()) {
                    errors += "  - Architecture '" + arch +
                              "' has a dedicated engine but EngineCapability mismatch: " +
                              reasons + "\n";
                }
            }
        }

        // Check 2: if the architecture has NO dedicated engine, verify the
        // fallback engine can handle it.
        if (!eng_reg.has(arch)) {
            // Determine fallback engine (same logic as in create())
            std::string fallback;
            if (cap.use_ssm)       fallback = "qwen35";
            else if (cap.use_mla)  fallback = "deepseek_v2";
            else if (cap.use_gqa)  fallback = "llama";

            if (!fallback.empty()) {
                auto cap_it = caps.find(fallback);
                if (cap_it != caps.end()) {
                    auto reasons = cap_it->second.check_compatibility(cap);
                    if (!reasons.empty()) {
                        errors += "  - Architecture '" + arch +
                                  "' will fallback to engine '" + fallback +
                                  "' but: " + reasons + "\n";
                    }
                } else {
                    errors += "  - Architecture '" + arch +
                              "' will fallback to engine '" + fallback +
                              "' but no EngineCapability declared for that engine\n";
                }
            }
        }
    }

    if (!errors.empty()) {
        LOG_ERROR("COMPATIBILITY ERRORS AT STARTUP:");
        throw std::runtime_error(
            "COMPATIBILITY ERROR at startup:\n" + errors +
            "Please register a dedicated engine or update EngineCapability for the affected architectures.");
    }

    LOG_INFO("Architecture compatibility check passed");
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
    // Lazy one-time compatibility check before first engine creation
    static std::once_flag compat_flag;
    std::call_once(compat_flag, ensure_arch_compatibility);

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
        // GQA (default) → GenericEngine (registered as "llama")
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
