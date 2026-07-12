#include "nanoinfer/engine.h"
#include "nanoinfer/model.h"
#include "nanoinfer/logger.h"

namespace nanoinfer {

EngineRegistry& EngineRegistry::instance() {
    static EngineRegistry registry;
    return registry;
}

void EngineRegistry::register_engine(const std::string& arch, EngineCreator creator) {
    creators_[arch] = std::move(creator);
}

std::unique_ptr<InferenceEngine> EngineRegistry::create(const std::string& arch,
                                                         Model& model, InferenceContext& ctx) {
    // Try exact match first
    auto it = creators_.find(arch);
    if (it != creators_.end()) return it->second(model, ctx);

    // Fallback: use architecture capability to auto-select engine
    auto& cap_registry = ArchCapabilityRegistry::instance();
    if (cap_registry.has(arch)) {
        auto cap = cap_registry.get(arch);
        // SSM → Qwen35Engine
        if (cap.use_ssm) {
            auto ssm_it = creators_.find("qwen35");
            if (ssm_it != creators_.end()) return ssm_it->second(model, ctx);
        }
        // MLA → DeepSeekEngine
        if (cap.use_mla) {
            auto mla_it = creators_.find("deepseek_v2");
            if (mla_it != creators_.end()) return mla_it->second(model, ctx);
        }
        // GQA (default) → LlamaEngine
        if (cap.use_gqa) {
            auto gqa_it = creators_.find("llama");
            if (gqa_it != creators_.end()) return gqa_it->second(model, ctx);
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

} // namespace nanoinfer
