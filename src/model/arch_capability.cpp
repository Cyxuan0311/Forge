#include "nanoinfer/arch_registry.h"

namespace nanoinfer {

ArchCapabilityRegistry& ArchCapabilityRegistry::instance() {
    static ArchCapabilityRegistry registry;
    return registry;
}

void ArchCapabilityRegistry::register_capability(const std::string& arch, const ArchCapability& cap) {
    capabilities_[arch] = cap;
}

bool ArchCapabilityRegistry::has(const std::string& arch) const {
    return capabilities_.find(arch) != capabilities_.end();
}

ArchCapability ArchCapabilityRegistry::get(const std::string& arch) const {
    auto it = capabilities_.find(arch);
    if (it != capabilities_.end()) return it->second;
    return ArchCapability{};
}

ArchCapabilityAutoRegister::ArchCapabilityAutoRegister(const std::string& arch, const ArchCapability& cap) {
    ArchCapabilityRegistry::instance().register_capability(arch, cap);
}

namespace {
// GQA architectures
static ArchCapabilityAutoRegister _reg_cap_llama("llama",
    ArchCapability{.use_gqa = true, .use_neox_rope = true});
static ArchCapabilityAutoRegister _reg_cap_mistral("mistral",
    ArchCapability{.use_gqa = true, .use_neox_rope = true});
static ArchCapabilityAutoRegister _reg_cap_qwen("qwen",
    ArchCapability{.use_gqa = true});
static ArchCapabilityAutoRegister _reg_cap_qwen2("qwen2",
    ArchCapability{.use_gqa = true});
static ArchCapabilityAutoRegister _reg_cap_yi("yi",
    ArchCapability{.use_gqa = true, .use_neox_rope = true});
static ArchCapabilityAutoRegister _reg_cap_deepseek("deepseek",
    ArchCapability{.use_gqa = true});

// MLA architectures
static ArchCapabilityAutoRegister _reg_cap_deepseek_v2("deepseek_v2",
    ArchCapability{.use_mla = true});
static ArchCapabilityAutoRegister _reg_cap_deepseek_v3("deepseek_v3",
    ArchCapability{.use_mla = true});

// SSM architectures
static ArchCapabilityAutoRegister _reg_cap_qwen35("qwen35",
    ArchCapability{.use_ssm = true, .use_mrope = true});

// Phi-3
static ArchCapabilityAutoRegister _reg_cap_phi3("phi3",
    ArchCapability{.use_gqa = true, .use_neox_rope = true});

// Gemma/Gemma2
static ArchCapabilityAutoRegister _reg_cap_gemma("gemma",
    ArchCapability{.use_gqa = true, .use_neox_rope = true});
static ArchCapabilityAutoRegister _reg_cap_gemma2("gemma2",
    ArchCapability{.use_gqa = true, .use_neox_rope = true});

// Falcon
static ArchCapabilityAutoRegister _reg_cap_falcon("falcon",
    ArchCapability{.use_gqa = true, .use_neox_rope = true});
} // anonymous namespace

} // namespace nanoinfer
