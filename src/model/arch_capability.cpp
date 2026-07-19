#include "forge/arch_registry.h"

namespace forge {

// ============================================================================
// ArchCapabilityRegistry implementation
// ============================================================================

ArchCapabilityRegistry& ArchCapabilityRegistry::instance() {
    static ArchCapabilityRegistry registry;
    return registry;
}

void ArchCapabilityRegistry::register_capability(const std::string& arch,
                                                 const ArchCapability& cap) {
    capabilities_[arch] = cap;
}

bool ArchCapabilityRegistry::has(const std::string& arch) const {
    return capabilities_.find(arch) != capabilities_.end();
}

ArchCapability ArchCapabilityRegistry::get(const std::string& arch) const {
    auto it = capabilities_.find(arch);
    if (it != capabilities_.end())
        return it->second;
    return ArchCapability{};
}

ArchCapabilityAutoRegister::ArchCapabilityAutoRegister(const std::string& arch,
                                                       const ArchCapability& cap) {
    ArchCapabilityRegistry::instance().register_capability(arch, cap);
}

}  // namespace forge
