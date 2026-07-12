#include "forge/engines/layer_graph_builder.h"
#include "forge/logger.h"

namespace forge {

GraphBuilderRegistry& GraphBuilderRegistry::instance() {
    static GraphBuilderRegistry reg;
    return reg;
}

void GraphBuilderRegistry::register_builder(const std::string& arch, BuilderCreator creator) {
    creators_[arch] = std::move(creator);
    LOG_INFO("Registered graph builder for arch: " + arch);
}

std::unique_ptr<LayerGraphBuilder> GraphBuilderRegistry::create(const std::string& arch) const {
    auto it = creators_.find(arch);
    if (it == creators_.end()) return nullptr;
    return it->second();
}

std::vector<std::string> GraphBuilderRegistry::registered_archs() const {
    std::vector<std::string> result;
    for (const auto& [arch, _] : creators_) {
        result.push_back(arch);
    }
    return result;
}

GraphBuilderAutoRegister::GraphBuilderAutoRegister(const std::string& arch,
                                                    GraphBuilderRegistry::BuilderCreator creator) {
    GraphBuilderRegistry::instance().register_builder(arch, std::move(creator));
}

} // namespace forge
