#include "nanoinfer/model_loader.h"

namespace nanoinfer {

ModelLoaderRegistry& ModelLoaderRegistry::instance() {
    static ModelLoaderRegistry reg;
    return reg;
}

void ModelLoaderRegistry::register_loader(const std::string& format_name, LoaderCreator creator) {
    creators_[format_name] = std::move(creator);
}

ModelLoaderPtr ModelLoaderRegistry::create_loader(const std::string& path) const {
    for (const auto& [name, creator] : creators_) {
        auto loader = creator();
        if (loader && loader->supports_format(path)) {
            return loader;
        }
    }
    return nullptr;
}

ModelLoaderPtr ModelLoaderRegistry::create_loader_by_format(const std::string& format_name) const {
    auto it = creators_.find(format_name);
    if (it != creators_.end()) {
        return it->second();
    }
    return nullptr;
}

std::vector<std::string> ModelLoaderRegistry::supported_formats() const {
    std::vector<std::string> result;
    result.reserve(creators_.size());
    for (const auto& [name, _] : creators_) {
        result.push_back(name);
    }
    return result;
}

LoaderAutoRegister::LoaderAutoRegister(const std::string& format_name, ModelLoaderRegistry::LoaderCreator creator) {
    ModelLoaderRegistry::instance().register_loader(format_name, std::move(creator));
}

} // namespace nanoinfer
