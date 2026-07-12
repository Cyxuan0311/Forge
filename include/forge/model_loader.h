#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensor.h"
#include "types.h"

namespace forge {

class ModelLoader {
public:
    virtual ~ModelLoader() = default;

    virtual bool load(const std::string& path) = 0;
    virtual void close() = 0;

    virtual bool has_tensor(const std::string& name) const = 0;
    virtual TensorPtr get_tensor(const std::string& name,
                                 DeviceType device = DeviceType::CPU) const = 0;

    virtual std::string get_metadata_str(const std::string& key,
                                         const std::string& default_val = "") const = 0;
    virtual int64_t get_metadata_int(const std::string& key, int64_t default_val = 0) const = 0;
    virtual double get_metadata_float(const std::string& key, double default_val = 0.0) const = 0;
    virtual std::vector<int32_t> get_metadata_int_array(
        const std::string& key, const std::vector<int32_t>& default_val = {}) const = 0;

    virtual bool supports_format(const std::string& path) const = 0;
    virtual std::string format_name() const = 0;

    virtual std::vector<std::string> tensor_names() const = 0;
    virtual size_t num_tensors() const = 0;

    virtual std::vector<int64_t> get_tensor_shape(const std::string& name) const = 0;
};

using ModelLoaderPtr = std::unique_ptr<ModelLoader>;

class ModelLoaderRegistry {
public:
    using LoaderCreator = std::function<ModelLoaderPtr()>;

    static ModelLoaderRegistry& instance();

    void register_loader(const std::string& format_name, LoaderCreator creator);
    ModelLoaderPtr create_loader(const std::string& path) const;
    ModelLoaderPtr create_loader_by_format(const std::string& format_name) const;
    std::vector<std::string> supported_formats() const;

private:
    ModelLoaderRegistry() = default;
    std::unordered_map<std::string, LoaderCreator> creators_;
};

struct LoaderAutoRegister {
    LoaderAutoRegister(const std::string& format_name, ModelLoaderRegistry::LoaderCreator creator);
};

#define FORGE_REGISTER_LOADER_IMPL2(line, name, creator) \
    static ::forge::LoaderAutoRegister _loader_reg_##line(name, creator)

#define FORGE_REGISTER_LOADER_IMPL(line, name, creator) \
    FORGE_REGISTER_LOADER_IMPL2(line, name, creator)

#define FORGE_REGISTER_LOADER(name, creator) FORGE_REGISTER_LOADER_IMPL(__LINE__, name, creator)

}  // namespace forge
