#pragma once

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensor.h"

namespace forge {

class Model;
class InferenceContext;

class InferenceEngine {
public:
    virtual ~InferenceEngine() = default;
    virtual TensorPtr forward(const TensorPtr& input_ids, int64_t start_pos) = 0;
    virtual TensorPtr forward_from_hidden(const TensorPtr& hidden, int64_t start_pos) {
        (void)hidden;
        (void)start_pos;
        throw std::runtime_error("forward_from_hidden not implemented for this engine");
    }
    virtual std::string name() const = 0;
    virtual void reset() {}
    virtual void set_gpu_layers(int layers) { (void)layers; }
    virtual int gpu_layers() const { return -1; }
};

using EngineCreator = std::function<std::unique_ptr<InferenceEngine>(Model&, InferenceContext&)>;

class EngineRegistry {
public:
    static EngineRegistry& instance();

    void register_engine(const std::string& arch, EngineCreator creator);
    std::unique_ptr<InferenceEngine> create(const std::string& arch, Model& model,
                                            InferenceContext& ctx);
    std::vector<std::string> registered_archs() const;
    bool has(const std::string& arch) const;

private:
    EngineRegistry() = default;
    std::unordered_map<std::string, EngineCreator> creators_;
};

struct EngineAutoRegister {
    EngineAutoRegister(const std::string& arch, EngineCreator creator);
};

#define FORGE_REGISTER_ENGINE_IMPL2(line, arch, creator) \
    static ::forge::EngineAutoRegister _engine_reg_##line(arch, creator)

#define FORGE_REGISTER_ENGINE_IMPL(line, arch, creator) \
    FORGE_REGISTER_ENGINE_IMPL2(line, arch, creator)

#define FORGE_REGISTER_ENGINE(arch, creator) FORGE_REGISTER_ENGINE_IMPL(__LINE__, arch, creator)

}  // namespace forge
