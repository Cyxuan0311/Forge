#pragma once

#include <string>
#include <functional>
#include <unordered_map>
#include "tensor.h"
#include "types.h"
#include "model_loader.h"
#include "model_config.h"
#include "weight_store.h"
#include "model_weights.h"

namespace forge {

// ============================================================================
// Config Parser Registry
// ============================================================================

using ConfigParseFn = std::function<ModelConfig(ModelLoader& loader, const std::string& arch)>;

class ConfigParserRegistry {
public:
    static ConfigParserRegistry& instance();

    void register_parser(const std::string& arch, ConfigParseFn fn);
    bool has(const std::string& arch) const;
    ModelConfig parse(ModelLoader& loader, const std::string& arch) const;

private:
    ConfigParserRegistry() = default;
    std::unordered_map<std::string, ConfigParseFn> parsers_;
};

struct ConfigParserAutoRegister {
    ConfigParserAutoRegister(const std::string& arch, ConfigParseFn fn);
};

#define FORGE_REGISTER_CONFIG_PARSER_IMPL2(line, arch, fn) \
    static ::forge::ConfigParserAutoRegister _config_parser_reg_##line(arch, fn)

#define FORGE_REGISTER_CONFIG_PARSER_IMPL(line, arch, fn) \
    FORGE_REGISTER_CONFIG_PARSER_IMPL2(line, arch, fn)

#define FORGE_REGISTER_CONFIG_PARSER(arch, fn) \
    FORGE_REGISTER_CONFIG_PARSER_IMPL(__LINE__, arch, fn)

// ============================================================================
// Weight Init Registry
// ============================================================================

struct LayerWeightInitContext {
    const WeightStore& store;
    const ModelConfig& config;
    int layer_idx;
    LayerWeights& lw;
};

using LayerWeightInitFn = std::function<void(LayerWeightInitContext& ctx)>;

class WeightInitRegistry {
public:
    static WeightInitRegistry& instance();

    void register_init(const std::string& arch, LayerWeightInitFn fn);
    bool has(const std::string& arch) const;
    void init_layer(const std::string& arch, LayerWeightInitContext& ctx) const;

private:
    WeightInitRegistry() = default;
    std::unordered_map<std::string, LayerWeightInitFn> inits_;
};

struct WeightInitAutoRegister {
    WeightInitAutoRegister(const std::string& arch, LayerWeightInitFn fn);
};

#define FORGE_REGISTER_WEIGHT_INIT_IMPL2(line, arch, fn) \
    static ::forge::WeightInitAutoRegister _weight_init_reg_##line(arch, fn)

#define FORGE_REGISTER_WEIGHT_INIT_IMPL(line, arch, fn) \
    FORGE_REGISTER_WEIGHT_INIT_IMPL2(line, arch, fn)

#define FORGE_REGISTER_WEIGHT_INIT(arch, fn) \
    FORGE_REGISTER_WEIGHT_INIT_IMPL(__LINE__, arch, fn)

// ============================================================================
// Arch Capability Registry
// ============================================================================

struct ArchCapability {
    bool use_gqa = false;
    bool use_mla = false;
    bool use_ssm = false;
    bool use_mrope = false;
    bool use_neox_rope = false;
    ActivationType ffn_activation = ActivationType::SiLU_GELU;
    NormType norm_type = NormType::RMSNorm;
};

class ArchCapabilityRegistry {
public:
    static ArchCapabilityRegistry& instance();

    void register_capability(const std::string& arch, const ArchCapability& cap);
    bool has(const std::string& arch) const;
    ArchCapability get(const std::string& arch) const;

private:
    ArchCapabilityRegistry() = default;
    std::unordered_map<std::string, ArchCapability> capabilities_;
};

struct ArchCapabilityAutoRegister {
    ArchCapabilityAutoRegister(const std::string& arch, const ArchCapability& cap);
};

#define FORGE_REGISTER_ARCH_CAPABILITY_IMPL2(line, arch, cap) \
    static ::forge::ArchCapabilityAutoRegister _arch_cap_reg_##line(arch, cap)

#define FORGE_REGISTER_ARCH_CAPABILITY_IMPL(line, arch, cap) \
    FORGE_REGISTER_ARCH_CAPABILITY_IMPL2(line, arch, cap)

#define FORGE_REGISTER_ARCH_CAPABILITY(arch, cap) \
    FORGE_REGISTER_ARCH_CAPABILITY_IMPL(__LINE__, arch, cap)

// ============================================================================
// Unified Architecture Registration
// ============================================================================

#define FORGE_REGISTER_ARCH_IMPL2(line, arch, engine_creator, config_fn, weight_init_fn, cap) \
    static ::forge::EngineAutoRegister _engine_reg_##line(arch, engine_creator); \
    static ::forge::ConfigParserAutoRegister _config_parser_reg_##line(arch, config_fn); \
    static ::forge::WeightInitAutoRegister _weight_init_reg_##line(arch, weight_init_fn); \
    static ::forge::ArchCapabilityAutoRegister _arch_cap_reg_##line(arch, cap)

#define FORGE_REGISTER_ARCH_IMPL(line, arch, engine_creator, config_fn, weight_init_fn, cap) \
    FORGE_REGISTER_ARCH_IMPL2(line, arch, engine_creator, config_fn, weight_init_fn, cap)

#define FORGE_REGISTER_ARCH(arch, engine_creator, config_fn, weight_init_fn, cap) \
    FORGE_REGISTER_ARCH_IMPL(__LINE__, arch, engine_creator, config_fn, weight_init_fn, cap)

} // namespace forge
