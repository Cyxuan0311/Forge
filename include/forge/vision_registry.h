#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "forge/vision_encoder.h"

namespace forge {

class ModelLoader;

// ============================================================================
// VisionEncoderRegistry
// ============================================================================
// Maps encoder name (e.g. "siglip") -> creator factory. Callers create encoders
// via create(name) instead of constructing concrete classes directly, so adding
// a new encoder requires only a new registration — no call-site changes.
// ============================================================================
using VisionEncoderCreator = std::function<std::unique_ptr<VisionEncoder>()>;

class VisionEncoderRegistry {
public:
    static VisionEncoderRegistry& instance();

    void register_encoder(const std::string& name, VisionEncoderCreator creator);
    bool has(const std::string& name) const;
    std::unique_ptr<VisionEncoder> create(const std::string& name) const;
    std::vector<std::string> registered_encoders() const;

private:
    VisionEncoderRegistry() = default;
    std::unordered_map<std::string, VisionEncoderCreator> creators_;
};

struct VisionEncoderAutoRegister {
    VisionEncoderAutoRegister(const std::string& name, VisionEncoderCreator creator);
};

#define FORGE_REGISTER_VISION_ENCODER_IMPL2(line, name, creator) \
    static ::forge::VisionEncoderAutoRegister _vision_enc_reg_##line(name, creator)

#define FORGE_REGISTER_VISION_ENCODER_IMPL(line, name, creator) \
    FORGE_REGISTER_VISION_ENCODER_IMPL2(line, name, creator)

#define FORGE_REGISTER_VISION_ENCODER(name, creator) \
    FORGE_REGISTER_VISION_ENCODER_IMPL(__LINE__, name, creator)

// ============================================================================
// VisionConfigParserRegistry
// ============================================================================
// Each vision architecture registers a parser that converts mmproj GGUF
// metadata into a VisionConfig. The parser is selected by architecture name
// (typically derived from the mmproj's "clip.vision.projector_type" or
// "general.architecture" metadata key).
// ============================================================================
using VisionConfigParseFn = std::function<VisionConfig(ModelLoader& loader)>;

class VisionConfigParserRegistry {
public:
    static VisionConfigParserRegistry& instance();

    void register_parser(const std::string& arch, VisionConfigParseFn fn);
    bool has(const std::string& arch) const;
    VisionConfig parse(ModelLoader& loader, const std::string& arch) const;
    std::vector<std::string> registered_parsers() const;

private:
    VisionConfigParserRegistry() = default;
    std::unordered_map<std::string, VisionConfigParseFn> parsers_;
};

struct VisionConfigParserAutoRegister {
    VisionConfigParserAutoRegister(const std::string& arch, VisionConfigParseFn fn);
};

#define FORGE_REGISTER_VISION_CONFIG_PARSER_IMPL2(line, arch, fn) \
    static ::forge::VisionConfigParserAutoRegister _vision_cfg_reg_##line(arch, fn)

#define FORGE_REGISTER_VISION_CONFIG_PARSER_IMPL(line, arch, fn) \
    FORGE_REGISTER_VISION_CONFIG_PARSER_IMPL2(line, arch, fn)

#define FORGE_REGISTER_VISION_CONFIG_PARSER(arch, fn) \
    FORGE_REGISTER_VISION_CONFIG_PARSER_IMPL(__LINE__, arch, fn)

// ============================================================================
// VisionWeightMapping — per-architecture weight name prefixes (P3)
// ============================================================================
// Maps canonical weight roles to GGUF tensor names. Different vision archs use
// different prefixes (e.g. SigLIP uses "v.patch_embd" / "v.blk.{i}", Gemma3
// uses "patch_embd" / "blk.{i}"). VisionWeights::init() looks up the mapping
// by architecture name instead of hardcoding names.
// ============================================================================
struct VisionWeightMapping {
    // Patch embedding
    std::string patch_embd_weight = "v.patch_embd.weight";
    std::string patch_embd_bias = "v.patch_embd.bias";
    // Position embedding
    std::string position_embd_weight = "v.position_embd.weight";
    // Post layer norm
    std::string post_ln_weight = "v.post_ln.weight";
    std::string post_ln_bias = "v.post_ln.bias";
    // ViT block prefix — "{i}" is replaced with the layer index
    std::string block_prefix = "v.blk.{i}";
    // Block sub-field suffixes (appended to block_prefix)
    std::string blk_ln1_w = ".ln1.weight";
    std::string blk_ln1_b = ".ln1.bias";
    std::string blk_ln2_w = ".ln2.weight";
    std::string blk_ln2_b = ".ln2.bias";
    std::string blk_q_w = ".attn_q.weight";
    std::string blk_q_b = ".attn_q.bias";
    std::string blk_k_w = ".attn_k.weight";
    std::string blk_k_b = ".attn_k.bias";
    std::string blk_v_w = ".attn_v.weight";
    std::string blk_v_b = ".attn_v.bias";
    std::string blk_o_w = ".attn_out.weight";
    std::string blk_o_b = ".attn_out.bias";
    std::string blk_ffn_up_w = ".ffn_up.weight";
    std::string blk_ffn_up_b = ".ffn_up.bias";
    std::string blk_ffn_down_w = ".ffn_down.weight";
    std::string blk_ffn_down_b = ".ffn_down.bias";
    // ViT merger prefix
    std::string merger_prefix = "v.vit_merger";
    // Projector prefix
    std::string projector_prefix = "mm";
};

class VisionWeightMapper {
public:
    static const VisionWeightMapping& get_mapping(const std::string& arch);
    static void register_mapping(const std::string& arch, const VisionWeightMapping& mapping);
    static std::string format_layer_prefix(const std::string& pattern, int layer_idx);
    static std::vector<std::string> registered_archs();

private:
    static std::unordered_map<std::string, VisionWeightMapping>& mappings();
};

struct VisionWeightMappingAutoRegister {
    VisionWeightMappingAutoRegister(const std::string& arch, const VisionWeightMapping& mapping);
};

#define FORGE_REGISTER_VISION_WEIGHT_MAPPING_IMPL2(line, arch, mapping) \
    static ::forge::VisionWeightMappingAutoRegister _vision_wmap_reg_##line(arch, mapping)

#define FORGE_REGISTER_VISION_WEIGHT_MAPPING_IMPL(line, arch, mapping) \
    FORGE_REGISTER_VISION_WEIGHT_MAPPING_IMPL2(line, arch, mapping)

#define FORGE_REGISTER_VISION_WEIGHT_MAPPING(arch, mapping) \
    FORGE_REGISTER_VISION_WEIGHT_MAPPING_IMPL(__LINE__, arch, mapping)

// Force-link helper. Referencing this symbol forces the linker to include
// vision_registry.cpp, ensuring static auto-registration objects are constructed.
extern volatile bool _vision_registrations_linked;

// Detect vision architecture name from mmproj GGUF metadata.
// Maps "clip.vision.projector_type" (and "general.architecture") to a
// registered encoder name. Falls back to "siglip" (the default ViT family).
std::string detect_vision_arch(ModelLoader& loader);

}  // namespace forge
