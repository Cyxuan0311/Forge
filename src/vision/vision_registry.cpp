// ============================================================================
// Vision Encoder Registry + Config Parser Registry + Weight Mapper
// ============================================================================
// Centralizes registration of vision encoders, config parsers, and weight
// name mappings. Adding a new vision architecture requires only:
//   1. Implement a VisionEncoder subclass
//   2. Register it here (or in its own .cpp via the FORGE_REGISTER_* macros)
// ============================================================================
#include "forge/vision_registry.h"

#include "forge/logger.h"
#include "forge/model_loader.h"

namespace forge {

// Force-link symbol — referencing this from the Python module / CLI forces the
// linker to keep this translation unit (and its static auto-registrations).
volatile bool _vision_registrations_linked = true;

// ============================================================================
// VisionEncoderRegistry
// ============================================================================

VisionEncoderRegistry& VisionEncoderRegistry::instance() {
    static VisionEncoderRegistry registry;
    return registry;
}

void VisionEncoderRegistry::register_encoder(const std::string& name,
                                             VisionEncoderCreator creator) {
    creators_[name] = std::move(creator);
}

bool VisionEncoderRegistry::has(const std::string& name) const {
    return creators_.find(name) != creators_.end();
}

std::unique_ptr<VisionEncoder> VisionEncoderRegistry::create(const std::string& name) const {
    auto it = creators_.find(name);
    if (it == creators_.end()) {
        LOG_ERROR("VisionEncoderRegistry: no encoder registered for '" + name + "'");
        return nullptr;
    }
    return it->second();
}

std::vector<std::string> VisionEncoderRegistry::registered_encoders() const {
    std::vector<std::string> names;
    names.reserve(creators_.size());
    for (const auto& kv : creators_)
        names.push_back(kv.first);
    return names;
}

VisionEncoderAutoRegister::VisionEncoderAutoRegister(const std::string& name,
                                                     VisionEncoderCreator creator) {
    VisionEncoderRegistry::instance().register_encoder(name, std::move(creator));
}

// ============================================================================
// VisionConfigParserRegistry
// ============================================================================

VisionConfigParserRegistry& VisionConfigParserRegistry::instance() {
    static VisionConfigParserRegistry registry;
    return registry;
}

void VisionConfigParserRegistry::register_parser(const std::string& arch,
                                                 VisionConfigParseFn fn) {
    parsers_[arch] = std::move(fn);
}

bool VisionConfigParserRegistry::has(const std::string& arch) const {
    return parsers_.find(arch) != parsers_.end();
}

VisionConfig VisionConfigParserRegistry::parse(ModelLoader& loader,
                                               const std::string& arch) const {
    auto it = parsers_.find(arch);
    if (it == parsers_.end()) {
        LOG_WARN("VisionConfigParserRegistry: no parser for '" + arch +
                 "', falling back to siglip parser");
        // Fall back to siglip parser (covers the most common case)
        auto siglip_it = parsers_.find("siglip");
        if (siglip_it != parsers_.end())
            return siglip_it->second(loader);
        return VisionConfig{};
    }
    return it->second(loader);
}

std::vector<std::string> VisionConfigParserRegistry::registered_parsers() const {
    std::vector<std::string> names;
    names.reserve(parsers_.size());
    for (const auto& kv : parsers_)
        names.push_back(kv.first);
    return names;
}

VisionConfigParserAutoRegister::VisionConfigParserAutoRegister(const std::string& arch,
                                                               VisionConfigParseFn fn) {
    VisionConfigParserRegistry::instance().register_parser(arch, std::move(fn));
}

// ============================================================================
// VisionWeightMapper
// ============================================================================

std::unordered_map<std::string, VisionWeightMapping>& VisionWeightMapper::mappings() {
    static std::unordered_map<std::string, VisionWeightMapping> map;
    return map;
}

const VisionWeightMapping& VisionWeightMapper::get_mapping(const std::string& arch) {
    auto& m = mappings();
    auto it = m.find(arch);
    if (it == m.end()) {
        // Fall back to siglip mapping (the default ViT naming convention)
        auto siglip_it = m.find("siglip");
        if (siglip_it != m.end())
            return siglip_it->second;
        static const VisionWeightMapping default_mapping{};
        return default_mapping;
    }
    return it->second;
}

void VisionWeightMapper::register_mapping(const std::string& arch,
                                          const VisionWeightMapping& mapping) {
    mappings()[arch] = mapping;
}

std::string VisionWeightMapper::format_layer_prefix(const std::string& pattern, int layer_idx) {
    const std::string placeholder = "{i}";
    auto pos = pattern.find(placeholder);
    if (pos == std::string::npos)
        return pattern;
    return pattern.substr(0, pos) + std::to_string(layer_idx) +
           pattern.substr(pos + placeholder.size());
}

std::vector<std::string> VisionWeightMapper::registered_archs() {
    std::vector<std::string> names;
    for (const auto& kv : mappings())
        names.push_back(kv.first);
    return names;
}

VisionWeightMappingAutoRegister::VisionWeightMappingAutoRegister(const std::string& arch,
                                                                 const VisionWeightMapping& mapping) {
    VisionWeightMapper::register_mapping(arch, mapping);
}

// ============================================================================
// detect_vision_arch — map mmproj metadata to a registered encoder name
// ============================================================================
std::string detect_vision_arch(ModelLoader& loader) {
    std::string proj_type = loader.get_metadata_str("clip.vision.projector_type", "");
    if (!proj_type.empty()) {
        // SigLIP-family projector types -> siglip encoder
        if (proj_type == "minicpmv4_6" || proj_type == "minicpmv" ||
            proj_type == "siglip" || proj_type == "gemma3" ||
            proj_type == "idefics3" || proj_type == "lfm2" ||
            proj_type == "janus_pro" || proj_type == "phi4") {
            return "siglip";
        }
        // Qwen3-VL projector type
        if (proj_type == "qwen3vl_merger" || proj_type == "qwen2vl_merger" ||
            proj_type == "qwen25vl_merger") {
            return "qwen3vl";
        }
        // Unknown projector types fall through to default
    }

    // Heuristic: detect Qwen3-VL by unique metadata keys that SigLIP doesn't have
    int spatial_merge = static_cast<int>(
        loader.get_metadata_int("clip.vision.spatial_merge_size", 0));
    auto ds_flags = loader.get_metadata_int_array("clip.vision.is_deepstack_layers", {});
    bool has_deepstack = false;
    for (auto v : ds_flags) {
        if (v) { has_deepstack = true; break; }
    }
    if (spatial_merge >= 2 || has_deepstack) {
        return "qwen3vl";
    }

    // Fall back to general.architecture if present
    std::string general_arch = loader.get_metadata_str("general.architecture", "");
    if (!general_arch.empty()) {
        if (general_arch == "clip")
            return "siglip";
    }
    return "siglip";  // default to SigLIP ViT family
}

// ============================================================================
// SigLIP config parser — reads mmproj GGUF metadata
// ============================================================================
// This is the shared config parsing logic previously duplicated inline in
// PyMultimodalModel::load() and forge_cli.cpp. It reads only from the mmproj
// loader; callers may supplement fields (e.g. image_size, projection_dim)
// from the LLM file afterwards.
// ============================================================================
static VisionConfig parse_siglip_vision_config(ModelLoader& loader) {
    VisionConfig cfg;

    // SigLIP-L / MiniCPM-V 4.6 默认值（与旧代码一致），metadata 缺失时兜底
    cfg.image_size =
        static_cast<int>(loader.get_metadata_int("clip.vision.image_size", 448));
    cfg.patch_size =
        static_cast<int>(loader.get_metadata_int("clip.vision.patch_size", 14));
    cfg.embedding_length =
        static_cast<int>(loader.get_metadata_int("v.embedding_length", 1152));
    cfg.feed_forward_length =
        static_cast<int>(loader.get_metadata_int("v.feed_forward_length", 4304));
    cfg.block_count =
        static_cast<int>(loader.get_metadata_int("v.block_count", 27));
    cfg.head_count =
        static_cast<int>(loader.get_metadata_int("v.attention.head_count", 16));
    cfg.scale_factor =
        static_cast<int>(loader.get_metadata_int("clip.vision.projector.scale_factor", 4));

    // ViT merger insertion point (MiniCPM-V 4.6 uses wa_layer_indexes)
    auto wa_layers = loader.get_metadata_int_array("clip.vision.wa_layer_indexes", {});
    if (!wa_layers.empty()) {
        cfg.insert_layer_id = static_cast<int>(wa_layers[0]);
    } else {
        cfg.insert_layer_id =
            static_cast<int>(loader.get_metadata_int("clip.vision.insert_layer_id", 6));
    }

    // Projector type drives position embedding type and projection logic.
    // Default to "minicpmv4_6" for backward compatibility when the metadata
    // key is absent (existing mmproj files predate this field).
    cfg.projector_type = loader.get_metadata_str("clip.vision.projector_type", "minicpmv4_6");

    // Infer position embedding type from projector_type.
    // SigLIP / MiniCPM-V use 2D bucket positional embeddings.
    if (cfg.projector_type == "minicpmv4_6" || cfg.projector_type == "minicpmv" ||
        cfg.projector_type == "siglip" || cfg.projector_type == "none") {
        cfg.pos_embed_type = PosEmbedType::Learned2D;
    } else {
        cfg.pos_embed_type = PosEmbedType::Learned2D;  // default for ViT family
    }

    return cfg;
}

// ============================================================================
// Registrations
// ============================================================================

// --- siglip encoder ---
FORGE_REGISTER_VISION_ENCODER("siglip", []() -> std::unique_ptr<VisionEncoder> {
    return std::make_unique<SiglipViTEncoder>();
});

// --- siglip config parser ---
FORGE_REGISTER_VISION_CONFIG_PARSER("siglip", parse_siglip_vision_config);

// --- siglip weight mapping (default ViT naming convention) ---
FORGE_REGISTER_VISION_WEIGHT_MAPPING("siglip", VisionWeightMapping{});

// ============================================================================
// Qwen3-VL config parser — reads mmproj GGUF metadata
// ============================================================================
static VisionConfig parse_qwen3vl_vision_config(ModelLoader& loader) {
    VisionConfig cfg;

    cfg.image_size =
        static_cast<int>(loader.get_metadata_int("clip.vision.image_size", 768));
    cfg.patch_size =
        static_cast<int>(loader.get_metadata_int("clip.vision.patch_size", 16));
    cfg.embedding_length =
        static_cast<int>(loader.get_metadata_int("clip.vision.embedding_length", 1024));
    cfg.feed_forward_length =
        static_cast<int>(loader.get_metadata_int("clip.vision.feed_forward_length", 4096));
    cfg.block_count =
        static_cast<int>(loader.get_metadata_int("clip.vision.block_count", 24));
    cfg.head_count =
        static_cast<int>(loader.get_metadata_int("clip.vision.attention.head_count", 16));
    cfg.spatial_merge_size =
        static_cast<int>(loader.get_metadata_int("clip.vision.spatial_merge_size", 2));
    cfg.scale_factor = cfg.spatial_merge_size * cfg.spatial_merge_size;  // 4

    // Deepstack layers from is_deepstack_layers metadata
    auto ds_flags = loader.get_metadata_int_array("clip.vision.is_deepstack_layers", {});
    if (!ds_flags.empty()) {
        for (int i = 0; i < static_cast<int>(ds_flags.size()); ++i) {
            if (ds_flags[i]) {
                cfg.deepstack_layers.push_back(i);
            }
        }
    }

    cfg.use_fused_qkv = true;
    cfg.use_gelu = true;
    cfg.projector_type = loader.get_metadata_str("clip.vision.projector_type", "qwen3vl_merger");
    cfg.pos_embed_type = PosEmbedType::Learned1D;  // learned additive position embedding
    cfg.insert_layer_id = -1;  // no merger insertion

    // Qwen3-VL uses ImageNet normalization
    cfg.image_mean = {0.48145466f, 0.4578125f, 0.40821073f};
    cfg.image_std = {0.26862954f, 0.26130258f, 0.27577711f};

    return cfg;
}

// --- qwen3vl encoder ---
FORGE_REGISTER_VISION_ENCODER("qwen3vl", []() -> std::unique_ptr<VisionEncoder> {
    return std::make_unique<Qwen3VLViTEncoder>();
});

// --- qwen3vl config parser ---
FORGE_REGISTER_VISION_CONFIG_PARSER("qwen3vl", parse_qwen3vl_vision_config);

// --- qwen3vl weight mapping ---
static const VisionWeightMapping qwen3vl_weight_mapping = {
    "v.patch_embd.weight",     // patch_embd_weight
    "v.patch_embd.bias",       // patch_embd_bias
    "v.position_embd.weight",  // position_embd_weight
    "v.post_ln.weight",        // post_ln_weight
    "v.post_ln.bias",          // post_ln_bias
    "v.blk.{i}",               // block_prefix
    ".ln1.weight",             // blk_ln1_w
    ".ln1.bias",               // blk_ln1_b
    ".ln2.weight",             // blk_ln2_w
    ".ln2.bias",               // blk_ln2_b
    ".attn_q.weight",          // blk_q_w
    ".attn_q.bias",            // blk_q_b
    ".attn_k.weight",          // blk_k_w
    ".attn_k.bias",            // blk_k_b
    ".attn_v.weight",          // blk_v_w
    ".attn_v.bias",            // blk_v_b
    ".attn_out.weight",        // blk_o_w
    ".attn_out.bias",          // blk_o_b
    ".ffn_up.weight",          // blk_ffn_up_w
    ".ffn_up.bias",            // blk_ffn_up_b
    ".ffn_down.weight",        // blk_ffn_down_w
    ".ffn_down.bias",          // blk_ffn_down_b
    "v.vit_merger",            // merger_prefix
    "mm",                      // projector_prefix
};
FORGE_REGISTER_VISION_WEIGHT_MAPPING("qwen3vl", qwen3vl_weight_mapping);

}  // namespace forge
