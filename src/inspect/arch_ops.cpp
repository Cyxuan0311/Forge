#include "arch_ops.h"

#include <algorithm>

#include "forge/arch_config_parsers.h"
#include "forge/arch_registry.h"
#include "forge/engine.h"
#include "forge/model.h"

namespace forge::inspect {

namespace {

ArchOpPipeline gqa_pipeline() {
    return {
        /*pre=*/{{"EMBEDDING", "token embedding"}},
        /*layer=*/
        {{"RMS_NORM", "attn norm"},
         {"MUL_MAT_TRANSB", "q/k/v projections"},
         {"ROPE", "rotary position"},
         {"FLASH_ATTN_GQA", "grouped attention"},
         {"MUL_MAT_TRANSB", "output projection"},
         {"ADD", "residual"},
         {"RMS_NORM", "ffn norm"},
         {"MUL_MAT_TRANSB", "gate/up projections"},
         {"SILU", "activation"},
         {"MUL_MAT_TRANSB", "down projection"},
         {"ADD", "residual"}},
        /*post=*/
        {{"RMS_NORM", "output norm"}, {"MUL_MAT_TRANSB", "output projection"}},
    };
}

ArchOpPipeline gemma_pipeline() {
    auto p = gqa_pipeline();
    p.pre_steps = {{"SCALE", "embedding scaling"}};
    // Gemma uses GeGLU (GELU on the gate branch).
    for (auto& s : p.layer_steps) {
        if (s.op == "SILU")
            s.op = "GELU";
    }
    return p;
}

ArchOpPipeline mla_pipeline() {
    return {
        /*pre=*/{{"EMBEDDING", "token embedding"}},
        /*layer=*/
        {{"RMS_NORM", "attn norm"},
         {"MUL_MAT_TRANSB", "q_a / q_b projections"},
         {"MUL_MAT_TRANSB", "kv_a / kv_b projections (MLA)"},
         {"ROPE", "rotary position"},
         {"FLASH_ATTN_GQA", "latent attention"},
         {"MUL_MAT_TRANSB", "output projection"},
         {"ADD", "residual"},
         {"RMS_NORM", "ffn norm"},
         {"MUL_MAT_TRANSB", "gate/up projections"},
         {"SILU", "activation"},
         {"MUL_MAT_TRANSB", "down projection"},
         {"ADD", "residual"}},
        /*post=*/
        {{"RMS_NORM", "output norm"}, {"MUL_MAT_TRANSB", "output projection"}},
    };
}

ArchOpPipeline moe_pipeline() {
    return {
        /*pre=*/{{"EMBEDDING", "token embedding"}},
        /*layer=*/
        {{"RMS_NORM", "attn norm"},
         {"MUL_MAT_TRANSB", "q/k/v projections"},
         {"ROPE", "rotary position"},
         {"FLASH_ATTN_GQA", "grouped attention"},
         {"MUL_MAT_TRANSB", "output projection"},
         {"ADD", "residual"},
         {"RMS_NORM", "ffn norm"},
         {"MUL_MAT_TRANSB", "router (gate input)"},
         {"MUL_MAT_TRANSB", "expert up/down projections (MoE)"},
         {"ADD", "residual"}},
        /*post=*/
        {{"RMS_NORM", "output norm"}, {"MUL_MAT_TRANSB", "output projection"}},
    };
}

ArchOpPipeline qwen35_pipeline() {
    return {
        /*pre=*/{{"EMBEDDING", "token embedding"}},
        /*layer=*/
        {{"RMS_NORM", "attn norm"},
         {"MUL_MAT_TRANSB", "q/k/v projections"},
         {"ROPE", "rotary position"},
         {"FLASH_ATTN_GQA", "full attention"},
         {"MUL_MAT_TRANSB", "linear attention / SSM (recurrent layers)"},
         {"MUL_MAT_TRANSB", "output projection"},
         {"ADD", "residual"},
         {"RMS_NORM", "ffn norm"},
         {"MUL_MAT_TRANSB", "gate/up projections"},
         {"SILU", "activation"},
         {"MUL_MAT_TRANSB", "down projection"},
         {"ADD", "residual"}},
        /*post=*/
        {{"RMS_NORM", "output norm"}, {"MUL_MAT_TRANSB", "output projection"}},
    };
}

}  // namespace

ArchFamily family_for_arch(const std::string& arch) {
    // Single source of truth for arch -> family. sniff_arch() and pipeline_for()
    // both call this, so they cannot disagree about what family an arch is in.
    static const struct {
        const char* arch;
        ArchFamily family;
    } kTable[] = {
        {"llama", ArchFamily::Gqa},     {"mistral", ArchFamily::Gqa},
        {"yi", ArchFamily::Gqa},        {"qwen", ArchFamily::Gqa},
        {"qwen2", ArchFamily::Gqa},     {"qwen3vl", ArchFamily::Gqa},
        {"phi3", ArchFamily::Gqa},      {"falcon", ArchFamily::Gqa},
        {"gemma", ArchFamily::Gemma},   {"gemma2", ArchFamily::Gemma},
        {"deepseek", ArchFamily::Mla},  {"deepseek_v2", ArchFamily::Mla},
        {"deepseek_v3", ArchFamily::Mla},
        {"phimoe", ArchFamily::Moe},    {"gemma4", ArchFamily::Moe},
        {"qwen35", ArchFamily::Hybrid},
    };
    for (const auto& e : kTable) {
        if (arch == e.arch)
            return e.family;
    }
    return ArchFamily::Unknown;
}

const char* family_name(ArchFamily f) {
    switch (f) {
    case ArchFamily::Gqa: return "gqa";
    case ArchFamily::Gemma: return "gemma";
    case ArchFamily::Mla: return "mla";
    case ArchFamily::Moe: return "moe";
    case ArchFamily::Hybrid: return "hybrid";
    case ArchFamily::Unknown: break;
    }
    return "";
}

const ArchOpPipeline* pipeline_for(const std::string& arch) {
    switch (family_for_arch(arch)) {
    case ArchFamily::Gqa: {
        static const auto p = gqa_pipeline();
        return &p;
    }
    case ArchFamily::Gemma: {
        static const auto p = gemma_pipeline();
        return &p;
    }
    case ArchFamily::Mla: {
        static const auto p = mla_pipeline();
        return &p;
    }
    case ArchFamily::Moe: {
        static const auto p = moe_pipeline();
        return &p;
    }
    case ArchFamily::Hybrid: {
        static const auto p = qwen35_pipeline();
        return &p;
    }
    case ArchFamily::Unknown:
        break;
    }
    return nullptr;
}

bool arch_supported(const std::string& arch) {
    static bool linked = (forge::force_link_arch_registrations(), true);
    (void)linked;
    bool has_cfg = ConfigParserRegistry::instance().has(arch);
    bool has_cap = ArchCapabilityRegistry::instance().has(arch);
    bool has_eng = EngineRegistry::instance().has(arch);
    return has_cfg || has_cap || has_eng;
}

std::vector<std::string> supported_archs() {
    forge::force_link_arch_registrations();
    std::vector<std::string> archs;
    for (const auto& [arch, cap] : ArchCapabilityRegistry::instance().all())
        archs.push_back(arch);
    for (const auto& a : EngineRegistry::instance().registered_archs())
        archs.push_back(a);
    std::sort(archs.begin(), archs.end());
    archs.erase(std::unique(archs.begin(), archs.end()), archs.end());
    return archs;
}

}  // namespace forge::inspect
