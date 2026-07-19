// ============================================================================
// Unified Architecture Registration
// ============================================================================
// All architecture registrations (engine, config parser, weight init, capability)
// are centralized in this single file.
//
// Adding a new architecture requires only adding a registration block here,
// plus implementing the config parser and weight init functions
// in arch_config_parser.cpp and arch_weight_init.cpp respectively.
//
// IMPORTANT: This file contains only static auto-registration objects.
// To prevent the linker from discarding this translation unit in shared
// libraries, call force_link_arch_registrations() from your module init.
// ============================================================================

#include "forge/arch_config_parsers.h"
#include "forge/arch_registry.h"
#include "forge/arch_weight_inits.h"
#include "forge/engine.h"
#include "forge/engines/deepseek_engine.h"
#include "forge/engines/gemma4_engine.h"
#include "forge/engines/generic_engine.h"
#include "forge/engines/qwen35_engine.h"
#include "forge/model.h"

namespace forge {

// Force-link this translation unit. Call from Python module init or CLI.
// Without this, the linker may discard static auto-registration objects
// in shared library builds.
volatile bool _arch_registrations_linked = true;

// ============================================================================
// Helper: GenericEngine creator lambda (reused by most GQA architectures)
// ============================================================================

static auto make_generic = [](Model& m, InferenceContext& c) -> std::unique_ptr<InferenceEngine> {
    return std::make_unique<GenericEngine>(m, c);
};

// ============================================================================
// Standard GQA architectures (share GenericEngine + init_gqa_layer_weights)
// ============================================================================

// --- Llama ---
static EngineAutoRegister           _eng_llama("llama", make_generic);
static ConfigParserAutoRegister     _cfg_llama("llama", parse_llama_config);
static WeightInitAutoRegister       _winit_llama("llama", init_gqa_layer_weights);
static ArchCapabilityAutoRegister   _cap_llama("llama",
    ArchCapability{.use_gqa = true, .use_neox_rope = true});

// --- Mistral ---
static EngineAutoRegister           _eng_mistral("mistral", make_generic);
static ConfigParserAutoRegister     _cfg_mistral("mistral", parse_mistral_config);
static WeightInitAutoRegister       _winit_mistral("mistral", init_gqa_layer_weights);
static ArchCapabilityAutoRegister   _cap_mistral("mistral",
    ArchCapability{.use_gqa = true, .use_neox_rope = true});

// --- Qwen ---
static EngineAutoRegister           _eng_qwen("qwen", make_generic);
static ConfigParserAutoRegister     _cfg_qwen("qwen", parse_qwen_config);
static WeightInitAutoRegister       _winit_qwen("qwen", init_gqa_layer_weights);
static ArchCapabilityAutoRegister   _cap_qwen("qwen",
    ArchCapability{.use_gqa = true});

// --- Qwen2 ---
static EngineAutoRegister           _eng_qwen2("qwen2", make_generic);
static ConfigParserAutoRegister     _cfg_qwen2("qwen2", parse_qwen2_config);
static WeightInitAutoRegister       _winit_qwen2("qwen2", init_gqa_layer_weights);
static ArchCapabilityAutoRegister   _cap_qwen2("qwen2",
    ArchCapability{.use_gqa = true});

// --- Yi ---
static EngineAutoRegister           _eng_yi("yi", make_generic);
static ConfigParserAutoRegister     _cfg_yi("yi", parse_yi_config);
static WeightInitAutoRegister       _winit_yi("yi", init_gqa_layer_weights);
static ArchCapabilityAutoRegister   _cap_yi("yi",
    ArchCapability{.use_gqa = true, .use_neox_rope = true});

// --- DeepSeek (GQA) ---
static EngineAutoRegister           _eng_deepseek("deepseek", make_generic);
static ConfigParserAutoRegister     _cfg_deepseek("deepseek", parse_deepseek_config);
static WeightInitAutoRegister       _winit_deepseek("deepseek", init_gqa_layer_weights);
static ArchCapabilityAutoRegister   _cap_deepseek("deepseek",
    ArchCapability{.use_gqa = true});

// --- Qwen3-VL (GQA + MRoPE + QK-Norm) ---
static EngineAutoRegister           _eng_qwen3vl("qwen3vl", make_generic);
static ConfigParserAutoRegister     _cfg_qwen3vl("qwen3vl", parse_qwen3vl_config);
static WeightInitAutoRegister       _winit_qwen3vl("qwen3vl", init_gqa_layer_weights);
static ArchCapabilityAutoRegister   _cap_qwen3vl("qwen3vl",
    ArchCapability{.use_gqa = true, .use_mrope = true, .use_qk_norm = true});

// --- Phi-3 (no dedicated engine — falls back to capability-based lookup) ---
// No EngineAutoRegister for phi3
static ConfigParserAutoRegister     _cfg_phi3("phi3", parse_phi3_config);
static WeightInitAutoRegister       _winit_phi3("phi3", init_gqa_layer_weights);
static ArchCapabilityAutoRegister   _cap_phi3("phi3",
    ArchCapability{.use_gqa = true, .use_neox_rope = true});

// ============================================================================
// Gemma family (GeGLU + embedding_scale)
// ============================================================================

// --- Gemma ---
static EngineAutoRegister           _eng_gemma("gemma", make_generic);
static ConfigParserAutoRegister     _cfg_gemma("gemma", parse_gemma_config);
static WeightInitAutoRegister       _winit_gemma("gemma", init_gqa_layer_weights);
static ArchCapabilityAutoRegister   _cap_gemma("gemma",
    ArchCapability{.use_gqa = true, .use_neox_rope = true,
                   .ffn_activation = ActivationType::GeGLU,
                   .embedding_scale = true});

// --- Gemma2 ---
static EngineAutoRegister           _eng_gemma2("gemma2", make_generic);
static ConfigParserAutoRegister     _cfg_gemma2("gemma2", parse_gemma2_config);
static WeightInitAutoRegister       _winit_gemma2("gemma2", init_gqa_layer_weights);
static ArchCapabilityAutoRegister   _cap_gemma2("gemma2",
    ArchCapability{.use_gqa = true, .use_neox_rope = true,
                   .ffn_activation = ActivationType::GeGLU,
                   .embedding_scale = true,
                   .has_post_attention_norm = true,
                   .has_post_ffn_norm = true});

// --- Gemma4 (dedicated Gemma4Engine: MoE + per-layer embeddings + SWA) ---
static EngineAutoRegister           _eng_gemma4("gemma4",
    [](Model& m, InferenceContext& c) -> std::unique_ptr<InferenceEngine> {
        return std::make_unique<Gemma4Engine>(m, c);
    });
static ConfigParserAutoRegister     _cfg_gemma4("gemma4", parse_gemma4_config);
static WeightInitAutoRegister       _winit_gemma4("gemma4", init_gemma4_layer_weights);
static ArchCapabilityAutoRegister   _cap_gemma4("gemma4",
    ArchCapability{.use_gqa = true, .use_neox_rope = true,
                   .ffn_activation = ActivationType::GeGLU,
                   .use_qk_norm = true,
                   .embedding_scale = true});

// ============================================================================
// Falcon (LayerNorm + bias + parallel residual)
// ============================================================================

static EngineAutoRegister           _eng_falcon("falcon", make_generic);
static ConfigParserAutoRegister     _cfg_falcon("falcon", parse_falcon_config);
static WeightInitAutoRegister       _winit_falcon("falcon", init_falcon_layer_weights);
static ArchCapabilityAutoRegister   _cap_falcon("falcon",
    ArchCapability{.use_gqa = true,
                   .ffn_activation = ActivationType::GELU,
                   .norm_type = NormType::LayerNorm,
                   .has_qkv_bias = true,
                   .has_norm_bias = true,
                   .use_parallel_residual = true});

// ============================================================================
// DeepSeek V2/V3 (MLA architecture — dedicated engine)
// ============================================================================

static EngineAutoRegister           _eng_deepseek_v2("deepseek_v2",
    [](Model& m, InferenceContext& c) -> std::unique_ptr<InferenceEngine> {
        return std::make_unique<DeepSeekEngine>(m, c);
    });
static ConfigParserAutoRegister     _cfg_deepseek_v2("deepseek_v2", parse_deepseek_v2_config);
static WeightInitAutoRegister       _winit_deepseek_v2("deepseek_v2", init_mla_layer_weights);
static ArchCapabilityAutoRegister   _cap_deepseek_v2("deepseek_v2",
    ArchCapability{.use_mla = true});

static EngineAutoRegister           _eng_deepseek_v3("deepseek_v3",
    [](Model& m, InferenceContext& c) -> std::unique_ptr<InferenceEngine> {
        return std::make_unique<DeepSeekEngine>(m, c);
    });
static ConfigParserAutoRegister     _cfg_deepseek_v3("deepseek_v3", parse_deepseek_v3_config);
static WeightInitAutoRegister       _winit_deepseek_v3("deepseek_v3", init_mla_layer_weights);
static ArchCapabilityAutoRegister   _cap_deepseek_v3("deepseek_v3",
    ArchCapability{.use_mla = true});

// ============================================================================
// Qwen3.5 (SSM hybrid — dedicated Qwen35Engine)
// ============================================================================

static EngineAutoRegister           _eng_qwen35("qwen35",
    [](Model& m, InferenceContext& c) -> std::unique_ptr<InferenceEngine> {
        return std::make_unique<Qwen35Engine>(m, c);
    });
static ConfigParserAutoRegister     _cfg_qwen35("qwen35", parse_qwen35_config);
static WeightInitAutoRegister       _winit_qwen35("qwen35", init_qwen35_layer_weights);
static ArchCapabilityAutoRegister   _cap_qwen35("qwen35",
    ArchCapability{.use_ssm = true, .use_mrope = true});

}  // namespace forge
