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
#include "forge/engines/bailingmoe3_engine.h"
#include "forge/engines/deepseek_engine.h"
#include "forge/engines/dflash_engine.h"
#include "forge/engines/gemma4_engine.h"
#include "forge/engines/gemma_embedding2_engine.h"
#include "forge/engines/generic_engine.h"
#include "forge/engines/lfm2_engine.h"
#include "forge/engines/mellum_engine.h"
#include "forge/engines/phimoe_engine.h"
#include "forge/engines/qwen35_engine.h"
#include "forge/engines/spark25_engine.h"
#include "forge/engines/xing4_engine.h"
#include "forge/logger.h"
#include "forge/model.h"

namespace forge {

// Force-link this translation unit. Call from Python module init or CLI.
// Without this, the linker may discard static auto-registration objects
// in shared library builds.
volatile bool _arch_registrations_linked = true;

void force_link_arch_registrations() {
    _arch_registrations_linked = true;
}

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
static EngineAutoRegister _eng_llama("llama", make_generic);
static ConfigParserAutoRegister _cfg_llama("llama", parse_llama_config);
static WeightInitAutoRegister _winit_llama("llama", init_gqa_layer_weights);
static ArchCapabilityAutoRegister _cap_llama("llama", ArchCapability{.use_gqa = true,
                                                                     .use_neox_rope = true});

// --- Mistral ---
static EngineAutoRegister _eng_mistral("mistral", make_generic);
static ConfigParserAutoRegister _cfg_mistral("mistral", parse_mistral_config);
static WeightInitAutoRegister _winit_mistral("mistral", init_gqa_layer_weights);
static ArchCapabilityAutoRegister _cap_mistral("mistral", ArchCapability{.use_gqa = true,
                                                                         .use_neox_rope = true});

// --- Qwen ---
static EngineAutoRegister _eng_qwen("qwen", make_generic);
static ConfigParserAutoRegister _cfg_qwen("qwen", parse_qwen_config);
static WeightInitAutoRegister _winit_qwen("qwen", init_gqa_layer_weights);
static ArchCapabilityAutoRegister _cap_qwen("qwen", ArchCapability{.use_gqa = true});

// --- Qwen2 ---
static EngineAutoRegister _eng_qwen2("qwen2", make_generic);
static ConfigParserAutoRegister _cfg_qwen2("qwen2", parse_qwen2_config);
static WeightInitAutoRegister _winit_qwen2("qwen2", init_gqa_layer_weights);
static ArchCapabilityAutoRegister _cap_qwen2("qwen2", ArchCapability{.use_gqa = true});

// --- Yi ---
static EngineAutoRegister _eng_yi("yi", make_generic);
static ConfigParserAutoRegister _cfg_yi("yi", parse_yi_config);
static WeightInitAutoRegister _winit_yi("yi", init_gqa_layer_weights);
static ArchCapabilityAutoRegister _cap_yi("yi",
                                          ArchCapability{.use_gqa = true, .use_neox_rope = true});

// --- DeepSeek (GQA) ---
static EngineAutoRegister _eng_deepseek("deepseek", make_generic);
static ConfigParserAutoRegister _cfg_deepseek("deepseek", parse_deepseek_config);
static WeightInitAutoRegister _winit_deepseek("deepseek", init_gqa_layer_weights);
static ArchCapabilityAutoRegister _cap_deepseek("deepseek", ArchCapability{.use_gqa = true});

// --- Qwen3-VL (GQA + MRoPE + QK-Norm) ---
static EngineAutoRegister _eng_qwen3vl("qwen3vl", make_generic);
static ConfigParserAutoRegister _cfg_qwen3vl("qwen3vl", parse_qwen3vl_config);
static WeightInitAutoRegister _winit_qwen3vl("qwen3vl", init_gqa_layer_weights);
static ArchCapabilityAutoRegister _cap_qwen3vl("qwen3vl", ArchCapability{.use_gqa = true,
                                                                         .use_mrope = true,
                                                                         .use_qk_norm = true});

// --- Phi-3 (no dedicated engine — falls back to capability-based lookup) ---
// No EngineAutoRegister for phi3
static ConfigParserAutoRegister _cfg_phi3("phi3", parse_phi3_config);
static WeightInitAutoRegister _winit_phi3("phi3", init_gqa_layer_weights);
static ArchCapabilityAutoRegister _cap_phi3("phi3",
                                            ArchCapability{.use_gqa = true, .use_neox_rope = true});

// ============================================================================
// phimoe (Phi-mini-MoE-instruct: GQA + NeoX RoPE + MoE FFN + norm biases)
// ============================================================================

static EngineAutoRegister _eng_phimoe("phimoe",
                                      [](Model& m,
                                         InferenceContext& c) -> std::unique_ptr<InferenceEngine> {
                                          return std::make_unique<PhimoeEngine>(m, c);
                                      });
static ConfigParserAutoRegister _cfg_phimoe("phimoe", parse_phimoe_config);
static WeightInitAutoRegister _winit_phimoe("phimoe", init_phimoe_layer_weights);
static ArchCapabilityAutoRegister _cap_phimoe("phimoe", ArchCapability{.use_gqa = true,
                                                                       .use_neox_rope = true,
                                                                       .has_qkv_bias = true,
                                                                       .has_norm_bias = true});

// ============================================================================
// Gemma family (GeGLU + embedding_scale)
// ============================================================================

// --- Gemma ---
static EngineAutoRegister _eng_gemma("gemma", make_generic);
static ConfigParserAutoRegister _cfg_gemma("gemma", parse_gemma_config);
static WeightInitAutoRegister _winit_gemma("gemma", init_gqa_layer_weights);
static ArchCapabilityAutoRegister _cap_gemma("gemma",
                                             ArchCapability{.use_gqa = true,
                                                            .use_neox_rope = true,
                                                            .ffn_activation = ActivationType::GeGLU,
                                                            .embedding_scale = true});

// --- Gemma2 ---
static EngineAutoRegister _eng_gemma2("gemma2", make_generic);
static ConfigParserAutoRegister _cfg_gemma2("gemma2", parse_gemma2_config);
static WeightInitAutoRegister _winit_gemma2("gemma2", init_gqa_layer_weights);
static ArchCapabilityAutoRegister _cap_gemma2("gemma2", ArchCapability{
                                                            .use_gqa = true,
                                                            .use_neox_rope = true,
                                                            .ffn_activation = ActivationType::GeGLU,
                                                            .embedding_scale = true,
                                                            .has_post_attention_norm = true,
                                                            .has_post_ffn_norm = true});

// --- Gemma4 (dedicated Gemma4Engine: MoE + per-layer embeddings + SWA) ---
static EngineAutoRegister _eng_gemma4("gemma4",
                                      [](Model& m,
                                         InferenceContext& c) -> std::unique_ptr<InferenceEngine> {
                                          return std::make_unique<Gemma4Engine>(m, c);
                                      });
static ConfigParserAutoRegister _cfg_gemma4("gemma4", parse_gemma4_config);
static WeightInitAutoRegister _winit_gemma4("gemma4", init_gemma4_layer_weights);
static ArchCapabilityAutoRegister _cap_gemma4("gemma4", ArchCapability{
                                                            .use_gqa = true,
                                                            .use_neox_rope = true,
                                                            .ffn_activation = ActivationType::GeGLU,
                                                            .use_qk_norm = true,
                                                            .embedding_scale = true});

// --- Gemma-embedding2 (EmbeddingGemma 2: embedding-only Gemma4 stack) ---
static EngineAutoRegister _eng_gemma_embedding2(
    "gemma-embedding2",
    [](Model& m, InferenceContext& c) -> std::unique_ptr<InferenceEngine> {
        return std::make_unique<GemmaEmbedding2Engine>(m, c);
    });
static ConfigParserAutoRegister _cfg_gemma_embedding2("gemma-embedding2",
                                                      parse_gemma_embedding2_config);
static WeightInitAutoRegister _winit_gemma_embedding2("gemma-embedding2",
                                                      init_gemma4_layer_weights);
static ArchCapabilityAutoRegister _cap_gemma_embedding2(
    "gemma-embedding2", ArchCapability{.use_gqa = true,
                                       .use_neox_rope = true,
                                       .ffn_activation = ActivationType::GeGLU,
                                       .use_qk_norm = true,
                                       .embedding_scale = true});

// ============================================================================
// Falcon (LayerNorm + bias + parallel residual)
// ============================================================================

static EngineAutoRegister _eng_falcon("falcon", make_generic);
static ConfigParserAutoRegister _cfg_falcon("falcon", parse_falcon_config);
static WeightInitAutoRegister _winit_falcon("falcon", init_falcon_layer_weights);
static ArchCapabilityAutoRegister _cap_falcon("falcon",
                                              ArchCapability{.use_gqa = true,
                                                             .ffn_activation = ActivationType::GELU,
                                                             .norm_type = NormType::LayerNorm,
                                                             .has_qkv_bias = true,
                                                             .has_norm_bias = true,
                                                             .use_parallel_residual = true});

// ============================================================================
// DeepSeek V2/V3 (MLA architecture — dedicated engine)
// ============================================================================

static EngineAutoRegister _eng_deepseek_v2(
    "deepseek_v2", [](Model& m, InferenceContext& c) -> std::unique_ptr<InferenceEngine> {
        return std::make_unique<DeepSeekEngine>(m, c);
    });
static ConfigParserAutoRegister _cfg_deepseek_v2("deepseek_v2", parse_deepseek_v2_config);
static WeightInitAutoRegister _winit_deepseek_v2("deepseek_v2", init_mla_layer_weights);
static ArchCapabilityAutoRegister _cap_deepseek_v2("deepseek_v2", ArchCapability{.use_mla = true});

static EngineAutoRegister _eng_deepseek_v3(
    "deepseek_v3", [](Model& m, InferenceContext& c) -> std::unique_ptr<InferenceEngine> {
        return std::make_unique<DeepSeekEngine>(m, c);
    });
static ConfigParserAutoRegister _cfg_deepseek_v3("deepseek_v3", parse_deepseek_v3_config);
static WeightInitAutoRegister _winit_deepseek_v3("deepseek_v3", init_mla_layer_weights);
static ArchCapabilityAutoRegister _cap_deepseek_v3("deepseek_v3", ArchCapability{.use_mla = true});

// ============================================================================
// Qwen3.5 (SSM hybrid — dedicated Qwen35Engine)
// ============================================================================

static EngineAutoRegister _eng_qwen35("qwen35",
                                      [](Model& m,
                                         InferenceContext& c) -> std::unique_ptr<InferenceEngine> {
                                          return std::make_unique<Qwen35Engine>(m, c);
                                      });
static ConfigParserAutoRegister _cfg_qwen35("qwen35", parse_qwen35_config);
static WeightInitAutoRegister _winit_qwen35("qwen35", init_qwen35_layer_weights);
static ArchCapabilityAutoRegister _cap_qwen35("qwen35",
                                              ArchCapability{.use_ssm = true, .use_mrope = true});

// ============================================================================
// Spark-X2.5 (iFlytek SparkLLM: mixed SWA/full attention, dual RoPE,
// per-head sigmoid output gate, GeGLU, tied embeddings)
// ============================================================================

static EngineAutoRegister _eng_spark25(
    "spark2_5", [](Model& m, InferenceContext& c) -> std::unique_ptr<InferenceEngine> {
        return std::make_unique<Spark25Engine>(m, c);
    });
static ConfigParserAutoRegister _cfg_spark25("spark2_5", parse_spark25_config);
static WeightInitAutoRegister _winit_spark25("spark2_5", init_spark25_layer_weights);
static ArchCapabilityAutoRegister _cap_spark25("spark2_5",
                                               ArchCapability{.use_gqa = true,
                                                              .use_neox_rope = true,
                                                              .ffn_activation = ActivationType::GeGLU});

// ============================================================================
// LFM2 / LFM2.5 (LiquidAI: hybrid shortconv operator + GQA attention,
// tied embeddings, QK-norm + NeoX RoPE)
// ============================================================================

static EngineAutoRegister _eng_lfm2(
    "lfm2", [](Model& m, InferenceContext& c) -> std::unique_ptr<InferenceEngine> {
        return std::make_unique<Lfm2Engine>(m, c);
    });
static ConfigParserAutoRegister _cfg_lfm2("lfm2", parse_lfm2_config);
static WeightInitAutoRegister _winit_lfm2("lfm2", init_lfm2_layer_weights);
static ArchCapabilityAutoRegister _cap_lfm2("lfm2",
                                            ArchCapability{.use_gqa = true,
                                                           .use_qk_norm = true});

// ============================================================================
// BailingMoE3 (Ling 3.0: hybrid MoE + MLA + SSM/KDA recurrent attention)
// ============================================================================
// The engine is registered here (like every other architecture) rather than via
// FORGE_REGISTER_ENGINE inside bailingmoe3_engine.cpp: registering centrally
// also references Bailingmoe3Engine's constructor, which force-links that
// translation unit. Left to self-registration alone, the linker drops
// bailingmoe3_engine.o from static-library builds and the engine never
// registers ("resolves to engine 'bailingmoe3' but that engine is not
// registered").

static EngineAutoRegister _eng_bailingmoe3(
    "bailingmoe3", [](Model& m, InferenceContext& c) -> std::unique_ptr<InferenceEngine> {
        return std::make_unique<Bailingmoe3Engine>(m, c);
    });
static ConfigParserAutoRegister _cfg_bailingmoe3("bailingmoe3", parse_bailingmoe3_config);
static WeightInitAutoRegister _winit_bailingmoe3("bailingmoe3",
                                                 init_bailingmoe3_layer_weights);
static ArchCapabilityAutoRegister _cap_bailingmoe3(
    "bailingmoe3",
    ArchCapability{.use_mla = true, .use_ssm = true});

// ============================================================================
// Xing4.0 (China Telecom): absorbed MLA + sigmoid MoE + mHC hyper-connections
// ============================================================================

static EngineAutoRegister _eng_xing4_0(
    "xing4_0", [](Model& m, InferenceContext& c) -> std::unique_ptr<InferenceEngine> {
        return std::make_unique<Xing4_0Engine>(m, c);
    });
static ConfigParserAutoRegister _cfg_xing4_0("xing4_0", parse_xing4_0_config);
static WeightInitAutoRegister _winit_xing4_0("xing4_0", init_xing4_0_layer_weights);
static ArchCapabilityAutoRegister _cap_xing4_0("xing4_0",
                                               ArchCapability{.use_mla = true, .use_ssm = false});

// ============================================================================
// Mellum (Mellum2.1: GQA + per-head QK-norm + SWA/full 交错 + 路由-only SiLU MoE,
// 全注意力层 yarn)。张量名与 gemma4 兼容 → 权重 init 直接复用 init_gemma4_layer_weights。
// ============================================================================

static EngineAutoRegister _eng_mellum(
    "mellum", [](Model& m, InferenceContext& c) -> std::unique_ptr<InferenceEngine> {
        return std::make_unique<MellumEngine>(m, c);
    });
static ConfigParserAutoRegister _cfg_mellum("mellum", parse_mellum_config);
static WeightInitAutoRegister _winit_mellum("mellum", init_gemma4_layer_weights);
static ArchCapabilityAutoRegister _cap_mellum("mellum",
                                              ArchCapability{.use_gqa = true,
                                                             .ffn_activation = ActivationType::SiLU_GELU,
                                                             .use_qk_norm = true});

// ============================================================================
// DFlash / DSPark (standalone lightweight drafter for speculative decoding)
// ============================================================================
// The draft engine lands in Phase 2. This creator returns nullptr with an
// explicit warning so Phase 0/1 can load, parse and inspect the GGUF
// (metadata + tensors) without being able to execute it.

static EngineAutoRegister _eng_dflash("dflash",
                                      [](Model& m,
                                         InferenceContext& c) -> std::unique_ptr<InferenceEngine> {
                                          return std::make_unique<DflashEngine>(m, c);
                                      });
static ConfigParserAutoRegister _cfg_dflash("dflash", parse_dflash_config);
static WeightInitAutoRegister _winit_dflash("dflash", init_gqa_layer_weights);
static ArchCapabilityAutoRegister _cap_dflash("dflash",
                                              ArchCapability{.use_gqa = true, .use_qk_norm = true});

}  // namespace forge
