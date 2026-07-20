#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "model_loader.h"
#include "quant_policy.h"
#include "tensor.h"
#include "types.h"

namespace forge {

enum class NormType : int { RMSNorm, LayerNorm };
enum class ActivationType : int { SiLU_GELU, GELU, ReLU, GeGLU };
enum class RopeType : int { None, Standard, LinearScaling, NTK_Scaled, NeoX, MRoPE, Proportional };
enum class FFNType : int { SiLUGated, GeGLU, SimpleGELU, MoE };

struct ModelConfig {
    int vocab_size = 0;
    int hidden_dim = 0;
    int intermediate_dim = 0;
    int num_layers = 0;
    int num_heads = 0;
    int num_kv_heads = 0;
    int head_dim = 0;

    float rope_theta = 10000.0f;
    float rope_scaling_factor = 1.0f;
    float rms_norm_eps = 1e-6f;
    float layer_norm_eps = 1e-12f;

    int max_seq_len = 4096;

    std::string arch_type = "llama";
    NormType norm_type = NormType::RMSNorm;
    ActivationType ffn_activation = ActivationType::SiLU_GELU;
    RopeType rope_type = RopeType::Standard;
    bool tie_embeddings = false;
    bool use_gqa = true;
    bool use_neox_rope = false;

    int kv_lora_rank = 0;
    int q_lora_rank = 0;
    bool use_mla = false;
    int n_routed_experts = 0;
    int n_shared_experts = 0;
    int num_expert_per_tok = 0;

    // SSM/Mamba hybrid layer support (e.g., Qwen3.5, MiniCPM-V 4.6)
    int ssm_group_count = 0;          // number of SSM groups (n_group)
    int ssm_time_step_rank = 0;       // rank of SSM time step (dt_rank)
    int ssm_inner_size = 0;           // SSM inner size (d_inner)
    int ssm_state_size = 0;           // SSM state size (d_state = head_k_dim)
    int ssm_conv_kernel = 0;          // SSM conv kernel size (d_conv)
    int full_attention_interval = 0;  // attention layer interval in hybrid arch
    bool use_ssm = false;             // whether model has SSM layers

    // MRoPE (Multi-dimensional RoPE) support for Qwen3.5
    int rope_dimension_count = 0;                   // number of dimensions to apply RoPE (e.g., 64)
    int rope_dimension_sections[4] = {0, 0, 0, 0};  // sections for MRoPE [t, h, w, extra]
    bool use_mrope = false;                         // whether to use MRoPE instead of standard RoPE

    // Gemma2 softcapping
    float f_attn_logit_softcapping = 0.0f;
    float f_final_logit_softcapping = 0.0f;

    bool use_parallel_residual = false;

    // GenericEngine: FFN type and rope Q-scaling
    FFNType ffn_type = FFNType::SiLUGated;
    float rope_q_scale = 0.0f;  // >0 means apply 1/sqrt(head_dim) scaling to Q after RoPE
    bool has_post_attention_norm = false;
    bool has_post_ffn_norm = false;

    // Gemma4-specific fields
    int n_embd_per_layer = 0;            // per-layer embedding dimension (0 = disabled)
    int n_ff_exp = 0;                    // expert FFN intermediate dimension
    int n_expert = 0;                    // total number of experts
    int n_expert_used = 0;               // number of experts per token (top-K)
    int n_swa = 0;                       // sliding window attention size
    std::vector<int> swa_layers;         // which layers use SWA (1=SWA, 0=full attention)
    int n_layer_kv_from_start = 0;       // layers with own KV cache (from start)
    bool use_qk_norm = false;            // whether to apply QK-norm before attention
    int head_dim_swa = 0;               // head dimension for SWA layers (0 = same as head_dim)
    int num_heads_swa = 0;              // number of attention heads for SWA layers (0 = same as num_heads)
    int num_kv_heads_swa = 0;           // number of KV heads for SWA layers (0 = same as num_kv_heads)
    float rope_theta_swa = 10000.0f;    // RoPE freq base for SWA layers
    int rope_dim_count = 0;             // RoPE dimension count for full-attention layers
    int rope_dim_count_swa = 0;         // RoPE dimension count for SWA layers
    std::vector<int32_t> suppress_tokens; // tokens to suppress during generation (set to -INF in logits)
};

class WeightStore {
public:
    void set(const std::string& name, TensorPtr tensor);
    TensorPtr get(const std::string& name) const;
    TensorPtr get_or_null(const std::string& name) const;
    bool has(const std::string& name) const;
    const std::unordered_map<std::string, TensorPtr>& all() const;
    size_t size() const;
    void clear();
    size_t total_bytes() const;

    std::vector<std::string> weight_names() const;

    void to_device(DeviceType device);
    void to_device_layer(int layer_idx, DeviceType device,
                         const std::string& prefix_pattern = "model.layers.{}");

private:
    std::unordered_map<std::string, TensorPtr> weights_;
};

// ============================================================================
// Unified Weight Representation
// ============================================================================
// Replaces architecture-specific weight structs (LlamaWeights, DeepSeekWeights,
// Qwen35Weights) with a single unified model that uses canonical weight names.
//
// Canonical naming convention:
//   Model-level:  "token_embedding", "output_norm", "output_weight"
//   Layer-level:  "layers.{i}.{field}" where field is one of:
//     Common:       attn_norm, ffn_norm, wo, w1(gate_proj), w2(down_proj), w3(up_proj)
//     GQA Attn:     wq, wk, wv, bq, bk, bv
//     MLA Attn:     wq_a, wq_b, kv_a_proj, kv_b_proj
//     Qwen35 Attn:  attn_q, attn_k, attn_v, attn_output, attn_q_norm, attn_k_norm,
//                   post_attention_norm, attn_qkv, attn_gate
//     Qwen35 SSM:   ssm_conv1d, ssm_dt, ssm_a, ssm_alpha, ssm_beta, ssm_norm, ssm_out
// ============================================================================

// Layer type classification for hybrid architectures (e.g., Qwen3.5)
enum class LayerType : int {
    FullAttention = 0,    // Standard full attention layer
    LinearAttention = 1,  // Linear/recurrent attention (e.g., Gated Delta Net)
    MLA = 2,              // Multi-head Latent Attention (DeepSeek V2/V3)
};

// Per-layer weight container using canonical names
struct LayerWeights {
    // Raw storage: canonical_name -> tensor
    std::unordered_map<std::string, TensorPtr> weights;

    // Layer type (for hybrid architectures)
    LayerType layer_type = LayerType::FullAttention;

    // --- Accessors ---
    TensorPtr get(const std::string& name) const {
        auto it = weights.find(name);
        return it != weights.end() ? it->second : nullptr;
    }

    void set(const std::string& name, TensorPtr tensor) { weights[name] = std::move(tensor); }

    bool has(const std::string& name) const { return weights.find(name) != weights.end(); }

    // Move all weights to target device
    void to_device(DeviceType device);

    // --- Common convenience accessors ---
    TensorPtr attn_norm() const { return get("attn_norm"); }
    TensorPtr ffn_norm() const { return get("ffn_norm"); }
    TensorPtr wo() const { return get("wo"); }
    TensorPtr w1() const { return get("w1"); }
    TensorPtr w2() const { return get("w2"); }
    TensorPtr w3() const { return get("w3"); }

    // GQA attention
    TensorPtr wq() const { return get("wq"); }
    TensorPtr wk() const { return get("wk"); }
    TensorPtr wv() const { return get("wv"); }
    TensorPtr bq() const { return get("bq"); }
    TensorPtr bk() const { return get("bk"); }
    TensorPtr bv() const { return get("bv"); }

    // MLA attention
    TensorPtr wq_a() const { return get("wq_a"); }
    TensorPtr wq_b() const { return get("wq_b"); }
    TensorPtr kv_a_proj() const { return get("kv_a_proj"); }
    TensorPtr kv_b_proj() const { return get("kv_b_proj"); }

    // Qwen35 full attention
    TensorPtr attn_q() const { return get("attn_q"); }
    TensorPtr attn_k() const { return get("attn_k"); }
    TensorPtr attn_v() const { return get("attn_v"); }
    TensorPtr attn_output() const { return get("attn_output"); }
    TensorPtr attn_q_norm() const { return get("attn_q_norm"); }
    TensorPtr attn_k_norm() const { return get("attn_k_norm"); }
    TensorPtr post_attention_norm() const { return get("post_attention_norm"); }
    TensorPtr post_ffn_norm() const { return get("post_ffn_norm"); }

    // Qwen35 linear attention / SSM
    TensorPtr attn_qkv() const { return get("attn_qkv"); }
    TensorPtr attn_gate() const { return get("attn_gate"); }
    TensorPtr ssm_conv1d() const { return get("ssm_conv1d"); }
    TensorPtr ssm_dt() const { return get("ssm_dt"); }
    TensorPtr ssm_a() const { return get("ssm_a"); }
    TensorPtr ssm_alpha() const { return get("ssm_alpha"); }
    TensorPtr ssm_beta() const { return get("ssm_beta"); }
    TensorPtr ssm_norm() const { return get("ssm_norm"); }
    TensorPtr ssm_out() const { return get("ssm_out"); }

    // Gemma4 attention norms
    TensorPtr attn_post_norm() const { return get("attn_post_norm"); }

    // Gemma4 MoE weights
    TensorPtr ffn_gate_inp() const { return get("ffn_gate_inp"); }
    TensorPtr ffn_gate_inp_s() const { return get("ffn_gate_inp_s"); }
    TensorPtr ffn_gate_exps() const { return get("ffn_gate_exps"); }
    TensorPtr ffn_up_exps() const { return get("ffn_up_exps"); }
    TensorPtr ffn_down_exps() const { return get("ffn_down_exps"); }
    TensorPtr ffn_gate_up_exps() const { return get("ffn_gate_up_exps"); }
    TensorPtr ffn_pre_norm_2() const { return get("ffn_pre_norm_2"); }
    TensorPtr ffn_post_norm_1() const { return get("ffn_post_norm_1"); }
    TensorPtr ffn_post_norm_2() const { return get("ffn_post_norm_2"); }
    TensorPtr layer_out_scale() const { return get("layer_out_scale"); }

    // Gemma4 per-layer embeddings
    TensorPtr per_layer_inp_gate() const { return get("per_layer_inp_gate"); }
    TensorPtr per_layer_proj() const { return get("per_layer_proj"); }
    TensorPtr per_layer_post_norm() const { return get("per_layer_post_norm"); }

    // Gemma4 proportional RoPE frequency factors (per-layer, full-attention layers only)
    TensorPtr rope_freqs() const { return get("rope_freqs"); }
};

// Unified model-level weight container
struct ModelWeights {
    TensorPtr token_embedding;
    TensorPtr token_embedding_fp32;  // Pre-dequantized FP32 cache for CPU transposed embedding
    TensorPtr output_norm;
    TensorPtr output_weight;
    TensorPtr output_weight_fp32;  // Pre-dequantized FP32 cache for CPU output_proj
    std::vector<LayerWeights> layers;

    // Gemma4 per-layer embedding weights
    TensorPtr per_layer_tok_embd;
    TensorPtr per_layer_model_proj;
    TensorPtr per_layer_proj_norm;

    bool init(const WeightStore& store, const ModelConfig& config);

    // Move output-level weights to target device
    void move_output_weights(DeviceType target_dev);

    // Move a specific layer's weights to target device
    void move_layer_weights(int layer_idx, DeviceType target_dev);
};

// ============================================================================
// Weight Name Mapping System
// ============================================================================
// Maps raw weight names (from GGUF/ninf files) to canonical names used by
// the unified LayerWeights/ModelWeights structures.
// ============================================================================

struct WeightAlias {
    std::vector<std::string> names;
};

struct LayerWeightMapping {
    // Common fields
    WeightAlias attn_norm;
    WeightAlias ffn_norm;
    WeightAlias wo;
    WeightAlias gate_proj;
    WeightAlias down_proj;
    WeightAlias up_proj;

    // GQA attention
    WeightAlias wq;
    WeightAlias wk;
    WeightAlias wv;

    // MLA attention (DeepSeek V2/V3)
    WeightAlias wq_a;
    WeightAlias wq_b;
    WeightAlias kv_a_proj;
    WeightAlias kv_b_proj;

    // Qwen35 full attention
    WeightAlias attn_q;
    WeightAlias attn_k;
    WeightAlias attn_v;
    WeightAlias attn_output;
    WeightAlias attn_q_norm;
    WeightAlias attn_k_norm;
    WeightAlias post_attention_norm;

    // Qwen35 linear attention / SSM
    WeightAlias attn_qkv;
    WeightAlias attn_gate;
    WeightAlias ssm_conv1d;
    WeightAlias ssm_dt;
    WeightAlias ssm_a;
    WeightAlias ssm_alpha;
    WeightAlias ssm_beta;
    WeightAlias ssm_norm;
    WeightAlias ssm_out;
};

struct ArchWeightMapping {
    WeightAlias token_embedding;
    WeightAlias output_norm;
    WeightAlias output_weight;
    LayerWeightMapping layer;
    std::string layer_prefix_pattern = "model.layers.{}";
    bool tie_embeddings = false;
};

class WeightMapper {
public:
    static const ArchWeightMapping& get_mapping(const std::string& arch_type);
    static std::string format_layer_prefix(const std::string& pattern, int layer_idx);
    static TensorPtr resolve(const WeightStore& store, const WeightAlias& alias,
                             const std::string& prefix = "");
};

// ============================================================================
// Config Parser Registry
// ============================================================================
// Allows each architecture to register its own config parsing logic,
// replacing the monolithic if-else chain in parse_config_from_gguf.
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

#define FORGE_REGISTER_CONFIG_PARSER(arch, fn) FORGE_REGISTER_CONFIG_PARSER_IMPL(__LINE__, arch, fn)

// ============================================================================
// Weight Init Registry
// ============================================================================
// Allows each architecture to register its own layer weight loading logic,
// replacing the if-else chain in ModelWeights::init.
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

#define FORGE_REGISTER_WEIGHT_INIT(arch, fn) FORGE_REGISTER_WEIGHT_INIT_IMPL(__LINE__, arch, fn)

// ============================================================================
// Architecture Capability Description
// ============================================================================
// Describes the computational capabilities of an architecture, enabling
// automatic engine selection when no explicit engine is registered.
// ============================================================================

struct ArchCapability {
    bool use_gqa = false;
    bool use_mla = false;
    bool use_ssm = false;
    bool use_mrope = false;
    bool use_neox_rope = false;
    ActivationType ffn_activation = ActivationType::SiLU_GELU;
    NormType norm_type = NormType::RMSNorm;
    bool has_qkv_bias = false;
    bool has_norm_bias = false;
    bool use_parallel_residual = false;
    bool use_qk_norm = false;
    bool embedding_scale = false;
    bool has_post_attention_norm = false;
    bool has_post_ffn_norm = false;
};

class ArchCapabilityRegistry {
public:
    static ArchCapabilityRegistry& instance();

    void register_capability(const std::string& arch, const ArchCapability& cap);
    bool has(const std::string& arch) const;
    ArchCapability get(const std::string& arch) const;
    const std::unordered_map<std::string, ArchCapability>& all() const { return capabilities_; }

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
// Single macro to register all aspects of a new architecture in one place.
// This registers: engine, config parser, weight init, and architecture capability.
//
// Usage example (in a single .cpp file):
//   FORGE_REGISTER_ARCH("gemma",
//       /*engine=*/[](Model& m, InferenceContext& ctx) { return std::make_unique<GenericEngine>(m,
//       ctx); },
//       /*config_parser=*/parse_gemma_config,
//       /*weight_init=*/init_gqa_layer_weights,
//       /*capability=*/ArchCapability{.use_gqa = true, .use_neox_rope = true}
//   );
//
// For architectures without a dedicated engine (falls back to capability-based lookup):
//   FORGE_REGISTER_ARCH_NO_ENGINE("phi3",
//       parse_phi3_config,
//       init_gqa_layer_weights,
//       ArchCapability{.use_gqa = true, .use_neox_rope = true}
//   );
// ============================================================================

// Full registration (with engine)
#define FORGE_REGISTER_ARCH_IMPL2(line, arch, engine_creator, config_fn, weight_init_fn, cap) \
    static ::forge::EngineAutoRegister _engine_reg_##line(arch, engine_creator);              \
    static ::forge::ConfigParserAutoRegister _config_parser_reg_##line(arch, config_fn);      \
    static ::forge::WeightInitAutoRegister _weight_init_reg_##line(arch, weight_init_fn);     \
    static ::forge::ArchCapabilityAutoRegister _arch_cap_reg_##line(arch, cap)

#define FORGE_REGISTER_ARCH_IMPL(line, arch, engine_creator, config_fn, weight_init_fn, cap) \
    FORGE_REGISTER_ARCH_IMPL2(line, arch, engine_creator, config_fn, weight_init_fn, cap)

#define FORGE_REGISTER_ARCH(arch, engine_creator, config_fn, weight_init_fn, cap) \
    FORGE_REGISTER_ARCH_IMPL(__LINE__, arch, engine_creator, config_fn, weight_init_fn, cap)

// Registration without engine (for architectures that fall back to capability-based lookup)
#define FORGE_REGISTER_ARCH_NO_ENGINE_IMPL2(line, arch, config_fn, weight_init_fn, cap)  \
    static ::forge::ConfigParserAutoRegister _config_parser_reg_##line(arch, config_fn); \
    static ::forge::WeightInitAutoRegister _weight_init_reg_##line(arch, weight_init_fn);\
    static ::forge::ArchCapabilityAutoRegister _arch_cap_reg_##line(arch, cap)

#define FORGE_REGISTER_ARCH_NO_ENGINE_IMPL(line, arch, config_fn, weight_init_fn, cap) \
    FORGE_REGISTER_ARCH_NO_ENGINE_IMPL2(line, arch, config_fn, weight_init_fn, cap)

#define FORGE_REGISTER_ARCH_NO_ENGINE(arch, config_fn, weight_init_fn, cap) \
    FORGE_REGISTER_ARCH_NO_ENGINE_IMPL(__LINE__, arch, config_fn, weight_init_fn, cap)

class Model {
public:
    Model() = default;
    ~Model() = default;

    bool load(const std::string& model_path, DeviceType device = DeviceType::CUDA);
    bool load_with_config(const std::string& model_path, const ModelConfig& config,
                          DeviceType device = DeviceType::CUDA);

    // Load vision/mmproj weights from a separate GGUF file (e.g., mmproj-model-f16.gguf)
    bool load_vision_weights(const std::string& mmproj_path, DeviceType device = DeviceType::CPU);

    const ModelConfig& config() const { return config_; }
    const WeightStore& weights() const { return weight_store_; }
    WeightStore& weights() { return weight_store_; }
    DeviceType device() const { return device_; }
    const std::string& path() const { return model_path_; }
    const std::string& format() const { return format_name_; }
    bool is_loaded() const { return is_loaded_; }

    void set_config(const ModelConfig& config);
    void set_device(DeviceType device);
    void set_quant_policy(const QuantPolicy& policy);

    TensorPtr get_weight(const std::string& name) const;

    static std::string detect_format(const std::string& path);

private:
    bool load_from_loader(ModelLoader& loader, DeviceType device);
    ModelConfig parse_config_from_gguf(ModelLoader& loader);
    ModelConfig parse_config_from_ninf(ModelLoader& loader);

    ModelConfig config_;
    WeightStore weight_store_;
    DeviceType device_ = DeviceType::CPU;
    QuantPolicy quant_policy_;
    std::string model_path_;
    std::string format_name_;
    bool is_loaded_ = false;

    // Keep loader alive so mmap'd data remains valid for zero-copy tensors
    ModelLoaderPtr loader_;
    ModelLoaderPtr vision_loader_;  // For mmproj/vision weights
};

}  // namespace forge
