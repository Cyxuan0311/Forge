#pragma once

#include <string>
#include <cstdint>

namespace nanoinfer {

enum class NormType : int { RMSNorm, LayerNorm };
enum class ActivationType : int { SiLU_GELU, GELU, ReLU, GeGLU };
enum class RopeType : int { None, Standard, LinearScaling, NTK_Scaled };

enum class LayerType : int {
    FullAttention = 0,
    LinearAttention = 1,
    MLA = 2,
};

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

    int ssm_group_count = 0;
    int ssm_time_step_rank = 0;
    int ssm_inner_size = 0;
    int ssm_state_size = 0;
    int ssm_conv_kernel = 0;
    int full_attention_interval = 0;
    bool use_ssm = false;

    int rope_dimension_count = 0;
    int rope_dimension_sections[4] = {0, 0, 0, 0};
    bool use_mrope = false;

    bool use_parallel_residual = false;

    float f_attn_logit_softcapping = 0.0f;
    float f_final_logit_softcapping = 0.0f;
};

} // namespace nanoinfer
