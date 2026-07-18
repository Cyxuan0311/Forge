#include "forge/weight_mapper.h"

#include <unordered_map>

namespace forge {

// ============================================================================
// Architecture weight name mappings
// ============================================================================

static const ArchWeightMapping llama_mapping = {
    .token_embedding = {{"model.embed_tokens.weight", "model.embedding.weight"}},
    .output_norm = {{"model.norm.weight", "model.ln_f.weight"}},
    .output_weight = {{"lm_head.weight", "model.output.weight"}},
    .layer =
        {
            .attn_norm = {{"input_layernorm.weight", "attention_norm.weight"}},
            .ffn_norm = {{"post_attention_layernorm.weight", "ffn_norm.weight"}},
            .wo = {{"self_attn.o_proj.weight", "attention.wo.weight"}},
            .gate_proj = {{"mlp.gate_proj.weight", "feed_forward.w1.weight"}},
            .down_proj = {{"mlp.down_proj.weight", "feed_forward.w2.weight"}},
            .up_proj = {{"mlp.up_proj.weight", "feed_forward.w3.weight"}},
            .wq = {{"self_attn.q_proj.weight", "attention.wq.weight"}},
            .wk = {{"self_attn.k_proj.weight", "attention.wk.weight"}},
            .wv = {{"self_attn.v_proj.weight", "attention.wv.weight"}},
        },
    .layer_prefix_pattern = "model.layers.{}",
    .tie_embeddings = false,
};

static const ArchWeightMapping mistral_mapping = {
    .token_embedding = {{"model.embed_tokens.weight"}},
    .output_norm = {{"model.norm.weight"}},
    .output_weight = {{"lm_head.weight"}},
    .layer =
        {
            .attn_norm = {{"input_layernorm.weight"}},
            .ffn_norm = {{"post_attention_layernorm.weight"}},
            .wo = {{"self_attn.o_proj.weight"}},
            .gate_proj = {{"mlp.gate_proj.weight"}},
            .down_proj = {{"mlp.down_proj.weight"}},
            .up_proj = {{"mlp.up_proj.weight"}},
            .wq = {{"self_attn.q_proj.weight"}},
            .wk = {{"self_attn.k_proj.weight"}},
            .wv = {{"self_attn.v_proj.weight"}},
        },
    .layer_prefix_pattern = "model.layers.{}",
    .tie_embeddings = false,
};

static const ArchWeightMapping qwen_mapping = {
    .token_embedding = {{"model.embed_tokens.weight"}},
    .output_norm = {{"model.norm.weight"}},
    .output_weight = {{"lm_head.weight"}},
    .layer =
        {
            .attn_norm = {{"input_layernorm.weight"}},
            .ffn_norm = {{"post_attention_layernorm.weight"}},
            .wo = {{"self_attn.o_proj.weight"}},
            .gate_proj = {{"mlp.gate_proj.weight"}},
            .down_proj = {{"mlp.down_proj.weight"}},
            .up_proj = {{"mlp.up_proj.weight"}},
            .wq = {{"self_attn.q_proj.weight"}},
            .wk = {{"self_attn.k_proj.weight"}},
            .wv = {{"self_attn.v_proj.weight"}},
        },
    .layer_prefix_pattern = "model.layers.{}",
    .tie_embeddings = false,
};

static const ArchWeightMapping deepseek_v2_mapping = {
    .token_embedding = {{"model.embed_tokens.weight"}},
    .output_norm = {{"model.norm.weight"}},
    .output_weight = {{"lm_head.weight"}},
    .layer =
        {
            .attn_norm = {{"input_layernorm.weight"}},
            .ffn_norm = {{"post_attention_layernorm.weight"}},
            .wo = {{"self_attn.o_proj.weight"}},
            .gate_proj = {{"mlp.gate_proj.weight"}},
            .down_proj = {{"mlp.down_proj.weight"}},
            .up_proj = {{"mlp.up_proj.weight"}},
            .wq_a = {{"self_attn.q_a_proj.weight"}},
            .wq_b = {{"self_attn.q_b_proj.weight"}},
            .kv_a_proj = {{"self_attn.kv_a_proj_with_mqa.weight"}},
            .kv_b_proj = {{"self_attn.kv_b_proj.weight"}},
        },
    .layer_prefix_pattern = "model.layers.{}",
    .tie_embeddings = false,
};

static const ArchWeightMapping qwen35_mapping = {
    .token_embedding = {{"model.embed_tokens.weight"}},
    .output_norm = {{"model.norm.weight"}},
    .output_weight = {{"lm_head.weight"}},
    .layer =
        {
            .attn_norm = {{"input_layernorm.weight"}},
            .ffn_norm = {{"post_attention_layernorm.weight"}},
            .wo = {{"self_attn.o_proj.weight"}},
            .gate_proj = {{"mlp.gate_proj.weight"}},
            .down_proj = {{"mlp.down_proj.weight"}},
            .up_proj = {{"mlp.up_proj.weight"}},
            .attn_q = {{"self_attn.q_proj.weight"}},
            .attn_k = {{"self_attn.k_proj.weight"}},
            .attn_v = {{"self_attn.v_proj.weight"}},
            .attn_output = {{"self_attn.o_proj.weight"}},
            .attn_q_norm = {{"self_attn.q_norm.weight"}},
            .attn_k_norm = {{"self_attn.k_norm.weight"}},
            .post_attention_norm = {{"post_attention_layernorm.weight"}},
            .attn_qkv = {{"self_attn.qkv_proj.weight"}},
            .attn_gate = {{"self_attn.gate.weight"}},
            .ssm_conv1d = {{"ssm.conv1d.weight"}},
            .ssm_dt = {{"ssm.dt.bias"}},
            .ssm_a = {{"ssm.a"}},
            .ssm_alpha = {{"ssm.alpha.weight"}},
            .ssm_beta = {{"ssm.beta.weight"}},
            .ssm_norm = {{"ssm.norm.weight"}},
            .ssm_out = {{"ssm.output.weight"}},
        },
    .layer_prefix_pattern = "model.layers.{}",
    .tie_embeddings = false,
};

static const std::unordered_map<std::string, const ArchWeightMapping*> arch_mappings = {
    {"llama", &llama_mapping},
    {"mistral", &mistral_mapping},
    {"qwen", &qwen_mapping},
    {"qwen2", &qwen_mapping},
    {"qwen3vl", &qwen_mapping},
    {"yi", &llama_mapping},
    {"deepseek", &llama_mapping},
    {"deepseek_v2", &deepseek_v2_mapping},
    {"deepseek_v3", &deepseek_v2_mapping},
    {"qwen35", &qwen35_mapping},
    {"gemma4", &llama_mapping},  // Gemma4 uses GGUF naming, not HuggingFace
};

const ArchWeightMapping& WeightMapper::get_mapping(const std::string& arch_type) {
    auto it = arch_mappings.find(arch_type);
    if (it != arch_mappings.end())
        return *it->second;
    return llama_mapping;
}

std::string WeightMapper::format_layer_prefix(const std::string& pattern, int layer_idx) {
    std::string result = pattern;
    auto pos = result.find("{}");
    if (pos != std::string::npos) {
        result.replace(pos, 2, std::to_string(layer_idx));
    }
    return result;
}

TensorPtr WeightMapper::resolve(const WeightStore& store, const WeightAlias& alias,
                                const std::string& prefix) {
    for (const auto& name : alias.names) {
        std::string full_name = prefix.empty() ? name : (prefix + "." + name);
        auto t = store.get(full_name);
        if (t)
            return t;
    }
    return nullptr;
}

}  // namespace forge
