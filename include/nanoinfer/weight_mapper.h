#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include "tensor.h"
#include "types.h"
#include "weight_store.h"
#include "model_config.h"

namespace nanoinfer {

struct WeightAlias {
    std::vector<std::string> names;
};

struct LayerWeightMapping {
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

} // namespace nanoinfer
