#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include "tensor.h"
#include "types.h"
#include "model_config.h"
#include "weight_store.h"

namespace forge {

struct LayerWeights {
    std::unordered_map<std::string, TensorPtr> weights;

    LayerType layer_type = LayerType::FullAttention;

    TensorPtr get(const std::string& name) const {
        auto it = weights.find(name);
        return it != weights.end() ? it->second : nullptr;
    }

    void set(const std::string& name, TensorPtr tensor) {
        weights[name] = std::move(tensor);
    }

    bool has(const std::string& name) const {
        return weights.find(name) != weights.end();
    }

    void to_device(DeviceType device);

    // Common accessors
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
};

struct ModelWeights {
    TensorPtr token_embedding;
    TensorPtr token_embedding_fp32;
    TensorPtr output_norm;
    TensorPtr output_weight;
    TensorPtr output_weight_fp32;
    std::vector<LayerWeights> layers;

    bool init(const WeightStore& store, const ModelConfig& config);

    void move_output_weights(DeviceType target_dev);
    void move_layer_weights(int layer_idx, DeviceType target_dev);
};

} // namespace forge
