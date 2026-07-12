#include "forge/model_weights.h"

#include <chrono>

#include "forge/arch_registry.h"
#include "forge/logger.h"
#include "forge/model_config.h"
#include "forge/operator_matmul.h"
#include "forge/weight_mapper.h"

namespace forge {

void LayerWeights::to_device(DeviceType device) {
    for (auto& [name, t] : weights) {
        if (t && t->device() != device) {
            t->to_device(device);
        }
    }
}

bool ModelWeights::init(const WeightStore& store, const ModelConfig& config) {
    token_embedding = store.get("token_embedding");
    if (!token_embedding) {
        LOG_ERROR("ModelWeights::init: token_embedding not found");
        return false;
    }

    output_norm = store.get("output_norm");
    output_weight = store.get("output_weight");

    if (config.tie_embeddings && !output_weight) {
        output_weight = token_embedding;
    }

    if (output_weight && output_weight->device() == DeviceType::CPU) {
        bool has_fused_kernel =
            (output_weight->dtype() == DataType::Q4_0 || output_weight->dtype() == DataType::Q8_0 ||
             output_weight->dtype() == DataType::Q4_1 || output_weight->dtype() == DataType::Q4_K);
        if (has_fused_kernel) {
            LOG_INFO("Output_weight is " +
                     std::to_string(static_cast<int>(output_weight->dtype())) + " (" +
                     std::to_string(output_weight->shape()[0]) + "x" +
                     std::to_string(output_weight->shape()[1]) + "), using fused GEMV directly");
        } else {
            TensorPtr fp32_weight = output_weight;
            if (is_quantized_type(output_weight->dtype())) {
                fp32_weight = ops::dequantize_weight(output_weight);
            }
            auto t_start = std::chrono::high_resolution_clock::now();
            auto q8 = ops::quantize_q8_0_weight(fp32_weight);
            auto t_end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            if (q8) {
                output_weight = q8;
                LOG_INFO("Quantized output_weight to Q8_0 (" +
                         std::to_string(output_weight->shape()[0]) + "x" +
                         std::to_string(output_weight->shape()[1]) +
                         "), size=" + std::to_string(q8->nbytes() / (1024 * 1024)) +
                         " MB, time=" + std::to_string(ms / 1000.0) + "s");
            }
        }
    }

    if (token_embedding && is_quantized_type(token_embedding->dtype()) &&
        token_embedding->device() == DeviceType::CPU) {
        bool transposed = (token_embedding->shape().size() >= 2 &&
                           token_embedding->shape()[0] < token_embedding->shape()[1]);
        if (transposed) {
            auto t_start = std::chrono::high_resolution_clock::now();
            token_embedding_fp32 = ops::dequantize_weight(token_embedding);
            auto t_end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            if (token_embedding_fp32) {
                LOG_INFO("Pre-dequantized token_embedding (" +
                         std::to_string(token_embedding->shape()[0]) + "x" +
                         std::to_string(token_embedding->shape()[1]) + ") to FP32, size=" +
                         std::to_string(token_embedding_fp32->nbytes() / (1024 * 1024)) +
                         " MB, time=" + std::to_string(ms / 1000.0) + "s");
            }
        }
    }

    layers.resize(config.num_layers);

    bool is_qwen35 = (config.arch_type == "qwen35");
    int full_attn_interval =
        config.full_attention_interval > 0 ? config.full_attention_interval : 4;

    auto& weight_init_registry = WeightInitRegistry::instance();
    bool has_registered_init = weight_init_registry.has(config.arch_type);

    for (int i = 0; i < config.num_layers; ++i) {
        auto& lw = layers[i];

        if (is_qwen35) {
            bool is_full_attn = ((i + 1) % full_attn_interval == 0);
            std::string base = "layers." + std::to_string(i);
            if (store.has(base + ".attn_q")) {
                is_full_attn = true;
            }
            lw.layer_type = is_full_attn ? LayerType::FullAttention : LayerType::LinearAttention;
        } else if (config.use_mla) {
            lw.layer_type = LayerType::MLA;
        } else {
            lw.layer_type = LayerType::FullAttention;
        }

        if (has_registered_init) {
            LayerWeightInitContext ctx{store, config, i, lw};
            weight_init_registry.init_layer(config.arch_type, ctx);
        } else {
            std::string base = "layers." + std::to_string(i);
            auto load_if = [&](const std::string& canonical, const std::string& store_name) {
                auto t = store.get(store_name);
                if (t)
                    lw.set(canonical, t);
            };

            load_if("attn_norm", base + ".attn_norm");
            load_if("ffn_norm", base + ".ffn_norm");
            load_if("w1", base + ".gate_proj");
            load_if("w2", base + ".down_proj");
            load_if("w3", base + ".up_proj");
            load_if("wq", base + ".wq");
            load_if("wk", base + ".wk");
            load_if("wv", base + ".wv");
            load_if("wo", base + ".wo");
            load_if("bq", base + ".bq");
            load_if("bk", base + ".bk");
            load_if("bv", base + ".bv");

            LOG_WARN(
                "No registered weight init for arch '" + config.arch_type +
                "', using default GQA init. Consider registering via FORGE_REGISTER_WEIGHT_INIT.");
        }

        if (lw.layer_type == LayerType::FullAttention && !is_qwen35) {
            if (!lw.wq() || !lw.wk() || !lw.wv() || !lw.wo()) {
                LOG_ERROR("Missing attention weights for layer " + std::to_string(i));
                return false;
            }
        } else if (lw.layer_type == LayerType::MLA) {
            if (!lw.wo() || !lw.kv_a_proj()) {
                LOG_ERROR("Missing MLA attention weights for layer " + std::to_string(i));
                return false;
            }
        }
    }

    return true;
}

void ModelWeights::move_output_weights(DeviceType target_dev) {
    if (output_norm && output_norm->device() != target_dev) {
        output_norm->to_device(target_dev);
    }
    if (output_weight && output_weight->device() != target_dev) {
        if (output_weight.get() != token_embedding.get()) {
            output_weight->to_device(target_dev);
        }
    }
}

void ModelWeights::move_layer_weights(int layer_idx, DeviceType target_dev) {
    layers[layer_idx].to_device(target_dev);
}

}  // namespace forge
