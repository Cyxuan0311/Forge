#include "forge/model_weights.h"

#include <chrono>
#include <cstring>

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

// DSpark (vLLM-style) markov head stores markov_w1 / markov_w2 in GGUF as
// [d, vocab]: dims[0] = d (outer) = 256, dims[1] = vocab (inner) = 130560, so
// token v's d-vector is COLUMN v in the raw bytes. forge's get_tensor reverses
// the shape vector (-> [vocab, d] = [130560, 256]) but NOT the bytes, so the
// loaded tensor has scrambled rows. dspark_sequential_sample needs
// markov_embed[prev] / markov_bias[v] as contiguous d-vectors, so we transpose to
// true [vocab, d] row-major.
//
// The head is consumed on CPU only, as FP32. It is loaded with the drafter's
// layer device, which for a BF16 drafter is CUDA -- and the loader keeps BF16
// weights native there (expanding them doubled the drafter's footprint and cost
// 19.5x on its weight-reading GEMMs). So the source may arrive as BF16 and is
// widened to FP32 here. This runs once at load, on CPU, for two [130560, 256]
// tables; correctness beats the memory here because the head never sees a GEMM.
// CPU-side FP32 view of a drafter head tensor. A BF16 GGUF weight arrives in its
// native form when loaded onto a CUDA device (the loader no longer expands it --
// see GgufModel::get_tensor), and the DSpark heads are read as float on the CPU.
static TensorPtr to_cpu_f32(const TensorPtr& src) {
    if (!src)
        return nullptr;
    if (src->device() == DeviceType::CPU && src->dtype() == DataType::FP32)
        return src;
    const int64_t numel = src->numel();
    auto out = std::make_shared<Tensor>(DataType::FP32, src->shape(), DeviceType::CPU);
    float* o = static_cast<float*>(out->data());
    if (src->dtype() == DataType::BF16) {
        const auto* in = static_cast<const uint16_t*>(src->data());
        for (int64_t i = 0; i < numel; ++i) {
            const uint32_t bits = static_cast<uint32_t>(in[i]) << 16;
            std::memcpy(o + i, &bits, sizeof(float));
        }
        return out;
    }
    if (src->dtype() != DataType::FP32)
        throw std::runtime_error("to_cpu_f32: expected FP32 or BF16, got '" +
                                 dtype_name(src->dtype()) + "'");
    // FP32 on another device: stage through a device copy.
    auto staged = std::make_shared<Tensor>(DataType::FP32, src->shape(), DeviceType::CPU);
    staged->copy_from(*src);
    return staged;
}

static TensorPtr transpose_markov_head(const TensorPtr& src) {
    if (!src)
        return nullptr;
    TensorPtr s = src;
    if (s->device() != DeviceType::CPU)
        s->to_device(DeviceType::CPU);  // in-place: head is CPU-only
    const auto& shp = s->shape();
    if (shp.size() != 2)
        return nullptr;
    // forge shape after reverse = [vocab, d] = [shp[0], shp[1]]
    const int64_t vocab = shp[0], D = shp[1];
    auto out = std::make_shared<Tensor>(DataType::FP32,
                                        std::vector<int64_t>{vocab, D}, DeviceType::CPU);
    const int64_t total = vocab * D;
    float* o = static_cast<float*>(out->data());
    if (s->dtype() == DataType::BF16) {
        const auto* in = static_cast<const uint16_t*>(s->data());
        for (int64_t v = 0; v < vocab; ++v)
            for (int64_t d = 0; d < D; ++d) {
                const uint32_t bits = static_cast<uint32_t>(in[d * vocab + v]) << 16;
                std::memcpy(o + v * D + d, &bits, sizeof(float));
            }
        (void)total;
        return out;
    }
    if (s->dtype() != DataType::FP32)
        throw std::runtime_error("transpose_markov_head: expected FP32 or BF16 head, got '" +
                                 dtype_name(s->dtype()) + "'");
    const float* in = static_cast<const float*>(s->data());
    // out[v, d] = in[d, v]  (raw GGUF [d, vocab] -> row-major [vocab, d])
    for (int64_t v = 0; v < vocab; ++v)
        for (int64_t d = 0; d < D; ++d)
            o[v * D + d] = in[d * vocab + v];
    return out;
}

bool ModelWeights::init(const WeightStore& store, const ModelConfig& config) {
    token_embedding = store.get("token_embedding");
    if (!token_embedding) {
        // DFlash / DSPark borrow the target model's token embedding at runtime,
        // so its absence is expected for these architectures.
        if (config.arch_type != "dflash" && config.arch_type != "dspark") {
            LOG_ERROR("ModelWeights::init: token_embedding not found");
            return false;
        }
        LOG_INFO("ModelWeights::init: '" + config.arch_type +
                 "' has no token_embedding (shared with target); borrowing at runtime");
    }

    output_norm = store.get("output_norm");
    output_norm_bias = store.get("output_norm_bias");
    output_weight = store.get("output_weight");
    output_bias = store.get("output_bias");

    if (config.tie_embeddings && !output_weight) {
        output_weight = token_embedding;
    }

    // Gemma4 per-layer embedding weights
    per_layer_tok_embd = store.get("per_layer_tok_embd");
    per_layer_model_proj = store.get("per_layer_model_proj");
    per_layer_proj_norm = store.get("per_layer_proj_norm");

    // DFlash / DSPark encoder weights (shared token_embd / lm_head come from target)
    dflash_fc = store.get("fc");
    if (!dflash_fc)
        dflash_fc = store.get("fc.weight");
    if (!dflash_fc)
        dflash_fc = store.get("model.fcs.0");
    dflash_output_norm_enc = store.get("output_norm_enc");
    if (!dflash_output_norm_enc)
        dflash_output_norm_enc = store.get("output_norm_enc.weight");

    // DSPark sequential Markov head (optional; absent for plain DFlash).
    // Two naming conventions are supported:
    //  - forge's older Qwen DFlash2 checkpoint: `markov_embed` / `markov_bias`
    //    already in [target_vocab, d] row-major form.
    //  - vLLM-style DSpark (e.g. MiniCPM5-2.6B-DSpark): `markov_w1`
    //    (Embedding[vocab, d], keyed by prev token) and `markov_w2`
    //    (LMHead[vocab, d], keyed by candidate v), stored in GGUF as [d, vocab];
    //    we transpose them to [vocab, d] so the markov walk can index
    //    markov_embed[prev] / markov_bias[v] as contiguous d-vectors.
    //    (vLLM qwen3_dspark.py:71-72: markov_w1 -> embed, markov_w2 -> bias.)
    dflash_markov_embed = store.get("markov_embed");
    if (!dflash_markov_embed)
        dflash_markov_embed = store.get("markov_embed.weight");
    dflash_markov_bias = store.get("markov_bias");
    if (!dflash_markov_bias)
        dflash_markov_bias = store.get("markov_bias.weight");
    if (!dflash_markov_embed || !dflash_markov_bias) {
        // vLLM DSpark (qwen3_dspark.py:71-72): markov_w1 = Embedding[vocab, d]
        // (prev-keyed embed), markov_w2 = LMHead[vocab, d] (candidate-keyed bias).
        // The GGUF loader already reverses dims to [vocab, d] row-major and converts
        // BF16 -> FP32, so we use them directly. The head is consumed on CPU only,
        // so stage it to CPU once (the drafter itself may live on GPU).
        TensorPtr mw1 = store.get("markov_w1");
        if (!mw1)
            mw1 = store.get("markov_w1.weight");
        TensorPtr mw2 = store.get("markov_w2");
        if (!mw2)
            mw2 = store.get("markov_w2.weight");
        if (mw1 && mw2) {
            // vLLM qwen3_dspark.py:71-72: markov_w1 = Embedding[vocab, d] (prev-keyed
            // embed), markov_w2 = LMHead[vocab, d] (candidate-keyed bias). GGUF stores
            // them as [d, vocab]; transpose to true [vocab, d] row-major so
            // markov_embed[prev] / markov_bias[v] are contiguous d-vectors. The head is
            // consumed on CPU, so transpose_markov_head also stages it to CPU.
            dflash_markov_embed = transpose_markov_head(mw1);  // prev-keyed
            dflash_markov_bias = transpose_markov_head(mw2);   // candidate-keyed
        }
    }
    dflash_draft_id_to_target_id = store.get("draft_id_to_target_id");

    // DSpark per-position confidence head (optional; requires has_confidence_head).
    // conf_proj.weight is [conf_in_dim, 1] (conf_in_dim = hidden_dim + markov_d); it
    // is consumed on CPU only, so the engine accessor stages it to CPU on demand.
    dflash_conf_proj_weight = store.get("conf_proj.weight");
    if (!dflash_conf_proj_weight)
        dflash_conf_proj_weight = store.get("confidence_head.proj.weight");
    dflash_conf_proj_bias = store.get("conf_proj.bias");
    if (!dflash_conf_proj_bias)
        dflash_conf_proj_bias = store.get("confidence_head.proj.bias");
    // dspark_sequential_sample reads the confidence projection as float, and it
    // is loaded with the drafter's layer device -- which for a BF16 drafter keeps
    // the weight in native BF16. Widen once here (a [conf_in_dim, 1] vector, so
    // this is free) so the engine accessor can hand out a plain CPU FP32 tensor.
    dflash_conf_proj_weight = to_cpu_f32(dflash_conf_proj_weight);
    dflash_conf_proj_bias = to_cpu_f32(dflash_conf_proj_bias);

    if (dflash_markov_embed && dflash_markov_bias) {
        LOG_INFO("DSpark markov head loaded: embed=" +
                 std::to_string(dflash_markov_embed->shape()[0]) + "x" +
                 std::to_string(dflash_markov_embed->shape()[1]) + " bias=" +
                 std::to_string(dflash_markov_bias->shape()[0]) + "x" +
                 std::to_string(dflash_markov_bias->shape()[1]) +
                 " confidence=" + (dflash_conf_proj_weight ? "yes" : "no"));
    } else {
        LOG_INFO("DSpark markov head NOT loaded");
    }

    // DFlash2 selector weights (present only for DFlash2 GGUFs).
    dflash_selector_prev = store.get("dflash_selector_prev");
    dflash_selector_next = store.get("dflash_selector_next");
    dflash_selector_hidden = store.get("dflash_selector_hidden");

    // Q8_0 re-quantization is a decode-GEMV optimization for LM heads (large
    // vocab, M==1 rows). Embedding models project to n_embd_out with
    // return_all_logits (M = seq_len > 1) and need the full-precision result,
    // so skip it there -- the Q8_0 prefill path is both lossy and, for this
    // shape, wrong.
    if (output_weight && output_weight->device() == DeviceType::CPU &&
        config.n_embd_out == 0) {
        bool has_fused_kernel =
            (output_weight->dtype() == DataType::Q4_0 || output_weight->dtype() == DataType::Q8_0 ||
             output_weight->dtype() == DataType::Q4_1 || output_weight->dtype() == DataType::Q4_K ||
             output_weight->dtype() == DataType::Q6_K);
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
        } else if (config.arch_type == "bailingmoe3") {
            // Ling 3.0: head_count_kv[i]==0 => GDN recurrent, >0 => absorbed MLA
            bool is_recurrent = (i < (int)config.head_count_kv.size()) &&
                                config.head_count_kv[i] == 0;
            lw.layer_type = is_recurrent ? LayerType::LinearAttention : LayerType::MLA;
        } else if (config.arch_type == "lfm2") {
            // LiquidAI LFM2: head_count_kv[i]==0 => shortconv operator layer,
            // >0 => GQA full attention. The tensor presence check covers GGUFs
            // whose head_count_kv is scalar (parse_lfm2_config then fills the
            // array with the scalar for every layer).
            std::string lbase = "layers." + std::to_string(i);
            bool is_conv = (i < (int)config.head_count_kv.size() &&
                            config.head_count_kv[i] == 0) ||
                           store.has(lbase + ".shortconv_in_proj");
            lw.layer_type = is_conv ? LayerType::ShortConv : LayerType::FullAttention;
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
            bool is_gemma4 = (config.arch_type == "gemma4" ||
                              config.arch_type == "gemma-embedding2");
            bool is_lfm2 = (config.arch_type == "lfm2");
            // Gemma4: wv is optional (may use wk as fallback);
            // lfm2 attention layers use the canonical attn_* names.
            if (is_lfm2) {
                if (!lw.attn_q() || !lw.attn_k() || !lw.attn_v() || !lw.attn_output()) {
                    LOG_ERROR("Missing attention weights for layer " + std::to_string(i));
                    return false;
                }
            } else if (!lw.wq() || !lw.wk() || (!is_gemma4 && !lw.wv()) || !lw.wo()) {
                LOG_ERROR("Missing attention weights for layer " + std::to_string(i));
                return false;
            }
        } else if (lw.layer_type == LayerType::ShortConv) {
            if (!lw.shortconv_conv() || !lw.shortconv_in_proj() || !lw.shortconv_out_proj()) {
                LOG_ERROR("Missing shortconv weights for layer " + std::to_string(i));
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
    if (output_norm_bias && output_norm_bias->device() != target_dev) {
        output_norm_bias->to_device(target_dev);
    }
    if (output_weight && output_weight->device() != target_dev) {
        if (output_weight.get() != token_embedding.get()) {
            output_weight->to_device(target_dev);
        }
    }
    if (output_bias && output_bias->device() != target_dev) {
        output_bias->to_device(target_dev);
    }
}

void ModelWeights::move_layer_weights(int layer_idx, DeviceType target_dev) {
    layers[layer_idx].to_device(target_dev);
}

}  // namespace forge
