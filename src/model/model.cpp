#include "forge/model.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "forge/gguf_model.h"
#include "forge/logger.h"
#include "forge/ninf_model.h"
#include "forge/operator_matmul.h"
#include "forge/types.h"

namespace forge {

static TensorPtr inverse_neox_permute_rows(const TensorPtr& tensor, int num_heads, int head_dim) {
    if (!tensor || tensor->ndim() != 2)
        return tensor;

    int64_t nrows = tensor->shape()[0];
    int64_t ncols = tensor->shape()[1];
    int half_dim = head_dim / 2;

    if (nrows != num_heads * head_dim)
        return tensor;

    bool was_cuda = (tensor->device() == DeviceType::CUDA);
    TensorPtr work = tensor;
    if (was_cuda) {
        work = std::make_shared<Tensor>(tensor->dtype(), tensor->shape(), DeviceType::CPU);
        work->copy_from(*tensor);
    }

    auto result = std::make_shared<Tensor>(work->dtype(), work->shape(), DeviceType::CPU);

    size_t row_bytes = work->nbytes() / nrows;
    const uint8_t* src = static_cast<const uint8_t*>(work->data());
    uint8_t* dst = static_cast<uint8_t*>(result->data());

    for (int64_t q = 0; q < nrows; ++q) {
        int head = static_cast<int>(q / head_dim);
        int dim_in_head = static_cast<int>(q % head_dim);
        int p;
        if (dim_in_head < half_dim) {
            p = head * head_dim + 2 * dim_in_head;
        } else {
            p = head * head_dim + 2 * (dim_in_head - half_dim) + 1;
        }
        std::memcpy(dst + q * row_bytes, src + p * row_bytes, row_bytes);
    }

    if (was_cuda) {
        result->to_device(DeviceType::CUDA);
    }

    return result;
}

void WeightStore::set(const std::string& name, TensorPtr tensor) {
    weights_[name] = std::move(tensor);
}

TensorPtr WeightStore::get(const std::string& name) const {
    auto it = weights_.find(name);
    if (it == weights_.end())
        return nullptr;
    return it->second;
}

TensorPtr WeightStore::get_or_null(const std::string& name) const {
    return get(name);
}

bool WeightStore::has(const std::string& name) const {
    return weights_.find(name) != weights_.end();
}

const std::unordered_map<std::string, TensorPtr>& WeightStore::all() const {
    return weights_;
}

size_t WeightStore::size() const {
    return weights_.size();
}

void WeightStore::clear() {
    weights_.clear();
}

size_t WeightStore::total_bytes() const {
    size_t total = 0;
    for (const auto& [_, t] : weights_) {
        if (t)
            total += t->nbytes();
    }
    return total;
}

std::vector<std::string> WeightStore::weight_names() const {
    std::vector<std::string> names;
    names.reserve(weights_.size());
    for (const auto& [name, _] : weights_) {
        names.push_back(name);
    }
    return names;
}

void WeightStore::to_device(DeviceType device) {
    for (auto& [_, t] : weights_) {
        if (t)
            t->to_device(device);
    }
}

void WeightStore::to_device_layer(int layer_idx, DeviceType device,
                                  const std::string& prefix_pattern) {
    std::string prefix = WeightMapper::format_layer_prefix(prefix_pattern, layer_idx);
    std::string base = "layers." + std::to_string(layer_idx);
    for (auto& [name, t] : weights_) {
        if (t && (name.find(base) == 0 || name.find(prefix) == 0)) {
            t->to_device(device);
        }
    }
}

// ============================================================================
// LayerWeights implementation
// ============================================================================
void LayerWeights::to_device(DeviceType device) {
    for (auto& [name, t] : weights) {
        if (t && t->device() != device) {
            t->to_device(device);
        }
    }
}

// ============================================================================
// ModelWeights implementation
// ============================================================================
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

    // Optimize output_weight for CPU inference using fused quantized GEMV kernels.
    // Strategy:
    //   - Q4_0/Q8_0: already have fast fused GEMV kernels, use directly
    //   - Q6_K/Q4_K/other: no fused kernel, quantize to Q8_0 for ~3x speedup over dequant+gemv
    //   - FP32/FP16: quantize to Q8_0 for ~3x memory bandwidth reduction
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
            // No fused GEMV kernel for this dtype — quantize to Q8_0
            // First dequantize to FP32 if quantized, then quantize to Q8_0
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

    // Pre-dequantize token_embedding to FP32 for CPU inference (transposed layout)
    // Transposed embedding [embed_dim, vocab_size] requires full dequant for row lookup,
    // which is very expensive per decode step. Cache the FP32 version once.
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

    // Determine layer types based on architecture
    bool is_qwen35 = (config/*arch_type=*/= "qwen35");
    int full_attn_interval =
        config.full_attention_interval > 0 ? config.full_attention_interval : 4;

    auto& weight_init_registry = WeightInitRegistry::instance();
    bool has_registered_init = weight_init_registry.has(config.arch_type);

    for (int i = 0; i < config.num_layers; ++i) {
        auto& lw = layers[i];

        // Determine layer type
        if (is_qwen35) {
            bool is_full_attn = ((i + 1) % full_attn_interval == 0);
            // Also check by weight presence (more robust)
            std::string base = "layers." + std::to_string(i);
            if (store.has(base + ".attn_q")) {
                is_full_attn = true;
            }
            lw/*layer_type=*/ is_full_attn ? LayerType::FullAttention : LayerType::LinearAttention;
        } else if (config.use_mla) {
            lw/*layer_type=*/ LayerType::MLA;
        } else {
            lw/*layer_type=*/ LayerType::FullAttention;
        }

        // Use registered weight init if available, otherwise fallback to inline logic
        if (has_registered_init) {
            LayerWeightInitContext ctx{store, config, i, lw};
            weight_init_registry.init_layer(config.arch_type, ctx);
        } else {
            // Fallback: load common weights inline for unregistered architectures
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

        // Validate required weights
        if (lw/*layer_type=*/= LayerType::FullAttention && !is_qwen35) {
            if (!lw.wq() || !lw.wk() || !lw.wv() || !lw.wo()) {
                LOG_ERROR("Missing attention weights for layer " + std::to_string(i));
                return false;
            }
        } else if (lw/*layer_type=*/= LayerType::MLA) {
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
        // Don't move if tied to token_embedding (shared pointer)
        if (output_weight.get() != token_embedding.get()) {
            output_weight->to_device(target_dev);
        }
    }
}

void ModelWeights::move_layer_weights(int layer_idx, DeviceType target_dev) {
    layers[layer_idx].to_device(target_dev);
}

std::string Model::detect_format(const std::string& path) {
    auto loader = ModelLoaderRegistry::instance().create_loader(path);
    if (loader)
        return loader->format_name();
    return "";
}

bool Model::load(const std::string& model_path, DeviceType device) {
    auto t_total_start = std::chrono::high_resolution_clock::now();

    loader_ = ModelLoaderRegistry::instance().create_loader(model_path);
    if (!loader_) {
        LOG_ERROR("No loader found for model: " + model_path);
        return false;
    }

    format_name_ = loader_->format_name();

    if (!loader_->load(model_path)) {
        LOG_ERROR("Failed to load model: " + model_path);
        return false;
    }

    if (format_name_ == "gguf") {
        config_ = parse_config_from_gguf(*loader_);
    } else if (format_name_ == "ninf") {
        config_ = parse_config_from_ninf(*loader_);
    }

    bool result = load_from_loader(*loader_, device);

    // Don't close the loader - keep mmap alive for zero-copy tensor references
    // loader_->close() will be called when Model is destroyed

    if (result) {
        model_path_ = model_path;
        device_ = device;
        is_loaded_ = true;
    }

    auto t_total_end = std::chrono::high_resolution_clock::now();
    double total_ms =
        std::chrono::duration<double, std::milli>(t_total_end - t_total_start).count();
    LOG_INFO("Model loaded in " + std::to_string(total_ms / 1000.0) + "s");

    return result;
}

bool Model::load_with_config(const std::string& model_path, const ModelConfig& config,
                             DeviceType device) {
    config_ = config;

    loader_ = ModelLoaderRegistry::instance().create_loader(model_path);
    if (!loader_) {
        LOG_ERROR("No loader found for model: " + model_path);
        return false;
    }

    format_name_ = loader_->format_name();
    if (!loader_->load(model_path)) {
        LOG_ERROR("Failed to load model: " + model_path);
        return false;
    }

    bool result = load_from_loader(*loader_, device);
    // Don't close the loader - keep mmap alive for zero-copy tensor references

    if (result) {
        model_path_ = model_path;
        device_ = device;
        is_loaded_ = true;
    }

    return result;
}

// Forward declaration for parse_common_gguf_config (defined later in this file)
static ModelConfig parse_common_gguf_config(ModelLoader& loader, const std::string& arch);

ModelConfig Model::parse_config_from_gguf(ModelLoader& loader) {
    std::string arch = loader.get_metadata_str("general.architecture", "llama");

    // Use registered config parser if available
    auto& registry = ConfigParserRegistry::instance();
    if (registry.has(arch)) {
        return registry.parse(loader, arch);
    }

    // Fallback: parse common fields for unregistered architectures
    auto cfg = parse_common_gguf_config(loader, arch);
    LOG_WARN("No registered config parser for arch '" + arch +
             "', using default parsing. Consider registering a parser via "
             "FORGE_REGISTER_CONFIG_PARSER.");
    return cfg;
}

ModelConfig Model::parse_config_from_ninf(ModelLoader& loader) {
    std::string arch =
        loader.get_metadata_str("model_type", loader.get_metadata_str("architecture", "llama"));

    ModelConfig cfg;

    auto meta_int = [&](const std::string& key, int64_t def) -> int64_t {
        int64_t v = loader.get_metadata_int(arch + "." + key, -1);
        if (v >= 0)
            return v;
        return loader.get_metadata_int(key, def);
    };

    auto meta_float = [&](const std::string& key, double def) -> double {
        std::string v = loader.get_metadata_str(arch + "." + key, "");
        if (!v.empty())
            return std::strtod(v.c_str(), nullptr);
        return loader.get_metadata_float(key, def);
    };

    auto meta_str = [&](const std::string& key, const std::string& def) -> std::string {
        std::string v = loader.get_metadata_str(arch + "." + key, "");
        if (!v.empty())
            return v;
        return loader.get_metadata_str(key, def);
    };

    cfg/*vocab_size=*/ static_cast<int>(meta_int("vocab_size", 32000));
    cfg/*hidden_dim=*/ static_cast<int>(meta_int("hidden_dim", 4096));
    cfg/*intermediate_dim=*/ static_cast<int>(meta_int("intermediate_dim", 11008));
    cfg/*num_layers=*/ static_cast<int>(meta_int("num_layers", 32));
    cfg/*num_heads=*/ static_cast<int>(meta_int("num_heads", 32));
    cfg/*num_kv_heads=*/ static_cast<int>(meta_int("num_kv_heads", cfg.num_heads));
    cfg/*head_dim=*/ static_cast<int>(meta_int("head_dim", 0));
    if (cfg/*head_dim=*/= 0)
        cfg/*head_dim=*/ cfg.hidden_dim / cfg.num_heads;
    cfg/*rope_theta=*/ static_cast<float>(meta_float("rope_theta", 10000.0));
    cfg/*rms_norm_eps=*/ static_cast<float>(meta_float("rms_norm_eps", 1e-6));
    cfg/*max_seq_len=*/ static_cast<int>(meta_int("max_seq_len", 4096));
    cfg/*arch_type=*/ arch;
    cfg/*norm_type=*/ NormType::RMSNorm;
    cfg/*ffn_activation=*/ ActivationType::SiLU_GELU;
    cfg/*use_gqa=*/ (cfg.num_kv_heads != cfg.num_heads);

    std::string tie_str = meta_str("tie_embeddings", "");
    if (tie_str == "true" || tie_str == "1") {
        cfg/*tie_embeddings=*/ true;
    }

    return cfg;
}

bool Model::load_from_loader(ModelLoader& loader, DeviceType device) {
    const auto& mapping = WeightMapper::get_mapping(config_.arch_type);

    LOG_INFO("Model::load_from_loader: arch=" + config_.arch_type +
             ", num_layers=" + std::to_string(config_.num_layers) +
             ", hidden_dim=" + std::to_string(config_.hidden_dim) + ", num_heads=" +
             std::to_string(config_.num_heads) + ", head_dim=" + std::to_string(config_.head_dim) +
             ", use_ssm=" + std::to_string(config_.use_ssm) +
             ", use_mrope=" + std::to_string(config_.use_mrope) +
             ", rope_dimension_count=" + std::to_string(config_.rope_dimension_count));

    auto load_tensor = [&](const std::string& name) -> TensorPtr {
        if (!loader.has_tensor(name))
            return nullptr;
        return loader.get_tensor(name, device);
    };

    auto resolve_weight = [&](const WeightAlias& alias,
                              const std::string& prefix = "") -> TensorPtr {
        for (const auto& name : alias.names) {
            std::string full_name = prefix.empty() ? name : (prefix + "." + name);
            auto t = load_tensor(full_name);
            if (t) {
                LOG_TRACE("Loaded weight: " + full_name);
                return t;
            }
        }
        return nullptr;
    };

    if (format_name_ == "gguf") {
        // GGUF uses "blk.{i}" prefix and different weight naming convention.
        // We build a GGUF-specific mapping and use the same WeightMapper::resolve logic.
        // GGUF canonical names: token_embd.weight, output_norm.weight, output.weight
        // Layer names: blk.{i}.attn_q.weight, blk.{i}.ffn_gate.weight, etc.

        auto te = load_tensor("token_embd.weight");
        if (!te) {
            LOG_ERROR("Failed to load token embedding");
            return false;
        }
        weight_store_.set("token_embedding", te);

        auto on = load_tensor("output_norm.weight");
        if (on)
            weight_store_.set("output_norm", on);

        auto ow = load_tensor("output.weight");
        if (ow) {
            weight_store_.set("output_weight", ow);
        } else if (config_.tie_embeddings) {
            weight_store_.set("output_weight", te);
        }

        // Build GGUF-specific layer mapping based on the arch mapping
        // GGUF uses "blk.{i}" prefix and names like attn_q.weight, ffn_gate.weight
        for (int i = 0; i < config_.num_layers; ++i) {
            std::string blk = "blk." + std::to_string(i);
            std::string base = "layers." + std::to_string(i);

            auto set_if = [&](const std::string& canonical, const std::string& gguf_name) {
                auto t = load_tensor(gguf_name);
                if (t)
                    weight_store_.set(canonical, t);
            };

            // Common weights (same for all architectures)
            set_if(base + ".attn_norm", blk + ".attn_norm.weight");
            set_if(base + ".ffn_norm", blk + ".ffn_norm.weight");
            set_if(base + ".gate_proj", blk + ".ffn_gate.weight");
            set_if(base + ".down_proj", blk + ".ffn_down.weight");
            set_if(base + ".up_proj", blk + ".ffn_up.weight");

            // Architecture-specific attention weights
            if (config_/*arch_type=*/= "qwen35") {
                set_if(base + ".attn_q", blk + ".attn_q.weight");
                set_if(base + ".attn_k", blk + ".attn_k.weight");
                set_if(base + ".attn_v", blk + ".attn_v.weight");
                set_if(base + ".attn_output", blk + ".attn_output.weight");
                // SSM weights
                set_if(base + ".attn_qkv", blk + ".attn_qkv.weight");
                set_if(base + ".attn_gate", blk + ".attn_gate.weight");
                set_if(base + ".attn_q_norm", blk + ".attn_q_norm.weight");
                set_if(base + ".attn_k_norm", blk + ".attn_k_norm.weight");
                set_if(base + ".post_attention_norm", blk + ".post_attention_norm.weight");
                set_if(base + ".ssm_norm", blk + ".ssm_norm.weight");
                set_if(base + ".ssm_a", blk + ".ssm_a");
                set_if(base + ".ssm_dt", blk + ".ssm_dt.bias");
                set_if(base + ".ssm_conv1d", blk + ".ssm_conv1d.weight");
                set_if(base + ".ssm_alpha", blk + ".ssm_alpha.weight");
                set_if(base + ".ssm_beta", blk + ".ssm_beta.weight");
                set_if(base + ".ssm_out", blk + ".ssm_out.weight");
            } else if (config_.use_mla) {
                set_if(base + ".wq_a", blk + ".attn_q_a.weight");
                set_if(base + ".wq_b", blk + ".attn_q_b.weight");
                set_if(base + ".kv_a_proj", blk + ".attn_kv_a.weight");
                set_if(base + ".kv_b_proj", blk + ".attn_kv_b.weight");
                set_if(base + ".wo", blk + ".attn_output.weight");
            } else {
                // Standard GQA attention
                set_if(base + ".wq", blk + ".attn_q.weight");
                set_if(base + ".wk", blk + ".attn_k.weight");
                set_if(base + ".wv", blk + ".attn_v.weight");
                set_if(base + ".wo", blk + ".attn_output.weight");
            }

            // Biases (may not exist for all architectures)
            set_if(base + ".bq", blk + ".attn_q.bias");
            set_if(base + ".bk", blk + ".attn_k.bias");
            set_if(base + ".bv", blk + ".attn_v.bias");

            // Neox RoPE: convert interleaved weights to half-split format
            // so that both CPU and GPU RoPE use (x[d], x[d+half_dim]) indexing
            if (config_.use_neox_rope) {
                auto wq = weight_store_.get(base + ".wq");
                if (wq)
                    weight_store_.set(base + ".wq", inverse_neox_permute_rows(wq, config_.num_heads,
                                                                              config_.head_dim));

                auto wk = weight_store_.get(base + ".wk");
                if (wk)
                    weight_store_.set(
                        base + ".wk",
                        inverse_neox_permute_rows(wk, config_.num_kv_heads, config_.head_dim));
            }
        }

        // Vision / Multimodal weights (mmproj)
        // Auto-scan all tensors with v.* or mm.* prefixes
        {
            for (const auto& name : loader.tensor_names()) {
                if (name.size() >= 2 &&
                    (name.compare(0, 2, "v.") == 0 || name.compare(0, 3, "mm.") == 0)) {
                    auto t = load_tensor(name);
                    if (t)
                        weight_store_.set(name, t);
                }
            }
        }
    } else {
        auto te = resolve_weight(mapping.token_embedding);
        if (!te) {
            LOG_ERROR("Failed to load token embedding");
            return false;
        }
        weight_store_.set("token_embedding", te);

        auto on = resolve_weight(mapping.output_norm);
        if (on)
            weight_store_.set("output_norm", on);

        if (!mapping.tie_embeddings) {
            auto ow = resolve_weight(mapping.output_weight);
            if (ow)
                weight_store_.set("output_weight", ow);
        }

        for (int i = 0; i < config_.num_layers; ++i) {
            std::string prefix = WeightMapper::format_layer_prefix(mapping.layer_prefix_pattern, i);

            auto set_if = [&](const std::string& canonical, const WeightAlias& alias) {
                auto t = resolve_weight(alias, prefix);
                if (t)
                    weight_store_.set(canonical, t);
            };

            std::string base = "layers." + std::to_string(i);

            // Common weights
            set_if(base + ".attn_norm", mapping.layer.attn_norm);
            set_if(base + ".ffn_norm", mapping.layer.ffn_norm);
            set_if(base + ".wo", mapping.layer.wo);
            set_if(base + ".gate_proj", mapping.layer.gate_proj);
            set_if(base + ".down_proj", mapping.layer.down_proj);
            set_if(base + ".up_proj", mapping.layer.up_proj);

            // GQA attention
            set_if(base + ".wq", mapping.layer.wq);
            set_if(base + ".wk", mapping.layer.wk);
            set_if(base + ".wv", mapping.layer.wv);

            // MLA attention
            set_if(base + ".wq_a", mapping.layer.wq_a);
            set_if(base + ".wq_b", mapping.layer.wq_b);
            set_if(base + ".kv_a_proj", mapping.layer.kv_a_proj);
            set_if(base + ".kv_b_proj", mapping.layer.kv_b_proj);

            // Qwen35 full attention
            set_if(base + ".attn_q", mapping.layer.attn_q);
            set_if(base + ".attn_k", mapping.layer.attn_k);
            set_if(base + ".attn_v", mapping.layer.attn_v);
            set_if(base + ".attn_output", mapping.layer.attn_output);
            set_if(base + ".attn_q_norm", mapping.layer.attn_q_norm);
            set_if(base + ".attn_k_norm", mapping.layer.attn_k_norm);
            set_if(base + ".post_attention_norm", mapping.layer.post_attention_norm);

            // Qwen35 SSM
            set_if(base + ".attn_qkv", mapping.layer.attn_qkv);
            set_if(base + ".attn_gate", mapping.layer.attn_gate);
            set_if(base + ".ssm_conv1d", mapping.layer.ssm_conv1d);
            set_if(base + ".ssm_dt", mapping.layer.ssm_dt);
            set_if(base + ".ssm_a", mapping.layer.ssm_a);
            set_if(base + ".ssm_alpha", mapping.layer.ssm_alpha);
            set_if(base + ".ssm_beta", mapping.layer.ssm_beta);
            set_if(base + ".ssm_norm", mapping.layer.ssm_norm);
            set_if(base + ".ssm_out", mapping.layer.ssm_out);
        }
    }

    LOG_INFO("Loaded " + std::to_string(weight_store_.size()) + " weights from " + format_name_ +
             " for arch: " + config_.arch_type);
    return true;
}

void Model::set_config(const ModelConfig& config) {
    config_ = config;
}

void Model::set_device(DeviceType device) {
    device_ = device;
}

TensorPtr Model::get_weight(const std::string& name) const {
    return weight_store_.get(name);
}

bool Model::load_vision_weights(const std::string& mmproj_path, DeviceType device) {
    vision_loader_ = ModelLoaderRegistry::instance().create_loader(mmproj_path);
    if (!vision_loader_) {
        LOG_ERROR("No loader found for mmproj: " + mmproj_path);
        return false;
    }

    if (!vision_loader_->load(mmproj_path)) {
        LOG_ERROR("Failed to load mmproj: " + mmproj_path);
        return false;
    }

    // Auto-scan all tensors from the mmproj file.
    // Vision tensors have prefixes: v.* (v.blk.*, v.patch_embd.*, v.position_embd.*, etc.)
    // Projector tensors have prefix: mm.*
    // Instead of hardcoding each tensor name, we load all tensors matching these prefixes.
    int loaded_count = 0;
    for (const auto& name : vision_loader_->tensor_names()) {
        if (name.size() >= 2 && (name.compare(0, 2, "v.") == 0 || name.compare(0, 3, "mm.") == 0)) {
            auto t = vision_loader_->get_tensor(name, device);
            if (t) {
                weight_store_.set(name, t);
                ++loaded_count;
            }
        }
    }

    // Don't close vision_loader_ - keep mmap alive for zero-copy tensor references

    LOG_INFO("Loaded " + std::to_string(loaded_count) + " vision/mmproj weights from " +
             mmproj_path + " (total weights: " + std::to_string(weight_store_.size()) + ")");
    return true;
}

static const ArchWeightMapping llama_mapping = {
    /*token_embedding=*/{{"model.embed_tokens.weight", "model.embedding.weight"}},
    /*output_norm=*/{{"model.norm.weight", "model.ln_f.weight"}},
    /*output_weight=*/{{"lm_head.weight", "model.output.weight"}},
    /*layer=*/{
        /*attn_norm=*/{{"input_layernorm.weight", "attention_norm.weight"}},
        /*ffn_norm=*/{{"post_attention_layernorm.weight", "ffn_norm.weight"}},
        /*wo=*/{{"self_attn.o_proj.weight", "attention.wo.weight"}},
        /*gate_proj=*/{{"mlp.gate_proj.weight", "feed_forward.w1.weight"}},
        /*down_proj=*/{{"mlp.down_proj.weight", "feed_forward.w2.weight"}},
        /*up_proj=*/{{"mlp.up_proj.weight", "feed_forward.w3.weight"}},
        /*wq=*/{{"self_attn.q_proj.weight", "attention.wq.weight"}},
        /*wk=*/{{"self_attn.k_proj.weight", "attention.wk.weight"}},
        /*wv=*/{{"self_attn.v_proj.weight", "attention.wv.weight"}},
    },
    /*layer_prefix_pattern=*/"model.layers.{}",
    /*tie_embeddings=*/false,
};

static const ArchWeightMapping mistral_mapping = {
    /*token_embedding=*/ {{"model.embed_tokens.weight"}},
    /*output_norm=*/ {{"model.norm.weight"}},
    /*output_weight=*/ {{"lm_head.weight"}},
    /*layer=*/
        {
            /*attn_norm=*/ {{"input_layernorm.weight"}},
            /*ffn_norm=*/ {{"post_attention_layernorm.weight"}},
            /*wo=*/ {{"self_attn.o_proj.weight"}},
            /*gate_proj=*/ {{"mlp.gate_proj.weight"}},
            /*down_proj=*/ {{"mlp.down_proj.weight"}},
            /*up_proj=*/ {{"mlp.up_proj.weight"}},
            /*wq=*/ {{"self_attn.q_proj.weight"}},
            /*wk=*/ {{"self_attn.k_proj.weight"}},
            /*wv=*/ {{"self_attn.v_proj.weight"}},
        },
    /*layer_prefix_pattern=*/ "model.layers.{}",
    /*tie_embeddings=*/ false,
};

static const ArchWeightMapping qwen_mapping = {
    /*token_embedding=*/ {{"model.embed_tokens.weight"}},
    /*output_norm=*/ {{"model.norm.weight"}},
    /*output_weight=*/ {{"lm_head.weight"}},
    /*layer=*/
        {
            /*attn_norm=*/ {{"input_layernorm.weight"}},
            /*ffn_norm=*/ {{"post_attention_layernorm.weight"}},
            /*wo=*/ {{"self_attn.o_proj.weight"}},
            /*gate_proj=*/ {{"mlp.gate_proj.weight"}},
            /*down_proj=*/ {{"mlp.down_proj.weight"}},
            /*up_proj=*/ {{"mlp.up_proj.weight"}},
            /*wq=*/ {{"self_attn.q_proj.weight"}},
            /*wk=*/ {{"self_attn.k_proj.weight"}},
            /*wv=*/ {{"self_attn.v_proj.weight"}},
        },
    /*layer_prefix_pattern=*/ "model.layers.{}",
    /*tie_embeddings=*/ false,
};

static const ArchWeightMapping deepseek_v2_mapping = {
    /*token_embedding=*/ {{"model.embed_tokens.weight"}},
    /*output_norm=*/ {{"model.norm.weight"}},
    /*output_weight=*/ {{"lm_head.weight"}},
    /*layer=*/
        {
            /*attn_norm=*/ {{"input_layernorm.weight"}},
            /*ffn_norm=*/ {{"post_attention_layernorm.weight"}},
            /*wo=*/ {{"self_attn.o_proj.weight"}},
            /*gate_proj=*/ {{"mlp.gate_proj.weight"}},
            /*down_proj=*/ {{"mlp.down_proj.weight"}},
            /*up_proj=*/ {{"mlp.up_proj.weight"}},
            /*wq_a=*/ {{"self_attn.q_a_proj.weight"}},
            /*wq_b=*/ {{"self_attn.q_b_proj.weight"}},
            /*kv_a_proj=*/ {{"self_attn.kv_a_proj_with_mqa.weight"}},
            /*kv_b_proj=*/ {{"self_attn.kv_b_proj.weight"}},
        },
    /*layer_prefix_pattern=*/ "model.layers.{}",
    /*tie_embeddings=*/ false,
};

static const ArchWeightMapping qwen35_mapping = {
    /*token_embedding=*/ {{"model.embed_tokens.weight"}},
    /*output_norm=*/ {{"model.norm.weight"}},
    /*output_weight=*/ {{"lm_head.weight"}},
    /*layer=*/
        {
            /*attn_norm=*/ {{"input_layernorm.weight"}},
            /*ffn_norm=*/ {{"post_attention_layernorm.weight"}},
            /*wo=*/ {{"self_attn.o_proj.weight"}},
            /*gate_proj=*/ {{"mlp.gate_proj.weight"}},
            /*down_proj=*/ {{"mlp.down_proj.weight"}},
            /*up_proj=*/ {{"mlp.up_proj.weight"}},
            /*attn_q=*/ {{"self_attn.q_proj.weight"}},
            /*attn_k=*/ {{"self_attn.k_proj.weight"}},
            /*attn_v=*/ {{"self_attn.v_proj.weight"}},
            /*attn_output=*/ {{"self_attn.o_proj.weight"}},
            /*attn_q_norm=*/ {{"self_attn.q_norm.weight"}},
            /*attn_k_norm=*/ {{"self_attn.k_norm.weight"}},
            /*post_attention_norm=*/ {{"post_attention_layernorm.weight"}},
            /*attn_qkv=*/ {{"self_attn.qkv_proj.weight"}},
            /*attn_gate=*/ {{"self_attn.gate.weight"}},
            /*ssm_conv1d=*/ {{"ssm.conv1d.weight"}},
            /*ssm_dt=*/ {{"ssm.dt.bias"}},
            /*ssm_a=*/ {{"ssm.a"}},
            /*ssm_alpha=*/ {{"ssm.alpha.weight"}},
            /*ssm_beta=*/ {{"ssm.beta.weight"}},
            /*ssm_norm=*/ {{"ssm.norm.weight"}},
            /*ssm_out=*/ {{"ssm.output.weight"}},
        },
    /*layer_prefix_pattern=*/ "model.layers.{}",
    /*tie_embeddings=*/ false,
};

static const std::unordered_map<std::string, const ArchWeightMapping*> arch_mappings = {
    {"llama", &llama_mapping},
    {"mistral", &mistral_mapping},
    {"qwen", &qwen_mapping},
    {"qwen2", &qwen_mapping},
    {"yi", &llama_mapping},
    {"deepseek", &llama_mapping},
    {"deepseek_v2", &deepseek_v2_mapping},
    {"deepseek_v3", &deepseek_v2_mapping},
    {"qwen35", &qwen35_mapping},
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

// ============================================================================
// ConfigParserRegistry implementation
// ============================================================================

ConfigParserRegistry& ConfigParserRegistry::instance() {
    static ConfigParserRegistry registry;
    return registry;
}

void ConfigParserRegistry::register_parser(const std::string& arch, ConfigParseFn fn) {
    parsers_[arch] = std::move(fn);
}

bool ConfigParserRegistry::has(const std::string& arch) const {
    return parsers_.find(arch) != parsers_.end();
}

ModelConfig ConfigParserRegistry::parse(ModelLoader& loader, const std::string& arch) const {
    auto it = parsers_.find(arch);
    if (it != parsers_.end())
        return it->second(loader, arch);
    return ModelConfig{};
}

ConfigParserAutoRegister::ConfigParserAutoRegister(const std::string& arch, ConfigParseFn fn) {
    ConfigParserRegistry::instance().register_parser(arch, std::move(fn));
}

// ============================================================================
// Default config parsing: common fields shared by all architectures
// ============================================================================

static ModelConfig parse_common_gguf_config(ModelLoader& loader, const std::string& arch) {
    ModelConfig cfg;
    cfg/*vocab_size=*/ static_cast<int>(loader.get_metadata_int(arch + ".vocab_size", 0));
    cfg/*hidden_dim=*/ static_cast<int>(loader.get_metadata_int(arch + ".embedding_length", 4096));
    cfg/*intermediate_dim=*/
        static_cast<int>(loader.get_metadata_int(arch + ".feed_forward_length", 11008));
    cfg/*num_layers=*/ static_cast<int>(loader.get_metadata_int(arch + ".block_count", 32));
    cfg/*num_heads=*/ static_cast<int>(loader.get_metadata_int(arch + ".attention.head_count", 32));
    cfg/*num_kv_heads=*/
        static_cast<int>(loader.get_metadata_int(arch + ".attention.head_count_kv", cfg.num_heads));
    cfg/*head_dim=*/ static_cast<int>(loader.get_metadata_int(arch + ".attention.key_length", 0));
    if (cfg/*head_dim=*/= 0)
        cfg/*head_dim=*/ cfg.hidden_dim / cfg.num_heads;
    cfg/*rope_theta=*/
        static_cast<float>(loader.get_metadata_float(arch + ".rope.freq_base", 10000.0));
    cfg/*rms_norm_eps=*/ static_cast<float>(
        loader.get_metadata_float(arch + ".attention.layer_norm_rms_epsilon", 1e-6));
    cfg/*max_seq_len=*/ static_cast<int>(loader.get_metadata_int(arch + ".context_length", 4096));
    cfg/*arch_type=*/ arch;
    cfg/*norm_type=*/ NormType::RMSNorm;
    cfg/*ffn_activation=*/ ActivationType::SiLU_GELU;
    cfg/*use_gqa=*/ (cfg.num_kv_heads != cfg.num_heads);

    if (cfg/*vocab_size=*/= 0) {
        auto embd_shape = loader.get_tensor_shape("token_embd.weight");
        if (!embd_shape.empty() && embd_shape.size() >= 2) {
            cfg/*vocab_size=*/ static_cast<int>(embd_shape[0]);
        } else if (loader.has_tensor("token_embd.weight")) {
            cfg/*vocab_size=*/ 32000;
        }
        if (cfg/*vocab_size=*/= 0)
            cfg/*vocab_size=*/ 32000;
    }

    return cfg;
}

// ============================================================================
// Architecture-specific config parsers (registered via static auto-register)
// ============================================================================

// LLaMA config parser
namespace {
static ConfigParserAutoRegister _reg_cfg_llama(
    "llama", [](ModelLoader& loader, const std::string& arch) -> ModelConfig {
        auto cfg = parse_common_gguf_config(loader, arch);
        cfg/*use_neox_rope=*/ true;
        return cfg;
    });

static ConfigParserAutoRegister _reg_cfg_mistral(
    "mistral", [](ModelLoader& loader, const std::string& arch) -> ModelConfig {
        auto cfg = parse_common_gguf_config(loader, arch);
        cfg/*use_neox_rope=*/ true;
        return cfg;
    });

static ConfigParserAutoRegister _reg_cfg_qwen("qwen",
                                              [](ModelLoader& loader,
                                                 const std::string& arch) -> ModelConfig {
                                                  auto cfg = parse_common_gguf_config(loader, arch);
                                                  if (cfg/*rope_theta=*/= 10000.0f)
                                                      cfg/*rope_theta=*/ 1000000.0f;
                                                  cfg/*tie_embeddings=*/ true;
                                                  return cfg;
                                              });

static ConfigParserAutoRegister _reg_cfg_qwen2(
    "qwen2", [](ModelLoader& loader, const std::string& arch) -> ModelConfig {
        auto cfg = parse_common_gguf_config(loader, arch);
        if (cfg/*rope_theta=*/= 10000.0f)
            cfg/*rope_theta=*/ 1000000.0f;
        cfg/*tie_embeddings=*/ true;
        return cfg;
    });

static ConfigParserAutoRegister _reg_cfg_yi("yi",
                                            [](ModelLoader& loader,
                                               const std::string& arch) -> ModelConfig {
                                                auto cfg = parse_common_gguf_config(loader, arch);
                                                cfg/*use_neox_rope=*/ true;
                                                return cfg;
                                            });

static ConfigParserAutoRegister _reg_cfg_deepseek("deepseek",
                                                  [](ModelLoader& loader,
                                                     const std::string& arch) -> ModelConfig {
                                                      return parse_common_gguf_config(loader, arch);
                                                  });

static ConfigParserAutoRegister _reg_cfg_deepseek_v2(
    "deepseek_v2", [](ModelLoader& loader, const std::string& arch) -> ModelConfig {
        auto cfg = parse_common_gguf_config(loader, arch);
        cfg/*kv_lora_rank=*/ static_cast<int>(loader.get_metadata_int(arch + ".kv_lora_rank", 0));
        cfg/*q_lora_rank=*/ static_cast<int>(loader.get_metadata_int(arch + ".q_lora_rank", 0));
        cfg/*use_mla=*/ (cfg.kv_lora_rank > 0);
        cfg/*n_routed_experts=*/
            static_cast<int>(loader.get_metadata_int(arch + ".n_routed_experts", 0));
        cfg/*n_shared_experts=*/
            static_cast<int>(loader.get_metadata_int(arch + ".n_shared_experts", 0));
        cfg/*num_expert_per_tok=*/
            static_cast<int>(loader.get_metadata_int(arch + ".num_expert_per_tok", 0));
        if (cfg.kv_lora_rank > 0) {
            cfg/*num_kv_heads=*/ 1;
            cfg/*head_dim=*/ cfg.kv_lora_rank;
        }
        return cfg;
    });

static ConfigParserAutoRegister _reg_cfg_deepseek_v3(
    "deepseek_v3", [](ModelLoader& loader, const std::string& arch) -> ModelConfig {
        auto cfg = parse_common_gguf_config(loader, arch);
        cfg/*kv_lora_rank=*/ static_cast<int>(loader.get_metadata_int(arch + ".kv_lora_rank", 0));
        cfg/*q_lora_rank=*/ static_cast<int>(loader.get_metadata_int(arch + ".q_lora_rank", 0));
        cfg/*use_mla=*/ (cfg.kv_lora_rank > 0);
        cfg/*n_routed_experts=*/
            static_cast<int>(loader.get_metadata_int(arch + ".n_routed_experts", 0));
        cfg/*n_shared_experts=*/
            static_cast<int>(loader.get_metadata_int(arch + ".n_shared_experts", 0));
        cfg/*num_expert_per_tok=*/
            static_cast<int>(loader.get_metadata_int(arch + ".num_expert_per_tok", 0));
        if (cfg.kv_lora_rank > 0) {
            cfg/*num_kv_heads=*/ 1;
            cfg/*head_dim=*/ cfg.kv_lora_rank;
        }
        return cfg;
    });

static ConfigParserAutoRegister _reg_cfg_qwen35(
    "qwen35", [](ModelLoader& loader, const std::string& arch) -> ModelConfig {
        auto cfg = parse_common_gguf_config(loader, arch);
        cfg/*tie_embeddings=*/ true;
        cfg/*use_ssm=*/ true;
        cfg/*ssm_group_count=*/
            static_cast<int>(loader.get_metadata_int(arch + ".ssm.group_count", 0));
        cfg/*ssm_time_step_rank=*/
            static_cast<int>(loader.get_metadata_int(arch + ".ssm.time_step_rank", 0));
        cfg/*ssm_inner_size=*/ static_cast<int>(loader.get_metadata_int(arch + ".ssm.inner_size", 0));
        cfg/*ssm_state_size=*/ static_cast<int>(loader.get_metadata_int(arch + ".ssm.state_size", 0));
        cfg/*ssm_conv_kernel=*/
            static_cast<int>(loader.get_metadata_int(arch + ".ssm.conv_kernel", 0));
        cfg/*full_attention_interval=*/
            static_cast<int>(loader.get_metadata_int(arch + ".full_attention_interval", 0));

        // MRoPE (Multi-dimensional RoPE) for Qwen3.5
        cfg/*rope_dimension_count=*/
            static_cast<int>(loader.get_metadata_int(arch + ".rope.dimension_count", 0));
        auto sections = loader.get_metadata_int_array(arch + ".rope.dimension_sections", {});
        if (!sections.empty() && cfg.rope_dimension_count > 0) {
            cfg/*use_mrope=*/ true;
            for (size_t i = 0; i < sections.size() && i < 4; ++i) {
                cfg.rope_dimension_sections[i] = sections[i];
            }
        }
        return cfg;
    });
}  // namespace

// ============================================================================
// WeightInitRegistry implementation
// ============================================================================

WeightInitRegistry& WeightInitRegistry::instance() {
    static WeightInitRegistry registry;
    return registry;
}

void WeightInitRegistry::register_init(const std::string& arch, LayerWeightInitFn fn) {
    inits_[arch] = std::move(fn);
}

bool WeightInitRegistry::has(const std::string& arch) const {
    return inits_.find(arch) != inits_.end();
}

void WeightInitRegistry::init_layer(const std::string& arch, LayerWeightInitContext& ctx) const {
    auto it = inits_.find(arch);
    if (it != inits_.end()) {
        it->second(ctx);
    }
}

WeightInitAutoRegister::WeightInitAutoRegister(const std::string& arch, LayerWeightInitFn fn) {
    WeightInitRegistry::instance().register_init(arch, std::move(fn));
}

// ============================================================================
// Architecture-specific weight init functions (registered via static auto-register)
// ============================================================================

namespace {

// Helper: load weight from store into LayerWeights if present
static void load_if_present(const WeightStore& store, LayerWeights& lw,
                            const std::string& canonical, const std::string& store_name) {
    auto t = store.get(store_name);
    if (t)
        lw.set(canonical, t);
}

// Standard GQA attention weight init (LLaMA, Mistral, Qwen2, Yi, DeepSeek)
static void init_gqa_layer_weights(LayerWeightInitContext& ctx) {
    const auto& store = ctx.store;
    auto& lw = ctx.lw;
    std::string base = "layers." + std::to_string(ctx.layer_idx);

    load_if_present(store, lw, "attn_norm", base + ".attn_norm");
    load_if_present(store, lw, "ffn_norm", base + ".ffn_norm");
    load_if_present(store, lw, "w1", base + ".gate_proj");
    load_if_present(store, lw, "w2", base + ".down_proj");
    load_if_present(store, lw, "w3", base + ".up_proj");
    load_if_present(store, lw, "wq", base + ".wq");
    load_if_present(store, lw, "wk", base + ".wk");
    load_if_present(store, lw, "wv", base + ".wv");
    load_if_present(store, lw, "wo", base + ".wo");
    load_if_present(store, lw, "bq", base + ".bq");
    load_if_present(store, lw, "bk", base + ".bk");
    load_if_present(store, lw, "bv", base + ".bv");
}

// MLA attention weight init (DeepSeek V2/V3)
static void init_mla_layer_weights(LayerWeightInitContext& ctx) {
    const auto& store = ctx.store;
    auto& lw = ctx.lw;
    std::string base = "layers." + std::to_string(ctx.layer_idx);

    load_if_present(store, lw, "attn_norm", base + ".attn_norm");
    load_if_present(store, lw, "ffn_norm", base + ".ffn_norm");
    load_if_present(store, lw, "w1", base + ".gate_proj");
    load_if_present(store, lw, "w2", base + ".down_proj");
    load_if_present(store, lw, "w3", base + ".up_proj");
    load_if_present(store, lw, "wq_a", base + ".wq_a");
    load_if_present(store, lw, "wq_b", base + ".wq_b");
    load_if_present(store, lw, "kv_a_proj", base + ".kv_a_proj");
    load_if_present(store, lw, "kv_b_proj", base + ".kv_b_proj");
    load_if_present(store, lw, "wo", base + ".wo");
}

// Qwen35 hybrid attention/SSM weight init
static void init_qwen35_layer_weights(LayerWeightInitContext& ctx) {
    const auto& store = ctx.store;
    auto& lw = ctx.lw;
    std::string base = "layers." + std::to_string(ctx.layer_idx);

    load_if_present(store, lw, "attn_norm", base + ".attn_norm");
    load_if_present(store, lw, "ffn_norm", base + ".ffn_norm");
    load_if_present(store, lw, "w1", base + ".gate_proj");
    load_if_present(store, lw, "w2", base + ".down_proj");
    load_if_present(store, lw, "w3", base + ".up_proj");

    // Full attention weights
    load_if_present(store, lw, "attn_q", base + ".attn_q");
    load_if_present(store, lw, "attn_k", base + ".attn_k");
    load_if_present(store, lw, "attn_v", base + ".attn_v");
    load_if_present(store, lw, "attn_output", base + ".attn_output");
    load_if_present(store, lw, "attn_q_norm", base + ".attn_q_norm");
    load_if_present(store, lw, "attn_k_norm", base + ".attn_k_norm");
    load_if_present(store, lw, "post_attention_norm", base + ".post_attention_norm");

    // Linear attention / SSM weights
    load_if_present(store, lw, "attn_qkv", base + ".attn_qkv");
    load_if_present(store, lw, "attn_gate", base + ".attn_gate");
    load_if_present(store, lw, "ssm_conv1d", base + ".ssm_conv1d");
    load_if_present(store, lw, "ssm_dt", base + ".ssm_dt");
    load_if_present(store, lw, "ssm_a", base + ".ssm_a");
    load_if_present(store, lw, "ssm_alpha", base + ".ssm_alpha");
    load_if_present(store, lw, "ssm_beta", base + ".ssm_beta");
    load_if_present(store, lw, "ssm_norm", base + ".ssm_norm");
    load_if_present(store, lw, "ssm_out", base + ".ssm_out");
}

// Register weight init functions for each architecture
static WeightInitAutoRegister _reg_winit_llama("llama", init_gqa_layer_weights);
static WeightInitAutoRegister _reg_winit_mistral("mistral", init_gqa_layer_weights);
static WeightInitAutoRegister _reg_winit_qwen("qwen", init_gqa_layer_weights);
static WeightInitAutoRegister _reg_winit_qwen2("qwen2", init_gqa_layer_weights);
static WeightInitAutoRegister _reg_winit_yi("yi", init_gqa_layer_weights);
static WeightInitAutoRegister _reg_winit_deepseek("deepseek", init_gqa_layer_weights);
static WeightInitAutoRegister _reg_winit_deepseek_v2("deepseek_v2", init_mla_layer_weights);
static WeightInitAutoRegister _reg_winit_deepseek_v3("deepseek_v3", init_mla_layer_weights);
static WeightInitAutoRegister _reg_winit_qwen35("qwen35", init_qwen35_layer_weights);

}  // namespace

// ============================================================================
// ArchCapabilityRegistry implementation
// ============================================================================

ArchCapabilityRegistry& ArchCapabilityRegistry::instance() {
    static ArchCapabilityRegistry registry;
    return registry;
}

void ArchCapabilityRegistry::register_capability(const std::string& arch,
                                                 const ArchCapability& cap) {
    capabilities_[arch] = cap;
}

bool ArchCapabilityRegistry::has(const std::string& arch) const {
    return capabilities_.find(arch) != capabilities_.end();
}

ArchCapability ArchCapabilityRegistry::get(const std::string& arch) const {
    auto it = capabilities_.find(arch);
    if (it != capabilities_.end())
        return it->second;
    return ArchCapability{};
}

ArchCapabilityAutoRegister::ArchCapabilityAutoRegister(const std::string& arch,
                                                       const ArchCapability& cap) {
    ArchCapabilityRegistry::instance().register_capability(arch, cap);
}

// ============================================================================
// Register architecture capabilities
// ============================================================================

namespace {
// GQA architectures (use LlamaEngine)
static ArchCapabilityAutoRegister _reg_cap_llama("llama", ArchCapability{/*use_gqa=*/ true,
                                                                         /*use_neox_rope=*/ true});
static ArchCapabilityAutoRegister _reg_cap_mistral("mistral",
                                                   ArchCapability{/*use_gqa=*/ true,
                                                                  /*use_neox_rope=*/ true});
static ArchCapabilityAutoRegister _reg_cap_qwen("qwen", ArchCapability{/*use_gqa=*/ true});
static ArchCapabilityAutoRegister _reg_cap_qwen2("qwen2", ArchCapability{/*use_gqa=*/ true});
static ArchCapabilityAutoRegister _reg_cap_yi("yi", ArchCapability{/*use_gqa=*/ true,
                                                                   /*use_neox_rope=*/ true});
static ArchCapabilityAutoRegister _reg_cap_deepseek("deepseek", ArchCapability{/*use_gqa=*/ true});

// MLA architectures (use DeepSeekEngine)
static ArchCapabilityAutoRegister _reg_cap_deepseek_v2("deepseek_v2",
                                                       ArchCapability{/*use_mla=*/ true});
static ArchCapabilityAutoRegister _reg_cap_deepseek_v3("deepseek_v3",
                                                       ArchCapability{/*use_mla=*/ true});

// SSM architectures (use Qwen35Engine)
static ArchCapabilityAutoRegister _reg_cap_qwen35("qwen35", ArchCapability{/*use_ssm=*/ true,
                                                                           /*use_mrope=*/ true});
}  // namespace

}  // namespace forge
