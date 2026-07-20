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
#include "forge/quant_policy.h"
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

    cfg.vocab_size = static_cast<int>(meta_int("vocab_size", 32000));
    cfg.hidden_dim = static_cast<int>(meta_int("hidden_dim", 4096));
    cfg.intermediate_dim = static_cast<int>(meta_int("intermediate_dim", 11008));
    cfg.num_layers = static_cast<int>(meta_int("num_layers", 32));
    cfg.num_heads = static_cast<int>(meta_int("num_heads", 32));
    cfg.num_kv_heads = static_cast<int>(meta_int("num_kv_heads", cfg.num_heads));
    cfg.head_dim = static_cast<int>(meta_int("head_dim", 0));
    if (cfg.head_dim == 0)
        cfg.head_dim = cfg.hidden_dim / cfg.num_heads;
    cfg.rope_theta = static_cast<float>(meta_float("rope_theta", 10000.0));
    cfg.rms_norm_eps = static_cast<float>(meta_float("rms_norm_eps", 1e-6));
    cfg.max_seq_len = static_cast<int>(meta_int("max_seq_len", 4096));
    cfg.arch_type = arch;
    cfg.norm_type = NormType::RMSNorm;
    cfg.ffn_activation = ActivationType::SiLU_GELU;
    cfg.use_gqa = (cfg.num_kv_heads != cfg.num_heads);

    std::string tie_str = meta_str("tie_embeddings", "");
    if (tie_str == "true" || tie_str == "1") {
        cfg.tie_embeddings = true;
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
        auto t = loader.get_tensor(name, device);
        // Per-tensor 混合精度: 按 tensor 名称模式匹配重新量化
        if (t && quant_policy_.enabled() && is_quantized_type(t->dtype())) {
            DataType preferred = select_quant_type(quant_policy_, name);
            if (t->dtype() != preferred && is_quantized_type(preferred)) {
                t = requant_tensor(t, preferred);
            }
        }
        return t;
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

        // Gemma4 per-layer embedding weights (model-level, not per-blk)
        if (config_.arch_type == "gemma4") {
            auto ple = load_tensor("per_layer_token_embd.weight");
            if (ple) weight_store_.set("per_layer_tok_embd", ple);
            auto plmp = load_tensor("per_layer_model_proj.weight");
            if (plmp) weight_store_.set("per_layer_model_proj", plmp);
            auto plpn = load_tensor("per_layer_proj_norm.weight");
            if (plpn) weight_store_.set("per_layer_proj_norm", plpn);
            // RoPE frequency factors for proportional RoPE (full-attention layers)
            auto rf = load_tensor("rope_freqs.weight");
            if (rf) weight_store_.set("rope_freqs", rf);
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
            if (config_.arch_type == "gemma4") {
                // Gemma4 attention weights with QK-norm
                set_if(base + ".wq", blk + ".attn_q.weight");
                set_if(base + ".wk", blk + ".attn_k.weight");
                set_if(base + ".wv", blk + ".attn_v.weight");
                set_if(base + ".wo", blk + ".attn_output.weight");
                set_if(base + ".attn_q_norm", blk + ".attn_q_norm.weight");
                set_if(base + ".attn_k_norm", blk + ".attn_k_norm.weight");
                set_if(base + ".attn_post_norm", blk + ".post_attention_norm.weight");
                // Per-layer proportional RoPE frequency factors (full-attention layers only)
                set_if(base + ".rope_freqs", blk + ".rope_freqs.weight");
                // Post-FFN norm
                set_if(base + ".ffn_post_norm", blk + ".post_ffw_norm.weight");
                // MoE weights
                set_if(base + ".ffn_gate_inp", blk + ".ffn_gate_inp.weight");
                set_if(base + ".ffn_gate_inp_s", blk + ".ffn_gate_inp.scale");
                set_if(base + ".ffn_gate_exps", blk + ".ffn_gate_exps.weight");
                set_if(base + ".ffn_up_exps", blk + ".ffn_up_exps.weight");
                set_if(base + ".ffn_down_exps", blk + ".ffn_down_exps.weight");
                set_if(base + ".ffn_gate_up_exps", blk + ".ffn_gate_up_exps.weight");
                set_if(base + ".ffn_pre_norm_2", blk + ".ffn_pre_norm_2.weight");
                set_if(base + ".ffn_post_norm_1", blk + ".ffn_post_norm_1.weight");
                set_if(base + ".ffn_post_norm_2", blk + ".ffn_post_norm_2.weight");
                // Per-layer embeddings
                set_if(base + ".per_layer_inp_gate", blk + ".inp_gate.weight");
                set_if(base + ".per_layer_proj", blk + ".proj.weight");
                set_if(base + ".per_layer_post_norm", blk + ".post_norm.weight");
                // Layer output scale
                set_if(base + ".layer_out_scale", blk + ".layer_output_scale.weight");
            } else if (config_.arch_type == "qwen35") {
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
            } else if (config_.arch_type == "qwen3vl") {
                // Qwen3-VL: standard GQA with QK-norm and MRoPE
                set_if(base + ".wq", blk + ".attn_q.weight");
                set_if(base + ".wk", blk + ".attn_k.weight");
                set_if(base + ".wv", blk + ".attn_v.weight");
                set_if(base + ".wo", blk + ".attn_output.weight");
                set_if(base + ".attn_q_norm", blk + ".attn_q_norm.weight");
                set_if(base + ".attn_k_norm", blk + ".attn_k_norm.weight");
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
                // For Gemma4, SWA and full-attention layers have different head dimensions
                bool is_swa = (i < (int)config_.swa_layers.size() && config_.swa_layers[i] == 1);
                int layer_head_dim = is_swa ? config_.head_dim_swa : config_.head_dim;
                int layer_num_heads = is_swa ? config_.num_heads_swa : config_.num_heads;

                auto wq = weight_store_.get(base + ".wq");
                if (wq) {
                    int wq_rows = static_cast<int>(wq->shape()[0]);
                    // Use actual head dim inferred from tensor shape for robustness
                    int wq_head_dim = (layer_num_heads > 0) ? wq_rows / layer_num_heads : wq_rows / config_.num_heads;
                    int wq_num_heads = wq_rows / wq_head_dim;
                    weight_store_.set(base + ".wq",
                                      inverse_neox_permute_rows(wq, wq_num_heads, wq_head_dim));
                }

                auto wk = weight_store_.get(base + ".wk");
                if (wk) {
                    int wk_rows = static_cast<int>(wk->shape()[0]);
                    // For Gemma4, WK head_dim matches the layer's head_dim (SWA or full)
                    int wk_head_dim = layer_head_dim;
                    int wk_kv_heads = wk_rows / wk_head_dim;
                    if (wk_kv_heads > 0) {
                        weight_store_.set(base + ".wk",
                                          inverse_neox_permute_rows(wk, wk_kv_heads, wk_head_dim));
                    }
                }
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

void Model::set_quant_policy(const QuantPolicy& policy) {
    quant_policy_ = policy;
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

// ConfigParserRegistry, WeightInitRegistry, ArchCapabilityRegistry implementations
// and their static auto-registrations are defined in:
//   arch_config_parser.cpp, arch_weight_init.cpp, arch_capability.cpp
// They were removed from this file to avoid ODR violations that caused
// static auto-registrations to be silently discarded by the linker.

// Keep the fallback parse_common_gguf_config for parse_config_from_gguf's fallback path.
// (arch_config_parser.cpp has its own copy used by registered parsers.)

static ModelConfig parse_common_gguf_config(ModelLoader& loader, const std::string& arch) {
    ModelConfig cfg;
    cfg.vocab_size = static_cast<int>(loader.get_metadata_int(arch + ".vocab_size", 0));
    cfg.hidden_dim = static_cast<int>(loader.get_metadata_int(arch + ".embedding_length", 4096));
    cfg.intermediate_dim =
        static_cast<int>(loader.get_metadata_int(arch + ".feed_forward_length", 11008));
    cfg.num_layers = static_cast<int>(loader.get_metadata_int(arch + ".block_count", 32));
    cfg.num_heads = static_cast<int>(loader.get_metadata_int(arch + ".attention.head_count", 32));
    cfg.num_kv_heads =
        static_cast<int>(loader.get_metadata_int(arch + ".attention.head_count_kv", cfg.num_heads));
    cfg.head_dim = static_cast<int>(loader.get_metadata_int(arch + ".attention.key_length", 0));
    if (cfg.head_dim == 0)
        cfg.head_dim = cfg.hidden_dim / cfg.num_heads;
    cfg.rope_theta =
        static_cast<float>(loader.get_metadata_float(arch + ".rope.freq_base", 10000.0));
    cfg.rms_norm_eps = static_cast<float>(
        loader.get_metadata_float(arch + ".attention.layer_norm_rms_epsilon", 1e-6));
    cfg.max_seq_len = static_cast<int>(loader.get_metadata_int(arch + ".context_length", 4096));
    cfg.arch_type = arch;
    cfg.norm_type = NormType::RMSNorm;
    cfg.ffn_activation = ActivationType::SiLU_GELU;
    cfg.use_gqa = (cfg.num_kv_heads != cfg.num_heads);

    if (cfg.vocab_size == 0) {
        auto embd_shape = loader.get_tensor_shape("token_embd.weight");
        if (!embd_shape.empty() && embd_shape.size() >= 2) {
            cfg.vocab_size = static_cast<int>(embd_shape[0]);
        } else if (loader.has_tensor("token_embd.weight")) {
            cfg.vocab_size = 32000;
        }
        if (cfg.vocab_size == 0)
            cfg.vocab_size = 32000;
    }

    // Read suppress_tokens from GGUF metadata (for archs without a registered parser)
    cfg.suppress_tokens = loader.get_metadata_int_array("tokenizer.ggml.suppress_tokens", {});

    return cfg;
}

}  // namespace forge
