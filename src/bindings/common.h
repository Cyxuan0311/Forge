#pragma once
// Shared header for Forge Python bindings modules.

#include <cstdlib>
#include <memory>
#include <string>

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../core/platform.h"
#include "../core/memory_counters.h"
#include "forge/arch_registry.h"
#include "forge/backend.h"
#include "forge/compute_graph.h"
#include "forge/context.h"
#include "forge/context_config.h"
#include "forge/engine.h"
#include "forge/engines/transformer_engine.h"
#include "forge/generator.h"
#include "forge/gguf_model.h"
#include "forge/inference/forward_request.h"
#include "forge/logger.h"
#include "forge/model.h"
#include "forge/model_loader.h"
#include "forge/ninf_model.h"
#include "forge/operators.h"
#include "forge/paged_kv_cache.h"
#include "forge/perf_profiler.h"
#include "forge/request_scheduler.h"
#include "forge/sampler.h"
#include "forge/speculative.h"
#include "forge/tensor.h"
#include "forge/tokenizer.h"
#include "forge/vision_encoder.h"
#include "forge/vision_registry.h"

namespace py = pybind11;
using namespace forge;

// Set by forge.set_num_threads(); create_context() applies it to the engine
// ContextParams (engine forward overrides the global omp thread count).
extern int forge_global_cpu_threads;

// ---- Registration helpers ----

inline void ensure_engines_registered() {
    // Force arch_registrations.cpp to be linked (contains all static auto-registrations).
    // Without this reference, the linker may discard that translation unit in .so builds.
    (void)forge::_arch_registrations_linked;
    // Trigger static initializers by accessing the registry.
    (void)EngineRegistry::instance().registered_archs();
}

inline void ensure_loaders_registered() {
    static bool registered = false;
    if (registered)
        return;
    registered = true;

    auto& reg = ModelLoaderRegistry::instance();
    reg.register_loader(
        "ninf", []() -> std::unique_ptr<ModelLoader> { return std::make_unique<NinfModel>(); });
    reg.register_loader(
        "gguf", []() -> std::unique_ptr<ModelLoader> { return std::make_unique<GgufModel>(); });
}

// Force-link vision_registry.cpp (contains all vision auto-registrations).
inline void ensure_vision_registered() {
    (void)forge::_vision_registrations_linked;
    (void)VisionEncoderRegistry::instance().registered_encoders();
    (void)VisionConfigParserRegistry::instance().registered_parsers();
}

// Force static initializers in shared libraries by touching registry instances.
// This ensures ConfigParserAutoRegister, WeightInitAutoRegister, and
// ArchCapabilityAutoRegister from arch_config_parser.cpp, arch_weight_init.cpp,
// and arch_capability.cpp are executed even if the linker would otherwise
// discard those translation units.
inline void ensure_config_and_weight_registered() {
    static bool registered = false;
    if (registered)
        return;
    registered = true;

    // Trigger static registrations by accessing the singleton instances.
    // The first access causes the static initializers in the linked translation
    // units to run (if they haven't already).
    (void)ConfigParserRegistry::instance().has("");
    (void)WeightInitRegistry::instance().has("");
    (void)ArchCapabilityRegistry::instance().has("");
}

// ---- Utility ----

inline py::array_t<float> tensor_to_numpy(const TensorPtr& tensor) {
    auto cpu_tensor = tensor;
    if (tensor->device() == DeviceType::CUDA) {
        auto t = std::make_shared<Tensor>(DataType::FP32, tensor->shape(), DeviceType::CPU);
        t->copy_from(*tensor);
        cpu_tensor = t;
    }

    auto shape = cpu_tensor->shape();
    std::vector<ssize_t> np_shape(shape.begin(), shape.end());
    std::vector<ssize_t> np_strides(shape.size());
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        np_strides[i] = static_cast<ssize_t>(cpu_tensor->strides()[i]) * sizeof(float);
    }

    return py::array_t<float>(np_shape, np_strides, static_cast<const float*>(cpu_tensor->data()),
                              py::cast(cpu_tensor));
}

// ---- Utility ----

// Parse KV cache dtype string. Supports: fp32, f16, q8_0, q4_0, q4_k
inline KVCacheDType parse_kv_dtype(const std::string& s) {
    if (s == "f16")  return KVCacheDType::F16;
    if (s == "q8_0") return KVCacheDType::Q8_0;
    if (s == "q4_0") return KVCacheDType::Q4_0;
    if (s == "q4_k") return KVCacheDType::Q4_K;
    return KVCacheDType::FP32;  // default
}

// ---- Wrapper classes ----

class PyInferenceContext {
public:
    explicit PyInferenceContext(Model& model) : ctx_(model) {}

    InferenceContext& get() { return ctx_; }

    void reset_kv() { ctx_.reset_kv_cache(); }
    void reset() { ctx_.reset(); }
    void warmup() { ctx_.warmup(); }
    void set_gpu_layers(int layers) { ctx_.set_gpu_layers(layers); }
    int gpu_layers() const { return ctx_.gpu_layers(); }
    void set_use_graph(bool use_graph) {
        auto* engine = ctx_.engine();
        if (!engine)
            throw std::runtime_error("No inference engine available");
        auto* tfm_eng = dynamic_cast<TransformerEngine*>(engine);
        if (tfm_eng) {
            tfm_eng->set_use_graph(use_graph);
        }
    }
    bool use_graph() const {
        auto* engine = const_cast<InferenceContext&>(ctx_).engine();
        if (!engine)
            return false;
        auto* tfm_eng = dynamic_cast<TransformerEngine*>(engine);
        return tfm_eng ? tfm_eng->use_graph() : false;
    }
    void set_cuda_graph_enabled(bool v) {
        auto* engine = ctx_.engine();
        if (!engine)
            throw std::runtime_error("No inference engine available");
        auto* tfm_eng = dynamic_cast<TransformerEngine*>(engine);
        if (tfm_eng) {
            tfm_eng->set_cuda_graph_enabled(v);
        }
    }
    bool cuda_graph_enabled() const {
        auto* engine = const_cast<InferenceContext&>(ctx_).engine();
        if (!engine)
            return false;
        auto* tfm_eng = dynamic_cast<TransformerEngine*>(engine);
        return tfm_eng ? tfm_eng->cuda_graph_enabled() : false;
    }
    DeviceType device() const { return ctx_.device(); }

    // ---- Phase 1: Generator primitives (reuse persistent KV cache) ----

    GenerationResult generate(const std::vector<int32_t>& tokens,
                              const GenerationConfig& config = GenerationConfig{}) {
        SamplerConfig sampler_cfg;
        sampler_cfg.temperature = config.temperature;
        sampler_cfg.top_k = config.top_k;
        sampler_cfg.top_p = config.top_p;
        sampler_cfg.repeat_penalty = config.repeat_penalty;
        sampler_cfg.repeat_last_n = config.repeat_last_n;
        sampler_cfg.do_sample = config.do_sample;
        sampler_cfg.seed = config.seed;
        sampler_cfg.logit_softcapping = ctx_.model().config().f_final_logit_softcapping;

        Generator gen(ctx_, sampler_cfg);
        {
            py::gil_scoped_release release;
            return gen.generate(tokens, config);
        }
    }

    void generate_stream(const std::vector<int32_t>& tokens,
                         const GenerationConfig& config,
                         py::object callback) {
        SamplerConfig sampler_cfg;
        sampler_cfg.temperature = config.temperature;
        sampler_cfg.top_k = config.top_k;
        sampler_cfg.top_p = config.top_p;
        sampler_cfg.repeat_penalty = config.repeat_penalty;
        sampler_cfg.repeat_last_n = config.repeat_last_n;
        sampler_cfg.do_sample = config.do_sample;
        sampler_cfg.seed = config.seed;
        sampler_cfg.logit_softcapping = ctx_.model().config().f_final_logit_softcapping;

        Generator gen(ctx_, sampler_cfg);

        auto token_cb = [&callback](int32_t token_id, int step) {
            py::gil_scoped_acquire acquire;
            callback(token_id, step);
        };

        {
            py::gil_scoped_release release;
            gen.generate(tokens, config, token_cb);
        }
    }

    // ---- Phase 2: KV-preserving generate (multi-turn context reuse) ----
    // Identical to generate()/generate_stream() but explicitly named to indicate
    // context-driven generation with configurable KV reset.

    GenerationResult generate_kv(const std::vector<int32_t>& tokens,
                                 const GenerationConfig& config = GenerationConfig{}) {
        return generate(tokens, config);
    }

    void generate_stream_kv(const std::vector<int32_t>& tokens,
                            const GenerationConfig& config,
                            py::object callback) {
        return generate_stream(tokens, config, std::move(callback));
    }


    py::array_t<float> forward(py::array_t<int32_t, py::array::c_style> input_ids,
                               int start_pos = 0) {
        auto buf = input_ids.request();
        if (buf.ndim != 1)
            throw std::runtime_error("input_ids must be 1D");

        int seq_len = static_cast<int>(buf.shape[0]);
        DeviceType dev = ctx_.device();

        auto ids_tensor = std::make_shared<Tensor>(DataType::INT32, std::vector<int64_t>{seq_len},
                                                   DeviceType::CPU);
        std::memcpy(ids_tensor->data(), buf.ptr, seq_len * sizeof(int32_t));

        if (dev == DeviceType::CUDA) {
            ids_tensor->to_device(DeviceType::CUDA);
        }

        auto* engine = ctx_.engine();
        if (!engine)
            throw std::runtime_error("No inference engine available");

        auto logits = engine->forward_request(ForwardRequest::from_ids(ids_tensor, start_pos));
        return tensor_to_numpy(logits);
    }

    int32_t forward_sample(py::array_t<int32_t, py::array::c_style> input_ids,
                           int start_pos, float temperature, int top_k, float top_p,
                           float repeat_penalty, const std::vector<int32_t>& token_history) {
        auto buf = input_ids.request();
        if (buf.ndim != 1)
            throw std::runtime_error("input_ids must be 1D");

        int seq_len = static_cast<int>(buf.shape[0]);
        DeviceType dev = ctx_.device();

        auto ids_tensor = std::make_shared<Tensor>(DataType::INT32,
                                                    std::vector<int64_t>{seq_len},
                                                    DeviceType::CPU);
        std::memcpy(ids_tensor->data(), buf.ptr, seq_len * sizeof(int32_t));

        if (dev == DeviceType::CUDA) {
            ids_tensor->to_device(DeviceType::CUDA);
        }

        auto* engine = ctx_.engine();
        if (!engine)
            throw std::runtime_error("No inference engine available");

        auto logits = engine->forward_request(ForwardRequest::from_ids(ids_tensor, start_pos));

        SamplerConfig sampler_cfg;
        sampler_cfg.temperature = temperature;
        sampler_cfg.top_k = top_k;
        sampler_cfg.top_p = top_p;
        sampler_cfg.repeat_penalty = repeat_penalty;
        sampler_cfg.do_sample = (temperature > 0.0f);
        sampler_cfg.logit_softcapping = ctx_.model().config().f_final_logit_softcapping;
        Sampler sampler(sampler_cfg);
        for (int32_t tid : token_history) {
            sampler.add_token_to_history(tid);
        }
        return static_cast<int32_t>(sampler.sample(logits, start_pos));
    }

    py::array_t<float> forward_with_embeddings(py::array_t<float, py::array::c_style> embeddings,
                                               int start_pos = 0) {
        auto buf = embeddings.request();
        if (buf.ndim != 2)
            throw std::runtime_error("embeddings must be 2D (seq_len, hidden_dim)");

        int seq_len = static_cast<int>(buf.shape[0]);
        int hidden_dim = static_cast<int>(buf.shape[1]);

        auto hidden_tensor = std::make_shared<Tensor>(
            DataType::FP32, std::vector<int64_t>{seq_len, hidden_dim}, DeviceType::CPU);
        std::memcpy(hidden_tensor->data(), buf.ptr, seq_len * hidden_dim * sizeof(float));

        DeviceType dev = ctx_.device();
        if (dev == DeviceType::CUDA) {
            hidden_tensor->to_device(DeviceType::CUDA);
        }

        auto* engine = ctx_.engine();
        if (!engine)
            throw std::runtime_error("No inference engine available");

        auto logits = engine->forward_from_hidden(hidden_tensor, start_pos);
        return tensor_to_numpy(logits);
    }

    py::array_t<float> get_embeddings(py::array_t<int32_t, py::array::c_style> input_ids) {
        auto buf = input_ids.request();
        if (buf.ndim != 1)
            throw std::runtime_error("input_ids must be 1D");

        int seq_len = static_cast<int>(buf.shape[0]);
        const int32_t* ids = static_cast<const int32_t*>(buf.ptr);

        const auto& model = ctx_.model();
        auto token_emb = model.weights().get("token_embedding");
        if (!token_emb)
            throw std::runtime_error("token_embedding weight not found");

        auto ids_tensor = std::make_shared<Tensor>(DataType::INT32, std::vector<int64_t>{seq_len},
                                                   DeviceType::CPU);
        std::memcpy(ids_tensor->data(), ids, seq_len * sizeof(int32_t));

        // If token_embedding is on CUDA, move indices to CUDA as well
        // (CUDA embedding kernels require indices on the same device)
        if (token_emb->device() == DeviceType::CUDA) {
            ids_tensor->to_device(DeviceType::CUDA);
        }

        auto hidden = ops::embedding(token_emb, ids_tensor);
        if (!hidden)
            throw std::runtime_error("embedding lookup failed");

        return tensor_to_numpy(hidden);
    }

    py::dict memory_stats() const {
        py::dict stats;
        auto* engine = ctx_.engine();
        if (!engine)
            return stats;
        auto* tfm_eng = dynamic_cast<const TransformerEngine*>(engine);
        if (!tfm_eng)
            return stats;

        const KVCache* cache = tfm_eng->kv_cache();
        stats["kv_cache_nbytes"] = static_cast<int64_t>(cache->nbytes());
        stats["kv_cache_active_bytes"] = static_cast<int64_t>(cache->active_bytes());
        stats["kv_cache_free_slots"] = cache->num_free_slots();
        stats["kv_cache_filled"] = cache->filled(0);  // layer 0 representative
        stats["kv_cache_page_size"] = 0;  // contiguous mode: no pages
        stats["kv_cache_free_pages"] = 0;
        // Report per-K/V types
        auto dtype_str = [](KVCacheDType dt) -> const char* {
            switch (dt) {
            case KVCacheDType::FP32: return "fp32";
            case KVCacheDType::F16:  return "f16";
            case KVCacheDType::Q8_0: return "q8_0";
            case KVCacheDType::Q4_0: return "q4_0";
            case KVCacheDType::Q4_K: return "q4_k";
            default: return "unknown";
            }
        };
        stats["kv_cache_dtype"] = dtype_str(cache->kv_dtype());
        stats["kv_cache_type_k"] = dtype_str(cache->type_k());
        stats["kv_cache_type_v"] = dtype_str(cache->type_v());
        // Phase 5: prefix cache stats (from KVMemory)
        auto* kv_mem = tfm_eng->kv_memory();
        if (kv_mem) {
            stats["prefix_cache_hits"] = kv_mem->prefix_hits();
            stats["prefix_cache_tokens"] = kv_mem->prefix_tokens();
        } else {
            stats["prefix_cache_hits"] = 0;
            stats["prefix_cache_tokens"] = 0;
        }
        return stats;
    }

    // ---- Sequence-level KV operations (Phase 5: Session fork / prefix sharing) ----

    bool seq_share(int src_seq, int dst_seq, int64_t p0, int64_t p1) {
        auto* engine = ctx_.engine();
        if (!engine) return false;
        auto* tfm_eng = dynamic_cast<TransformerEngine*>(engine);
        if (!tfm_eng || !tfm_eng->kv_memory()) return false;
        return tfm_eng->kv_memory()->seq_share(src_seq, dst_seq, p0, p1);
    }

    bool seq_remove(int seq_id, int64_t p0, int64_t p1) {
        auto* engine = ctx_.engine();
        if (!engine) return false;
        auto* tfm_eng = dynamic_cast<TransformerEngine*>(engine);
        if (!tfm_eng || !tfm_eng->kv_memory()) return false;
        return tfm_eng->kv_memory()->seq_remove(seq_id, p0, p1);
    }

    bool seq_keep(int seq_id) {
        auto* engine = ctx_.engine();
        if (!engine) return false;
        auto* tfm_eng = dynamic_cast<TransformerEngine*>(engine);
        if (!tfm_eng || !tfm_eng->kv_memory()) return false;
        return tfm_eng->kv_memory()->seq_keep(seq_id);
    }

    void seq_release(int seq_id) {
        auto* engine = ctx_.engine();
        if (!engine) return;
        auto* tfm_eng = dynamic_cast<TransformerEngine*>(engine);
        if (!tfm_eng || !tfm_eng->kv_memory()) return;
        tfm_eng->kv_memory()->release(seq_id);
    }

    bool is_paged() const {
        auto* engine = ctx_.engine();
        if (!engine) return false;
        auto* tfm_eng = dynamic_cast<const TransformerEngine*>(engine);
        if (!tfm_eng || !tfm_eng->kv_memory()) return false;
        return tfm_eng->kv_memory()->is_paged();
    }

private:
    InferenceContext ctx_;
};

class PyModel {
public:
    PyModel() = default;

    void load(const std::string& path, int vocab_size, int hidden_dim, int intermediate_dim,
              int num_layers, int num_heads, int num_kv_heads, int head_dim, float rope_theta,
              float rms_norm_eps, int max_seq_len, const std::string& arch_type,
              const std::string& norm_type_str, const std::string& activation_str,
              bool tie_embeddings, const std::string& device_str,
              int n_swa = 0, const std::vector<int>& swa_layers = {}) {
        ensure_loaders_registered();
        ensure_engines_registered();
        ModelConfig cfg;
        cfg.vocab_size = vocab_size;
        cfg.hidden_dim = hidden_dim;
        cfg.intermediate_dim = intermediate_dim;
        cfg.num_layers = num_layers;
        cfg.num_heads = num_heads;
        cfg.num_kv_heads = num_kv_heads > 0 ? num_kv_heads : num_heads;
        cfg.head_dim = head_dim > 0 ? head_dim : hidden_dim / num_heads;
        cfg.rope_theta = rope_theta;
        cfg.rms_norm_eps = rms_norm_eps;
        cfg.max_seq_len = max_seq_len;
        cfg.arch_type = arch_type;
        cfg.tie_embeddings = tie_embeddings;
        cfg.n_swa = n_swa;
        if (!swa_layers.empty()) cfg.swa_layers = swa_layers;

        if (norm_type_str == "layernorm") {
            cfg.norm_type = NormType::LayerNorm;
        } else {
            cfg.norm_type = NormType::RMSNorm;
        }

        if (activation_str == "gelu") {
            cfg.ffn_activation = ActivationType::GELU;
        } else if (activation_str == "relu") {
            cfg.ffn_activation = ActivationType::ReLU;
        } else {
            cfg.ffn_activation = ActivationType::SiLU_GELU;
        }

        cfg.use_gqa = (cfg.num_kv_heads != cfg.num_heads);

        DeviceType dev =
            (device_str == "cuda" || device_str == "cuda:0") ? DeviceType::CUDA : DeviceType::CPU;

        if (!model_.load_with_config(path, cfg, dev)) {
            throw std::runtime_error("Failed to load model from: " + path);
        }
    }

    void load_gguf(const std::string& path, const std::string& device_str,
                   const QuantPolicy& policy = QuantPolicy{}) {
        ensure_loaders_registered();
        ensure_engines_registered();
        DeviceType dev =
            (device_str == "cuda" || device_str == "cuda:0") ? DeviceType::CUDA : DeviceType::CPU;

        if (policy.enabled()) {
            model_.set_quant_policy(policy);
        }

        if (!model_.load(path, dev)) {
            throw std::runtime_error("Failed to load GGUF model from: " + path);
        }
    }

    void load_auto(const std::string& path, const std::string& device_str) {
        ensure_loaders_registered();
        ensure_engines_registered();
        DeviceType dev =
            (device_str == "cuda" || device_str == "cuda:0") ? DeviceType::CUDA : DeviceType::CPU;

        if (!model_.load(path, dev)) {
            throw std::runtime_error("Failed to load model from: " + path);
        }
    }

    PyInferenceContext* create_context(const std::string& kv_cache_dtype_str, int gpu_layers,
                                       bool offload_kqv = true,
                                       const forge::SpeculativeConfig& speculative_config = {}) {
        ensure_engines_registered();

        auto ctx = std::make_unique<PyInferenceContext>(model_);

        // Apply the forge.set_num_threads() value to the engine context so the
        // engine's per-forward omp_set_num_threads() doesn't reset it to the
        // ContextParams default (4).
        ctx->get().params_mut().n_threads = forge_global_cpu_threads;
        ctx->get().params_mut().n_threads_batch = forge_global_cpu_threads;

        // Internal feature flag: enable paged KV storage via environment variable.
        // Not exposed as a Python API parameter (Phase 3 requirement).
        const char* storage_mode_env = std::getenv("FORGE_KV_STORAGE_MODE");
        if (storage_mode_env && std::string(storage_mode_env) == "paged") {
            ctx->get().params_mut().kv_storage_mode = KVStorageMode::Paged;
        }

        ctx->get().params_mut().offload_kqv = offload_kqv;
        ctx->get().params_mut().speculative_config = speculative_config;

        const auto& cfg = model_.config();
        auto engine = EngineRegistry::instance().create(cfg.arch_type, model_, ctx->get());
        if (!engine) {
            throw std::runtime_error("No engine registered for arch: " + cfg.arch_type);
        }

        auto* tfm_eng = dynamic_cast<TransformerEngine*>(engine.get());
        if (tfm_eng) {
            tfm_eng->set_kv_cache_dtype(parse_kv_dtype(kv_cache_dtype_str));
            tfm_eng->set_gpu_layers(gpu_layers);
        }

        ctx->get().set_engine(std::move(engine));
        return ctx.release();
    }

    py::dict generate(py::array_t<int32_t, py::array::c_style> prompt_ids, int max_new_tokens,
                      float temperature, int top_k, float top_p, float repeat_penalty,
                      bool do_sample, uint64_t seed, int eos_token_id,
                      const std::string& kv_cache_dtype_str, int gpu_layers,
                      const std::vector<int32_t>& stop_token_ids = {});
    void generate_stream(py::array_t<int32_t, py::array::c_style> prompt_ids, py::object callback,
                         int max_new_tokens, float temperature, int top_k, float top_p,
                         float repeat_penalty, bool do_sample, uint64_t seed, int eos_token_id,
                         const std::string& kv_cache_dtype_str, int gpu_layers,
                         const std::vector<int32_t>& stop_token_ids = {});

    const ModelConfig& config() const { return model_.config(); }
    Model& get_model() { return model_; }
    DeviceType device() const { return model_.device(); }

    py::list registered_archs() const {
        ensure_engines_registered();
        auto archs = EngineRegistry::instance().registered_archs();
        py::list result;
        for (const auto& a : archs)
            result.append(a);
        return result;
    }

    static std::string detect_format(const std::string& path) {
        ensure_loaders_registered();
        return Model::detect_format(path);
    }

    /// Phase 1: ensure a default context exists (lazily created on first use).
    /// The default context preserves KV cache across generate() calls.
    /// Phase 2: FORGE_DISABLE_CONTEXT_REUSE=1 disables caching (rollback to old path).
    PyInferenceContext* ensure_default_context(const std::string& kv_cache_dtype_str = "fp32",
                                                int gpu_layers = -1) {
        // Phase 2 rollback: disable context reuse via env var
        const char* disable_reuse = std::getenv("FORGE_DISABLE_CONTEXT_REUSE");
        if (disable_reuse && std::string(disable_reuse) == "1") {
            default_ctx_.reset(create_context(kv_cache_dtype_str, gpu_layers));
            return default_ctx_.get();
        }
        if (default_ctx_) {
            // Already created; the dtype/layers are locked to the first call's values.
            return default_ctx_.get();
        }
        default_ctx_.reset(create_context(kv_cache_dtype_str, gpu_layers));
        return default_ctx_.get();
    }

    /// Phase 1: release the default context (explicit reset).
    void release_default_context() {
        default_ctx_.reset();
    }

    /// Access the default context (may be nullptr if never created).
    PyInferenceContext* default_context() const { return default_ctx_.get(); }

private:
    Model model_;
    std::unique_ptr<PyInferenceContext> default_ctx_;  // Phase 1: persistent KV cache context
};

class PyRequestScheduler {
public:
    // Accept py::object (not PyModel&) so we can store a strong reference,
    // preventing the Python Model from being garbage-collected while the
    // scheduler holds a C++ Model& reference to it.
    PyRequestScheduler(py::object model, int block_size = 16, int max_num_seqs = 4)
        : model_ref_(std::move(model)),
          scheduler_(init_scheduler(model_ref_.cast<PyModel&>().get_model(), block_size,
                                    max_num_seqs)) {}

    static RequestScheduler init_scheduler(Model& model, int block_size, int max_num_seqs) {
        ensure_engines_registered();
        ensure_loaders_registered();
        return RequestScheduler(model, block_size, max_num_seqs);
    }

    int submit(const std::vector<int32_t>& prompt_tokens, int max_new_tokens = 256,
               int eos_token_id = -1, const SamplerConfig& sampler_cfg = SamplerConfig{}) {
        return scheduler_.submit(prompt_tokens, max_new_tokens, eos_token_id, sampler_cfg);
    }

    bool step() { return scheduler_.step(); }

    py::list get_finished() {
        auto finished = scheduler_.get_finished();
        py::list result;
        for (auto& req : finished)
            result.append(std::move(req));
        return result;
    }

    int num_active() const { return scheduler_.num_active(); }
    int num_waiting() const { return scheduler_.num_waiting(); }
    bool has_pending() const { return scheduler_.has_pending(); }
    void abort(int request_id) { scheduler_.abort(request_id); }
    void reset() { scheduler_.reset(); }

    int prefix_cache_hits() const { return scheduler_.prefix_cache_hits(); }
    int prefix_cache_misses() const { return scheduler_.prefix_cache_misses(); }

    // Phase 6: expose per-layer page pool sizes for SWA isolation verification.
    py::dict memory_stats() const {
        py::dict stats;
        auto* engine = scheduler_.context().engine();
        if (!engine) return stats;
        auto* tfm_eng = dynamic_cast<const TransformerEngine*>(engine);
        if (!tfm_eng) return stats;

        auto* mem = tfm_eng->kv_memory();
        auto* cache = tfm_eng->kv_cache();

        if (mem && mem->is_paged()) {
            auto& storage = mem->storage();
            stats["kv_cache_nbytes"] = static_cast<int64_t>(storage.nbytes());
            stats["kv_cache_active_bytes"] = static_cast<int64_t>(storage.active_bytes());
            stats["kv_cache_free_slots"] = storage.num_free_slots();
            stats["kv_cache_page_size"] = storage.page_size();
            stats["kv_cache_free_pages"] = storage.num_free_pages();
            stats["kv_cache_filled"] = storage.filled(0);
            // Phase 6: per-layer pool max_pages (SWA layers should be smaller)
            auto pool_sizes = storage.layer_pool_max_pages();
            py::list pool_list;
            for (int s : pool_sizes) pool_list.append(s);
            stats["layer_pool_max_pages"] = pool_list;
        } else {
            stats["kv_cache_nbytes"] = static_cast<int64_t>(cache->nbytes());
            stats["kv_cache_active_bytes"] = static_cast<int64_t>(cache->active_bytes());
            stats["kv_cache_free_slots"] = cache->num_free_slots();
            stats["kv_cache_filled"] = cache->filled(0);
            stats["kv_cache_page_size"] = 0;
            stats["kv_cache_free_pages"] = 0;
        }
        // Phase 5: prefix cache stats
        stats["prefix_cache_hits"] = prefix_cache_hits();
        stats["prefix_cache_misses"] = prefix_cache_misses();
        if (mem) {
            stats["prefix_cache_tokens"] = mem->prefix_tokens();
        } else {
            stats["prefix_cache_tokens"] = 0;
        }
        return stats;
    }

    // Phase 6: high-level generate() – submit → step loop → collect finished
    py::list generate(const std::vector<int32_t>& prompt_tokens,
                      const GenerationConfig& gen_cfg = GenerationConfig{},
                      const SamplerConfig& sampler_cfg = SamplerConfig{}) {
        py::list result;
        scheduler_.submit(prompt_tokens, gen_cfg.max_new_tokens,
                          gen_cfg.eos_token_id, sampler_cfg);
        while (scheduler_.has_pending()) {
            scheduler_.step();
            auto finished = scheduler_.get_finished();
            for (auto& req : finished)
                result.append(std::move(req));
        }
        return result;
    }

    int n_batch() const { return scheduler_.context().params().n_batch; }
    void set_n_batch(int v) { scheduler_.context().params_mut().n_batch = v; }
    int n_ubatch() const { return scheduler_.context().params().n_ubatch; }
    void set_n_ubatch(int v) { scheduler_.context().params_mut().n_ubatch = v; }

    int n_threads() const { return scheduler_.context().params().n_threads; }
    void set_n_threads(int v) { scheduler_.context().params_mut().n_threads = v; }
    int n_threads_batch() const { return scheduler_.context().params().n_threads_batch; }
    void set_n_threads_batch(int v) { scheduler_.context().params_mut().n_threads_batch = v; }

private:
    py::object model_ref_;    // keeps PyModel alive (prevents GC of Model&)
    RequestScheduler scheduler_;
};

class PyLogger {
public:
    static void set_level(int level) { Logger::instance().set_level(static_cast<LogLevel>(level)); }
    static int level() { return static_cast<int>(Logger::instance().level()); }
    static void set_python_sink(const py::function& fn) {
        Logger::instance().set_sink(
            [fn](LogLevel lvl, const std::string& msg) { fn(static_cast<int>(lvl), msg); });
    }
    static void reset_sink() { Logger::instance().reset_sink(); }
};

class PyMultimodalModel {
public:
    PyMultimodalModel() = default;

    void load(const std::string& model_path, const std::string& device_str = "cuda") {
        load(model_path, "", device_str);
    }

    void load(const std::string& model_path, const std::string& mmproj_path,
              const std::string& device_str);

    py::array_t<float> encode_image(py::array_t<uint8_t, py::array::c_style> image);

    py::dict generate(py::array_t<int32_t, py::array::c_style> prompt_ids, int max_new_tokens,
                      float temperature, int top_k, float top_p, float repeat_penalty,
                      bool do_sample, uint64_t seed, int eos_token_id,
                      const std::string& kv_cache_dtype_str, int gpu_layers,
                      const std::vector<int32_t>& stop_token_ids = {});

    void generate_stream(py::array_t<int32_t, py::array::c_style> prompt_ids, py::object callback,
                         int max_new_tokens, float temperature, int top_k, float top_p,
                         float repeat_penalty, bool do_sample, uint64_t seed, int eos_token_id,
                         const std::string& kv_cache_dtype_str, int gpu_layers,
                         const std::vector<int32_t>& stop_token_ids = {});

    const ModelConfig& config() const { return model_.config(); }
    const VisionConfig& vision_config() const {
        static const VisionConfig empty{};
        return vision_ ? vision_->config() : empty;
    }

    PyInferenceContext* create_context(const std::string& kv_cache_dtype_str = "fp32",
                                       int gpu_layers = -1, bool offload_kqv = true,
                                       const forge::SpeculativeConfig& speculative_config = {}) {
        auto ctx = new PyInferenceContext(model_);

        ctx->get().params_mut().offload_kqv = offload_kqv;
        ctx->get().params_mut().speculative_config = speculative_config;

        auto engine =
            EngineRegistry::instance().create(model_.config().arch_type, model_, ctx->get());
        if (!engine) {
            delete ctx;
            throw std::runtime_error("No engine registered for arch: " + model_.config().arch_type);
        }

        auto* tfm_eng = dynamic_cast<TransformerEngine*>(engine.get());
        if (tfm_eng) {
            tfm_eng->set_kv_cache_dtype(parse_kv_dtype(kv_cache_dtype_str));
            tfm_eng->set_gpu_layers(gpu_layers);
        }

        ctx->get().set_engine(std::move(engine));
        return ctx;
    }

private:
    Model model_;
    std::unique_ptr<VisionEncoder> vision_;
};

// ---- Module registration functions ----

void register_core_types(py::module_& m);
void register_model(py::module_& m);
void register_tokenizer(py::module_& m);
void register_chat_template(py::module_& m);
void register_multimodal(py::module_& m);
void register_scheduler(py::module_& m);
void register_backend(py::module_& m);
void register_logger(py::module_& m);
void register_profiler(py::module_& m);
