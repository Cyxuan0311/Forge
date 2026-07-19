#pragma once
// Shared header for Forge Python bindings modules.

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../core/platform.h"
#include "forge/arch_registry.h"
#include "forge/backend.h"
#include "forge/compute_graph.h"
#include "forge/context.h"
#include "forge/engine.h"
#include "forge/engines/transformer_engine.h"
#include "forge/generator.h"
#include "forge/gguf_model.h"
#include "forge/logger.h"
#include "forge/model.h"
#include "forge/model_loader.h"
#include "forge/ninf_model.h"
#include "forge/operators.h"
#include "forge/paged_kv_cache.h"
#include "forge/perf_profiler.h"
#include "forge/request_scheduler.h"
#include "forge/sampler.h"
#include "forge/tensor.h"
#include "forge/tokenizer.h"
#include "forge/vision_encoder.h"

namespace py = pybind11;
using namespace forge;

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
    DeviceType device() const { return ctx_.device(); }

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

        auto logits = engine->forward(ids_tensor, start_pos);
        return tensor_to_numpy(logits);
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

        const auto& cache = tfm_eng->kv_cache();
        stats["kv_cache_nbytes"] = static_cast<int64_t>(cache.nbytes());
        stats["kv_cache_dtype"] = (cache.kv_dtype() == KVCacheDType::Q4_0) ? "q4_0" : "fp32";
        return stats;
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
              bool tie_embeddings, const std::string& device_str) {
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

    void load_gguf(const std::string& path, const std::string& device_str) {
        ensure_loaders_registered();
        ensure_engines_registered();
        DeviceType dev =
            (device_str == "cuda" || device_str == "cuda:0") ? DeviceType::CUDA : DeviceType::CPU;

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

    PyInferenceContext* create_context(const std::string& kv_cache_dtype_str, int gpu_layers) {
        ensure_engines_registered();

        auto ctx = std::make_unique<PyInferenceContext>(model_);

        const auto& cfg = model_.config();
        auto engine = EngineRegistry::instance().create(cfg.arch_type, model_, ctx->get());
        if (!engine) {
            throw std::runtime_error("No engine registered for arch: " + cfg.arch_type);
        }

        auto* tfm_eng = dynamic_cast<TransformerEngine*>(engine.get());
        if (tfm_eng) {
            KVCacheDType kv_dtype = KVCacheDType::FP32;
            if (kv_cache_dtype_str == "q4_0")
                kv_dtype = KVCacheDType::Q4_0;
            tfm_eng->set_kv_cache_dtype(kv_dtype);
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

private:
    Model model_;
};

class PyRequestScheduler {
public:
    PyRequestScheduler(PyModel& model, int block_size = 16, int max_num_seqs = 4)
        : scheduler_(init_scheduler(model.get_model(), block_size, max_num_seqs)) {}

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

private:
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
    const VisionConfig& vision_config() const { return vision_.config(); }

    PyInferenceContext* create_context(const std::string& kv_cache_dtype_str = "fp32",
                                       int gpu_layers = -1) {
        auto ctx = new PyInferenceContext(model_);

        auto engine =
            EngineRegistry::instance().create(model_.config().arch_type, model_, ctx->get());
        if (!engine) {
            delete ctx;
            throw std::runtime_error("No engine registered for arch: " + model_.config().arch_type);
        }

        auto* tfm_eng = dynamic_cast<TransformerEngine*>(engine.get());
        if (tfm_eng) {
            KVCacheDType kv_dtype = KVCacheDType::FP32;
            if (kv_cache_dtype_str == "q4_0")
                kv_dtype = KVCacheDType::Q4_0;
            tfm_eng->set_kv_cache_dtype(kv_dtype);
            tfm_eng->set_gpu_layers(gpu_layers);
        }

        ctx->get().set_engine(std::move(engine));
        return ctx;
    }

private:
    Model model_;
    VisionEncoder vision_;
};

// ---- Module registration functions ----

void register_core_types(py::module_& m);
void register_model(py::module_& m);
void register_tokenizer(py::module_& m);
void register_multimodal(py::module_& m);
void register_scheduler(py::module_& m);
void register_backend(py::module_& m);
void register_logger(py::module_& m);
void register_profiler(py::module_& m);
