#include "common.h"

// ---- PyModel method implementations (require full context) ----

py::dict PyModel::generate(py::array_t<int32_t, py::array::c_style> prompt_ids, int max_new_tokens,
                           float temperature, int top_k, float top_p, float repeat_penalty,
                           bool do_sample, uint64_t seed, int eos_token_id,
                           const std::string& kv_cache_dtype_str, int gpu_layers,
                           const std::vector<int32_t>& stop_token_ids) {
    // Phase 1: 旧签名保留，内部复用持久 context。打 DeprecationWarning。
    PyErr_WarnEx(PyExc_DeprecationWarning,
                 "Model.generate() with scalar parameters is deprecated; "
                 "use ctx.generate(tokens, GenerationConfig(...)) for KV cache reuse",
                 1);

    auto* ctx = ensure_default_context(kv_cache_dtype_str, gpu_layers);

    auto buf = prompt_ids.request();
    if (buf.ndim != 1)
        throw std::runtime_error("prompt_ids must be 1D");
    int prompt_len = static_cast<int>(buf.shape[0]);
    std::vector<int32_t> tokens(prompt_len);
    std::memcpy(tokens.data(), buf.ptr, prompt_len * sizeof(int32_t));

    GenerationConfig gen_cfg;
    gen_cfg.max_new_tokens = max_new_tokens;
    gen_cfg.temperature = temperature;
    gen_cfg.top_k = top_k;
    gen_cfg.top_p = top_p;
    gen_cfg.repeat_penalty = repeat_penalty;
    gen_cfg.do_sample = do_sample;
    gen_cfg.seed = seed;
    gen_cfg.eos_token_id = eos_token_id;
    gen_cfg.stop_token_ids = stop_token_ids;

    SamplerConfig sampler_cfg;
    sampler_cfg.temperature = temperature;
    sampler_cfg.top_k = top_k;
    sampler_cfg.top_p = top_p;
    sampler_cfg.repeat_penalty = repeat_penalty;
    sampler_cfg.do_sample = do_sample;
    sampler_cfg.seed = seed;
    sampler_cfg.logit_softcapping = model_.config().f_final_logit_softcapping;
    Generator gen(ctx->get(), sampler_cfg);
    auto result = gen.generate(tokens, gen_cfg);

    py::dict out;
    out["token_ids"] = py::cast(result.token_ids);
    out["text"] = result.text;
    out["num_prompt_tokens"] = result.num_prompt_tokens;
    out["num_generated_tokens"] = result.num_generated_tokens;
    out["finished"] = result.finished;
    out["finish_reason"] = result.finish_reason;
    return out;
}

void PyModel::generate_stream(py::array_t<int32_t, py::array::c_style> prompt_ids,
                              py::object callback, int max_new_tokens, float temperature, int top_k,
                              float top_p, float repeat_penalty, bool do_sample, uint64_t seed,
                              int eos_token_id, const std::string& kv_cache_dtype_str,
                              int gpu_layers, const std::vector<int32_t>& stop_token_ids) {
    // Phase 1: 旧签名保留，内部复用持久 context。打 DeprecationWarning。
    PyErr_WarnEx(PyExc_DeprecationWarning,
                 "Model.generate_stream() with scalar parameters is deprecated; "
                 "use ctx.generate_stream(tokens, callback, GenerationConfig(...))",
                 1);

    auto* ctx = ensure_default_context(kv_cache_dtype_str, gpu_layers);

    auto buf = prompt_ids.request();
    if (buf.ndim != 1)
        throw std::runtime_error("prompt_ids must be 1D");
    int prompt_len = static_cast<int>(buf.shape[0]);
    std::vector<int32_t> tokens(prompt_len);
    std::memcpy(tokens.data(), buf.ptr, prompt_len * sizeof(int32_t));

    GenerationConfig gen_cfg;
    gen_cfg.max_new_tokens = max_new_tokens;
    gen_cfg.temperature = temperature;
    gen_cfg.top_k = top_k;
    gen_cfg.top_p = top_p;
    gen_cfg.repeat_penalty = repeat_penalty;
    gen_cfg.do_sample = do_sample;
    gen_cfg.seed = seed;
    gen_cfg.eos_token_id = eos_token_id;
    gen_cfg.stop_token_ids = stop_token_ids;

    SamplerConfig sampler_cfg;
    sampler_cfg.temperature = temperature;
    sampler_cfg.top_k = top_k;
    sampler_cfg.top_p = top_p;
    sampler_cfg.repeat_penalty = repeat_penalty;
    sampler_cfg.do_sample = do_sample;
    sampler_cfg.seed = seed;
    sampler_cfg.logit_softcapping = model_.config().f_final_logit_softcapping;
    Generator gen(ctx->get(), sampler_cfg);

    auto token_cb = [&callback](int32_t token_id, int step) {
        py::gil_scoped_acquire acquire;
        callback(token_id, step);
    };

    {
        py::gil_scoped_release release;
        gen.generate(tokens, gen_cfg, token_cb);
    }
}

void register_model(py::module_& m) {
    py::class_<PyModel>(m, "Model")
        .def(py::init<>())
        .def("load", &PyModel::load, py::arg("path"), py::arg("vocab_size"), py::arg("hidden_dim"),
             py::arg("intermediate_dim"), py::arg("num_layers"), py::arg("num_heads"),
             py::arg("num_kv_heads") = 0, py::arg("head_dim") = 0, py::arg("rope_theta") = 10000.0f,
             py::arg("rms_norm_eps") = 1e-6f, py::arg("max_seq_len") = 4096,
             py::arg("arch_type") = "llama", py::arg("norm_type") = "rmsnorm",
             py::arg("activation") = "silu_gelu", py::arg("tie_embeddings") = false,
             py::arg("device") = "cuda", py::arg("n_swa") = 0,
             py::arg("swa_layers") = std::vector<int>{})
        .def("load_gguf", &PyModel::load_gguf, py::arg("path"), py::arg("device") = "cuda",
             py::arg("quant_policy") = QuantPolicy{})
        .def("load_auto", &PyModel::load_auto, py::arg("path"), py::arg("device") = "cuda")
        .def(
            "load_vision_weights",
            [](PyModel& self, const std::string& mmproj_path, const std::string& device_str) {
                DeviceType dev = (device_str == "cuda" || device_str == "cuda:0") ? DeviceType::CUDA
                                                                                  : DeviceType::CPU;
                return self.get_model().load_vision_weights(mmproj_path, dev);
            },
            py::arg("mmproj_path"), py::arg("device") = "cpu")
        .def("create_context", &PyModel::create_context, py::arg("kv_cache_dtype") = "fp32",
             py::arg("gpu_layers") = -1, py::arg("kv_cache_precision") = "auto",
             py::arg("offload_kqv") = true,
             py::arg("speculative_config") = forge::SpeculativeConfig{},
             py::return_value_policy::take_ownership)
        .def("generate", &PyModel::generate, py::arg("prompt_ids"), py::arg("max_new_tokens") = 256,
             py::arg("temperature") = 1.0f, py::arg("top_k") = 0, py::arg("top_p") = 1.0f,
             py::arg("repeat_penalty") = 1.0f, py::arg("do_sample") = true, py::arg("seed") = 0,
             py::arg("eos_token_id") = -1, py::arg("kv_cache_dtype") = "fp32",
             py::arg("gpu_layers") = -1, py::arg("stop_token_ids") = std::vector<int32_t>{})
        .def("generate_stream", &PyModel::generate_stream, py::arg("prompt_ids"),
             py::arg("callback"), py::arg("max_new_tokens") = 256, py::arg("temperature") = 1.0f,
             py::arg("top_k") = 0, py::arg("top_p") = 1.0f, py::arg("repeat_penalty") = 1.0f,
             py::arg("do_sample") = true, py::arg("seed") = 0, py::arg("eos_token_id") = -1,
             py::arg("kv_cache_dtype") = "fp32", py::arg("gpu_layers") = -1,
             py::arg("stop_token_ids") = std::vector<int32_t>{})
        .def("registered_archs", &PyModel::registered_archs)
        .def("detect_format", &PyModel::detect_format)
        .def_property_readonly("config", &PyModel::config,
                               py::return_value_policy::reference_internal)
        .def_property_readonly("device", &PyModel::device)
        .def("release_default_context", &PyModel::release_default_context)
        .def_property_readonly("default_context", &PyModel::default_context,
                               py::return_value_policy::reference_internal)
        .def("ensure_default_context", &PyModel::ensure_default_context,
             py::arg("kv_cache_dtype") = "fp32", py::arg("gpu_layers") = -1,
             py::arg("kv_cache_precision") = "auto", py::return_value_policy::reference_internal);

    py::class_<PyInferenceContext>(m, "InferenceContext")
        .def("forward", &PyInferenceContext::forward, py::arg("input_ids"),
             py::arg("start_pos") = 0)
        .def("forward_sample", &PyInferenceContext::forward_sample, py::arg("input_ids"),
             py::arg("start_pos"), py::arg("temperature"), py::arg("top_k"), py::arg("top_p"),
             py::arg("repeat_penalty"), py::arg("token_history"))
        .def("forward_with_embeddings", &PyInferenceContext::forward_with_embeddings,
             py::arg("embeddings"), py::arg("start_pos") = 0)
        .def("get_embeddings", &PyInferenceContext::get_embeddings, py::arg("input_ids"))
        .def("reset_kv", &PyInferenceContext::reset_kv)
        .def("reset", &PyInferenceContext::reset)
        .def("warmup", &PyInferenceContext::warmup)
        .def("set_gpu_layers", &PyInferenceContext::set_gpu_layers)
        .def("gpu_layers", &PyInferenceContext::gpu_layers)
        .def("set_use_graph", &PyInferenceContext::set_use_graph, py::arg("use_graph"))
        .def("use_graph", &PyInferenceContext::use_graph)
        .def("set_cuda_graph_enabled", &PyInferenceContext::set_cuda_graph_enabled, py::arg("v"))
        .def("cuda_graph_enabled", &PyInferenceContext::cuda_graph_enabled)
        .def("memory_stats", &PyInferenceContext::memory_stats)
        // Phase 5: seq-level KV operations for Session fork / prefix sharing
        .def("seq_share", &PyInferenceContext::seq_share, py::arg("src_seq"), py::arg("dst_seq"),
             py::arg("p0"), py::arg("p1"))
        .def("seq_remove", &PyInferenceContext::seq_remove, py::arg("seq_id"), py::arg("p0"),
             py::arg("p1"))
        .def("seq_keep", &PyInferenceContext::seq_keep, py::arg("seq_id"))
        .def("seq_release", &PyInferenceContext::seq_release, py::arg("seq_id"))
        .def("is_paged", &PyInferenceContext::is_paged)
        .def_property(
            "n_batch", [](PyInferenceContext& self) { return self.get().params().n_batch; },
            [](PyInferenceContext& self, int v) { self.get().params_mut().n_batch = v; })
        .def_property(
            "n_ubatch", [](PyInferenceContext& self) { return self.get().params().n_ubatch; },
            [](PyInferenceContext& self, int v) { self.get().params_mut().n_ubatch = v; })
        .def_property(
            "n_threads", [](PyInferenceContext& self) { return self.get().params().n_threads; },
            [](PyInferenceContext& self, int v) { self.get().params_mut().n_threads = v; })
        .def_property(
            "n_threads_batch",
            [](PyInferenceContext& self) { return self.get().params().n_threads_batch; },
            [](PyInferenceContext& self, int v) { self.get().params_mut().n_threads_batch = v; })
        .def("generate", &PyInferenceContext::generate, py::arg("tokens"),
             py::arg("config") = GenerationConfig{},
             "Generate tokens using this context (KV cache may be reused across calls).")
        .def("generate_stream", &PyInferenceContext::generate_stream, py::arg("tokens"),
             py::arg("config"), py::arg("callback"), "Streaming generation using this context.")
        // Phase 2: explicit KV-driven generation aliases (multi-turn context reuse)
        .def("generate_kv", &PyInferenceContext::generate_kv, py::arg("tokens"),
             py::arg("config") = GenerationConfig{},
             "Legacy Compatibility: Alias for generate() to explicitly indicate context-KV "
             "generation.")
        .def("generate_stream_kv", &PyInferenceContext::generate_stream_kv, py::arg("tokens"),
             py::arg("config"), py::arg("callback"),
             "Legacy Compatibility: Alias for generate_stream() to explicitly indicate context-KV "
             "generation.")
        .def_property_readonly("device", &PyInferenceContext::device);
}
