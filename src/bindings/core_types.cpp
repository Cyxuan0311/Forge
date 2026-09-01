#include "common.h"
#include "forge/context_config.h"

void register_core_types(py::module_& m) {
    // ---- Enums ----
    py::enum_<DataType>(m, "DataType")
        .value("FP32", DataType::FP32)
        .value("FP16", DataType::FP16)
        .value("Q4_0", DataType::Q4_0)
        .value("Q4_1", DataType::Q4_1)
        .value("Q4_K", DataType::Q4_K)
        .value("INT8", DataType::INT8)
        .value("INT32", DataType::INT32)
        .value("Q8_0", DataType::Q8_0)
        .value("Q5_0", DataType::Q5_0)
        .value("Q5_1", DataType::Q5_1)
        .value("Q2_K", DataType::Q2_K)
        .value("Q3_K", DataType::Q3_K)
        .value("Q5_K", DataType::Q5_K)
        .value("Q6_K", DataType::Q6_K)
        .value("IQ2_S", DataType::IQ2_S)
        .value("IQ2_XXS", DataType::IQ2_XXS)
        .value("IQ4_NL", DataType::IQ4_NL)
        .value("IQ2_XS", DataType::IQ2_XS)
        .value("IQ3_S", DataType::IQ3_S)
        .value("IQ4_XS", DataType::IQ4_XS)
        .value("BF16", DataType::BF16);

    py::enum_<DeviceType>(m, "DeviceType")
        .value("CPU", DeviceType::CPU)
        .value("CUDA", DeviceType::CUDA);

    py::enum_<LogLevel>(m, "LogLevel")
        .value("NONE", LogLevel::NONE)
        .value("LOG_ERROR", LogLevel::LOG_ERROR)
        .value("WARN", LogLevel::WARN)
        .value("INFO", LogLevel::INFO)
        .value("DEBUG", LogLevel::DEBUG)
        .value("TRACE", LogLevel::TRACE);

    py::enum_<NormType>(m, "NormType")
        .value("RMSNorm", NormType::RMSNorm)
        .value("LayerNorm", NormType::LayerNorm);

    py::enum_<ActivationType>(m, "ActivationType")
        .value("SiLU_GELU", ActivationType::SiLU_GELU)
        .value("GELU", ActivationType::GELU)
        .value("ReLU", ActivationType::ReLU);

    py::enum_<RopeType>(m, "RopeType")
        .value("None", RopeType::None)
        .value("Standard", RopeType::Standard)
        .value("LinearScaling", RopeType::LinearScaling)
        .value("NTK_Scaled", RopeType::NTK_Scaled)
        .value("NeoX", RopeType::NeoX)
        .value("MRoPE", RopeType::MRoPE)
        .value("Proportional", RopeType::Proportional);

    py::enum_<FFNType>(m, "FFNType")
        .value("SiLUGated", FFNType::SiLUGated)
        .value("GeGLU", FFNType::GeGLU)
        .value("SimpleGELU", FFNType::SimpleGELU)
        .value("MoE", FFNType::MoE);

    py::enum_<KVCacheDType>(m, "KVCacheDType")
        .value("FP32", KVCacheDType::FP32)
        .value("F16", KVCacheDType::F16)
        .value("Q8_0", KVCacheDType::Q8_0)
        .value("Q4_0", KVCacheDType::Q4_0)
        .value("Q4_K", KVCacheDType::Q4_K)
        .value("FP8_E4M3", KVCacheDType::FP8_E4M3)
        .value("FP8_E5M2", KVCacheDType::FP8_E5M2);

    // Phase 6: per-layer memory policy (None_ avoids the Python `None` keyword).
    py::enum_<KVLayerPolicy>(m, "KVLayerPolicy")
        .value("None_", KVLayerPolicy::None)
        .value("Full", KVLayerPolicy::Full)
        .value("SlidingWindow", KVLayerPolicy::SlidingWindow)
        .value("Recurrent", KVLayerPolicy::Recurrent)
        .export_values();

    // ---- QuantPolicy ----
    py::class_<QuantPolicy>(m, "QuantPolicy")
        .def(py::init<>())
        .def_readwrite("default_type", &QuantPolicy::default_type)
        .def_readwrite("attn_wv_type", &QuantPolicy::attn_wv_type)
        .def_readwrite("ffn_down_type", &QuantPolicy::ffn_down_type)
        .def_readwrite("output_type", &QuantPolicy::output_type)
        .def("enabled", &QuantPolicy::enabled)
        .def_static("q4_k_m", &QuantPolicy::q4_k_m);

    // ---- SpeculativeConfig ----
    // SpeculativeConfig is an aggregate (no multi-arg constructor), so the legacy
    // positional py::init<...> overloads below build it field-by-field via a
    // lambda (C++17-safe; aggregate cannot be parenthesized-constructed).
    py::class_<SpeculativeConfig>(m, "SpeculativeConfig")
        .def(py::init<>())
        // Legacy 10-arg overload for backward compat (positional).
        .def(py::init([](bool enabled, int n_draft, int n_min, float p_min, bool print_stats,
                         const std::string& draft_model_path, int draft_gpu_layers, bool use_ngram,
                         int ngram_n, int ngram_min) {
                 SpeculativeConfig c;
                 c.enabled = enabled;
                 c.n_draft = n_draft;
                 c.n_min = n_min;
                 c.p_min = p_min;
                 c.print_stats = print_stats;
                 c.draft_model_path = draft_model_path;
                 c.draft_gpu_layers = draft_gpu_layers;
                 c.use_ngram = use_ngram;
                 c.ngram_n = ngram_n;
                 c.ngram_min = ngram_min;
                 return c;
             }),
             py::arg("enabled") = false, py::arg("n_draft") = 5, py::arg("n_min") = 0,
             py::arg("p_min") = 0.0f, py::arg("print_stats") = false,
             py::arg("draft_model_path") = "", py::arg("draft_gpu_layers") = -1,
             py::arg("use_ngram") = true, py::arg("ngram_n") = 5, py::arg("ngram_min") = 2)
        // Legacy 15-arg overload for backward compat (positional).
        .def(py::init([](bool enabled, int n_draft, int n_min, float p_min, bool print_stats,
                         const std::string& draft_model_path, int draft_gpu_layers, bool use_mtp,
                         bool use_ngram, int ngram_n, int ngram_min, bool use_ngram_mod,
                         int ngram_mod_n, int ngram_mod_n_min, int ngram_mod_n_max) {
                 SpeculativeConfig c;
                 c.enabled = enabled;
                 c.n_draft = n_draft;
                 c.n_min = n_min;
                 c.p_min = p_min;
                 c.print_stats = print_stats;
                 c.draft_model_path = draft_model_path;
                 c.draft_gpu_layers = draft_gpu_layers;
                 c.use_mtp = use_mtp;
                 c.use_ngram = use_ngram;
                 c.ngram_n = ngram_n;
                 c.ngram_min = ngram_min;
                 c.use_ngram_mod = use_ngram_mod;
                 c.ngram_mod_n = ngram_mod_n;
                 c.ngram_mod_n_min = ngram_mod_n_min;
                 c.ngram_mod_n_max = ngram_mod_n_max;
                 return c;
             }),
             py::arg("enabled") = false, py::arg("n_draft") = 5, py::arg("n_min") = 0,
             py::arg("p_min") = 0.0f, py::arg("print_stats") = false,
             py::arg("draft_model_path") = "", py::arg("draft_gpu_layers") = -1,
             py::arg("use_mtp") = false, py::arg("use_ngram") = true, py::arg("ngram_n") = 5,
             py::arg("ngram_min") = 2, py::arg("use_ngram_mod") = false,
             py::arg("ngram_mod_n") = 24, py::arg("ngram_mod_n_min") = 48,
             py::arg("ngram_mod_n_max") = 64)
        .def_readwrite("enabled", &SpeculativeConfig::enabled)
        .def_readwrite("n_draft", &SpeculativeConfig::n_draft)
        .def_readwrite("n_min", &SpeculativeConfig::n_min)
        .def_readwrite("p_min", &SpeculativeConfig::p_min)
        .def_readwrite("draft_model_path", &SpeculativeConfig::draft_model_path)
        .def_readwrite("draft_gpu_layers", &SpeculativeConfig::draft_gpu_layers)
        .def_readwrite("use_mtp", &SpeculativeConfig::use_mtp)
        .def_readwrite("use_ngram", &SpeculativeConfig::use_ngram)
        .def_readwrite("ngram_n", &SpeculativeConfig::ngram_n)
        .def_readwrite("ngram_min", &SpeculativeConfig::ngram_min)
        .def_readwrite("use_ngram_mod", &SpeculativeConfig::use_ngram_mod)
        .def_readwrite("ngram_mod_n", &SpeculativeConfig::ngram_mod_n)
        .def_readwrite("ngram_mod_n_min", &SpeculativeConfig::ngram_mod_n_min)
        .def_readwrite("ngram_mod_n_max", &SpeculativeConfig::ngram_mod_n_max)
        .def_readwrite("ngram_mod_pool_size", &SpeculativeConfig::ngram_mod_pool_size)
        // DFlash / DSPark standalone drafter (DFLASH_DSPARK_PLAN.md, Phase 0).
        .def_readwrite("draft_arch", &SpeculativeConfig::draft_arch)
        .def_readwrite("draft_target_layers", &SpeculativeConfig::draft_target_layers)
        .def_readwrite("draft_mask_token_id", &SpeculativeConfig::draft_mask_token_id)
        .def_readwrite("draft_n_spec", &SpeculativeConfig::draft_n_spec);

    // ---- SpeculativeStats ----
    py::class_<SpeculativeStats>(m, "SpeculativeStats")
        .def_readwrite("n_spec_steps", &SpeculativeStats::n_spec_steps)
        .def_readwrite("n_fallback_steps", &SpeculativeStats::n_fallback_steps)
        .def_readwrite("n_draft_tokens", &SpeculativeStats::n_draft_tokens)
        .def_readwrite("n_accepted_tokens", &SpeculativeStats::n_accepted_tokens)
        .def_readwrite("n_output_tokens", &SpeculativeStats::n_output_tokens)
        .def("acceptance_rate", &SpeculativeStats::acceptance_rate)
        .def("tokens_per_step", &SpeculativeStats::tokens_per_step);

    // ---- GenerationConfig ----
    py::class_<GenerationConfig>(m, "GenerationConfig")
        .def(py::init<>())
        .def_readwrite("max_new_tokens", &GenerationConfig::max_new_tokens)
        .def_readwrite("temperature", &GenerationConfig::temperature)
        .def_readwrite("top_k", &GenerationConfig::top_k)
        .def_readwrite("top_p", &GenerationConfig::top_p)
        .def_readwrite("repeat_penalty", &GenerationConfig::repeat_penalty)
        .def_readwrite("repeat_last_n", &GenerationConfig::repeat_last_n)
        .def_readwrite("do_sample", &GenerationConfig::do_sample)
        .def_readwrite("seed", &GenerationConfig::seed)
        .def_readwrite("eos_token_id", &GenerationConfig::eos_token_id)
        .def_readwrite("stop_token_ids", &GenerationConfig::stop_token_ids)
        .def_readwrite("reset_kv_cache", &GenerationConfig::reset_kv_cache);

    // ---- GenerationResult ----
    py::class_<GenerationResult>(m, "GenerationResult")
        .def(py::init<>())
        .def_readonly("token_ids", &GenerationResult::token_ids)
        .def_readonly("text", &GenerationResult::text)
        .def_readonly("num_prompt_tokens", &GenerationResult::num_prompt_tokens)
        .def_readonly("num_generated_tokens", &GenerationResult::num_generated_tokens)
        .def_readonly("finished", &GenerationResult::finished)
        .def_readonly("finish_reason", &GenerationResult::finish_reason)
        .def_property_readonly("spec_stats", [](const GenerationResult& r) -> py::object {
            if (!r.spec_stats)
                return py::none();
            return py::cast(*r.spec_stats);
        });

    // ---- KVCachePrecision ----
    py::enum_<KVCachePrecision>(m, "KVCachePrecision")
        .value("AUTO", KVCachePrecision::AUTO)
        .value("HIGH_ACCURACY", KVCachePrecision::HIGH_ACCURACY)
        .value("HIGH_THROUGHPUT", KVCachePrecision::HIGH_THROUGHPUT)
        .export_values();

    // ---- ContextConfig ----
    py::class_<ContextConfig>(m, "ContextConfig")
        .def(py::init<>())
        .def_readwrite("kv_cache_dtype", &ContextConfig::kv_cache_dtype)
        .def_readwrite("kv_cache_precision", &ContextConfig::kv_cache_precision)
        .def_readwrite("kv_cache_type_k", &ContextConfig::kv_cache_type_k)
        .def_readwrite("kv_cache_type_v", &ContextConfig::kv_cache_type_v)
        .def_readwrite("kv_storage", &ContextConfig::kv_storage)
        .def_readwrite("page_size", &ContextConfig::page_size)
        .def_readwrite("max_seq_len", &ContextConfig::max_seq_len)
        .def_readwrite("max_num_seqs", &ContextConfig::max_num_seqs)
        .def_readwrite("gpu_layers", &ContextConfig::gpu_layers)
        .def_readwrite("n_batch", &ContextConfig::n_batch)
        .def_readwrite("n_ubatch", &ContextConfig::n_ubatch)
        .def_readwrite("n_threads", &ContextConfig::n_threads)
        .def_readwrite("n_threads_batch", &ContextConfig::n_threads_batch)
        .def_readwrite("prefix_cache", &ContextConfig::prefix_cache)
        .def_readwrite("prefix_cache_bytes", &ContextConfig::prefix_cache_bytes)
        .def_readwrite("swa_window", &ContextConfig::swa_window)
        .def_readwrite("device", &ContextConfig::device);

    // ---- Tensor ----
    py::class_<Tensor, TensorPtr>(m, "Tensor")
        .def(py::init<DataType, std::vector<int64_t>, DeviceType>())
        .def("shape", &Tensor::shape)
        .def("dtype", &Tensor::dtype)
        .def("device", &Tensor::device)
        .def("numel", &Tensor::numel)
        .def("nbytes", &Tensor::nbytes)
        .def("strides", [](const Tensor& t) -> std::vector<int64_t> { return t.strides(); })
        .def("zero_", &Tensor::zero_)
        .def(
            "copy_from", [](Tensor& self, const Tensor& other) { self.copy_from(other); },
            py::arg("other"))
        .def(
            "to_device",
            [](Tensor& t, DeviceType dev) {
                t.to_device(dev);
                return &t;
            },
            py::return_value_policy::reference_internal)
        .def(
            "view",
            [](const Tensor& t, const std::vector<int64_t>& new_shape) {
                return std::make_shared<Tensor>(t.view(new_shape));
            },
            py::arg("new_shape"))
        .def(
            "slice",
            [](const Tensor& t, int64_t dim, int64_t start, int64_t end) {
                return std::make_shared<Tensor>(t.slice(dim, start, end));
            },
            py::arg("dim"), py::arg("start"), py::arg("end"))
        .def("byte_offset", &Tensor::byte_offset, "Byte offset of data() relative to storage base")
        .def("allocation_bytes", &Tensor::allocation_bytes,
             "Bytes needed for allocation (planner uses this)")
        .def_static(
            "from_buffer",
            [](uintptr_t ptr, DataType dtype, const std::vector<int64_t>& shape, DeviceType device,
               bool own) {
                return std::make_shared<Tensor>(
                    Tensor::from_buffer(reinterpret_cast<void*>(ptr), dtype, shape, device, own));
            },
            py::arg("ptr"), py::arg("dtype"), py::arg("shape"), py::arg("device") = DeviceType::CPU,
            py::arg("own") = false)
        .def("numpy", [](TensorPtr& t) { return tensor_to_numpy(t); });

    // ---- dtype helper functions (quant_traits.h) ----
    m.def("dtype_size", &dtype_size, py::arg("dt"),
          "Element size in bytes (0 for quantized types)");
    m.def("dtype_name", &dtype_name, py::arg("dt"), "Human-readable dtype name");
    m.def("dtype_block_size", &dtype_block_size, py::arg("dt"),
          "Bytes per quantization block (0 for non-quantized)");
    m.def("dtype_block_elements", &dtype_block_elements, py::arg("dt"),
          "Elements per quantization block (1 for non-quantized)");
    m.def("is_quantized_type", &is_quantized_type, py::arg("dt"),
          "Whether this DataType is a quantized format");
    m.def("compute_quantized_bytes", &compute_quantized_bytes, py::arg("numel"), py::arg("dt"),
          "Compute bytes needed for numel elements of quantized type");
    m.def(
        "get_dequant_row_fn",
        [](DataType dt) -> uintptr_t {
            return reinterpret_cast<uintptr_t>(get_dequant_row_fn(dt));
        },
        py::arg("dt"), "Get dequant_row function pointer (0 if N/A)");

    // ---- Memory counters (Phase 0 baseline) ----
    m.def(
        "get_memory_counters",
        []() -> py::dict {
            auto snap = MemoryCounters::instance().snapshot();
            py::dict d;
            d["cpu_malloc"] = snap.cpu_malloc;
            d["cpu_free"] = snap.cpu_free;
            d["cuda_malloc"] = snap.cuda_malloc;
            d["cuda_free"] = snap.cuda_free;
            d["h2d_copies"] = snap.h2d_copies;
            d["d2h_copies"] = snap.d2h_copies;
            d["d2d_copies"] = snap.d2d_copies;
            d["h2d_bytes"] = snap.h2d_bytes;
            d["d2h_bytes"] = snap.d2h_bytes;
            d["d2d_bytes"] = snap.d2d_bytes;
            return d;
        },
        "Get current memory allocation/copy counters");
    m.def(
        "reset_memory_counters", []() { MemoryCounters::instance().reset(); },
        "Reset all memory counters to zero");

    // ---- ModelConfig ----
    py::class_<ModelConfig>(m, "ModelConfig")
        .def(py::init<>())
        .def_readwrite("vocab_size", &ModelConfig::vocab_size)
        .def_readwrite("hidden_dim", &ModelConfig::hidden_dim)
        .def_readwrite("intermediate_dim", &ModelConfig::intermediate_dim)
        .def_readwrite("num_layers", &ModelConfig::num_layers)
        .def_readwrite("num_heads", &ModelConfig::num_heads)
        .def_readwrite("num_kv_heads", &ModelConfig::num_kv_heads)
        .def_readwrite("head_dim", &ModelConfig::head_dim)
        .def_readwrite("rope_theta", &ModelConfig::rope_theta)
        .def_readwrite("rms_norm_eps", &ModelConfig::rms_norm_eps)
        .def_readwrite("max_seq_len", &ModelConfig::max_seq_len)
        .def_readwrite("arch_type", &ModelConfig::arch_type)
        .def_readwrite("tie_embeddings", &ModelConfig::tie_embeddings)
        .def_readwrite("use_gqa", &ModelConfig::use_gqa)
        .def_readwrite("use_neox_rope", &ModelConfig::use_neox_rope)
        .def_readwrite("norm_type", &ModelConfig::norm_type)
        .def_readwrite("ffn_activation", &ModelConfig::ffn_activation)
        .def_readwrite("use_ssm", &ModelConfig::use_ssm)
        .def_readwrite("ssm_group_count", &ModelConfig::ssm_group_count)
        .def_readwrite("ssm_time_step_rank", &ModelConfig::ssm_time_step_rank)
        .def_readwrite("ssm_inner_size", &ModelConfig::ssm_inner_size)
        .def_readwrite("ssm_state_size", &ModelConfig::ssm_state_size)
        .def_readwrite("ssm_conv_kernel", &ModelConfig::ssm_conv_kernel)
        .def_readwrite("full_attention_interval", &ModelConfig::full_attention_interval)
        .def_readwrite("rope_dimension_count", &ModelConfig::rope_dimension_count)
        .def_readwrite("use_mrope", &ModelConfig::use_mrope)
        .def_readwrite("f_attn_logit_softcapping", &ModelConfig::f_attn_logit_softcapping)
        .def_readwrite("f_final_logit_softcapping", &ModelConfig::f_final_logit_softcapping)
        .def_readwrite("use_parallel_residual", &ModelConfig::use_parallel_residual)
        .def_readwrite("n_embd_per_layer", &ModelConfig::n_embd_per_layer)
        .def_readwrite("n_ff_exp", &ModelConfig::n_ff_exp)
        .def_readwrite("n_expert", &ModelConfig::n_expert)
        .def_readwrite("n_expert_used", &ModelConfig::n_expert_used)
        .def_readwrite("n_swa", &ModelConfig::n_swa)
        .def_readwrite("n_layer_kv_from_start", &ModelConfig::n_layer_kv_from_start)
        .def_readwrite("use_qk_norm", &ModelConfig::use_qk_norm)
        .def_readwrite("head_dim_swa", &ModelConfig::head_dim_swa)
        .def_readwrite("num_heads_swa", &ModelConfig::num_heads_swa)
        .def_readwrite("num_kv_heads_swa", &ModelConfig::num_kv_heads_swa)
        .def_readwrite("suppress_tokens", &ModelConfig::suppress_tokens)
        .def_readwrite("rope_type", &ModelConfig::rope_type)
        .def_readwrite("ffn_type", &ModelConfig::ffn_type)
        .def_readwrite("rope_q_scale", &ModelConfig::rope_q_scale)
        .def_readwrite("has_post_attention_norm", &ModelConfig::has_post_attention_norm)
        .def_readwrite("has_post_ffn_norm", &ModelConfig::has_post_ffn_norm);

    // ---- VisionConfig ----
    py::class_<VisionConfig>(m, "VisionConfig")
        .def(py::init<>())
        .def_readwrite("image_size", &VisionConfig::image_size)
        .def_readwrite("patch_size", &VisionConfig::patch_size)
        .def_readwrite("embedding_length", &VisionConfig::embedding_length)
        .def_readwrite("feed_forward_length", &VisionConfig::feed_forward_length)
        .def_readwrite("block_count", &VisionConfig::block_count)
        .def_readwrite("head_count", &VisionConfig::head_count)
        .def_readwrite("projection_dim", &VisionConfig::projection_dim)
        .def_readwrite("scale_factor", &VisionConfig::scale_factor)
        .def_readwrite("insert_layer_id", &VisionConfig::insert_layer_id);
}
