#include "common.h"

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
        .value("Q4_K", KVCacheDType::Q4_K);

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
    py::class_<SpeculativeConfig>(m, "SpeculativeConfig")
        .def(py::init<>())
        .def_readwrite("n_draft", &SpeculativeConfig::n_draft)
        .def_readwrite("p_min", &SpeculativeConfig::p_min)
        .def_readwrite("use_ngram", &SpeculativeConfig::use_ngram)
        .def_readwrite("ngram_n", &SpeculativeConfig::ngram_n)
        .def_readwrite("ngram_min", &SpeculativeConfig::ngram_min)
        .def_readwrite("enabled", &SpeculativeConfig::enabled);

    // ---- Tensor ----
    py::class_<Tensor, TensorPtr>(m, "Tensor")
        .def(py::init<DataType, std::vector<int64_t>, DeviceType>())
        .def("shape", &Tensor::shape)
        .def("dtype", &Tensor::dtype)
        .def("device", &Tensor::device)
        .def("numel", &Tensor::numel)
        .def("nbytes", &Tensor::nbytes)
        .def(
            "strides",
            [](const Tensor& t) -> std::vector<int64_t> { return t.strides(); })
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
            py::arg("ptr"), py::arg("dtype"), py::arg("shape"),
            py::arg("device") = DeviceType::CPU, py::arg("own") = false)
        .def("numpy", [](TensorPtr& t) { return tensor_to_numpy(t); });

    // ---- dtype helper functions (quant_traits.h) ----
    m.def("dtype_size", &dtype_size, py::arg("dt"), "Element size in bytes (0 for quantized types)");
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
    m.def("get_memory_counters", []() -> py::dict {
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
    }, "Get current memory allocation/copy counters");
    m.def("reset_memory_counters", []() {
        MemoryCounters::instance().reset();
    }, "Reset all memory counters to zero");

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
