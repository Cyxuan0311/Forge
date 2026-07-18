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
        .value("IQ2_S", DataType::IQ2_S);

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
        .value("NTK_Scaled", RopeType::NTK_Scaled);

    py::enum_<KVCacheDType>(m, "KVCacheDType")
        .value("FP32", KVCacheDType::FP32)
        .value("Q4_0", KVCacheDType::Q4_0);

    // ---- Tensor ----
    py::class_<Tensor, TensorPtr>(m, "Tensor")
        .def(py::init<DataType, std::vector<int64_t>, DeviceType>())
        .def("shape", &Tensor::shape)
        .def("dtype", &Tensor::dtype)
        .def("device", &Tensor::device)
        .def("numel", &Tensor::numel)
        .def("nbytes", &Tensor::nbytes)
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
        .def("numpy", [](TensorPtr& t) { return tensor_to_numpy(t); });

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
        .def_readwrite("suppress_tokens", &ModelConfig::suppress_tokens);

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
