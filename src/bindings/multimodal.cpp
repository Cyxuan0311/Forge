#include "common.h"

// ---- PyMultimodalModel method implementations ----

void PyMultimodalModel::load(const std::string& model_path, const std::string& mmproj_path,
                             const std::string& device_str) {
    ensure_loaders_registered();
    ensure_engines_registered();

    DeviceType dev =
        (device_str == "cuda" || device_str == "cuda:0") ? DeviceType::CUDA : DeviceType::CPU;

    if (!model_.load(model_path, dev)) {
        throw std::runtime_error("Failed to load model from: " + model_path);
    }

    if (!mmproj_path.empty()) {
        if (!model_.load_vision_weights(mmproj_path, DeviceType::CPU)) {
            throw std::runtime_error("Failed to load mmproj from: " + mmproj_path);
        }
    }

    VisionConfig vis_cfg;
    const auto& cfg = model_.config();

    // Read vision config from mmproj file first (it has the most complete metadata)
    // Then fall back to LLM file for any missing values
    if (!mmproj_path.empty()) {
        auto mmproj_loader = ModelLoaderRegistry::instance().create_loader(mmproj_path);
        if (mmproj_loader && mmproj_loader->load(mmproj_path)) {
            vis_cfg.image_size =
                static_cast<int>(mmproj_loader->get_metadata_int("clip.vision.image_size", 448));
            vis_cfg.patch_size =
                static_cast<int>(mmproj_loader->get_metadata_int("clip.vision.patch_size", 14));
            vis_cfg.embedding_length =
                static_cast<int>(mmproj_loader->get_metadata_int("v.embedding_length", 1152));
            vis_cfg.feed_forward_length =
                static_cast<int>(mmproj_loader->get_metadata_int("v.feed_forward_length", 4304));
            vis_cfg.block_count =
                static_cast<int>(mmproj_loader->get_metadata_int("v.block_count", 27));
            vis_cfg.head_count =
                static_cast<int>(mmproj_loader->get_metadata_int("v.attention.head_count", 16));
            vis_cfg.projection_dim = cfg.hidden_dim;
            vis_cfg.scale_factor = static_cast<int>(
                mmproj_loader->get_metadata_int("clip.vision.projector.scale_factor", 4));
            // wa_layer_indexes stores the ViT merger insertion point (insert_layer_id)
            auto wa_layers =
                mmproj_loader->get_metadata_int_array("clip.vision.wa_layer_indexes", {});
            if (!wa_layers.empty()) {
                vis_cfg.insert_layer_id = static_cast<int>(wa_layers[0]);
            } else {
                vis_cfg.insert_layer_id = static_cast<int>(
                    mmproj_loader->get_metadata_int("clip.vision.insert_layer_id", 6));
            }
            mmproj_loader->close();
        }
    }

    // Supplement from LLM file
    auto loader = ModelLoaderRegistry::instance().create_loader(model_path);
    if (loader && loader->load(model_path)) {
        // Override image_size from LLM if mmproj didn't set it
        if (vis_cfg.image_size == 448) {
            vis_cfg.image_size = static_cast<int>(
                loader->get_metadata_int("minicpmv.image_size", vis_cfg.image_size));
        }
        vis_cfg.patch_size =
            static_cast<int>(loader->get_metadata_int("minicpmv.patch_size", vis_cfg.patch_size));
        // If insert_layer_id still not set, try LLM file
        if (vis_cfg.insert_layer_id < 0) {
            auto wa_layers = loader->get_metadata_int_array("clip.vision.wa_layer_indexes", {});
            if (!wa_layers.empty()) {
                vis_cfg.insert_layer_id = static_cast<int>(wa_layers[0]);
            }
        }
        loader->close();
    }

    if (!vision_.init(model_.weights(), vis_cfg)) {
        LOG_WARN("VisionEncoder initialization failed - model may not have vision components");
    }
}

py::array_t<float> PyMultimodalModel::encode_image(py::array_t<uint8_t, py::array::c_style> image) {
    auto buf = image.request();
    int height, width, channels;

    if (buf.ndim == 3) {
        channels = static_cast<int>(buf.shape[2]);
        height = static_cast<int>(buf.shape[0]);
        width = static_cast<int>(buf.shape[1]);
    } else if (buf.ndim == 2) {
        channels = 1;
        height = static_cast<int>(buf.shape[0]);
        width = static_cast<int>(buf.shape[1]);
    } else {
        throw std::runtime_error("Image must be 2D (H,W) or 3D (H,W,C)");
    }

    const uint8_t* raw = static_cast<const uint8_t*>(buf.ptr);
    std::vector<float> rgb(height * width * 3, 0.0f);

    if (channels == 3) {
        for (int i = 0; i < height * width * 3; ++i) {
            rgb[i] = static_cast<float>(raw[i]);
        }
    } else if (channels == 4) {
        for (int i = 0; i < height * width; ++i) {
            rgb[i * 3 + 0] = static_cast<float>(raw[i * 4 + 0]);
            rgb[i * 3 + 1] = static_cast<float>(raw[i * 4 + 1]);
            rgb[i * 3 + 2] = static_cast<float>(raw[i * 4 + 2]);
        }
    } else if (channels == 1) {
        for (int i = 0; i < height * width; ++i) {
            rgb[i * 3 + 0] = static_cast<float>(raw[i]);
            rgb[i * 3 + 1] = static_cast<float>(raw[i]);
            rgb[i * 3 + 2] = static_cast<float>(raw[i]);
        }
    }

    auto embeddings = vision_.encode(rgb.data(), width, height, 3);
    int num_tokens = embeddings.size() / vision_.config().projection_dim;

    return py::array_t<float>({num_tokens, vision_.config().projection_dim}, embeddings.data());
}

py::dict PyMultimodalModel::generate(py::array_t<int32_t, py::array::c_style> prompt_ids,
                                     int max_new_tokens, float temperature, int top_k, float top_p,
                                     float repeat_penalty, bool do_sample, uint64_t seed,
                                     int eos_token_id, const std::string& kv_cache_dtype_str,
                                     int gpu_layers,
                                     const std::vector<int32_t>& stop_token_ids) {
    auto ctx = std::make_unique<InferenceContext>(model_);
    auto engine = EngineRegistry::instance().create(model_.config().arch_type, model_, *ctx);
    if (!engine) {
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
    ctx->set_engine(std::move(engine));

    auto buf = prompt_ids.request();
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

    Generator gen(*ctx);
    auto result = gen.generate(tokens, gen_cfg);

    py::dict out;
    out["token_ids"] = py::cast(result.token_ids);
    out["num_prompt_tokens"] = result.num_prompt_tokens;
    out["num_generated_tokens"] = result.num_generated_tokens;
    out["finished"] = result.finished;
    out["finish_reason"] = result.finish_reason;
    return out;
}

void PyMultimodalModel::generate_stream(py::array_t<int32_t, py::array::c_style> prompt_ids,
                                        py::object callback, int max_new_tokens, float temperature,
                                        int top_k, float top_p, float repeat_penalty,
                                        bool do_sample, uint64_t seed, int eos_token_id,
                                        const std::string& kv_cache_dtype_str, int gpu_layers,
                                        const std::vector<int32_t>& stop_token_ids) {
    auto ctx = std::make_unique<InferenceContext>(model_);
    auto engine = EngineRegistry::instance().create(model_.config().arch_type, model_, *ctx);
    if (!engine) {
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
    ctx->set_engine(std::move(engine));

    auto buf = prompt_ids.request();
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

    Generator gen(*ctx);

    auto token_cb = [&callback](int32_t token_id, int step) {
        py::gil_scoped_acquire acquire;
        callback(token_id, step);
    };

    {
        py::gil_scoped_release release;
        gen.generate(tokens, gen_cfg, token_cb);
    }
}

void register_multimodal(py::module_& m) {
    py::class_<PyMultimodalModel>(m, "MultimodalModel")
        .def(py::init<>())
        .def("load",
             (void(PyMultimodalModel::*)(const std::string&, const std::string&)) &
                 PyMultimodalModel::load,
             py::arg("model_path"), py::arg("device") = "cuda")
        .def("load_with_mmproj",
             (void(PyMultimodalModel::*)(const std::string&, const std::string&,
                                         const std::string&)) &
                 PyMultimodalModel::load,
             py::arg("model_path"), py::arg("mmproj_path"), py::arg("device") = "cuda")
        .def("encode_image", &PyMultimodalModel::encode_image, py::arg("image"))
        .def("generate", &PyMultimodalModel::generate, py::arg("prompt_ids"),
             py::arg("max_new_tokens") = 256, py::arg("temperature") = 1.0f, py::arg("top_k") = 0,
             py::arg("top_p") = 1.0f, py::arg("repeat_penalty") = 1.0f, py::arg("do_sample") = true,
             py::arg("seed") = 0, py::arg("eos_token_id") = -1, py::arg("kv_cache_dtype") = "fp32",
             py::arg("gpu_layers") = -1, py::arg("stop_token_ids") = std::vector<int32_t>{})
        .def("generate_stream", &PyMultimodalModel::generate_stream, py::arg("prompt_ids"),
             py::arg("callback"), py::arg("max_new_tokens") = 256, py::arg("temperature") = 1.0f,
             py::arg("top_k") = 0, py::arg("top_p") = 1.0f, py::arg("repeat_penalty") = 1.0f,
             py::arg("do_sample") = true, py::arg("seed") = 0, py::arg("eos_token_id") = -1,
             py::arg("kv_cache_dtype") = "fp32", py::arg("gpu_layers") = -1,
             py::arg("stop_token_ids") = std::vector<int32_t>{})
        .def_property_readonly("config", &PyMultimodalModel::config,
                               py::return_value_policy::reference)
        .def_property_readonly("vision_config", &PyMultimodalModel::vision_config,
                               py::return_value_policy::reference)
        .def("create_context", &PyMultimodalModel::create_context,
             py::arg("kv_cache_dtype") = "fp32", py::arg("gpu_layers") = -1,
             py::return_value_policy::take_ownership);
}
