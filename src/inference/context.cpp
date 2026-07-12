#include "forge/context.h"

#include <cstring>
#include <stdexcept>

#include "forge/backend.h"
#include "forge/engine.h"
#include "forge/engines/llama_engine.h"
#include "forge/logger.h"
#include "forge/model.h"
#include "forge/perf_profiler.h"

#ifdef USE_CUDA
#    include <cuda_runtime.h>
#endif

namespace forge {

InferenceContext::InferenceContext(const Model& model) : model_(model), params_{} {
    params_.device = model.device();
    params_.max_seq_len = model.config().max_seq_len;
}

InferenceContext::InferenceContext(const Model& model, const ContextParams& params)
    : model_(model), params_(params) {
    if (params_.max_seq_len <= 0) {
        params_.max_seq_len = model.config().max_seq_len;
    }
    if (params_.device == DeviceType::CPU && model.device() == DeviceType::CUDA) {
        params_.device = model.device();
    }
}

InferenceContext::~InferenceContext() = default;

bool InferenceContext::init_kv_cache() {
    if (kv_cache_initialized_)
        return true;

    const auto& cfg = model_.config();
    kv_cache_ = std::make_unique<KVCache>();

    DeviceType kv_dev = params_.device;
    if (params_.gpu_layers >= 0 && params_.gpu_layers < cfg.num_layers) {
        kv_dev = DeviceType::CPU;
    }

    int kv_max_seq = params_.max_seq_len;
    const int KV_MAX_SEQ_CAP = 4096;

    // Cap max_seq_len to avoid OOM
    if (kv_max_seq > KV_MAX_SEQ_CAP) {
        LOG_INFO("Capping KV cache max_seq_len from " + std::to_string(kv_max_seq) + " to " +
                 std::to_string(KV_MAX_SEQ_CAP) + " to avoid OOM");
        kv_max_seq = KV_MAX_SEQ_CAP;
    }

    // If CUDA, check available memory and further reduce if needed
    if (kv_dev == DeviceType::CUDA) {
        size_t kv_bytes = (size_t)cfg.num_layers * 2 * (size_t)kv_max_seq * cfg.num_kv_heads *
                          cfg.head_dim * sizeof(float);
        auto backend = BackendManager::instance().get_cuda_backend();
        if (backend) {
            size_t free_mem = backend->device_memory_free();
            // Leave 256MB headroom for other allocations
            const size_t headroom = 256 * 1024 * 1024;
            if (kv_bytes + headroom > free_mem && free_mem > headroom) {
                size_t available = free_mem - headroom;
                int reduced_seq =
                    static_cast<int>(available / ((size_t)cfg.num_layers * 2 * cfg.num_kv_heads *
                                                  cfg.head_dim * sizeof(float)));
                if (reduced_seq < kv_max_seq) {
                    // Round down to nearest 256 for alignment
                    reduced_seq = (reduced_seq / 256) * 256;
                    if (reduced_seq < 256)
                        reduced_seq = 256;
                    LOG_WARN("KV cache needs " + std::to_string(kv_bytes / (1024 * 1024)) +
                             "MB but only " + std::to_string(free_mem / (1024 * 1024)) +
                             "MB free, reducing max_seq_len from " + std::to_string(kv_max_seq) +
                             " to " + std::to_string(reduced_seq));
                    kv_max_seq = reduced_seq;
                }
            }
        }
    }

    kv_cache_->init_quantized(cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, kv_max_seq, kv_dev,
                              params_.kv_cache_dtype);

    kv_cache_initialized_ = true;
    LOG_INFO("KV cache initialized: " + std::to_string(cfg.num_layers) +
             " layers, max_seq=" + std::to_string(kv_max_seq) +
             ", dtype=" + (params_.kv_cache_dtype == KVCacheDType::Q4_0 ? "q4_0" : "fp32"));
    return true;
}

TensorPtr InferenceContext::forward(const TensorPtr& input_ids, int64_t start_pos) {
    if (!engine_) {
        throw std::runtime_error("InferenceContext: no engine set");
    }

    if (!kv_cache_initialized_) {
        init_kv_cache();
    }

    return engine_->forward(input_ids, start_pos);
}

TensorPtr InferenceContext::decode(int token_id, int64_t start_pos) {
    auto input =
        std::make_shared<Tensor>(DataType::INT32, std::vector<int64_t>{1}, DeviceType::CPU);
    *static_cast<int32_t*>(input->data()) = token_id;

    if (params_.device == DeviceType::CUDA) {
        input->to_device(DeviceType::CUDA);
    }

    return forward(input, start_pos);
}

void InferenceContext::reset() {
    current_pos_ = 0;
    reset_kv_cache();
}

void InferenceContext::reset_kv_cache() {
    if (kv_cache_) {
        kv_cache_->reset();
    }
    if (engine_) {
        engine_->reset();
    }
    kv_cache_initialized_ = false;
}

int InferenceContext::generate(int start_token, int max_tokens,
                               std::function<int(float*, int)> sampler_fn) {
    int token = start_token;
    for (int i = 0; i < max_tokens; ++i) {
        auto logits = decode(token, current_pos_);
        current_pos_++;

        std::vector<float> host_logits(logits->numel());
        if (logits->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
            cudaMemcpy(host_logits.data(), logits->data(), logits->numel() * sizeof(float),
                       cudaMemcpyDeviceToHost);
#endif
        } else {
            std::memcpy(host_logits.data(), logits->data(), logits->nbytes());
        }

        token = sampler_fn(host_logits.data(), static_cast<int>(host_logits.size()));
    }
    return token;
}

void InferenceContext::set_engine(std::unique_ptr<InferenceEngine> engine) {
    engine_ = std::move(engine);
}

InferenceEngine* InferenceContext::engine() {
    return engine_.get();
}

const InferenceEngine* InferenceContext::engine() const {
    return engine_.get();
}

void InferenceContext::set_gpu_layers(int layers) {
    params_.gpu_layers = layers;
    if (engine_) {
        engine_->set_gpu_layers(layers);
    }
}

void InferenceContext::set_kv_cache_dtype(KVCacheDType dtype) {
    params_.kv_cache_dtype = dtype;
}

void InferenceContext::warmup() {
    if (!engine_) {
        LOG_WARN("InferenceContext::warmup: no engine, skipping");
        return;
    }

    // Save profiler state and disable during warmup
    bool profiler_was_enabled = PerfProfiler::instance().enabled();
    PerfProfiler::instance().disable();

    const auto& cfg = model_.config();

    // Ensure KV cache is initialized
    if (!kv_cache_initialized_) {
        init_kv_cache();
    }

    // Run a short dummy forward pass to trigger CUDA kernel compilation
    // Use 2 tokens for prefill + 1 token for decode to cover both paths
    auto input_ids =
        std::make_shared<Tensor>(DataType::INT32, std::vector<int64_t>{2}, DeviceType::CPU);
    auto* ids = static_cast<int32_t*>(input_ids->data());
    ids[0] = 0;
    ids[1] = 0;

    if (params_.device == DeviceType::CUDA) {
        input_ids->to_device(DeviceType::CUDA);
    }

    // Prefill warmup
    engine_->forward(input_ids, 0);

    // Decode warmup
    auto decode_input =
        std::make_shared<Tensor>(DataType::INT32, std::vector<int64_t>{1}, DeviceType::CPU);
    *static_cast<int32_t*>(decode_input->data()) = 0;
    if (params_.device == DeviceType::CUDA) {
        decode_input->to_device(DeviceType::CUDA);
    }
    engine_->forward(decode_input, 2);

#ifdef USE_CUDA
    cudaDeviceSynchronize();
#endif

    // Reset state after warmup
    reset_kv_cache();

    // Restore profiler
    if (profiler_was_enabled) {
        PerfProfiler::instance().enable();
    }

    LOG_INFO("Warmup completed");
}

}  // namespace forge
