#include "forge/context.h"

#include <cstring>
#include <stdexcept>

#include "forge/backend.h"
#include "forge/engine.h"
#include "forge/engines/llama_engine.h"
#include "forge/inference/forward_request.h"
#include "forge/logger.h"
#include "forge/model.h"
#include "forge/perf_profiler.h"

#ifdef USE_CUDA
#    include <cuda_runtime.h>
#endif

namespace forge {

namespace {

// 根据模型架构选择 memory 类型: Qwen3.5 hybrid 需要 KV cache + recurrent state,
// 其余架构只需 KV cache。
std::unique_ptr<InferenceMemory> make_memory(const Model& model) {
    const auto& cfg = model.config();
    if (cfg.use_ssm) {
        return std::make_unique<HybridMemory>();
    }
    // ArchCapability 也可能声明 use_ssm, 但 context 构造时 ExecutionPlan 尚未构建。
    // 通过 arch_type 名做简单判断即可, ExecutionPlan 会在 engine 构建时做权威判断。
    if (cfg.arch_type == "qwen35") {
        return std::make_unique<HybridMemory>();
    }
    return std::make_unique<KvMemory>();
}

}  // namespace

InferenceContext::InferenceContext(const Model& model)
    : model_(model), params_{}, memory_(make_memory(model)) {
    params_.device = model.device();
    params_.max_seq_len = model.config().max_seq_len;
}

InferenceContext::InferenceContext(const Model& model, const ContextParams& params)
    : model_(model), params_(params), memory_(make_memory(model)) {
    if (params_.max_seq_len <= 0) {
        params_.max_seq_len = model.config().max_seq_len;
    }
    if (params_.device == DeviceType::CPU && model.device() == DeviceType::CUDA) {
        params_.device = model.device();
    }
}

InferenceContext::~InferenceContext() = default;

// KV cache 的实际分配由 engine 完成: 只有 engine 知道该架构的层数、per-layer KV
// 维度和 ring buffer 配置(Gemma4 的 per-layer KV 就无法用统一逻辑分配)。
// context 作为 owner 只负责持有 memory 并触发分配, 不再自己 init 一份。
bool InferenceContext::init_kv_cache() {
    if (memory_->initialized())
        return true;
    if (!engine_) {
        LOG_WARN("InferenceContext::init_kv_cache: no engine set, deferring allocation");
        return false;
    }
    engine_->init_memory();
    return memory_->initialized();
}

TensorPtr InferenceContext::forward(const TensorPtr& input_ids, int64_t start_pos) {
    if (!engine_) {
        throw std::runtime_error("InferenceContext: no engine set");
    }

    // engine 会在 forward 内部按需分配 memory, 这里不再预先 init 一份。
    return engine_->forward_request(ForwardRequest::from_ids(input_ids, start_pos));
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
    // memory_ 是唯一所有者, reset 一次即可。engine_->reset() 仍需调用,
    // 因为 engine 可能有额外的内部状态(如 graph cache)需要清理。
    memory_->reset();
    if (engine_) {
        engine_->reset();
    }
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

    // Run a short dummy forward pass to trigger CUDA kernel compilation.
    // The engine allocates KV cache on its first forward; since InferenceContext
    // is now the sole owner, no redundant cache is created here.

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
    engine_->forward_request(ForwardRequest::from_ids(input_ids, 0));

    // Decode warmup
    auto decode_input =
        std::make_shared<Tensor>(DataType::INT32, std::vector<int64_t>{1}, DeviceType::CPU);
    *static_cast<int32_t*>(decode_input->data()) = 0;
    if (params_.device == DeviceType::CUDA) {
        decode_input->to_device(DeviceType::CUDA);
    }
    engine_->forward_request(ForwardRequest::from_ids(decode_input, 2));

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
