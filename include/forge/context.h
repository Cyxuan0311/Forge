#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "kv_cache.h"
#include "quant_policy.h"
#include "speculative.h"
#include "tensor.h"
#include "types.h"

namespace forge {

class Model;
class InferenceEngine;

struct ContextParams {
    int max_seq_len = 4096;
    int gpu_layers = -1;
    KVCacheDType kv_cache_dtype = KVCacheDType::FP32;  // legacy: sets both K and V
    KVCacheTypeConfig kv_cache_config;                  // per-K/V type config
    DeviceType device = DeviceType::CUDA;
    int batch_size = 1;
    int n_batch = 512;          // max tokens per forward_batch() call
    int n_ubatch = 256;         // max tokens per internal micro-batch
    int n_threads = 4;          // decode (single-token) thread count
    int n_threads_batch = 8;    // prefill/batch (multi-token) thread count
    QuantPolicy quant_policy;   // per-tensor 混合精度策略
    SpeculativeConfig speculative_config;  // speculative decoding 配置

    KVCacheDType type_k() const { return kv_cache_config.type_k; }
    KVCacheDType type_v() const { return kv_cache_config.type_v; }
};

class InferenceContext {
public:
    explicit InferenceContext(const Model& model);
    InferenceContext(const Model& model, const ContextParams& params);
    ~InferenceContext();

    TensorPtr forward(const TensorPtr& input_ids, int64_t start_pos);
    TensorPtr decode(int token_id, int64_t start_pos);

    void reset();
    void reset_kv_cache();

    int generate(int start_token, int max_tokens, std::function<int(float*, int)> sampler_fn);

    const Model& model() const { return model_; }
    const KVCache& kv_cache() const { return *kv_cache_; }
    KVCache& kv_cache() { return *kv_cache_; }

    const ContextParams& params() const { return params_; }
    ContextParams& params_mut() { return params_; }
    int current_pos() const { return current_pos_; }

    void set_engine(std::unique_ptr<InferenceEngine> engine);
    InferenceEngine* engine();
    const InferenceEngine* engine() const;

    void set_gpu_layers(int layers);
    int gpu_layers() const { return params_.gpu_layers; }

    void set_kv_cache_dtype(KVCacheDType dtype);
    KVCacheDType kv_cache_dtype() const { return params_.kv_cache_dtype; }

    DeviceType device() const { return params_.device; }

    bool init_kv_cache();

    /// Run a warmup forward pass to trigger CUDA kernel JIT compilation.
    /// Call this before timing inference to avoid first-run overhead.
    void warmup();

private:
    const Model& model_;
    ContextParams params_;
    std::unique_ptr<KVCache> kv_cache_;
    std::unique_ptr<InferenceEngine> engine_;
    int current_pos_ = 0;
    bool kv_cache_initialized_ = false;
};

}  // namespace forge
