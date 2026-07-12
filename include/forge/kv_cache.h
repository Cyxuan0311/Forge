#pragma once

#include <memory>
#include <vector>

#include "tensor.h"

namespace forge {

enum class KVCacheDType : int {
    FP32 = 0,
    Q4_0 = 1,
};

struct KVCacheLayer {
    TensorPtr key_cache;
    TensorPtr value_cache;
    std::vector<uint8_t> q_key_cache;
    std::vector<uint8_t> q_value_cache;
    // CUDA-side quantized KV cache buffers
    void* d_q_key_cache = nullptr;
    void* d_q_value_cache = nullptr;
    size_t d_q_cache_bytes = 0;
    int filled = 0;
};

class KVCache {
public:
    KVCache() = default;
    ~KVCache();

    bool init(int num_layers, int num_kv_heads, int head_dim, int max_seq_len, DeviceType device);

    bool init_quantized(int num_layers, int num_kv_heads, int head_dim, int max_seq_len,
                        DeviceType device, KVCacheDType kv_dtype);

    void reset();

    int update(int layer, const TensorPtr& new_key, const TensorPtr& new_value, int seq_len);

    TensorPtr get_key(int layer) const;
    TensorPtr get_value(int layer) const;

    TensorPtr get_key_filled(int layer) const;
    TensorPtr get_value_filled(int layer) const;

    int filled(int layer) const;
    int max_seq_len() const { return max_seq_len_; }
    int num_layers() const { return static_cast<int>(layers_.size()); }
    DeviceType device() const { return device_; }
    KVCacheDType kv_dtype() const { return kv_dtype_; }

    size_t nbytes() const;

    static size_t q4_0_block_nbytes(int n);

    void dequantize_layer(int layer);

private:
    int update_quantized(int layer, const TensorPtr& new_key, const TensorPtr& new_value,
                         int seq_len);
    int update_quantized_cuda(int layer, const TensorPtr& new_key, const TensorPtr& new_value,
                              int seq_len);
    void dequantize_layer_cuda(int layer);
    void alloc_cuda_q_cache(int layer, size_t bytes);

    std::vector<KVCacheLayer> layers_;
    int num_kv_heads_ = 0;
    int head_dim_ = 0;
    int max_seq_len_ = 0;
    DeviceType device_ = DeviceType::CPU;
    KVCacheDType kv_dtype_ = KVCacheDType::FP32;
};

}  // namespace forge
