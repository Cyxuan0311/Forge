#include "nanoinfer/kv_cache.h"
#include "nanoinfer/cuda_kernels.h"
#include "nanoinfer/logger.h"
#include "nanoinfer/perf_profiler.h"
#include <cstring>
#include <cmath>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace nanoinfer {

KVCache::~KVCache() {
#ifdef USE_CUDA
    for (auto& layer : layers_) {
        if (layer.d_q_key_cache) cudaFree(layer.d_q_key_cache);
        if (layer.d_q_value_cache) cudaFree(layer.d_q_value_cache);
        layer.d_q_key_cache = nullptr;
        layer.d_q_value_cache = nullptr;
    }
#endif
}

size_t KVCache::q4_0_block_nbytes(int n) {
    int num_blocks = (n + 31) / 32;
    return num_blocks * (sizeof(float) + 16);
}

bool KVCache::init(int num_layers, int num_kv_heads, int head_dim,
                    int max_seq_len, DeviceType device) {
    return init_quantized(num_layers, num_kv_heads, head_dim,
                          max_seq_len, device, KVCacheDType::FP32);
}

bool KVCache::init_quantized(int num_layers, int num_kv_heads, int head_dim,
                              int max_seq_len, DeviceType device, KVCacheDType kv_dtype) {
    num_kv_heads_ = num_kv_heads;
    head_dim_ = head_dim;
    max_seq_len_ = max_seq_len;
    device_ = device;
    kv_dtype_ = kv_dtype;

    layers_.resize(num_layers);

    for (int i = 0; i < num_layers; ++i) {
        auto k_shape = std::vector<int64_t>{max_seq_len, num_kv_heads * head_dim};
        auto v_shape = std::vector<int64_t>{max_seq_len, num_kv_heads * head_dim};

        layers_[i].key_cache = std::make_shared<Tensor>(DataType::FP32, k_shape, device);
        layers_[i].value_cache = std::make_shared<Tensor>(DataType::FP32, v_shape, device);
        layers_[i].key_cache->zero_();
        layers_[i].value_cache->zero_();
        layers_[i].filled = 0;

        if (kv_dtype == KVCacheDType::Q4_0) {
            int per_row = num_kv_heads * head_dim;
            size_t q_size_per_row = q4_0_block_nbytes(per_row);
            layers_[i].q_key_cache.resize(max_seq_len * q_size_per_row, 0);
            layers_[i].q_value_cache.resize(max_seq_len * q_size_per_row, 0);
        }
    }

    LOG_INFO("KVCache initialized: " + std::to_string(num_layers) + " layers, " +
             std::to_string(num_kv_heads) + " kv_heads, " +
             std::to_string(head_dim) + " head_dim, " +
             std::to_string(max_seq_len) + " max_seq_len, " +
             (device == DeviceType::CUDA ? "CUDA" : "CPU") + ", " +
             (kv_dtype == KVCacheDType::Q4_0 ? "Q4_0" : "FP32"));

    return true;
}

void KVCache::reset() {
    for (auto& layer : layers_) {
        layer.filled = 0;
    }
}

static void quantize_q4_0_cpu(const float* data, uint8_t* q_data, int n) {
    int block_size = 32;
    int num_blocks = (n + block_size - 1) / block_size;

    for (int b = 0; b < num_blocks; ++b) {
        int start = b * block_size;
        int end = std::min(start + block_size, n);

        float amax = 0.0f;
        for (int i = start; i < end; ++i) {
            amax = std::max(amax, std::fabs(data[i]));
        }
        float scale = amax / 7.0f;
        if (scale == 0.0f) scale = 1.0f;
        float inv_scale = 1.0f / scale;

        uint8_t* block_ptr = q_data + b * (sizeof(float) + 16);
        std::memcpy(block_ptr, &scale, sizeof(float));
        uint8_t* qs = block_ptr + sizeof(float);

        for (int i = 0; i < 16; ++i) {
            int idx0 = start + i * 2;
            int idx1 = start + i * 2 + 1;

            int8_t v0 = (idx0 < end) ? static_cast<int8_t>(std::round(std::max(-8.0f, std::min(7.0f, data[idx0] * inv_scale)))) : 0;
            int8_t v1 = (idx1 < end) ? static_cast<int8_t>(std::round(std::max(-8.0f, std::min(7.0f, data[idx1] * inv_scale)))) : 0;

            uint8_t packed = (static_cast<uint8_t>(v0) & 0x0F) | ((static_cast<uint8_t>(v1) & 0x0F) << 4);
            qs[i] = packed;
        }
    }
}

static void dequantize_q4_0_cpu(const uint8_t* q_data, float* out, int n) {
    int block_size = 32;
    int num_blocks = (n + block_size - 1) / block_size;

    for (int b = 0; b < num_blocks; ++b) {
        const uint8_t* block_ptr = q_data + b * (sizeof(float) + 16);
        float scale;
        std::memcpy(&scale, block_ptr, sizeof(float));
        const uint8_t* qs = block_ptr + sizeof(float);

        int start = b * block_size;
        for (int i = 0; i < 16; ++i) {
            uint8_t packed = qs[i];
            int8_t v0 = packed & 0x0F;
            if (v0 & 0x08) v0 -= 16;
            int8_t v1 = (packed >> 4) & 0x0F;
            if (v1 & 0x08) v1 -= 16;

            int idx0 = start + i * 2;
            int idx1 = start + i * 2 + 1;
            if (idx0 < n) out[idx0] = static_cast<float>(v0) * scale;
            if (idx1 < n) out[idx1] = static_cast<float>(v1) * scale;
        }
    }
}

int KVCache::update(int layer, const TensorPtr& new_key, const TensorPtr& new_value, int seq_len) {
    if (kv_dtype_ == KVCacheDType::Q4_0) {
        return update_quantized(layer, new_key, new_value, seq_len);
    }

    if (layer < 0 || layer >= static_cast<int>(layers_.size())) {
        LOG_ERROR("KVCache::update: invalid layer " + std::to_string(layer));
        return -1;
    }

    auto& kv = layers_[layer];
    int filled = kv.filled;

    if (filled + seq_len > max_seq_len_) {
        LOG_ERROR("KVCache::update: cache overflow, filled=" + std::to_string(filled) +
                  " seq_len=" + std::to_string(seq_len) +
                  " max=" + std::to_string(max_seq_len_));
        return -1;
    }

    int kv_dim = num_kv_heads_ * head_dim_;

    if (device_ == DeviceType::CUDA) {
#ifdef USE_CUDA
        float* k_dst = static_cast<float*>(kv.key_cache->data()) + filled * kv_dim;
        float* v_dst = static_cast<float*>(kv.value_cache->data()) + filled * kv_dim;
        const float* k_src = static_cast<const float*>(new_key->data());
        const float* v_src = static_cast<const float*>(new_value->data());
        size_t copy_bytes = seq_len * kv_dim * sizeof(float);
        if (new_key->device() == DeviceType::CUDA) {
            cudaMemcpyAsync(k_dst, k_src, copy_bytes, cudaMemcpyDeviceToDevice);
            cudaMemcpyAsync(v_dst, v_src, copy_bytes, cudaMemcpyDeviceToDevice);
        } else {
            cudaMemcpyAsync(k_dst, k_src, copy_bytes, cudaMemcpyHostToDevice);
            cudaMemcpyAsync(v_dst, v_src, copy_bytes, cudaMemcpyHostToDevice);
        }
#endif
    } else {
        float* k_dst = static_cast<float*>(kv.key_cache->data()) + filled * kv_dim;
        float* v_dst = static_cast<float*>(kv.value_cache->data()) + filled * kv_dim;

        if (new_key->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
            std::vector<float> h_key(seq_len * kv_dim), h_value(seq_len * kv_dim);
            cudaMemcpy(h_key.data(), new_key->data(), seq_len * kv_dim * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_value.data(), new_value->data(), seq_len * kv_dim * sizeof(float), cudaMemcpyDeviceToHost);
            std::memcpy(k_dst, h_key.data(), seq_len * kv_dim * sizeof(float));
            std::memcpy(v_dst, h_value.data(), seq_len * kv_dim * sizeof(float));
#endif
        } else {
            const float* k_src = static_cast<const float*>(new_key->data());
            const float* v_src = static_cast<const float*>(new_value->data());
            std::memcpy(k_dst, k_src, seq_len * kv_dim * sizeof(float));
            std::memcpy(v_dst, v_src, seq_len * kv_dim * sizeof(float));
        }
    }

    kv.filled = filled + seq_len;
    return kv.filled;
}

int KVCache::update_quantized(int layer, const TensorPtr& new_key, const TensorPtr& new_value, int seq_len) {
    // Use CUDA path if device is CUDA
    if (device_ == DeviceType::CUDA) {
        int result = update_quantized_cuda(layer, new_key, new_value, seq_len);
        if (result >= 0) return result;
        // Fall through to CPU path if CUDA path fails
    }

    if (layer < 0 || layer >= static_cast<int>(layers_.size())) return -1;

    auto& kv = layers_[layer];
    int filled = kv.filled;
    if (filled + seq_len > max_seq_len_) return -1;

    int kv_dim = num_kv_heads_ * head_dim_;
    size_t q_row_size = q4_0_block_nbytes(kv_dim);

    std::vector<float> h_key(seq_len * kv_dim), h_value(seq_len * kv_dim);

    if (new_key->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        cudaMemcpy(h_key.data(), new_key->data(), seq_len * kv_dim * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_value.data(), new_value->data(), seq_len * kv_dim * sizeof(float), cudaMemcpyDeviceToHost);
#endif
    } else {
        std::memcpy(h_key.data(), new_key->data(), seq_len * kv_dim * sizeof(float));
        std::memcpy(h_value.data(), new_value->data(), seq_len * kv_dim * sizeof(float));
    }

    for (int s = 0; s < seq_len; ++s) {
        uint8_t* qk_dst = kv.q_key_cache.data() + (filled + s) * q_row_size;
        uint8_t* qv_dst = kv.q_value_cache.data() + (filled + s) * q_row_size;
        quantize_q4_0_cpu(h_key.data() + s * kv_dim, qk_dst, kv_dim);
        quantize_q4_0_cpu(h_value.data() + s * kv_dim, qv_dst, kv_dim);
    }

    kv.filled = filled + seq_len;
    return kv.filled;
}

void KVCache::dequantize_layer(int layer) {
    if (layer < 0 || layer >= static_cast<int>(layers_.size())) return;
    if (kv_dtype_ != KVCacheDType::Q4_0) return;

    // Use CUDA path if device is CUDA and CUDA quantized cache exists
    if (device_ == DeviceType::CUDA && layers_[layer].d_q_key_cache) {
        dequantize_layer_cuda(layer);
        return;
    }

    auto& kv = layers_[layer];
    int filled = kv.filled;
    int kv_dim = num_kv_heads_ * head_dim_;
    size_t q_row_size = q4_0_block_nbytes(kv_dim);

    std::vector<float> h_out(filled * kv_dim);

    for (int s = 0; s < filled; ++s) {
        const uint8_t* qk_src = kv.q_key_cache.data() + s * q_row_size;
        dequantize_q4_0_cpu(qk_src, h_out.data() + s * kv_dim, kv_dim);
    }

    if (device_ == DeviceType::CUDA) {
#ifdef USE_CUDA
        cudaMemcpy(kv.key_cache->data(), h_out.data(), filled * kv_dim * sizeof(float), cudaMemcpyHostToDevice);
#endif
    } else {
        std::memcpy(kv.key_cache->data(), h_out.data(), filled * kv_dim * sizeof(float));
    }

    h_out.resize(filled * kv_dim);
    for (int s = 0; s < filled; ++s) {
        const uint8_t* qv_src = kv.q_value_cache.data() + s * q_row_size;
        dequantize_q4_0_cpu(qv_src, h_out.data() + s * kv_dim, kv_dim);
    }

    if (device_ == DeviceType::CUDA) {
#ifdef USE_CUDA
        cudaMemcpy(kv.value_cache->data(), h_out.data(), filled * kv_dim * sizeof(float), cudaMemcpyHostToDevice);
#endif
    } else {
        std::memcpy(kv.value_cache->data(), h_out.data(), filled * kv_dim * sizeof(float));
    }
}

TensorPtr KVCache::get_key(int layer) const {
    if (layer < 0 || layer >= static_cast<int>(layers_.size())) return nullptr;
    return layers_[layer].key_cache;
}

TensorPtr KVCache::get_value(int layer) const {
    if (layer < 0 || layer >= static_cast<int>(layers_.size())) return nullptr;
    return layers_[layer].value_cache;
}

TensorPtr KVCache::get_key_filled(int layer) const {
    if (layer < 0 || layer >= static_cast<int>(layers_.size())) return nullptr;
    const auto& kv = layers_[layer];
    int f = kv.filled;
    if (f == max_seq_len_) return kv.key_cache;
    return std::make_shared<Tensor>(kv.key_cache->slice(0, 0, f));
}

TensorPtr KVCache::get_value_filled(int layer) const {
    if (layer < 0 || layer >= static_cast<int>(layers_.size())) return nullptr;
    const auto& kv = layers_[layer];
    int f = kv.filled;
    if (f == max_seq_len_) return kv.value_cache;
    return std::make_shared<Tensor>(kv.value_cache->slice(0, 0, f));
}

int KVCache::filled(int layer) const {
    if (layer < 0 || layer >= static_cast<int>(layers_.size())) return 0;
    return layers_[layer].filled;
}

size_t KVCache::nbytes() const {
    if (kv_dtype_ == KVCacheDType::Q4_0) {
        int kv_dim = num_kv_heads_ * head_dim_;
        size_t q_row_size = q4_0_block_nbytes(kv_dim);
        size_t per_layer = max_seq_len_ * q_row_size * 2;
        return per_layer * layers_.size();
    }
    size_t per_layer = max_seq_len_ * num_kv_heads_ * head_dim_ * sizeof(float) * 2;
    return per_layer * layers_.size();
}

void KVCache::alloc_cuda_q_cache(int layer, size_t bytes) {
#ifdef USE_CUDA
    auto& kv = layers_[layer];
    if (kv.d_q_cache_bytes >= bytes) return;
    if (kv.d_q_key_cache) cudaFree(kv.d_q_key_cache);
    if (kv.d_q_value_cache) cudaFree(kv.d_q_value_cache);
    cudaMalloc(&kv.d_q_key_cache, bytes);
    cudaMalloc(&kv.d_q_value_cache, bytes);
    kv.d_q_cache_bytes = bytes;
#endif
}

int KVCache::update_quantized_cuda(int layer, const TensorPtr& new_key, const TensorPtr& new_value, int seq_len) {
#ifdef USE_CUDA
    if (layer < 0 || layer >= static_cast<int>(layers_.size())) return -1;

    auto& kv = layers_[layer];
    int filled = kv.filled;
    if (filled + seq_len > max_seq_len_) return -1;

    int kv_dim = num_kv_heads_ * head_dim_;
    size_t q_row_size = q4_0_block_nbytes(kv_dim);
    size_t total_q_bytes = max_seq_len_ * q_row_size;

    // Allocate CUDA quantized cache if needed
    alloc_cuda_q_cache(layer, total_q_bytes);

    // Ensure new_key/new_value are on CUDA
    const float* k_src = static_cast<const float*>(new_key->data());
    const float* v_src = static_cast<const float*>(new_value->data());

    std::vector<float> h_key, h_value;
    if (new_key->device() == DeviceType::CPU) {
        h_key.resize(seq_len * kv_dim);
        h_value.resize(seq_len * kv_dim);
        std::memcpy(h_key.data(), k_src, seq_len * kv_dim * sizeof(float));
        std::memcpy(h_value.data(), v_src, seq_len * kv_dim * sizeof(float));

        // Upload to temp CUDA buffer, then quantize on GPU
        float* d_temp;
        cudaMalloc(&d_temp, seq_len * kv_dim * sizeof(float) * 2);
        cudaMemcpy(d_temp, h_key.data(), seq_len * kv_dim * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_temp + seq_len * kv_dim, h_value.data(), seq_len * kv_dim * sizeof(float), cudaMemcpyHostToDevice);

        // Quantize on GPU
        uint8_t* q_key_dst = static_cast<uint8_t*>(kv.d_q_key_cache) + filled * q_row_size;
        uint8_t* q_val_dst = static_cast<uint8_t*>(kv.d_q_value_cache) + filled * q_row_size;
        cuda::launch_quantize_q4_0_matrix(d_temp, q_key_dst, seq_len, kv_dim);
        cuda::launch_quantize_q4_0_matrix(d_temp + seq_len * kv_dim, q_val_dst, seq_len, kv_dim);

        cudaFree(d_temp);
    } else {
        // Both already on CUDA - quantize directly on GPU
        uint8_t* q_key_dst = static_cast<uint8_t*>(kv.d_q_key_cache) + filled * q_row_size;
        uint8_t* q_val_dst = static_cast<uint8_t*>(kv.d_q_value_cache) + filled * q_row_size;
        cuda::launch_quantize_q4_0_matrix(k_src, q_key_dst, seq_len, kv_dim);
        cuda::launch_quantize_q4_0_matrix(v_src, q_val_dst, seq_len, kv_dim);
    }

    kv.filled = filled + seq_len;
    return kv.filled;
#else
    return -1;
#endif
}

void KVCache::dequantize_layer_cuda(int layer) {
#ifdef USE_CUDA
    if (layer < 0 || layer >= static_cast<int>(layers_.size())) return;
    if (kv_dtype_ != KVCacheDType::Q4_0) return;

    auto& kv = layers_[layer];
    int filled = kv.filled;
    int kv_dim = num_kv_heads_ * head_dim_;

    if (!kv.d_q_key_cache || filled == 0) return;

    // Dequantize directly from CUDA quantized cache to CUDA FP32 cache
    float* k_out = static_cast<float*>(kv.key_cache->data());
    float* v_out = static_cast<float*>(kv.value_cache->data());

    // Dequantize all filled rows at once
    cuda::launch_dequant_q4_0_matrix(
        kv.d_q_key_cache, k_out, filled, kv_dim);
    cuda::launch_dequant_q4_0_matrix(
        kv.d_q_value_cache, v_out, filled, kv_dim);
#endif
}

} // namespace nanoinfer
