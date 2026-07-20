#include "forge/kv_cache.h"

#include <cmath>
#include <cstring>

#include "forge/cuda_kernels.h"
#include "forge/logger.h"
#include "forge/perf_profiler.h"

#ifdef USE_CUDA
#    include <cuda_runtime.h>
#endif

namespace forge {

// ---- fp16 helpers (CPU) ----

static inline uint16_t fp32_to_fp16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(uint32_t));
    uint32_t sign = (x >> 31) & 1;
    uint32_t exp = (x >> 23) & 0xFF;
    uint32_t mant = x & 0x7FFFFF;
    uint16_t h;

    if (exp == 0xFF) {
        // Inf or NaN
        h = (sign << 15) | 0x7C00 | (mant ? 1 : 0);
    } else if (exp >= 143) {
        // Overflow → Inf
        h = (sign << 15) | 0x7C00;
    } else if (exp >= 113) {
        // Normal
        h = (sign << 15) | ((exp - 112) << 10) | (mant >> 13);
    } else if (exp >= 103) {
        // Subnormal
        uint32_t sub_exp = 126 - exp;
        uint32_t sub_mant = (mant | 0x800000) >> (sub_exp + 13);
        h = (sign << 15) | sub_mant;
    } else {
        h = (sign << 15);  // underflow → 0
    }
    return h;
}

static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t x;

    if (exp == 0) {
        if (mant == 0) {
            x = sign << 31;
        } else {
            // Subnormal: normalize
            int e = -1;
            while (!(mant & 0x400)) {
                mant <<= 1;
                --e;
            }
            mant &= 0x3FF;
            x = (sign << 31) | ((127 + e - 14) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        x = (sign << 31) | (0xFF << 23) | (mant << 13);
    } else {
        x = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &x, sizeof(float));
    return f;
}

// =========================================================================
// Per-type CPU quantize/dequantize
// =========================================================================

// ---- F16 ----

static void quantize_f16_cpu(const float* data, uint8_t* q_data, int n) {
    auto* qs = reinterpret_cast<uint16_t*>(q_data);
    for (int i = 0; i < n; ++i) {
        qs[i] = fp32_to_fp16(data[i]);
    }
}

static void dequantize_f16_cpu(const uint8_t* q_data, float* out, int n) {
    auto* qs = reinterpret_cast<const uint16_t*>(q_data);
    for (int i = 0; i < n; ++i) {
        out[i] = fp16_to_fp32(qs[i]);
    }
}

// ---- Q8_0 (fp16 d + int8 qs[32] = 34 bytes/block) ----

static void quantize_q8_0_cpu(const float* data, uint8_t* q_data, int n) {
    constexpr int BLOCK = 32;
    int num_blocks = (n + BLOCK - 1) / BLOCK;

    for (int b = 0; b < num_blocks; ++b) {
        int start = b * BLOCK;
        int end = std::min(start + BLOCK, n);

        float amax = 0.0f;
        for (int i = start; i < end; ++i)
            amax = std::max(amax, std::fabs(data[i]));

        float d = amax / 127.0f;
        if (d == 0.0f)
            d = 1.0f;
        float id = 1.0f / d;

        uint16_t d_fp16 = fp32_to_fp16(d);
        std::memcpy(q_data + b * 34, &d_fp16, 2);

        auto* qs = reinterpret_cast<int8_t*>(q_data + b * 34 + 2);
        for (int i = 0; i < BLOCK; ++i) {
            int idx = start + i;
            if (idx < end) {
                float v = data[idx] * id;
                qs[i] = static_cast<int8_t>(std::round(std::max(-128.0f, std::min(127.0f, v))));
            } else {
                qs[i] = 0;
            }
        }
    }
}

static void dequantize_q8_0_cpu(const uint8_t* q_data, float* out, int n) {
    constexpr int BLOCK = 32;
    int num_blocks = (n + BLOCK - 1) / BLOCK;

    for (int b = 0; b < num_blocks; ++b) {
        uint16_t d_bits;
        std::memcpy(&d_bits, q_data + b * 34, 2);
        float d = fp16_to_fp32(d_bits);

        auto* qs = reinterpret_cast<const int8_t*>(q_data + b * 34 + 2);
        int start = b * BLOCK;
        for (int i = 0; i < BLOCK; ++i) {
            int idx = start + i;
            if (idx < n)
                out[idx] = static_cast<float>(qs[i]) * d;
        }
    }
}

// ---- Q4_0 (fp16 d + 16 packed bytes = 18 bytes/block, standard format) ----

static void quantize_q4_0_cpu(const float* data, uint8_t* q_data, int n) {
    constexpr int BLOCK = 32;
    int num_blocks = (n + BLOCK - 1) / BLOCK;

    for (int b = 0; b < num_blocks; ++b) {
        int start = b * BLOCK;
        int end = std::min(start + BLOCK, n);

        float amax = 0.0f;
        for (int i = start; i < end; ++i)
            amax = std::max(amax, std::fabs(data[i]));

        float d = amax / -8.0f;
        if (d == 0.0f)
            d = 1.0f;
        float id = 1.0f / d;

        uint16_t d_fp16 = fp32_to_fp16(d);
        std::memcpy(q_data + b * 18, &d_fp16, 2);

        uint8_t* qs = q_data + b * 18 + 2;
        for (int i = 0; i < 16; ++i) {
            int idx0 = start + i;
            int idx1 = start + i + 16;
            int8_t v0 = (idx0 < end) ? static_cast<int8_t>(std::round(
                                           std::max(-8.0f, std::min(7.0f, data[idx0] * id))))
                                     : 0;
            int8_t v1 = (idx1 < end) ? static_cast<int8_t>(std::round(
                                           std::max(-8.0f, std::min(7.0f, data[idx1] * id))))
                                     : 0;
            qs[i] = (static_cast<uint8_t>(v0) & 0x0F) | ((static_cast<uint8_t>(v1) & 0x0F) << 4);
        }
    }
}

static void dequantize_q4_0_cpu(const uint8_t* q_data, float* out, int n) {
    constexpr int BLOCK = 32;
    int num_blocks = (n + BLOCK - 1) / BLOCK;

    for (int b = 0; b < num_blocks; ++b) {
        uint16_t d_bits;
        std::memcpy(&d_bits, q_data + b * 18, 2);
        float d = fp16_to_fp32(d_bits);

        const uint8_t* qs = q_data + b * 18 + 2;
        int start = b * BLOCK;
        for (int i = 0; i < 16; ++i) {
            uint8_t packed = qs[i];
            int8_t v0 = packed & 0x0F;
            if (v0 & 0x08)
                v0 -= 16;
            int8_t v1 = (packed >> 4) & 0x0F;
            if (v1 & 0x08)
                v1 -= 16;
            int idx0 = start + i;
            int idx1 = start + i + 16;
            if (idx0 < n)
                out[idx0] = static_cast<float>(v0) * d;
            if (idx1 < n)
                out[idx1] = static_cast<float>(v1) * d;
        }
    }
}

// ---- Q4_K (144 bytes/block of 256 elements) ----

static void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

static void quantize_q4_k_cpu(const float* data, uint8_t* q_data, int n) {
    // Simplified Q4_K quantization: use uniform scale per block (no sub-block min)
    // Block layout: d[2] + dmin[2] + scales[12] + qs[128] = 144 bytes
    constexpr int BLOCK = 256;
    int num_blocks = (n + BLOCK - 1) / BLOCK;

    for (int b = 0; b < num_blocks; ++b) {
        int start = b * BLOCK;
        int end = std::min(start + BLOCK, n);

        float amax = 0.0f;
        for (int i = start; i < end; ++i)
            amax = std::max(amax, std::fabs(data[i]));

        float d = amax / 7.0f;
        if (d == 0.0f)
            d = 1.0f;
        float id = 1.0f / d;

        uint8_t* block = q_data + b * 144;
        uint16_t d_fp16 = fp32_to_fp16(d);
        uint16_t dmin_fp16 = fp32_to_fp16(0.0f);
        std::memcpy(block, &d_fp16, 2);
        std::memcpy(block + 2, &dmin_fp16, 2);

        // Set all 6-bit scales to 1 (sc=1, m=0 for each of 8 sub-blocks)
        // scales[12] encodes 8 (sc, m) pairs via get_scale_min_k4
        // sc=1, m=0 → 6-bit encoding: (m << 4) | sc = 0x01
        std::memset(block + 4, 0x01, 12);

        // Quantize 256 elements into 128 packed bytes (4-bit each)
        uint8_t* qs = block + 16;
        for (int i = 0; i < 128; ++i) {
            int idx0 = start + i;
            int idx1 = start + i + 128;
            int8_t v0 = (idx0 < end) ? static_cast<int8_t>(std::round(
                                           std::max(-8.0f, std::min(7.0f, data[idx0] * id))))
                                     : 0;
            int8_t v1 = (idx1 < end) ? static_cast<int8_t>(std::round(
                                           std::max(-8.0f, std::min(7.0f, data[idx1] * id))))
                                     : 0;
            qs[i] = (static_cast<uint8_t>(v0) & 0x0F) | ((static_cast<uint8_t>(v1) & 0x0F) << 4);
        }
    }
}

static void dequantize_q4_k_cpu(const uint8_t* q_data, float* out, int n, int row) {
    constexpr int QK_K = 256;
    constexpr int Q4_K_BLOCK_SIZE = 144;
    int blocks_per_row = (n + QK_K - 1) / QK_K;
    const uint8_t* row_ptr = q_data + row * blocks_per_row * Q4_K_BLOCK_SIZE;

    for (int bi = 0; bi < blocks_per_row; ++bi) {
        const uint8_t* block_ptr = row_ptr + bi * Q4_K_BLOCK_SIZE;
        uint16_t d_bits, dmin_bits;
        std::memcpy(&d_bits, block_ptr, 2);
        std::memcpy(&dmin_bits, block_ptr + 2, 2);
        float d = fp16_to_fp32(d_bits);
        float dmin = fp16_to_fp32(dmin_bits);
        const uint8_t* scales = block_ptr + 4;
        const uint8_t* qs = block_ptr + 16;

        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(is, scales, &sc1, &m1);
            get_scale_min_k4(is + 1, scales, &sc2, &m2);
            float d1 = d * sc1;
            float m1_val = dmin * m1;
            float d2 = d * sc2;
            float m2_val = dmin * m2;
            int base = bi * QK_K + j;
            for (int l = 0; l < 32; ++l) {
                if (base + l < n)
                    out[base + l] = d1 * static_cast<float>(qs[l] & 0xF) - m1_val;
            }
            for (int l = 0; l < 32; ++l) {
                if (base + 32 + l < n)
                    out[base + 32 + l] = d2 * static_cast<float>(qs[l] >> 4) - m2_val;
            }
            qs += 32;
            is += 2;
        }
    }
}

// =========================================================================
// Block size calculation
// =========================================================================

size_t KVCache::q4_0_block_nbytes(int n) {
    int num_blocks = (n + 31) / 32;
    return num_blocks * 18;  // standard: fp16 d (2B) + 16B qs
}

size_t KVCache::block_nbytes(KVCacheDType dtype, int n) {
    switch (dtype) {
    case KVCacheDType::FP32:
        return n * sizeof(float);
    case KVCacheDType::F16:
        return n * sizeof(uint16_t);
    case KVCacheDType::Q8_0: {
        int num_blocks = (n + 31) / 32;
        return num_blocks * 34;  // fp16 d (2B) + int8 qs[32] (32B)
    }
    case KVCacheDType::Q4_0:
        return q4_0_block_nbytes(n);
    case KVCacheDType::Q4_K: {
        int num_blocks = (n + 255) / 256;
        return num_blocks * 144;  // d(2) + dmin(2) + scales(12) + qs(128)
    }
    default:
        return n * sizeof(float);
    }
}

// =========================================================================
// Destructor
// =========================================================================

KVCache::~KVCache() {
#ifdef USE_CUDA
    for (auto& layer : layers_) {
        if (layer.d_q_key_cache)
            cudaFree(layer.d_q_key_cache);
        if (layer.d_q_value_cache)
            cudaFree(layer.d_q_value_cache);
        layer.d_q_key_cache = nullptr;
        layer.d_q_value_cache = nullptr;
    }
#endif
}

// =========================================================================
// Initialization
// =========================================================================

void KVCache::init_cells(int layer) {
    layers_[layer].cells.assign(max_seq_len_, KVCellMeta{});
}

bool KVCache::init(int num_layers, int num_kv_heads, int head_dim, int max_seq_len,
                   DeviceType device) {
    return init_quantized(num_layers, num_kv_heads, head_dim, max_seq_len, device,
                          KVCacheDType::FP32);
}

bool KVCache::init_quantized(int num_layers, int num_kv_heads, int head_dim, int max_seq_len,
                             DeviceType device, KVCacheDType kv_dtype) {
    KVCacheTypeConfig config;
    config.type_k = kv_dtype;
    config.type_v = kv_dtype;
    return init_quantized(num_layers, num_kv_heads, head_dim, max_seq_len, device, config);
}

static const char* kv_dtype_name(KVCacheDType dt) {
    switch (dt) {
    case KVCacheDType::FP32: return "FP32";
    case KVCacheDType::F16:  return "F16";
    case KVCacheDType::Q8_0: return "Q8_0";
    case KVCacheDType::Q4_0: return "Q4_0";
    case KVCacheDType::Q4_K: return "Q4_K";
    default: return "unknown";
    }
}

bool KVCache::init_quantized(int num_layers, int num_kv_heads, int head_dim, int max_seq_len,
                             DeviceType device, const KVCacheTypeConfig& kv_config) {
    num_kv_heads_ = num_kv_heads;
    head_dim_ = head_dim;
    max_seq_len_ = max_seq_len;
    device_ = device;
    kv_config_ = kv_config;
    // Legacy: kv_dtype_ = max(type_k, type_v) for backward compat
    kv_dtype_ = static_cast<KVCacheDType>(
        std::max(static_cast<int>(kv_config.type_k), static_cast<int>(kv_config.type_v)));

    layers_.resize(num_layers);

    int kv_dim = num_kv_heads * head_dim;

    for (int i = 0; i < num_layers; ++i) {
        // FP32 shadow cache (always allocated — used as dequant target and for attention)
        auto k_shape = std::vector<int64_t>{max_seq_len, kv_dim};
        auto v_shape = std::vector<int64_t>{max_seq_len, kv_dim};
        layers_[i].key_cache = std::make_shared<Tensor>(DataType::FP32, k_shape, device);
        layers_[i].value_cache = std::make_shared<Tensor>(DataType::FP32, v_shape, device);
        layers_[i].key_cache->zero_();
        layers_[i].value_cache->zero_();
        layers_[i].filled = 0;

        // Allocate quantized caches if needed
        if (kv_config.type_k != KVCacheDType::FP32) {
            size_t q_size = max_seq_len * block_nbytes(kv_config.type_k, kv_dim);
            layers_[i].q_key_cache.resize(q_size, 0);
        }
        if (kv_config.type_v != KVCacheDType::FP32) {
            size_t q_size = max_seq_len * block_nbytes(kv_config.type_v, kv_dim);
            layers_[i].q_value_cache.resize(q_size, 0);
        }

        init_cells(i);
    }

    LOG_INFO("KVCache initialized: " + std::to_string(num_layers) + " layers, " +
             std::to_string(num_kv_heads) + " kv_heads, " + std::to_string(head_dim) +
             " head_dim, " + std::to_string(max_seq_len) + " max_seq_len, " +
             (device == DeviceType::CUDA ? "CUDA" : "CPU") + ", K=" + kv_dtype_name(kv_config.type_k) +
             ", V=" + kv_dtype_name(kv_config.type_v));

    return true;
}

// =========================================================================
// Per-layer init (Gemma4 mixed-attention)
// =========================================================================

bool KVCache::init_per_layer(int num_layers, const std::vector<int>& kv_dims, int max_seq_len,
                             DeviceType device) {
    kv_dim_per_layer_ = kv_dims;
    num_kv_heads_ = 0;
    head_dim_ = 0;
    max_seq_len_ = max_seq_len;
    device_ = device;
    kv_dtype_ = KVCacheDType::FP32;
    kv_config_ = {};

    layers_.resize(num_layers);

    for (int i = 0; i < num_layers; ++i) {
        int dim = (i < (int)kv_dims.size()) ? kv_dims[i] : kv_dims.back();
        if (dim > 0) {
            auto shape = std::vector<int64_t>{max_seq_len, dim};
            layers_[i].key_cache = std::make_shared<Tensor>(DataType::FP32, shape, device);
            layers_[i].value_cache = std::make_shared<Tensor>(DataType::FP32, shape, device);
            layers_[i].key_cache->zero_();
            layers_[i].value_cache->zero_();
        }
        layers_[i].filled = 0;
        init_cells(i);
    }

    LOG_INFO("KVCache init_per_layer: " + std::to_string(num_layers) + " layers, max_seq_len=" +
             std::to_string(max_seq_len) + ", " + (device == DeviceType::CUDA ? "CUDA" : "CPU"));
    return true;
}

// =========================================================================
// Reset
// =========================================================================

void KVCache::reset() {
    for (auto& layer : layers_) {
        layer.filled = 0;
        for (auto& cell : layer.cells) {
            cell.pos = -1;
            cell.seq_id_mask = 0;
        }
    }
}

// =========================================================================
// Update (legacy single-seq)
// =========================================================================

int KVCache::update(int layer, const TensorPtr& new_key, const TensorPtr& new_value, int seq_len) {
    if (layer < 0 || layer >= static_cast<int>(layers_.size())) {
        LOG_ERROR("KVCache::update: invalid layer " + std::to_string(layer));
        return -1;
    }
    int64_t start_pos = static_cast<int64_t>(layers_[layer].filled);
    return update(layer, 0, start_pos, new_key, new_value, seq_len);
}

// =========================================================================
// Update (sequence-aware)
// =========================================================================

int KVCache::update(int layer, int seq_id, int64_t pos,
                    const TensorPtr& new_key, const TensorPtr& new_value, int seq_len) {
    if (layer < 0 || layer >= static_cast<int>(layers_.size())) {
        LOG_ERROR("KVCache::update: invalid layer " + std::to_string(layer));
        return -1;
    }
    if (seq_id < 0 || seq_id >= max_seqs_) {
        LOG_ERROR("KVCache::update: invalid seq_id " + std::to_string(seq_id) +
                  ", max=" + std::to_string(max_seqs_));
        return -1;
    }
    if (pos + seq_len > max_seq_len_) {
        LOG_ERROR("KVCache::update: cache overflow, pos=" + std::to_string(pos) +
                  " seq_len=" + std::to_string(seq_len) + " max=" + std::to_string(max_seq_len_));
        return -1;
    }

    int result;
    if (kv_config_.type_k == KVCacheDType::FP32 && kv_config_.type_v == KVCacheDType::FP32) {
        result = update_fp32(layer, pos, new_key, new_value, seq_len);
    } else {
        result = update_quantized(layer, pos, new_key, new_value, seq_len);
    }
    if (result < 0)
        return result;

    // Update cell metadata
    auto& kv = layers_[layer];
    for (int s = 0; s < seq_len; ++s) {
        int slot = static_cast<int>(pos + s);
        kv.cells[slot].pos = pos + s;
        kv.cells[slot].add_seq(seq_id);
    }

    int new_end = static_cast<int>(pos + seq_len);
    if (new_end > kv.filled) {
        kv.filled = new_end;
    }

    return kv.filled;
}

// =========================================================================
// FP32 update
// =========================================================================

int KVCache::update_fp32(int layer, int64_t start_pos,
                         const TensorPtr& new_key, const TensorPtr& new_value, int seq_len) {
    auto& kv = layers_[layer];
    int kv_dim = this->kv_dim(layer);
    int filled = static_cast<int>(start_pos);

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
            cudaMemcpy(h_key.data(), new_key->data(), seq_len * kv_dim * sizeof(float),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(h_value.data(), new_value->data(), seq_len * kv_dim * sizeof(float),
                       cudaMemcpyDeviceToHost);
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

    return static_cast<int>(start_pos + seq_len);
}

// =========================================================================
// Per-type CPU quantize dispatch
// =========================================================================

static void quantize_cpu(KVCacheDType dtype, const float* data, uint8_t* q_data, int n) {
    switch (dtype) {
    case KVCacheDType::F16:  quantize_f16_cpu(data, q_data, n); break;
    case KVCacheDType::Q8_0: quantize_q8_0_cpu(data, q_data, n); break;
    case KVCacheDType::Q4_0: quantize_q4_0_cpu(data, q_data, n); break;
    case KVCacheDType::Q4_K: quantize_q4_k_cpu(data, q_data, n); break;
    default: break;
    }
}

static void dequantize_cpu(KVCacheDType dtype, const uint8_t* q_data, float* out, int n, int row) {
    switch (dtype) {
    case KVCacheDType::F16:  dequantize_f16_cpu(q_data, out, n); break;
    case KVCacheDType::Q8_0: dequantize_q8_0_cpu(q_data, out, n); break;
    case KVCacheDType::Q4_0: dequantize_q4_0_cpu(q_data, out, n); break;
    case KVCacheDType::Q4_K: dequantize_q4_k_cpu(q_data, out, n, row); break;
    default: break;
    }
}

// =========================================================================
// Quantized update (dispatches per type_k / type_v)
// =========================================================================

int KVCache::update_quantized(int layer, int64_t start_pos,
                              const TensorPtr& new_key, const TensorPtr& new_value, int seq_len) {
    // Use CUDA path if device is CUDA
    if (device_ == DeviceType::CUDA) {
        int result = update_quantized_cuda(layer, start_pos, new_key, new_value, seq_len);
        if (result >= 0)
            return result;
        // Fall through to CPU path if CUDA path fails
    }

    if (layer < 0 || layer >= static_cast<int>(layers_.size()))
        return -1;

    auto& kv = layers_[layer];
    int filled = static_cast<int>(start_pos);
    if (filled + seq_len > max_seq_len_)
        return -1;

    int kv_dim = num_kv_heads_ * head_dim_;
    size_t k_row_size = block_nbytes(kv_config_.type_k, kv_dim);
    size_t v_row_size = block_nbytes(kv_config_.type_v, kv_dim);

    std::vector<float> h_key(seq_len * kv_dim), h_value(seq_len * kv_dim);

    if (new_key->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        cudaMemcpy(h_key.data(), new_key->data(), seq_len * kv_dim * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(h_value.data(), new_value->data(), seq_len * kv_dim * sizeof(float),
                   cudaMemcpyDeviceToHost);
#endif
    } else {
        std::memcpy(h_key.data(), new_key->data(), seq_len * kv_dim * sizeof(float));
        std::memcpy(h_value.data(), new_value->data(), seq_len * kv_dim * sizeof(float));
    }

    for (int s = 0; s < seq_len; ++s) {
        if (kv_config_.type_k != KVCacheDType::FP32) {
            uint8_t* qk_dst = kv.q_key_cache.data() + (filled + s) * k_row_size;
            quantize_cpu(kv_config_.type_k, h_key.data() + s * kv_dim, qk_dst, kv_dim);
        }
        if (kv_config_.type_v != KVCacheDType::FP32) {
            uint8_t* qv_dst = kv.q_value_cache.data() + (filled + s) * v_row_size;
            quantize_cpu(kv_config_.type_v, h_value.data() + s * kv_dim, qv_dst, kv_dim);
        }
    }

    return static_cast<int>(start_pos + seq_len);
}

// =========================================================================
// Dequantize layer (dispatches per type_k / type_v)
// =========================================================================

void KVCache::dequantize_layer(int layer) {
    if (layer < 0 || layer >= static_cast<int>(layers_.size()))
        return;

    bool need_k = kv_config_.type_k != KVCacheDType::FP32;
    bool need_v = kv_config_.type_v != KVCacheDType::FP32;
    if (!need_k && !need_v)
        return;

    // Use CUDA path if device is CUDA and CUDA quantized cache exists
    if (device_ == DeviceType::CUDA && layers_[layer].d_q_key_cache) {
        dequantize_layer_cuda(layer);
        return;
    }

    auto& kv = layers_[layer];
    int filled = kv.filled;
    int kv_dim = num_kv_heads_ * head_dim_;
    size_t k_row_size = block_nbytes(kv_config_.type_k, kv_dim);
    size_t v_row_size = block_nbytes(kv_config_.type_v, kv_dim);

    if (need_k && filled > 0) {
        std::vector<float> h_out(filled * kv_dim);
        for (int s = 0; s < filled; ++s) {
            const uint8_t* qk_src = kv.q_key_cache.data() + s * k_row_size;
            dequantize_cpu(kv_config_.type_k, qk_src, h_out.data() + s * kv_dim, kv_dim, s);
        }
        if (device_ == DeviceType::CUDA) {
#ifdef USE_CUDA
            cudaMemcpy(kv.key_cache->data(), h_out.data(), filled * kv_dim * sizeof(float),
                       cudaMemcpyHostToDevice);
#endif
        } else {
            std::memcpy(kv.key_cache->data(), h_out.data(), filled * kv_dim * sizeof(float));
        }
    }

    if (need_v && filled > 0) {
        std::vector<float> h_out(filled * kv_dim);
        for (int s = 0; s < filled; ++s) {
            const uint8_t* qv_src = kv.q_value_cache.data() + s * v_row_size;
            dequantize_cpu(kv_config_.type_v, qv_src, h_out.data() + s * kv_dim, kv_dim, s);
        }
        if (device_ == DeviceType::CUDA) {
#ifdef USE_CUDA
            cudaMemcpy(kv.value_cache->data(), h_out.data(), filled * kv_dim * sizeof(float),
                       cudaMemcpyHostToDevice);
#endif
        } else {
            std::memcpy(kv.value_cache->data(), h_out.data(), filled * kv_dim * sizeof(float));
        }
    }
}

// =========================================================================
// Accessors
// =========================================================================

TensorPtr KVCache::get_key(int layer) const {
    if (layer < 0 || layer >= static_cast<int>(layers_.size()))
        return nullptr;
    return layers_[layer].key_cache;
}

TensorPtr KVCache::get_value(int layer) const {
    if (layer < 0 || layer >= static_cast<int>(layers_.size()))
        return nullptr;
    return layers_[layer].value_cache;
}

TensorPtr KVCache::get_key_filled(int layer) const {
    if (layer < 0 || layer >= static_cast<int>(layers_.size()))
        return nullptr;
    const auto& kv = layers_[layer];
    if (!kv.key_cache)
        return nullptr;
    int f = kv.filled;
    if (f == max_seq_len_)
        return kv.key_cache;
    return std::make_shared<Tensor>(kv.key_cache->slice(0, 0, f));
}

TensorPtr KVCache::get_value_filled(int layer) const {
    if (layer < 0 || layer >= static_cast<int>(layers_.size()))
        return nullptr;
    const auto& kv = layers_[layer];
    if (!kv.value_cache)
        return nullptr;
    int f = kv.filled;
    if (f == max_seq_len_)
        return kv.value_cache;
    return std::make_shared<Tensor>(kv.value_cache->slice(0, 0, f));
}

int KVCache::filled(int layer) const {
    if (layer < 0 || layer >= static_cast<int>(layers_.size()))
        return 0;
    return layers_[layer].filled;
}

size_t KVCache::nbytes() const {
    if (!kv_dim_per_layer_.empty()) {
        size_t total = 0;
        for (int i = 0; i < (int)layers_.size(); ++i) {
            int dim = (i < (int)kv_dim_per_layer_.size()) ? kv_dim_per_layer_[i] : 0;
            if (dim > 0) {
                total += max_seq_len_ * dim * sizeof(float) * 2;
            }
        }
        return total;
    }
    int kv_dim = num_kv_heads_ * head_dim_;
    size_t per_layer = 0;
    per_layer += max_seq_len_ * block_nbytes(kv_config_.type_k, kv_dim);
    per_layer += max_seq_len_ * block_nbytes(kv_config_.type_v, kv_dim);
    return per_layer * layers_.size();
}

// =========================================================================
// CUDA quantized helpers
// =========================================================================

void KVCache::alloc_cuda_q_cache(int layer, size_t bytes) {
#ifdef USE_CUDA
    auto& kv = layers_[layer];
    if (kv.d_q_cache_bytes >= bytes)
        return;
    if (kv.d_q_key_cache)
        cudaFree(kv.d_q_key_cache);
    if (kv.d_q_value_cache)
        cudaFree(kv.d_q_value_cache);
    cudaMalloc(&kv.d_q_key_cache, bytes);
    cudaMalloc(&kv.d_q_value_cache, bytes);
    kv.d_q_cache_bytes = bytes;
#endif
}

int KVCache::update_quantized_cuda(int layer, int64_t start_pos,
                                   const TensorPtr& new_key, const TensorPtr& new_value,
                                   int seq_len) {
#ifdef USE_CUDA
    if (layer < 0 || layer >= static_cast<int>(layers_.size()))
        return -1;

    auto& kv = layers_[layer];
    int filled = static_cast<int>(start_pos);
    if (filled + seq_len > max_seq_len_)
        return -1;

    int kv_dim = num_kv_heads_ * head_dim_;
    size_t k_row_size = block_nbytes(kv_config_.type_k, kv_dim);
    size_t v_row_size = block_nbytes(kv_config_.type_v, kv_dim);
    size_t total_k_bytes = max_seq_len_ * k_row_size;
    size_t total_v_bytes = max_seq_len_ * v_row_size;

    alloc_cuda_q_cache(layer, std::max(total_k_bytes, total_v_bytes));

    const float* k_src = static_cast<const float*>(new_key->data());
    const float* v_src = static_cast<const float*>(new_value->data());

    std::vector<float> h_key, h_value;  // CPU staging buffers (only used if input is on CPU)
    const float* d_k = k_src;
    const float* d_v = v_src;
    float* d_temp = nullptr;

    if (new_key->device() == DeviceType::CPU) {
        h_key.resize(seq_len * kv_dim);
        h_value.resize(seq_len * kv_dim);
        std::memcpy(h_key.data(), k_src, seq_len * kv_dim * sizeof(float));
        std::memcpy(h_value.data(), v_src, seq_len * kv_dim * sizeof(float));
        cudaMalloc(&d_temp, seq_len * kv_dim * sizeof(float) * 2);
        cudaMemcpy(d_temp, h_key.data(), seq_len * kv_dim * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_temp + seq_len * kv_dim, h_value.data(), seq_len * kv_dim * sizeof(float),
                   cudaMemcpyHostToDevice);
        d_k = d_temp;
        d_v = d_temp + seq_len * kv_dim;
    }

    // Quantize K
    if (kv_config_.type_k != KVCacheDType::FP32) {
        uint8_t* q_dst = static_cast<uint8_t*>(kv.d_q_key_cache) + filled * k_row_size;
        switch (kv_config_.type_k) {
        case KVCacheDType::F16:
            cuda::launch_quantize_f16_matrix(d_k, q_dst, seq_len, kv_dim);
            break;
        case KVCacheDType::Q8_0:
            cuda::launch_quantize_q8_0_matrix(d_k, q_dst, seq_len, kv_dim);
            break;
        case KVCacheDType::Q4_0:
            cuda::launch_quantize_q4_0_matrix(d_k, q_dst, seq_len, kv_dim);
            break;
        case KVCacheDType::Q4_K:
            cuda::launch_quantize_q4_k_matrix(d_k, q_dst, seq_len, kv_dim);
            break;
        default:
            break;
        }
    }

    // Quantize V
    if (kv_config_.type_v != KVCacheDType::FP32) {
        uint8_t* q_dst = static_cast<uint8_t*>(kv.d_q_value_cache) + filled * v_row_size;
        switch (kv_config_.type_v) {
        case KVCacheDType::F16:
            cuda::launch_quantize_f16_matrix(d_v, q_dst, seq_len, kv_dim);
            break;
        case KVCacheDType::Q8_0:
            cuda::launch_quantize_q8_0_matrix(d_v, q_dst, seq_len, kv_dim);
            break;
        case KVCacheDType::Q4_0:
            cuda::launch_quantize_q4_0_matrix(d_v, q_dst, seq_len, kv_dim);
            break;
        case KVCacheDType::Q4_K:
            cuda::launch_quantize_q4_k_matrix(d_v, q_dst, seq_len, kv_dim);
            break;
        default:
            break;
        }
    }

    if (d_temp)
        cudaFree(d_temp);
    return static_cast<int>(start_pos + seq_len);
#else
    return -1;
#endif
}

void KVCache::dequantize_layer_cuda(int layer) {
#ifdef USE_CUDA
    if (layer < 0 || layer >= static_cast<int>(layers_.size()))
        return;

    auto& kv = layers_[layer];
    int filled = kv.filled;
    int kv_dim = num_kv_heads_ * head_dim_;

    if (!kv.d_q_key_cache || filled == 0)
        return;

    float* k_out = static_cast<float*>(kv.key_cache->data());
    float* v_out = static_cast<float*>(kv.value_cache->data());

    // Dequantize K
    if (kv_config_.type_k != KVCacheDType::FP32) {
        switch (kv_config_.type_k) {
        case KVCacheDType::F16:
            cuda::launch_dequant_f16_matrix(kv.d_q_key_cache, k_out, filled, kv_dim);
            break;
        case KVCacheDType::Q8_0:
            cuda::launch_dequant_q8_0_matrix(kv.d_q_key_cache, k_out, filled, kv_dim);
            break;
        case KVCacheDType::Q4_0:
            cuda::launch_dequant_q4_0_matrix(kv.d_q_key_cache, k_out, filled, kv_dim);
            break;
        case KVCacheDType::Q4_K:
            cuda::launch_dequant_q4_k_matrix(kv.d_q_key_cache, k_out, filled, kv_dim);
            break;
        default:
            break;
        }
    }

    // Dequantize V
    if (kv_config_.type_v != KVCacheDType::FP32) {
        switch (kv_config_.type_v) {
        case KVCacheDType::F16:
            cuda::launch_dequant_f16_matrix(kv.d_q_value_cache, v_out, filled, kv_dim);
            break;
        case KVCacheDType::Q8_0:
            cuda::launch_dequant_q8_0_matrix(kv.d_q_value_cache, v_out, filled, kv_dim);
            break;
        case KVCacheDType::Q4_0:
            cuda::launch_dequant_q4_0_matrix(kv.d_q_value_cache, v_out, filled, kv_dim);
            break;
        case KVCacheDType::Q4_K:
            cuda::launch_dequant_q4_k_matrix(kv.d_q_value_cache, v_out, filled, kv_dim);
            break;
        default:
            break;
        }
    }
#endif
}

// =========================================================================
// Sequence operations (unchanged from original)
// =========================================================================

void KVCache::seq_rm(int seq_id, int64_t p0, int64_t p1) {
    if (seq_id < 0 || seq_id >= max_seqs_) {
        LOG_ERROR("KVCache::seq_rm: invalid seq_id " + std::to_string(seq_id));
        return;
    }
    uint32_t bit = 1u << seq_id;

    for (auto& layer : layers_) {
        for (int i = 0; i < static_cast<int>(layer.cells.size()); ++i) {
            auto& cell = layer.cells[i];
            if (cell.is_free())
                continue;
            if (cell.pos < p0 || cell.pos >= p1)
                continue;
            if (!(cell.seq_id_mask & bit))
                continue;

            cell.seq_id_mask &= ~bit;

            if (cell.no_seqs()) {
                cell.pos = -1;
            }
        }

        int max_pos = -1;
        for (const auto& cell : layer.cells) {
            if (!cell.is_free() && cell.pos > max_pos) {
                max_pos = static_cast<int>(cell.pos);
            }
        }
        layer.filled = (max_pos >= 0) ? max_pos + 1 : 0;
    }
}

void KVCache::seq_cp(int src_seq, int dst_seq, int64_t p0, int64_t p1) {
    if (src_seq < 0 || src_seq >= max_seqs_ || dst_seq < 0 || dst_seq >= max_seqs_) {
        LOG_ERROR("KVCache::seq_cp: invalid seq_id src=" + std::to_string(src_seq) +
                  " dst=" + std::to_string(dst_seq));
        return;
    }
    uint32_t src_bit = 1u << src_seq;
    uint32_t dst_bit = 1u << dst_seq;

    for (auto& layer : layers_) {
        for (auto& cell : layer.cells) {
            if (cell.is_free())
                continue;
            if (cell.pos < p0 || cell.pos >= p1)
                continue;
            if (!(cell.seq_id_mask & src_bit))
                continue;

            cell.seq_id_mask |= dst_bit;
        }
    }
}

void KVCache::seq_keep(int seq_id) {
    if (seq_id < 0 || seq_id >= max_seqs_) {
        LOG_ERROR("KVCache::seq_keep: invalid seq_id " + std::to_string(seq_id));
        return;
    }
    uint32_t keep_bit = 1u << seq_id;

    for (auto& layer : layers_) {
        for (auto& cell : layer.cells) {
            if (cell.is_free())
                continue;

            if (cell.seq_id_mask & keep_bit) {
                cell.seq_id_mask = keep_bit;
            } else {
                cell.pos = -1;
                cell.seq_id_mask = 0;
            }
        }

        int max_pos = -1;
        for (const auto& cell : layer.cells) {
            if (!cell.is_free() && cell.pos > max_pos) {
                max_pos = static_cast<int>(cell.pos);
            }
        }
        layer.filled = (max_pos >= 0) ? max_pos + 1 : 0;
    }
}

int KVCache::find_slot(int layer) const {
    if (layer < 0 || layer >= static_cast<int>(layers_.size()))
        return -1;

    const auto& cells = layers_[layer].cells;
    for (int i = 0; i < static_cast<int>(cells.size()); ++i) {
        if (cells[i].is_free())
            return i;
    }
    return -1;
}

int KVCache::seq_filled(int layer, int seq_id) const {
    if (layer < 0 || layer >= static_cast<int>(layers_.size()))
        return 0;
    if (seq_id < 0 || seq_id >= max_seqs_)
        return 0;

    uint32_t bit = 1u << seq_id;
    int count = 0;
    for (const auto& cell : layers_[layer].cells) {
        if (!cell.is_free() && (cell.seq_id_mask & bit)) {
            ++count;
        }
    }
    return count;
}

}  // namespace forge
