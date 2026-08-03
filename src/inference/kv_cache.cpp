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
        h = (sign << 15) | 0x7C00 | (mant ? 1 : 0);
    } else if (exp >= 143) {
        h = (sign << 15) | 0x7C00;
    } else if (exp >= 113) {
        h = (sign << 15) | ((exp - 112) << 10) | (mant >> 13);
    } else if (exp >= 103) {
        uint32_t sub_exp = 126 - exp;
        uint32_t sub_mant = (mant | 0x800000) >> (sub_exp + 13);
        h = (sign << 15) | sub_mant;
    } else {
        h = (sign << 15);
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

        std::memset(block + 4, 0x01, 12);

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
// Per-type CPU quantize/dequantize dispatch
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
// Block size calculation
// =========================================================================

size_t KVCache::q4_0_block_nbytes(int n) {
    int num_blocks = (n + 31) / 32;
    return num_blocks * 18;
}

size_t KVCache::block_nbytes(KVCacheDType dtype, int n) {
    switch (dtype) {
    case KVCacheDType::FP32:
        return n * sizeof(float);
    case KVCacheDType::F16:
        return n * sizeof(uint16_t);
    case KVCacheDType::Q8_0: {
        int num_blocks = (n + 31) / 32;
        return num_blocks * 34;
    }
    case KVCacheDType::Q4_0:
        return q4_0_block_nbytes(n);
    case KVCacheDType::Q4_K: {
        int num_blocks = (n + 255) / 256;
        return num_blocks * 144;
    }
    default:
        return n * sizeof(float);
    }
}

// =========================================================================
// KVCacheStorage implementation
// =========================================================================

KVCacheStorage::~KVCacheStorage() {
#ifdef USE_CUDA
    if (d_data) {
        cudaFree(d_data);
        d_data = nullptr;
    }
#endif
}

KVCacheStorage::KVCacheStorage(KVCacheStorage&& o) noexcept
    : dtype(o.dtype), device(o.device), tensor(std::move(o.tensor)),
      h_data(std::move(o.h_data)), d_data(o.d_data), d_bytes(o.d_bytes),
      max_rows(o.max_rows), row_bytes(o.row_bytes) {
    o.d_data = nullptr;
    o.d_bytes = 0;
}

KVCacheStorage& KVCacheStorage::operator=(KVCacheStorage&& o) noexcept {
    if (this != &o) {
#ifdef USE_CUDA
        if (d_data)
            cudaFree(d_data);
#endif
        dtype = o.dtype;
        device = o.device;
        tensor = std::move(o.tensor);
        h_data = std::move(o.h_data);
        d_data = o.d_data;
        d_bytes = o.d_bytes;
        max_rows = o.max_rows;
        row_bytes = o.row_bytes;
        o.d_data = nullptr;
        o.d_bytes = 0;
    }
    return *this;
}

void KVCacheStorage::alloc(KVCacheDType dt, DeviceType dev, int rows, size_t rbytes) {
    dtype = dt;
    device = dev;
    max_rows = rows;
    row_bytes = rbytes;

    size_t total = static_cast<size_t>(max_rows) * row_bytes;

    if (dtype == KVCacheDType::FP32) {
        // FP32: allocate as Tensor
        int dim = static_cast<int>(row_bytes / sizeof(float));
        auto shape = std::vector<int64_t>{max_rows, dim};
        tensor = std::make_shared<Tensor>(DataType::FP32, shape, device);
        tensor->zero_();
    } else {
        // Quantized: allocate h_data or d_data depending on device
        if (device == DeviceType::CPU) {
            h_data.resize(total, 0);
        } else {
#ifdef USE_CUDA
            if (d_data)
                cudaFree(d_data);
            d_bytes = total;
            cudaMalloc(&d_data, total);
            cudaMemset(d_data, 0, total);
#endif
        }
    }
}

void KVCacheStorage::zero_fill() {
    if (dtype == KVCacheDType::FP32) {
        if (tensor)
            tensor->zero_();
    } else {
        if (device == DeviceType::CPU) {
            std::fill(h_data.begin(), h_data.end(), 0);
        } else {
#ifdef USE_CUDA
            if (d_data && d_bytes > 0)
                cudaMemset(d_data, 0, d_bytes);
#endif
        }
    }
}

size_t KVCacheStorage::capacity_bytes() const {
    if (dtype == KVCacheDType::FP32) {
        return tensor ? tensor->nbytes() : 0;
    }
    if (device == DeviceType::CPU) {
        return h_data.size();
    }
    return d_bytes;
}

float* KVCacheStorage::fp32_data() {
    return (dtype == KVCacheDType::FP32 && tensor) ? static_cast<float*>(tensor->data()) : nullptr;
}

const float* KVCacheStorage::fp32_data() const {
    return (dtype == KVCacheDType::FP32 && tensor) ? static_cast<const float*>(tensor->data())
                                                    : nullptr;
}

uint8_t* KVCacheStorage::q_data() {
    return (device == DeviceType::CPU && dtype != KVCacheDType::FP32) ? h_data.data() : nullptr;
}

const uint8_t* KVCacheStorage::q_data() const {
    return (device == DeviceType::CPU && dtype != KVCacheDType::FP32) ? h_data.data() : nullptr;
}

void* KVCacheStorage::d_q_data() {
    return (device == DeviceType::CUDA && dtype != KVCacheDType::FP32) ? d_data : nullptr;
}

const void* KVCacheStorage::d_q_data() const {
    return (device == DeviceType::CUDA && dtype != KVCacheDType::FP32) ? d_data : nullptr;
}

uint8_t* KVCacheStorage::q_row(int row) {
    return h_data.data() + static_cast<size_t>(row) * row_bytes;
}

const uint8_t* KVCacheStorage::q_row(int row) const {
    return h_data.data() + static_cast<size_t>(row) * row_bytes;
}

float* KVCacheStorage::fp32_row(int row) {
    return fp32_data() + static_cast<size_t>(row) * (row_bytes / sizeof(float));
}

const float* KVCacheStorage::fp32_row(int row) const {
    return fp32_data() + static_cast<size_t>(row) * (row_bytes / sizeof(float));
}

// =========================================================================
// KVCache destructor
// =========================================================================

KVCache::~KVCache() = default;

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
    kv_dtype_ = static_cast<KVCacheDType>(
        std::max(static_cast<int>(kv_config.type_k), static_cast<int>(kv_config.type_v)));

    layers_.resize(num_layers);

    int kv_dim = num_kv_heads * head_dim;

    for (int i = 0; i < num_layers; ++i) {
        DeviceType layer_dev = layer_device(i);

        // Allocate key storage in native dtype — no FP32 shadow for quantized modes
        size_t k_row_bytes = block_nbytes(kv_config.type_k, kv_dim);
        size_t v_row_bytes = block_nbytes(kv_config.type_v, kv_dim);
        layers_[i].key_store.alloc(kv_config.type_k, layer_dev, max_seq_len, k_row_bytes);
        layers_[i].value_store.alloc(kv_config.type_v, layer_dev, max_seq_len, v_row_bytes);
        layers_[i].filled = 0;

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
        DeviceType layer_dev = layer_device(i);
        if (dim > 0) {
            size_t row_bytes = dim * sizeof(float);
            layers_[i].key_store.alloc(KVCacheDType::FP32, layer_dev, max_seq_len, row_bytes);
            layers_[i].value_store.alloc(KVCacheDType::FP32, layer_dev, max_seq_len, row_bytes);
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
        layer.key_store.zero_fill();
        layer.value_store.zero_fill();
    }
    for (auto& c : ring_cursor_)
        c = 0;
}

void KVCache::rollback(int64_t to_pos) {
    for (auto& layer : layers_) {
        if (layer.filled > to_pos) {
            for (int64_t p = to_pos; p < layer.filled; ++p) {
                if (p >= 0 && p < static_cast<int64_t>(layer.cells.size())) {
                    layer.cells[p].pos = -1;
                    layer.cells[p].seq_id_mask = 0;
                }
            }
            layer.filled = static_cast<int>(to_pos);
        }
    }
}

// =========================================================================
// Ring buffer mode
// =========================================================================

void KVCache::set_ring_buffer(int window_size, int layer) {
    if (window_size <= 0) {
        window_size_ = 0;
        use_ring_buffer_.assign(layers_.size(), false);
        ring_cursor_.assign(layers_.size(), 0);
        return;
    }
    window_size_ = window_size;
    if (use_ring_buffer_.size() != layers_.size()) {
        use_ring_buffer_.assign(layers_.size(), false);
        ring_cursor_.assign(layers_.size(), 0);
    }
    if (layer < 0) {
        use_ring_buffer_.assign(layers_.size(), true);
    } else if (layer < static_cast<int>(layers_.size())) {
        use_ring_buffer_[layer] = true;
    }
    int num_ring = 0;
    for (auto b : use_ring_buffer_) if (b) ++num_ring;
    LOG_INFO("KVCache ring buffer: window_size=" + std::to_string(window_size) +
             ", ring_layers=" + std::to_string(num_ring) + "/" + std::to_string(layers_.size()));
}

bool KVCache::use_ring_buffer(int layer) const {
    if (layer < 0 || layer >= static_cast<int>(use_ring_buffer_.size()))
        return false;
    return use_ring_buffer_[layer];
}

// =========================================================================
// Per-layer device management
// =========================================================================

void KVCache::set_layer_devices(const std::vector<DeviceType>& layer_devices) {
    if (layer_devices.empty()) {
        layer_devices_.clear();
        return;
    }
    layer_devices_ = layer_devices;
    if (static_cast<int>(layer_devices_.size()) < static_cast<int>(layers_.size())) {
        layer_devices_.resize(layers_.size(), device_);
    }

    // Re-allocate each layer's KV storage on its target device
    for (int i = 0; i < static_cast<int>(layers_.size()); ++i) {
        DeviceType target = layer_devices_[i];
        auto& kv = layers_[i];

        // For FP32 mode: move tensors via copy
        if (kv.key_store.dtype == KVCacheDType::FP32 && kv.key_store.tensor &&
            kv.key_store.tensor->device() != target) {
            auto new_key = std::make_shared<Tensor>(kv.key_store.tensor->dtype(),
                                                    kv.key_store.tensor->shape(), target);
            new_key->copy_from(*kv.key_store.tensor);
            kv.key_store.tensor = new_key;
            kv.key_store.device = target;
        }
        if (kv.value_store.dtype == KVCacheDType::FP32 && kv.value_store.tensor &&
            kv.value_store.tensor->device() != target) {
            auto new_val = std::make_shared<Tensor>(kv.value_store.tensor->dtype(),
                                                    kv.value_store.tensor->shape(), target);
            new_val->copy_from(*kv.value_store.tensor);
            kv.value_store.tensor = new_val;
            kv.value_store.device = target;
        }

        // For quantized mode: if the device changed, we need to re-allocate
        // (copy from host to device or vice versa)
        if (kv.key_store.dtype != KVCacheDType::FP32 && kv.key_store.device != target) {
            size_t k_row_bytes = kv.key_store.row_bytes;
            size_t v_row_bytes = kv.value_store.row_bytes;
            int rows = kv.key_store.max_rows;

            // Save existing data temporarily (only host data can be saved)
            // For simplicity, re-allocate and zero-fill; the caller should
            // call set_layer_devices before any KV updates
            kv.key_store = KVCacheStorage();
            kv.value_store = KVCacheStorage();
            kv.key_store.alloc(kv_config_.type_k, target, rows, k_row_bytes);
            kv.value_store.alloc(kv_config_.type_v, target, rows, v_row_bytes);
        }
    }

    int num_cuda = 0;
    for (auto d : layer_devices_) if (d == DeviceType::CUDA) ++num_cuda;
    LOG_INFO("KVCache per-layer device: " + std::to_string(num_cuda) + "/" +
             std::to_string(layers_.size()) + " layers on CUDA");
}

DeviceType KVCache::layer_device(int layer) const {
    if (layer_devices_.empty() || layer < 0 || layer >= static_cast<int>(layer_devices_.size()))
        return device_;
    return layer_devices_[layer];
}

// =========================================================================
// CUDA stream management
// =========================================================================

void KVCache::set_cuda_stream(void* stream) {
    cuda_stream_ = stream;
}

void* KVCache::cuda_stream() const {
    return cuda_stream_;
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

    bool layer_uses_ring = use_ring_buffer(layer);
    int64_t write_pos = pos;
    if (layer_uses_ring) {
        write_pos = static_cast<int64_t>(ring_cursor_[layer]);
    }

    if (write_pos + seq_len > max_seq_len_) {
        LOG_ERROR("KVCache::update: cache overflow, write_pos=" + std::to_string(write_pos) +
                  " seq_len=" + std::to_string(seq_len) + " max=" + std::to_string(max_seq_len_));
        return -1;
    }

    int result;
    if (kv_config_.type_k == KVCacheDType::FP32 && kv_config_.type_v == KVCacheDType::FP32) {
        result = update_fp32(layer, write_pos, new_key, new_value, seq_len);
    } else {
        result = update_quantized(layer, write_pos, new_key, new_value, seq_len);
    }
    if (result < 0)
        return result;

    // Update cell metadata
    auto& kv = layers_[layer];
    for (int s = 0; s < seq_len; ++s) {
        int slot = static_cast<int>(write_pos + s);
        kv.cells[slot].pos = pos + s;
        kv.cells[slot].add_seq(seq_id);
    }

    int new_end = static_cast<int>(write_pos + seq_len);
    if (new_end > kv.filled) {
        kv.filled = new_end;
    }

    if (layer_uses_ring) {
        ring_cursor_[layer] = (ring_cursor_[layer] + seq_len) % window_size_;
    }

    if (layer_uses_ring) {
        return std::min(kv.filled, window_size_);
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
    DeviceType layer_dev = kv.key_store.device;

#ifdef USE_CUDA
    cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream());
#endif

    if (layer_dev == DeviceType::CUDA) {
#ifdef USE_CUDA
        float* k_dst = kv.key_store.fp32_row(filled);
        float* v_dst = kv.value_store.fp32_row(filled);
        const float* k_src = static_cast<const float*>(new_key->data());
        const float* v_src = static_cast<const float*>(new_value->data());
        size_t copy_bytes = seq_len * kv_dim * sizeof(float);
        if (new_key->device() == DeviceType::CUDA) {
            cudaMemcpyAsync(k_dst, k_src, copy_bytes, cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(v_dst, v_src, copy_bytes, cudaMemcpyDeviceToDevice, stream);
        } else {
            cudaMemcpyAsync(k_dst, k_src, copy_bytes, cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(v_dst, v_src, copy_bytes, cudaMemcpyHostToDevice, stream);
        }
#endif
    } else {
        float* k_dst = kv.key_store.fp32_row(filled);
        float* v_dst = kv.value_store.fp32_row(filled);

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
// Quantized update (dispatches per type_k / type_v)
// =========================================================================

int KVCache::update_quantized(int layer, int64_t start_pos,
                              const TensorPtr& new_key, const TensorPtr& new_value, int seq_len) {
    DeviceType layer_dev = this->layer_device(layer);
    if (layer_dev == DeviceType::CUDA) {
        int result = update_quantized_cuda(layer, start_pos, new_key, new_value, seq_len);
        if (result >= 0)
            return result;
    }

    if (layer < 0 || layer >= static_cast<int>(layers_.size()))
        return -1;

    auto& kv = layers_[layer];
    int filled = static_cast<int>(start_pos);
    if (filled + seq_len > max_seq_len_)
        return -1;

    int kv_dim = num_kv_heads_ * head_dim_;

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
            uint8_t* qk_dst = kv.key_store.q_row(filled + s);
            quantize_cpu(kv_config_.type_k, h_key.data() + s * kv_dim, qk_dst, kv_dim);
        }
        if (kv_config_.type_v != KVCacheDType::FP32) {
            uint8_t* qv_dst = kv.value_store.q_row(filled + s);
            quantize_cpu(kv_config_.type_v, h_value.data() + s * kv_dim, qv_dst, kv_dim);
        }
    }

    return static_cast<int>(start_pos + seq_len);
}

// =========================================================================
// Dequantize layer (on-demand, for non-fused attention paths)
// =========================================================================

void KVCache::dequantize_layer(int layer) {
    if (layer < 0 || layer >= static_cast<int>(layers_.size()))
        return;

    bool need_k = kv_config_.type_k != KVCacheDType::FP32;
    bool need_v = kv_config_.type_v != KVCacheDType::FP32;
    if (!need_k && !need_v)
        return;

    // Use CUDA path if this layer's device is CUDA and CUDA quantized cache exists
    if (this->layer_device(layer) == DeviceType::CUDA && layers_[layer].key_store.d_q_data()) {
        dequantize_layer_cuda(layer);
        return;
    }

    auto& kv = layers_[layer];
    int filled = kv.filled;
    int kv_dim = num_kv_heads_ * head_dim_;

    // Dequantize into the FP32 shadow tensor (which is the key_store/value_store tensor
    // in the old model). But now we don't have a permanent FP32 shadow — so we create
    // a temporary FP32 result. The caller (attention_executor) will use get_key_filled()
    // which handles this.
    //
    // For CPU quantized: dequantize into a temporary buffer, then write back to
    // the key_store/value_store FP32 tensor. But since we no longer have a permanent
    // FP32 tensor for quantized modes, we need a different approach.
    //
    // For backward compatibility: dequantize_layer() still needs to produce FP32 data
    // that get_key_filled() can return. We temporarily allocate FP32 tensors for this.
    // This is only called for non-fused paths (rare in production with quantized KV).

    if (need_k && filled > 0) {
        // Allocate a temporary FP32 tensor if not already present
        if (!kv.key_store.tensor) {
            auto shape = std::vector<int64_t>{max_seq_len_, kv_dim};
            kv.key_store.tensor = std::make_shared<Tensor>(DataType::FP32, shape, kv.key_store.device);
        }
        std::vector<float> h_out(filled * kv_dim);
        for (int s = 0; s < filled; ++s) {
            const uint8_t* qk_src = kv.key_store.q_row(s);
            dequantize_cpu(kv_config_.type_k, qk_src, h_out.data() + s * kv_dim, kv_dim, s);
        }
        if (kv.key_store.device == DeviceType::CUDA) {
#ifdef USE_CUDA
            cudaMemcpy(kv.key_store.tensor->data(), h_out.data(), filled * kv_dim * sizeof(float),
                       cudaMemcpyHostToDevice);
#endif
        } else {
            std::memcpy(kv.key_store.tensor->data(), h_out.data(), filled * kv_dim * sizeof(float));
        }
    }

    if (need_v && filled > 0) {
        if (!kv.value_store.tensor) {
            auto shape = std::vector<int64_t>{max_seq_len_, kv_dim};
            kv.value_store.tensor = std::make_shared<Tensor>(DataType::FP32, shape, kv.value_store.device);
        }
        std::vector<float> h_out(filled * kv_dim);
        for (int s = 0; s < filled; ++s) {
            const uint8_t* qv_src = kv.value_store.q_row(s);
            dequantize_cpu(kv_config_.type_v, qv_src, h_out.data() + s * kv_dim, kv_dim, s);
        }
        if (kv.value_store.device == DeviceType::CUDA) {
#ifdef USE_CUDA
            cudaMemcpy(kv.value_store.tensor->data(), h_out.data(), filled * kv_dim * sizeof(float),
                       cudaMemcpyHostToDevice);
#endif
        } else {
            std::memcpy(kv.value_store.tensor->data(), h_out.data(), filled * kv_dim * sizeof(float));
        }
    }
}

// =========================================================================
// Accessors
// =========================================================================

TensorPtr KVCache::get_key(int layer) const {
    if (layer < 0 || layer >= static_cast<int>(layers_.size()))
        return nullptr;
    const auto& store = layers_[layer].key_store;

    // FP32 mode: return the tensor directly
    if (store.dtype == KVCacheDType::FP32 && store.tensor)
        return store.tensor;

    // Quantized mode: return the dequantized FP32 tensor if available
    // (populated by dequantize_layer)
    if (store.tensor)
        return store.tensor;

    return nullptr;
}

TensorPtr KVCache::get_value(int layer) const {
    if (layer < 0 || layer >= static_cast<int>(layers_.size()))
        return nullptr;
    const auto& store = layers_[layer].value_store;

    if (store.dtype == KVCacheDType::FP32 && store.tensor)
        return store.tensor;

    if (store.tensor)
        return store.tensor;

    return nullptr;
}

TensorPtr KVCache::get_key_filled(int layer) const {
    if (layer < 0 || layer >= static_cast<int>(layers_.size()))
        return nullptr;
    const auto& kv = layers_[layer];
    const auto& store = kv.key_store;

    // Need FP32 tensor for attention (either native FP32 or dequantized)
    if (!store.tensor)
        return nullptr;

    int f = kv.filled;

    if (use_ring_buffer(layer) && f > window_size_) {
        int cursor = ring_cursor_[layer];
        int kvd = this->kv_dim(layer);
        auto out = std::make_shared<Tensor>(DataType::FP32,
                                            std::vector<int64_t>{window_size_, kvd},
                                            store.device);
        float* dst = static_cast<float*>(out->data());
        const float* src = static_cast<const float*>(store.tensor->data());
        int seg1_len = window_size_ - cursor;
        if (seg1_len > 0)
            std::memcpy(dst, src + cursor * kvd, seg1_len * kvd * sizeof(float));
        if (cursor > 0)
            std::memcpy(dst + seg1_len * kvd, src, cursor * kvd * sizeof(float));
        return out;
    }

    int eff = use_ring_buffer(layer) ? std::min(f, window_size_) : f;
    if (eff == max_seq_len_)
        return store.tensor;
    return std::make_shared<Tensor>(store.tensor->slice(0, 0, eff));
}

TensorPtr KVCache::get_value_filled(int layer) const {
    if (layer < 0 || layer >= static_cast<int>(layers_.size()))
        return nullptr;
    const auto& kv = layers_[layer];
    const auto& store = kv.value_store;

    if (!store.tensor)
        return nullptr;

    int f = kv.filled;

    if (use_ring_buffer(layer) && f > window_size_) {
        int cursor = ring_cursor_[layer];
        int kvd = this->kv_dim(layer);
        auto out = std::make_shared<Tensor>(DataType::FP32,
                                            std::vector<int64_t>{window_size_, kvd},
                                            store.device);
        float* dst = static_cast<float*>(out->data());
        const float* src = static_cast<const float*>(store.tensor->data());
        int seg1_len = window_size_ - cursor;
        if (seg1_len > 0)
            std::memcpy(dst, src + cursor * kvd, seg1_len * kvd * sizeof(float));
        if (cursor > 0)
            std::memcpy(dst + seg1_len * kvd, src, cursor * kvd * sizeof(float));
        return out;
    }

    int eff = use_ring_buffer(layer) ? std::min(f, window_size_) : f;
    if (eff == max_seq_len_)
        return store.tensor;
    return std::make_shared<Tensor>(store.tensor->slice(0, 0, eff));
}

int KVCache::filled(int layer) const {
    if (layer < 0 || layer >= static_cast<int>(layers_.size()))
        return 0;
    int raw = layers_[layer].filled;
    if (use_ring_buffer(layer))
        return std::min(raw, window_size_);
    return raw;
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

void* KVCache::d_q_key_cache(int layer) const {
    if (layer < 0 || layer >= static_cast<int>(layers_.size()))
        return nullptr;
    return const_cast<void*>(layers_[layer].key_store.d_q_data());
}

void* KVCache::d_q_value_cache(int layer) const {
    if (layer < 0 || layer >= static_cast<int>(layers_.size()))
        return nullptr;
    return const_cast<void*>(layers_[layer].value_store.d_q_data());
}

size_t KVCache::key_row_bytes(int layer) const {
    if (layer < 0 || layer >= static_cast<int>(layers_.size()))
        return 0;
    return layers_[layer].key_store.row_bytes;
}

size_t KVCache::value_row_bytes(int layer) const {
    if (layer < 0 || layer >= static_cast<int>(layers_.size()))
        return 0;
    return layers_[layer].value_store.row_bytes;
}

// =========================================================================
// CUDA quantized update
// =========================================================================

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
    size_t k_row_size = kv.key_store.row_bytes;
    size_t v_row_size = kv.value_store.row_bytes;

    // Ensure CUDA quantized buffers exist and are large enough
    size_t total_k_bytes = static_cast<size_t>(max_seq_len_) * k_row_size;
    size_t total_v_bytes = static_cast<size_t>(max_seq_len_) * v_row_size;

    if (!kv.key_store.d_q_data() || kv.key_store.d_bytes < total_k_bytes) {
        // Need to allocate/reallocate
        if (kv.key_store.d_data)
            cudaFree(kv.key_store.d_data);
        kv.key_store.d_bytes = total_k_bytes;
        cudaMalloc(&kv.key_store.d_data, total_k_bytes);
        cudaMemset(kv.key_store.d_data, 0, total_k_bytes);
        kv.key_store.device = DeviceType::CUDA;
    }
    if (!kv.value_store.d_q_data() || kv.value_store.d_bytes < total_v_bytes) {
        if (kv.value_store.d_data)
            cudaFree(kv.value_store.d_data);
        kv.value_store.d_bytes = total_v_bytes;
        cudaMalloc(&kv.value_store.d_data, total_v_bytes);
        cudaMemset(kv.value_store.d_data, 0, total_v_bytes);
        kv.value_store.device = DeviceType::CUDA;
    }

    cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream());

    const float* k_src = static_cast<const float*>(new_key->data());
    const float* v_src = static_cast<const float*>(new_value->data());

    std::vector<float> h_key, h_value;
    const float* d_k = k_src;
    const float* d_v = v_src;
    float* d_temp = nullptr;

    if (new_key->device() == DeviceType::CPU) {
        h_key.resize(seq_len * kv_dim);
        h_value.resize(seq_len * kv_dim);
        std::memcpy(h_key.data(), k_src, seq_len * kv_dim * sizeof(float));
        std::memcpy(h_value.data(), v_src, seq_len * kv_dim * sizeof(float));
        cudaMalloc(&d_temp, seq_len * kv_dim * sizeof(float) * 2);
        cudaMemcpyAsync(d_temp, h_key.data(), seq_len * kv_dim * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_temp + seq_len * kv_dim, h_value.data(), seq_len * kv_dim * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
        d_k = d_temp;
        d_v = d_temp + seq_len * kv_dim;
    }

    // Quantize K
    if (kv_config_.type_k != KVCacheDType::FP32) {
        uint8_t* q_dst = static_cast<uint8_t*>(kv.key_store.d_data) + filled * k_row_size;
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
        uint8_t* q_dst = static_cast<uint8_t*>(kv.value_store.d_data) + filled * v_row_size;
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

    if (!kv.key_store.d_q_data() || filled == 0)
        return;

    // Ensure FP32 tensors exist for output
    if (!kv.key_store.tensor) {
        auto shape = std::vector<int64_t>{max_seq_len_, kv_dim};
        kv.key_store.tensor = std::make_shared<Tensor>(DataType::FP32, shape, DeviceType::CUDA);
    }
    if (!kv.value_store.tensor) {
        auto shape = std::vector<int64_t>{max_seq_len_, kv_dim};
        kv.value_store.tensor = std::make_shared<Tensor>(DataType::FP32, shape, DeviceType::CUDA);
    }

    float* k_out = static_cast<float*>(kv.key_store.tensor->data());
    float* v_out = static_cast<float*>(kv.value_store.tensor->data());

    // Dequantize K
    if (kv_config_.type_k != KVCacheDType::FP32) {
        const void* q_src = kv.key_store.d_q_data();
        switch (kv_config_.type_k) {
        case KVCacheDType::F16:
            cuda::launch_dequant_f16_matrix(q_src, k_out, filled, kv_dim);
            break;
        case KVCacheDType::Q8_0:
            cuda::launch_dequant_q8_0_matrix(q_src, k_out, filled, kv_dim);
            break;
        case KVCacheDType::Q4_0:
            cuda::launch_dequant_q4_0_matrix(q_src, k_out, filled, kv_dim);
            break;
        case KVCacheDType::Q4_K:
            cuda::launch_dequant_q4_k_matrix(q_src, k_out, filled, kv_dim);
            break;
        default:
            break;
        }
    }

    // Dequantize V
    if (kv_config_.type_v != KVCacheDType::FP32) {
        const void* q_src = kv.value_store.d_q_data();
        switch (kv_config_.type_v) {
        case KVCacheDType::F16:
            cuda::launch_dequant_f16_matrix(q_src, v_out, filled, kv_dim);
            break;
        case KVCacheDType::Q8_0:
            cuda::launch_dequant_q8_0_matrix(q_src, v_out, filled, kv_dim);
            break;
        case KVCacheDType::Q4_0:
            cuda::launch_dequant_q4_0_matrix(q_src, v_out, filled, kv_dim);
            break;
        case KVCacheDType::Q4_K:
            cuda::launch_dequant_q4_k_matrix(q_src, v_out, filled, kv_dim);
            break;
        default:
            break;
        }
    }
#endif
}

// =========================================================================
// Sequence operations
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
