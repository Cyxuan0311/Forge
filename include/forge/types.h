#pragma once

#include <cstdint>

namespace forge {

enum class DataType : uint32_t {
    FP32 = 0,
    FP16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q4_K = 4,
    INT8 = 5,
    INT32 = 6,
    Q8_0 = 7,
    Q5_0 = 8,
    Q5_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q5_K = 12,
    Q6_K = 13,
    IQ2_S = 14,    // 82 bytes/block, 256 elements/block
    IQ2_XXS = 16,  // 66 bytes/block, 256 elements/block (2.0625 bpw)
    IQ4_NL = 17,   // 18 bytes/block, 32 elements/block (non-linear 4-bit)
    BF16 = 15,     // bfloat16 (converted to FP32 at load time)
    IQ2_XS = 18,   // 74 bytes/block, 256 elements/block (2.3125 bpw)
    IQ3_S = 19,    // 110 bytes/block, 256 elements/block (3.4375 bpw)
    IQ4_XS = 20,   // 136 bytes/block, 256 elements/block (non-linear 4-bit)
    IQ1_S = 22,    // 50 bytes/block, 256 elements/block (1.5625 bpw)
    IQ3_XXS = 23,  // 98 bytes/block, 256 elements/block (3.0625 bpw)
    IQ1_M = 24,    // 56 bytes/block, 256 elements/block (1.75 bpw)
    FP8_E4M3 = 21, // e4m3, 1 byte/element (Forge codec: bias 7, max finite 240,
                   // no NaN). Phase 6-C W8A8 storage dtype. KV cache uses the
                   // separate KVCacheDType::FP8_E4M3 -- two independent systems.
};

enum class DeviceType : uint32_t {
    CPU = 0,
    CUDA = 1,
};

// DeviceTarget extends DeviceType with a GPU device index (0-based).
// For CPU devices, device_id is ignored.
// Used for multi-GPU layer splitting where different layers reside on
// different GPUs. Backward-compatible: DeviceTarget::cuda(0) == legacy CUDA.
struct DeviceTarget {
    DeviceType type = DeviceType::CPU;
    int device_id = 0;

    bool operator==(const DeviceTarget& o) const {
        return type == o.type && (type == DeviceType::CPU || device_id == o.device_id);
    }
    bool operator!=(const DeviceTarget& o) const { return !(*this == o); }

    bool is_cuda() const { return type == DeviceType::CUDA; }
    bool is_cpu() const  { return type == DeviceType::CPU; }

    static DeviceTarget cpu() { return {DeviceType::CPU, 0}; }
    static DeviceTarget cuda(int id = 0) { return {DeviceType::CUDA, id}; }

    // Implicit conversion from DeviceType for backward compatibility
    DeviceTarget(DeviceType dev = DeviceType::CPU, int id = 0) : type(dev), device_id(id) {}
};

}  // namespace forge

// Include quant_traits.h after DataType enum is defined.
// Provides QuantTraits<DT>, DataTypeTraits table, and accessor functions
// (dtype_size, dtype_name, dtype_block_size, dtype_block_elements,
//  is_quantized_type, compute_quantized_bytes, get_dequant_row_fn).
#include "forge/quant_traits.h"
