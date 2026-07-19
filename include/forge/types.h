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
    IQ2_S = 14,   // 82 bytes/block, 256 elements/block
    BF16 = 15,    // bfloat16 (converted to FP32 at load time)
};

enum class DeviceType : uint32_t {
    CPU = 0,
    CUDA = 1,
};

}  // namespace forge

// Include quant_traits.h after DataType enum is defined.
// Provides QuantTraits<DT>, DataTypeTraits table, and accessor functions
// (dtype_size, dtype_name, dtype_block_size, dtype_block_elements,
//  is_quantized_type, compute_quantized_bytes, get_dequant_row_fn).
#include "forge/quant_traits.h"
