#pragma once

#include <cstdint>
#include <string>

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
};

enum class DeviceType : uint32_t {
    CPU = 0,
    CUDA = 1,
};

inline size_t dtype_size(DataType dt) {
    switch (dt) {
    case DataType::FP32:
        return 4;
    case DataType::FP16:
        return 2;
    case DataType::INT8:
        return 1;
    case DataType::INT32:
        return 4;
    default:
        return 0;
    }
}

inline std::string dtype_name(DataType dt) {
    switch (dt) {
    case DataType::FP32:
        return "fp32";
    case DataType::FP16:
        return "fp16";
    case DataType::Q4_0:
        return "q4_0";
    case DataType::Q4_1:
        return "q4_1";
    case DataType::Q4_K:
        return "q4_k";
    case DataType::INT8:
        return "int8";
    case DataType::INT32:
        return "int32";
    case DataType::Q8_0:
        return "q8_0";
    case DataType::Q5_0:
        return "q5_0";
    case DataType::Q5_1:
        return "q5_1";
    case DataType::Q2_K:
        return "q2_k";
    case DataType::Q3_K:
        return "q3_k";
    case DataType::Q5_K:
        return "q5_k";
    case DataType::Q6_K:
        return "q6_k";
    default:
        return "unknown";
    }
}

inline bool is_quantized_type(DataType dt) {
    switch (dt) {
    case DataType::Q4_0:
    case DataType::Q4_1:
    case DataType::Q4_K:
    case DataType::Q8_0:
    case DataType::Q5_0:
    case DataType::Q5_1:
    case DataType::Q2_K:
    case DataType::Q3_K:
    case DataType::Q5_K:
    case DataType::Q6_K:
        return true;
    default:
        return false;
    }
}

inline int64_t dtype_block_size(DataType dt) {
    switch (dt) {
    case DataType::Q4_0:
        return 18;
    case DataType::Q4_1:
        return 20;
    case DataType::Q8_0:
        return 34;
    case DataType::Q5_0:
        return 22;
    case DataType::Q5_1:
        return 24;
    case DataType::Q2_K:
        return 64;
    case DataType::Q3_K:
        return 110;
    case DataType::Q4_K:
        return 144;
    case DataType::Q5_K:
        return 176;
    case DataType::Q6_K:
        return 210;
    default:
        return 0;
    }
}

inline int64_t dtype_block_elements(DataType dt) {
    switch (dt) {
    case DataType::Q4_0:
    case DataType::Q4_1:
    case DataType::Q8_0:
    case DataType::Q5_0:
    case DataType::Q5_1:
        return 32;
    case DataType::Q2_K:
    case DataType::Q3_K:
    case DataType::Q4_K:
    case DataType::Q5_K:
    case DataType::Q6_K:
        return 256;
    default:
        return 1;
    }
}

inline size_t compute_quantized_bytes(int64_t numel, DataType dt) {
    if (!is_quantized_type(dt)) {
        return numel * dtype_size(dt);
    }
    int64_t block_el = dtype_block_elements(dt);
    int64_t block_sz = dtype_block_size(dt);
    int64_t n_blocks = (numel + block_el - 1) / block_el;
    return n_blocks * block_sz;
}

}  // namespace forge
