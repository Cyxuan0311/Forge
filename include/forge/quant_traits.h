#pragma once

// Type traits system for Forge's DataType enum.
// Two-layer design (inspired by llama.cpp's ggml_type_traits):
//   1. QuantTraits<DT> — compile-time template for CUDA kernels
//   2. DataTypeTraits[] — runtime table indexed by DataType, eliminates switch statements
//
// This header is included by forge/types.h after the DataType enum is defined.
// Do NOT include this header directly unless types.h has already been included.

#include <cstdint>
#include <string>

namespace forge {

// ---- Compile-time traits (for CUDA templates) ----
template <DataType DT> struct QuantTraits;

template <> struct QuantTraits<DataType::Q4_0> {
    static constexpr int block_elements = 32;
    static constexpr int block_size    = 18;
    static constexpr const char* name  = "q4_0";
};

template <> struct QuantTraits<DataType::Q4_1> {
    static constexpr int block_elements = 32;
    static constexpr int block_size    = 20;
    static constexpr const char* name  = "q4_1";
};

template <> struct QuantTraits<DataType::Q8_0> {
    static constexpr int block_elements = 32;
    static constexpr int block_size    = 34;
    static constexpr const char* name  = "q8_0";
};

template <> struct QuantTraits<DataType::Q5_0> {
    static constexpr int block_elements = 32;
    static constexpr int block_size    = 22;
    static constexpr const char* name  = "q5_0";
};

template <> struct QuantTraits<DataType::Q5_1> {
    static constexpr int block_elements = 32;
    static constexpr int block_size    = 24;
    static constexpr const char* name  = "q5_1";
};

template <> struct QuantTraits<DataType::Q2_K> {
    static constexpr int block_elements = 256;
    static constexpr int block_size    = 84;
    static constexpr const char* name  = "q2_k";
};

template <> struct QuantTraits<DataType::Q3_K> {
    static constexpr int block_elements = 256;
    static constexpr int block_size    = 110;
    static constexpr const char* name  = "q3_k";
};

template <> struct QuantTraits<DataType::Q4_K> {
    static constexpr int block_elements = 256;
    static constexpr int block_size    = 144;
    static constexpr const char* name  = "q4_k";
};

template <> struct QuantTraits<DataType::Q5_K> {
    static constexpr int block_elements = 256;
    static constexpr int block_size    = 176;
    static constexpr const char* name  = "q5_k";
};

template <> struct QuantTraits<DataType::Q6_K> {
    static constexpr int block_elements = 256;
    static constexpr int block_size    = 210;
    static constexpr const char* name  = "q6_k";
};

template <> struct QuantTraits<DataType::IQ2_S> {
    static constexpr int block_elements = 256;
    static constexpr int block_size    = 82;
    static constexpr const char* name  = "iq2_s";
};

template <> struct QuantTraits<DataType::IQ2_XXS> {
    static constexpr int block_elements = 256;
    static constexpr int block_size    = 66;
    static constexpr const char* name  = "iq2_xxs";
};

template <> struct QuantTraits<DataType::IQ4_NL> {
    static constexpr int block_elements = 32;
    static constexpr int block_size    = 18;
    static constexpr const char* name  = "iq4_nl";
};

// ---- Runtime traits table ----

// Dequantize row function: dequantize one row of a quantized weight matrix.
// Signature: (quant_data, output, K, row_index)
using DequantRowFn = void (*)(const uint8_t*, float*, int, int);

struct DataTypeTraits {
    const char* name;
    int64_t type_size;       // element size in bytes (0 for quantized types)
    int64_t block_elements;  // elements per quantization block (1 for non-quantized)
    int64_t block_size;      // bytes per quantization block (0 for non-quantized)
    bool is_quantized;
    DequantRowFn dequant_row;  // nullptr if not available
};

// Global traits table indexed by DataType enum value.
// Defined in quant_traits.cpp alongside the dequant functions it references.
extern const DataTypeTraits data_type_traits[];

// ---- Backward-compatible accessors (replace switch statements) ----

inline size_t dtype_size(DataType dt) {
    return static_cast<size_t>(data_type_traits[static_cast<int>(dt)].type_size);
}

inline std::string dtype_name(DataType dt) {
    return data_type_traits[static_cast<int>(dt)].name;
}

inline int64_t dtype_block_size(DataType dt) {
    return data_type_traits[static_cast<int>(dt)].block_size;
}

inline int64_t dtype_block_elements(DataType dt) {
    return data_type_traits[static_cast<int>(dt)].block_elements;
}

inline bool is_quantized_type(DataType dt) {
    return data_type_traits[static_cast<int>(dt)].is_quantized;
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

// Convenience: get dequant_row function pointer for a DataType (nullptr if N/A)
inline DequantRowFn get_dequant_row_fn(DataType dt) {
    return data_type_traits[static_cast<int>(dt)].dequant_row;
}

}  // namespace forge
