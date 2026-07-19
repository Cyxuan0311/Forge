#include "forge/types.h"

// Forward declarations of dequant row functions defined in matmul.cpp.
// These are non-static functions in forge::ops namespace.
namespace forge {
namespace ops {
extern void dequantize_q4_0_row(const uint8_t*, float*, int, int);
extern void dequantize_q4_1_row(const uint8_t*, float*, int, int);
extern void dequantize_q4_k_row(const uint8_t*, float*, int, int);
extern void dequantize_q8_0_row(const uint8_t*, float*, int, int);
extern void dequantize_q3_k_row(const uint8_t*, float*, int, int);
extern void dequantize_q5_k_row(const uint8_t*, float*, int, int);
extern void dequantize_q6_k_row(const uint8_t*, float*, int, int);
extern void dequantize_iq2_s_row(const uint8_t*, float*, int, int);
}  // namespace ops
}  // namespace forge

namespace forge {

// Runtime type traits table — one entry per DataType enum value.
// Mirrors llama.cpp's ggml_type_traits[] designator-initialization style.
// Adding a new quant type requires only:
//   1. Add a QuantTraits<> specialization in quant_traits.h
//   2. Add one row to this table
//   3. Implement the dequant kernel
const DataTypeTraits data_type_traits[] = {
    // [enum_value] = { name, type_size, block_elements, block_size, is_quantized, dequant_row }
    [0]  = { "fp32",   4, 1,   0,   false, nullptr },
    [1]  = { "fp16",   2, 1,   0,   false, nullptr },
    [2]  = { "q4_0",   0, 32,  18,  true,  ops::dequantize_q4_0_row },
    [3]  = { "q4_1",   0, 32,  20,  true,  ops::dequantize_q4_1_row },
    [4]  = { "q4_k",   0, 256, 144, true,  ops::dequantize_q4_k_row },
    [5]  = { "int8",   1, 1,   0,   false, nullptr },
    [6]  = { "int32",  4, 1,   0,   false, nullptr },
    [7]  = { "q8_0",   0, 32,  34,  true,  ops::dequantize_q8_0_row },
    [8]  = { "q5_0",   0, 32,  22,  true,  nullptr },
    [9]  = { "q5_1",   0, 32,  24,  true,  nullptr },
    [10] = { "q2_k",   0, 256, 84,  true,  nullptr },
    [11] = { "q3_k",   0, 256, 110, true,  ops::dequantize_q3_k_row },
    [12] = { "q5_k",   0, 256, 176, true,  ops::dequantize_q5_k_row },
    [13] = { "q6_k",   0, 256, 210, true,  ops::dequantize_q6_k_row },
    [14] = { "iq2_s",  0, 256, 82,  true,  ops::dequantize_iq2_s_row },
    [15] = { "bf16",   2, 1,   0,   false, nullptr },
};

}  // namespace forge
