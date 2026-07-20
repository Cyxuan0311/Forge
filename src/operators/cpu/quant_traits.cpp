#include "forge/types.h"

// Forward declarations of dequant row functions defined in matmul.cpp.
// These are non-static functions in forge::ops namespace.
namespace forge {
namespace ops {
extern void dequantize_q4_0_row(const uint8_t*, float*, int, int);
extern void dequantize_q4_1_row(const uint8_t*, float*, int, int);
extern void dequantize_q4_k_row(const uint8_t*, float*, int, int);
extern void dequantize_q8_0_row(const uint8_t*, float*, int, int);
extern void dequantize_q5_0_row(const uint8_t*, float*, int, int);
extern void dequantize_q5_1_row(const uint8_t*, float*, int, int);
extern void dequantize_q2_k_row(const uint8_t*, float*, int, int);
extern void dequantize_q3_k_row(const uint8_t*, float*, int, int);
extern void dequantize_q5_k_row(const uint8_t*, float*, int, int);
extern void dequantize_q6_k_row(const uint8_t*, float*, int, int);
extern void dequantize_iq2_s_row(const uint8_t*, float*, int, int);
extern void dequantize_iq2_xxs_row(const uint8_t*, float*, int, int);
extern void dequantize_iq4_nl_row(const uint8_t*, float*, int, int);
}  // namespace ops
}  // namespace forge

namespace forge {

// Runtime type traits table — one entry per DataType enum value.
// Mirrors llama.cpp's ggml_type_traits[] designator-initialization style.
// Adding a new quant type requires only:
//   1. Add a QuantTraits<> specialization in quant_traits.h
//   2. Add one row to this table
//   3. Implement the dequant kernel
// MSVC does not support C99 designated initializers, so use positional initialization.
// Order must match the DataType enum values (0..17) exactly.
const DataTypeTraits data_type_traits[] = {
    /*  0 FP32     */ { "fp32",     4, 1,   0,   false, nullptr },
    /*  1 FP16     */ { "fp16",     2, 1,   0,   false, nullptr },
    /*  2 Q4_0     */ { "q4_0",     0, 32,  18,  true,  ops::dequantize_q4_0_row },
    /*  3 Q4_1     */ { "q4_1",     0, 32,  20,  true,  ops::dequantize_q4_1_row },
    /*  4 Q4_K     */ { "q4_k",     0, 256, 144, true,  ops::dequantize_q4_k_row },
    /*  5 INT8     */ { "int8",     1, 1,   0,   false, nullptr },
    /*  6 INT32    */ { "int32",    4, 1,   0,   false, nullptr },
    /*  7 Q8_0     */ { "q8_0",     0, 32,  34,  true,  ops::dequantize_q8_0_row },
    /*  8 Q5_0     */ { "q5_0",     0, 32,  22,  true,  ops::dequantize_q5_0_row },
    /*  9 Q5_1     */ { "q5_1",     0, 32,  24,  true,  ops::dequantize_q5_1_row },
    /* 10 Q2_K     */ { "q2_k",     0, 256, 84,  true,  ops::dequantize_q2_k_row },
    /* 11 Q3_K     */ { "q3_k",     0, 256, 110, true,  ops::dequantize_q3_k_row },
    /* 12 Q5_K     */ { "q5_k",     0, 256, 176, true,  ops::dequantize_q5_k_row },
    /* 13 Q6_K     */ { "q6_k",     0, 256, 210, true,  ops::dequantize_q6_k_row },
    /* 14 IQ2_S    */ { "iq2_s",    0, 256, 82,  true,  ops::dequantize_iq2_s_row },
    /* 15 BF16     */ { "bf16",     2, 1,   0,   false, nullptr },
    /* 16 IQ2_XXS  */ { "iq2_xxs",  0, 256, 66,  true,  ops::dequantize_iq2_xxs_row },
    /* 17 IQ4_NL   */ { "iq4_nl",   0, 32,  18,  true,  ops::dequantize_iq4_nl_row },
};

}  // namespace forge
