#pragma once
// Scalar dequant row functions. Implemented in common/dequant.cpp.
// Referenced by forge::data_type_traits[].dequant_row (quant_traits.cpp) and
// by the CPU drivers as a scalar fallback. Names stay in forge::ops so the
// existing quant_traits.cpp extern declarations keep linking.

#include <cstdint>

namespace forge {
namespace ops {

void dequantize_q4_0_row(const uint8_t* q_data, float* out, int K, int row);
void dequantize_q4_1_row(const uint8_t* q_data, float* out, int K, int row);
void dequantize_q4_k_row(const uint8_t* q_data, float* out, int K, int row);
void dequantize_q8_0_row(const uint8_t* q_data, float* out, int K, int row);
void dequantize_q5_0_row(const uint8_t* q_data, float* out, int K, int row);
void dequantize_q5_1_row(const uint8_t* q_data, float* out, int K, int row);
void dequantize_q2_k_row(const uint8_t* q_data, float* out, int K, int row);
void dequantize_q3_k_row(const uint8_t* q_data, float* out, int K, int row);
void dequantize_q5_k_row(const uint8_t* q_data, float* out, int K, int row);
void dequantize_q6_k_row(const uint8_t* q_data, float* out, int K, int row);
void dequantize_iq2_s_row(const uint8_t* q_data, float* out, int K, int row);
void dequantize_iq2_xxs_row(const uint8_t* q_data, float* out, int K, int row);
void dequantize_iq4_nl_row(const uint8_t* q_data, float* out, int K, int row);
void dequantize_iq2_xs_row(const uint8_t* q_data, float* out, int K, int row);
void dequantize_iq3_s_row(const uint8_t* q_data, float* out, int K, int row);

}  // namespace ops
}  // namespace forge
