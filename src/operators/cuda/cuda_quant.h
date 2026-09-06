#pragma once

#include <cstdint>

#include "forge/types.h"

struct CUstream_st;
typedef CUstream_st* cudaStream_t;

namespace forge {
namespace cuda {

void launch_dequant_q4_0(const void* q_data, float* out, int n, cudaStream_t stream = 0);
void launch_dequant_q4_1(const void* q_data, float* out, int n, cudaStream_t stream = 0);
void launch_dequant_q4_k(const void* q_data, float* out, int n, cudaStream_t stream = 0);
void launch_dequant_q4_0_matrix(const void* q_data, float* out, int N, int K,
                                cudaStream_t stream = 0);
void launch_dequant_q4_1_matrix(const void* q_data, float* out, int N, int K,
                                cudaStream_t stream = 0);
void launch_dequant_q4_k_matrix(const void* q_data, float* out, int N, int K,
                                cudaStream_t stream = 0);
void launch_dequant_q6_k_matrix(const void* q_data, float* out, int N, int K,
                                cudaStream_t stream = 0);
void launch_quantize_q4_0(const float* data, void* q_data, int n, cudaStream_t stream = 0);
void launch_quantize_q4_0_matrix(const float* data, void* q_data, int num_rows, int row_len,
                                 cudaStream_t stream = 0);
void launch_dequant_q4_0_matrix_to_rows(const void* q_data, float* out, int num_rows, int row_len,
                                        int stride, int start_row, cudaStream_t stream = 0);
void launch_dequant_q4_0_kv(const void* q_key, const void* q_value, float* key_out,
                            float* value_out, int seq_len, int kv_dim, int filled,
                            cudaStream_t stream = 0);
void launch_argmax(const float* data, int32_t* out_idx, int n, cudaStream_t stream = 0);
void launch_cublas_sgemm(const float* A, const float* B, float* C, int M, int K, int N, bool transB,
                         cudaStream_t stream = 0);

void launch_cublas_gemm_fp16_fp32(const float* A, const void* B, float* C, int M, int K, int N,
                                   bool transB, cudaStream_t stream = 0);

void launch_dequant_q4_0_matrix_fp16(const void* q_data, void* out, int N, int K,
                                      cudaStream_t stream = 0);

void launch_dequant_q4_k_matrix_fp16(const void* q_data, void* out, int N, int K,
                                      cudaStream_t stream = 0);

void launch_dequant_q6_k_matrix_fp16(const void* q_data, void* out, int N, int K,
                                      cudaStream_t stream = 0);
void launch_dequant_q8_0_matrix_fp16(const void* q_data, void* out, int N, int K,
                                     cudaStream_t stream = 0);

// Phase 3 (Tensor Core): dequantize the whole quantized B [N, K] matrix to FP16
// and run the FP16 cuBLAS GEMM (FP32 accumulate) on Tensor Cores. Supported for
// every dtype with an FP16 matrix dequant (Q4_0/Q4_K/Q6_K/Q8_0). Returns false
// when dt has no FP16 dequant path or when built without FORGE_USE_CUBLAS (in
// which case nothing is launched), so callers can fall back to their SIMT path.
bool launch_cublas_gemm_quant_fp16(DataType dt, const float* A, const void* B_q, float* C, int M,
                                   int K, int N, bool transB, cudaStream_t stream);

}  // namespace cuda
}  // namespace forge
