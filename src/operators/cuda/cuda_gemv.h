#pragma once

#include <cstdint>

struct CUstream_st;
typedef CUstream_st* cudaStream_t;

namespace forge {
namespace cuda {

void launch_gemv_transB(const float* x, const float* W, float* out,
                        int K, int N, cudaStream_t stream = 0);

void launch_gemv(const float* x, const float* W, float* out,
                 int K, int N, cudaStream_t stream = 0);

void launch_gemv_q4_0_transB(const float* x, const void* q_weight, float* out,
                             int K, int N, cudaStream_t stream = 0);

void launch_gemv_q4_0_transB_batch(const float* x, const void* q_weight, float* out,
                                   int M, int K, int N, cudaStream_t stream = 0);

void launch_gemv_q4_0_transB_dual(const float* x,
                                  const void* q_weight1, int N1,
                                  const void* q_weight2, int N2,
                                  float* out, int K,
                                  cudaStream_t stream = 0);

void launch_gemv_q4_k_transB(const float* x, const void* q_weight, float* out,
                             int K, int N, cudaStream_t stream = 0);

void launch_gemv_q4_k_transB_batch(const float* x, const void* q_weight, float* out,
                                   int M, int K, int N, cudaStream_t stream = 0);

void launch_gemv_q4_1_transB(const float* x, const void* q_weight, float* out,
                             int K, int N, cudaStream_t stream = 0);

void launch_gemv_q4_1_transB_batch(const float* x, const void* q_weight, float* out,
                                   int M, int K, int N, cudaStream_t stream = 0);

void launch_gemv_q6_k_transB(const float* x, const void* q_weight, float* out,
                             int K, int N, cudaStream_t stream = 0);

void launch_gemv_q6_k_transB_batch(const float* x, const void* q_weight, float* out,
                                   int M, int K, int N, cudaStream_t stream = 0);

} // namespace cuda
} // namespace forge
