#pragma once

#include <cstdint>

struct CUstream_st;
typedef CUstream_st* cudaStream_t;

namespace forge {
namespace cuda {

// ---- FP32 GEMV ----
void launch_gemv_transB(const float* x, const float* W, float* out, int K, int N,
                        cudaStream_t stream = 0);

void launch_gemv(const float* x, const float* W, float* out, int K, int N, cudaStream_t stream = 0);

// ---- Q4_0 special GEMV (smem + splitK + dual) ----
void launch_gemv_q4_0_transB(const float* x, const void* q_weight, float* out, int K, int N,
                             cudaStream_t stream = 0);

void launch_gemv_q4_0_transB_dual(const float* x, const void* q_weight1, int N1,
                                  const void* q_weight2, int N2, float* out, int K,
                                  cudaStream_t stream = 0);

// ---- Typed GEMV dispatch tables ----
// Indexed by DataType enum value. nullptr for unsupported types.
// Defined in cuda_gemv_instances.cu alongside explicit template instantiations.

using GemvFn = void (*)(const float*, const void*, float*, int, int, cudaStream_t);
using GemvBatchFn = void (*)(const float*, const void*, float*, int, int, int, cudaStream_t);

extern const GemvFn gemv_dispatch[18];
extern const GemvBatchFn gemv_batch_dispatch[18];

}  // namespace cuda
}  // namespace forge
