#pragma once

struct CUstream_st;
typedef CUstream_st* cudaStream_t;

namespace forge {
namespace cuda {

void launch_silu(const float* x, float* out, int n, cudaStream_t stream = 0);
// Phase 6-A-1: `launch_silu_fp16` removed (fake FP16 entry point, no callers);
// see cuda_activation.cu.
void launch_gelu(const float* x, float* out, int n, cudaStream_t stream = 0);
void launch_gelu_tanh(const float* x, float* out, int n, cudaStream_t stream = 0);
void launch_tanh(const float* x, float* out, int n, cudaStream_t stream = 0);

}  // namespace cuda
}  // namespace forge
