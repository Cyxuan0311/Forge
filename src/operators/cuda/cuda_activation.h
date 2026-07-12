#pragma once

struct CUstream_st;
typedef CUstream_st* cudaStream_t;

namespace forge {
namespace cuda {

void launch_silu(const float* x, float* out, int n, cudaStream_t stream = 0);
void launch_silu_fp16(const void* x, void* out, int n, cudaStream_t stream = 0);
void launch_gelu(const float* x, float* out, int n, cudaStream_t stream = 0);
void launch_gelu_tanh(const float* x, float* out, int n, cudaStream_t stream = 0);
void launch_tanh(const float* x, float* out, int n, cudaStream_t stream = 0);

} // namespace cuda
} // namespace forge
