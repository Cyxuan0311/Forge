#pragma once

struct CUstream_st;
typedef CUstream_st* cudaStream_t;

namespace forge {
namespace cuda {

void launch_rms_norm(const float* x, const float* weight, float* out,
                     int rows, int cols, float eps, cudaStream_t stream = 0);

void launch_rms_norm_fp16(const void* x, const void* weight, void* out,
                          int rows, int cols, float eps, cudaStream_t stream = 0);

} // namespace cuda
} // namespace forge
