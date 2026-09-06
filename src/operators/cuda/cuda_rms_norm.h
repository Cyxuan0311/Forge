#pragma once

#include <cuda_fp16.h>

struct CUstream_st;
typedef CUstream_st* cudaStream_t;

namespace forge {
namespace cuda {

void launch_rms_norm(const float* x, const float* weight, float* out, int rows, int cols, float eps,
                     cudaStream_t stream = 0);

// Phase 6-A-2: real native FP16 RMSNorm. FP16 storage/arithmetic with an FP32
// reduction (square-sum -> rsqrt -> scale) so numerical stability is preserved --
// this is a native FP16 *path*, not "all intermediates in half". Gated at the
// operator boundary by forge/dtype_contract.h (op_supports_lowp / RMS_NORM).
void launch_rms_norm_fp16(const __half* x, const __half* weight, __half* out, int rows, int cols,
                          float eps, cudaStream_t stream = 0);

}  // namespace cuda
}  // namespace forge
