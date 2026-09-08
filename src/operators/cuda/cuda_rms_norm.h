#pragma once

#include <cuda_bf16.h>
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

// Phase 6-B-2: real native BF16 RMSNorm (weighted + unweighted). BF16
// storage/arithmetic with an FP32 reduction -- mirrors the 6-A-2 FP16 policy
// (norm_compute stays FP32 while activation is BF16). Routed from
// ops::rms_norm / rms_norm_kernel_dst; other dtypes keep loud rejection.
void launch_rms_norm_bf16(const __nv_bfloat16* x, const __nv_bfloat16* weight, __nv_bfloat16* out,
                           int rows, int cols, float eps, cudaStream_t stream = 0);
void launch_rms_norm_unweighted_bf16(const __nv_bfloat16* x, __nv_bfloat16* out, int rows, int cols,
                                     float eps, cudaStream_t stream = 0);

// Phase 6-C-2: real native FP8 E4M3 RMSNorm (weighted + unweighted). FP8
// storage with an FP32 reduction (forge E4M3 codec via forge/fp8_utils.h).
// Routed from ops::rms_norm / rms_norm_kernel_dst (FP8 x + FP8 weight only).
void launch_rms_norm_fp8(const uint8_t* x, const uint8_t* weight, uint8_t* out, int rows, int cols,
                         float eps, cudaStream_t stream = 0);
void launch_rms_norm_unweighted_fp8(const uint8_t* x, uint8_t* out, int rows, int cols, float eps,
                                    cudaStream_t stream = 0);

}  // namespace cuda
}  // namespace forge
