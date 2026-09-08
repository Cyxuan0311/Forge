// Stub mirror of cuda_fp8_gemv.cu for structural parity with the CUTLASS-backed
// GEMM split (cuda_fp8_gemm.cu / cuda_fp8_gemm_stub.cu).
//
// NOTE: unlike the FP8 GEMM path, the FP8 GEMV kernel is PURE CUDA (no CUTLASS
// dependency), so it always compiles for the sm_89+ target. Because of that the
// real implementation in cuda_fp8_gemv.cu is ALWAYS linked and this stub is NOT
// added to the build (linking both would be a duplicate-symbol error). It exists
// only as a reference/fallback template if a build ever needs to exclude the real
// GEMV TU entirely.

#include "forge/fp8_gemv.h"

namespace forge {
namespace cuda {

bool launch_fp8_gemv_e4m3_fp32(const uint8_t*, const uint8_t*, float*, int, int,
                                const Fp8W8A8Scales&, cudaStream_t) {
    return false;
}

}  // namespace cuda
}  // namespace forge
