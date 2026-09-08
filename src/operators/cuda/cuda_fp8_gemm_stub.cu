// Stub used when CUTLASS is unavailable or the build targets no Ada arch
// (sm_89+). Keeps the production launcher symbols resolvable so forge_ops
// always links; the real implementation lives in cuda_fp8_gemm.cu.

#include "forge/fp8_gemm.h"

namespace forge {
namespace cuda {

bool launch_fp8_gemm_e4m3_fp32(const uint8_t*, const uint8_t*, float*, int, int, int,
                               const Fp8W8A8Scales&, cudaStream_t) {
    return false;
}

bool launch_fp8_gemm_e4m3_fp8out(const uint8_t*, const uint8_t*, uint8_t*, int, int, int,
                                 const Fp8W8A8Scales&, cudaStream_t) {
    return false;
}

}  // namespace cuda
}  // namespace forge
