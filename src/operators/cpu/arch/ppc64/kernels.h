#pragma once
// PPC64 VSX kernels aggregate header (FORGE_ARCH_PPC64 + USE_VSX).
// Common CPU detection: always false on PPC64 (no AVX-512).
// Includes all PPC64 VSX-accelerated kernel sub-headers.

#ifdef USE_VSX
#    include <altivec.h>
#endif

#include "elementwise_kernels.h"
#include "norm_kernels.h"
#include "attn_kernels.h"
#include "sampling_kernels.h"
#include "kv_kernels.h"

#include "vec.h"
#include "vec_dot.h"
#include "gemv.h"
#include "gemm.h"
#include "quants.h"

namespace forge {
namespace cpu {
inline bool cached_has_avx512_vnni() { return false; }
}  // namespace cpu
}  // namespace forge
