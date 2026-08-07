#pragma once
// ARM64 NEON kernels aggregate header (FORGE_ARCH_ARM64 + USE_NEON).
// Common CPU detection: always false on ARM64 (no AVX-512).
// Includes all ARM64 NEON-accelerated kernel sub-headers.
//
// Apple Silicon (M1/M2/M3/M4): NEON + dotprod + i8mm are always available.
//   Compiled with -mcpu=apple-m1 (min base for all M-series).
// Mobile/server ARM64: NEON always, dotprod/i8mm optional.
//   Compiled with -march=armv8.2-a+dotprod (or -mcpu=native).
//
// Kernel sub-headers use #ifdef USE_NEON to guard NEON intrinsics.
// #ifdef USE_DOTPROD guards vdotq_s32 (ARMv8.2+dotprod, Apple M1+).
// #ifdef USE_MATMUL_INT8 guards vmmlaq_s32 (ARMv8.6+i8mm, Apple M1+).
//
// See cpu_arch_split_plan.md §3.3 — only one arch kernels.h is compiled per TU,
// selected by simd.h based on compile-time FORGE_ARCH_ARM64 + USE_NEON.

#ifdef USE_NEON
#    include <arm_neon.h>
#endif

#include "elementwise_kernels.h"
#include "norm_kernels.h"
#include "attn_kernels.h"
#include "sampling_kernels.h"
#include "kv_kernels.h"

// Quantized dot product and GEMV kernels
#include "vec.h"
#include "vec_dot.h"
#include "gemv.h"
#include "gemm.h"
#include "quants.h"

// Future: more quantized kernels
// #include "gemm_microkernel.h"
// #include "fused.h"
// #include "cpu_gemv.h"

namespace forge {
namespace cpu {

// Common cpu info stub (ARM64 has no AVX-512)
inline bool cached_has_avx512_vnni() { return false; }

}  // namespace cpu
}  // namespace forge
