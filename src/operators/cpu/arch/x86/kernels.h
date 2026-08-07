#pragma once
// x86 SIMD kernels aggregate header (FORGE_ARCH_X86).
// Includes all x86-specific GEMV/GEMM/dot kernels and primitives.
// See cpu_arch_split_plan.md — only one arch kernels.h is compiled per TU,
// selected by simd.h based on compile-time architecture macros.

#ifdef USE_AVX2
#    include <immintrin.h>
#endif

#include "vec.h"
#include "vec_dot.h"
#include "scales.h"
#include "gemv.h"
#include "gemm.h"
#include "gemm_microkernel.h"
#include "fused.h"
#include "cpu_gemv.h"
#include "repack.h"
#include "quants.h"
#include "cpuinfo.h"
#include "elementwise_kernels.h"
#include "norm_kernels.h"
#include "attn_kernels.h"
#include "sampling_kernels.h"
#include "kv_kernels.h"