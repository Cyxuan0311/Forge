#pragma once
// Compile-time architecture switch (see cpu_arch_split_plan.md §3.3).
// Guarantees a single TU includes exactly one arch kernels.h, so same-named
// static kernels across arch variants never collide.
//
// Selection logic:
//   FORGE_ARCH_X86    + USE_AVX2          -> arch/x86/kernels.h
//   FORGE_ARCH_ARM64  + USE_NEON          -> arch/arm64/kernels.h
//   FORGE_ARCH_PPC64  + USE_VSX           -> arch/ppc64/kernels.h
//   otherwise                              -> arch/generic/kernels.h

#if defined(FORGE_ARCH_X86) && defined(USE_AVX2)
#    include "arch/x86/kernels.h"
#elif defined(FORGE_ARCH_ARM64) && defined(USE_NEON)
#    include "arch/arm64/kernels.h"
#elif defined(FORGE_ARCH_PPC64) && defined(USE_VSX)
#    include "arch/ppc64/kernels.h"
#else
#    include "arch/generic/kernels.h"
#endif
