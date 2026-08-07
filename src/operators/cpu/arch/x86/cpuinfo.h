#pragma once
// Runtime CPU feature detection for x86.
// Extracted from matmul.cpp — used by AVX-512 VNNI dispatch at runtime.
// The cpuid result is cached in a function-local static for one-shot cost.

#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64)

#if defined(_MSC_VER)
#    include <intrin.h>
#else
#    include <cpuid.h>
#endif

namespace forge {
namespace cpu {

inline bool cpu_has_avx512_vnni() {
    uint32_t eax, ebx, ecx, edx;
#if defined(_MSC_VER)
    int regs[4];
    __cpuidex(regs, 7, 0);
    eax = static_cast<uint32_t>(regs[0]);
    ebx = static_cast<uint32_t>(regs[1]);
    ecx = static_cast<uint32_t>(regs[2]);
    edx = static_cast<uint32_t>(regs[3]);
#else
    // CPUID leaf 7, sub-leaf 0
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) == 0)
        return false;
#endif
    // AVX-512F:  EBX[16]
    if (!(ebx & (1u << 16)))
        return false;
    // AVX-512BW: EBX[30]
    if (!(ebx & (1u << 30)))
        return false;
    // AVX-512VNNI: ECX[11]
    if (!(ecx & (1u << 11)))
        return false;
    return true;
}

inline bool cached_has_avx512_vnni() {
    static bool result = cpu_has_avx512_vnni();
    return result;
}

}  // namespace cpu
}  // namespace forge

#else

namespace forge {
namespace cpu {

inline bool cached_has_avx512_vnni() { return false; }

}  // namespace cpu
}  // namespace forge

#endif