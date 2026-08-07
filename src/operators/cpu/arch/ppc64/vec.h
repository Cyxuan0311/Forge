#pragma once
// PowerPC64 VSX vector primitives.
// Provides hsum, fp16 conversion, and broadcast utilities for VSX.
// VSX registers are 128-bit, matching ARM NEON register width.

#ifdef USE_VSX
#include <altivec.h>
#endif
#include <cstdint>
#include <cstring>
#include <cmath>

namespace forge {
namespace cpu {

#ifdef USE_VSX

// ---- Horizontal sum of __vector float ----
// VSX does not have a single-instruction horizontal sum like ARM vaddvq_f32.
// Use union-based element extraction.
static inline float hsum_f32x4(__vector float v) {
    union { __vector float vf; float f[4]; } u;
    u.vf = v;
    return u.f[0] + u.f[1] + u.f[2] + u.f[3];
}

// ---- fp16 to fp32 scalar (software, identical to arm64/x86) ----
static inline float fp16_to_fp32_scalar(uint16_t bits) {
    uint32_t sign     = (bits >> 15) & 1;
    uint32_t exponent = (bits >> 10) & 0x1F;
    uint32_t mantissa = bits & 0x3FF;
    float value;
    if (exponent == 0) {
        value = std::ldexp(static_cast<float>(mantissa) / 1024.0f, -14);
    } else {
        value = std::ldexp(1.0f + static_cast<float>(mantissa) / 1024.0f,
                           static_cast<int>(exponent) - 15);
    }
    return sign ? -value : value;
}

// ---- Broadcast fp16 to 4-way __vector float ----
// Scalar-based: load fp16 value, expand to 4 identical floats via vec_splats.
static inline __vector float fp16_to_fp32_broadcast_vsx(uint16_t fp16_val) {
    float f = fp16_to_fp32_scalar(fp16_val);
    return vec_splats(f);
}

#endif // USE_VSX

}  // namespace cpu
}  // namespace forge
