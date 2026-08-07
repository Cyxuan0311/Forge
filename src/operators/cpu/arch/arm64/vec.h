#pragma once
// ARM64 NEON vector primitives.
// Provides hsum, fp16 conversion, dot product, and nibble expansion
// for quantized weight dot products. NEON registers are 128-bit (half AVX2),
// so loop counts are doubled compared to x86 counterparts.

#ifdef USE_NEON
#include <arm_neon.h>
#endif
#include <cstring>
#include <cmath>

namespace forge {
namespace cpu {

#ifdef USE_NEON

// ---- Horizontal sum of float32x4_t ----
static inline float hsum_f32x4(float32x4_t v) {
    return vaddvq_f32(v);
}

// ---- fp16 to fp32 scalar (software, identical to x86) ----
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

// ---- Broadcast fp16 to 4-way float32x4_t ----
// Scalar-based: load fp16 value, expand to 4 identical floats.
// ARMv8.2+ has F16C via __fp16 type and vcvt_f32_f16, but we use
// scalar fallback for portability across compilers.
static inline float32x4_t fp16_to_fp32_broadcast_neon(uint16_t fp16_val) {
    float f = fp16_to_fp32_scalar(fp16_val);
    return vdupq_n_f32(f);
}

#endif // USE_NEON

}  // namespace cpu
}  // namespace forge
