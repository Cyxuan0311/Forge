#pragma once
// Architecture-independent scalar primitives shared by all arch variants.
// Hosts the fp16<->fp32 conversion helpers and scalar dot product fallbacks.

#include <cmath>
#include <cstdint>

namespace forge {
namespace cpu {

static inline float fp16_to_float_scalar(uint16_t bits) {
    uint32_t sign = (bits >> 15) & 1;
    uint32_t exponent = (bits >> 10) & 0x1F;
    uint32_t mantissa = bits & 0x3FF;
    if (exponent == 0) {
        if (mantissa == 0)
            return 0.0f;
        float v = std::ldexp(static_cast<float>(mantissa) / 1024.0f, -14);
        return sign ? -v : v;
    }
    float v =
        std::ldexp(1.0f + static_cast<float>(mantissa) / 1024.0f, static_cast<int>(exponent) - 15);
    return sign ? -v : v;
}

}  // namespace cpu
}  // namespace forge
