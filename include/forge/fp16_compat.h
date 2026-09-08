#pragma once
// Host-side (non-CUDA) compatibility types for the two CUDA scalar storage
// types that Forge host code references by name: IEEE 754 binary16 (`__half`)
// and bfloat16 (`__nv_bfloat16`).
//
// This header is included ONLY on CPU-only builds -- i.e. when no CUDA toolkit
// is available and <cuda_fp16.h> / <cuda_bf16.h> cannot be included. On CUDA
// builds those toolkit headers are used instead (same names/semantics); the
// switch lives in forge/cuda_kernels.h.
//
// Design rules:
//   - Bit layout matches CUDA exactly: one little-endian uint16_t payload, so
//     forge FP16/BF16 buffers are interchangeable between CPU and GPU builds.
//   - FP32 <-> FP16/BF16 conversions round to nearest, ties to even (RN),
//     matching CUDA's default __float2half / __float2bfloat16.
//   - Only the scalar surface forge host code uses is implemented (storage +
//     FP32 conversions and CUDA's convenience conversion members).
//
// Kept deliberately small so CPU-only builds never need the CUDA toolkit
// headers (they previously failed with "cuda_bf16.h not found" on the
// macOS/Windows/ppc64le CI jobs).

#include <cstdint>
#include <cstring>

namespace forge {
namespace fp16_detail {

// Bit reinterpretation between float and uint32_t (endian-independent layout:
// memcpy is always safe; the float bit patterns themselves are IEEE 754).
inline std::uint32_t float_bits(float f) {
    std::uint32_t b;
    std::memcpy(&b, &f, sizeof(b));
    return b;
}
inline float float_from_bits(std::uint32_t b) {
    float f;
    std::memcpy(&f, &b, sizeof(f));
    return f;
}

// Round `v / 2^d` to the nearest integer, ties to even (d in [1, 24]).
inline std::uint32_t rne_shift(std::uint32_t v, int d) {
    const std::uint32_t half = 1u << (d - 1);
    const std::uint32_t rem = v & ((half << 1) - 1u);
    std::uint32_t q = v >> d;
    if (rem > half || (rem == half && (q & 1u)))
        ++q;
    return q;
}

// IEEE 754 binary16 -> binary32 bits.
inline std::uint32_t half_to_float_bits(std::uint16_t h) {
    const std::uint32_t sign = (std::uint32_t)(h & 0x8000u) << 16;
    const std::uint32_t exp = (h >> 10) & 0x1fu;
    const std::uint32_t mant = h & 0x3ffu;
    std::uint32_t b;
    if (exp == 0) {
        if (mant == 0) {
            b = sign;  // +/-0
        } else {
            // Subnormal: value = mant * 2^-24. Normalize so bit p (top set
            // bit of the 10-bit mantissa) lands on float's implicit 1.
            int p = 9;
            while (p > 0 && !(mant & (1u << p)))
                --p;
            const std::uint32_t low = mant & ((1u << p) - 1u);
            b = sign | ((103u + (std::uint32_t)p) << 23) | (low << (23 - p));
        }
    } else if (exp == 0x1f) {
        b = sign | 0x7f800000u | (mant << 13);  // +/-inf or NaN
    } else {
        b = sign | ((exp - 15u + 127u) << 23) | (mant << 13);
    }
    return b;
}

// Binary32 bits -> IEEE 754 binary16 bits (round-to-nearest-even).
inline std::uint16_t float_to_half_bits(std::uint32_t b) {
    const std::uint32_t sign16 = (b >> 16) & 0x8000u;
    const std::uint32_t exp = (b >> 23) & 0xffu;
    const std::uint32_t mant = b & 0x7fffffu;

    if (exp == 0xffu) {  // +/-inf or NaN
        std::uint32_t hbits = 0x7c00u;
        if (mant != 0) {  // NaN: keep high payload bits (quieted if empty)
            std::uint32_t hm = (mant >> 13) & 0x3ffu;
            hbits |= hm ? hm : 1u;
        }
        return (std::uint16_t)(sign16 | hbits);
    }
    if (exp == 0 || b == 0)
        return (std::uint16_t)sign16;  // +/-0 / float subnormal

    const std::uint32_t sig = 0x800000u | mant;  // 24-bit significand

    if (exp >= 113u) {
        // Normal half range: E_half = exp - 112 in [1, 30]; keep 11 bits.
        std::uint32_t ehalf = exp - 112u;
        std::uint32_t t = rne_shift(sig, 13);  // in [1024, 2048)
        if (t == 2048u) {
            ++ehalf;
            t = 1024u;
        }
        if (ehalf >= 31u)
            return (std::uint16_t)(sign16 | 0x7c00u);  // overflow -> inf
        return (std::uint16_t)(sign16 | (ehalf << 10) | (t - 1024u));
    }
    if (exp >= 102u) {
        // Half subnormal: value >= 2^-25, expressed in 2^-24 units.
        const std::uint32_t k = rne_shift(sig, 126 - (int)exp);  // in [0, 1024]
        if (k == 0)
            return (std::uint16_t)sign16;
        if (k >= 1024u)
            return (std::uint16_t)(sign16 | 0x0400u);  // rounds to min normal
        return (std::uint16_t)(sign16 | k);
    }
    return (std::uint16_t)sign16;  // |value| < 2^-25 -> +/-0
}

// Bfloat16 -> binary32 bits (zero-extend the high half).
inline std::uint32_t bf16_to_float_bits(std::uint16_t h) {
    return (std::uint32_t)h << 16;
}

// Binary32 bits -> bfloat16 bits (round-to-nearest-even).
inline std::uint16_t float_to_bf16_bits(std::uint32_t b) {
    const std::uint32_t sign16 = (b >> 16) & 0x8000u;
    const std::uint32_t exp = (b >> 23) & 0xffu;
    const std::uint32_t mant = b & 0x7fffffu;
    if (exp == 0xffu) {  // +/-inf or NaN
        return (std::uint16_t)(sign16 | (mant == 0 ? 0x7f80u : 0x7fc0u));
    }
    const std::uint32_t rounded = b + 0x7fffu + ((b >> 16) & 1u);
    return (std::uint16_t)(rounded >> 16);
}

}  // namespace fp16_detail
}  // namespace forge

// CUDA scalar storage types (only when the toolkit headers are unavailable).
struct __half {
    std::uint16_t x;
    __half() = default;
    __half(const __half&) = default;
    __half& operator=(const __half&) = default;
    __half(float f)
        : x(forge::fp16_detail::float_to_half_bits(forge::fp16_detail::float_bits(f))) {}
    operator float() const {
        return forge::fp16_detail::float_from_bits(forge::fp16_detail::half_to_float_bits(x));
    }
};

struct __nv_bfloat16 {
    std::uint16_t x;
    __nv_bfloat16() = default;
    __nv_bfloat16(const __nv_bfloat16&) = default;
    __nv_bfloat16& operator=(const __nv_bfloat16&) = default;
    __nv_bfloat16(float f)
        : x(forge::fp16_detail::float_to_bf16_bits(forge::fp16_detail::float_bits(f))) {}
    operator float() const {
        return forge::fp16_detail::float_from_bits(forge::fp16_detail::bf16_to_float_bits(x));
    }
};

// CUDA-style conversion intrinsics used by Forge host code (see cuda_kernels.h
// and the scalar CPU fallbacks in src/core/op_kernels.cpp).
inline __half __float2half(float f) {
    return __half(f);
}
inline float __half2float(const __half& h) {
    return forge::fp16_detail::float_from_bits(forge::fp16_detail::half_to_float_bits(h.x));
}
inline __nv_bfloat16 __float2bfloat16(float f) {
    return __nv_bfloat16(f);
}
inline float __bfloat162float(const __nv_bfloat16& h) {
    return forge::fp16_detail::float_from_bits(forge::fp16_detail::bf16_to_float_bits(h.x));
}
