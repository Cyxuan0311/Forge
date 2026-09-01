#pragma once

#include <cmath>
#include <cstdint>

namespace forge {

// This header is included by both host (C++) and device (NVCC) translation
// units. CUDA-only qualifiers must be hidden from the host compiler.
#ifdef __CUDACC__
#    define FORGE_FP8_QUALIFIER __host__ __device__
#else
#    define FORGE_FP8_QUALIFIER
#endif

// =========================================================================
// FP8 <-> FP32 conversion utilities (host + device).
//
// Two standard 8-bit float formats are supported:
//   - E4M3: 1 sign, 4 exp (bias 7), 3 mantissa.  Finite range ~[-240, 240].
//   - E5M2: 1 sign, 5 exp (bias 15), 2 mantissa. Finite range ~[-57344, 57344].
//
// Used by:
//   - KV cache quantize_row / dequantize_row (CPU)
//   - CUDA quantize_fp8 matrix kernels (write path)
//   - Fused / paged flash-attention decode kernels (online dequant read path)
// =========================================================================

// ---- E4M3 decode ----
inline FORGE_FP8_QUALIFIER float fp8_e4m3_to_fp32(uint8_t b) {
    uint32_t sign = (b >> 7) & 0x1u;
    uint32_t exp = (b >> 3) & 0xFu;
    uint32_t mant = b & 0x7u;
    float s = sign ? -1.0f : 1.0f;
    if (exp == 0) {
        // subnormal: 2^-6 * (mant / 8)
        if (mant == 0)
            return s * 0.0f;
        return s * exp2f(-6.0f) * (mant / 8.0f);
    } else if (exp == 0xF) {
        // E4M3 has no NaN; inf is clamped to the max finite to avoid NaN in
        // downstream attention math (KV magnitudes are small after RMSNorm).
        return s * 240.0f;
    }
    return s * exp2f(static_cast<float>(exp) - 7.0f) * (1.0f + mant / 8.0f);
}

// ---- E5M2 decode ----
inline FORGE_FP8_QUALIFIER float fp8_e5m2_to_fp32(uint8_t b) {
    uint32_t sign = (b >> 7) & 0x1u;
    uint32_t exp = (b >> 2) & 0x1Fu;
    uint32_t mant = b & 0x3u;
    float s = sign ? -1.0f : 1.0f;
    if (exp == 0) {
        if (mant == 0)
            return s * 0.0f;
        return s * exp2f(-14.0f) * (mant / 4.0f);
    } else if (exp == 0x1F) {
        if (mant == 0)
            return s * 57344.0f;  // inf -> clamp to max finite
        return 0.0f;              // NaN  -> treat as 0
    }
    return s * exp2f(static_cast<float>(exp) - 15.0f) * (1.0f + mant / 4.0f);
}

// ---- E4M3 encode (round-to-nearest) ----
inline FORGE_FP8_QUALIFIER uint8_t fp32_to_fp8_e4m3(float x) {
    float ax = fabsf(x);
    uint8_t sign = (x < 0.0f) ? 0x80 : 0x00;
    if (ax <= 0.0f)
        return sign;
    if (ax > 240.0f)
        ax = 240.0f;  // clamp to max normal finite

    int e0 = static_cast<int>(floorf(log2f(ax)));
    int E = e0 + 7;
    if (E < 1) {
        // subnormal: 2^-6 * (m / 8), m in [0,7]
        int m = static_cast<int>(roundf(ax * 512.0f));  // ax * 8 * 2^6
        if (m > 7)
            m = 7;
        if (m < 0)
            m = 0;
        return static_cast<uint8_t>(sign | m);
    }
    if (E > 14) {
        return static_cast<uint8_t>(sign | 0xE7);  // E=14, m=7 = 240
    }
    float norm = ax * exp2f(static_cast<float>(7 - E));  // in [1, 2)
    int m = static_cast<int>(roundf((norm - 1.0f) * 8.0f));
    if (m > 7)
        m = 7;
    if (m < 0)
        m = 0;
    return static_cast<uint8_t>(sign | (static_cast<uint8_t>(E) << 3) | static_cast<uint8_t>(m));
}

// ---- E5M2 encode (round-to-nearest) ----
inline FORGE_FP8_QUALIFIER uint8_t fp32_to_fp8_e5m2(float x) {
    float ax = fabsf(x);
    uint8_t sign = (x < 0.0f) ? 0x80 : 0x00;
    if (ax <= 0.0f)
        return sign;
    if (ax > 57344.0f)
        ax = 57344.0f;  // clamp to max normal finite

    int e0 = static_cast<int>(floorf(log2f(ax)));
    int E = e0 + 15;
    if (E < 1) {
        // subnormal: 2^-14 * (m / 4), m in [0,3]
        int m = static_cast<int>(roundf(ax * 65536.0f));  // ax * 4 * 2^14
        if (m > 3)
            m = 3;
        if (m < 0)
            m = 0;
        return static_cast<uint8_t>(sign | m);
    }
    if (E > 30) {
        return static_cast<uint8_t>(sign | 0xFB);  // E=30, m=3 = 57344
    }
    float norm = ax * exp2f(static_cast<float>(15 - E));  // in [1, 2)
    int m = static_cast<int>(roundf((norm - 1.0f) * 4.0f));
    if (m > 3)
        m = 3;
    if (m < 0)
        m = 0;
    return static_cast<uint8_t>(sign | (static_cast<uint8_t>(E) << 2) | static_cast<uint8_t>(m));
}

// ---- Device helpers for fused/paged attention decode kernels ----
// Tag types select the FP8 variant; fp8_load() performs the per-element
// online dequant inside the attention kernel. Only needed under NVCC.
#ifdef __CUDACC__
struct Fp8E4M3 {};
struct Fp8E5M2 {};

template <typename FMT>
__device__ inline float fp8_load(const uint8_t* p, int i);

template <>
__device__ inline float fp8_load<Fp8E4M3>(const uint8_t* p, int i) {
    return fp8_e4m3_to_fp32(p[i]);
}

template <>
__device__ inline float fp8_load<Fp8E5M2>(const uint8_t* p, int i) {
    return fp8_e5m2_to_fp32(p[i]);
}

// Device online quantize helpers (used by scaled FP8 KV quantize kernels).
template <typename FMT>
__device__ inline uint8_t fp8_store(float x);

template <>
__device__ inline uint8_t fp8_store<Fp8E4M3>(float x) {
    return fp32_to_fp8_e4m3(x);
}

template <>
__device__ inline uint8_t fp8_store<Fp8E5M2>(float x) {
    return fp32_to_fp8_e5m2(x);
}
#endif  // __CUDACC__

}  // namespace forge
