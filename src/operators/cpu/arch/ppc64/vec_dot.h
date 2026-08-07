#pragma once
// PowerPC64 VSX quantized dot product kernels.
// Provides vec_dot_q4_0_q8_0, vec_dot_q8_0_q8_0, vec_dot_q4_1_q8_0
// — matching the x86 AVX2 and ARM64 NEON API.
//
// VSX has NO dotprod instruction, so we always use the vmull fallback
// path via vec_mule/vec_mulo + vec_sum4s.
//
// This header is self-contained and does NOT include vec.h or
// elementwise_kernels.h.

#ifdef USE_VSX
#include <altivec.h>
#endif
#include <cstdint>
#include <cstring>
#include <cmath>

namespace forge {
namespace cpu {

// ---- Q8_0 activation block (for dot products, not for storage) ----
struct block_q8_0_act {
    float   d;       // scale (FP32)
    int8_t  qs[32];  // quantized values
};

#ifdef USE_VSX

// ---- Internal helpers (namespace vsx_dot) ----

namespace vsx_dot {

// Software fp16-to-fp32 (same as vec.h — kept self-contained for vec_dot)
static inline float fp16_to_fp32(uint16_t bits) {
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

// Expand 16 packed nibble bytes into 32 unsigned bytes in [0..15].
// Input: 16 bytes with 2 nibbles each (low nibble = even element, high = odd).
// Output: lo=[e0,e2,...,e30], hi=[e1,e3,...,e31].
static inline void nibbles_to_bytes_32(const uint8_t* src,
                                       __vector unsigned char& lo,
                                       __vector unsigned char& hi) {
    __vector unsigned char tmp = vec_xl(0, src);
    __vector unsigned char mask = vec_splats((unsigned char)0x0F);
    lo = tmp & mask;
    hi = vec_sr(tmp, vec_splats((unsigned char)4));
}

// Interleave even/odd nibbles to produce consecutive element order.
// Input: lo=[e0,e2,...,e30], hi=[e1,e3,...,e31]
// Output: q0=[e0,e1,...,e15], q1=[e16,e17,...,e31]
// Uses vec_perm with shuffle masks equivalent to NEON vzip1/vzip2.
static inline void interleave_nibbles_vsx(const __vector unsigned char& lo,
                                          const __vector unsigned char& hi,
                                          __vector unsigned char& q0,
                                          __vector unsigned char& q1) {
    static const __vector unsigned char perm_mask_z1 = {
         0, 16,  1, 17,  2, 18,  3, 19,
         4, 20,  5, 21,  6, 22,  7, 23
    };
    static const __vector unsigned char perm_mask_z2 = {
         8, 24,  9, 25, 10, 26, 11, 27,
        12, 28, 13, 29, 14, 30, 15, 31
    };
    q0 = vec_perm(lo, hi, perm_mask_z1);
    q1 = vec_perm(lo, hi, perm_mask_z2);
}

// Dot product of 16 unsigned nibble bytes with 16 signed q8 bytes.
// Uses vec_mule/vec_mulo fallback (VSX has no dotprod instruction).
// Returns scalar int sum. Caller handles -8*sum(q8) bias correction.
static inline int dot_16_nibble_q8(__vector unsigned char q4_nibbles,
                                   __vector signed char q8_signed) {
    __vector signed char q4_s = (__vector signed char)q4_nibbles;
    __vector signed short pe = vec_mule(q4_s, q8_signed);
    __vector signed short po = vec_mulo(q4_s, q8_signed);
    __vector signed int zero = vec_splats((int)0);
    __vector signed int se = vec_sum4s(pe, zero);
    __vector signed int so = vec_sum4s(po, zero);
    __vector signed int s32 = se + so;
    // Horizontal sum of 4 int32 lanes
    union { __vector signed int vi; int i[4]; } u;
    u.vi = s32;
    return u.i[0] + u.i[1] + u.i[2] + u.i[3];
}

// Dot product of 16 signed bytes with 16 signed bytes.
// Used for Q8_0 × Q8_0 (both signed int8, no bias correction needed).
static inline int dot_16_signed_signed(__vector signed char a, __vector signed char b) {
    __vector signed short pe = vec_mule(a, b);
    __vector signed short po = vec_mulo(a, b);
    __vector signed int zero = vec_splats((int)0);
    __vector signed int se = vec_sum4s(pe, zero);
    __vector signed int so = vec_sum4s(po, zero);
    __vector signed int s32 = se + so;
    union { __vector signed int vi; int i[4]; } u;
    u.vi = s32;
    return u.i[0] + u.i[1] + u.i[2] + u.i[3];
}

}  // namespace vsx_dot

// ---- Quantize FP32 row to Q8_0_act activation format ----
// Block size = 32 elements. Same scalar logic as arm64 / x86 paths.
static inline void quantize_row_q8_0_act(const float* src, block_q8_0_act* dst, int k) {
    const int QK = 32;
    int nb = k / QK;
    for (int i = 0; i < nb; ++i) {
        float amax = 0.0f;
        for (int j = 0; j < QK; ++j) {
            float v = std::fabs(src[i * QK + j]);
            if (v > amax) amax = v;
        }
        float d = amax / 127.0f;
        float id = (amax > 0.0f) ? (1.0f / d) : 0.0f;
        dst[i].d = d;
        for (int j = 0; j < QK; ++j) {
            float x = src[i * QK + j] * id;
            x = (x >= 0.0f) ? (x + 0.5f) : (x - 0.5f);
            if (x > 127.0f) x = 127.0f;
            if (x < -128.0f) x = -128.0f;
            dst[i].qs[j] = static_cast<int8_t>(static_cast<int>(x));
        }
    }
    // Partial tail block
    int rem = k - nb * QK;
    if (rem > 0) {
        float amax = 0.0f;
        for (int j = 0; j < rem; ++j) {
            float v = std::fabs(src[nb * QK + j]);
            if (v > amax) amax = v;
        }
        float d = amax / 127.0f;
        float id = (amax > 0.0f) ? (1.0f / d) : 0.0f;
        dst[nb].d = d;
        int j;
        for (j = 0; j < rem; ++j) {
            float x = src[nb * QK + j] * id;
            x = (x >= 0.0f) ? (x + 0.5f) : (x - 0.5f);
            if (x > 127.0f) x = 127.0f;
            if (x < -128.0f) x = -128.0f;
            dst[nb].qs[j] = static_cast<int8_t>(static_cast<int>(x));
        }
        for (; j < QK; ++j) dst[nb].qs[j] = 0;
    }
}

// ---- vec_dot_q4_0_q8_0_vsx ----
// Q4_0 block layout (18 bytes): d[2] (fp16) + qs[16] (32 × 4-bit nibbles)
// Q4_0 value = (nibble - 8) * scale_w
// dot = scale_w * scale_a * (sum(nibble * q8) - 8 * sum(q8))
static inline float vec_dot_q4_0_q8_0_vsx(const uint8_t* w_row,
                                           const block_q8_0_act* act,
                                           int nb) {
    constexpr int BLOCK_BYTES = 18;
    float result = 0.0f;

    for (int i = 0; i < nb; ++i) {
        uint16_t scale_bits;
        std::memcpy(&scale_bits, w_row + i * BLOCK_BYTES, 2);
        float scale = vsx_dot::fp16_to_fp32(scale_bits) * act[i].d;

        // Expand 32 nibbles and interleave to correct element order:
        // lo=[e0, e2, ..., e30], hi=[e1, e3, ..., e31]
        // After interleave: q0=[e0, e1, ..., e15], q1=[e16, e17, ..., e31]
        __vector unsigned char lo, hi;
        vsx_dot::nibbles_to_bytes_32(w_row + i * BLOCK_BYTES + 2, lo, hi);
        __vector unsigned char q0, q1;
        vsx_dot::interleave_nibbles_vsx(lo, hi, q0, q1);

        // Load activation quants as signed int8
        __vector signed char q8_lo = (__vector signed char)vec_xl(0, act[i].qs);
        __vector signed char q8_hi = (__vector signed char)vec_xl(0, act[i].qs + 16);

        // Dot product of 2×16 elements
        int dot_i32 = vsx_dot::dot_16_nibble_q8(q0, q8_lo);
        dot_i32 += vsx_dot::dot_16_nibble_q8(q1, q8_hi);

        // Bias correction: subtract 8 * sum(q8) since nibble values
        // are [0..15] but mathematically represent (nibble - 8) * scale.
        int sum_q8 = 0;
        for (int j = 0; j < 32; ++j) sum_q8 += (int)act[i].qs[j];

        result += scale * static_cast<float>(dot_i32 - sum_q8 * 8);
    }
    return result;
}

// ---- vec_dot_q8_0_q8_0_vsx ----
// Q8_0 block layout (34 bytes): d[2] (fp16) + qs[32] (signed int8)
// Both weight and activation are signed int8 — no bias correction needed.
static inline float vec_dot_q8_0_q8_0_vsx(const uint8_t* w_row,
                                           const block_q8_0_act* act,
                                           int nb) {
    constexpr int BLOCK_BYTES = 34;
    float result = 0.0f;

    for (int i = 0; i < nb; ++i) {
        uint16_t scale_bits;
        std::memcpy(&scale_bits, w_row + i * BLOCK_BYTES, 2);
        float scale = vsx_dot::fp16_to_fp32(scale_bits) * act[i].d;

        // Load weight quants as signed int8
        __vector signed char w_lo = (__vector signed char)vec_xl(0,
            w_row + i * BLOCK_BYTES + 2);
        __vector signed char w_hi = (__vector signed char)vec_xl(0,
            w_row + i * BLOCK_BYTES + 18);
        // Load activation quants
        __vector signed char a_lo = (__vector signed char)vec_xl(0, act[i].qs);
        __vector signed char a_hi = (__vector signed char)vec_xl(0, act[i].qs + 16);

        int dot = vsx_dot::dot_16_signed_signed(w_lo, a_lo);
        dot += vsx_dot::dot_16_signed_signed(w_hi, a_hi);

        result += scale * static_cast<float>(dot);
    }
    return result;
}

// ---- vec_dot_q4_1_q8_0_vsx ----
// Q4_1 block layout (20 bytes): d[2] + m[2] (both fp16) + qs[16] (32 nibbles)
// Q4_1 value = nibble * d + m  (nibble values 0..15 are raw, no -8 bias)
// dot = d_scale * sum(nibble * q8) + m_scale * sum(q8)
static inline float vec_dot_q4_1_q8_0_vsx(const uint8_t* w_row,
                                           const block_q8_0_act* act,
                                           int nb) {
    constexpr int BLOCK_BYTES = 20;
    float acc_d = 0.0f, acc_m = 0.0f;

    for (int i = 0; i < nb; ++i) {
        uint16_t d_bits, m_bits;
        std::memcpy(&d_bits, w_row + i * BLOCK_BYTES, 2);
        std::memcpy(&m_bits, w_row + i * BLOCK_BYTES + 2, 2);
        float d_val = vsx_dot::fp16_to_fp32(d_bits);
        float m_val = vsx_dot::fp16_to_fp32(m_bits);
        float d_scale = d_val * act[i].d;
        float m_scale = m_val * act[i].d;

        // Expand 32 nibbles and interleave (same interleave as Q4_0)
        __vector unsigned char lo, hi;
        vsx_dot::nibbles_to_bytes_32(w_row + i * BLOCK_BYTES + 4, lo, hi);
        __vector unsigned char q0, q1;
        vsx_dot::interleave_nibbles_vsx(lo, hi, q0, q1);

        __vector signed char q8_lo = (__vector signed char)vec_xl(0, act[i].qs);
        __vector signed char q8_hi = (__vector signed char)vec_xl(0, act[i].qs + 16);

        // Raw dot: nibble values [0..15] — no bias correction
        int dot_i32 = vsx_dot::dot_16_nibble_q8(q0, q8_lo);
        dot_i32 += vsx_dot::dot_16_nibble_q8(q1, q8_hi);

        // Sum of activation quants for the m term
        int sum_act = 0;
        for (int j = 0; j < 32; ++j) sum_act += (int)act[i].qs[j];

        acc_d += d_scale * static_cast<float>(dot_i32);
        acc_m += m_scale * static_cast<float>(sum_act);
    }
    return acc_d + acc_m;
}

#endif // USE_VSX

}  // namespace cpu
}  // namespace forge
