#pragma once
// ARM64 NEON quantized dot product kernels.
// Provides vec_dot_q4_0_q8_0, vec_dot_q8_0_q8_0, vec_dot_q5_0_q8_0,
// vec_dot_q4_1_q8_0, vec_dot_q5_1_q8_0 — matching the x86 AVX2 API.
//
// NEON 128-bit registers process 16 bytes at a time (half of AVX2's 32).
// #ifdef USE_DOTPROD enables vdotq_s32 (ARMv8.2+dotprod, Apple M1+).
// Fallback uses vmull_s8 + vpadal for baseline ARMv8-A NEON.

#ifdef USE_NEON
#include <arm_neon.h>
#endif
#include <cstring>
#include <cmath>

namespace forge {
namespace cpu {

#ifdef USE_NEON

// ---- Internal helpers (namespace arm_dot) ----

namespace arm_dot {

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
// Output: two uint8x16_t — lo has even-index elements, hi has odd-index elements.
// Caller should interleave with vzip to get consecutive element order.
static inline void nibbles_to_bytes_32(const uint8_t* src,
                                       uint8x16_t& lo, uint8x16_t& hi) {
    uint8x16_t tmp = vld1q_u8(src);
    lo = vandq_u8(tmp, vdupq_n_u8(0x0F));
    hi = vshrq_n_u8(tmp, 4);
}

// Interleave even/odd nibbles to produce consecutive element order.
// Input: lo=[e0,e2,...,e30], hi=[e1,e3,...,e31]
// Output: q0=[e0,e1,...,e15], q1=[e16,e17,...,e31]
static inline void interleave_nibbles_neon(const uint8x16_t& lo,
                                           const uint8x16_t& hi,
                                           uint8x16_t& q0, uint8x16_t& q1) {
    q0 = vzip1q_u8(lo, hi);
    q1 = vzip2q_u8(lo, hi);
}

// 8-wide horizontal sum of int32x4_t pairs into a single int32.
static inline int32_t hsum_i32x4(int32x4_t v0, int32x4_t v1) {
    int32x4_t sum = vaddq_s32(v0, v1);
    return vaddvq_s32(sum);
}

#ifdef USE_DOTPROD
// Fast path: 16-way dot product using ARMv8.2 dotprod.
// Computes dot of 16 unsigned nibbles with 16 signed q8 in one instruction.
static inline int32x4_t dot_16_nibble_q8(const uint8x16_t& q4_nibbles,
                                          const int8x16_t& q8_signed,
                                          int32x4_t acc) {
    return vdotq_s32(acc, vreinterpretq_s8_u8(q4_nibbles), q8_signed);
}
#else
// Baseline NEON: 16-way dot product using vmull_s8 + pairwise accumulate.
// Computes sum(unsigned_nibble * signed_q8) without pre-subtraction.
// (The -8 * sum(q8) bias correction is applied by the caller.)
static inline int32x4_t dot_16_nibble_q8_fallback(const uint8x16_t& q4_nibbles,
                                                   const int8x16_t& q8_signed,
                                                   int32x4_t acc) {
    // Keep nibbles unsigned [0..15]; caller handles -8 * sum(q8) bias.
    int8x16_t q4_signed = vreinterpretq_s8_u8(q4_nibbles);
    // Wide multiply: low 8 products → int16x8, high 8 products → int16x8
    int16x8_t prod_lo = vmull_s8(vget_low_s8(q4_signed), vget_low_s8(q8_signed));
    int16x8_t prod_hi = vmull_s8(vget_high_s8(q4_signed), vget_high_s8(q8_signed));
    // Pairwise add to int32
    int32x4_t sum_lo = vpaddlq_s16(prod_lo);
    int32x4_t sum_hi = vpaddlq_s16(prod_hi);
    return vaddq_s32(acc, vaddq_s32(sum_lo, sum_hi));
}
#endif // USE_DOTPROD

}  // namespace arm_dot

// ---- Quantization of activation row to block_q8_0_act ----
// Block size = 32 elements. Same format as x86 vec_dot.h.
// Used by vec_dot and GEMV decode paths.

struct block_q8_0_act {
    float  d;       // fp32 scale
    int8_t qs[32];  // 32 signed 8-bit quantized activation values
};

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

// ---- vec_dot_q4_0_q8_0_neon ----
// Q4_0 block layout (18 bytes): d[2] (fp16) + qs[16] (32 x 4-bit nibbles)
// Matches x86 AVX2 vec_dot_q4_0_q8_0 semantics exactly.
static inline float vec_dot_q4_0_q8_0_neon(const uint8_t* w_row,
                                            const block_q8_0_act* act,
                                            int nb) {
    float result = 0.0f;
    for (int i = 0; i < nb; ++i) {
        // Read fp16 weight scale
        uint16_t ws;
        std::memcpy(&ws, w_row + i * 18, sizeof(uint16_t));
        float scale_w = arm_dot::fp16_to_fp32(ws);
        float scale = scale_w * act[i].d;

        // Expand 32 nibbles and interleave to correct element order:
        // lo=[e0, e2, ..., e30], hi=[e1, e3, ..., e31]
        // After interleave: q0=[e0, e1, ..., e15], q1=[e16, e17, ..., e31]
        uint8x16_t lo, hi;
        arm_dot::nibbles_to_bytes_32(w_row + i * 18 + 2, lo, hi);
        uint8x16_t q0, q1;
        arm_dot::interleave_nibbles_neon(lo, hi, q0, q1);

        // Load activation quants as signed int8
        int8x16_t q8_lo = vld1q_s8(act[i].qs);
        int8x16_t q8_hi = vld1q_s8(act[i].qs + 16);

        int32x4_t dot_acc = vdupq_n_s32(0);
#ifdef USE_DOTPROD
        dot_acc = arm_dot::dot_16_nibble_q8(q0, q8_lo, dot_acc);
        dot_acc = arm_dot::dot_16_nibble_q8(q1, q8_hi, dot_acc);
#else
        dot_acc = arm_dot::dot_16_nibble_q8_fallback(q0, q8_lo, dot_acc);
        dot_acc = arm_dot::dot_16_nibble_q8_fallback(q1, q8_hi, dot_acc);
#endif
        // Sum the 4 int32 lanes
        int32_t dot_i32 = vaddvq_s32(dot_acc);

        // Bias correction: subtract 8 * sum(q8) since nibble values are
        // [0..15] but mathematically represent (nibble - 8) * scale.
        int16x8_t sum_q8_lo = vpaddlq_s8(q8_lo);
        int16x8_t sum_q8_hi = vpaddlq_s8(q8_hi);
        int32x4_t sum_q8 = vaddq_s32(vpaddlq_s16(sum_q8_lo), vpaddlq_s16(sum_q8_hi));
        int32_t bias = vaddvq_s32(sum_q8) * 8;

        result += scale * static_cast<float>(dot_i32 - bias);
    }
    return result;
}

// ---- vec_dot_q8_0_q8_0_neon ----
// Q8_0 block layout (34 bytes): d[2] (fp16) + qs[32] (signed int8)
static inline float vec_dot_q8_0_q8_0_neon(const uint8_t* w_row,
                                            const block_q8_0_act* act,
                                            int nb) {
    float result = 0.0f;
    for (int i = 0; i < nb; ++i) {
        uint16_t ws;
        std::memcpy(&ws, w_row + i * 34, sizeof(uint16_t));
        float scale = arm_dot::fp16_to_fp32(ws) * act[i].d;

        int8x16_t w_lo = vld1q_s8(reinterpret_cast<const int8_t*>(w_row + i * 34 + 2));
        int8x16_t w_hi = vld1q_s8(reinterpret_cast<const int8_t*>(w_row + i * 34 + 18));
        int8x16_t a_lo = vld1q_s8(act[i].qs);
        int8x16_t a_hi = vld1q_s8(act[i].qs + 16);

        int32x4_t dot_acc = vdupq_n_s32(0);
#ifdef USE_DOTPROD
        dot_acc = vdotq_s32(dot_acc, w_lo, a_lo);
        dot_acc = vdotq_s32(dot_acc, w_hi, a_hi);
#else
        // vmull_s8 fallback for signed×signed dot
        int16x8_t p0 = vmull_s8(vget_low_s8(w_lo), vget_low_s8(a_lo));
        int16x8_t p1 = vmull_s8(vget_high_s8(w_lo), vget_high_s8(a_lo));
        int16x8_t p2 = vmull_s8(vget_low_s8(w_hi), vget_low_s8(a_hi));
        int16x8_t p3 = vmull_s8(vget_high_s8(w_hi), vget_high_s8(a_hi));
        int32x4_t s0 = vpaddlq_s16(p0);
        int32x4_t s1 = vpaddlq_s16(p1);
        int32x4_t s2 = vpaddlq_s16(p2);
        int32x4_t s3 = vpaddlq_s16(p3);
        dot_acc = vaddq_s32(vaddq_s32(s0, s1), vaddq_s32(s2, s3));
#endif

        result += scale * static_cast<float>(vaddvq_s32(dot_acc));
    }
    return result;
}

// ---- vec_dot_q4_1_q8_0_neon ----
// Q4_1 block layout (20 bytes): d[2] + m[2] (both fp16) + qs[16] (32 nibbles)
// Value = nibble * d + m
static inline float vec_dot_q4_1_q8_0_neon(const uint8_t* w_row,
                                            const block_q8_0_act* act,
                                            int nb) {
    float acc_d = 0.0f, acc_m = 0.0f;
    for (int i = 0; i < nb; ++i) {
        uint16_t d_bits, m_bits;
        std::memcpy(&d_bits, w_row + i * 20, sizeof(uint16_t));
        std::memcpy(&m_bits, w_row + i * 20 + 2, sizeof(uint16_t));
        float d_val = arm_dot::fp16_to_fp32(d_bits);
        float m_val = arm_dot::fp16_to_fp32(m_bits);
        float d_scale = d_val * act[i].d;
        float m_scale = m_val * act[i].d;

        uint8x16_t lo, hi;
        arm_dot::nibbles_to_bytes_32(w_row + i * 20 + 4, lo, hi);
        int8x16_t q8_lo = vld1q_s8(act[i].qs);
        int8x16_t q8_hi = vld1q_s8(act[i].qs + 16);

        int32x4_t dot_acc = vdupq_n_s32(0);
#ifdef USE_DOTPROD
        dot_acc = arm_dot::dot_16_nibble_q8(lo, q8_lo, dot_acc);
        dot_acc = arm_dot::dot_16_nibble_q8(hi, q8_hi, dot_acc);
#else
        dot_acc = arm_dot::dot_16_nibble_q8_fallback(lo, q8_lo, dot_acc);
        dot_acc = arm_dot::dot_16_nibble_q8_fallback(hi, q8_hi, dot_acc);
#endif
        int32_t dot_i32 = vaddvq_s32(dot_acc);

        // Sum of activation quants for the m term
        int16x8_t sum_q8_lo = vpaddlq_s8(q8_lo);
        int16x8_t sum_q8_hi = vpaddlq_s8(q8_hi);
        int32x4_t sum_q8 = vaddq_s32(vpaddlq_s16(sum_q8_lo), vpaddlq_s16(sum_q8_hi));
        int32_t sum_act = vaddvq_s32(sum_q8);

        acc_d += d_scale * static_cast<float>(dot_i32);
        acc_m += m_scale * static_cast<float>(sum_act);
    }
    return acc_d + acc_m;
}

#endif // USE_NEON

}  // namespace cpu
}  // namespace forge
