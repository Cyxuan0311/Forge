#pragma once
// PowerPC64 VSX sampling kernels.
// Provides vectorised argmax, max, softcap, exp-and-sum, scale-normalize
// for the greedy sampling pipeline.
//
// Uses fast-exp polynomial approximation identical to the x86/arm64 sampler
// path (max error ~1.5%), and Cephes-style exp-based tanh for softcap.
// cephes_exp_f32_vsx is defined in elementwise_kernels.h.

#ifdef USE_VSX
#include <altivec.h>
#endif
#include <cmath>
#include <cstring>

namespace forge {
namespace cpu {

#ifdef USE_VSX

// ---- vector_tanh: tanh(z) = (exp(2z)-1)/(exp(2z)+1) via cephes_exp_f32_vsx ----
// (cephes_exp_f32_vsx is defined in elementwise_kernels.h)
static inline __vector float vector_tanh_vsx(__vector float z) {
    __vector float two_z = z + z;
    __vector float e2z = cephes_exp_f32_vsx(two_z);
    __vector float one = vec_splats(1.0f);
    return (e2z - one) / (e2z + one);
}

// ---- softcap_and_argmax_f32 ----
// Applies tanh softcap in-place, returns index of max.
static inline int softcap_and_argmax_f32(float* logits, int n, float cap) {
    float inv_cap = 1.0f / cap;
    int best = 0;
    float best_val = -__builtin_huge_valf();

    int i = 0;
    for (; i + 3 < n; i += 4) {
        __vector float v = vec_xl(0, (const float*)(logits + i));
        __vector float z = v * vec_splats(inv_cap);

        // tanh(z) = (exp(2z)-1)/(exp(2z)+1) via Cephes exp
        __vector float tanh_z = vector_tanh_vsx(z);

        __vector float result = tanh_z * vec_splats(cap);
        vec_xst(result, 0, (float*)(logits + i));

        // Find max within this vector
        union { __vector float vf; float f[4]; } u;
        u.vf = result;
        for (int j = 0; j < 4; ++j) {
            if (u.f[j] > best_val) { best_val = u.f[j]; best = i + j; }
        }
    }
    for (; i < n; ++i) {
        logits[i] = std::tanh(logits[i] / cap) * cap;
        if (logits[i] > best_val) { best_val = logits[i]; best = i; }
    }
    return best;
}

// ---- argmax_f32 ----
static inline int argmax_f32(const float* data, int n) {
    int best = 0;
    float best_val = data[0];

    int i = 1;
    for (; i + 3 < n; i += 4) {
        __vector float v = vec_xl(0, (const float*)(data + i));
        __vector float max_vec = vec_max(v, vec_splats(best_val));
        // Check if any element in this vector exceeds current best
        float max_v;
        {
            union { __vector float vf; float f[4]; } um;
            um.vf = max_vec;
            max_v = um.f[0];
            if (um.f[1] > max_v) max_v = um.f[1];
            if (um.f[2] > max_v) max_v = um.f[2];
            if (um.f[3] > max_v) max_v = um.f[3];
        }
        if (max_v > best_val) {
            // Find the exact index within this vector
            union { __vector float vf; float f[4]; } u;
            u.vf = v;
            for (int j = 0; j < 4; ++j) {
                if (u.f[j] > best_val) { best_val = u.f[j]; best = i + j; }
            }
        }
    }
    for (; i < n; ++i) {
        if (data[i] > best_val) { best_val = data[i]; best = i; }
    }
    return best;
}

// ---- max_f32 ----
static inline float max_f32(const float* data, int n) {
    __vector float max_v = vec_splats(data[0]);
    int i = 0;
    for (; i + 3 < n; i += 4) {
        __vector float v = vec_xl(0, (const float*)(data + i));
        max_v = vec_max(max_v, v);
    }
    union { __vector float vf; float f[4]; } u;
    u.vf = max_v;
    float result = u.f[0];
    if (u.f[1] > result) result = u.f[1];
    if (u.f[2] > result) result = u.f[2];
    if (u.f[3] > result) result = u.f[3];
    for (; i < n; ++i) {
        if (data[i] > result) result = data[i];
    }
    return result;
}

// ---- softcap_and_max_f32 ----
static inline float softcap_and_max_f32(float* data, int n, float cap) {
    float inv_cap = 1.0f / cap;
    // Handle element 0 separately
    data[0] = std::tanh(data[0] / cap) * cap;
    float best_val = data[0];

    int i = 1;
    for (; i + 3 < n; i += 4) {
        __vector float v = vec_xl(0, (const float*)(data + i));
        __vector float z = v * vec_splats(inv_cap);
        __vector float tanh_z = vector_tanh_vsx(z);
        __vector float result = tanh_z * vec_splats(cap);
        vec_xst(result, 0, (float*)(data + i));

        union { __vector float vf; float f[4]; } u;
        u.vf = result;
        for (int j = 0; j < 4; ++j) {
            if (u.f[j] > best_val) best_val = u.f[j];
        }
    }
    for (; i < n; ++i) {
        data[i] = std::tanh(data[i] / cap) * cap;
        if (data[i] > best_val) best_val = data[i];
    }
    return best_val;
}

// ---- exp_and_sum_f32 (fast-exp approximation, ~1.5% error) ----
// exp(x) approximated as 2^(integer part) * 2^(fractional part):
//   2^f ≈ 1 + f*(c1 + f*(c2 + f*c3))
// where c1=0.693147, c2=0.240227, c3=0.055504
// Matches the x86 AVX2 / ARM64 NEON fast-exp sampler path exactly.
static inline float exp_and_sum_f32(const float* data, float* out, int n,
                                    float max_val, float inv_temp) {
    const float log2e = 1.4426950408889634f;
    const float c1 = 0.6931471805599453f;
    const float c2 = 0.2402265069591006f;
    const float c3 = 0.05550410866482158f;

    __vector float sum_all = vec_splats(0.0f);
    __vector float v_max = vec_splats(max_val);
    __vector float v_factor = vec_splats(inv_temp * log2e);
    __vector float v_c1 = vec_splats(c1);
    __vector float v_c2 = vec_splats(c2);
    __vector float v_c3 = vec_splats(c3);
    __vector float v_one = vec_splats(1.0f);

    int i = 0;
    for (; i + 3 < n; i += 4) {
        __vector float d = vec_xl(0, (const float*)(data + i));
        // x = (d - max_val) * inv_temp * log2(e)
        __vector float x = (d - v_max) * v_factor;
        // Floor to get integer exponent
        __vector signed int x_int = vec_cts(x, 0);
        // Fractional part f = x - floor(x) in [0, 1)
        __vector float xf = vec_ctf(x_int, 0);
        __vector unsigned int lt_mask = (__vector unsigned int)vec_cmplt(x, xf);
        x_int = x_int - (__vector signed int)(lt_mask & vec_splats((unsigned int)1));
        __vector float f = x - vec_ctf(x_int, 0);

        // Polynomial: 1 + f*(c1 + f*(c2 + f*c3))
        __vector float poly = v_c3;
        poly = vec_madd(poly, f, v_c2);
        poly = vec_madd(poly, f, v_c1);
        poly = vec_madd(poly, f, v_one);

        // Scale by 2^int: add exponent bits
        __vector signed int exp_bits = vec_sl(x_int, vec_splats((unsigned int)23));
        __vector float result = (__vector float)((__vector signed int)poly + exp_bits);

        vec_xst(result, 0, (float*)(out + i));
        sum_all = sum_all + result;
    }

    // Horizontal sum of the vector accumulator
    union { __vector float vf; float f[4]; } us;
    us.vf = sum_all;
    float sum = us.f[0] + us.f[1] + us.f[2] + us.f[3];

    for (; i < n; ++i) {
        float x = (data[i] - max_val) * inv_temp * log2e;
        int x_int = (int)x;
        if (x < (float)x_int) x_int--;  // floor for negative
        float f = x - (float)x_int;
        float poly = 1.0f + f * (c1 + f * (c2 + f * c3));
        // Scale by 2^x_int via integer bit addition
        int exp_bits_i = x_int << 23;
        int poly_bits;
        std::memcpy(&poly_bits, &poly, sizeof(int));
        poly_bits += exp_bits_i;
        float poly_scaled;
        std::memcpy(&poly_scaled, &poly_bits, sizeof(float));
        out[i] = poly_scaled;
        sum += poly_scaled;
    }
    return sum;
}

// ---- scale_normalize_f32 ----
static inline void scale_normalize_f32(float* data, int n, float inv_sum) {
    __vector float s = vec_splats(inv_sum);
    int i = 0;
    for (; i + 15 < n; i += 16) {
        __vector float d0 = vec_xl(0, (const float*)(data + i));
        __vector float d1 = vec_xl(0, (const float*)(data + i + 4));
        __vector float d2 = vec_xl(0, (const float*)(data + i + 8));
        __vector float d3 = vec_xl(0, (const float*)(data + i + 12));
        vec_xst(d0 * s, 0, (float*)(data + i));
        vec_xst(d1 * s, 0, (float*)(data + i + 4));
        vec_xst(d2 * s, 0, (float*)(data + i + 8));
        vec_xst(d3 * s, 0, (float*)(data + i + 12));
    }
    for (; i + 3 < n; i += 4) {
        __vector float d = vec_xl(0, (const float*)(data + i));
        vec_xst(d * s, 0, (float*)(data + i));
    }
    for (; i < n; ++i) {
        data[i] *= inv_sum;
    }
}

#endif // USE_VSX

}  // namespace cpu
}  // namespace forge
