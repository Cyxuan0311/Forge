#pragma once
// PowerPC64 VSX attention building-block kernels.
// Provides dot product, scale, and fused-multiply-add for online-softmax
// attention (both decode and prefill paths).

#ifdef USE_VSX
#include <altivec.h>
#endif

namespace forge {
namespace cpu {

#ifdef USE_VSX

// Horizontal sum of a __vector float (delegates to hsum_f32x4 in vec.h).
static inline float hsum_f32(__vector float v) {
    union { __vector float vf; float f[4]; } u;
    u.vf = v;
    return u.f[0] + u.f[1] + u.f[2] + u.f[3];
}

// Dot product of two float32 vectors.
// Quad-accumulator pattern for ILP (instruction-level parallelism).
static inline float dot_f32(const float* a, const float* b, int n) {
    __vector float sum0 = vec_splats(0.0f);
    __vector float sum1 = vec_splats(0.0f);
    __vector float sum2 = vec_splats(0.0f);
    __vector float sum3 = vec_splats(0.0f);

    int i = 0;
    for (; i + 15 < n; i += 16) {
        __vector float a0 = vec_xl(0, (const float*)(a + i));
        __vector float a1 = vec_xl(0, (const float*)(a + i + 4));
        __vector float a2 = vec_xl(0, (const float*)(a + i + 8));
        __vector float a3 = vec_xl(0, (const float*)(a + i + 12));
        __vector float b0 = vec_xl(0, (const float*)(b + i));
        __vector float b1 = vec_xl(0, (const float*)(b + i + 4));
        __vector float b2 = vec_xl(0, (const float*)(b + i + 8));
        __vector float b3 = vec_xl(0, (const float*)(b + i + 12));
        sum0 = vec_madd(a0, b0, sum0);
        sum1 = vec_madd(a1, b1, sum1);
        sum2 = vec_madd(a2, b2, sum2);
        sum3 = vec_madd(a3, b3, sum3);
    }
    for (; i + 3 < n; i += 4) {
        __vector float a0 = vec_xl(0, (const float*)(a + i));
        __vector float b0 = vec_xl(0, (const float*)(b + i));
        sum0 = vec_madd(a0, b0, sum0);
    }

    sum0 = sum0 + sum1;
    sum2 = sum2 + sum3;
    sum0 = sum0 + sum2;
    float total = hsum_f32(sum0);

    // Scalar tail
    for (; i < n; ++i) {
        total += a[i] * b[i];
    }
    return total;
}

// Scale a float32 vector in-place: data[i] *= scale.
static inline void scale_f32(float* data, int n, float scale) {
    __vector float s = vec_splats(scale);
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
        data[i] *= scale;
    }
}

// Fused multiply-add: acc[i] += weight * src[i].
static inline void fmadd_f32(float* acc, const float* src, int n, float weight) {
    __vector float w = vec_splats(weight);
    int i = 0;
    for (; i + 15 < n; i += 16) {
        __vector float s0 = vec_xl(0, (const float*)(src + i));
        __vector float s1 = vec_xl(0, (const float*)(src + i + 4));
        __vector float s2 = vec_xl(0, (const float*)(src + i + 8));
        __vector float s3 = vec_xl(0, (const float*)(src + i + 12));
        __vector float a0 = vec_xl(0, (const float*)(acc + i));
        __vector float a1 = vec_xl(0, (const float*)(acc + i + 4));
        __vector float a2 = vec_xl(0, (const float*)(acc + i + 8));
        __vector float a3 = vec_xl(0, (const float*)(acc + i + 12));
        vec_xst(vec_madd(s0, w, a0), 0, (float*)(acc + i));
        vec_xst(vec_madd(s1, w, a1), 0, (float*)(acc + i + 4));
        vec_xst(vec_madd(s2, w, a2), 0, (float*)(acc + i + 8));
        vec_xst(vec_madd(s3, w, a3), 0, (float*)(acc + i + 12));
    }
    for (; i + 3 < n; i += 4) {
        __vector float s = vec_xl(0, (const float*)(src + i));
        __vector float a = vec_xl(0, (const float*)(acc + i));
        vec_xst(vec_madd(s, w, a), 0, (float*)(acc + i));
    }
    for (; i < n; ++i) {
        acc[i] += weight * src[i];
    }
}

#endif // USE_VSX

}  // namespace cpu
}  // namespace forge
