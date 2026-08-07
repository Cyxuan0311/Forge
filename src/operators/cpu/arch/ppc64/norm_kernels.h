#pragma once
// PowerPC64 VSX RMS normalization kernel.
// Processes one row: o_row = x_row * rms * w_row (optional weight).
// rms = 1/sqrt(sum(x^2)/cols + eps).

#ifdef USE_VSX
#include <altivec.h>
#endif
#include <cmath>

namespace forge {
namespace cpu {

#ifdef USE_VSX

static inline void rms_norm_row_f32(const float* x_row, const float* w_row, float* o_row,
                                    int cols, float eps) {
    // Compute sum of squares using VSX
    __vector float sum_sq0 = vec_splats(0.0f);
    __vector float sum_sq1 = vec_splats(0.0f);
    __vector float sum_sq2 = vec_splats(0.0f);
    __vector float sum_sq3 = vec_splats(0.0f);

    int c = 0;
    for (; c + 15 < cols; c += 16) {
        __vector float v0 = vec_xl(0, (const float*)(x_row + c));
        __vector float v1 = vec_xl(0, (const float*)(x_row + c + 4));
        __vector float v2 = vec_xl(0, (const float*)(x_row + c + 8));
        __vector float v3 = vec_xl(0, (const float*)(x_row + c + 12));
        sum_sq0 = vec_madd(v0, v0, sum_sq0);
        sum_sq1 = vec_madd(v1, v1, sum_sq1);
        sum_sq2 = vec_madd(v2, v2, sum_sq2);
        sum_sq3 = vec_madd(v3, v3, sum_sq3);
    }
    for (; c + 3 < cols; c += 4) {
        __vector float v = vec_xl(0, (const float*)(x_row + c));
        sum_sq0 = vec_madd(v, v, sum_sq0);
    }

    // Horizontal reduction of 4 accumulators
    sum_sq0 = sum_sq0 + sum_sq1;
    sum_sq2 = sum_sq2 + sum_sq3;
    sum_sq0 = sum_sq0 + sum_sq2;
    union { __vector float vf; float f[4]; } u;
    u.vf = sum_sq0;
    float sum_sq = u.f[0] + u.f[1] + u.f[2] + u.f[3];

    // Scalar tail
    for (; c < cols; ++c) {
        float v = x_row[c];
        sum_sq += v * v;
    }

    float rms = 1.0f / std::sqrt(sum_sq / cols + eps);

    // Apply norm with optional weight
    if (w_row) {
        int c2 = 0;
        for (; c2 + 15 < cols; c2 += 16) {
            __vector float x0 = vec_xl(0, (const float*)(x_row + c2));
            __vector float x1 = vec_xl(0, (const float*)(x_row + c2 + 4));
            __vector float x2 = vec_xl(0, (const float*)(x_row + c2 + 8));
            __vector float x3 = vec_xl(0, (const float*)(x_row + c2 + 12));
            __vector float w0 = vec_xl(0, (const float*)(w_row + c2));
            __vector float w1 = vec_xl(0, (const float*)(w_row + c2 + 4));
            __vector float w2 = vec_xl(0, (const float*)(w_row + c2 + 8));
            __vector float w3 = vec_xl(0, (const float*)(w_row + c2 + 12));
            __vector float s = vec_splats(rms);
            vec_xst((x0 * s) * w0, 0, (float*)(o_row + c2));
            vec_xst((x1 * s) * w1, 0, (float*)(o_row + c2 + 4));
            vec_xst((x2 * s) * w2, 0, (float*)(o_row + c2 + 8));
            vec_xst((x3 * s) * w3, 0, (float*)(o_row + c2 + 12));
        }
        for (; c2 + 3 < cols; c2 += 4) {
            __vector float x = vec_xl(0, (const float*)(x_row + c2));
            __vector float w = vec_xl(0, (const float*)(w_row + c2));
            __vector float s = vec_splats(rms);
            vec_xst((x * s) * w, 0, (float*)(o_row + c2));
        }
        for (; c2 < cols; ++c2) {
            o_row[c2] = x_row[c2] * rms * w_row[c2];
        }
    } else {
        int c2 = 0;
        __vector float s = vec_splats(rms);
        for (; c2 + 15 < cols; c2 += 16) {
            __vector float x0 = vec_xl(0, (const float*)(x_row + c2));
            __vector float x1 = vec_xl(0, (const float*)(x_row + c2 + 4));
            __vector float x2 = vec_xl(0, (const float*)(x_row + c2 + 8));
            __vector float x3 = vec_xl(0, (const float*)(x_row + c2 + 12));
            vec_xst(x0 * s, 0, (float*)(o_row + c2));
            vec_xst(x1 * s, 0, (float*)(o_row + c2 + 4));
            vec_xst(x2 * s, 0, (float*)(o_row + c2 + 8));
            vec_xst(x3 * s, 0, (float*)(o_row + c2 + 12));
        }
        for (; c2 + 3 < cols; c2 += 4) {
            __vector float x = vec_xl(0, (const float*)(x_row + c2));
            vec_xst(x * s, 0, (float*)(o_row + c2));
        }
        for (; c2 < cols; ++c2) {
            o_row[c2] = x_row[c2] * rms;
        }
    }
}

#endif // USE_VSX

}  // namespace cpu
}  // namespace forge
