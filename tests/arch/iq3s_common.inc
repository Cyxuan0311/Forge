/**
 * Shared IQ3_S dot-product validation (included by per-arch wrappers).
 *
 * Generates random IQ3_S weight blocks and FP32 activations, quantizes the
 * activations to Q8_K, then checks the arch SIMD dot kernel against a scalar
 * reference that mirrors forge's dequantize_iq3_s_row semantics.
 *
 * Links against src/operators/cpu/common/quant_tables.cpp so the kernel and
 * the reference share the real iq3s_grid table.
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <limits>

#include "quant_tables.h"

static uint16_t to_fp16(float v) {
    union { float f; uint32_t u; } c; c.f = v;
    uint32_t s = (c.u >> 16) & 0x8000u;
    int32_t e = (int32_t)((c.u >> 23) & 0xff) - 127 + 15;
    uint32_t m = c.u & 0x7fffffu;
    if (e >= 31) return (uint16_t)(s | 0x7c00);
    if (e <= 0) {
        if (e < -10) return (uint16_t)s;
        m = (m | 0x800000u) >> (1 - e);
        return (uint16_t)(s | (m >> 13));
    }
    return (uint16_t)(s | ((uint32_t)e << 10) | (m >> 13));
}

static float from_fp16(uint16_t h) {
    uint32_t s = (h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1f;
    uint32_t m = h & 0x3ffu;
    if (e == 0) return (m == 0) ? 0.0f : std::ldexp((float)m / 1024.0f, -14) * (s ? -1.f : 1.f);
    if (e == 31) return (s ? -1.f : 1.f) * (m ? std::numeric_limits<float>::infinity()
                                             : std::numeric_limits<float>::infinity());
    return std::ldexp(1.0f + (float)m / 1024.0f, (int)e - 15) * (s ? -1.f : 1.f);
}

// Reference: same math as forge::ops::dequantize_iq3_s_row, accumulated as a
// dot product against a Q8_K activation row (double accumulator for precision).
static float ref_dot_iq3_s_q8_K(const uint8_t* blk, const forge::cpu::block_q8_K* q8, int nb) {
    static const uint8_t kmask[8] = {1,2,4,8,16,32,64,128};
    double sum = 0.0;
    for (int bi = 0; bi < nb; ++bi) {
        const uint8_t* b = blk + (size_t)bi * 110;
        float d = from_fp16(*reinterpret_cast<const uint16_t*>(b)) * q8[bi].d;
        const uint8_t* qs = b + 2;
        const uint8_t* qh = b + 66;
        const uint8_t* signs = b + 74;
        const uint8_t* sc = b + 106;
        const int8_t* y = q8[bi].qs;

        for (int ib32 = 0; ib32 < 8; ib32 += 2) {
            float db1 = d * (1 + 2 * (sc[ib32/2] & 0xf));
            float db2 = d * (1 + 2 * (sc[ib32/2] >> 4));
            int base = ib32 * 32;
            // group ib32 -> qh[0], group ib32+1 -> qh[1]
            for (int l = 0; l < 4; ++l) {
                const uint8_t* g1a = reinterpret_cast<const uint8_t*>(
                    &forge::ops::iq3s_grid[qs[2*l]     | ((qh[0] << (8 - 2*l)) & 256)]);
                const uint8_t* g2a = reinterpret_cast<const uint8_t*>(
                    &forge::ops::iq3s_grid[qs[2*l + 1] | ((qh[0] << (7 - 2*l)) & 256)]);
                for (int j = 0; j < 4; ++j) {
                    float s1 = (signs[l] & kmask[j]) ? -1.f : 1.f;
                    float s2 = (signs[l] & kmask[j+4]) ? -1.f : 1.f;
                    sum += (double)(db1 * g1a[j] * s1) * y[base + 8*l + j];
                    sum += (double)(db1 * g2a[j] * s2) * y[base + 8*l + j + 4];
                }
            }
            qs += 8; signs += 4;
            for (int l = 0; l < 4; ++l) {
                const uint8_t* g1b = reinterpret_cast<const uint8_t*>(
                    &forge::ops::iq3s_grid[qs[2*l]     | ((qh[1] << (8 - 2*l)) & 256)]);
                const uint8_t* g2b = reinterpret_cast<const uint8_t*>(
                    &forge::ops::iq3s_grid[qs[2*l + 1] | ((qh[1] << (7 - 2*l)) & 256)]);
                for (int j = 0; j < 4; ++j) {
                    float s1 = (signs[l] & kmask[j]) ? -1.f : 1.f;
                    float s2 = (signs[l] & kmask[j+4]) ? -1.f : 1.f;
                    sum += (double)(db2 * g1b[j] * s1) * y[(base+32) + 8*l + j];
                    sum += (double)(db2 * g2b[j] * s2) * y[(base+32) + 8*l + j + 4];
                }
            }
            qh += 2; qs += 8; signs += 4;
        }
    }
    return (float)sum;
}

static void quantize_row_q8_K_local(const float* x, forge::cpu::block_q8_K* y, int k) {
    constexpr int QK_K = 256;
    int nb = (k + QK_K - 1) / QK_K;
    for (int bi = 0; bi < nb; ++bi) {
        int base = bi * QK_K;
        int n = std::min(QK_K, k - base);
        float amax = 0;
        for (int j = 0; j < n; ++j) amax = std::max(amax, std::fabs(x[base+j]));
        float d = amax / 127.0f;
        float id = d > 0 ? 1.0f/d : 0.0f;
        y[bi].d = d;
        for (int j = 0; j < n; ++j) {
            float v = x[base+j] * id;
            y[bi].qs[j] = (int8_t)(v >= 0 ? (int)(v + 0.5f) : (int)(v - 0.5f));
        }
        for (int j = n; j < QK_K; ++j) y[bi].qs[j] = 0;
        for (int j = 0; j < 16; ++j) y[bi].bsums[j] = 0;
    }
}

// Build a random IQ3_S weight row in raw block format.
static void make_iq3s_row(uint8_t* row, int nb, uint32_t seed) {
    uint32_t r = seed;
    auto rnd = [&]() { r = r * 1664525u + 1013904223u; return r; };
    for (int bi = 0; bi < nb; ++bi) {
        uint8_t* b = row + (size_t)bi * 110;
        uint16_t d16 = to_fp16(0.01f + (float)((rnd() >> 8) % 100) / 100.0f);
        std::memcpy(b, &d16, 2);
        for (int j = 0; j < 64; ++j) b[2 + j] = (uint8_t)(rnd() % 256);
        for (int j = 0; j < 8; ++j)  b[66 + j] = (uint8_t)(rnd() % 256);
        for (int j = 0; j < 32; ++j) b[74 + j] = (uint8_t)(rnd() % 256);
        for (int j = 0; j < 4; ++j)  b[106 + j] = (uint8_t)(rnd() % 256);
    }
}
