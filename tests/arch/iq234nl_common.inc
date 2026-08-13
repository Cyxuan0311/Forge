/**
 * Shared IQ2_S / IQ2_XS / IQ4_NL dot-product validation
 * (included by per-arch wrappers).
 *
 * Generates random quantized weight blocks and FP32 activations, quantizes
 * activations to Q8_K, and checks the arch SIMD dot kernels against scalar
 * references that mirror forge's dequantize_*_row semantics.
 *
 * Links against src/operators/cpu/common/quant_tables.cpp so kernels and
 * references share the real grid/LUT tables.
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
    if (e == 31) return (s ? -1.f : 1.f) * std::numeric_limits<float>::infinity();
    return std::ldexp(1.0f + (float)m / 1024.0f, (int)e - 15) * (s ? -1.f : 1.f);
}

static const uint8_t kmask_iq2[8] = {1,2,4,8,16,32,64,128};

// ---- Reference: IQ2_S (82-byte blocks) ----
static float ref_dot_iq2_s_q8_K(const uint8_t* blk, const forge::cpu::block_q8_K* q8, int nb) {
    double sum = 0.0;
    for (int bi = 0; bi < nb; ++bi) {
        const uint8_t* b = blk + (size_t)bi * 82;
        float d = from_fp16(*reinterpret_cast<const uint16_t*>(b)) * q8[bi].d;
        const uint8_t* qs    = b + 2;
        const uint8_t* signs = qs + 32;
        const uint8_t* qh    = b + 66;
        const uint8_t* sc    = b + 74;
        const int8_t* y      = q8[bi].qs;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            float db0 = d * (0.5f + (sc[ib32] & 0xf)) * 0.25f;
            float db1 = d * (0.5f + (sc[ib32] >> 4)) * 0.25f;
            int base = ib32 * 32;
            for (int l = 0; l < 4; ++l) {
                int grid_idx = qs[l] | ((qh[ib32] << (8 - 2 * l)) & 0x300);
                const uint8_t* g = reinterpret_cast<const uint8_t*>(&forge::ops::iq2s_grid[grid_idx]);
                float db = (l < 2) ? db0 : db1;
                for (int j = 0; j < 8; ++j) {
                    float s = (signs[l] & kmask_iq2[j]) ? -1.f : 1.f;
                    sum += (double)(db * g[j] * s) * y[base + 8 * l + j];
                }
            }
            qs += 4;
            signs += 4;
        }
    }
    return (float)sum;
}

// ---- Reference: IQ2_XS (74-byte blocks) ----
static float ref_dot_iq2_xs_q8_K(const uint8_t* blk, const forge::cpu::block_q8_K* q8, int nb) {
    double sum = 0.0;
    for (int bi = 0; bi < nb; ++bi) {
        const uint8_t* b = blk + (size_t)bi * 74;
        float d = from_fp16(*reinterpret_cast<const uint16_t*>(b)) * q8[bi].d;
        const uint16_t* q2 = reinterpret_cast<const uint16_t*>(b + 2);
        const uint8_t* sc  = b + 66;
        const int8_t* y    = q8[bi].qs;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            float db0 = d * (0.5f + (sc[ib32] & 0xf)) * 0.25f;
            float db1 = d * (0.5f + (sc[ib32] >> 4)) * 0.25f;
            int base = ib32 * 32;
            for (int l = 0; l < 4; ++l) {
                const uint16_t v = q2[4 * ib32 + l];
                const uint8_t* g = reinterpret_cast<const uint8_t*>(&forge::ops::iq2xs_grid[v & 511]);
                const uint8_t sb = forge::ops::ksigns_iq2xs[v >> 9];
                float db = (l < 2) ? db0 : db1;
                for (int j = 0; j < 8; ++j) {
                    float s = (sb & kmask_iq2[j]) ? -1.f : 1.f;
                    sum += (double)(db * g[j] * s) * y[base + 8 * l + j];
                }
            }
        }
    }
    return (float)sum;
}

// ---- Reference: IQ4_NL (18-byte sub-blocks of 32 elements) ----
static float ref_dot_iq4_nl_q8_K(const uint8_t* blk, const forge::cpu::block_q8_K* q8, int nb) {
    double sum = 0.0;
    for (int bi = 0; bi < nb; ++bi) {
        const int8_t* y = q8[bi].qs;
        const uint8_t* b = blk + (size_t)bi * (256 / 32) * 18;
        for (int sb = 0; sb < 8; ++sb) {
            float d = from_fp16(*reinterpret_cast<const uint16_t*>(b)) * q8[bi].d;
            const uint8_t* qs = b + 2;
            int base = sb * 32;
            for (int i = 0; i < 16; ++i) {
                sum += (double)(d * forge::ops::kvalues_iq4nl[qs[i] & 0x0F]) * y[base + i];
                sum += (double)(d * forge::ops::kvalues_iq4nl[(qs[i] >> 4) & 0x0F]) * y[base + 16 + i];
            }
            b += 18;
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

// Random FP32 activation rows.
static void make_act(float* x, int n, uint32_t seed) {
    uint32_t r = seed;
    auto rnd = [&]() { r = r * 1664525u + 1013904223u; return r; };
    for (int i = 0; i < n; ++i) {
        float v = (float)((int)(rnd() % 2000) - 1000) / 1000.0f;
        x[i] = v;
    }
}

// Build a random IQ2_S weight row (82-byte blocks).
static void make_iq2s_row(uint8_t* row, int nb, uint32_t seed) {
    uint32_t r = seed;
    auto rnd = [&]() { r = r * 1664525u + 1013904223u; return r; };
    for (int bi = 0; bi < nb; ++bi) {
        uint8_t* b = row + (size_t)bi * 82;
        uint16_t d16 = to_fp16(0.01f + (float)((rnd() >> 8) % 100) / 100.0f);
        std::memcpy(b, &d16, 2);
        for (int j = 0; j < 32; ++j) b[2 + j]  = (uint8_t)(rnd() % 256);
        for (int j = 0; j < 32; ++j) b[34 + j] = (uint8_t)(rnd() % 256);
        for (int j = 0; j < 8; ++j)  b[66 + j] = (uint8_t)(rnd() % 256);
        for (int j = 0; j < 8; ++j)  b[74 + j] = (uint8_t)(rnd() % 256);
    }
}

// Build a random IQ2_XS weight row (74-byte blocks).
static void make_iq2xs_row(uint8_t* row, int nb, uint32_t seed) {
    uint32_t r = seed;
    auto rnd = [&]() { r = r * 1664525u + 1013904223u; return r; };
    for (int bi = 0; bi < nb; ++bi) {
        uint8_t* b = row + (size_t)bi * 74;
        uint16_t d16 = to_fp16(0.01f + (float)((rnd() >> 8) % 100) / 100.0f);
        std::memcpy(b, &d16, 2);
        for (int j = 0; j < 32; ++j) {
            uint16_t v = (uint16_t)(rnd() % 65536);
            std::memcpy(b + 2 + 2 * j, &v, 2);
        }
        for (int j = 0; j < 8; ++j) b[66 + j] = (uint8_t)(rnd() % 256);
    }
}

// Build a random IQ4_NL weight row (18-byte sub-blocks).
static void make_iq4nl_row(uint8_t* row, int nb, uint32_t seed) {
    uint32_t r = seed;
    auto rnd = [&]() { r = r * 1664525u + 1013904223u; return r; };
    for (int bi = 0; bi < nb; ++bi) {
        for (int sb = 0; sb < 8; ++sb) {
            uint8_t* b = row + ((size_t)bi * 8 + sb) * 18;
            uint16_t d16 = to_fp16(0.01f + (float)((rnd() >> 8) % 100) / 100.0f);
            std::memcpy(b, &d16, 2);
            for (int j = 0; j < 16; ++j) b[2 + j] = (uint8_t)(rnd() % 256);
        }
    }
}
