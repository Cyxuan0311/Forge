// Regression lock for the fused AVX2 IQ kernels (IQ2_XXS / IQ3_XXS / IQ1_M),
// registered with CTest as "iq13xxs-compare" (bench mode: run with "bench" argv).
// Compares each Forge AVX2 dot kernel against a direct port of llama.cpp's
// ggml_vec_dot_iq*_q8_K_generic (ggml-cpu/quants.c L906/L1050/L1193) on the
// SAME packed weight bytes and SAME q8_K activation blocks. The integer dot
// semantics are identical, so the only allowed divergence is fp32 accumulation
// noise (tolerance 1e-4 relative; observed max ~8e-5).
//
// History: these three dtypes had no fused AVX2 q8_K kernel and fell into the
// dequant+gemv fallback (full-weight dequant per matmul call). The kernels were
// ported from llama.cpp (x86/quants.c AVX2 branches); this test pins grid-index
// pairing, sign handling, scale decode, and the IQ1_M delta/scale layout.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <string>
#include <vector>

#include <immintrin.h>

#include "src/operators/cpu/common/quant_helpers.h"
#include "src/operators/cpu/common/quant_tables.h"

namespace forge {
namespace cpu {
// Defined in matmul.cpp; declared only to satisfy quants.h references.
__m256 q3_k_sb_dot_avx2(const uint8_t* q3_sb, const block_q8_K* q8);
float dot_q2_K_q8_K_avx2(const uint8_t* q2_row, const block_q8_K* q8, int nb,
                         const uint8_t* scales_row);
}
}

#include "src/operators/cpu/arch/x86/quants.h"

using namespace forge;

// ---------------------------------------------------------------------------
// fp16 <-> fp32
// ---------------------------------------------------------------------------
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

static const uint8_t kmask_iq2xs[8] = {1,2,4,8,16,32,64,128};

// ---------------------------------------------------------------------------
// llama.cpp generic references (direct ports, ggml-cpu/quants.c)
// ---------------------------------------------------------------------------
static float ref_dot_iq2_xxs_q8_K(const uint8_t* blk, const cpu::block_q8_K* q8, int nb) {
    uint32_t aux32[2];
    const uint8_t* aux8 = (const uint8_t*)aux32;
    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const uint8_t* b = blk + (size_t)i * 66;
        const float d = from_fp16(*reinterpret_cast<const uint16_t*>(b)) * q8[i].d;
        const uint16_t* q2 = reinterpret_cast<const uint16_t*>(b + 2);
        const int8_t* q8v = q8[i].qs;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < 256/32; ++ib32) {
            memcpy(aux32, q2, 2*sizeof(uint32_t));
            q2 += 4;
            const uint32_t ls = 2*(aux32[1] >> 28) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t* grid = (const uint8_t*)(forge::ops::iq2xxs_grid + aux8[l]);
                const uint8_t signs = forge::ops::ksigns_iq2xs[(aux32[1] >> 7*l) & 127];
                for (int j = 0; j < 8; ++j)
                    sumi += grid[j] * q8v[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
                q8v += 8;
            }
            bsum += sumi * ls;
        }
        sumf += d * bsum;
    }
    return 0.125f * sumf;
}

static float ref_dot_iq3_xxs_q8_K(const uint8_t* blk, const cpu::block_q8_K* q8, int nb) {
    uint32_t aux32;
    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const uint8_t* b = blk + (size_t)i * 98;
        const float d = from_fp16(*reinterpret_cast<const uint16_t*>(b)) * q8[i].d;
        const uint8_t* q3 = b + 2;
        const uint8_t* gas = q3 + 256/4;
        const int8_t* q8v = q8[i].qs;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < 256/32; ++ib32) {
            memcpy(&aux32, gas, sizeof(uint32_t)); gas += sizeof(uint32_t);
            const uint32_t ls = 2*(aux32 >> 28) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t* grid1 = (const uint8_t*)(forge::ops::iq3xxs_grid + q3[2*l+0]);
                const uint8_t* grid2 = (const uint8_t*)(forge::ops::iq3xxs_grid + q3[2*l+1]);
                const uint8_t signs = forge::ops::ksigns_iq2xs[(aux32 >> 7*l) & 127];
                for (int j = 0; j < 4; ++j) {
                    sumi += grid1[j] * q8v[j+0] * (signs & kmask_iq2xs[j+0] ? -1 : 1);
                    sumi += grid2[j] * q8v[j+4] * (signs & kmask_iq2xs[j+4] ? -1 : 1);
                }
                q8v += 8;
            }
            q3 += 8;
            bsum += sumi * ls;
        }
        sumf += d * bsum;
    }
    return 0.25f * sumf;
}

static float ref_dot_iq1_m_q8_K(const uint8_t* blk, const cpu::block_q8_K* q8, int nb) {
    double sumf = 0;
    for (int i = 0; i < nb; i++) {
        const uint8_t* b = blk + (size_t)i * 56;
        const int8_t* q8v = q8[i].qs;
        const uint8_t* qs = b;
        const uint8_t* qh = b + 32;
        const uint16_t* sc = reinterpret_cast<const uint16_t*>(b + 48);

        const uint16_t scale_bits = (uint16_t)((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
                                               ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
        int sum1[2], sum2[2], delta[4];
        int sumi1 = 0, sumi2 = 0;
        for (int ib = 0; ib < 256/32; ++ib) {
            delta[0] = qh[0] & 0x08 ? -1 : 1;
            delta[1] = qh[0] & 0x80 ? -1 : 1;
            delta[2] = qh[1] & 0x08 ? -1 : 1;
            delta[3] = qh[1] & 0x80 ? -1 : 1;
            sum1[0] = sum1[1] = sum2[0] = sum2[1] = 0;
            for (int l = 0; l < 4; ++l) {
                const int8_t* grid = (const int8_t*)(forge::ops::iq1s_grid +
                    (qs[l] | (((uint16_t)qh[l/2] << (8 - 4*(l%2))) & 0x700)));
                int lsum1 = 0, lsum2 = 0;
                for (int j = 0; j < 8; ++j) {
                    lsum1 += q8v[j] * grid[j];
                    lsum2 += q8v[j];
                }
                q8v += 8;
                sum1[l/2] += lsum1;
                sum2[l/2] += lsum2*delta[l];
            }
            const int ls1 = 2*((sc[ib/2] >> (6*(ib%2)+0)) & 0x7) + 1;
            const int ls2 = 2*((sc[ib/2] >> (6*(ib%2)+3)) & 0x7) + 1;
            sumi1 += sum1[0] * ls1 + sum1[1] * ls2;
            sumi2 += sum2[0] * ls1 + sum2[1] * ls2;
            qs += 4;
            qh += 2;
        }
        sumf += (double)from_fp16(scale_bits) * q8[i].d * (sumi1 + 0.125 * sumi2);
    }
    return (float)sumf;
}

// ---------------------------------------------------------------------------
// Random valid packed-row generators
// ---------------------------------------------------------------------------
static uint32_t rnd_next(uint32_t& r) { r = r * 1664525u + 1013904223u; return r; }

static void make_iq2xxs_row(uint8_t* row, int nb, uint32_t seed) {
    uint32_t r = seed;
    for (int bi = 0; bi < nb; ++bi) {
        uint8_t* b = row + (size_t)bi * 66;
        uint16_t d16 = to_fp16(0.01f + (float)((rnd_next(r) >> 8) % 100) / 100.0f);
        memcpy(b, &d16, 2);
        for (int j = 0; j < 64; ++j) b[2 + j] = (uint8_t)(rnd_next(r) % 256);
        // top nibble of the 2nd uint32 = scale (valid 0..15), rest random signs
    }
}
static void make_iq3xxs_row(uint8_t* row, int nb, uint32_t seed) {
    uint32_t r = seed;
    for (int bi = 0; bi < nb; ++bi) {
        uint8_t* b = row + (size_t)bi * 98;
        uint16_t d16 = to_fp16(0.01f + (float)((rnd_next(r) >> 8) % 100) / 100.0f);
        memcpy(b, &d16, 2);
        for (int j = 0; j < 64; ++j) b[2 + j] = (uint8_t)(rnd_next(r) % 256);   // grid idx
        for (int j = 0; j < 32; ++j) b[66 + j] = (uint8_t)(rnd_next(r) % 256);  // signs+scale nibble
    }
}
static void make_iq1m_row(uint8_t* row, int nb, uint32_t seed) {
    uint32_t r = seed;
    for (int bi = 0; bi < nb; ++bi) {
        uint8_t* b = row + (size_t)bi * 56;
        for (int j = 0; j < 32; ++j) b[j] = (uint8_t)(rnd_next(r) % 256);       // qs
        for (int j = 0; j < 16; ++j) b[32 + j] = (uint8_t)(rnd_next(r) % 256);  // qh
        // scales: pack a sane fp16 super-scale into the top nibbles (llama.cpp
        // iq1m_scale_t layout) so d is a normal value, keep low bits random.
        const uint16_t target = to_fp16(0.05f + (float)((rnd_next(r) >> 8) % 200) / 100.0f);
        uint16_t sc[4];
        sc[0] = (uint16_t)((target & 0x000f) << 12) | (rnd_next(r) & 0x0fff);
        sc[1] = (uint16_t)((target & 0x00f0) << 8)  | (rnd_next(r) & 0x0fff);
        sc[2] = (uint16_t)((target & 0x0f00) << 4)  | (rnd_next(r) & 0x0fff);
        sc[3] = (uint16_t)((target & 0xf000))       | (rnd_next(r) & 0x0fff);
        memcpy(b + 48, sc, 8);
    }
}
static void make_act(float* x, int n, uint32_t seed) {
    uint32_t r = seed;
    for (int i = 0; i < n; ++i)
        x[i] = (float)((int)(rnd_next(r) % 2000) - 1000) / 1000.0f;
}

static double rel_err(double a, double b) {
    return std::fabs(a - b) / (std::fabs(a) + std::fabs(b) + 1.0);
}

static int run_valid(const char* name, size_t blk,
                     void (*mk)(uint8_t*, int, uint32_t),
                     float (*avx2)(const uint8_t*, const cpu::block_q8_K*, int),
                     float (*ref)(const uint8_t*, const cpu::block_q8_K*, int),
                     int nb) {
    std::vector<uint8_t> row((size_t)nb * blk);
    std::vector<cpu::block_q8_K> q8(nb);
    double max_err = 0.0;
    int fails = 0;
    for (int trial = 0; trial < 4000; ++trial) {
        const uint32_t seed = 7000 + (uint32_t)trial;
        mk(row.data(), nb, seed);
        std::vector<float> act((size_t)nb * 256);
        make_act(act.data(), (int)act.size(), seed + 13);
        cpu::quantize_row_q8_K(act.data(), q8.data(), (int)act.size());

        const float got = avx2(row.data(), q8.data(), nb);
        const float refv = ref(row.data(), q8.data(), nb);
        const double e = rel_err((double)got, (double)refv);
        max_err = std::max(max_err, e);
        if (e > 1e-4) {
            if (fails < 5)
                printf("%s trial %d MISMATCH: got=%.6f ref=%.6f rel=%g\n", name, trial, got, refv, e);
            ++fails;
        }
    }
    printf("%-8s nb=%d  max_rel=%e  fails=%d\n", name, nb, max_err, fails);
    return fails;
}

static double bench_dot(int nb, int iters, const uint8_t* row,
                        const cpu::block_q8_K* q8,
                        float (*fn)(const uint8_t*, const cpu::block_q8_K*, int)) {
    volatile float sink = 0;
    for (int i = 0; i < 200; ++i) sink += fn(row, q8, nb);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) sink += fn(row, q8, nb);
    auto t1 = std::chrono::high_resolution_clock::now();
    (void)sink;
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
}

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "bench";
    int fails = 0;

    for (int nb : {4, 6}) {
        fails += run_valid("IQ2_XXS", 66, make_iq2xxs_row, ops::dot_iq2_xxs_q8_K_avx2,
                           ref_dot_iq2_xxs_q8_K, nb);
        fails += run_valid("IQ3_XXS", 98, make_iq3xxs_row, ops::dot_iq3_xxs_q8_K_avx2,
                           ref_dot_iq3_xxs_q8_K, nb);
        fails += run_valid("IQ1_M", 56, make_iq1m_row, ops::dot_iq1_m_q8_K_avx2,
                           ref_dot_iq1_m_q8_K, nb);
    }

    if (bench) {
        const int nb = 6;  // K = 1536
        std::vector<uint8_t> r2((size_t)nb * 66), r3((size_t)nb * 98), r1((size_t)nb * 56);
        make_iq2xxs_row(r2.data(), nb, 12345);
        make_iq3xxs_row(r3.data(), nb, 12345);
        make_iq1m_row(r1.data(), nb, 12345);
        std::vector<float> act((size_t)nb * 256);
        make_act(act.data(), (int)act.size(), 999);
        std::vector<cpu::block_q8_K> q8(nb);
        cpu::quantize_row_q8_K(act.data(), q8.data(), (int)act.size());

        const int iters = 300000;
        printf("nb=%d (K=1536)  iq2_xxs=%6.3f us  iq3_xxs=%6.3f us  iq1_m=%6.3f us\n",
               nb,
               bench_dot(nb, iters, r2.data(), q8.data(), ops::dot_iq2_xxs_q8_K_avx2),
               bench_dot(nb, iters, r3.data(), q8.data(), ops::dot_iq3_xxs_q8_K_avx2),
               bench_dot(nb, iters, r1.data(), q8.data(), ops::dot_iq1_m_q8_K_avx2));
    }
    return fails != 0;
}
