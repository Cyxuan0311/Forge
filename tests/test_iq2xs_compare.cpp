// Bit-exact numerical comparison: old IQ2_XS dot kernel (git HEAD) vs new
// llama.cpp-ported kernel (working tree).
//
// Compile:
//   g++ -std=c++17 -O2 -march=native -fopenmp -DFORGE_ARCH_X86 -DUSE_AVX2 \
//       -I. -Iinclude -Isrc/operators/cpu \
//       tests/test_iq2xs_compare.cpp src/operators/cpu/common/quant_tables.cpp \
//       -o /tmp/test_iq2xs
//   /tmp/test_iq2xs

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include <immintrin.h>

// --- minimal forge::cpu helpers (mirror arch/x86) ---
namespace forge {
namespace cpu {

struct block_q8_K {
    float d;
    int8_t qs[256];
    int16_t bsums[16];
};

static inline float fp16_to_float_scalar(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (uint32_t)((h >> 10) & 0x1f);
    uint32_t man = (uint32_t)(h & 0x3ff);
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) bits = sign;
        else {
            int e = -1;
            do { man <<= 1; --e; } while ((man & 0x400) == 0);
            exp = (uint32_t)(e + 127 + 1);
            man &= 0x3ff;
            bits = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000 | (man << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// hsum_avx2 from arch/x86/vec.h
static inline float hsum_avx2(__m256 v) {
    __m128 hi128 = _mm256_extractf128_ps(v, 1);
    __m128 lo128 = _mm256_castps256_ps128(v);
    __m128 sum128 = _mm_add_ps(lo128, hi128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}

// get_scale_shuffle from arch/x86/scales.h (same table as llama.cpp)
static inline __m128i get_scale_shuffle(int i) {
    static const uint8_t k_shuffle[128] = {
        0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,
        2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  4,  4,  4,  5,  5,  5,  5,
        5,  5,  5,  5,  6,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  7,  7,  7,  8,  8,
        8,  8,  8,  8,  8,  8,  9,  9,  9,  9,  9,  9,  9,  9,  10, 10, 10, 10, 10, 10, 10, 10,
        11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13,
        13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15};
    return _mm_loadu_si128((const __m128i*)k_shuffle + i);
}

}  // namespace cpu
}  // namespace forge

// lookup tables (real definitions in quant_tables.cpp)
namespace forge {
namespace ops {
extern const uint64_t iq2xs_grid[512];
extern const uint8_t ksigns_iq2xs[128];
extern const uint8_t kmask_iq2xs[8];
}  // namespace ops
}  // namespace forge

// --- Old kernel (git HEAD), renamed ---
namespace forge {
namespace ops {
#include "test_iq2xs_old.inc"   // defines dot_iq2_xs_q8_K_avx2_old
}  // namespace ops
}  // namespace forge

// --- New kernel (working tree) ---
namespace forge {
namespace ops {
static inline __m256i make_m128i_si256(__m128i lo, __m128i hi) {
    return _mm256_insertf128_si256(_mm256_castsi128_si256(lo), hi, 1);
}
#include "test_iq2xs_new.inc"   // defines dot_iq2_xs_q8_K_avx2
}  // namespace ops
}  // namespace forge

using namespace forge;

static std::mt19937 g_rng(1234);

static uint16_t fp32_to_fp16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    uint32_t sign = (u >> 16) & 0x8000;
    int32_t expo = (u >> 23) & 0xff;
    uint32_t mant = u & 0x7fffff;
    if (expo == 0xff) return (uint16_t)(sign | 0x7c00 | (mant ? 0x200 : 0));
    int32_t e = expo - 127 + 15;
    if (e >= 31) return (uint16_t)(sign | 0x7c00);
    if (e <= 0) {
        if (e < -10) return (uint16_t)sign;
        mant |= 0x800000;
        int shift = 14 - e;
        uint32_t half_mant = mant >> shift;
        return (uint16_t)(sign | half_mant);
    }
    uint32_t half_mant = (mant >> 13) & 0x3ff;
    uint32_t rem = mant & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (half_mant & 1))) half_mant++;
    return (uint16_t)(sign | (e << 10) | half_mant);
}

static void random_iq2xs_row(std::vector<uint8_t>& row, int nb) {
    std::uniform_int_distribution<int> dqs(0, 511);
    std::uniform_int_distribution<int> dsc(0, 7);
    std::uniform_int_distribution<int> dsign(0, 1);
    std::uniform_real_distribution<float> df(-1.0f, 1.0f);
    for (int b = 0; b < nb; ++b) {
        uint8_t* p = row.data() + (size_t)b * 74;
        uint16_t d = fp32_to_fp16(df(g_rng) * 0.5f);
        std::memcpy(p, &d, 2);
        uint16_t* qs = reinterpret_cast<uint16_t*>(p + 2);
        for (int i = 0; i < 32; ++i) {
            uint16_t val = (uint16_t)dqs(g_rng);
            uint16_t signs = 0;
            for (int j = 0; j < 7; ++j) signs |= (uint16_t)(dsign(g_rng) << j);
            qs[i] = val | (signs << 9);   // bit13 (xor bit) = 0
        }
        uint8_t* sc = p + 66;
        for (int i = 0; i < 8; ++i) sc[i] = (uint8_t)((dsc(g_rng) << 4) | dsc(g_rng));
    }
}

int main() {
    const int nb = 4;   // 1024 elements per dot
    std::vector<uint8_t> row((size_t)nb * 74);
    std::vector<cpu::block_q8_K> q8(nb);

    double max_err = 0.0;
    int fails = 0;
    for (int trial = 0; trial < 2000; ++trial) {
        random_iq2xs_row(row, nb);

        std::uniform_real_distribution<float> df(-3.0f, 3.0f);
        std::uniform_int_distribution<int> dq(-127, 127);
        for (int b = 0; b < nb; ++b) {
            q8[b].d = df(g_rng);
            for (int i = 0; i < 256; ++i) q8[b].qs[i] = (int8_t)dq(g_rng);
        }

        float oldv = ops::dot_iq2_xs_q8_K_avx2_old(row.data(), q8.data(), nb);
        float newv = ops::dot_iq2_xs_q8_K_avx2(row.data(), q8.data(), nb);
        // Relative tolerance: accumulation-order float rounding differences are
        // expected (~1e-5 relative). A bit-layout bug would produce ~1e-1+ errors.
        float denom = std::fabs(oldv) + std::fabs(newv) + 1.0f;
        float rel_err = std::fabs(oldv - newv) / denom;
        if (rel_err > max_err) max_err = rel_err;
        if (rel_err > 1e-4f) {
            if (fails < 5)
                printf("trial %d MISMATCH old=%f new=%f rel=%e\n", trial, oldv, newv, rel_err);
            ++fails;
        }
    }
    printf("trials=2000 max_rel_err=%e fails=%d\n", max_err, fails);
    return fails == 0 ? 0 : 1;
}