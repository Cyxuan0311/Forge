// Validation + micro-benchmark for the vectorized IQ2_XS AVX2 dot kernel.
//
// Compares the current (working tree) dot_iq2_xs_q8_K_avx2 against (a) the
// old scalar-grid version (tests/test_iq2xs_old.inc) and (b) the scalar
// reference in tests/arch/iq234nl_common.inc.
//
// Compile:
//   g++ -std=c++17 -O3 -march=native -fopenmp -DFORGE_ARCH_X86 -DUSE_AVX2 \
//       -I. -Iinclude -Isrc/operators/cpu -Isrc/operators/cpu/common \
//       tests/test_iq2xs_compare.cpp src/operators/cpu/common/quant_tables.cpp \
//       -o /tmp/test_iq2xs
//   /tmp/test_iq2xs            (validation, 4000 trials)
//   /tmp/test_iq2xs bench      (micro-benchmark of new vs old vs ref)

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <string>
#include <vector>

#include <immintrin.h>

#include "src/operators/cpu/common/quant_helpers.h"
namespace forge {
namespace cpu {
__m256 q3_k_sb_dot_avx2(const uint8_t* q3_sb, const block_q8_K* q8);
float dot_q2_K_q8_K_avx2(const uint8_t* q2_row, const block_q8_K* q8, int nb,
                         const uint8_t* scales_row);
}
}

#include "src/operators/cpu/arch/x86/quants.h"

namespace forge {
namespace ops {
#include "test_iq2xs_old.inc"
}
}

namespace ref_iq234 {
#include "arch/iq234nl_common.inc"
}

using namespace forge;

static bool approx(float got, float want) {
    float abs_err = std::fabs(got - want);
    float rel = std::fabs(want) > 1e-30f ? abs_err / std::fabs(want) : abs_err;
    return rel <= 1e-3f;
}

static double bench_dot(int nb, int iters, const uint8_t* row,
                        const cpu::block_q8_K* q8, float (*fn)(const uint8_t*, const cpu::block_q8_K*, int)) {
    // Warmup
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

    // Validation
    {
        const int nb = 4;
        std::vector<uint8_t> row((size_t)nb * 74);
        std::vector<cpu::block_q8_K> q8(nb);
        double max_nv = 0.0, max_nr = 0.0;
        int fails = 0;

        for (int trial = 0; trial < 4000; ++trial) {
            const uint32_t seed = 5000 + (uint32_t)trial;
            ref_iq234::make_iq2xs_row(row.data(), nb, seed);
            std::vector<float> act((size_t)nb * 256);
            ref_iq234::make_act(act.data(), (int)act.size(), seed + 13);
            ref_iq234::quantize_row_q8_K_local(act.data(), q8.data(), (int)act.size());

            const float newd = ops::dot_iq2_xs_q8_K_avx2(row.data(), q8.data(), nb);
            const float oldd = ops::dot_iq2_xs_q8_K_avx2_old(row.data(), q8.data(), nb);
            const float refd = ref_iq234::ref_dot_iq2_xs_q8_K(row.data(), q8.data(), nb);

            double e_nv = std::fabs((double)newd - oldd) / (std::fabs(newd) + std::fabs(oldd) + 1.0);
            double e_nr = std::fabs((double)newd - refd) / (std::fabs(newd) + std::fabs(refd) + 1.0);
            max_nv = std::max(max_nv, e_nv);
            max_nr = std::max(max_nr, e_nr);

            bool bad = !approx(newd, refd) || !approx(newd, oldd);
            if (bad) {
                if (fails < 5)
                    printf("trial %d MISMATCH: new=%f old=%f ref=%f\n", trial, newd, oldd, refd);
                ++fails;
            }
        }

        printf("trials=4000\n");
        printf("iq2xs new-vs-old max_rel=%e  new-vs-ref max_rel=%e\n", max_nv, max_nr);
        printf("fails=%d\n", fails);
        if (fails != 0) return 1;
    }

    if (bench) {
        for (int nb : {4, 8, 16}) {
            std::vector<uint8_t> row((size_t)nb * 74);
            std::vector<cpu::block_q8_K> q8(nb);
            ref_iq234::make_iq2xs_row(row.data(), nb, 12345);
            std::vector<float> act((size_t)nb * 256);
            ref_iq234::make_act(act.data(), (int)act.size(), 999);
            ref_iq234::quantize_row_q8_K_local(act.data(), q8.data(), (int)act.size());

            const int iters = 2000000 / nb;
            double t_new = bench_dot(nb, iters, row.data(), q8.data(), ops::dot_iq2_xs_q8_K_avx2);
            double t_old = bench_dot(nb, iters, row.data(), q8.data(), ops::dot_iq2_xs_q8_K_avx2_old);
            printf("nb=%3d  new=%7.2f us  old=%7.2f us  speedup=%.2fx\n",
                   nb, t_new, t_old, t_old / t_new);
        }
    }
    return 0;
}
