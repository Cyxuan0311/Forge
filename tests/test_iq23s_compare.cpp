// Validation for the vectorized IQ2_S / IQ3_S AVX2 dot kernels.
//
// Compares the new (working tree) kernels from
// src/operators/cpu/arch/x86/quants.h against (a) the old scalar-grid kernels
// from git HEAD (tests/test_iq23s_old.inc) and (b) the scalar reference
// implementations in tests/arch/iq234nl_common.inc and tests/arch/iq3s_common.inc.
//
// Compile:
//   g++ -std=c++17 -O2 -march=native -fopenmp -DFORGE_ARCH_X86 -DUSE_AVX2 \
//       -I. -Iinclude -Isrc/operators/cpu -Isrc/operators/cpu/common \
//       tests/test_iq23s_compare.cpp src/operators/cpu/common/quant_tables.cpp \
//       -o /tmp/test_iq23s
//   /tmp/test_iq23s

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

#include <immintrin.h>

// Provide forge::cpu::block_q8_K first (quants.h references it), then
// forward-declare the Q2_K/Q3_K kernels that quants.h's unused GEMV helpers
// call (defined in cpu_gemv.h).  Never invoked by this harness.
#include "src/operators/cpu/common/quant_helpers.h"
namespace forge {
namespace cpu {
__m256 q3_k_sb_dot_avx2(const uint8_t* q3_sb, const block_q8_K* q8);
float dot_q2_K_q8_K_avx2(const uint8_t* q2_row, const block_q8_K* q8, int nb,
                         const uint8_t* scales_row);
}
}

// Real headers: provide fp16_to_float_scalar, hsum_avx2, and the forge::ops
// grid/LUT table externs.
#include "src/operators/cpu/arch/x86/quants.h"

// Old kernels (git HEAD), renamed *_old, inside forge::ops.
namespace forge {
namespace ops {
#include "test_iq23s_old.inc"
}
}

// Scalar references (namespace-wrapped to avoid static-name collisions).
namespace ref_iq234 {
#include "arch/iq234nl_common.inc"
}
namespace ref_iq3s {
#include "arch/iq3s_common.inc"
}

using namespace forge;

// Relative tolerance against the reference magnitude, matching the project's
// arch test convention (tests/arch/*: rel > 1e-3 is a mismatch).
static bool approx(float got, float want) {
    float abs_err = std::fabs(got - want);
    float rel = std::fabs(want) > 1e-30f ? abs_err / std::fabs(want) : abs_err;
    return rel <= 1e-3f;
}

int main() {
    const int nb = 4;   // 1024 elements per dot
    std::vector<uint8_t> row2s((size_t)nb * 82);
    std::vector<uint8_t> row3s((size_t)nb * 110);
    std::vector<cpu::block_q8_K> q8(nb);

    double max_nv2 = 0.0, max_nr2 = 0.0, max_nv3 = 0.0, max_nr3 = 0.0;
    int fails = 0;

    for (int trial = 0; trial < 4000; ++trial) {
        const uint32_t seed = 1000 + (uint32_t)trial;
        ref_iq234::make_iq2s_row(row2s.data(), nb, seed);
        ref_iq3s::make_iq3s_row(row3s.data(), nb, seed + 7);

        std::vector<float> act((size_t)nb * 256);
        ref_iq234::make_act(act.data(), (int)act.size(), seed + 13);
        ref_iq234::quantize_row_q8_K_local(act.data(), q8.data(), (int)act.size());

        const float new2 = ops::dot_iq2_s_q8_K_avx2(row2s.data(), q8.data(), nb);
        const float old2 = ops::dot_iq2_s_q8_K_avx2_old(row2s.data(), q8.data(), nb);
        const float ref2 = ref_iq234::ref_dot_iq2_s_q8_K(row2s.data(), q8.data(), nb);
        const float new3 = ops::dot_iq3_s_q8_K_avx2(row3s.data(), q8.data(), nb);
        const float old3 = ops::dot_iq3_s_q8_K_avx2_old(row3s.data(), q8.data(), nb);
        const float ref3 = ref_iq3s::ref_dot_iq3_s_q8_K(row3s.data(), q8.data(), nb);

        double e_nv2 = std::fabs((double)new2 - old2) / (std::fabs(new2) + std::fabs(old2) + 1.0);
        double e_nr2 = std::fabs((double)new2 - ref2) / (std::fabs(new2) + std::fabs(ref2) + 1.0);
        double e_nv3 = std::fabs((double)new3 - old3) / (std::fabs(new3) + std::fabs(old3) + 1.0);
        double e_nr3 = std::fabs((double)new3 - ref3) / (std::fabs(new3) + std::fabs(ref3) + 1.0);

        max_nv2 = std::max(max_nv2, e_nv2);
        max_nr2 = std::max(max_nr2, e_nr2);
        max_nv3 = std::max(max_nv3, e_nv3);
        max_nr3 = std::max(max_nr3, e_nr3);

        bool bad = !approx(new2, ref2) || !approx(new3, ref3);
        if (bad) {
            if (fails < 5)
                printf("trial %d MISMATCH: iq2s new=%f old=%f ref=%f | iq3s new=%f old=%f ref=%f\n",
                       trial, new2, old2, ref2, new3, old3, ref3);
            ++fails;
        }
    }

    printf("trials=4000\n");
    printf("iq2s  new-vs-old max_rel=%e  new-vs-ref max_rel=%e\n", max_nv2, max_nr2);
    printf("iq3s  new-vs-old max_rel=%e  new-vs-ref max_rel=%e\n", max_nv3, max_nr3);
    printf("fails=%d\n", fails);
    return fails == 0 ? 0 : 1;
}
