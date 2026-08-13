/**
 * Standalone QEMU test: IQ3_S x Q8_K dot product on ARM64 NEON.
 *
 * Cross-compile:
 *   aarch64-linux-gnu-g++ -std=gnu++17 -O2 -march=armv8-a+simd -DUSE_NEON \
 *     -I src/operators/cpu/arch/arm64 -I src/operators/cpu/common \
 *     -I src -I include \
 *     tests/arch/arm64/test_iq3s_dot.cpp \
 *     src/operators/cpu/common/quant_tables.cpp -static \
 *     -o test_iq3s_dot_neon
 * Run:
 *   qemu-aarch64 ./test_iq3s_dot_neon
 */

#include <cstring>
#include "vec.h"
#include "vec_dot.h"
#include "gemv.h"
#include "gemm.h"
#include "quants.h"
#include "quant_tables.h"

#include "iq3s_common.inc"

static int errors = 0;

static void run_case(int nb, uint32_t seed) {
    const int K = nb * 256;
    std::vector<uint8_t> row((size_t)nb * 110);
    make_iq3s_row(row.data(), nb, seed);

    std::vector<float> act(K);
    for (int i = 0; i < K; ++i) act[i] = std::sin((float)i * 0.13f) * 3.0f + (float)(i % 7);

    std::vector<forge::cpu::block_q8_K> q8(nb);
    quantize_row_q8_K_local(act.data(), q8.data(), K);

    float got = forge::cpu::dot_iq3_s_q8_K_neon(row.data(), q8.data(), nb);
    float want = ref_dot_iq3_s_q8_K(row.data(), q8.data(), nb);

    float abs_err = std::fabs(got - want);
    float rel = std::fabs(want) > 1e-30f ? abs_err / std::fabs(want) : abs_err;
    if (rel > 1e-3f) {
        printf("FAIL nb=%d seed=%u got=%.6f want=%.6f abs=%.3e\n",
               nb, seed, got, want, abs_err);
        ++errors;
    } else {
        printf("ok   nb=%d seed=%u got=%.6f want=%.6f\n", nb, seed, got, want);
    }
}

int main() {
    run_case(1, 0x12345678u);
    run_case(2, 0xdeadbeefu);
    run_case(3, 0xabcdef01u);
    run_case(4, 0x0badcafeu);
    if (errors) { printf("FAILED (%d)\n", errors); return 1; }
    printf("ALL PASSED\n");
    return 0;
}
