/**
 * Standalone QEMU test: IQ2_S / IQ2_XS / IQ4_NL x Q8_K dot products on PPC64 VSX.
 *
 * Cross-compile:
 *   powerpc64le-linux-gnu-g++ -std=gnu++17 -O2 -mcpu=power8 -DUSE_VSX \
 *     -I src/operators/cpu/arch/ppc64 -I src/operators/cpu/common \
 *     -I src -I include \
 *     tests/arch/ppc64/test_iq234nl_dot.cpp \
 *     src/operators/cpu/common/quant_tables.cpp -static \
 *     -o test_iq234nl_dot_vsx
 * Run:
 *   qemu-ppc64le ./test_iq234nl_dot_vsx
 */

#include "quants.h"
#include "quant_tables.h"

#include "iq234nl_common.inc"

static int errors = 0;

template <typename Fn>
static void run_dot(const char* name, Fn fn, const uint8_t* row,
                    const std::vector<forge::cpu::block_q8_K>& q8, int nb,
                    float want, int kcase, uint32_t seed) {
    float got = fn(row, q8.data(), nb);
    float abs_err = std::fabs(got - want);
    float rel = std::fabs(want) > 1e-30f ? abs_err / std::fabs(want) : abs_err;
    if (rel > 1e-3f) {
        printf("FAIL %s nb=%d seed=%u got=%.6f want=%.6f abs=%.3e\n",
               name, nb, seed, got, want, abs_err);
        ++errors;
    } else {
        printf("ok   %s nb=%d seed=%u got=%.6f want=%.6f\n", name, nb, seed, got, want);
    }
}

static void run_iq2_s(int nb, uint32_t seed) {
    const int K = nb * 256;
    std::vector<uint8_t> row((size_t)nb * 82);
    make_iq2s_row(row.data(), nb, seed);

    std::vector<float> act(K);
    make_act(act.data(), K, seed ^ 0x55aa55aa);

    std::vector<forge::cpu::block_q8_K> q8(nb);
    quantize_row_q8_K_local(act.data(), q8.data(), K);

    float want = ref_dot_iq2_s_q8_K(row.data(), q8.data(), nb);
    run_dot("iq2_s", forge::cpu::dot_iq2_s_q8_K_vsx, row.data(), q8, nb, want, 1, seed);
}

static void run_iq2_xs(int nb, uint32_t seed) {
    const int K = nb * 256;
    std::vector<uint8_t> row((size_t)nb * 74);
    make_iq2xs_row(row.data(), nb, seed);

    std::vector<float> act(K);
    make_act(act.data(), K, seed ^ 0x0badcafe);

    std::vector<forge::cpu::block_q8_K> q8(nb);
    quantize_row_q8_K_local(act.data(), q8.data(), K);

    float want = ref_dot_iq2_xs_q8_K(row.data(), q8.data(), nb);
    run_dot("iq2_xs", forge::cpu::dot_iq2_xs_q8_K_vsx, row.data(), q8, nb, want, 2, seed);
}

static void run_iq4_nl(int nb, uint32_t seed) {
    const int K = nb * 256;
    std::vector<uint8_t> row((size_t)nb * 8 * 18);
    make_iq4nl_row(row.data(), nb, seed);

    std::vector<float> act(K);
    make_act(act.data(), K, seed ^ 0xdeadbeef);

    std::vector<forge::cpu::block_q8_K> q8(nb);
    quantize_row_q8_K_local(act.data(), q8.data(), K);

    float want = ref_dot_iq4_nl_q8_K(row.data(), q8.data(), nb);
    run_dot("iq4_nl", forge::cpu::dot_iq4_nl_q8_K_vsx, row.data(), q8, nb, want, 3, seed);
}

int main() {
    for (int nb = 1; nb <= 3; ++nb) {
        run_iq2_s(nb, 0x12345678u + nb);
        run_iq2_xs(nb, 0xabcdef01u + nb);
        run_iq4_nl(nb, 0x0badcafeu + nb);
    }
    if (errors) { printf("FAILED (%d)\n", errors); return 1; }
    printf("ALL PASSED\n");
    return 0;
}
