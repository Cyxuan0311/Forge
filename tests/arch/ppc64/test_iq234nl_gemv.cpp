/**
 * Standalone QEMU test: IQ2_S / IQ2_XS / IQ4_NL GEMV (transB) on PPC64 VSX,
 * including the M>1 path.
 *
 * Cross-compile:
 *   powerpc64le-linux-gnu-g++ -std=gnu++17 -O2 -mcpu=power8 -DUSE_VSX \
 *     -I src/operators/cpu/arch/ppc64 -I src/operators/cpu/common \
 *     -I src -I include -I tests/arch \
 *     tests/arch/ppc64/test_iq234nl_gemv.cpp \
 *     src/operators/cpu/common/quant_tables.cpp -static \
 *     -o test_iq234nl_gemv_vsx
 * Run:
 *   qemu-ppc64le ./test_iq234nl_gemv_vsx
 */

#include "quants.h"
#include "quant_tables.h"

#include "iq234nl_common.inc"

static int errors = 0;

// Scalar reference: out[m][n] = dot(row_n, q8(m))
template <typename RefFn>
static void ref_gemv(RefFn dot, const uint8_t* w, int block_size,
                     const std::vector<forge::cpu::block_q8_K>& q8_all,
                     int M, int K, int N, float* out) {
    const int nb = (K + 255) / 256;
    for (int n = 0; n < N; ++n) {
        const uint8_t* row = w + (size_t)n * nb * block_size;
        for (int m = 0; m < M; ++m)
            out[(size_t)m * N + n] = dot(row, q8_all.data() + (size_t)m * nb, nb);
    }
}

static void check(const char* name, int M, int K, int N, const float* got, const float* want) {
    for (int i = 0; i < M * N; ++i) {
        float abs_err = std::fabs(got[i] - want[i]);
        float rel = std::fabs(want[i]) > 1e-30f ? abs_err / std::fabs(want[i]) : abs_err;
        if (rel > 1e-3f) {
            printf("FAIL %s M=%d K=%d N=%d i=%d got=%.6f want=%.6f\n",
                   name, M, K, N, i, got[i], want[i]);
            ++errors;
            return;
        }
    }
    printf("ok   %s M=%d K=%d N=%d\n", name, M, K, N);
}

int main() {
    const int K = 256, N = 5, M = 3;

    // ---- IQ2_S ----
    {
        std::vector<uint8_t> w((size_t)N * 82);
        make_iq2s_row(w.data(), 1, 0x1111u);
        std::vector<float> act(M * K);
        for (int m = 0; m < M; ++m) make_act(act.data() + (size_t)m * K, K, 0x2222u + m);
        std::vector<forge::cpu::block_q8_K> q8_all((size_t)M);
        for (int m = 0; m < M; ++m) quantize_row_q8_K_local(act.data() + (size_t)m * K, &q8_all[m], K);

        std::vector<float> got(M * N), want(M * N);
        forge::cpu::gemv_iq2_s_q8k_transB_vsx(act.data(), w.data(), got.data(), M, K, N);
        ref_gemv(ref_dot_iq2_s_q8_K, w.data(), 82, q8_all, M, K, N, want.data());
        check("iq2_s_gemv", M, K, N, got.data(), want.data());
    }

    // ---- IQ2_XS ----
    {
        std::vector<uint8_t> w((size_t)N * 74);
        make_iq2xs_row(w.data(), 1, 0x3333u);
        std::vector<float> act(M * K);
        for (int m = 0; m < M; ++m) make_act(act.data() + (size_t)m * K, K, 0x4444u + m);
        std::vector<forge::cpu::block_q8_K> q8_all((size_t)M);
        for (int m = 0; m < M; ++m) quantize_row_q8_K_local(act.data() + (size_t)m * K, &q8_all[m], K);

        std::vector<float> got(M * N), want(M * N);
        forge::cpu::gemv_iq2_xs_q8k_transB_vsx(act.data(), w.data(), got.data(), M, K, N);
        ref_gemv(ref_dot_iq2_xs_q8_K, w.data(), 74, q8_all, M, K, N, want.data());
        check("iq2_xs_gemv", M, K, N, got.data(), want.data());
    }

    // ---- IQ4_NL ----
    {
        std::vector<uint8_t> w((size_t)N * 8 * 18);
        make_iq4nl_row(w.data(), 1, 0x5555u);
        std::vector<float> act(M * K);
        for (int m = 0; m < M; ++m) make_act(act.data() + (size_t)m * K, K, 0x6666u + m);
        std::vector<forge::cpu::block_q8_K> q8_all((size_t)M);
        for (int m = 0; m < M; ++m) quantize_row_q8_K_local(act.data() + (size_t)m * K, &q8_all[m], K);

        std::vector<float> got(M * N), want(M * N);
        forge::cpu::gemv_iq4_nl_q8k_transB_vsx(act.data(), w.data(), got.data(), M, K, N);
        ref_gemv(ref_dot_iq4_nl_q8_K, w.data(), 8 * 18, q8_all, M, K, N, want.data());
        check("iq4_nl_gemv", M, K, N, got.data(), want.data());
    }

    if (errors) { printf("FAILED (%d)\n", errors); return 1; }
    printf("ALL PASSED\n");
    return 0;
}
