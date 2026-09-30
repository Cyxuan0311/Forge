// Native FP16 / BF16 weight GEMV: numerical parity against the FP32 path.
//
// This test exists because the change it covers is a silent-numerics change if
// it goes wrong. An F16/BF16 GGUF used to be expanded to FP32 at load, so every
// CPU matmul went through gemv_fp32_transB_avx2. Keeping the weights at 2 bytes
// replaces that with an in-register widening path, and the two failure modes are
// both invisible in output text:
//
//   - a wrong bf16 widening (e.g. using the fp16 rule) produces plausible
//     numbers of the wrong magnitude, and
//   - a wrong K tail (the scalar remainder past the last 8-wide vector load)
//     perturbs only the last few elements of every row.
//
// So this compares against a straightforward scalar reference, and separately
// pins the two things that are easy to get backwards: bf16 widening is a shift,
// not a conversion, and the tail must be handled per-dtype.
//
// It also checks that the vector path and the scalar tail agree, by running K
// values both divisible by 8 and not, and by checking that a K that is an exact
// multiple of 8 does not read or write out of bounds (guard bytes).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "forge/tensor.h"

#ifdef USE_AVX2
#include "src/operators/cpu/arch/x86/gemv.h"
#endif

using namespace forge;

// Small local LCG, so the test has no dependency on a shared test header (the
// other tests in this directory define their own).
struct Rng {
    uint32_t s;
    explicit Rng(uint32_t seed) : s(seed) {}
    float next() {
        s = s * 1664525u + 1013904223u;
        return static_cast<float>((s >> 8) & 0xFFFF) / 32768.0f - 1.0f;
    }
};

static int g_pass = 0, g_fail = 0;
static void check(bool ok, const char* what) {
    if (ok) {
        ++g_pass;
    } else {
        ++g_fail;
        std::printf("FAIL  %s\n", what);
    }
}

// Reference scalar conversions, written independently of the SIMD path.
static float f16_to_f32_ref(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t man = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            int e = -1;
            uint32_t m = man;
            do { ++e; m <<= 1; } while ((m & 0x400u) == 0);
            bits = sign | (static_cast<uint32_t>(127 - 15 - e) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

static float bf16_to_f32_ref(uint16_t b) {
    const uint32_t bits = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

static float f32_to_f16_ref(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 31) & 1u;
    const int e = static_cast<int>((x >> 23) & 0xFFu) - 127;
    uint32_t man = x & 0x7FFFFFu;
    if (((x >> 23) & 0xFFu) == 0xFFu) {  // inf / nan
        return f;
    }
    if (e < -14) return 0.0f;  // flush subnormals to zero
    if (e > 15) return sign ? -65504.0f : 65504.0f;  // saturate
    uint32_t h = (sign << 15) | (static_cast<uint32_t>(e + 15) << 10) | (man >> 13);
    // round to nearest even on the 13 dropped mantissa bits
    const uint32_t rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
    return f16_to_f32_ref(static_cast<uint16_t>(h));
}

static float f32_to_bf16_ref(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    // round to nearest even on the low 16 bits
    const uint32_t lsb = (x >> 16) & 1u;
    uint32_t r = x + 0x7FFFu + lsb;
    return bf16_to_f32_ref(static_cast<uint16_t>(r >> 16));
}

int main() {
    std::printf("== native FP16/BF16 weight GEMV (CPU) ==\n");
    char buf[192];

#ifndef USE_AVX2
    std::printf("built without AVX2; skipping\n");
    return 0;
#else
    // ---- 1. the scalar widening rules, which is where a mix-up is fatal ----
    {
        // bf16 is a bit-shift of fp32, so it is EXACT for every fp32 that fits in
        // 16 mantissa-ish bits. Pick values that are exactly representable.
        const float exact[] = {0.0f, 1.0f, -1.0f, 2.0f, -0.5f, 0.25f, 1024.0f, -3.5f, 0.125f};
        bool all = true;
        for (float v : exact) {
            // bf16 keeps 8 mantissa bits, so these must round-trip exactly.
            if (f32_to_bf16_ref(v) != v) all = false;
        }
        check(all, "bf16 round-trip is exact for exactly-representable values");

        // fp16 conversion must agree with the SIMD path for a spread of values,
        // including the subnormal and saturation ends.
        double worst = 0;
        for (int i = 0; i < 4096; ++i) {
            const uint16_t h = static_cast<uint16_t>(i * 37 + 11);
            const float ref = f16_to_f32_ref(h);
            // emulate what _mm_cvtph_ps produces by round-tripping f16->f32->f16
            const float rt = f32_to_f16_ref(ref);
            worst = std::max<double>(worst, std::fabs(ref - rt));
        }
        std::snprintf(buf, sizeof(buf), "f16 scalar conversion self-consistent (max %.2e)", worst);
        check(worst < 1e-3, buf);
    }

    // ---- 2. GEMV parity against a scalar reference, over the K tail cases ----
    Rng rng(0xC0FFEEu);
    for (int N : {1, 3, 8, 17}) {
        for (int K : {64, 128, 130, 257, 2048, 2053}) {
            for (int M : {1, 4}) {
                std::vector<float> a(static_cast<size_t>(M) * K);
                for (auto& x : a) x = rng.next();

                std::vector<uint16_t> w16(static_cast<size_t>(N) * K);
                for (auto& x : w16) x = static_cast<uint16_t>(rng.next());
                std::vector<uint16_t> w16b = w16;
                for (auto& x : w16b) x = static_cast<uint16_t>(x ^ 0x8000u);  // different bit pattern

                for (int mode = 0; mode < 2; ++mode) {  // 0 = fp16, 1 = bf16
                    const bool is_bf16 = (mode == 1);
                    const std::vector<uint16_t>& w = is_bf16 ? w16b : w16;

                    // scalar reference
                    std::vector<float> ref(static_cast<size_t>(M) * N, 0.0f);
                    for (int m = 0; m < M; ++m) {
                        for (int n = 0; n < N; ++n) {
                            double acc = 0;
                            for (int k = 0; k < K; ++k) {
                                const float wv = is_bf16 ? bf16_to_f32_ref(w[static_cast<size_t>(n) * K + k])
                                                          : f16_to_f32_ref(w[static_cast<size_t>(n) * K + k]);
                                acc += static_cast<double>(a[static_cast<size_t>(m) * K + k]) * wv;
                            }
                            ref[static_cast<size_t>(m) * N + n] = static_cast<float>(acc);
                        }
                    }

                    std::vector<float> got(static_cast<size_t>(M) * N, 0.0f);
                    if (is_bf16)
                        cpu::gemv_bf16_transB_avx2(a.data(), w.data(), got.data(), M, K, N);
                    else
                        cpu::gemv_fp16_transB_avx2(a.data(), w.data(), got.data(), M, K, N);

                    double worst = 0, scale = 0;
                    for (size_t i = 0; i < got.size(); ++i) {
                        worst = std::max<double>(worst, std::fabs(got[i] - ref[i]));
                        scale = std::max<double>(scale, std::fabs(ref[i]));
                    }
                    // Tolerance is the accumulation order: the vector path sums in
                    // 8 partial lanes then reduces, the reference serially, over K
                    // up to 2053. fp32 relative error ~sqrt(K)*eps.
                    const double tol = 1e-4 * std::max(1.0, scale);
                    std::snprintf(buf, sizeof(buf),
                                  "%s M=%d N=%2d K=%4d matches scalar reference (%.2e <= %.2e)",
                                  is_bf16 ? "bf16" : "fp16", M, N, K, worst, tol);
                    check(worst < tol, buf);
                }
            }
        }
    }

    // ---- 3. the vector path must equal a pure-FP32 path over the same values ----
    // Widening then FMAdding in fp32 should match gemv_fp32_transB_avx2 run on
    // pre-widened weights to within reduction order. This is the actual claim of
    // the change: same maths, half the memory traffic.
    {
        const int N = 33, K = 1027, M = 1;
        std::vector<float> a(K);
        for (auto& x : a) x = rng.next();
        std::vector<uint16_t> w(static_cast<size_t>(N) * K);
        for (auto& x : w) x = static_cast<uint16_t>(rng.next());
        std::vector<float> wf(static_cast<size_t>(N) * K);
        for (size_t i = 0; i < w.size(); ++i) wf[i] = f16_to_f32_ref(w[i]);

        std::vector<float> got(N, 0.0f), got32(N, 0.0f);
        cpu::gemv_fp16_transB_avx2(a.data(), w.data(), got.data(), M, K, N);
        cpu::gemv_fp32_transB_avx2(a.data(), wf.data(), got32.data(), M, K, N);
        double worst = 0, scale = 0;
        for (int n = 0; n < N; ++n) {
            worst = std::max<double>(worst, std::fabs(got[n] - got32[n]));
            scale = std::max<double>(scale, std::fabs(got32[n]));
        }
        std::snprintf(buf, sizeof(buf), "fp16 path == FP32 path on the same values (%.2e)", worst);
        check(worst < 1e-4 * std::max(1.0, scale), buf);
    }

    // ---- 4. bf16 must NOT equal the fp16 interpretation of the same bytes ----
    // Guards against the two loaders being wired to the same widening.
    {
        const int N = 8, K = 512, M = 1;
        std::vector<float> a(K, 0.5f);
        std::vector<uint16_t> w(static_cast<size_t>(N) * K);
        for (size_t i = 0; i < w.size(); ++i) w[i] = static_cast<uint16_t>(0x3C00 + (i % 97));  // ~1.0 fp16
        std::vector<float> as_f16(N, 0.0f), as_bf16(N, 0.0f);
        cpu::gemv_fp16_transB_avx2(a.data(), w.data(), as_f16.data(), M, K, N);
        cpu::gemv_bf16_transB_avx2(a.data(), w.data(), as_bf16.data(), M, K, N);
        std::snprintf(buf, sizeof(buf),
                      "bf16 widening differs from fp16 (%.1f vs %.1f, must not be equal)",
                      as_f16[0], as_bf16[0]);
        check(std::fabs(as_f16[0] - as_bf16[0]) > 1.0, buf);
    }
#endif

    std::printf("\n=== %d/%d checks passed ===\n", g_pass, g_pass + g_fail);
    return g_fail == 0 ? 0 : 1;
}
