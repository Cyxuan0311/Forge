// Standalone CPU decode-kernel benchmark (synthetic weights, zero file IO).
//
// Isolates kernel + thread-pool performance from storage so P2/P3/P4 changes
// are measurable independently of the WSL DrvFs page-fault bottleneck.
//
//   g++ -std=c++17 -O3 -march=native -fopenmp -DFORGE_ARCH_X86 -DUSE_AVX2 \
//       -I. -Iinclude -Isrc -Isrc/operators/cpu -Isrc/operators/cpu/common \
//       tests/bench_cpu_decode.cpp -o /tmp/bench_cpu_decode -lpthread
//   /tmp/bench_cpu_decode [iters]          (FORGE_PIN_THREADS=first|last opt)

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "src/operators/cpu/arch/x86/fused.h"
#include "src/operators/cpu/arch/x86/gemm_microkernel.h"
#include "src/operators/cpu/arch/x86/repack.h"

using namespace forge;

namespace {

uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    const uint32_t man = x & 0x7FFFFFu;
    if (exp <= 0) return static_cast<uint16_t>(sign);
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (man >> 13));
}

float urand(std::mt19937& g, float lo, float hi) {
    return lo + (hi - lo) * (static_cast<float>(g() % 100000) / 100000.0f);
}

void fill_q4_0(std::vector<uint8_t>& w, int64_t K, int64_t N, std::mt19937& g) {
    const int64_t nb = (K + 31) / 32;
    w.assign(static_cast<size_t>(nb) * 18 * N, 0);
    for (size_t i = 0; i < w.size(); i += 18) {
        const uint16_t sb_bits[] = {
            static_cast<uint16_t>(f32_to_f16(urand(g, 0.01f, 1.0f)))};
        std::memcpy(w.data() + i, sb_bits, 2);
        for (int j = 0; j < 16; ++j) w[i + 2 + j] = static_cast<uint8_t>(g() & 0xFF);
    }
    (void)K; (void)N;
}

void fill_q6_k(std::vector<uint8_t>& w, int64_t N, int64_t nb, std::mt19937& g) {
    w.resize(static_cast<size_t>(N) * nb * 210);
    for (size_t off = 0; off < w.size(); off += 210) {
        for (int i = 0; i < 192; ++i) w[off + i] = static_cast<uint8_t>(g() & 0xFF);
        for (int i = 0; i < 16; ++i)
            w[off + 192 + i] = static_cast<uint8_t>(static_cast<int>(g() % 127) - 63);
        const uint16_t d_bits[] = {
            static_cast<uint16_t>(f32_to_f16(urand(g, 0.01f, 1.0f)))};
        std::memcpy(w.data() + off + 208, d_bits, 2);
    }
}

template <typename Fn>
double time_ms(Fn&& fn, int iters, int warmup) {
    for (int i = 0; i < warmup; ++i) fn();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) fn();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

}  // namespace

int main(int argc, char** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 200;
    std::mt19937 rng(7);

    // ---- Case 1: TinyLlama-style fused QKV, Q4_0 8x8 --------------------
    // K=2048, Nq=2048 (32 heads x 64), Nk=Nv=256 (4 kv heads x 64).
    {
        constexpr int64_t K = 2048, Nq = 2048, Nk = 256, Nv = 256;
        std::vector<uint8_t> wq_raw, wk_raw, wv_raw;
        fill_q4_0(wq_raw, K, Nq, rng);
        fill_q4_0(wk_raw, K, Nk, rng);
        fill_q4_0(wv_raw, K, Nv, rng);

        const auto* wq8 = ops::get_repacked_q4_0_8x8(wq_raw.data(), K, Nq);
        const auto* wk8 = ops::get_repacked_q4_0_8x8(wk_raw.data(), K, Nk);
        const auto* wv8 = ops::get_repacked_q4_0_8x8(wv_raw.data(), K, Nv);
        if (!wq8 || !wk8 || !wv8) {
            std::printf("qkv repack failed\n");
            return 1;
        }

        std::vector<float> a(K), oq(Nq), ok(Nk), ov(Nv);
        for (auto& v : a) v = urand(rng, -1.f, 1.f);

        const double mbytes =
            static_cast<double>(Nq + Nk + Nv) * K * 0.5 / (1024.0 * 1024.0);
        const double ms = time_ms(
            [&] { cpu::gemv_q4_0_fused_qkv_8x8_avx2(a.data(), wq8, wk8, wv8,
                                                    oq.data(), ok.data(), ov.data(),
                                                    (int)K, Nq, Nk, Nv); },
            iters, 10);
        std::printf("qkv_8x8  K=%lld N=%lld : %.3f ms/call  %.2f GB/s\n",
                    (long long)K, (long long)(Nq + Nk + Nv), ms,
                    mbytes / 1024.0 / (ms / 1000.0));
    }

    // ---- Case 2: output projection, Q6_K 8x8 (llama-ish vocab) ----------
    {
        constexpr int64_t K = 2048, N = 32000;
        const int64_t nb = K / 256;
        std::vector<uint8_t> w_raw;
        fill_q6_k(w_raw, N, nb, rng);

        const auto* w8 = ops::get_repacked_q6_K(w_raw.data(), K, N);
        if (!w8) {
            std::printf("q6_K repack failed\n");
            return 1;
        }

        std::vector<float> a(K), out(N);
        for (auto& v : a) v = urand(rng, -1.f, 1.f);

        const double mbytes = static_cast<double>(N) * K * 0.84375 / (1024.0 * 1024.0);
        const double ms = time_ms(
            [&] { cpu::gemv_q6_K_8x8_q8_K_avx2(a.data(), w8, out.data(), (int)K, (int)N); },
            iters, 5);
        std::printf("q6_proj  K=%lld N=%lld : %.3f ms/call  %.2f GB/s\n",
                    (long long)K, (long long)N, ms,
                    mbytes / 1024.0 / (ms / 1000.0));
    }

    return 0;
}
