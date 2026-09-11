// Standalone GEMV micro-benchmark for the Forge CPU decode kernels.
//
// Purpose: iterating on kernel-level optimisations via the full model costs
// ~80 s of model load plus a 30 s generation per run, on a machine whose
// load time alone varies 57-76 s. This harness times the actual decode kernels
// directly, so a tuning iteration takes about a second and is reproducible.
//
// Weights are synthesised (valid block layout, realistic scales) rather than
// loaded from a GGUF -- the kernels are bandwidth/compute bound and do not care
// about the weight values, only their size and alignment.
//
// Build (from the repo root):
//   g++ -std=c++20 -O2 -mavx2 -mfma -mf16c -DUSE_AVX2 \
//       -I src/operators -I include -pthread \
//       benchmarks/gemv_bench.cpp -o /tmp/gemv_bench
// Add -mavx512f -mavx512bw -mavx512vnni -mavx512vl -DUSE_AVX512_VNNI for the
// AVX-512 variants.
//
// Usage:
//   /tmp/gemv_bench            # the real Qwen2.5-7B decode shapes
//   /tmp/gemv_bench --sweep    # also sweep N to show bandwidth vs size
//
// Threads follow FORGE_NUM_THREADS (default: hardware concurrency).

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "cpu/arch/x86/gemm_microkernel.h"
#include "cpu/arch/x86/fused.h"

using namespace forge::cpu;
using clk = std::chrono::steady_clock;

// Minimal stand-in for forge::host_mem (used by scratch_vec) so this file links
// without the full core library. Mirrors tests/cross_host_mem_stub.cpp.
namespace forge {
namespace host_mem {
void* allocate(size_t bytes) {
    void* p = nullptr;
    if (posix_memalign(&p, 64, bytes) != 0) return nullptr;
    return p;
}
void deallocate(void* ptr) { free(ptr); }
}  // namespace host_mem
}  // namespace forge

// ---- synthesised weights ----------------------------------------------------

static uint32_t g_seed = 12345u;
static uint32_t rnd() {
    g_seed = g_seed * 1103515245u + 12345u;
    return g_seed >> 8;
}

// Q4_0 block: 18 bytes = uint16 fp16 scale + 16 bytes of packed 4-bit quants.
static std::vector<uint8_t> make_q4_0(int64_t K, int64_t N) {
    const int64_t nb = K / 32;
    std::vector<uint8_t> w((size_t)N * nb * 18);
    for (size_t off = 0; off < w.size(); off += 18) {
        const uint16_t d = 0x3C00;  // fp16 1.0 -- keeps dot products well scaled
        std::memcpy(w.data() + off, &d, 2);
        for (int j = 0; j < 16; ++j) w[off + 2 + j] = (uint8_t)rnd();
    }
    return w;
}

// Q6_K block: 210 bytes = ql[128] qh[64] scales[16] d(uint16 fp16).
static std::vector<uint8_t> make_q6_K(int64_t K, int64_t N) {
    const int64_t nb = K / 256;
    std::vector<uint8_t> w((size_t)N * nb * 210);
    for (size_t off = 0; off < w.size(); off += 210) {
        for (int i = 0; i < 208; ++i) w[off + i] = (uint8_t)rnd();
        const uint16_t d = 0x3800;  // fp16 0.5
        std::memcpy(w.data() + off + 208, &d, 2);
    }
    return w;
}

static std::vector<float> make_act(int64_t K) {
    std::vector<float> a((size_t)K);
    for (int64_t i = 0; i < K; ++i)
        a[i] = (float)((int)(rnd() % 2001) - 1000) / 1000.0f;
    return a;
}

// ---- timing helpers ---------------------------------------------------------

template <typename F>
static double bench_ms(F&& f, int iters, int warmup = 3) {
    for (int i = 0; i < warmup; ++i) f();
    auto t0 = clk::now();
    for (int i = 0; i < iters; ++i) f();
    auto t1 = clk::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

static bool any_bad(const float* p, size_t n) {
    for (size_t i = 0; i < n; ++i)
        if (!std::isfinite(p[i])) return true;
    return false;
}

static void report(const char* name, int64_t K, int64_t N, double bytes, double ms,
                   bool bad) {
    const double gbs = bytes / (ms * 1e-3) / 1e9;
    std::printf("  %-34s K=%6lld N=%7lld  %9.4f ms  %7.2f GB/s%s\n", name,
                (long long)K, (long long)N, ms, gbs, bad ? "   <-- NaN/Inf!" : "");
}

int main(int argc, char** argv) {
    bool sweep = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--sweep") sweep = true;

    const int nth = get_num_threads();
    std::printf("GEMV benchmark -- threads=%d (FORGE_NUM_THREADS to change)\n\n", nth);

    // Bytes per element for each quant type.
    const double BPE_Q4_0 = 18.0 / 32.0;    // 4.5 bits
    const double BPE_Q6_K = 210.0 / 256.0;  // 6.5625 bits

    std::printf("Qwen2.5-7B decode shapes (per layer, M=1):\n");

    // ---- 1. fused ffn_up (gate + up, two weight matrices) ----
    {
        const int64_t K = 3584, N = 18944;  // intermediate_size = 18944
        auto gate = make_q4_0(K, N);
        auto up = make_q4_0(K, N);
        auto r1 = repack_q4_0_weights_8x8(gate.data(), K, N);
        auto r2 = repack_q4_0_weights_8x8(up.data(), K, N);
        if (!r1.first || !r2.first) {
            std::printf("  ffn_up: repack failed\n");
        } else {
            auto a = make_act(K);
            std::vector<float> out((size_t)N * 2);
            double ms = bench_ms([&] {
                gemv_q4_0_fused_ffn_up_8x8_avx2(a.data(), r1.first, r2.first, out.data(), K, N);
            }, 200);
            report("gemv_q4_0_fused_ffn_up_8x8", K, N * 2,
                   2.0 * (double)K * N * BPE_Q4_0, ms, any_bad(out.data(), out.size()));
            delete[] r1.first;
            delete[] r2.first;
        }
    }

    // ---- 2. ffn_down (K=intermediate, N=hidden) with residual ----
    {
        const int64_t K = 18944, N = 3584;
        auto w = make_q4_0(K, N);
        auto r = repack_q4_0_weights_8x8(w.data(), K, N);
        if (!r.first) {
            std::printf("  ffn_down: repack failed\n");
        } else {
            auto a = make_act(K);
            std::vector<float> res((size_t)N, 0.5f), out((size_t)N);
            double ms = bench_ms([&] {
                gemv_q4_0_8x8_residual_avx2(a.data(), r.first, res.data(), out.data(), K, N);
            }, 200);
            report("gemv_q4_0_8x8_residual (ffn_down)", K, N,
                   (double)K * N * BPE_Q4_0, ms, any_bad(out.data(), out.size()));
            delete[] r.first;
        }
    }

    // ---- 3. attn output proj (K=hidden, N=hidden) with residual ----
    {
        const int64_t K = 3584, N = 3584;
        auto w = make_q4_0(K, N);
        auto r = repack_q4_0_weights_8x8(w.data(), K, N);
        if (!r.first) {
            std::printf("  attn_proj: repack failed\n");
        } else {
            auto a = make_act(K);
            std::vector<float> res((size_t)N, 0.5f), out((size_t)N);
            double ms = bench_ms([&] {
                gemv_q4_0_8x8_residual_avx2(a.data(), r.first, res.data(), out.data(), K, N);
            }, 500);
            report("gemv_q4_0_8x8_residual (attn_proj)", K, N,
                   (double)K * N * BPE_Q4_0, ms, any_bad(out.data(), out.size()));
            delete[] r.first;
        }
    }

    // ---- 4. fused qkv: q=N_q, k=N_k, v=N_v in one pass ----
    {
        const int64_t K = 3584, N_q = 3584, N_k = 512, N_v = 512;
        auto wq = make_q4_0(K, N_q);
        auto wk = make_q4_0(K, N_k);
        auto wv = make_q4_0(K, N_v);
        auto rq = repack_q4_0_weights_8x8(wq.data(), K, N_q);
        auto rk = repack_q4_0_weights_8x8(wk.data(), K, N_k);
        auto rv = repack_q4_0_weights_8x8(wv.data(), K, N_v);
        if (!rq.first || !rk.first || !rv.first) {
            std::printf("  qkv: repack failed\n");
        } else {
            auto a = make_act(K);
            std::vector<float> oq((size_t)N_q), ok((size_t)N_k), ov((size_t)N_v);
            double ms = bench_ms([&] {
                gemv_q4_0_fused_qkv_8x8_avx2(a.data(), rq.first, rk.first, rv.first,
                                             oq.data(), ok.data(), ov.data(),
                                             K, N_q, N_k, N_v);
            }, 500);
            const double bytes = (double)K * (N_q + N_k + N_v) * BPE_Q4_0;
            report("gemv_q4_0_fused_qkv_8x8", K, N_q + N_k + N_v, bytes, ms,
                   any_bad(oq.data(), oq.size()));
            delete[] rq.first;
            delete[] rk.first;
            delete[] rv.first;
        }
    }

    // ---- 5. output projection, Q6_K (largest single tensor) ----
    {
        const int64_t K = 3584, N = 152064;  // vocab
        auto w = make_q6_K(K, N);
        auto r = repack_q6_K_weights(w.data(), K, N);
        if (!r.first) {
            std::printf("  output_proj: repack failed\n");
        } else {
            auto a = make_act(K);
            std::vector<float> out((size_t)N);
            double ms = bench_ms([&] {
                gemv_q6_K_8x8_q8_K_avx2(a.data(), r.first, out.data(), (int)K, (int)N);
            }, 30);
            report("gemv_q6_K_8x8_q8_K", K, N, (double)K * N * BPE_Q6_K, ms,
                   any_bad(out.data(), out.size()));
            delete[] r.first;
        }
    }

    // ---- sweep: bandwidth as a function of matrix size ----
    if (sweep) {
        std::printf("\nSweep -- K=3584, single Q4_0 GEMV with residual, varying N:\n");
        std::printf("  %-12s %12s %12s %10s\n", "N", "MB", "ms", "GB/s");
        for (int64_t N : {512, 1024, 2048, 3584, 7168, 14336, 18944, 37888, 75776}) {
            const int64_t K = 3584;
            auto w = make_q4_0(K, N);
            auto r = repack_q4_0_weights_8x8(w.data(), K, N);
            if (!r.first) continue;
            auto a = make_act(K);
            std::vector<float> res((size_t)N, 0.5f), out((size_t)N);
            const int iters = N <= 4096 ? 2000 : (N <= 16384 ? 500 : 200);
            double ms = bench_ms([&] {
                gemv_q4_0_8x8_residual_avx2(a.data(), r.first, res.data(), out.data(), K, N);
            }, iters);
            const double mb = (double)K * N * BPE_Q4_0 / 1e6;
            std::printf("  %-12lld %12.2f %12.4f %10.2f\n", (long long)N, mb, ms,
                        mb / 1e3 / (ms * 1e-3));
            delete[] r.first;
        }
    }

    std::printf("\ndone\n");
    return 0;
}
