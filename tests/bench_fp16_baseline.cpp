// =========================================================================
// bench_fp16_baseline.cpp -- Phase 2 round 1 (6-A-7 Phase 2): FP16 baseline
// vs the existing FP32 path, real kernel timing only.
//
// Round-1 question (the ONLY question this bench answers):
//     "How much faster is the Native FP16 baseline than the existing FP32
//      path?" -- separately for Decode and Prefill.
//
// Round 2 (NOT here) will compare __half2 / TensorCore candidates against
// THIS baseline. Per the Phase 2 discipline:
//   * no occupancy / register / theoretical-throughput rankings -- only real
//     kernel timings (cudaEvent around back-to-back launches);
//   * Decode (streaming, latency-sensitive) and Prefill (vectorized, larger
//     parallel workload) are measured SEPARATELY, never merged;
//   * two workload regimes per op: small = launch-bound, large =
//     bandwidth-bound (Phase 6-A benchmark taxonomy).
//
// Sections:
//   1. attention decode  : FP16 decode kernel vs FP32 decode kernel
//                          (q_len=1, GQA 8/2, HD=64, kv sweep)
//   2. attention prefill : FP16 vectorized vs FP32 vectorized (q=kv sweep)
//   3. elementwise       : 6-A-8.0 ADD/MUL FP16 vs FP32 (the two regimes)
//
// Other chain ops (RMSNorm / SiLU / matmul) already have recorded benches in
// their 6-A test deliverables and are intentionally NOT duplicated here.
//
// Run: ./build/tests/forge-bench-fp16-baseline   (skips without GPU)
// =========================================================================

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <vector>

#include "forge/cuda_kernels.h"

using forge::cuda::launch_add_fp16;
using forge::cuda::launch_flash_attention_gqa_decode;
using forge::cuda::launch_flash_attention_gqa_decode_fp16;
using forge::cuda::launch_flash_attention_gqa_decode_fp16_h2;
using forge::cuda::launch_flash_attention_gqa_prefill_fp16;
using forge::cuda::launch_flash_attention_gqa_prefill_fp16_vectorized;
using forge::cuda::launch_flash_attention_gqa_prefill_mma_cfg;
using forge::cuda::launch_flash_attention_gqa_prefill_mma_fp16;
using forge::cuda::launch_flash_attention_gqa_vectorized;
using forge::cuda::launch_multiply_fp16;

namespace {

void check_cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[CUDA] %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// Real kernel timing: warmup, then cudaEvent around iters back-to-back launches.
template <typename Fn>
float time_ms(Fn&& fn, int iters) {
    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    fn();  // warmup
    cudaDeviceSynchronize();
    cudaEventRecord(s);
    for (int i = 0; i < iters; ++i) fn();
    cudaEventRecord(e);
    cudaDeviceSynchronize();
    float ms = 0;
    cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s);
    cudaEventDestroy(e);
    return ms / iters;
}

uint16_t f32_to_f16(float f) {
    __half h = __float2half(f);
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return u;
}

struct DevBuf {
    void* p = nullptr;
    size_t bytes = 0;
    DevBuf(size_t b) : bytes(b) { check_cuda(cudaMalloc(&p, b), "cudaMalloc"); }
    ~DevBuf() { cudaFree(p); }
    template <typename T>
    T* as() const {
        return static_cast<T*>(p);
    }
    void fill_random(std::mt19937& g, float lo, float hi) {
        std::uniform_real_distribution<float> d(lo, hi);
        std::vector<float> host(bytes / 4);
        for (auto& v : host) v = d(g);
        check_cuda(cudaMemcpy(p, host.data(), bytes, cudaMemcpyHostToDevice), "fill H2D");
    }
    void fill_fp16(std::mt19937& g, float lo, float hi) {
        std::uniform_real_distribution<float> d(lo, hi);
        size_t n = bytes / 2;
        std::vector<uint16_t> host(n);
        for (size_t i = 0; i < n; ++i) host[i] = f32_to_f16(d(g));
        check_cuda(cudaMemcpy(p, host.data(), bytes, cudaMemcpyHostToDevice), "fill H2D");
    }
};

constexpr int kNh = 8, kNkv = 2, kHd = 64;  // the 6-A GQA shape family

// ---- section 1: attention decode (q_len = 1) -----------------------------
// Decode streams the whole KV cache once per token: bandwidth-bound for large
// kv, launch-bound for small kv. Report ms/token + implied tokens/s.
void bench_decode(std::mt19937& g) {
    std::printf("  [decode] q_len=1 GQA nh=%d nkv=%d hd=%d (FP16 kernel vs FP32 kernel)\n",
                kNh, kNkv, kHd);
    struct Cfg {
        int kv;
        int iters;
    };
    const Cfg cfgs[] = {{128, 500}, {512, 300}, {2048, 100}};  // small -> large
    for (const auto& c : cfgs) {
        const int kv_dim = kNkv * kHd, out_dim = kNh * kHd;
        DevBuf q16(1 * out_dim * 2), k16((size_t)c.kv * kv_dim * 2),
            v16((size_t)c.kv * kv_dim * 2), o16(1 * out_dim * 2);
        DevBuf q32(1 * out_dim * 4), k32((size_t)c.kv * kv_dim * 4),
            v32((size_t)c.kv * kv_dim * 4), o32(1 * out_dim * 4);
        q16.fill_fp16(g, -1, 1);
        k16.fill_fp16(g, -1, 1);
        v16.fill_fp16(g, -1, 1);
        q32.fill_random(g, -1, 1);
        k32.fill_random(g, -1, 1);
        v32.fill_random(g, -1, 1);

        float ms16 = time_ms(
            [&] {
                launch_flash_attention_gqa_decode_fp16(q16.as<__half>(), k16.as<__half>(),
                                                       v16.as<__half>(), o16.as<__half>(), c.kv,
                                                       kNh, kNkv, kHd, nullptr, 0);
            },
            c.iters);
        float ms32 = time_ms(
            [&] {
                launch_flash_attention_gqa_decode(q32.as<float>(), k32.as<float>(), v32.as<float>(),
                                                  o32.as<float>(), c.kv, kNh, kNkv, kHd, nullptr, 0);
            },
            c.iters);
        std::printf(
            "    kv=%4d : FP16 %8.4f ms/tok (%7.0f tok/s) | FP32 %8.4f ms/tok (%7.0f tok/s) | "
            "speedup %.2fx\n",
            c.kv, ms16, 1000.0 / ms16, ms32, 1000.0 / ms32, ms32 / ms16);
    }
}

// ---- section 1b (round 2): __half2 decode candidate vs frozen baseline ----
// Round-2 question: "Is the __half2 candidate actually faster than THIS
// baseline?" Same configs, same measurement discipline as section 1. The
// candidate keeps the FP32 math contract (no HFMA2) -- it only pairs the
// f16->f32 conversions and hoists the V loads.
// Aggregation: the kv=512 transition point showed 12-40% run-to-run noise
// (clock/thermal states), so this section interleaves baseline/candidate
// measurements 5 times and reports each kernel's MINIMUM (interference-
// robust); the inner protocol stays warmup + cudaEvent + mean as elsewhere.
void bench_decode_h2(std::mt19937& g) {
    std::printf(
        "  [decode-h2] q_len=1 GQA nh=%d nkv=%d hd=%d (candidate vs frozen baseline, best-of-5)\n",
        kNh, kNkv, kHd);
    struct Cfg {
        int kv;
        int iters;
    };
    const Cfg cfgs[] = {{128, 500}, {512, 300}, {2048, 100}};  // small -> large
    for (const auto& c : cfgs) {
        const int kv_dim = kNkv * kHd, out_dim = kNh * kHd;
        DevBuf q16(1 * out_dim * 2), k16((size_t)c.kv * kv_dim * 2),
            v16((size_t)c.kv * kv_dim * 2), o16(1 * out_dim * 2);
        DevBuf o16b(1 * out_dim * 2);
        q16.fill_fp16(g, -1, 1);
        k16.fill_fp16(g, -1, 1);
        v16.fill_fp16(g, -1, 1);

        auto base_fn = [&] {
            launch_flash_attention_gqa_decode_fp16(q16.as<__half>(), k16.as<__half>(),
                                                   v16.as<__half>(), o16.as<__half>(), c.kv,
                                                   kNh, kNkv, kHd, nullptr, 0);
        };
        auto h2_fn = [&] {
            launch_flash_attention_gqa_decode_fp16_h2(q16.as<__half>(), k16.as<__half>(),
                                                      v16.as<__half>(), o16b.as<__half>(), c.kv,
                                                      kNh, kNkv, kHd, nullptr, 0);
        };
        float ms_base = 1e30f, ms_h2 = 1e30f;
        for (int rep = 0; rep < 5; ++rep) {  // interleaved to share clock state
            ms_base = std::min(ms_base, time_ms(base_fn, c.iters));
            ms_h2 = std::min(ms_h2, time_ms(h2_fn, c.iters));
        }
        std::printf("    kv=%4d : baseline %8.4f ms/tok | h2 candidate %8.4f ms/tok | "
                    "h2 vs baseline %.3fx\n",
                    c.kv, ms_base, ms_h2, ms_base / ms_h2);
    }
}

// ---- section 2b (round 3/4): TensorCore (wmma) prefill candidate -----------
// Round-4 question: "Is the q=1024 1.09x the TensorCore's ceiling or the tile
// config's ceiling?" Tile-config scan: for each (Br, Bc, warps) config,
// best-of-5 interleaved (baseline, candidate) A/B vs the frozen 6-A-7
// baseline -- same protocol as round 2/3. The (32,32,2) row is the round-3
// shape. Br=32 & 4 warps is not measurable (wmma M=16).
void bench_prefill_mma_scan(std::mt19937& g) {
    struct MmaCfg {
        int br, bc, warps;
    };
    const MmaCfg mcfgs[] = {{32, 16, 2}, {32, 32, 2}, {32, 64, 2}, {64, 16, 2}, {64, 32, 2},
                            {64, 64, 2}, {64, 16, 4}, {64, 32, 4}, {64, 64, 4}};
    struct QCfg {
        int q;
        int iters;
    };
    const QCfg qcfgs[] = {{64, 300}, {256, 100}, {1024, 30}};  // small -> large
    float best_ratio[3] = {0.f, 0.f, 0.f};
    int best_cfg[3] = {-1, -1, -1};

    std::printf(
        "  [prefill-mma-scan] q=kv causal GQA nh=%d nkv=%d hd=%d (config scan vs baseline, "
        "best-of-5 interleaved)\n",
        kNh, kNkv, kHd);
    for (size_t qi = 0; qi < 3; ++qi) {
        const QCfg& c = qcfgs[qi];
        const int kv_dim = kNkv * kHd, out_dim = kNh * kHd;
        DevBuf q16((size_t)c.q * out_dim * 2), k16((size_t)c.q * kv_dim * 2),
            v16((size_t)c.q * kv_dim * 2), o16((size_t)c.q * out_dim * 2),
            o16b((size_t)c.q * out_dim * 2);
        q16.fill_fp16(g, -1, 1);
        k16.fill_fp16(g, -1, 1);
        v16.fill_fp16(g, -1, 1);

        auto base_fn = [&] {
            // Frozen 6-A-7.2 baseline arm -- explicit vectorized entry (the
            // generic production entry now routes hd=64 to the promoted wmma).
            launch_flash_attention_gqa_prefill_fp16_vectorized(
                q16.as<__half>(), k16.as<__half>(), v16.as<__half>(), o16.as<__half>(), c.q, c.q,
                kNh, kNkv, kHd, nullptr, true, 0);
        };
        for (size_t ci = 0; ci < 9; ++ci) {
            const MmaCfg& m = mcfgs[ci];
            auto mma_fn = [&] {
                launch_flash_attention_gqa_prefill_mma_cfg(q16.as<__half>(), k16.as<__half>(),
                                                           v16.as<__half>(), o16b.as<__half>(),
                                                           c.q, c.q, kNh, kNkv, kHd, true, m.br,
                                                           m.bc, m.warps, 0);
            };
            float ms_base = 1e30f, ms_mma = 1e30f;
            for (int rep = 0; rep < 5; ++rep) {  // interleaved to share clock state
                ms_base = std::min(ms_base, time_ms(base_fn, c.iters));
                ms_mma = std::min(ms_mma, time_ms(mma_fn, c.iters));
            }
            const float ratio = ms_base / ms_mma;
            if (ratio > best_ratio[qi]) {
                best_ratio[qi] = ratio;
                best_cfg[qi] = static_cast<int>(ci);
            }
            std::printf("    q=%4d Br=%2d Bc=%2d w=%d : baseline %8.4f ms | cfg %8.4f ms | "
                        "%.3fx%s\n",
                        c.q, m.br, m.bc, m.warps, ms_base, ms_mma, ratio,
                        (m.br == 32 && m.bc == 32 && m.warps == 2) ? "  (round-3 shape)" : "");
        }
    }
    for (size_t qi = 0; qi < 3; ++qi) {
        const MmaCfg& m = mcfgs[best_cfg[qi]];
        std::printf("    best q=%4d : Br=%d Bc=%d w=%d -> %.3fx\n", qcfgs[qi].q, m.br, m.bc,
                    m.warps, best_ratio[qi]);
    }
}

// ---- section 2: attention prefill (q_len == kv_len, causal) ---------------
// Prefill is the vectorized regime: q_len in a sweep from small to large.
void bench_prefill(std::mt19937& g) {
    std::printf("  [prefill] q=kv causal GQA nh=%d nkv=%d hd=%d (FP16 vectorized vs FP32 "
                "vectorized)\n",
                kNh, kNkv, kHd);
    struct Cfg {
        int q;
        int iters;
    };
    const Cfg cfgs[] = {{64, 300}, {256, 100}, {1024, 30}};  // small -> large
    for (const auto& c : cfgs) {
        const int kv_dim = kNkv * kHd, out_dim = kNh * kHd;
        DevBuf q16((size_t)c.q * out_dim * 2), k16((size_t)c.q * kv_dim * 2),
            v16((size_t)c.q * kv_dim * 2), o16((size_t)c.q * out_dim * 2);
        DevBuf q32((size_t)c.q * out_dim * 4), k32((size_t)c.q * kv_dim * 4),
            v32((size_t)c.q * kv_dim * 4), o32((size_t)c.q * out_dim * 4);
        q16.fill_fp16(g, -1, 1);
        k16.fill_fp16(g, -1, 1);
        v16.fill_fp16(g, -1, 1);
        q32.fill_random(g, -1, 1);
        k32.fill_random(g, -1, 1);
        v32.fill_random(g, -1, 1);

        float ms16 = time_ms(
            [&] {
                launch_flash_attention_gqa_prefill_fp16_vectorized(
                    q16.as<__half>(), k16.as<__half>(), v16.as<__half>(), o16.as<__half>(), c.q,
                    c.q, kNh, kNkv, kHd, nullptr, true, 0);
            },
            c.iters);
        float ms32 = time_ms(
            [&] {
                launch_flash_attention_gqa_vectorized(q32.as<float>(), k32.as<float>(),
                                                      v32.as<float>(), o32.as<float>(), c.q, c.q,
                                                      kNh, kNkv, kHd, nullptr, true, 0);
            },
            c.iters);
        std::printf(
            "    q=%4d : FP16 %8.4f ms (%7.0f tok/s) | FP32 %8.4f ms (%7.0f tok/s) | speedup "
            "%.2fx\n",
            c.q, ms16, c.q * 1000.0 / ms16, ms32, c.q * 1000.0 / ms32, ms32 / ms16);
    }
}

// ---- section 3: 6-A-8.0 elementwise ADD / MUL -----------------------------
// Pure streaming ops: n=4096 is launch-bound, n=16M is bandwidth-bound.
void bench_elementwise(std::mt19937& g) {
    std::printf("  [elementwise] ADD / MUL (FP16 vs FP32)\n");
    struct Cfg {
        size_t n;
        int iters;
        const char* regime;
    };
    const Cfg cfgs[] = {{4096, 1000, "launch-bound"}, {16u << 20, 50, "bandwidth-bound"}};
    for (const auto& c : cfgs) {
        DevBuf a16(c.n * 2), b16(c.n * 2), o16(c.n * 2);
        DevBuf a32(c.n * 4), b32(c.n * 4), o32(c.n * 4);
        a16.fill_fp16(g, -2, 2);
        b16.fill_fp16(g, -2, 2);
        a32.fill_random(g, -2, 2);
        b32.fill_random(g, -2, 2);

        auto bench_one = [&](auto&& launch16, auto&& launch32, const char* op) {
            float ms16 = time_ms([&] { launch16(); }, c.iters);
            float ms32 = time_ms([&] { launch32(); }, c.iters);
            // traffic: 2 reads + 1 write
            double gbs16 = 3.0 * c.n * 2 / (ms16 / 1000.0) / 1e9;
            double gbs32 = 3.0 * c.n * 4 / (ms32 / 1000.0) / 1e9;
            std::printf("    %s n=%6zu %-15s : FP16 %8.4f ms (%6.1f GB/s) | FP32 %8.4f ms "
                        "(%6.1f GB/s) | speedup %.2fx\n",
                        op, c.n, c.regime, ms16, gbs16, ms32, gbs32, ms32 / ms16);
        };
        bench_one(
            [&] { launch_add_fp16(a16.as<__half>(), b16.as<__half>(), o16.as<__half>(),
                                  (int)c.n, 0); },
            [&] { forge::cuda::launch_add_bias(a32.as<float>(), b32.as<float>(), o32.as<float>(),
                                               (int)c.n, 0); },
            "add");
        bench_one(
            [&] { launch_multiply_fp16(a16.as<__half>(), b16.as<__half>(), o16.as<__half>(),
                                       (int)c.n, 0); },
            [&] { forge::cuda::launch_multiply(a32.as<float>(), b32.as<float>(), o32.as<float>(),
                                               (int)c.n, 0); },
            "mul");
    }
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("SKIP: no CUDA device available (bench_fp16_baseline)\n");
        return 0;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::printf("[Phase 2 round 1] FP16 baseline vs FP32 path (real kernel timing)\n");
    std::printf("device: %s\n\n", prop.name);

    std::mt19937 rng(0x6A70);
    bench_decode(rng);
    std::printf("\n");
    bench_decode_h2(rng);
    std::printf("\n");
    bench_prefill(rng);
    std::printf("\n");
    bench_prefill_mma_scan(rng);
    std::printf("\n");
    bench_elementwise(rng);
    return 0;
}
