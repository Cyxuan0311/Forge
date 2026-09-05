// Phase 2-B + 2-C micro-benchmark: quantify the KV-tiled kernel win.
//
// The tiling removes the per-thread local_acc[HEAD_DIM] register array, so the
// structural benefit is higher occupancy on the decode path (memory bound: each
// decode streams the whole KV cache once). This benchmark reports, per format
// and context length, the decode latency (ms / token) and implied tokens/sec
// for:
//   * the streaming decode kernel (flash_attn_gqa_decode_kernel; shown under
//     its historical label "legacy" in the table below), and
//   * the new tiled kernel (decode == prefill with q_len == 1).
// It also compares tiled vs vectorized FP32 prefill at long context.
//
// Run: ./build/tests/forge-bench-tiled-attn

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "forge/cuda_kernels.h"

using forge::cuda::launch_flash_attention_gqa_decode;
using forge::cuda::launch_flash_attention_gqa_vectorized;
using forge::cuda::launch_fused_flash_attention_gqa_prefill_f16;
using forge::cuda::launch_fused_flash_attention_gqa_prefill_fp32;
using forge::cuda::launch_fused_flash_attention_gqa_prefill_q8_0;
using forge::cuda::launch_fused_flash_attention_gqa_decode_f16;
using forge::cuda::launch_fused_flash_attention_gqa_decode_q8_0;

static void check(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { fprintf(stderr, "[CUDA] %s: %s\n", w, cudaGetErrorString(e)); }
}

template <typename Fn>
static float time_ms(Fn fn, int iters) {
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
    return ms / iters;
}

// --- minimal quantizers (host) ------------------------------------------------

static void quant_f16(const std::vector<float>& src, std::vector<uint8_t>& dst) {
    dst.resize(src.size() * 2);
    for (size_t i = 0; i < src.size(); ++i) {
        __half h = __float2half(src[i]);
        std::memcpy(dst.data() + i * 2, &h, 2);
    }
}

static void quant_q8_0(const std::vector<float>& src, std::vector<uint8_t>& dst) {
    // Row-major K/V: every 32 floats within a row form one scale block. Rows are
    // multiples of 32, so a flat quantize of the whole buffer stays block-aligned.
    const size_t nblk = src.size() / 32;
    dst.assign(nblk * 34, 0);
    for (size_t b = 0; b < nblk; ++b) {
        const float* blk = &src[b * 32];
        float amax = 0;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(blk[i]));
        float s = (amax > 0) ? amax / 127.0f : 1.0f;
        uint8_t* out = dst.data() + b * 34;
        __half hs = __float2half(s);
        std::memcpy(out, &hs, 2);
        int8_t* qs = reinterpret_cast<int8_t*>(out + 2);
        for (int i = 0; i < 32; ++i) qs[i] = (int8_t)std::round(blk[i] / s);
    }
}

// Optional focus knobs so a single section/kv_len can be profiled with ncu:
//   FORGE_BENCH = decode | prefill_fp32 | prefill_q8   (default: run all)
//   FORGE_KVLEN = single kv_len to run                   (default: run all)
static bool want(const char* section) {
    const char* e = getenv("FORGE_BENCH");
    return !e || !*e || !strcmp(e, "all") || !strcmp(e, section);
}
static bool wantkv(int kv) {
    const char* e = getenv("FORGE_KVLEN");
    return !e || !*e || atoi(e) == 0 || atoi(e) == kv;
}

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        printf("SKIP: no CUDA device\n");
        return 0;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("device: %s (sm_%d%d), %.0f MiB HBM, mem bw ~%.0f GB/s\n", prop.name, prop.major,
           prop.minor, prop.totalGlobalMem / (1024.0 * 1024.0),
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6);

    const int nh = 32, nkv = 8, hd = 128, kvh_dim = nkv * hd;
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> ur(-1.0f, 1.0f);

    printf("\n=== DECODE throughput (ms/token, all %d heads; lower=better) ===\n", nh);
    printf("%-6s %-6s | %10s %10s %8s | %10s %10s %10s\n", "fmt", "kvlen", "legacy_ms",
           "tiled_ms", "speedup", "legacy_t/s", "tiled_t/s", "read GB/s");
    if (want("decode")) {
    for (int kv_len : {2048, 8192, 32768}) {
        if (!wantkv(kv_len)) continue;
        const size_t kv_elems = (size_t)kv_len * kvh_dim;
        std::vector<float> K(kv_elems), V(kv_elems), Q((size_t)nh * hd);
        for (auto& x : K) x = ur(rng);
        for (auto& x : V) x = ur(rng);
        for (auto& x : Q) x = ur(rng);

        float *d_Q, *d_K, *d_V, *d_O;
        std::vector<uint8_t> qK, qV;
        uint8_t *d_qK, *d_qV;
        cudaMalloc(&d_Q, Q.size() * 4);
        cudaMalloc(&d_K, K.size() * 4);
        cudaMalloc(&d_V, V.size() * 4);
        cudaMalloc(&d_O, Q.size() * 4);
        cudaMemcpy(d_Q, Q.data(), Q.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_K, K.data(), K.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, V.data(), V.size() * 4, cudaMemcpyHostToDevice);

        auto run_legacy_fp32 = [&]() {
            launch_flash_attention_gqa_decode(d_Q, d_K, d_V, d_O, kv_len, nh, nkv, hd, nullptr, 0);
        };
        auto run_tiled_fp32 = [&]() {
            launch_fused_flash_attention_gqa_prefill_fp32(d_Q, d_K, d_V, d_O, 1, kv_len, nh, nkv,
                                                          hd, nullptr, false, 0);
        };
        float lg = time_ms(run_legacy_fp32, 50), tg = time_ms(run_tiled_fp32, 50);

        // Quantized variants.
        quant_f16(K, qK);
        quant_f16(V, qV);
        cudaMalloc(&d_qK, qK.size());
        cudaMalloc(&d_qV, qV.size());
        cudaMemcpy(d_qK, qK.data(), qK.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_qV, qV.data(), qV.size(), cudaMemcpyHostToDevice);
        auto run_legacy_f16 = [&]() {
            launch_fused_flash_attention_gqa_decode_f16(d_Q, d_qK, d_qV, d_O, kv_len, nh, nkv, hd,
                                                        kvh_dim * 2, nullptr, 0);
        };
        auto run_tiled_f16 = [&]() {
            launch_fused_flash_attention_gqa_prefill_f16(d_Q, d_qK, d_qV, d_O, 1, kv_len, nh, nkv,
                                                         hd, kvh_dim * 2, nullptr, false, nullptr,
                                                         nullptr, 0);
        };
        float lg16 = time_ms(run_legacy_f16, 50), tg16 = time_ms(run_tiled_f16, 50);

        quant_q8_0(K, qK);
        quant_q8_0(V, qV);
        cudaMemcpy(d_qK, qK.data(), qK.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_qV, qV.data(), qV.size(), cudaMemcpyHostToDevice);
        auto run_legacy_q8 = [&]() {
            launch_fused_flash_attention_gqa_decode_q8_0(d_Q, d_qK, d_qV, d_O, kv_len, nh, nkv, hd,
                                                          (size_t)(nkv * hd / 32) * 34, nullptr, 0);
        };
        auto run_tiled_q8 = [&]() {
            launch_fused_flash_attention_gqa_prefill_q8_0(d_Q, d_qK, d_qV, d_O, 1, kv_len, nh, nkv,
                                                          hd, kvh_dim / 32 * 34, nullptr, false,
                                                          nullptr, nullptr, 0);
        };
        float lg8 = time_ms(run_legacy_q8, 50), tg8 = time_ms(run_tiled_q8, 50);

        const double bw = 2.0 * kv_elems * 4;  // FP32-equivalent K+V bytes; quantized rows read bw/2 (f16) / bw/4 (q8_0)
        printf("FP32  %-6d | %10.4f %10.4f %7.2fx | %10.1f %10.1f %10.2f\n", kv_len, lg, tg,
               lg / tg, 1000.0 / lg, 1000.0 / tg, bw / (tg * 1e-3) / 1e9);
        printf("F16   %-6d | %10.4f %10.4f %7.2fx | %10.1f %10.1f %10.2f\n", kv_len, lg16, tg16,
               lg16 / tg16, 1000.0 / lg16, 1000.0 / tg16, (bw / 2) / (tg16 * 1e-3) / 1e9);
        printf("Q8_0  %-6d | %10.4f %10.4f %7.2fx | %10.1f %10.1f %10.2f\n", kv_len, lg8, tg8,
               lg8 / tg8, 1000.0 / lg8, 1000.0 / tg8, (bw / 4) / (tg8 * 1e-3) / 1e9);

        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
        cudaFree(d_qK); cudaFree(d_qV);
    }
    }  // want("decode")

    printf("\n=== PREFILL FP32: vectorized vs tiled (q_len=512, ms; lower=better) ===\n");
    if (want("prefill_fp32")) {
    for (int kv_len : {4096, 16384, 65536}) {
        if (!wantkv(kv_len)) continue;
        const size_t kv_elems = (size_t)kv_len * kvh_dim;
        const int q_len = 512;
        std::vector<float> K(kv_elems), V(kv_elems), Q((size_t)q_len * nh * hd);
        for (auto& x : K) x = ur(rng);
        for (auto& x : V) x = ur(rng);
        for (auto& x : Q) x = ur(rng);
        float *d_Q, *d_K, *d_V, *d_O;
        cudaMalloc(&d_Q, Q.size() * 4);
        cudaMalloc(&d_K, K.size() * 4);
        cudaMalloc(&d_V, V.size() * 4);
        cudaMalloc(&d_O, Q.size() * 4);
        cudaMemcpy(d_Q, Q.data(), Q.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_K, K.data(), K.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, V.data(), V.size() * 4, cudaMemcpyHostToDevice);
        auto run_vec = [&]() {
            launch_flash_attention_gqa_vectorized(d_Q, d_K, d_V, d_O, q_len, kv_len, nh, nkv, hd,
                                                  nullptr, false, 0);
        };
        auto run_tiled = [&]() {
            launch_fused_flash_attention_gqa_prefill_fp32(d_Q, d_K, d_V, d_O, q_len, kv_len, nh,
                                                          nkv, hd, nullptr, false, 0);
        };
        float v = time_ms(run_vec, 20), t = time_ms(run_tiled, 20);
        printf("  kv=%d q=%d  vectorized=%.3f ms  tiled=%.3f ms  speedup=%.2fx\n", kv_len, q_len, v,
               t, v / t);
        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    }
    }  // want("prefill_fp32")

    printf("\n=== PREFILL Q8_0: tiled (no materialize) vs materialize+vectorized (q_len=512, ms) ===\n");
    if (want("prefill_q8")) {
    auto dequant_q8_0 = [](const std::vector<uint8_t>& src) {
        std::vector<float> out(src.size() / 34 * 32, 0.f);
        int nblk = (int)out.size() / 32;
        for (int b = 0; b < nblk; ++b) {
            const uint8_t* in = src.data() + b * 34;
            uint16_t sb;
            std::memcpy(&sb, in, 2);
            float s = __half2float(*(const __half*)&sb);
            const int8_t* qs = reinterpret_cast<const int8_t*>(in + 2);
            for (int i = 0; i < 32; ++i) out[b * 32 + i] = qs[i] * s;
        }
        return out;
    };
    for (int kv_len : {4096, 16384, 65536}) {
        if (!wantkv(kv_len)) continue;
        const size_t kv_elems = (size_t)kv_len * kvh_dim;
        const int q_len = 512;
        std::vector<float> K(kv_elems), V(kv_elems), Q((size_t)q_len * nh * hd);
        for (auto& x : K) x = ur(rng);
        for (auto& x : V) x = ur(rng);
        for (auto& x : Q) x = ur(rng);
        std::vector<uint8_t> qK, qV;
        quant_q8_0(K, qK);
        quant_q8_0(V, qV);

        float *d_Q, *d_qK, *d_qV, *d_O, *d_Kf, *d_Vf;
        cudaMalloc(&d_Q, Q.size() * 4);
        cudaMalloc(&d_qK, qK.size());
        cudaMalloc(&d_qV, qV.size());
        cudaMalloc(&d_O, Q.size() * 4);
        cudaMalloc(&d_Kf, kv_elems * 4);
        cudaMalloc(&d_Vf, kv_elems * 4);
        cudaMemcpy(d_Q, Q.data(), Q.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_qK, qK.data(), qK.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_qV, qV.data(), qV.size(), cudaMemcpyHostToDevice);
        std::vector<float> Kf = dequant_q8_0(qK), Vf = dequant_q8_0(qV);
        cudaMemcpy(d_Kf, Kf.data(), kv_elems * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_Vf, Vf.data(), kv_elems * 4, cudaMemcpyHostToDevice);

        auto run_tiled = [&]() {
            launch_fused_flash_attention_gqa_prefill_q8_0(d_Q, d_qK, d_qV, d_O, q_len, kv_len, nh,
                                                          nkv, hd, kvh_dim / 32 * 34, nullptr, false,
                                                          nullptr, nullptr, 0);
        };
        auto run_mat = [&]() {
            launch_flash_attention_gqa_vectorized(d_Q, d_Kf, d_Vf, d_O, q_len, kv_len, nh, nkv, hd,
                                                  nullptr, false, 0);
        };
        float tg = time_ms(run_tiled, 20), tm = time_ms(run_mat, 20);
        printf("  kv=%d q=%d  tiled_q8_0=%.3f ms  materialize+vec=%.3f ms  speedup=%.2fx\n", kv_len,
               q_len, tg, tm, tm / tg);

        cudaFree(d_Q); cudaFree(d_qK); cudaFree(d_qV); cudaFree(d_O); cudaFree(d_Kf); cudaFree(d_Vf);
    }
    }  // want("prefill_q8")
    return 0;
}
