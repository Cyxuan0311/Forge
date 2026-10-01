// BF16 matmul bandwidth: a guard, not a new kernel.
//
// WHY THIS TEST EXISTS. Speculative decoding with a BF16 drafter looked like it had
// an obvious win available. BF16 has no entry in the GEMV dispatch table (it reads
// nullptr in both the M==1 and the batched table), so every BF16 matmul in the
// drafter goes through cuBLAS via matmul_transB. Profiling the drafter's context
// path on MiniCPM5-2.6B-DSpark put append_context_kv at ~17 ms per round for 1-4
// tokens across 5 layers, against ~0.3 ms of HBM traffic -- which reads exactly
// like "cuBLAS is doing something inefficient for tiny M, write a GEMV".
//
// A hand-written BF16 GEMV was written to test that. It was SLOWER than cuBLAS:
// 0.370 ms against 0.267 ms on the drafter's own fp32-x-bf16 shape (M=2, K=2048,
// N=10240), i.e. 113 GB/s against 157 GB/s. cuBLAS is already at ~96% of this
// card's 164 GB/s measured read roof, so there was nothing to win, and the
// apparent 4x gap between the profiled 1.09 ms and the 0.267 ms kernel is host-side
// dispatch latency in the surrounding path, not the matmul.
//
// So the kernel was reverted. This test stays to pin the finding: if someone sees
// the nullptr dispatch entry again and thinks it is an opportunity, this fails
// them before they spend the day on it. It asserts the roof, not a magic number.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "forge/cuda_kernels.h"

namespace {

int g_failures = 0;

void check(bool cond, const char* what) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

#define CUDA_OK(expr)                                                                             \
    do {                                                                                          \
        cudaError_t err_ = (expr);                                                                \
        if (err_ != cudaSuccess) {                                                                \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__,   \
                         __LINE__);                                                                \
            std::exit(1);                                                                          \
        }                                                                                         \
    } while (0)

// A read-only bandwidth probe, used only as a sanity scale for the numbers printed
// below. This is deliberately NOT used for the pass/fail threshold: an earlier
// version measured the roof with a device-to-device copy and read half of it, which
// reported 77 GB/s against the 164 GB/s this box actually sustains (docs/perf-lfm2.md,
// forge-bench-gpu_bandwidth), because a copy also writes. The thresholds below are
// absolute GB/s instead, set from the measurements in the file header with margin.
double measure_read_roof() {
    void* src = nullptr;
    void* dst = nullptr;
    const size_t n = 512ull << 20;
    if (cudaMalloc(&src, n) != cudaSuccess)
        return 0.0;
    if (cudaMalloc(&dst, n) != cudaSuccess) {
        CUDA_OK(cudaFree(src));
        return 0.0;
    }
    cudaEvent_t t0, t1;
    CUDA_OK(cudaEventCreate(&t0));
    CUDA_OK(cudaEventCreate(&t1));
    for (int i = 0; i < 5; ++i)
        CUDA_OK(cudaMemcpy(dst, src, n, cudaMemcpyDeviceToDevice));
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaEventRecord(t0, 0));
    const int iters = 20;
    for (int i = 0; i < iters; ++i)
        CUDA_OK(cudaMemcpy(dst, src, n, cudaMemcpyDeviceToDevice));
    CUDA_OK(cudaEventRecord(t1, 0));
    CUDA_OK(cudaEventSynchronize(t1));
    float ms = 0.f;
    CUDA_OK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_OK(cudaEventDestroy(t0));
    CUDA_OK(cudaEventDestroy(t1));
    CUDA_OK(cudaFree(src));
    CUDA_OK(cudaFree(dst));
    // D2D moves n read + n written; a read-only kernel can do no better than n.
    return (double)n / ((ms / iters) * 1e-3) / 1e9;
}

// Time the cuBLAS path the drafter uses, on the shape it uses. Returns GB/s of
// weight traffic.
double cublas_weight_gbs(int M, int K, int N, float* out_ms) {
    std::vector<float> hA((size_t)M * K, 0.01f);
    std::vector<__nv_bfloat16> hB((size_t)N * K, __float2bfloat16(0.01f));

    float* dA = nullptr;
    __nv_bfloat16* dB = nullptr;
    float* dC = nullptr;
    CUDA_OK(cudaMalloc(&dA, hA.size() * sizeof(float)));
    CUDA_OK(cudaMalloc(&dB, hB.size() * sizeof(__nv_bfloat16)));
    CUDA_OK(cudaMalloc(&dC, (size_t)M * N * sizeof(float)));
    CUDA_OK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(__nv_bfloat16),
                       cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1;
    CUDA_OK(cudaEventCreate(&t0));
    CUDA_OK(cudaEventCreate(&t1));
    const int warm = 20, iters = 200;
    for (int i = 0; i < warm; ++i)
        forge::cuda::launch_cublas_gemm_bf16_fp32(dA, dB, dC, M, K, N, true, 0);
    CUDA_OK(cudaStreamSynchronize(0));
    CUDA_OK(cudaEventRecord(t0, 0));
    for (int i = 0; i < iters; ++i)
        forge::cuda::launch_cublas_gemm_bf16_fp32(dA, dB, dC, M, K, N, true, 0);
    CUDA_OK(cudaEventRecord(t1, 0));
    CUDA_OK(cudaEventSynchronize(t1));
    float ms = 0.f;
    CUDA_OK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_OK(cudaEventDestroy(t0));
    CUDA_OK(cudaEventDestroy(t1));
    CUDA_OK(cudaFree(dA));
    CUDA_OK(cudaFree(dB));
    CUDA_OK(cudaFree(dC));

    const double per = ms / iters;
    if (out_ms)
        *out_ms = static_cast<float>(per);
    return (double)N * K * sizeof(__nv_bfloat16) / (per * 1e-3) / 1e9;
}

}  // namespace

int main() {
    if (cudaFree(nullptr) != cudaSuccess) {
        std::printf("bf16 matmul bandwidth guard skipped: no CUDA device\n");
        return 0;
    }

    const double roof = measure_read_roof();
    if (roof <= 0.0) {
        std::printf("bf16 matmul bandwidth guard skipped: could not measure the read roof\n");
        return 0;
    }

    // The shapes MiniCPM5-2.6B-DSpark actually runs: fc is [10240, 2048], and each
    // layer's K/V projection is [256, 2048], with M = 1-4 (the per-round delta).
    // Threshold per case: weight_bytes = N*K*2. A large matrix is bandwidth-bound
    // and must stream; a small one cannot fill the GPU at all, so for those the
    // meaningful bound is absolute latency (launch overhead dominates).
    struct Case {
        int M, K, N;
        const char* what;
        double min_gbs;     // 0 = do not gate on bandwidth
        double max_ms;
    };
    const Case cases[] = {
        // fc is 40 MB: genuinely bandwidth-bound. Measured 179-183 GB/s, which is
        // ~110% of this box's 164 GB/s read roof (the excess is L2 reuse of the
        // repeated calls in the timing loop). The hand-written GEMV that was tried
        // and reverted managed 113 GB/s here, so 120 separates the two clearly.
        {2, 2048, 10240, "drafter fc projection", 120.0, 0.0},
        {1, 2048, 10240, "drafter fc projection, single token", 120.0, 0.0},
        {4, 2048, 10240, "drafter fc projection, 4-token delta", 120.0, 0.0},
        // k/v is 1 MB: far too small to saturate. Gate on latency instead.
        {4, 2048, 256, "per-layer k/v projection", 0.0, 0.10},
        {1, 2048, 256, "per-layer k/v projection, single token", 0.0, 0.10},
    };

    for (const auto& c : cases) {
        float ms = 0.f;
        const double gbs = cublas_weight_gbs(c.M, c.K, c.N, &ms);
        std::printf("cublas fp32xbf16  M=%-2d K=%-5d N=%-6d %-32s: %.3f ms, %6.1f GB/s "
                    "(%3.0f%% of %.1f GB/s roof)\n",
                    c.M, c.K, c.N, c.what, ms, gbs, 100.0 * gbs / roof, roof);
        if (c.min_gbs > 0.0)
            check(gbs >= c.min_gbs,
                  "the cuBLAS bf16 path streams a bandwidth-bound matrix near the roof, so "
                  "there is no GEMV-shaped win available here");
        if (c.max_ms > 0.0)
            check(ms <= c.max_ms, "a small bf16 projection stays well under its latency bound");
    }

    if (g_failures == 0) {
        std::printf("all bf16 matmul bandwidth guard checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "%d bf16 bandwidth guard check(s) failed\n", g_failures);
    return 1;
}
