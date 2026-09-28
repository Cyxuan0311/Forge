// bench_dspark_gemm.cpp -- DSpark/DFlash drafter GEMM shapes: what does the
// small-M dispatch actually cost, and what would keeping the weight in its
// native 2-byte form buy?
//
// Why this bench exists
// ---------------------
// The DSpark drafter (MiniCPM5-2.6B-DSpark.gguf) reports general.file_type 32
// (MOSTLY_BF16), but gguf_model.cpp expands every BF16 tensor to FP32 at load
// time. So the drafter's layer weights reach the matmul as FP32, which makes
// kernel_selector take the
//     if (!is_quantized_type(p.dtype)) return KernelVariant::MATMUL_CUBLAS;
// branch and land on launch_cublas_sgemm -- a plain FP32 SIMT GEMM with no
// Tensor Core involvement. In-situ profiling attributed ~170 ms of device time
// to one 5-layer drafter forward.
//
// Two arms, same shapes, same math:
//   arm "fp32"  ops::matmul_transB with an FP32 weight  == what runs TODAY
//   arm "fp16"  cuda::launch_cublas_gemm_fp16_fp32 with an FP16 copy of the
//               same weight == a proxy for "weight stayed 2 bytes and hit
//               Tensor Cores". FP16 stands in for BF16 deliberately: both are
//               2-byte and both go through cublasGemmEx with CUBLAS_COMPUTE_32F,
//               so the ratio is the decision number without needing the
//               not-yet-written BF16 variant. Measured here so the projection
//               is grounded rather than assumed.
//
// The per-op GEMM shapes come from the drafter GGUF metadata
// (dflash.embedding_length=2048, dflash.feed_forward_length=6144,
//  dflash.attention.head_count=16, head_count_kv=2, key_length=128,
//  5 layers, vocab 130560).
//
// Methodology notes (mirrors bench_prefill_gemm's 5-D discipline):
//   * cudaEventSynchronize BEFORE recording t0, so each measurement brackets
//     exactly one call. This is not cosmetic: with an async pipeline the host
//     runs ahead, and a stream event recorded "now" lands after every kernel
//     already queued -- which is what makes in-situ nested scopes read high.
//     See the note in perf_profiler.h.
//   * min / median / max over N runs, never a single latency.
//   * achieved bandwidth printed against the weight's own byte size, so the
//     numbers can be sanity-checked against the card's peak without trusting
//     any dispatch heuristic.
//
// Run: ./build/tests/forge-bench-dspark_gemm [iters]

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "forge/cuda_kernels.h"
#include "forge/operator_matmul.h"
#include "forge/tensor.h"
#include "forge/types.h"

#ifdef USE_CUDA
#include <cuda_fp16.h>
#include <cuda_runtime.h>

using forge::DataType;
using forge::DeviceType;
using forge::Tensor;
using forge::TensorPtr;

namespace {

constexpr int kHidden = 2048;
constexpr int kInter = 6144;
constexpr int kHeads = 16;
constexpr int kKvHeads = 2;
constexpr int kHeadDim = 128;
constexpr int kLayers = 5;
constexpr int kVocab = 130560;
constexpr int kQDim = kHeads * kHeadDim;        // 2048
constexpr int kKvDim = kKvHeads * kHeadDim;     // 256

struct Shape {
    const char* name;
    int K;
    int N;
    int per_layer;  // 1 if this GEMM runs once per drafter layer, 0 if once per forward
};

// Exactly the matmuls a DFlashEngine::forward_layer + the borrowed target
// lm_head issue for one drafter forward.
const Shape kShapes[] = {
    {"qkv.wq  ", kHidden, kQDim, 1},   {"qkv.wk  ", kHidden, kKvDim, 1},
    {"qkv.wv  ", kHidden, kKvDim, 1},   {"attn.wo ", kHidden, kQDim, 1},
    {"ffn.w1  ", kHidden, kInter, 1},   {"ffn.w3  ", kHidden, kInter, 1},
    {"ffn.w2  ", kInter, kQDim, 1},     {"lm_head ", kHidden, kVocab, 0},
};

const int kMs[] = {1, 2, 5, 8};

TensorPtr make_device_f32(const std::vector<float>& host, int rows, int cols) {
    auto t = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{rows, cols},
                                      DeviceType::CUDA);
    if (!host.empty())
        cudaMemcpy(t->data(), host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice);
    return t;
}

TensorPtr make_device_f16(const std::vector<__half>& host, int rows, int cols) {
    auto t = std::make_shared<Tensor>(DataType::FP16, std::vector<int64_t>{rows, cols},
                                      DeviceType::CUDA);
    if (!host.empty())
        cudaMemcpy(t->data(), host.data(), host.size() * sizeof(__half), cudaMemcpyHostToDevice);
    return t;
}

struct Timing {
    double best_ms = 0, med_ms = 0, max_ms = 0;
};

// One measurement of a single launch, bracketed by a pre-sync so the events
// cannot absorb unrelated queued work.
template <typename Fn>
Timing time_it(int iters, Fn&& fn) {
    std::vector<double> s;
    s.reserve((size_t)iters);
    for (int i = 0; i < iters; ++i) {
        cudaDeviceSynchronize();
        cudaEvent_t t0, t1;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);
        cudaEventRecord(t0, 0);
        fn();
        cudaEventRecord(t1, 0);
        cudaEventSynchronize(t1);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, t0, t1);
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
        s.push_back((double)ms);
    }
    std::vector<double> sorted = s;
    std::sort(sorted.begin(), sorted.end());
    Timing t;
    t.best_ms = sorted.front();
    t.med_ms = sorted[sorted.size() / 2];
    t.max_ms = sorted.back();
    return t;
}

// Max absolute deviation normalised by the RMS of the reference arm.
//
// A per-element relative error is meaningless here: a length-K dot product of
// zero-mean random values lands arbitrarily close to zero, so |a-b|/|a| blows
// up on a column that is merely small, not wrong. RMS-relative is the standard
// GEMM comparison and stays meaningful for K=2048.
double compare(const std::vector<float>& a, const std::vector<float>& b, int M, int N, int K) {
    double sumsq = 0.0;
    for (float v : a)
        sumsq += (double)v * (double)v;
    const double rms = std::sqrt(sumsq / std::max<size_t>(1, a.size()));
    if (rms <= 0.0)
        return 0.0;
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        worst = std::max(worst, (double)std::fabs((double)a[i] - (double)b[i]));
    (void)M;
    (void)N;
    (void)K;
    return worst / rms;
}

}  // namespace

int main(int argc, char** argv) {
    const int iters = (argc > 1) ? atoi(argv[1]) : 10;

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        printf("no CUDA device -- skipping.\n");
        return 0;
    }
    printf("DSpark drafter GEMM bench: %s (cc %d.%d), iters=%d\n", prop.name, prop.major,
           prop.minor, iters);
    printf("shapes: hidden=%d inter=%d q=%d kv=%d layers=%d vocab=%d\n", kHidden, kInter, kQDim,
           kKvDim, kLayers, kVocab);
    printf("arms:  fp32 = ops::matmul_transB w/ FP32 weight (what runs today)\n");
    printf("       fp16 = launch_cublas_gemm_fp16_fp16_fp32 path, 2-byte weight + Tensor Core\n");
    printf("       (fp16 stands in for bf16: both 2-byte, both cublasGemmEx/COMPUTE_32F)\n\n");

    std::mt19937 rng(20240928);
    std::normal_distribution<float> nd(0.0f, 0.5f);

    // Rollup of the per-forward best case, so the numbers can be compared
    // directly against the ~170 ms an in-situ sync attributed to one drafter
    // forward.
    double rollup_fp32 = 0.0, rollup_fp16 = 0.0;
    int nMs = (int)(sizeof(kMs) / sizeof(kMs[0]));

    printf("%-9s %5s %5s | %-27s | %-27s | %s\n", "shape", "K", "N", "fp32 ms (min/med/max)",
           "fp16 ms (min/med/max)", "fp32/fp16  fp32 GB/s  fp16 GB/s");
    printf("%s\n", std::string(118, '-').c_str());

    for (const Shape& sh : kShapes) {
        const int K = sh.K, N = sh.N;
        const size_t wn = (size_t)K * (size_t)N;

        // Weights: same values, FP32 master + FP16 round-trip of the same
        // numbers, so the two arms differ only in weight width / kernel.
        std::vector<float> w32(wn);
        for (size_t i = 0; i < wn; ++i)
            w32[i] = nd(rng);
        std::vector<__half> w16(wn);
        for (size_t i = 0; i < wn; ++i)
            w16[i] = __float2half(w32[i]);

        TensorPtr t_w32 = make_device_f32(w32, N, K);
        TensorPtr t_w16 = make_device_f16(w16, N, K);

        for (int mi = 0; mi < nMs; ++mi) {
            const int M = kMs[mi];
            std::vector<float> a((size_t)M * (size_t)K);
            for (auto& v : a)
                v = nd(rng);
            TensorPtr t_a2d = make_device_f32(a, M, K);
            const TensorPtr& w32_2d = t_w32;
            const TensorPtr& w16_2d = t_w16;

            TensorPtr out32;
            Timing t32 = time_it(iters, [&] { out32 = forge::ops::matmul_transB(t_a2d, w32_2d); });

            TensorPtr out16 = std::make_shared<Tensor>(
                DataType::FP32, std::vector<int64_t>{M, N}, DeviceType::CUDA);
            Timing t16 = time_it(iters, [&] {
                forge::cuda::launch_cublas_gemm_fp16_fp32(
                    static_cast<const float*>(t_a2d->data()), w16_2d->data(),
                    static_cast<float*>(out16->data()), M, K, N, /*transB=*/true, 0);
            });

            // Correctness proxy check: the fp16 arm must agree with the fp32
            // arm well inside the loose tolerance the other benches use, or it
            // is not a stand-in and the ratio means nothing.
            std::vector<float> o32((size_t)M * N), o16((size_t)M * N);
            cudaMemcpy(o32.data(), out32->data(), o32.size() * sizeof(float),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(o16.data(), out16->data(), o16.size() * sizeof(float),
                       cudaMemcpyDeviceToHost);
            const double rel = compare(o32, o16, M, N, K);

            const double gb = (double)wn * sizeof(float) / 1e9;
            const double gb16 = (double)wn * sizeof(__half) / 1e9;
            printf("%-9s %5d %5d | %6.3f %6.3f %6.3f      | %6.3f %6.3f %6.3f      | %6.2fx  %7.1f  %7.1f"
                   "  rms_rel=%.2e%s\n",
                   sh.name, K, N, t32.best_ms, t32.med_ms, t32.max_ms, t16.best_ms, t16.med_ms,
                   t16.max_ms, t32.best_ms / std::max(1e-9, t16.best_ms), gb / (t32.best_ms / 1e3),
                   gb16 / (t16.best_ms / 1e3), rel, rel > 0.05 ? "  <-- DISAGREE" : "  ok");

            // Rollup uses M = 5 (the configured n_spec) only, and only the
            // first per-layer shape set plus the once-per-forward lm_head.
            if (M == 5) {
                const int reps = sh.per_layer ? kLayers : 1;
                rollup_fp32 += reps * t32.best_ms;
                rollup_fp16 += reps * t16.best_ms;
            }
        }
        printf("\n");
    }

    printf("=== one drafter forward at M=5 (best-of-%d), summed over %d layers + lm_head ===\n",
           iters, kLayers);
    printf("  fp32 (today)          : %8.2f ms\n", rollup_fp32);
    printf("  fp16/bf16 (2-byte +TC): %8.2f ms\n", rollup_fp16);
    printf("  ratio                 : %8.2fx\n", rollup_fp32 / std::max(1e-9, rollup_fp16));
    printf("\nCompare against the ~170 ms an in-situ sync attributed to one drafter\n"
           "forward. If the fp32 rollup is far below that, the GEMMs are not the\n"
           "bottleneck and the in-situ number was measuring queue drain.\n");
    return 0;
}

#else  // !USE_CUDA

int main() {
    printf("built without CUDA -- skipping.\n");
    return 0;
}

#endif
