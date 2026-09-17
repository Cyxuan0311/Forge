// Phase 5-D: Q3_K FP16 whole-matrix dequant + TensorCore prefill wiring.
//
// Q3_K used to be excluded from tensorcore_eligible(), so every Q3_K prefill
// GEMM was pinned to the SIMT dp4a MMQ kernel and never reached the Tensor
// Cores. This test locks in the two things that unblock it:
//
//   1. launch_dequant_q3_k_matrix_fp16() agrees with the existing FP32
//      launch_dequant_q3_k_matrix() (same block walk, FP16 store).
//   2. launch_cublas_gemm_quant_fp16() accepts DataType::Q3_K and produces the
//      same result as a host FP32 golden built on the FP32 dequant — the
//      kernel is wired, just not auto-selected.
//   3. KernelSelector keeps a Q3_K prefill on MMQ: the TensorCore route
//      measured 3.2x slower (573-token prefill, RTX 4050: 8687 ms vs
//      2700 ms) because the FP16 whole-matrix dequant runs on every call.
//
// Skips gracefully without a CUDA device or without cuBLAS.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "forge/cuda_kernels.h"
#include "forge/kernel_selector.h"
#include "forge/types.h"

static int g_failures = 0;
static bool g_tc_available = false;  // set once the cuBLAS probe runs

static void check(bool ok, const char* what) {
    if (ok) {
        std::printf("  [PASS] %s\n", what);
    } else {
        std::printf("  [FAIL] %s\n", what);
        ++g_failures;
    }
}

static void skip(const char* what) {
    std::printf("  [SKIP] %s\n", what);
}

// Bits of the half value 0.0625 (2^-4): sign 0, exp 15-4 = 11, mantissa 0.
static constexpr uint16_t kHalfD = 0x2C00;

// Host-side binary16 -> binary32. __half2float() is a device intrinsic, and
// the golden is computed on the host.
static float half_to_float(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t f;
    if (exp == 0u) {
        if (mant == 0u) {
            f = sign;
        } else {
            exp = 1;  // normalise the subnormal
            while ((mant & 0x400u) == 0u) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x3FFu;
            f = sign | (static_cast<uint32_t>(exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31u) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | (static_cast<uint32_t>(exp + 127 - 15) << 23) | (mant << 13);
    }
    float out = 0.0f;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

// Fill nb Q3_K super-blocks with random payload but a finite, well-scaled d.
// Every bit pattern is a legal Q3_K block; only d needs pinning so the
// comparison is not dominated by NaN/Inf scales.
static void make_q3_k_blocks(uint8_t* p, int nb, std::mt19937& rng) {
    std::uniform_int_distribution<int> byte_dist(0, 255);
    for (int i = 0; i < nb; ++i) {
        uint8_t* b = p + static_cast<size_t>(i) * 110;
        for (int j = 0; j < 110; ++j)
            b[j] = static_cast<uint8_t>(byte_dist(rng));
        std::memcpy(b + 108, &kHalfD, sizeof(kHalfD));
    }
}

// ---- 1. FP32 vs FP16 whole-matrix dequant parity ----------------------------

static void test_dequant_parity() {
    constexpr int N = 8;
    constexpr int K = 512;  // 2 super-blocks per row
    constexpr int total = N * K;

    std::mt19937 rng(1234);
    std::vector<uint8_t> blocks(static_cast<size_t>(N) * (K / 256) * 110);
    make_q3_k_blocks(blocks.data(), static_cast<int>(blocks.size() / 110), rng);

    void* d_q = nullptr;
    float* d_f32 = nullptr;
    void* d_f16 = nullptr;
    cudaMalloc(&d_q, blocks.size());
    cudaMalloc(&d_f32, static_cast<size_t>(total) * sizeof(float));
    cudaMalloc(&d_f16, static_cast<size_t>(total) * sizeof(uint16_t));
    cudaMemcpy(d_q, blocks.data(), blocks.size(), cudaMemcpyHostToDevice);

    forge::cuda::launch_dequant_q3_k_matrix(d_q, d_f32, N, K, 0);
    forge::cuda::launch_dequant_q3_k_matrix_fp16(d_q, d_f16, N, K, 0);

    std::vector<float> h_f32(total);
    std::vector<uint16_t> h_f16(total);
    cudaMemcpy(h_f32.data(), d_f32, total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_f16.data(), d_f16, total * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    double max_abs = 0.0;
    double max_diff = 0.0;
    for (int i = 0; i < total; ++i) {
        const float got = half_to_float(h_f16[i]);
        max_abs = std::fmax(max_abs, std::fabs(h_f32[i]));
        max_diff = std::fmax(max_diff, std::fabs(h_f32[i] - got));
    }

    // FP16 carries 11 mantissa bits: 2^-11 relative, plus the subnormal floor.
    const double tol = 6e-4 * max_abs + 1e-5;
    std::printf("  dequant: max|ref|=%.6f max|diff|=%.3e tol=%.3e\n", max_abs, max_diff, tol);
    check(max_diff <= tol, "Q3_K FP16 matrix dequant matches the FP32 dequant");

    cudaFree(d_q);
    cudaFree(d_f32);
    cudaFree(d_f16);
}

// ---- 2. TensorCore GEMM (Q3_K -> FP16 -> cuBLAS) vs host FP32 golden --------

static void test_tc_gemm_parity() {
    constexpr int M = 64;
    constexpr int K = 512;
    constexpr int N = 32;

    std::mt19937 rng(4321);
    std::uniform_real_distribution<float> a_dist(-1.0f, 1.0f);

    std::vector<float> h_a(static_cast<size_t>(M) * K);
    for (auto& v : h_a)
        v = a_dist(rng);

    std::vector<uint8_t> blocks(static_cast<size_t>(N) * (K / 256) * 110);
    make_q3_k_blocks(blocks.data(), static_cast<int>(blocks.size() / 110), rng);

    void* d_q = nullptr;
    float* d_a = nullptr;
    float* d_bf32 = nullptr;
    float* d_c = nullptr;
    cudaMalloc(&d_q, blocks.size());
    cudaMalloc(&d_a, h_a.size() * sizeof(float));
    cudaMalloc(&d_bf32, static_cast<size_t>(N) * K * sizeof(float));
    cudaMalloc(&d_c, static_cast<size_t>(M) * N * sizeof(float));
    cudaMemcpy(d_q, blocks.data(), blocks.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Golden: dequantize B to FP32 on device, then dot on the host.
    forge::cuda::launch_dequant_q3_k_matrix(d_q, d_bf32, N, K, 0);
    std::vector<float> h_b(static_cast<size_t>(N) * K);
    cudaMemcpy(h_b.data(), d_bf32, h_b.size() * sizeof(float), cudaMemcpyDeviceToHost);

    const bool launched = forge::cuda::launch_cublas_gemm_quant_fp16(
        forge::DataType::Q3_K, static_cast<const float*>(d_a), d_q, static_cast<float*>(d_c), M, K,
        N, /*transB=*/true, 0);

    if (!launched) {
        skip("launch_cublas_gemm_quant_fp16(Q3_K) unavailable (no cuBLAS)");
        cudaFree(d_q);
        cudaFree(d_a);
        cudaFree(d_bf32);
        cudaFree(d_c);
        return;
    }
    g_tc_available = true;

    std::vector<float> h_c(static_cast<size_t>(M) * N);
    cudaMemcpy(h_c.data(), d_c, h_c.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    double max_abs = 0.0;
    double max_diff = 0.0;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k)
                acc += static_cast<double>(h_a[static_cast<size_t>(m) * K + k]) *
                       h_b[static_cast<size_t>(n) * K + k];
            const double got = h_c[static_cast<size_t>(m) * N + n];
            max_abs = std::fmax(max_abs, std::fabs(acc));
            max_diff = std::fmax(max_diff, std::fabs(acc - got));
        }
    }

    const double tol = 0.02 * max_abs;
    std::printf("  tc gemm: max|golden|=%.6f max|diff|=%.3e tol=%.3e\n", max_abs, max_diff, tol);
    check(max_diff <= tol, "Q3_K TensorCore GEMM matches the FP32 golden");

    cudaFree(d_q);
    cudaFree(d_a);
    cudaFree(d_bf32);
    cudaFree(d_c);
}

// ---- 3. KernelSelector wiring: Q3_K MMQ below the TC threshold, TC above ---

static void test_selector_wiring() {
    // Regression lock (Q3_K MMQ-default contract): the M==1 decode route stays
    // on dp4a GEMV and ALL prefill M stays on MMQ. Phase 6-MMQ briefly claimed
    // Q3_K wins on the dequant-FP16 + cuBLAS TC path at prefill M, but measured
    // on shipping GPUs (RTX 4050) the TC path is ~2.6x SLOWER there (the Q3_K
    // block/scale dequant does not amortize on tensor cores), so Q3_K is
    // intentionally not TC-eligible by default; FORGE_AUTOTUNE=1 can still opt
    // it back in per-signature where it actually wins.
    forge::KernelProblem p;
    p.domain = forge::KernelDomain::MatMul;
    p.dtype = forge::DataType::Q3_K;
    p.K = 512;
    p.N = 512;
    p.sm_arch = 89;

    p.M = 1;
    const forge::KernelVariant gemv = forge::select_kernel(p);
    std::printf("  selector: Q3_K M=1 -> variant=%d\n", static_cast<int>(gemv));
    check(gemv != forge::KernelVariant::MMQ_TENSORCORE,
          "Q3_K decode (M==1) never routes to TensorCore");

    p.M = 128;
    const forge::KernelVariant small_m = forge::select_kernel(p);
    std::printf("  selector: Q3_K M=128 -> variant=%d (MMQ=%d)\n", static_cast<int>(small_m),
                static_cast<int>(forge::KernelVariant::MMQ));
    check(small_m == forge::KernelVariant::MMQ,
          "Q3_K small-M prefill stays on MMQ (below tc_min_m=256)");

    p.M = 256;
    const forge::KernelVariant prefill = forge::select_kernel(p);
    std::printf("  selector: Q3_K M=256 -> variant=%d (MMQ=%d)\n", static_cast<int>(prefill),
                static_cast<int>(forge::KernelVariant::MMQ));
    check(prefill == forge::KernelVariant::MMQ,
          "Q3_K prefill at M>=tc_min_m=256 stays on MMQ (TC disabled by default)");
}

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) {
        std::printf("No CUDA device present - skipping Q3_K FP16 dequant test.\n");
        return 0;
    }

    std::printf("--- Q3_K FP16 whole-matrix dequant (Phase 5-D) ---\n");
    test_dequant_parity();
    test_tc_gemm_parity();
    test_selector_wiring();

    if (g_failures) {
        std::printf("FAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("All Q3_K FP16 dequant checks passed.\n");
    return 0;
}
