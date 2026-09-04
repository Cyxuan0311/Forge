// Phase 1-B + 1-C correctness oracle (requires CUDA; skips gracefully otherwise).
//
// 1-C: launch_output_proj_q4_0_q8_1 was changed from memset + per-warp atomicAdd
//     to a cooperative reduction (one block per row, shared-memory warp combine,
//     single write). The summed value is mathematically identical to the old
//     atomicAdd total, but the new code is deterministic (bit-exact across runs)
//     and needs no memset. This test pins that against a double-precision FP32
//     CPU reference (x dotted with the CPU-dequantized Q4_0 weight rows) and
//     checks run-to-run bit-exactness.
//
// 1-B: cuda_matmul_transB_dual's M == 1 hot path now writes b1 into the first N1
//     columns and b2 into the last N2 columns of `out` directly (no intermediate
//     tensor, no D2D interleave copies). The exact numerical + layout contract is
//     validated here by launching launch_gemv_q4_0_q8_1 for each part into its
//     slice and comparing the interleaved `out` to the CPU reference.
//
// Build & run:
//   cmake --build build -j --target forge-phase1-bc-ref-test && \
//   ./build/tests/forge-phase1-bc-ref-test

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "forge/cuda_kernels.h"
#include "operators/cuda/cuda_gemv.h"
#include "operators/cpu/common/dequant.h"

using forge::cuda::launch_output_proj_q4_0_q8_1;
using forge::cuda::launch_gemv_q4_0_q8_1;
using forge::ops::dequantize_q4_0_row;

static uint16_t halfbits(float f) {
    __half h = __float2half(f);
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return u;
}

// Q4_0 row: 18 bytes/block (fp16 scale + 16 bytes of 32 int4 nibbles). Random
// payload bytes, valid positive fp16 scale so the dot never produces NaN/Inf.
static void fill_q4_0_row(uint8_t* row, int row_bytes, int num_blocks_row, std::mt19937& rng) {
    std::uniform_int_distribution<int> u8(0, 255);
    for (int i = 0; i < row_bytes; ++i) row[i] = (uint8_t)u8(rng);
    for (int k = 0; k < num_blocks_row; ++k) {
        uint16_t s = halfbits(0.02f);
        row[k * 18] = s & 0xff;
        row[k * 18 + 1] = (s >> 8) & 0xff;
    }
}

static bool compare(const char* tag, const std::vector<float>& got,
                    const std::vector<float>& ref) {
    double dot = 0, na = 0, nb = 0, maxabs = 0, ref_scale = 0;
    int nan = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (std::isnan(got[i])) ++nan;
        dot += (double)got[i] * (double)ref[i];
        na += (double)got[i] * (double)got[i];
        nb += (double)ref[i] * (double)ref[i];
        maxabs = std::max(maxabs, std::fabs((double)got[i] - (double)ref[i]));
        ref_scale = std::max(ref_scale, std::fabs((double)ref[i]));
    }
    double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
    bool ok = (cos >= 0.9999) && (maxabs <= 1e-2 * (ref_scale + 1e-3)) && (nan == 0);
    printf("  %-12s cos=%.6f maxabs=%.3e ref_scale=%.3e nan=%d  %s\n", tag, cos, maxabs,
           ref_scale, nan, ok ? "ok" : "FAIL");
    return ok;
}

// 1-C: output_proj cooperative reduction vs FP32 reference + reproducibility.
static bool test_output_proj(int K, int N, std::mt19937& rng) {
    const int num_blocks_row = (K + 31) / 32;
    const int row_bytes = num_blocks_row * 18;
    const size_t w_bytes = (size_t)N * row_bytes;

    std::vector<uint8_t> w_host(w_bytes);
    for (int r = 0; r < N; ++r)
        fill_q4_0_row(w_host.data() + (size_t)r * row_bytes, row_bytes, num_blocks_row, rng);

    std::vector<float> x_host(K);
    std::uniform_real_distribution<float> ur(-1.0f, 1.0f);
    for (auto& v : x_host) v = ur(rng);

    float *x_gpu, *out1, *out2;
    uint8_t* w_gpu;
    cudaMalloc(&x_gpu, K * 4);
    cudaMalloc(&w_gpu, w_bytes);
    cudaMalloc(&out1, N * 4);
    cudaMalloc(&out2, N * 4);
    cudaMemcpy(x_gpu, x_host.data(), K * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(w_gpu, w_host.data(), w_bytes, cudaMemcpyHostToDevice);

    // Two runs to check bit-exact reproducibility (1-C claim: no atomicAdd).
    launch_output_proj_q4_0_q8_1(x_gpu, w_gpu, out1, K, N, 0);
    launch_output_proj_q4_0_q8_1(x_gpu, w_gpu, out2, K, N, 0);
    cudaDeviceSynchronize();

    std::vector<float> got1(N), got2(N);
    cudaMemcpy(got1.data(), out1, N * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(got2.data(), out2, N * 4, cudaMemcpyDeviceToHost);

    std::vector<float> ref(N), wrow(K);
    for (int r = 0; r < N; ++r) {
        dequantize_q4_0_row(w_host.data() + (size_t)r * row_bytes, wrow.data(), K, 0);
        double s = 0;
        for (int k = 0; k < K; ++k) s += (double)x_host[k] * (double)wrow[k];
        ref[r] = (float)s;
    }

    bool repro = true;
    for (int i = 0; i < N; ++i)
        if (got1[i] != got2[i]) repro = false;

    bool ok = compare("output_proj", got1, ref);
    printf("                 repro=%s\n", repro ? "yes" : "NO");
    ok = ok && repro;

    cudaFree(x_gpu);
    cudaFree(w_gpu);
    cudaFree(out1);
    cudaFree(out2);
    return ok;
}

// 1-B: M == 1 dual slice layout (b1 -> [0,N1), b2 -> [N1,N)) vs FP32 reference.
static bool test_dual(int K, int N1, int N2, std::mt19937& rng) {
    const int N = N1 + N2;
    const int nb1 = (K + 31) / 32, rb1 = nb1 * 18;
    const int nb2 = (K + 31) / 32, rb2 = nb2 * 18;

    auto make_w = [&](int nrows, int rb) {
        std::vector<uint8_t> w((size_t)nrows * rb);
        for (int r = 0; r < nrows; ++r) fill_q4_0_row(w.data() + (size_t)r * rb, rb, (K + 31) / 32, rng);
        return w;
    };
    auto w1 = make_w(N1, rb1);
    auto w2 = make_w(N2, rb2);

    std::vector<float> x_host(K);
    std::uniform_real_distribution<float> ur(-1.0f, 1.0f);
    for (auto& v : x_host) v = ur(rng);

    float *x_gpu, *out_gpu;
    uint8_t *w1_gpu, *w2_gpu;
    cudaMalloc(&x_gpu, K * 4);
    cudaMalloc(&w1_gpu, w1.size());
    cudaMalloc(&w2_gpu, w2.size());
    cudaMalloc(&out_gpu, N * 4);
    cudaMemcpy(x_gpu, x_host.data(), K * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(w1_gpu, w1.data(), w1.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(w2_gpu, w2.data(), w2.size(), cudaMemcpyHostToDevice);

    // M == 1 dual hot path: b1 into the first N1 columns, b2 into the last N2.
    launch_gemv_q4_0_q8_1(x_gpu, w1_gpu, out_gpu, K, N1, 0);
    launch_gemv_q4_0_q8_1(x_gpu, w2_gpu, out_gpu + N1, K, N2, 0);
    cudaDeviceSynchronize();

    std::vector<float> got(N);
    cudaMemcpy(got.data(), out_gpu, N * 4, cudaMemcpyDeviceToHost);

    std::vector<float> ref(N), wrow(K);
    for (int r = 0; r < N1; ++r) {
        dequantize_q4_0_row(w1.data() + (size_t)r * rb1, wrow.data(), K, 0);
        double s = 0;
        for (int k = 0; k < K; ++k) s += (double)x_host[k] * (double)wrow[k];
        ref[r] = (float)s;
    }
    for (int r = 0; r < N2; ++r) {
        dequantize_q4_0_row(w2.data() + (size_t)r * rb2, wrow.data(), K, 0);
        double s = 0;
        for (int k = 0; k < K; ++k) s += (double)x_host[k] * (double)wrow[k];
        ref[N1 + r] = (float)s;
    }

    bool ok = compare("dual(M=1)", got, ref);

    cudaFree(x_gpu);
    cudaFree(w1_gpu);
    cudaFree(w2_gpu);
    cudaFree(out_gpu);
    return ok;
}

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        printf("SKIP: no CUDA device available\n");
        return 0;
    }
    std::mt19937 rng(20260904);
    bool ok = true;
    ok &= test_output_proj(1024, 512, rng);
    ok &= test_output_proj(2048, 4096, rng);  // vocab-like large N
    ok &= test_dual(1024, 256, 128, rng);
    ok &= test_dual(2048, 1024, 512, rng);
    printf("%s\n", ok ? "phase1-bc-ref: PASS" : "phase1-bc-ref: FAIL");
    return ok ? 0 : 1;
}
