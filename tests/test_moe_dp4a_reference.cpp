// Phase 1-A correctness oracle for the MoE expert GEMV (Q8_1 + dp4a) path.
//
// The MoE kernel was migrated from a scalar dot_block to the SAME Q8_1 × dp4a
// dot primitives used by the decode GEMV path (now shared in
// cuda_gemv_common.cuh). Because both paths use identical numerics, the MoE
// output for (n_expert=1, n_expert_used=1, weight=1) must equal the main-path
// 2D batch GEMV output for the same weights and activation — bit-for-bit except
// for harmless fp rounding in the shared quantize step. This test pins that
// contract down for all four MoE weight types, catching any Q8_1 mapping or
// 3D block-indexing bug introduced by the migration.
//
// The binary skips gracefully when no CUDA device is available.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "forge/types.h"
#include "forge/cuda_kernels.h"
#include "operators/cuda/cuda_gemv.h"

using forge::DataType;
using forge::cuda::launch_moe_expert_gemv;
using forge::cuda::launch_gemv_q4_0_q8_1_batch;
using forge::cuda::launch_gemv_q8_0_q8_1_batch;
using forge::cuda::launch_gemv_q4_k_q8_1_batch;
using forge::cuda::launch_gemv_q6_k_q8_1_batch;

static const char* name(DataType dt) {
    switch (dt) {
        case DataType::Q4_0: return "Q4_0";
        case DataType::Q8_0: return "Q8_0";
        case DataType::Q4_K: return "Q4_K";
        case DataType::Q6_K: return "Q6_K";
        default:             return "?";
    }
}

struct QP { int BE, BS; };
static QP qp(DataType dt) {
    switch (dt) {
        case DataType::Q4_0: return {32, 18};
        case DataType::Q8_0: return {32, 34};
        case DataType::Q4_K: return {256, 144};
        case DataType::Q6_K: return {256, 210};
        default:             return {32, 18};
    }
}

static uint16_t halfbits(float f) {
    __half h = __float2half(f);
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return u;
}

// Fill one quantized weight row with random data, then sanitize the scale
// half-field(s) of EVERY block so the dot never produces NaN/Inf. (We only care
// about matching the two kernels on identical bytes, not meaningful quantization.)
static void fill_row(uint8_t* row, int row_bytes, int num_blocks_row, DataType dt,
                    std::mt19937& rng) {
    std::uniform_int_distribution<int> u8(0, 255);
    for (int i = 0; i < row_bytes; ++i) row[i] = (uint8_t)u8(rng);
    auto set_half = [&](int off, float f) {
        uint16_t s = halfbits(f);
        row[off] = s & 0xff; row[off + 1] = (s >> 8) & 0xff;
    };
    if (dt == DataType::Q4_0) {
        for (int k = 0; k < num_blocks_row; ++k) set_half(k * 18, 0.02f);
    } else if (dt == DataType::Q8_0) {
        for (int k = 0; k < num_blocks_row; ++k) set_half(k * 34, 0.02f);
    } else if (dt == DataType::Q4_K) {
        for (int k = 0; k < num_blocks_row; ++k) {
            set_half(k * 144 + 0, 0.02f);  // d
            set_half(k * 144 + 2, 0.01f);  // dmin
        }
    } else if (dt == DataType::Q6_K) {
        for (int k = 0; k < num_blocks_row; ++k) set_half(k * 210 + 208, 0.02f);  // d
    }
}

template <DataType DT>
static bool test_type(int K, int N, int n_tokens, std::mt19937& rng) {
    const QP p = qp(DT);
    const int num_blocks_row = (K + p.BE - 1) / p.BE;
    const int row_bytes = num_blocks_row * p.BS;
    const size_t w_bytes = (size_t)N * row_bytes;
    const size_t out_elems = (size_t)n_tokens * N;

    // Quantized weight (n_expert = 1 -> 3D layout == 2D layout).
    std::vector<uint8_t> w_host(w_bytes);
    for (int r = 0; r < N; ++r) fill_row(w_host.data() + (size_t)r * row_bytes, row_bytes, num_blocks_row, DT, rng);

    // Activation.
    std::vector<float> x_host((size_t)n_tokens * K);
    std::uniform_real_distribution<float> ur(-1.0f, 1.0f);
    for (auto& v : x_host) v = ur(rng);

    // Router: single fixed expert, unit weight (so MoE == bare GEMV).
    std::vector<int> idx((size_t)n_tokens, 0);
    std::vector<float> wgt((size_t)n_tokens, 1.0f);

    float *x_gpu = nullptr, *out_moe = nullptr, *out_ref = nullptr;
    uint8_t* w_gpu = nullptr;
    int* idx_gpu = nullptr;
    float* wgt_gpu = nullptr;
    cudaMalloc(&x_gpu, x_host.size() * 4);
    cudaMalloc(&w_gpu, w_bytes);
    cudaMalloc(&out_moe, out_elems * 4);
    cudaMalloc(&out_ref, out_elems * 4);
    cudaMalloc(&idx_gpu, idx.size() * 4);
    cudaMalloc(&wgt_gpu, wgt.size() * 4);
    cudaMemcpy(x_gpu, x_host.data(), x_host.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(w_gpu, w_host.data(), w_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(idx_gpu, idx.data(), idx.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(wgt_gpu, wgt.data(), wgt.size() * 4, cudaMemcpyHostToDevice);
    cudaMemset(out_moe, 0, out_elems * 4);

    launch_moe_expert_gemv<DT>(x_gpu, w_gpu, out_moe, idx_gpu, wgt_gpu,
                               K, N, /*n_expert=*/1, /*n_expert_used=*/1, n_tokens, 0);

    if constexpr (DT == DataType::Q4_0)
        launch_gemv_q4_0_q8_1_batch(x_gpu, w_gpu, out_ref, n_tokens, K, N, 0);
    else if constexpr (DT == DataType::Q8_0)
        launch_gemv_q8_0_q8_1_batch(x_gpu, w_gpu, out_ref, n_tokens, K, N, 0);
    else if constexpr (DT == DataType::Q4_K)
        launch_gemv_q4_k_q8_1_batch(x_gpu, w_gpu, out_ref, n_tokens, K, N, 0);
    else if constexpr (DT == DataType::Q6_K)
        launch_gemv_q6_k_q8_1_batch(x_gpu, w_gpu, out_ref, n_tokens, K, N, 0);

    cudaDeviceSynchronize();

    std::vector<float> moe(out_elems), ref(out_elems);
    cudaMemcpy(moe.data(), out_moe, out_elems * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(ref.data(), out_ref, out_elems * 4, cudaMemcpyDeviceToHost);

    double dot = 0, na = 0, nb = 0, maxabs = 0, ref_scale = 0;
    int nan_moe = 0, nan_ref = 0;
    for (size_t i = 0; i < out_elems; ++i) {
        if (std::isnan(moe[i])) ++nan_moe;
        if (std::isnan(ref[i])) ++nan_ref;
        dot += (double)moe[i] * (double)ref[i];
        na += (double)moe[i] * (double)moe[i];
        nb += (double)ref[i] * (double)ref[i];
        maxabs = std::max(maxabs, std::fabs((double)moe[i] - (double)ref[i]));
        ref_scale = std::max(ref_scale, std::fabs((double)ref[i]));
    }
    double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
    // MoE and main-path share the exact same dot + quantize, so they should be
    // near bit-exact. Allow a tiny slack for the atomicAdd/reduction ordering.
    bool ok = (cos >= 0.9999) && (maxabs <= 1e-2 * (ref_scale + 1e-3));

    printf("  %-5s cos=%.6f maxabs=%.3e ref_scale=%.3e  %s\n",
           name(DT), cos, maxabs, ref_scale, ok ? "ok" : "FAIL");

    cudaFree(x_gpu); cudaFree(w_gpu); cudaFree(out_moe); cudaFree(out_ref);
    cudaFree(idx_gpu); cudaFree(wgt_gpu);
    return ok;
}

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        printf("SKIP: no CUDA device available\n");
        return 0;
    }
    std::mt19937 rng(20260903);
    const int K = 1024, N = 256, n_tokens = 4;

    bool all_ok = true;
    all_ok &= test_type<DataType::Q4_0>(K, N, n_tokens, rng);
    all_ok &= test_type<DataType::Q8_0>(K, N, n_tokens, rng);
    all_ok &= test_type<DataType::Q4_K>(K, N, n_tokens, rng);
    all_ok &= test_type<DataType::Q6_K>(K, N, n_tokens, rng);

    printf("%s\n", all_ok ? "moe-dp4a-ref: PASS" : "moe-dp4a-ref: FAIL");
    return all_ok ? 0 : 1;
}
