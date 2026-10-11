// Correctness oracle for the slot-indexed fused MoE GEMV (launch_moe_expert_
// {gateup,down}_slots).
//
// Reference MoE semantics (llama.cpp, and the CPU path of every engine here):
//
//   out[t] = Σ_k w[t,k] · down_{e[t,k]}( silu(gate_{e[t,k]}(x[t])) ⊙ up_{e[t,k]}(x[t]) )
//
// The nonlinearity is per expert, so gate/up results must be kept per slot
// (slot = t*n_expert_used + k). The older launch_moe_expert_gemv accumulates
// every routed expert into out[token*N + row] with atomicAdd, which can only
// express Σ_k w_k·W_{e_k}·x — routing it through gate -> activation -> down
// applies silu to the cross-expert SUM and produces garbage whenever
// n_expert_used > 1. This test pins the correct composition down across
// n_expert_used ∈ {1, 2, 8} and all five MoE weight dtypes.
//
// The reference reuses the already-oracle-tested per-dtype 2D batch GEMV (one
// call per slot, i.e. one expert at a time) and does the SiLU + weighted
// accumulation on the host in double, so it is independent of the code under
// test apart from the shared dequant primitives.
//
// The binary skips gracefully when no CUDA device is available.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "forge/cuda_kernels.h"
#include "forge/types.h"
#include "operators/cuda/cuda_gemv.h"

using forge::DataType;
using forge::cuda::launch_moe_expert_down_slots;
using forge::cuda::launch_moe_expert_gateup_slots;
using forge::cuda::launch_moe_expert_gemv;
using forge::cuda::launch_silu_multiply;

static const char* name(DataType dt) {
    switch (dt) {
    case DataType::Q4_0:
        return "Q4_0";
    case DataType::Q8_0:
        return "Q8_0";
    case DataType::Q4_K:
        return "Q4_K";
    case DataType::Q6_K:
        return "Q6_K";
    case DataType::Q5_0:
        return "Q5_0";
    default:
        return "?";
    }
}

struct QP {
    int BE, BS;
};
static QP qp(DataType dt) {
    switch (dt) {
    case DataType::Q4_0:
        return {32, 18};
    case DataType::Q8_0:
        return {32, 34};
    case DataType::Q4_K:
        return {256, 144};
    case DataType::Q6_K:
        return {256, 210};
    case DataType::Q5_0:
        return {32, 22};
    default:
        return {32, 18};
    }
}

static uint16_t halfbits(float f) {
    __half h = __float2half(f);
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return u;
}

// Fill one quantized weight row with random bytes, then sanitize the scale
// half-field(s) of EVERY block so the dot never produces NaN/Inf.
static void fill_row(uint8_t* row, int row_bytes, int num_blocks_row, DataType dt,
                     std::mt19937& rng) {
    std::uniform_int_distribution<int> u8(0, 255);
    for (int i = 0; i < row_bytes; ++i)
        row[i] = (uint8_t)u8(rng);
    auto set_half = [&](int off, float f) {
        uint16_t s = halfbits(f);
        row[off] = s & 0xff;
        row[off + 1] = (s >> 8) & 0xff;
    };
    if (dt == DataType::Q4_0) {
        for (int k = 0; k < num_blocks_row; ++k)
            set_half(k * 18, 0.02f);
    } else if (dt == DataType::Q8_0) {
        for (int k = 0; k < num_blocks_row; ++k)
            set_half(k * 34, 0.02f);
    } else if (dt == DataType::Q5_0) {
        for (int k = 0; k < num_blocks_row; ++k)
            set_half(k * 22, 0.02f);
    } else if (dt == DataType::Q4_K) {
        for (int k = 0; k < num_blocks_row; ++k) {
            set_half(k * 144 + 0, 0.02f);  // d
            set_half(k * 144 + 2, 0.01f);  // dmin
        }
    } else if (dt == DataType::Q6_K) {
        for (int k = 0; k < num_blocks_row; ++k)
            set_half(k * 210 + 208, 0.02f);  // d
    }
}

// Per-dtype 2D batch GEMV, used to build the independent per-slot reference.
template <DataType DT>
static void ref_gemv(const float* x, const void* w, float* out, int M, int K, int N,
                     cudaStream_t stream = 0) {
    if constexpr (DT == DataType::Q4_0)
        forge::cuda::launch_gemv_q4_0_q8_1_batch(x, w, out, M, K, N, stream);
    else if constexpr (DT == DataType::Q8_0)
        forge::cuda::launch_gemv_q8_0_q8_1_batch(x, w, out, M, K, N, stream);
    else if constexpr (DT == DataType::Q4_K)
        forge::cuda::launch_gemv_q4_k_q8_1_batch(x, w, out, M, K, N, stream);
    else if constexpr (DT == DataType::Q6_K)
        forge::cuda::launch_gemv_q6_k_q8_1_batch(x, w, out, M, K, N, stream);
    else if constexpr (DT == DataType::Q5_0)
        forge::cuda::launch_gemv_q5_0_q8_1_batch(x, w, out, M, K, N, stream);
}

template <DataType DT>
static bool test_case(int K, int N_ff, int hidden, int n_expert, int n_expert_used, int seq_len,
                      std::mt19937& rng) {
    const QP p = qp(DT);
    const int num_blocks_row = (K + p.BE - 1) / p.BE;
    const size_t row_bytes = (size_t)num_blocks_row * p.BS;
    const size_t slab_bytes = (size_t)N_ff * row_bytes;
    const int n_slots = seq_len * n_expert_used;

    // ---- Weights: [n_expert, N, K] expert-major (axis 0), matching the GGUF
    // ne = [K, N, E] byte layout the slot kernels index. ----
    std::vector<uint8_t> wg(n_expert * slab_bytes), wu(n_expert * slab_bytes),
        wd(n_expert * hidden * row_bytes);
    for (int e = 0; e < n_expert; ++e) {
        for (int r = 0; r < N_ff; ++r)
            fill_row(wg.data() + e * slab_bytes + r * row_bytes, (int)row_bytes, num_blocks_row, DT,
                     rng);
        for (int r = 0; r < N_ff; ++r)
            fill_row(wu.data() + e * slab_bytes + r * row_bytes, (int)row_bytes, num_blocks_row, DT,
                     rng);
        for (int r = 0; r < hidden; ++r)
            fill_row(wd.data() + e * hidden * row_bytes + r * row_bytes, (int)row_bytes,
                     num_blocks_row, DT, rng);
    }

    std::uniform_real_distribution<float> ur(-1.0f, 1.0f);
    std::vector<float> x_host((size_t)seq_len * K);
    for (auto& v : x_host)
        v = ur(rng);

    // Router decisions: k distinct experts per token, weights summing to 1.
    std::vector<int> idx(n_slots);
    std::vector<float> wgt(n_slots);
    std::uniform_int_distribution<int> pick(0, n_expert - 1);
    std::uniform_real_distribution<float> wu01(0.1f, 1.0f);
    for (int t = 0; t < seq_len; ++t) {
        std::vector<int> used;
        for (int k = 0; k < n_expert_used; ++k) {
            int e;
            do {
                e = pick(rng);
            } while (std::find(used.begin(), used.end(), e) != used.end());
            used.push_back(e);
            idx[t * n_expert_used + k] = e;
        }
        float sum = 0.0f;
        for (int k = 0; k < n_expert_used; ++k) {
            wgt[t * n_expert_used + k] = wu01(rng);
            sum += wgt[t * n_expert_used + k];
        }
        for (int k = 0; k < n_expert_used; ++k)
            wgt[t * n_expert_used + k] /= sum;
    }

    float *x_gpu, *gate_gpu, *up_gpu, *gated_gpu, *out_gpu;
    uint8_t *wg_gpu, *wu_gpu, *wd_gpu;
    int* idx_gpu;
    float* wgt_gpu;
    cudaMalloc(&x_gpu, x_host.size() * 4);
    cudaMalloc(&gate_gpu, (size_t)n_slots * N_ff * 4);
    cudaMalloc(&up_gpu, (size_t)n_slots * N_ff * 4);
    cudaMalloc(&gated_gpu, (size_t)n_slots * N_ff * 4);
    cudaMalloc(&out_gpu, (size_t)seq_len * hidden * 4);
    cudaMalloc(&wg_gpu, wg.size());
    cudaMalloc(&wu_gpu, wu.size());
    cudaMalloc(&wd_gpu, wd.size());
    cudaMalloc(&idx_gpu, idx.size() * 4);
    cudaMalloc(&wgt_gpu, wgt.size() * 4);
    cudaMemcpy(x_gpu, x_host.data(), x_host.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(wg_gpu, wg.data(), wg.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(wu_gpu, wu.data(), wu.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(wd_gpu, wd.data(), wd.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(idx_gpu, idx.data(), idx.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(wgt_gpu, wgt.data(), wgt.size() * 4, cudaMemcpyHostToDevice);
    // Poison the outputs so a kernel that forgets to write every slot fails.
    cudaMemset(gate_gpu, 0x7f, (size_t)n_slots * N_ff * 4);
    cudaMemset(up_gpu, 0x7f, (size_t)n_slots * N_ff * 4);
    cudaMemset(out_gpu, 0x7f, (size_t)seq_len * hidden * 4);

    // ---- Device path under test: per-slot gate/up -> silu -> weighted down ----
    launch_moe_expert_gateup_slots<DT>(x_gpu, wg_gpu, wu_gpu, gate_gpu, up_gpu, idx_gpu, K, N_ff,
                                       n_expert, n_expert_used, seq_len, 0);
    launch_silu_multiply(gate_gpu, up_gpu, gated_gpu, n_slots * N_ff, 0);
    launch_moe_expert_down_slots<DT>(gated_gpu, wd_gpu, out_gpu, idx_gpu, wgt_gpu, N_ff, hidden,
                                     n_expert, n_expert_used, seq_len, 0);
    cudaDeviceSynchronize();

    // ---- Reference: one expert at a time through the 2D batch GEMV, host-side
    // SiLU + weighted accumulation in double. ----
    float *tmp_gate, *tmp_up, *tmp_down, *gated_ref_gpu;
    cudaMalloc(&tmp_gate, N_ff * 4);
    cudaMalloc(&tmp_up, N_ff * 4);
    cudaMalloc(&tmp_down, hidden * 4);
    cudaMalloc(&gated_ref_gpu, (size_t)N_ff * 4);

    std::vector<float> g_ref((size_t)N_ff), u_ref((size_t)N_ff);
    std::vector<double> ref((size_t)seq_len * hidden, 0.0);
    for (int slot = 0; slot < n_slots; ++slot) {
        const int token = slot / n_expert_used;
        const int e = idx[slot];

        ref_gemv<DT>(x_gpu + (size_t)token * K, wg_gpu + (size_t)e * slab_bytes, tmp_gate, 1, K,
                     N_ff, 0);
        ref_gemv<DT>(x_gpu + (size_t)token * K, wu_gpu + (size_t)e * slab_bytes, tmp_up, 1, K, N_ff,
                     0);
        cudaDeviceSynchronize();
        cudaMemcpy(g_ref.data(), tmp_gate, N_ff * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(u_ref.data(), tmp_up, N_ff * 4, cudaMemcpyDeviceToHost);

        for (int i = 0; i < N_ff; ++i) {
            double g = g_ref[i];
            g_ref[i] = (float)(g / (1.0 + std::exp(-g)) * (double)u_ref[i]);
        }
        cudaMemcpy(gated_ref_gpu, g_ref.data(), N_ff * 4, cudaMemcpyHostToDevice);
        ref_gemv<DT>(gated_ref_gpu, wd_gpu + (size_t)e * hidden * row_bytes, tmp_down, 1, N_ff,
                     hidden, 0);
        cudaDeviceSynchronize();
        std::vector<float> d_ref((size_t)hidden);
        cudaMemcpy(d_ref.data(), tmp_down, hidden * 4, cudaMemcpyDeviceToHost);

        for (int d = 0; d < hidden; ++d)
            ref[(size_t)token * hidden + d] += (double)wgt[slot] * (double)d_ref[d];
    }

    std::vector<float> out((size_t)seq_len * hidden);
    cudaMemcpy(out.data(), out_gpu, out.size() * 4, cudaMemcpyDeviceToHost);

    double dot = 0, na = 0, nb = 0, maxabs = 0, ref_scale = 0;
    int nan_out = 0;
    for (size_t i = 0; i < out.size(); ++i) {
        if (std::isnan(out[i]))
            ++nan_out;
        dot += (double)out[i] * ref[i];
        na += (double)out[i] * (double)out[i];
        nb += ref[i] * ref[i];
        maxabs = std::max(maxabs, std::fabs((double)out[i] - ref[i]));
        ref_scale = std::max(ref_scale, std::fabs(ref[i]));
    }
    double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
    // Both paths share the same dequant primitives; only the host-vs-device
    // accumulation order differs, so this should be near bit-exact.
    bool ok = (nan_out == 0) && (cos >= 0.9999) && (maxabs <= 1e-2 * (ref_scale + 1e-3));

    // ---- Teeth check: run the legacy atomicAdd composition (gate/up summed
    // across experts, then activated) through the same reference. It must NOT
    // match once n_expert_used > 1 — that is the bug this oracle exists for.
    // Printed, not asserted: it documents why the slot kernels are required.
    double legacy_cos = -1.0;
    if (n_expert_used > 1) {
        float *lg, *lu, *lo;
        cudaMalloc(&lg, (size_t)seq_len * N_ff * 4);
        cudaMalloc(&lu, (size_t)seq_len * N_ff * 4);
        cudaMalloc(&lo, (size_t)seq_len * hidden * 4);
        cudaMemset(lg, 0, (size_t)seq_len * N_ff * 4);
        cudaMemset(lu, 0, (size_t)seq_len * N_ff * 4);
        launch_moe_expert_gemv<DT>(x_gpu, wg_gpu, lg, idx_gpu, wgt_gpu, K, N_ff, n_expert,
                                   n_expert_used, seq_len, 0, /*apply_weight=*/false);
        launch_moe_expert_gemv<DT>(x_gpu, wu_gpu, lu, idx_gpu, wgt_gpu, K, N_ff, n_expert,
                                   n_expert_used, seq_len, 0, /*apply_weight=*/false);
        launch_silu_multiply(lg, lu, lg, seq_len * N_ff, 0);
        launch_moe_expert_gemv<DT>(lg, wd_gpu, lo, idx_gpu, wgt_gpu, N_ff, hidden, n_expert,
                                   n_expert_used, seq_len, 0, /*apply_weight=*/true);
        cudaDeviceSynchronize();
        std::vector<float> lout((size_t)seq_len * hidden);
        cudaMemcpy(lout.data(), lo, lout.size() * 4, cudaMemcpyDeviceToHost);
        double ld = 0, lna = 0, lnb = 0;
        for (size_t i = 0; i < lout.size(); ++i) {
            ld += (double)lout[i] * ref[i];
            lna += (double)lout[i] * (double)lout[i];
            lnb += ref[i] * ref[i];
        }
        legacy_cos = ld / (std::sqrt(lna) * std::sqrt(lnb) + 1e-12);
        cudaFree(lg);
        cudaFree(lu);
        cudaFree(lo);
    }

    printf(
        "  %-5s k=%-2d seq=%-2d cos=%.6f maxabs=%.3e ref_scale=%.3e nan=%d legacy_cos=%.4f  %s\n",
        name(DT), n_expert_used, seq_len, cos, maxabs, ref_scale, nan_out, legacy_cos,
        ok ? "ok" : "FAIL");

    cudaFree(x_gpu);
    cudaFree(gate_gpu);
    cudaFree(up_gpu);
    cudaFree(gated_gpu);
    cudaFree(out_gpu);
    cudaFree(wg_gpu);
    cudaFree(wu_gpu);
    cudaFree(wd_gpu);
    cudaFree(idx_gpu);
    cudaFree(wgt_gpu);
    cudaFree(tmp_gate);
    cudaFree(tmp_up);
    cudaFree(tmp_down);
    cudaFree(gated_ref_gpu);
    return ok;
}

template <DataType DT>
static bool test_type(std::mt19937& rng) {
    // K/N_ff/hidden are multiples of 256 so the K-quant super-block types
    // (Q4_K/Q6_K/Q5_0) never see a ragged tail block.
    const int K = 256, N_ff = 256, hidden = 256, n_expert = 16;
    bool ok = true;
    ok &= test_case<DT>(K, N_ff, hidden, n_expert, 1, 1, rng);  // top-1 decode
    ok &= test_case<DT>(K, N_ff, hidden, n_expert, 2, 3, rng);  // multi-expert, multi-token
    ok &= test_case<DT>(K, N_ff, hidden, n_expert, 8, 1, rng);  // mellum's 8-of-64 decode
    ok &= test_case<DT>(K, N_ff, hidden, n_expert, 8, 4, rng);  // ...and prefill chunk
    return ok;
}

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        printf("SKIP: no CUDA device available\n");
        return 0;
    }
    std::mt19937 rng(20260910);

    bool all_ok = true;
    all_ok &= test_type<DataType::Q4_0>(rng);
    all_ok &= test_type<DataType::Q8_0>(rng);
    all_ok &= test_type<DataType::Q4_K>(rng);
    all_ok &= test_type<DataType::Q6_K>(rng);
    all_ok &= test_type<DataType::Q5_0>(rng);

    printf("%s\n", all_ok ? "moe-slot-ref: PASS" : "moe-slot-ref: FAIL");
    return all_ok ? 0 : 1;
}