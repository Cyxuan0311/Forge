// Phase 2-C: Vectorized FP32 prefill parity gate.
//
// The Vectorized prefill kernel (flash_attn_gqa_kernel_vectorized) is a
// reformatting of the Scalar prefill kernel (flash_attn_gqa_kernel): Q is loaded
// once into shared memory (Q tile) and K/V use float4 vectorized loads, but the
// QK/V accumulation order is unchanged, so the result must be bit-exact.
//
// This test pins the contract three ways:
//   1. Vectorized vs CPU naive GQA reference  — correctness (cos >= 0.999)
//   2. Scalar     vs CPU naive GQA reference  — correctness (cos >= 0.999)
//   3. Vectorized vs Scalar                   — bit-exact reformat (maxabs ~ 0)
//
// The binary skips gracefully when no CUDA device is available.

#include <cstdio>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "operators/cuda/cuda_flash_attn.h"

using forge::cuda::launch_flash_attention_gqa;
using forge::cuda::launch_flash_attention_gqa_vectorized;

static double cosine(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += (double)a[i] * (double)b[i];
        na += (double)a[i] * (double)a[i];
        nb += (double)b[i] * (double)b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
}

static double max_abs(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs((double)a[i] - (double)b[i]));
    return m;
}

// Naive 2-pass GQA SDPA reference (causal + fp32), matching the kernel's layout:
//   Q: [q_len, num_heads, head_dim]   -> qi*num_heads*HD + h*HD
//   K,V: [kv_len, num_kv_heads, head_dim] -> j*num_kv_heads*HD + kv_h*HD
static void ref_gqa(const std::vector<float>& Q, const std::vector<float>& K,
                    const std::vector<float>& V, int q_len, int kv_len, int num_heads,
                    int num_kv_heads, int head_dim, bool causal, std::vector<float>& O) {
    O.assign((size_t)q_len * num_heads * head_dim, 0.f);
    const float scale = 1.f / std::sqrt((float)head_dim);
    const int kv_groups = num_heads / num_kv_heads;
    for (int qi = 0; qi < q_len; ++qi) {
        int q_pos = kv_len - q_len + qi;
        for (int h = 0; h < num_heads; ++h) {
            int kv_h = h / kv_groups;
            const float* q = &Q[(size_t)qi * num_heads * head_dim + h * head_dim];
            std::vector<float> scores(kv_len);
            float mx = -1e30f;
            for (int j = 0; j < kv_len; ++j) {
                if (causal && j > q_pos) {
                    scores[j] = -1e30f;
                    continue;
                }
                const float* k = &K[(size_t)j * num_kv_heads * head_dim + kv_h * head_dim];
                float dot = 0.f;
                for (int d = 0; d < head_dim; ++d) dot += q[d] * k[d];
                dot *= scale;
                scores[j] = dot;
                mx = std::max(mx, dot);
            }
            float sum = 0.f;
            for (int j = 0; j < kv_len; ++j) {
                if (scores[j] <= -1e20f) {
                    scores[j] = 0.f;
                    continue;
                }
                float e = std::exp(scores[j] - mx);
                scores[j] = e;
                sum += e;
            }
            float inv = 1.f / (sum + 1e-20f);
            float* o = &O[(size_t)qi * num_heads * head_dim + h * head_dim];
            for (int j = 0; j < kv_len; ++j) {
                if (scores[j] == 0.f) continue;
                const float* v = &V[(size_t)j * num_kv_heads * head_dim + kv_h * head_dim];
                float w = scores[j] * inv;
                for (int d = 0; d < head_dim; ++d) o[d] += w * v[d];
            }
        }
    }
}

static bool run_case(int head_dim, int q_len, int kv_len, int num_heads, int num_kv_heads,
                     bool causal, std::mt19937& rng) {
    const size_t q_elems = (size_t)q_len * num_heads * head_dim;
    const size_t k_elems = (size_t)kv_len * num_kv_heads * head_dim;

    std::vector<float> Q(q_elems), K(k_elems), V(k_elems);
    std::uniform_real_distribution<float> ur(-1.0f, 1.0f);
    for (auto& v : Q) v = ur(rng);
    for (auto& v : K) v = ur(rng);
    for (auto& v : V) v = ur(rng);

    float *d_Q = nullptr, *d_K = nullptr, *d_V = nullptr;
    float *d_os = nullptr, *d_ov = nullptr;
    cudaMalloc(&d_Q, Q.size() * 4);
    cudaMalloc(&d_K, K.size() * 4);
    cudaMalloc(&d_V, V.size() * 4);
    cudaMalloc(&d_os, q_elems * 4);
    cudaMalloc(&d_ov, q_elems * 4);
    cudaMemcpy(d_Q, Q.data(), Q.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K.data(), K.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V.data(), V.size() * 4, cudaMemcpyHostToDevice);

    launch_flash_attention_gqa(d_Q, d_K, d_V, d_os, q_len, kv_len, num_heads, num_kv_heads,
                               head_dim, nullptr, causal, 0);
    launch_flash_attention_gqa_vectorized(d_Q, d_K, d_V, d_ov, q_len, kv_len, num_heads,
                                           num_kv_heads, head_dim, nullptr, causal, 0);
    cudaDeviceSynchronize();

    std::vector<float> o_s(q_elems), o_v(q_elems);
    cudaMemcpy(o_s.data(), d_os, q_elems * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(o_v.data(), d_ov, q_elems * 4, cudaMemcpyDeviceToHost);

    std::vector<float> o_ref;
    ref_gqa(Q, K, V, q_len, kv_len, num_heads, num_kv_heads, head_dim, causal, o_ref);

    double cos_s = cosine(o_s, o_ref);
    double cos_v = cosine(o_v, o_ref);
    double maxabs = max_abs(o_s, o_v);

    bool ok = (cos_v >= 0.999) && (cos_s >= 0.999) && (maxabs <= 1e-4);
    printf("  HD=%-3d q=%d kv=%d h=%d kvh=%d causal=%d  cos(scalar)=%.6f cos(vec)=%.6f "
           "maxabs(scalar,vec)=%.3e  %s\n",
           head_dim, q_len, kv_len, num_heads, num_kv_heads, (int)causal, cos_s, cos_v, maxabs,
           ok ? "ok" : "FAIL");

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_os);
    cudaFree(d_ov);
    return ok;
}

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        printf("SKIP: no CUDA device available\n");
        return 0;
    }
    std::mt19937 rng(20260904);
    bool all_ok = true;
    for (int hd : {16, 32, 64, 128}) {
        for (bool causal : {false, true}) {
            all_ok &= run_case(hd, 32, 128, 8, 2, causal, rng);
            all_ok &= run_case(hd, 16, 256, 8, 4, causal, rng);
        }
    }
    all_ok &= run_case(64, 64, 512, 16, 4, true, rng);
    printf("%s\n", all_ok ? "attn-backend-parity: PASS" : "attn-backend-parity: FAIL");
    return all_ok ? 0 : 1;
}
