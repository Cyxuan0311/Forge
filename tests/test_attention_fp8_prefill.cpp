// Phase 2-B: FP8 (E4M3 / E5M2) fused prefill parity gate.
//
// The FP8 prefill kernel reads scaled FP8 KV from the cache and dequantizes it
// online. The golden path dequantizes the SAME FP8 bytes with the SAME
// per-(row, kv_head) scales into FP32 and runs the Vectorized FP32 prefill over
// them. Because both consume identical dequantized values, the two must agree
// to ~bit-exactness — this isolates kernel correctness (offsets, scale lookup,
// dequant, mask/causal handling) from FP8 quantization error itself.
//
// A CPU naive GQA reference over the dequantized KV is checked as well, to pin
// absolute correctness rather than only self-consistency.
//
// The binary skips gracefully when no CUDA device is available.

#include <cstdio>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "forge/cuda_kernels.h"
#include "forge/fp8_utils.h"

using forge::fp8_e4m3_to_fp32;
using forge::fp8_e5m2_to_fp32;
using forge::fp32_to_fp8_e4m3;
using forge::fp32_to_fp8_e5m2;
using forge::cuda::launch_flash_attention_gqa_vectorized;
using forge::cuda::launch_fused_flash_attention_gqa_prefill_fp8_e4m3;
using forge::cuda::launch_fused_flash_attention_gqa_prefill_fp8_e5m2;

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

// Naive 2-pass GQA SDPA reference over already-FP32 K/V.
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

// Quantize one (row, kv_head) slice with a shared scale, into the FP8 cache
// layout: bytes at [j][kv_h][d], scale at [j * num_kv_heads + kv_h].
// `amax` is the finite max of the FP8 format (240 for E4M3, 57344 for E5M2).
static void quantize_kv(const std::vector<float>& src, int kv_len, int num_kv_heads, int head_dim,
                        float amax, bool e4m3, std::vector<uint8_t>& dst_bytes,
                        std::vector<float>& dst_scales) {
    const size_t row_bytes = (size_t)num_kv_heads * head_dim;
    dst_bytes.assign((size_t)kv_len * row_bytes, 0);
    dst_scales.assign((size_t)kv_len * num_kv_heads, 1.0f);
    for (int j = 0; j < kv_len; ++j) {
        for (int kh = 0; kh < num_kv_heads; ++kh) {
            const float* row = &src[(size_t)j * num_kv_heads * head_dim + kh * head_dim];
            float mx = 0.f;
            for (int d = 0; d < head_dim; ++d) mx = std::max(mx, std::fabs(row[d]));
            float s = (mx > 0.f) ? (mx / amax) : 1.0f;
            dst_scales[(size_t)j * num_kv_heads + kh] = s;
            uint8_t* out = dst_bytes.data() + (size_t)j * row_bytes + kh * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                float q = row[d] / s;
                out[d] = e4m3 ? fp32_to_fp8_e4m3(q) : fp32_to_fp8_e5m2(q);
            }
        }
    }
}

// Dequantize the FP8 cache back to FP32, reproducing exactly what the kernel
// computes: fp8_value * per-(row, kv_head) scale.
static void dequant_kv(const std::vector<uint8_t>& bytes, const std::vector<float>& scales,
                       int kv_len, int num_kv_heads, int head_dim, bool e4m3,
                       std::vector<float>& dst) {
    const size_t row_bytes = (size_t)num_kv_heads * head_dim;
    dst.assign((size_t)kv_len * num_kv_heads * head_dim, 0.f);
    for (int j = 0; j < kv_len; ++j) {
        for (int kh = 0; kh < num_kv_heads; ++kh) {
            const uint8_t* in = bytes.data() + (size_t)j * row_bytes + kh * head_dim;
            float s = scales[(size_t)j * num_kv_heads + kh];
            float* out = &dst[(size_t)j * num_kv_heads * head_dim + kh * head_dim];
            for (int d = 0; d < head_dim; ++d)
                out[d] = (e4m3 ? fp8_e4m3_to_fp32(in[d]) : fp8_e5m2_to_fp32(in[d])) * s;
        }
    }
}

static bool run_case(bool e4m3, int head_dim, int q_len, int kv_len, int num_heads,
                     int num_kv_heads, bool causal, std::mt19937& rng) {
    const size_t q_elems = (size_t)q_len * num_heads * head_dim;
    const size_t kv_elems = (size_t)kv_len * num_kv_heads * head_dim;

    std::vector<float> Q(q_elems), K(kv_elems), V(kv_elems);
    std::uniform_real_distribution<float> ur(-1.0f, 1.0f);
    for (auto& v : Q) v = ur(rng);
    for (auto& v : K) v = ur(rng);
    for (auto& v : V) v = ur(rng);

    // Quantize K/V to FP8 with per-(row, kv_head) scales.
    const float amax = e4m3 ? 240.0f : 57344.0f;
    std::vector<uint8_t> q_K, q_V;
    std::vector<float> ks, vs;
    quantize_kv(K, kv_len, num_kv_heads, head_dim, amax, e4m3, q_K, ks);
    quantize_kv(V, kv_len, num_kv_heads, head_dim, amax, e4m3, q_V, vs);

    // Golden FP32 K/V = the same dequantized values the kernel will use.
    std::vector<float> K_deq, V_deq;
    dequant_kv(q_K, ks, kv_len, num_kv_heads, head_dim, e4m3, K_deq);
    dequant_kv(q_V, vs, kv_len, num_kv_heads, head_dim, e4m3, V_deq);

    float *d_Q = nullptr, *d_Kd = nullptr, *d_Vd = nullptr;
    uint8_t *d_qK = nullptr, *d_qV = nullptr;
    float *d_ks = nullptr, *d_vs = nullptr;
    float *d_ofp8 = nullptr, *d_ogold = nullptr;
    cudaMalloc(&d_Q, Q.size() * 4);
    cudaMalloc(&d_Kd, K_deq.size() * 4);
    cudaMalloc(&d_Vd, V_deq.size() * 4);
    cudaMalloc(&d_qK, q_K.size());
    cudaMalloc(&d_qV, q_V.size());
    cudaMalloc(&d_ks, ks.size() * 4);
    cudaMalloc(&d_vs, vs.size() * 4);
    cudaMalloc(&d_ofp8, q_elems * 4);
    cudaMalloc(&d_ogold, q_elems * 4);
    cudaMemcpy(d_Q, Q.data(), Q.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Kd, K_deq.data(), K_deq.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Vd, V_deq.data(), V_deq.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_qK, q_K.data(), q_K.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qV, q_V.data(), q_V.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ks, ks.data(), ks.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vs, vs.data(), vs.size() * 4, cudaMemcpyHostToDevice);

    const size_t q_row_size = (size_t)num_kv_heads * head_dim;  // FP8: 1 byte/element

    if (e4m3) {
        launch_fused_flash_attention_gqa_prefill_fp8_e4m3(d_Q, d_qK, d_qV, d_ofp8, q_len, kv_len,
                                                          num_heads, num_kv_heads, head_dim,
                                                          q_row_size, nullptr, causal, d_ks, d_vs,
                                                          0);
    } else {
        launch_fused_flash_attention_gqa_prefill_fp8_e5m2(d_Q, d_qK, d_qV, d_ofp8, q_len, kv_len,
                                                          num_heads, num_kv_heads, head_dim,
                                                          q_row_size, nullptr, causal, d_ks, d_vs,
                                                          0);
    }
    launch_flash_attention_gqa_vectorized(d_Q, d_Kd, d_Vd, d_ogold, q_len, kv_len, num_heads,
                                          num_kv_heads, head_dim, nullptr, causal, 0);
    cudaDeviceSynchronize();

    std::vector<float> o_fp8(q_elems), o_gold(q_elems);
    cudaMemcpy(o_fp8.data(), d_ofp8, q_elems * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(o_gold.data(), d_ogold, q_elems * 4, cudaMemcpyDeviceToHost);

    std::vector<float> o_ref;
    ref_gqa(Q, K_deq, V_deq, q_len, kv_len, num_heads, num_kv_heads, head_dim, causal, o_ref);

    double cos_gold = cosine(o_fp8, o_gold);
    double maxabs = max_abs(o_fp8, o_gold);
    double cos_ref = cosine(o_fp8, o_ref);

    bool ok = (cos_gold >= 0.9999) && (maxabs <= 1e-4) && (cos_ref >= 0.999);
    printf("  %-5s HD=%-3d q=%d kv=%d h=%d kvh=%d causal=%d  cos(fp8,gold)=%.6f "
           "maxabs=%.3e cos(fp8,cpu)=%.6f  %s\n",
           e4m3 ? "E4M3" : "E5M2", head_dim, q_len, kv_len, num_heads, num_kv_heads, (int)causal,
           cos_gold, maxabs, cos_ref, ok ? "ok" : "FAIL");

    cudaFree(d_Q); cudaFree(d_Kd); cudaFree(d_Vd); cudaFree(d_qK); cudaFree(d_qV);
    cudaFree(d_ks); cudaFree(d_vs); cudaFree(d_ofp8); cudaFree(d_ogold);
    return ok;
}

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        printf("SKIP: no CUDA device available\n");
        return 0;
    }
    std::mt19937 rng(20260905);
    bool all_ok = true;
    for (bool e4m3 : {true, false}) {
        for (int hd : {16, 32, 64, 128}) {
            for (bool causal : {false, true}) {
                all_ok &= run_case(e4m3, hd, 32, 128, 8, 2, causal, rng);
            }
        }
        all_ok &= run_case(e4m3, 128, 16, 256, 8, 4, true, rng);
    }
    printf("%s\n", all_ok ? "attn-fp8-prefill: PASS" : "attn-fp8-prefill: FAIL");
    return all_ok ? 0 : 1;
}
