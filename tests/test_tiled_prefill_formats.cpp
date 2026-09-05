// Phase 2-B + 2-C: KV-tiled prefill across every supported KV format.
//
// The tiled prefill kernel (cuda_tiled_prefill.cu) is parameterised by a KV
// format policy, so the same template serves FP32 / F16 / Q8_0 / FP8-E4M3 /
// FP8-E5M2, contiguous or paged. This test pins all of them:
//
//   1. Correctness: each format's prefill is compared against a CPU naive GQA
//      reference run over the *dequantized* KV (absolute correctness).
//   2. Self-consistency: each format is compared against the FP32 vectorized
//      prefill over the same dequantized KV. The kernel consumes exactly those
//      dequantized values, so agreement must be tight (~1e-6).
//   3. Paged vs contiguous: the paged variant must reproduce the contiguous
//      result for identical KV content (validates page-table row addressing).
//
// The binary skips gracefully when no CUDA device is available.

#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "forge/cuda_kernels.h"
#include "forge/fp8_utils.h"

using forge::cuda::launch_flash_attention_gqa_vectorized;
using forge::cuda::launch_flash_attention_gqa_decode;
using forge::cuda::launch_fused_flash_attention_gqa_prefill_f16;
using forge::cuda::launch_fused_flash_attention_gqa_prefill_fp32;
using forge::cuda::launch_fused_flash_attention_gqa_prefill_fp8_e4m3;
using forge::cuda::launch_fused_flash_attention_gqa_prefill_fp8_e5m2;
using forge::cuda::launch_fused_flash_attention_gqa_prefill_q8_0;
using forge::cuda::launch_fused_flash_attention_gqa_prefill_q4_0;
using forge::cuda::launch_fused_flash_attention_gqa_decode_f16;
using forge::cuda::launch_fused_flash_attention_gqa_decode_fp8_e4m3;
using forge::cuda::launch_fused_flash_attention_gqa_decode_fp8_e5m2;
using forge::cuda::launch_fused_flash_attention_gqa_decode_q8_0;
using forge::cuda::launch_fused_flash_attention_gqa_decode_q4_0;
using forge::cuda::launch_paged_flash_attention_gqa_prefill_f16;
using forge::cuda::launch_paged_flash_attention_gqa_prefill_fp32;
using forge::cuda::launch_paged_flash_attention_gqa_prefill_fp8_e4m3;
using forge::cuda::launch_paged_flash_attention_gqa_prefill_fp8_e5m2;
using forge::cuda::launch_paged_flash_attention_gqa_prefill_q8_0;
using forge::cuda::launch_paged_flash_attention_gqa_prefill_q4_0;

namespace {

enum class Fmt { Fp32, Q4_0, F16, Q8_0, Fp8E4M3, Fp8E5M2 };

const char* fmt_name(Fmt f) {
    switch (f) {
    case Fmt::Fp32:     return "FP32";
    case Fmt::Q4_0:     return "Q4_0";
    case Fmt::F16:      return "F16";
    case Fmt::Q8_0:     return "Q8_0";
    case Fmt::Fp8E4M3:  return "E4M3";
    case Fmt::Fp8E5M2:  return "E5M2";
    default:            return "?";
    }
}

// Minimum head_dim per format (Q4_0/Q8_0 quantize in 32-element blocks).
int fmt_min_hd(Fmt f) {
    return (f == Fmt::Q8_0 || f == Fmt::Q4_0) ? 32 : 16;
}

// Bytes per KV row for `nkv` heads of `hd` elements.
size_t fmt_row_bytes(Fmt f, int nkv, int hd) {
    switch (f) {
    case Fmt::Fp32:    return (size_t)nkv * hd * sizeof(float);
    case Fmt::Q4_0:    return (size_t)((nkv * hd + 31) / 32) * 18;
    case Fmt::F16:     return (size_t)nkv * hd * sizeof(__half);
    case Fmt::Q8_0:    return (size_t)((nkv * hd + 31) / 32) * 34;
    case Fmt::Fp8E4M3:
    case Fmt::Fp8E5M2: return (size_t)nkv * hd;
    default:           return 0;
    }
}

// ---- Host-side quantization -------------------------------------------------

void quantize(Fmt f, const std::vector<float>& src, int rows, int nkv, int hd,
              std::vector<uint8_t>& dst, std::vector<float>& k_scales,
              std::vector<float>& v_scales_unused) {
    (void)v_scales_unused;
    const size_t kv_dim = (size_t)nkv * hd;
    const size_t rb = fmt_row_bytes(f, nkv, hd);
    dst.assign(rows * rb, 0);

    if (f == Fmt::Fp32) {
        std::memcpy(dst.data(), src.data(), src.size() * sizeof(float));
        return;
    }
    if (f == Fmt::F16) {
        for (size_t i = 0; i < src.size(); ++i) {
            __half h = __float2half(src[i]);
            std::memcpy(dst.data() + i * sizeof(__half), &h, sizeof(__half));
        }
        return;
    }
    if (f == Fmt::Q4_0) {
        // Blocks of 32; hd % 32 == 0 keeps head boundaries block-aligned,
        // matching the kernel's kv_h * (hd/32) * 18 offset. Values are stored
        // as unsigned 4-bit (val + 8) so the kernel can subtract 8.
        const int nblk = (nkv * hd) / 32;
        for (int r = 0; r < rows; ++r) {
            for (int b = 0; b < nblk; ++b) {
                const float* blk = &src[(size_t)r * kv_dim + (size_t)b * 32];
                float amax = 0.f;
                for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(blk[i]));
                const float s = (amax > 0.f) ? (amax / 8.0f) : 1.0f;
                uint8_t* out = dst.data() + (size_t)r * rb + (size_t)b * 18;
                __half hs = __float2half(s);
                std::memcpy(out, &hs, sizeof(__half));
                uint8_t* qs = out + 2;
                // Engine layout: byte i packs element i (low) and element i+16 (high).
                for (int i = 0; i < 16; ++i) {
                    int q0 = static_cast<int>(std::round(blk[i] / s));
                    int q1 = static_cast<int>(std::round(blk[i + 16] / s));
                    q0 = std::min(7, std::max(-8, q0)) + 8;
                    q1 = std::min(7, std::max(-8, q1)) + 8;
                    qs[i] = static_cast<uint8_t>((q0 & 0x0F) | ((q1 & 0x0F) << 4));
                }
            }
        }
        return;
    }
    if (f == Fmt::Q8_0) {
        // Blocks of 32 over the whole row; hd % 32 == 0 keeps head boundaries
        // block-aligned, matching the kernel's kv_h * (hd/32) * 34 offset.
        const int nblk = (nkv * hd) / 32;
        for (int r = 0; r < rows; ++r) {
            for (int b = 0; b < nblk; ++b) {
                const float* blk = &src[(size_t)r * kv_dim + (size_t)b * 32];
                float amax = 0.f;
                for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(blk[i]));
                const float s = (amax > 0.f) ? (amax / 127.0f) : 1.0f;
                uint8_t* out = dst.data() + (size_t)r * rb + (size_t)b * 34;
                __half hs = __float2half(s);
                std::memcpy(out, &hs, sizeof(__half));
                int8_t* qs = reinterpret_cast<int8_t*>(out + 2);
                for (int i = 0; i < 32; ++i)
                    qs[i] = static_cast<int8_t>(std::round(blk[i] / s));
            }
        }
        return;
    }
    // FP8: one scale per (row, kv_head) over `hd` elements.
    const float amax_fmt = (f == Fmt::Fp8E4M3) ? 240.0f : 57344.0f;
    k_scales.assign((size_t)rows * nkv, 1.0f);
    for (int r = 0; r < rows; ++r) {
        for (int kh = 0; kh < nkv; ++kh) {
            const float* row = &src[(size_t)r * kv_dim + (size_t)kh * hd];
            float amax = 0.f;
            for (int d = 0; d < hd; ++d) amax = std::max(amax, std::fabs(row[d]));
            const float s = (amax > 0.f) ? (amax / amax_fmt) : 1.0f;
            k_scales[(size_t)r * nkv + kh] = s;
            uint8_t* out = dst.data() + (size_t)r * rb + (size_t)kh * hd;
            for (int d = 0; d < hd; ++d) {
                float q = row[d] / s;
                out[d] = (f == Fmt::Fp8E4M3) ? forge::fp32_to_fp8_e4m3(q)
                                             : forge::fp32_to_fp8_e5m2(q);
            }
        }
    }
}

// ---- Host-side dequantization (mirrors the kernel's policy) -----------------

void dequantize(Fmt f, const std::vector<uint8_t>& src, int rows, int nkv, int hd,
                const std::vector<float>& scales, std::vector<float>& dst) {
    const size_t kv_dim = (size_t)nkv * hd;
    const size_t rb = fmt_row_bytes(f, nkv, hd);
    dst.assign((size_t)rows * kv_dim, 0.f);

    if (f == Fmt::Fp32) {
        std::memcpy(dst.data(), src.data(), dst.size() * sizeof(float));
        return;
    }
    if (f == Fmt::F16) {
        for (size_t i = 0; i < dst.size(); ++i) {
            __half h;
            std::memcpy(&h, src.data() + i * sizeof(__half), sizeof(__half));
            dst[i] = __half2float(h);
        }
        return;
    }
    if (f == Fmt::Q4_0) {
        const int nblk = (nkv * hd) / 32;
        for (int r = 0; r < rows; ++r) {
            for (int b = 0; b < nblk; ++b) {
                const uint8_t* in = src.data() + (size_t)r * rb + (size_t)b * 18;
                __half hs;
                std::memcpy(&hs, in, sizeof(__half));
                const float s = __half2float(hs);
                const uint8_t* qs = in + 2;
                // Engine layout: byte i packs element i (low) and element i+16 (high).
                for (int i = 0; i < 16; ++i) {
                    const int lo = (qs[i] & 0x0F) - 8;
                    const int hi = ((qs[i] >> 4) & 0x0F) - 8;
                    dst[(size_t)r * kv_dim + (size_t)b * 32 + i] = lo * s;
                    dst[(size_t)r * kv_dim + (size_t)b * 32 + i + 16] = hi * s;
                }
            }
        }
        return;
    }
    if (f == Fmt::Q8_0) {
        const int nblk = (nkv * hd) / 32;
        for (int r = 0; r < rows; ++r) {
            for (int b = 0; b < nblk; ++b) {
                const uint8_t* in = src.data() + (size_t)r * rb + (size_t)b * 34;
                __half hs;
                std::memcpy(&hs, in, sizeof(__half));
                const float s = __half2float(hs);
                const int8_t* qs = reinterpret_cast<const int8_t*>(in + 2);
                for (int i = 0; i < 32; ++i) dst[(size_t)r * kv_dim + (size_t)b * 32 + i] = qs[i] * s;
            }
        }
        return;
    }
    for (int r = 0; r < rows; ++r) {
        for (int kh = 0; kh < nkv; ++kh) {
            const uint8_t* in = src.data() + (size_t)r * rb + (size_t)kh * hd;
            const float s = scales[(size_t)r * nkv + kh];
            for (int d = 0; d < hd; ++d) {
                float v = (f == Fmt::Fp8E4M3) ? forge::fp8_e4m3_to_fp32(in[d])
                                              : forge::fp8_e5m2_to_fp32(in[d]);
                dst[(size_t)r * kv_dim + (size_t)kh * hd + d] = v * s;
            }
        }
    }
}

// ---- CPU naive GQA reference ------------------------------------------------

void ref_gqa(const std::vector<float>& Q, const std::vector<float>& K,
             const std::vector<float>& V, int q_len, int kv_len, int num_heads, int num_kv_heads,
             int head_dim, bool causal, std::vector<float>& O) {
    O.assign((size_t)q_len * num_heads * head_dim, 0.f);
    const float scale = 1.f / std::sqrt((float)head_dim);
    const int kv_groups = num_heads / num_kv_heads;
    for (int qi = 0; qi < q_len; ++qi) {
        const int q_pos = kv_len - q_len + qi;
        for (int h = 0; h < num_heads; ++h) {
            const int kv_h = h / kv_groups;
            const float* q = &Q[(size_t)qi * num_heads * head_dim + h * head_dim];
            std::vector<float> sc(kv_len);
            float mx = -1e30f;
            for (int j = 0; j < kv_len; ++j) {
                if (causal && j > q_pos) { sc[j] = -1e30f; continue; }
                const float* k = &K[(size_t)j * num_kv_heads * head_dim + kv_h * head_dim];
                float dot = 0.f;
                for (int d = 0; d < head_dim; ++d) dot += q[d] * k[d];
                dot *= scale;
                sc[j] = dot;
                mx = std::max(mx, dot);
            }
            float sum = 0.f;
            for (int j = 0; j < kv_len; ++j) {
                if (sc[j] <= -1e20f) { sc[j] = 0.f; continue; }
                float e = std::exp(sc[j] - mx);
                sc[j] = e;
                sum += e;
            }
            const float inv = 1.f / (sum + 1e-20f);
            float* o = &O[(size_t)qi * num_heads * head_dim + h * head_dim];
            for (int j = 0; j < kv_len; ++j) {
                if (sc[j] == 0.f) continue;
                const float* v = &V[(size_t)j * num_kv_heads * head_dim + kv_h * head_dim];
                const float w = sc[j] * inv;
                for (int d = 0; d < head_dim; ++d) o[d] += w * v[d];
            }
        }
    }
}

double cosine(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += (double)a[i] * (double)b[i];
        na += (double)a[i] * (double)a[i];
        nb += (double)b[i] * (double)b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
}

double max_abs(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs((double)a[i] - (double)b[i]));
    return m;
}

// ---- Contiguous format case -------------------------------------------------

bool run_contiguous(Fmt fmt, int hd, int q_len, int kv_len, int nh, int nkv, bool causal,
                    std::mt19937& rng) {
    const size_t q_elems = (size_t)q_len * nh * hd;
    const size_t kv_elems = (size_t)kv_len * nkv * hd;

    std::vector<float> Q(q_elems), K(kv_elems), V(kv_elems);
    std::uniform_real_distribution<float> ur(-1.0f, 1.0f);
    for (auto& v : Q) v = ur(rng);
    for (auto& v : K) v = ur(rng);
    for (auto& v : V) v = ur(rng);

    std::vector<uint8_t> qK, qV;
    std::vector<float> ks, vs, unused;
    quantize(fmt, K, kv_len, nkv, hd, qK, ks, unused);
    quantize(fmt, V, kv_len, nkv, hd, qV, vs, unused);

    std::vector<float> Kd, Vd;
    dequantize(fmt, qK, kv_len, nkv, hd, ks, Kd);
    dequantize(fmt, qV, kv_len, nkv, hd, vs, Vd);

    float *d_Q = nullptr, *d_Kd = nullptr, *d_Vd = nullptr;
    float *d_ofmt = nullptr, *d_ogold = nullptr;
    uint8_t *d_qK = nullptr, *d_qV = nullptr;
    float *d_ks = nullptr, *d_vs = nullptr;
    cudaMalloc(&d_Q, Q.size() * 4);
    cudaMalloc(&d_Kd, Kd.size() * 4);
    cudaMalloc(&d_Vd, Vd.size() * 4);
    cudaMalloc(&d_ofmt, q_elems * 4);
    cudaMalloc(&d_ogold, q_elems * 4);
    cudaMalloc(&d_qK, qK.size());
    cudaMalloc(&d_qV, qV.size());
    cudaMemcpy(d_Q, Q.data(), Q.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Kd, Kd.data(), Kd.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Vd, Vd.data(), Vd.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_qK, qK.data(), qK.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qV, qV.data(), qV.size(), cudaMemcpyHostToDevice);

    const bool fp8 = (fmt == Fmt::Fp8E4M3 || fmt == Fmt::Fp8E5M2);
    if (fp8) {
        cudaMalloc(&d_ks, ks.size() * 4);
        cudaMalloc(&d_vs, vs.size() * 4);
        cudaMemcpy(d_ks, ks.data(), ks.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_vs, vs.data(), vs.size() * 4, cudaMemcpyHostToDevice);
    }

    const size_t rb = fmt_row_bytes(fmt, nkv, hd);
    switch (fmt) {
    case Fmt::Fp32:
        launch_fused_flash_attention_gqa_prefill_fp32(d_Q, d_Kd, d_Vd, d_ofmt, q_len, kv_len, nh,
                                                      nkv, hd, nullptr, causal, 0);
        break;
    case Fmt::Q4_0:
        launch_fused_flash_attention_gqa_prefill_q4_0(d_Q, d_qK, d_qV, d_ofmt, q_len, kv_len, nh,
                                                      nkv, hd, rb, nullptr, causal, nullptr,
                                                      nullptr, 0);
        break;
    case Fmt::F16:
        launch_fused_flash_attention_gqa_prefill_f16(d_Q, d_qK, d_qV, d_ofmt, q_len, kv_len, nh,
                                                     nkv, hd, rb, nullptr, causal, nullptr, nullptr,
                                                     0);
        break;
    case Fmt::Q8_0:
        launch_fused_flash_attention_gqa_prefill_q8_0(d_Q, d_qK, d_qV, d_ofmt, q_len, kv_len, nh,
                                                      nkv, hd, rb, nullptr, causal, nullptr,
                                                      nullptr, 0);
        break;
    case Fmt::Fp8E4M3:
        launch_fused_flash_attention_gqa_prefill_fp8_e4m3(d_Q, d_qK, d_qV, d_ofmt, q_len, kv_len,
                                                          nh, nkv, hd, rb, nullptr, causal, d_ks,
                                                          d_vs, 0);
        break;
    case Fmt::Fp8E5M2:
        launch_fused_flash_attention_gqa_prefill_fp8_e5m2(d_Q, d_qK, d_qV, d_ofmt, q_len, kv_len,
                                                          nh, nkv, hd, rb, nullptr, causal, d_ks,
                                                          d_vs, 0);
        break;
    }
    // Golden: FP32 vectorized prefill over the same dequantized KV.
    launch_flash_attention_gqa_vectorized(d_Q, d_Kd, d_Vd, d_ogold, q_len, kv_len, nh, nkv, hd,
                                          nullptr, causal, 0);
    cudaDeviceSynchronize();

    std::vector<float> o_fmt(q_elems), o_gold(q_elems);
    cudaMemcpy(o_fmt.data(), d_ofmt, q_elems * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(o_gold.data(), d_ogold, q_elems * 4, cudaMemcpyDeviceToHost);

    std::vector<float> o_ref;
    ref_gqa(Q, Kd, Vd, q_len, kv_len, nh, nkv, hd, causal, o_ref);

    const double cos_gold = cosine(o_fmt, o_gold);
    const double mabs = max_abs(o_fmt, o_gold);
    const double cos_ref = cosine(o_fmt, o_ref);
    const bool ok = (cos_gold >= 0.9999) && (mabs <= 1e-3) && (cos_ref >= 0.999);

    printf("  %-5s HD=%-3d q=%d kv=%d h=%d kvh=%d causal=%d  cos(gold)=%.6f maxabs=%.3e "
           "cos(cpu)=%.6f  %s\n",
           fmt_name(fmt), hd, q_len, kv_len, nh, nkv, (int)causal, cos_gold, mabs, cos_ref,
           ok ? "ok" : "FAIL");

    cudaFree(d_Q); cudaFree(d_Kd); cudaFree(d_Vd); cudaFree(d_ofmt); cudaFree(d_ogold);
    cudaFree(d_qK); cudaFree(d_qV);
    if (d_ks) cudaFree(d_ks);
    if (d_vs) cudaFree(d_vs);
    return ok;
}

// ---- Paged vs contiguous ----------------------------------------------------

bool run_paged(Fmt fmt, int hd, int q_len, int kv_len, int nh, int nkv, int page_size,
               bool causal, std::mt19937& rng) {
    const size_t q_elems = (size_t)q_len * nh * hd;
    const size_t kv_elems = (size_t)kv_len * nkv * hd;

    std::vector<float> Q(q_elems), K(kv_elems), V(kv_elems);
    std::uniform_real_distribution<float> ur(-1.0f, 1.0f);
    for (auto& v : Q) v = ur(rng);
    for (auto& v : K) v = ur(rng);
    for (auto& v : V) v = ur(rng);

    std::vector<uint8_t> qK, qV;
    std::vector<float> ks, vs, unused;
    quantize(fmt, K, kv_len, nkv, hd, qK, ks, unused);
    quantize(fmt, V, kv_len, nkv, hd, qV, vs, unused);

    const size_t rb = fmt_row_bytes(fmt, nkv, hd);
    const int n_pages = (kv_len + page_size - 1) / page_size;
    const size_t page_bytes = (size_t)page_size * rb;

    // Host-side page table: page p holds rows [p*page_size, (p+1)*page_size).
    std::vector<int32_t> page_ids(n_pages);
    for (int p = 0; p < n_pages; ++p) page_ids[p] = p;

    // Device: allocate each page, fill it, and collect the device pointers.
    std::vector<void*> h_kpages(n_pages), h_vpages(n_pages);
    for (int p = 0; p < n_pages; ++p) {
        const int rows = std::min(page_size, kv_len - p * page_size);
        cudaMalloc(&h_kpages[p], page_bytes);
        cudaMalloc(&h_vpages[p], page_bytes);
        cudaMemset(h_kpages[p], 0, page_bytes);
        cudaMemset(h_vpages[p], 0, page_bytes);
        cudaMemcpy(h_kpages[p], qK.data() + (size_t)p * page_size * rb, (size_t)rows * rb,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(h_vpages[p], qV.data() + (size_t)p * page_size * rb, (size_t)rows * rb,
                   cudaMemcpyHostToDevice);
    }

    void **d_kpages = nullptr, **d_vpages = nullptr;
    int32_t* d_ids = nullptr;
    cudaMalloc(&d_kpages, n_pages * sizeof(void*));
    cudaMalloc(&d_vpages, n_pages * sizeof(void*));
    cudaMalloc(&d_ids, n_pages * sizeof(int32_t));
    cudaMemcpy(d_kpages, h_kpages.data(), n_pages * sizeof(void*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vpages, h_vpages.data(), n_pages * sizeof(void*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ids, page_ids.data(), n_pages * sizeof(int32_t), cudaMemcpyHostToDevice);

    float *d_Q = nullptr, *d_opaged = nullptr, *d_ocont = nullptr;
    uint8_t *d_qK = nullptr, *d_qV = nullptr;
    float *d_ks = nullptr, *d_vs = nullptr;
    cudaMalloc(&d_Q, Q.size() * 4);
    cudaMalloc(&d_opaged, q_elems * 4);
    cudaMalloc(&d_ocont, q_elems * 4);
    cudaMalloc(&d_qK, qK.size());
    cudaMalloc(&d_qV, qV.size());
    cudaMemcpy(d_Q, Q.data(), Q.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_qK, qK.data(), qK.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qV, qV.data(), qV.size(), cudaMemcpyHostToDevice);

    const bool fp8 = (fmt == Fmt::Fp8E4M3 || fmt == Fmt::Fp8E5M2);
    if (fp8) {
        cudaMalloc(&d_ks, ks.size() * 4);
        cudaMalloc(&d_vs, vs.size() * 4);
        cudaMemcpy(d_ks, ks.data(), ks.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_vs, vs.data(), vs.size() * 4, cudaMemcpyHostToDevice);
    }

    // Contiguous FP32 K/V for the paged-FP32 reference launch.
    float *d_Kd = nullptr, *d_Vd = nullptr;
    if (fmt == Fmt::Fp32) {
        std::vector<float> Kd, Vd;
        dequantize(fmt, qK, kv_len, nkv, hd, ks, Kd);
        dequantize(fmt, qV, kv_len, nkv, hd, vs, Vd);
        cudaMalloc(&d_Kd, Kd.size() * 4);
        cudaMalloc(&d_Vd, Vd.size() * 4);
        cudaMemcpy(d_Kd, Kd.data(), Kd.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_Vd, Vd.data(), Vd.size() * 4, cudaMemcpyHostToDevice);
    }

    // Paged launch. (Paged FP8 KV carries no row scales in this engine.)
    switch (fmt) {
    case Fmt::Fp32:
        launch_paged_flash_attention_gqa_prefill_fp32(d_Q, d_kpages, d_vpages, d_ids, d_opaged,
                                                      q_len, kv_len, nh, nkv, hd, page_size, rb,
                                                      nullptr, causal, nullptr, nullptr, 0);
        launch_fused_flash_attention_gqa_prefill_fp32(d_Q, d_Kd, d_Vd, d_ocont, q_len, kv_len, nh,
                                                      nkv, hd, nullptr, causal, 0);
        break;
    case Fmt::Q4_0:
        launch_paged_flash_attention_gqa_prefill_q4_0(d_Q, d_kpages, d_vpages, d_ids, d_opaged,
                                                      q_len, kv_len, nh, nkv, hd, page_size, rb,
                                                      nullptr, causal, nullptr, nullptr, 0);
        launch_fused_flash_attention_gqa_prefill_q4_0(d_Q, d_qK, d_qV, d_ocont, q_len, kv_len, nh,
                                                      nkv, hd, rb, nullptr, causal, nullptr,
                                                      nullptr, 0);
        break;
    case Fmt::F16:
        launch_paged_flash_attention_gqa_prefill_f16(d_Q, d_kpages, d_vpages, d_ids, d_opaged,
                                                     q_len, kv_len, nh, nkv, hd, page_size, rb,
                                                     nullptr, causal, nullptr, nullptr, 0);
        launch_fused_flash_attention_gqa_prefill_f16(d_Q, d_qK, d_qV, d_ocont, q_len, kv_len, nh,
                                                     nkv, hd, rb, nullptr, causal, nullptr, nullptr,
                                                     0);
        break;
    case Fmt::Q8_0:
        launch_paged_flash_attention_gqa_prefill_q8_0(d_Q, d_kpages, d_vpages, d_ids, d_opaged,
                                                      q_len, kv_len, nh, nkv, hd, page_size, rb,
                                                      nullptr, causal, nullptr, nullptr, 0);
        launch_fused_flash_attention_gqa_prefill_q8_0(d_Q, d_qK, d_qV, d_ocont, q_len, kv_len, nh,
                                                      nkv, hd, rb, nullptr, causal, nullptr,
                                                      nullptr, 0);
        break;
    case Fmt::Fp8E4M3:
        launch_paged_flash_attention_gqa_prefill_fp8_e4m3(d_Q, d_kpages, d_vpages, d_ids, d_opaged,
                                                          q_len, kv_len, nh, nkv, hd, page_size, rb,
                                                          nullptr, causal, nullptr, nullptr, 0);
        launch_fused_flash_attention_gqa_prefill_fp8_e4m3(d_Q, d_qK, d_qV, d_ocont, q_len, kv_len,
                                                          nh, nkv, hd, rb, nullptr, causal, d_ks,
                                                          d_vs, 0);
        break;
    case Fmt::Fp8E5M2:
        launch_paged_flash_attention_gqa_prefill_fp8_e5m2(d_Q, d_kpages, d_vpages, d_ids, d_opaged,
                                                          q_len, kv_len, nh, nkv, hd, page_size, rb,
                                                          nullptr, causal, nullptr, nullptr, 0);
        launch_fused_flash_attention_gqa_prefill_fp8_e5m2(d_Q, d_qK, d_qV, d_ocont, q_len, kv_len,
                                                          nh, nkv, hd, rb, nullptr, causal, d_ks,
                                                          d_vs, 0);
        break;
    default:
        break;
    }
    cudaDeviceSynchronize();

    std::vector<float> o_paged(q_elems), o_cont(q_elems);
    cudaMemcpy(o_paged.data(), d_opaged, q_elems * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(o_cont.data(), d_ocont, q_elems * 4, cudaMemcpyDeviceToHost);

    // Paged FP8 KV carries no row scales in this engine (the paged FP8 *decode*
    // kernel dequantizes unscaled too), so its reference is the CPU GQA over
    // the raw FP8 values (scale = 1).
    double cos = 0, mabs = 0;
    bool ok = false;
    if (fp8) {
        std::vector<float> ones(ks.size(), 1.0f);
        std::vector<float> Kd, Vd, o_ref;
        dequantize(fmt, qK, kv_len, nkv, hd, ones, Kd);
        dequantize(fmt, qV, kv_len, nkv, hd, ones, Vd);
        ref_gqa(Q, Kd, Vd, q_len, kv_len, nh, nkv, hd, causal, o_ref);
        cos = cosine(o_paged, o_ref);
        mabs = max_abs(o_paged, o_ref);
        // Raw (unscaled) FP8 values reach 240 (E4M3) / 57344 (E5M2), so the
        // absolute error must be judged against the output magnitude.
        double ref_scale = 0;
        for (float v : o_ref) ref_scale = std::max(ref_scale, std::fabs((double)v));
        ok = (cos >= 0.9999) && (mabs <= 1e-3 * (ref_scale + 1e-6));
        printf("  %-5s HD=%-3d paged q=%d kv=%d page=%d causal=%d  cos(cpu,scale=1)=%.6f "
               "maxabs=%.3e ref_scale=%.3e  %s\n",
               fmt_name(fmt), hd, q_len, kv_len, page_size, (int)causal, cos, mabs, ref_scale,
               ok ? "ok" : "FAIL");
    } else {
        cos = cosine(o_paged, o_cont);
        mabs = max_abs(o_paged, o_cont);
        ok = (cos >= 0.9999) && (mabs <= 1e-6);
        printf("  %-5s HD=%-3d paged q=%d kv=%d page=%d causal=%d  cos(contig)=%.6f "
               "maxabs=%.3e  %s\n",
               fmt_name(fmt), hd, q_len, kv_len, page_size, (int)causal, cos, mabs,
               ok ? "ok" : "FAIL");
    }

    for (int p = 0; p < n_pages; ++p) {
        cudaFree(h_kpages[p]);
        cudaFree(h_vpages[p]);
    }
    cudaFree(d_kpages); cudaFree(d_vpages); cudaFree(d_ids);
    cudaFree(d_Q); cudaFree(d_opaged); cudaFree(d_ocont); cudaFree(d_qK); cudaFree(d_qV);
    if (d_ks) cudaFree(d_ks);
    if (d_vs) cudaFree(d_vs);
    if (d_Kd) cudaFree(d_Kd);
    if (d_Vd) cudaFree(d_Vd);
    return ok;
}

// ---- Tiled DECODE (q_len == 1) ----------------------------------------------
//
// Decoding reuses the same tiled kernel with q_len == 1 (Q/O layouts coincide,
// the 2-D mask degenerates to a 1-D row, causal is a no-op). It is checked
// against the previously-shipped decode kernel for the same format — which must
// agree to ~1e-7 — and against the CPU reference.

bool run_decode(Fmt fmt, int hd, int kv_len, int nh, int nkv, std::mt19937& rng) {
    const size_t q_elems = (size_t)nh * hd;  // q_len == 1
    const size_t kv_elems = (size_t)kv_len * nkv * hd;

    std::vector<float> Q(q_elems), K(kv_elems), V(kv_elems);
    std::uniform_real_distribution<float> ur(-1.0f, 1.0f);
    for (auto& v : Q) v = ur(rng);
    for (auto& v : K) v = ur(rng);
    for (auto& v : V) v = ur(rng);

    std::vector<uint8_t> qK, qV;
    std::vector<float> ks, vs, unused;
    quantize(fmt, K, kv_len, nkv, hd, qK, ks, unused);
    quantize(fmt, V, kv_len, nkv, hd, qV, vs, unused);

    std::vector<float> Kd, Vd;
    dequantize(fmt, qK, kv_len, nkv, hd, ks, Kd);
    dequantize(fmt, qV, kv_len, nkv, hd, vs, Vd);

    float *d_Q = nullptr, *d_Kd = nullptr, *d_Vd = nullptr;
    float *d_onew = nullptr, *d_oold = nullptr;
    uint8_t *d_qK = nullptr, *d_qV = nullptr;
    float *d_ks = nullptr, *d_vs = nullptr;
    cudaMalloc(&d_Q, Q.size() * 4);
    cudaMalloc(&d_Kd, Kd.size() * 4);
    cudaMalloc(&d_Vd, Vd.size() * 4);
    cudaMalloc(&d_onew, q_elems * 4);
    cudaMalloc(&d_oold, q_elems * 4);
    cudaMalloc(&d_qK, qK.size());
    cudaMalloc(&d_qV, qV.size());
    cudaMemcpy(d_Q, Q.data(), Q.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Kd, Kd.data(), Kd.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Vd, Vd.data(), Vd.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_qK, qK.data(), qK.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qV, qV.data(), qV.size(), cudaMemcpyHostToDevice);

    const bool fp8 = (fmt == Fmt::Fp8E4M3 || fmt == Fmt::Fp8E5M2);
    if (fp8) {
        cudaMalloc(&d_ks, ks.size() * 4);
        cudaMalloc(&d_vs, vs.size() * 4);
        cudaMemcpy(d_ks, ks.data(), ks.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_vs, vs.data(), vs.size() * 4, cudaMemcpyHostToDevice);
    }

    const size_t rb = fmt_row_bytes(fmt, nkv, hd);
    switch (fmt) {
    case Fmt::Fp32:
        launch_fused_flash_attention_gqa_prefill_fp32(d_Q, d_Kd, d_Vd, d_onew, 1, kv_len, nh, nkv,
                                                      hd, nullptr, false, 0);
        launch_flash_attention_gqa_decode(d_Q, d_Kd, d_Vd, d_oold, kv_len, nh, nkv, hd, nullptr, 0);
        break;
    case Fmt::Q4_0:
        launch_fused_flash_attention_gqa_prefill_q4_0(d_Q, d_qK, d_qV, d_onew, 1, kv_len, nh, nkv,
                                                      hd, rb, nullptr, false, nullptr, nullptr, 0);
        launch_fused_flash_attention_gqa_decode_q4_0(d_Q, d_qK, d_qV, d_oold, kv_len, nh, nkv, hd,
                                                     rb, nullptr, 0);
        break;
    case Fmt::F16:
        launch_fused_flash_attention_gqa_prefill_f16(d_Q, d_qK, d_qV, d_onew, 1, kv_len, nh, nkv,
                                                     hd, rb, nullptr, false, nullptr, nullptr, 0);
        launch_fused_flash_attention_gqa_decode_f16(d_Q, d_qK, d_qV, d_oold, kv_len, nh, nkv, hd,
                                                    rb, nullptr, 0);
        break;
    case Fmt::Q8_0:
        launch_fused_flash_attention_gqa_prefill_q8_0(d_Q, d_qK, d_qV, d_onew, 1, kv_len, nh, nkv,
                                                      hd, rb, nullptr, false, nullptr, nullptr, 0);
        launch_fused_flash_attention_gqa_decode_q8_0(d_Q, d_qK, d_qV, d_oold, kv_len, nh, nkv, hd,
                                                     rb, nullptr, 0);
        break;
    case Fmt::Fp8E4M3:
        launch_fused_flash_attention_gqa_prefill_fp8_e4m3(d_Q, d_qK, d_qV, d_onew, 1, kv_len, nh,
                                                          nkv, hd, rb, nullptr, false, d_ks, d_vs, 0);
        launch_fused_flash_attention_gqa_decode_fp8_e4m3(d_Q, d_qK, d_qV, d_oold, kv_len, nh, nkv,
                                                         hd, rb, nullptr, d_ks, d_vs, 0);
        break;
    case Fmt::Fp8E5M2:
        launch_fused_flash_attention_gqa_prefill_fp8_e5m2(d_Q, d_qK, d_qV, d_onew, 1, kv_len, nh,
                                                          nkv, hd, rb, nullptr, false, d_ks, d_vs, 0);
        launch_fused_flash_attention_gqa_decode_fp8_e5m2(d_Q, d_qK, d_qV, d_oold, kv_len, nh, nkv,
                                                         hd, rb, nullptr, d_ks, d_vs, 0);
        break;
    }
    cudaDeviceSynchronize();

    std::vector<float> o_new(q_elems), o_old(q_elems);
    cudaMemcpy(o_new.data(), d_onew, q_elems * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(o_old.data(), d_oold, q_elems * 4, cudaMemcpyDeviceToHost);

    std::vector<float> o_ref;
    ref_gqa(Q, Kd, Vd, 1, kv_len, nh, nkv, hd, false, o_ref);

    const double cos_old = cosine(o_new, o_old);
    const double mabs = max_abs(o_new, o_old);
    const double cos_ref = cosine(o_new, o_ref);
    const bool ok = (cos_old >= 0.9999) && (mabs <= 1e-4) && (cos_ref >= 0.999);

    printf("  %-5s HD=%-3d kv=%d h=%d kvh=%d  cos(vs legacy decode)=%.6f maxabs=%.3e "
           "cos(cpu)=%.6f  %s\n",
           fmt_name(fmt), hd, kv_len, nh, nkv, cos_old, mabs, cos_ref, ok ? "ok" : "FAIL");

    cudaFree(d_Q); cudaFree(d_Kd); cudaFree(d_Vd); cudaFree(d_onew); cudaFree(d_oold);
    cudaFree(d_qK); cudaFree(d_qV);
    if (d_ks) cudaFree(d_ks);
    if (d_vs) cudaFree(d_vs);
    return ok;
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        printf("SKIP: no CUDA device available\n");
        return 0;
    }
    std::mt19937 rng(20260906);
    bool all_ok = true;

    printf("[contiguous]\n");
    for (Fmt f : {Fmt::Fp32, Fmt::Q4_0, Fmt::F16, Fmt::Q8_0, Fmt::Fp8E4M3, Fmt::Fp8E5M2}) {
        for (int hd : {32, 64, 128}) {
            if (hd < fmt_min_hd(f)) continue;
            for (bool causal : {false, true}) {
                all_ok &= run_contiguous(f, hd, 32, 128, 8, 2, causal, rng);
            }
        }
    }

    printf("[paged vs contiguous]\n");
    for (Fmt f : {Fmt::Fp32, Fmt::Q4_0, Fmt::F16, Fmt::Q8_0, Fmt::Fp8E4M3, Fmt::Fp8E5M2}) {
        for (bool causal : {false, true}) {
            all_ok &= run_paged(f, 64, 32, 128, 8, 2, 64, causal, rng);
        }
        // Multi-page with a page size that does not divide kv_len evenly.
        all_ok &= run_paged(f, 64, 16, 200, 8, 2, 64, true, rng);
    }

    printf("[tiled decode vs legacy decode kernel]\n");
    for (Fmt f : {Fmt::Fp32, Fmt::Q4_0, Fmt::F16, Fmt::Q8_0, Fmt::Fp8E4M3, Fmt::Fp8E5M2}) {
        for (int hd : {64, 128}) {
            if (hd < fmt_min_hd(f)) continue;
            for (int kv_len : {128, 2048}) {
                all_ok &= run_decode(f, hd, kv_len, 8, 2, rng);
            }
        }
    }

    printf("%s\n", all_ok ? "tiled-attn-formats: PASS" : "tiled-attn-formats: FAIL");
    return all_ok ? 0 : 1;
}
