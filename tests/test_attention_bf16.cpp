// =========================================================================
// test_attention_bf16.cpp -- Phase 6-B-6: native BF16-Q attention (decode-first)
//
// Mixed storage contract (NOT "swap __half for __nv_bfloat16"): Q arrives
// BF16, K/V arrive FP32 (authoritative KV store -- bf16_policy.kv stays FP32),
// all math stays FP32 (QK^T / online softmax / PV, same scale convention as
// the FP32/FP16 kernels), output is BF16.
//
// Acceptance (decode slice):
//   1. BF16 Q + FP32 K/V -> BF16 output (dtype hard asserts, not just numerics)
//   2. vs CPU FP32 reference on the same decoded-BF16 Q (parity, BF16 ULP
//      analysis) + vs FP32 kernel self-consistency
//   3. FP32 math by construction (kernel diff review; parity catches
//      scale/ordering regressions)
//   4. ordinary stream == capture stream (bitwise)
//   5. ComputeGraph FLASH_ATTN_GQA node capture/replay (bitwise)
//   6. Selector routes q-BF16 seq1 -> CUDA_BF16_DECODE; BF16 prefill / BF16 KV /
//      non-GQA BF16 / hd256 all fail loudly
//
// Requires CUDA; skips gracefully without a GPU.
// =========================================================================

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

#include "forge/attention_selector.h"
#include "forge/compute_graph.h"
#include "forge/cuda_graph_runner.h"
#include "forge/cuda_kernels.h"
#include "forge/dtype_contract.h"
#include "forge/op_dispatch.h"
#include "forge/op_enum.h"
#include "forge/tensor.h"
#include "forge/types.h"

namespace {

using forge::DataType;
using forge::DeviceType;
using forge::Tensor;
using forge::TensorPtr;

uint16_t f32_to_bf16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t rounding = 0x7FFFu + ((x >> 16) & 1u);  // round to nearest-even
    return static_cast<uint16_t>((x + rounding) >> 16);
}
float bf16_to_f32(uint16_t h) {
    uint32_t x = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

TensorPtr make_dev_bf16(const std::vector<uint16_t>& h, int64_t r, int64_t c) {
    auto host =
        std::make_shared<Tensor>(DataType::BF16, std::vector<int64_t>{r, c}, DeviceType::CPU);
    std::memcpy(host->data(), h.data(), h.size() * 2);
    auto dev =
        std::make_shared<Tensor>(DataType::BF16, std::vector<int64_t>{r, c}, DeviceType::CUDA);
    dev->copy_from(*host);
    return dev;
}
TensorPtr make_dev_fp32(const std::vector<float>& h, int64_t r, int64_t c) {
    auto host =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{r, c}, DeviceType::CPU);
    std::memcpy(host->data(), h.data(), h.size() * 4);
    auto dev =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{r, c}, DeviceType::CUDA);
    dev->copy_from(*host);
    return dev;
}
std::vector<uint16_t> to_host_u16(const TensorPtr& t) {
    auto host = std::make_shared<Tensor>(t->dtype(), t->shape(), DeviceType::CPU);
    host->copy_from(*t);
    std::vector<uint16_t> v(static_cast<size_t>(t->numel()));
    std::memcpy(v.data(), host->data(), v.size() * 2);
    return v;
}
std::vector<float> to_host_f32(const TensorPtr& t) {
    auto host = std::make_shared<Tensor>(t->dtype(), t->shape(), DeviceType::CPU);
    host->copy_from(*t);
    std::vector<float> v(static_cast<size_t>(t->numel()));
    if (t->dtype() == DataType::BF16) {
        std::vector<uint16_t> c(v.size());
        std::memcpy(c.data(), host->data(), c.size() * 2);
        for (size_t i = 0; i < v.size(); ++i)
            v[i] = bf16_to_f32(c[i]);
    } else {
        std::memcpy(v.data(), host->data(), v.size() * 4);
    }
    return v;
}

int failures = 0;
#define CHECK(cond)                                                     \
    do {                                                                \
        if (!(cond)) {                                                  \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                 \
        }                                                               \
    } while (0)

void rand_bf16(int64_t n, float lo, float hi, std::mt19937& g, std::vector<uint16_t>& codes) {
    std::uniform_real_distribution<float> d(lo, hi);
    codes.resize(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
        codes[i] = f32_to_bf16(d(g));
}
void rand_f32(int64_t n, float lo, float hi, std::mt19937& g, std::vector<float>& vals) {
    std::uniform_real_distribution<float> d(lo, hi);
    vals.resize(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
        vals[i] = d(g);
}

// CPU FP32 GQA reference (2-pass softmax). score = dot(q,k) * rsqrt(head_dim)
// -- the same scale convention as the CUDA kernels (cuda_flash_attn.cu).
void ref_gqa_decode_fp32(const float* Q, const float* K, const float* V, float* O, int kv_len,
                         int nh, int nkv, int hd) {
    float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    int kg = nh / nkv;
    for (int h = 0; h < nh; ++h) {
        int kvh = h / kg;
        std::vector<float> scores(static_cast<size_t>(kv_len));
        float mx = -1e30f;
        for (int j = 0; j < kv_len; ++j) {
            float dot = 0.0f;
            for (int d = 0; d < hd; ++d)
                dot += Q[h * hd + d] * K[(size_t)j * nkv * hd + kvh * hd + d];
            scores[j] = dot * scale;
            if (scores[j] > mx)
                mx = scores[j];
        }
        float sum = 0.0f;
        for (int j = 0; j < kv_len; ++j) {
            scores[j] = std::exp(scores[j] - mx);
            sum += scores[j];
        }
        for (int d = 0; d < hd; ++d) {
            float acc = 0.0f;
            for (int j = 0; j < kv_len; ++j)
                acc += scores[j] * V[(size_t)j * nkv * hd + kvh * hd + d];
            O[h * hd + d] = acc / sum;
        }
    }
}

// 6-B-6 decode correctness: BF16-Q kernel vs CPU FP32 ref on the same
// decoded-BF16 Q, dtype hard asserts, ordinary == capture stream (bitwise).
void check_decode(int kv_len, int nh, int nkv, int hd, std::mt19937& g) {
    std::vector<uint16_t> qc;
    std::vector<float> kf, vf;
    rand_bf16((int64_t)nh * hd, -1.0f, 1.0f, g, qc);
    rand_f32((int64_t)kv_len * nkv * hd, -1.0f, 1.0f, g, kf);
    rand_f32((int64_t)kv_len * nkv * hd, -1.0f, 1.0f, g, vf);
    auto q = make_dev_bf16(qc, 1, nh * hd);
    auto k = make_dev_fp32(kf, kv_len, nkv * hd);
    auto v = make_dev_fp32(vf, kv_len, nkv * hd);
    auto o = std::make_shared<Tensor>(DataType::BF16, std::vector<int64_t>{1, nh * hd},
                                      DeviceType::CUDA);

    // A: legacy stream.
    forge::cuda::launch_flash_attention_gqa_decode_bf16q(
        reinterpret_cast<const __nv_bfloat16*>(q->data()), static_cast<const float*>(k->data()),
        static_cast<const float*>(v->data()), reinterpret_cast<__nv_bfloat16*>(o->data()), kv_len,
        nh, nkv, hd, nullptr, 0);
    cudaDeviceSynchronize();
    auto codesA = to_host_u16(o);

    // B: explicit capture stream -- must match A bitwise.
    cudaStream_t cs = nullptr;
    CHECK(cudaStreamCreate(&cs) == cudaSuccess);
    auto oB = std::make_shared<Tensor>(DataType::BF16, std::vector<int64_t>{1, nh * hd},
                                       DeviceType::CUDA);
    forge::cuda::launch_flash_attention_gqa_decode_bf16q(
        reinterpret_cast<const __nv_bfloat16*>(q->data()), static_cast<const float*>(k->data()),
        static_cast<const float*>(v->data()), reinterpret_cast<__nv_bfloat16*>(oB->data()), kv_len,
        nh, nkv, hd, nullptr, cs);
    cudaStreamSynchronize(cs);
    auto codesB = to_host_u16(oB);
    CHECK(codesA == codesB);
    cudaStreamDestroy(cs);

    // Reference: decoded-BF16 Q + same FP32 K/V through the CPU FP32 GQA.
    // K/V are never quantized here (authoritative store); the only rounding
    // under test is the BF16 Q input store + the BF16 output store.
    std::vector<float> qf(qc.size());
    for (size_t i = 0; i < qc.size(); ++i)
        qf[i] = bf16_to_f32(qc[i]);
    std::vector<float> ref(static_cast<size_t>(nh * hd));
    ref_gqa_decode_fp32(qf.data(), kf.data(), vf.data(), ref.data(), kv_len, nh, nkv, hd);

    auto got = to_host_f32(o);
    float max_abs = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i)
        max_abs = std::max(max_abs, std::fabs(got[i] - ref[i]));
    std::printf("  decode kv=%d nh=%d nkv=%d hd=%d : max_abs=%.3e (A==B: %d)\n", kv_len, nh, nkv,
                hd, max_abs, codesA == codesB ? 1 : 0);
    // Calibration (RTX 4050, 2026-09-07): outputs are softmax averages of
    // V in [-1,1] so |out| <= 1; observed 1.2-2.3e-4 (BF16 Q-input + output
    // store rounding; kernel math is FP32). Set at ~4x observed; structural
    // defects (wrong scale/axis/ordering) blow past by 100x+.
    CHECK(max_abs < 1e-3f);
}

// FP32 self-consistency: the same decoded values through the FP32 decode
// kernel must match the BF16-Q kernel to BF16 rounding (no structural
// deviation between the two compilation units).
void check_fp32_consistency(int kv_len, int nh, int nkv, int hd, std::mt19937& g) {
    std::vector<uint16_t> qc;
    std::vector<float> kf, vf;
    rand_bf16((int64_t)nh * hd, -1.0f, 1.0f, g, qc);
    rand_f32((int64_t)kv_len * nkv * hd, -1.0f, 1.0f, g, kf);
    rand_f32((int64_t)kv_len * nkv * hd, -1.0f, 1.0f, g, vf);
    std::vector<float> qf(qc.size());
    for (size_t i = 0; i < qc.size(); ++i)
        qf[i] = bf16_to_f32(qc[i]);
    auto q32 = make_dev_fp32(qf, 1, nh * hd);
    auto k32 = make_dev_fp32(kf, kv_len, nkv * hd);
    auto v32 = make_dev_fp32(vf, kv_len, nkv * hd);
    auto o32 = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, nh * hd},
                                        DeviceType::CUDA);
    forge::cuda::launch_flash_attention_gqa_decode(
        static_cast<const float*>(q32->data()), static_cast<const float*>(k32->data()),
        static_cast<const float*>(v32->data()), static_cast<float*>(o32->data()), kv_len, nh, nkv,
        hd, nullptr, 0);
    cudaDeviceSynchronize();

    auto qb = make_dev_bf16(qc, 1, nh * hd);
    auto ob = std::make_shared<Tensor>(DataType::BF16, std::vector<int64_t>{1, nh * hd},
                                       DeviceType::CUDA);
    forge::cuda::launch_flash_attention_gqa_decode_bf16q(
        reinterpret_cast<const __nv_bfloat16*>(qb->data()), static_cast<const float*>(k32->data()),
        static_cast<const float*>(v32->data()), reinterpret_cast<__nv_bfloat16*>(ob->data()),
        kv_len, nh, nkv, hd, nullptr, 0);
    cudaDeviceSynchronize();

    auto got32 = to_host_f32(o32);
    auto gotbf = to_host_f32(ob);
    float max_abs = 0.0f;
    for (size_t i = 0; i < got32.size(); ++i)
        max_abs = std::max(max_abs, std::fabs(gotbf[i] - got32[i]));
    std::printf("  fp32-consistency kv=%d nh=%d nkv=%d hd=%d : max_abs=%.3e\n", kv_len, nh, nkv, hd,
                max_abs);
    // Same decoded values through both kernels: only the BF16 output store
    // differs. Observed 1.2-2.4e-4; same ~4x bound as above.
    CHECK(max_abs < 1e-3f);
}

// 6-B-6.7 all-BF16 decode correctness: BF16 Q/K/V storage, FP32 math, BF16
// output, vs the CPU FP32 ref on the same decoded-BF16 QKV. K/V rounding now
// contributes (unlike the BF16-Q slice), so the bound is set from measurement
// (see calibration below), with stream A/B bitwise as the exactness anchor.
void check_decode_bf16(int kv_len, int nh, int nkv, int hd, std::mt19937& g) {
    std::vector<uint16_t> qc, kc, vc;
    rand_bf16((int64_t)nh * hd, -1.0f, 1.0f, g, qc);
    rand_bf16((int64_t)kv_len * nkv * hd, -1.0f, 1.0f, g, kc);
    rand_bf16((int64_t)kv_len * nkv * hd, -1.0f, 1.0f, g, vc);
    auto q = make_dev_bf16(qc, 1, nh * hd);
    auto k = make_dev_bf16(kc, kv_len, nkv * hd);
    auto v = make_dev_bf16(vc, kv_len, nkv * hd);
    auto o = std::make_shared<Tensor>(DataType::BF16, std::vector<int64_t>{1, nh * hd},
                                      DeviceType::CUDA);

    // A: legacy stream.
    forge::cuda::launch_flash_attention_gqa_decode_bf16(
        reinterpret_cast<const __nv_bfloat16*>(q->data()),
        reinterpret_cast<const __nv_bfloat16*>(k->data()),
        reinterpret_cast<const __nv_bfloat16*>(v->data()),
        reinterpret_cast<__nv_bfloat16*>(o->data()), kv_len, nh, nkv, hd, nullptr, 0);
    cudaDeviceSynchronize();
    auto codesA = to_host_u16(o);

    // B: explicit capture stream -- must match A bitwise.
    cudaStream_t cs = nullptr;
    CHECK(cudaStreamCreate(&cs) == cudaSuccess);
    auto oB = std::make_shared<Tensor>(DataType::BF16, std::vector<int64_t>{1, nh * hd},
                                       DeviceType::CUDA);
    forge::cuda::launch_flash_attention_gqa_decode_bf16(
        reinterpret_cast<const __nv_bfloat16*>(q->data()),
        reinterpret_cast<const __nv_bfloat16*>(k->data()),
        reinterpret_cast<const __nv_bfloat16*>(v->data()),
        reinterpret_cast<__nv_bfloat16*>(oB->data()), kv_len, nh, nkv, hd, nullptr, cs);
    cudaStreamSynchronize(cs);
    auto codesB = to_host_u16(oB);
    CHECK(codesA == codesB);
    cudaStreamDestroy(cs);

    // Reference: all inputs decoded from the same BF16 codes.
    std::vector<float> qf(qc.size()), kf(kc.size()), vf(vc.size());
    for (size_t i = 0; i < qc.size(); ++i)
        qf[i] = bf16_to_f32(qc[i]);
    for (size_t i = 0; i < kc.size(); ++i)
        kf[i] = bf16_to_f32(kc[i]);
    for (size_t i = 0; i < vc.size(); ++i)
        vf[i] = bf16_to_f32(vc[i]);
    std::vector<float> ref(static_cast<size_t>(nh * hd));
    ref_gqa_decode_fp32(qf.data(), kf.data(), vf.data(), ref.data(), kv_len, nh, nkv, hd);

    auto got = to_host_f32(o);
    float max_abs = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i)
        max_abs = std::max(max_abs, std::fabs(got[i] - ref[i]));
    std::printf("  decode-bf16kv kv=%d nh=%d nkv=%d hd=%d : max_abs=%.3e (A==B: %d)\n", kv_len, nh,
                nkv, hd, max_abs, codesA == codesB ? 1 : 0);
    // Calibration (RTX 4050, 2026-09-07): |out| <= 1; observed 1.1-2.2e-4 --
    // the same band as the BF16-Q slice (K/V rounding adds no measurable
    // bias at kv=512; softmax averaging absorbs it). Same ~4x bound.
    CHECK(max_abs < 1e-3f);
}

// CPU FP32 GQA prefill reference (causal). q_pos = kv_len - q_len + qi --
// the same convention as the vectorized CUDA kernels.
void ref_gqa_prefill_fp32(const float* Q, const float* K, const float* V, float* O, int q_len,
                          int kv_len, int nh, int nkv, int hd) {
    float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    int kg = nh / nkv;
    for (int qi = 0; qi < q_len; ++qi) {
        int q_pos = kv_len - q_len + qi;
        for (int h = 0; h < nh; ++h) {
            int kvh = h / kg;
            const float* qr = Q + ((size_t)qi * nh + h) * hd;
            std::vector<float> scores(static_cast<size_t>(kv_len));
            float mx = -1e30f;
            for (int j = 0; j < kv_len; ++j) {
                if (j > q_pos) {
                    scores[j] = -1e30f;
                    continue;
                }
                float dot = 0.0f;
                for (int d = 0; d < hd; ++d)
                    dot += qr[d] * K[(size_t)j * nkv * hd + kvh * hd + d];
                scores[j] = dot * scale;
                if (scores[j] > mx)
                    mx = scores[j];
            }
            float sum = 0.0f;
            for (int j = 0; j < kv_len; ++j) {
                scores[j] = (scores[j] < -1e29f) ? 0.0f : std::exp(scores[j] - mx);
                sum += scores[j];
            }
            for (int d = 0; d < hd; ++d) {
                float acc = 0.0f;
                for (int j = 0; j < kv_len; ++j)
                    acc += scores[j] * V[(size_t)j * nkv * hd + kvh * hd + d];
                O[((size_t)qi * nh + h) * hd + d] = acc / sum;
            }
        }
    }
}

// 6-B-7.0 all-BF16 prefill correctness: direct launch vs CPU causal ref,
// dtype hard assert, stream A/B bitwise.
void check_prefill_bf16(int q_len, int kv_len, int nh, int nkv, int hd, std::mt19937& g) {
    std::vector<uint16_t> qc, kc, vc;
    rand_bf16((int64_t)q_len * nh * hd, -1.0f, 1.0f, g, qc);
    rand_bf16((int64_t)kv_len * nkv * hd, -1.0f, 1.0f, g, kc);
    rand_bf16((int64_t)kv_len * nkv * hd, -1.0f, 1.0f, g, vc);
    auto q = make_dev_bf16(qc, q_len, nh * hd);
    auto k = make_dev_bf16(kc, kv_len, nkv * hd);
    auto v = make_dev_bf16(vc, kv_len, nkv * hd);
    auto o = std::make_shared<Tensor>(DataType::BF16, std::vector<int64_t>{q_len, nh * hd},
                                      DeviceType::CUDA);

    forge::cuda::launch_flash_attention_gqa_prefill_bf16(
        reinterpret_cast<const __nv_bfloat16*>(q->data()),
        reinterpret_cast<const __nv_bfloat16*>(k->data()),
        reinterpret_cast<const __nv_bfloat16*>(v->data()),
        reinterpret_cast<__nv_bfloat16*>(o->data()), q_len, kv_len, nh, nkv, hd, nullptr, true, 0);
    cudaDeviceSynchronize();
    auto codesA = to_host_u16(o);

    cudaStream_t cs = nullptr;
    CHECK(cudaStreamCreate(&cs) == cudaSuccess);
    auto oB = std::make_shared<Tensor>(DataType::BF16, std::vector<int64_t>{q_len, nh * hd},
                                       DeviceType::CUDA);
    forge::cuda::launch_flash_attention_gqa_prefill_bf16(
        reinterpret_cast<const __nv_bfloat16*>(q->data()),
        reinterpret_cast<const __nv_bfloat16*>(k->data()),
        reinterpret_cast<const __nv_bfloat16*>(v->data()),
        reinterpret_cast<__nv_bfloat16*>(oB->data()), q_len, kv_len, nh, nkv, hd, nullptr, true,
        cs);
    cudaStreamSynchronize(cs);
    CHECK(to_host_u16(oB) == codesA);
    cudaStreamDestroy(cs);

    std::vector<float> qf(qc.size()), kf(kc.size()), vf(vc.size());
    for (size_t i = 0; i < qc.size(); ++i)
        qf[i] = bf16_to_f32(qc[i]);
    for (size_t i = 0; i < kc.size(); ++i)
        kf[i] = bf16_to_f32(kc[i]);
    for (size_t i = 0; i < vc.size(); ++i)
        vf[i] = bf16_to_f32(vc[i]);
    std::vector<float> ref(static_cast<size_t>(q_len * nh * hd));
    ref_gqa_prefill_fp32(qf.data(), kf.data(), vf.data(), ref.data(), q_len, kv_len, nh, nkv, hd);

    auto got = to_host_f32(o);
    float max_abs = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i)
        max_abs = std::max(max_abs, std::fabs(got[i] - ref[i]));
    std::printf("  prefill-bf16 q=%d kv=%d nh=%d nkv=%d hd=%d : max_abs=%.3e (A==B: %d)\n", q_len,
                kv_len, nh, nkv, hd, max_abs, to_host_u16(oB) == codesA ? 1 : 0);
    // Calibration (RTX 4050, 2026-09-07): 9.7e-4 (hd16/64), 1.9e-3 (hd128) --
    // outputs are softmax averages (|out| <= 1), so 1.9e-3 is 0.5 BF16 ULP:
    // pure output-store rounding. Bound at ~1 ULP (4e-3); structural defects
    // give O(0.1)+.
    CHECK(max_abs < 4e-3f);
}

// Selector unit test: q BF16 seq1 -> BF16_DECODE, q BF16 seq>1 -> BF16_PREFILL
// (6-B-7.0 unlocked); names resolve.
void check_selector() {
    using forge::AttentionPath;
    forge::AttentionProblem p;
    p.device = DeviceType::CUDA;
    p.num_heads = 8;
    p.num_kv_heads = 2;
    p.head_dim = 64;

    p.seq_len = 1;
    p.q_dtype = DataType::BF16;
    CHECK(forge::choose_attention_path(p) == AttentionPath::CUDA_BF16_DECODE);
    p.q_dtype = DataType::FP32;
    CHECK(forge::choose_attention_path(p) == AttentionPath::CUDA_FP32_DECODE);

    p.seq_len = 16;
    p.q_dtype = DataType::BF16;
    CHECK(forge::choose_attention_path(p) == AttentionPath::CUDA_BF16_PREFILL);

    CHECK(std::string(forge::attention_path_name(AttentionPath::CUDA_BF16_DECODE)) ==
          "CUDA_BF16_DECODE");
    CHECK(std::string(forge::attention_path_name(AttentionPath::CUDA_BF16_PREFILL)) ==
          "CUDA_BF16_PREFILL");
    std::printf("  selector: BF16 decode + prefill routes OK\n");
}

// Loud-rejection surface (decode scope retired; everything else still rejects):
// BF16 K/V with BF16 Q, BF16-Q prefill (seq>1), non-GQA BF16, hd256 -- all via
// the dispatch layer, all must throw with a BF16 marker.
void check_loud(int kv_len, int nh, int nkv, int hd, std::mt19937& g) {
    auto run_dispatch = [&](const TensorPtr& q, const TensorPtr& k, const TensorPtr& v, int qlen,
                            int hdm, const char* label) {
        int32_t params[forge::OP_PARAMS_MAX_SIZE / sizeof(int32_t)] = {0};
        params[0] = nh;
        params[1] = nkv;
        params[2] = hdm;
        params[3] = 1;  // causal
        params[4] = 1;  // CUDA
        bool threw = false;
        std::string what;
        try {
            forge::OpDispatch::instance().execute(forge::OpType::FLASH_ATTN_GQA, DeviceType::CUDA,
                                                  {q, k, v}, params, 0);
        } catch (const std::runtime_error& e) {
            threw = true;
            what = e.what();
        }
        if (!threw) {
            std::printf("FAIL %s: expected loud reject\n", label);
            ++failures;
        } else {
            std::printf("  %s: loud reject OK\n", label);
        }
    };

    std::vector<uint16_t> qc;
    std::vector<float> kf, vf;
    rand_bf16((int64_t)nh * hd, -1.0f, 1.0f, g, qc);
    rand_f32((int64_t)kv_len * nkv * hd, -1.0f, 1.0f, g, kf);
    rand_f32((int64_t)kv_len * nkv * hd, -1.0f, 1.0f, g, vf);
    auto qb = make_dev_bf16(qc, 1, nh * hd);
    auto k32 = make_dev_fp32(kf, kv_len, nkv * hd);
    auto v32 = make_dev_fp32(vf, kv_len, nkv * hd);
    // 6-B-6.7: BF16 K with BF16 Q is now NATIVE (all-BF16 decode) -- verify it
    // does NOT throw, and returns BF16. (The old "authoritative FP32 KV" loud
    // reject is retired for the all-BF16 pairing only.)
    std::vector<uint16_t> kc2;
    rand_bf16((int64_t)kv_len * nkv * hd, -1.0f, 1.0f, g, kc2);
    {
        auto kb = make_dev_bf16(kc2, kv_len, nkv * hd);
        int32_t params[forge::OP_PARAMS_MAX_SIZE / sizeof(int32_t)] = {0};
        params[0] = nh;
        params[1] = nkv;
        params[2] = hd;
        params[3] = 1;
        params[4] = 1;
        TensorPtr out;
        try {
            out = forge::OpDispatch::instance().execute(forge::OpType::FLASH_ATTN_GQA,
                                                        DeviceType::CUDA, {qb, kb, kb}, params, 0);
        } catch (const std::exception& e) {
            std::printf("FAIL BF16-QKV dispatch threw: %s\n", e.what());
            ++failures;
        }
        if (out && out->dtype() != DataType::BF16) {
            std::printf("FAIL BF16-QKV dispatch: output dtype is not BF16\n");
            ++failures;
        } else if (out) {
            std::printf("  BF16-QKV dispatch: native BF16 out OK\n");
        }
    }
    // Mixed K/V (K BF16 + V FP32) must still throw loudly.
    {
        auto kb = make_dev_bf16(kc2, kv_len, nkv * hd);
        run_dispatch(qb, kb, v32, 1, hd, "BF16-Q + mixed K/V");
    }

    // 6-B-7.0: all-BF16 prefill (seq_len_q > 1) is now NATIVE -- verify it
    // does NOT throw, and returns BF16.
    {
        std::vector<uint16_t> qpc, kpc, vpc;
        rand_bf16((int64_t)4 * nh * hd, -1.0f, 1.0f, g, qpc);
        rand_bf16((int64_t)kv_len * nkv * hd, -1.0f, 1.0f, g, kpc);
        rand_bf16((int64_t)kv_len * nkv * hd, -1.0f, 1.0f, g, vpc);
        auto qpre = make_dev_bf16(qpc, 4, nh * hd);
        auto kpre = make_dev_bf16(kpc, kv_len, nkv * hd);
        auto vpre = make_dev_bf16(vpc, kv_len, nkv * hd);
        int32_t params[forge::OP_PARAMS_MAX_SIZE / sizeof(int32_t)] = {0};
        params[0] = nh;
        params[1] = nkv;
        params[2] = hd;
        params[3] = 1;
        params[4] = 1;
        TensorPtr out;
        try {
            out = forge::OpDispatch::instance().execute(
                forge::OpType::FLASH_ATTN_GQA, DeviceType::CUDA, {qpre, kpre, vpre}, params, 0);
        } catch (const std::exception& e) {
            std::printf("FAIL all-BF16 prefill dispatch threw: %s\n", e.what());
            ++failures;
        }
        if (out && out->dtype() != DataType::BF16) {
            std::printf("FAIL all-BF16 prefill dispatch: output dtype is not BF16\n");
            ++failures;
        } else if (out) {
            std::printf("  all-BF16 prefill dispatch: native BF16 out OK\n");
        }
    }
    // BF16-Q + FP32-KV prefill must still throw (no FP32-KV prefill kernel).
    {
        std::vector<uint16_t> qpc;
        rand_bf16((int64_t)4 * nh * hd, -1.0f, 1.0f, g, qpc);
        auto qpre = make_dev_bf16(qpc, 4, nh * hd);
        run_dispatch(qpre, k32, v32, 4, hd, "BF16-Q + FP32-KV prefill");
    }

    // hd256 BF16 must throw (16/32/64/96/128 only).
    run_dispatch(qb, k32, v32, 1, 256, "BF16-Q hd256");

    // Non-GQA BF16 (MHA) must throw.
    {
        int32_t params[forge::OP_PARAMS_MAX_SIZE / sizeof(int32_t)] = {0};
        params[0] = nh;
        params[1] = nh;  // nkv == nh -> MHA
        params[2] = hd;
        params[3] = 1;
        params[4] = 1;
        bool threw = false;
        try {
            forge::OpDispatch::instance().execute(forge::OpType::FLASH_ATTN_GQA, DeviceType::CUDA,
                                                  {qb, k32, v32}, params, 0);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        if (!threw) {
            std::printf("FAIL BF16-Q MHA: expected loud reject\n");
            ++failures;
        } else {
            std::printf("  BF16-Q MHA: loud reject OK\n");
        }
    }
}

// Graph integration: FLASH_ATTN_GQA node (BF16 Q + FP32 KV) -- eager vs
// dispatch vs capture vs replay, all bitwise; output dtype BF16.
void check_graph_node(int kv_len, int nh, int nkv, int hd, std::mt19937& g) {
    std::vector<uint16_t> qc;
    std::vector<float> kf, vf;
    rand_bf16((int64_t)nh * hd, -1.0f, 1.0f, g, qc);
    rand_f32((int64_t)kv_len * nkv * hd, -1.0f, 1.0f, g, kf);
    rand_f32((int64_t)kv_len * nkv * hd, -1.0f, 1.0f, g, vf);
    auto q = make_dev_bf16(qc, 1, nh * hd);
    auto k = make_dev_fp32(kf, kv_len, nkv * hd);
    auto v = make_dev_fp32(vf, kv_len, nkv * hd);

    auto graph = std::make_unique<forge::ComputeGraph>();
    int qi = graph->add_input(q);
    int ki = graph->add_input(k);
    int vi = graph->add_input(v);
    int32_t params[forge::OP_PARAMS_MAX_SIZE / sizeof(int32_t)] = {0};
    params[0] = nh;
    params[1] = nkv;
    params[2] = hd;
    params[3] = 1;  // causal
    params[4] = 1;  // CUDA
    graph->add_node("attn", forge::OpType::FLASH_ATTN_GQA, {qi, ki, vi}, params, DeviceType::CUDA);

    CHECK(forge::CudaGraphRunner::is_compatible(*graph));

    // A: implicit stream. Output dtype must be BF16 (contract, not just values).
    auto outA = graph->execute();
    CHECK(outA != nullptr);
    CHECK(outA->dtype() == DataType::BF16);
    cudaDeviceSynchronize();
    auto codesA = to_host_u16(graph->last_output());
    CHECK(graph->last_output()->nbytes() == static_cast<size_t>(nh * hd * 2));

    // B: explicit legacy stream.
    graph->execute(0);
    cudaDeviceSynchronize();
    CHECK(to_host_u16(graph->last_output()) == codesA);

    // C/D: capture + replay on a capture stream.
    cudaStream_t cs = nullptr;
    CHECK(cudaStreamCreate(&cs) == cudaSuccess);
    forge::CudaGraphRunner runner;
    runner.set_enabled(true);
    bool captured = runner.try_replay_or_capture(
        *graph, [&]() { graph->execute(cs); }, cs);
    cudaStreamSynchronize(cs);
    CHECK(captured);
    CHECK(to_host_u16(graph->last_output()) == codesA);
    bool replayed = runner.try_replay_or_capture(
        *graph, [&]() { graph->execute(cs); }, cs);
    cudaStreamSynchronize(cs);
    CHECK(replayed);
    CHECK(to_host_u16(graph->last_output()) == codesA);
    cudaStreamDestroy(cs);
    std::printf(
        "  graph node kv=%d nh=%d nkv=%d hd=%d : eager/capture/replay bitwise OK (BF16-Q)\n",
        kv_len, nh, nkv, hd);
}

// 6-B-6.7 graph node variant: all-BF16 QKV -- eager vs capture vs replay,
// all bitwise; output dtype BF16.
void check_graph_node_bf16kv(int kv_len, int nh, int nkv, int hd, std::mt19937& g) {
    std::vector<uint16_t> qc, kc, vc;
    rand_bf16((int64_t)nh * hd, -1.0f, 1.0f, g, qc);
    rand_bf16((int64_t)kv_len * nkv * hd, -1.0f, 1.0f, g, kc);
    rand_bf16((int64_t)kv_len * nkv * hd, -1.0f, 1.0f, g, vc);
    auto q = make_dev_bf16(qc, 1, nh * hd);
    auto k = make_dev_bf16(kc, kv_len, nkv * hd);
    auto v = make_dev_bf16(vc, kv_len, nkv * hd);

    auto graph = std::make_unique<forge::ComputeGraph>();
    int qi = graph->add_input(q);
    int ki = graph->add_input(k);
    int vi = graph->add_input(v);
    int32_t params[forge::OP_PARAMS_MAX_SIZE / sizeof(int32_t)] = {0};
    params[0] = nh;
    params[1] = nkv;
    params[2] = hd;
    params[3] = 1;  // causal
    params[4] = 1;  // CUDA
    graph->add_node("attn", forge::OpType::FLASH_ATTN_GQA, {qi, ki, vi}, params, DeviceType::CUDA);

    CHECK(forge::CudaGraphRunner::is_compatible(*graph));

    auto outA = graph->execute();
    CHECK(outA != nullptr);
    CHECK(outA->dtype() == DataType::BF16);
    cudaDeviceSynchronize();
    auto codesA = to_host_u16(graph->last_output());
    CHECK(graph->last_output()->nbytes() == static_cast<size_t>(nh * hd * 2));

    graph->execute(0);
    cudaDeviceSynchronize();
    CHECK(to_host_u16(graph->last_output()) == codesA);

    cudaStream_t cs = nullptr;
    CHECK(cudaStreamCreate(&cs) == cudaSuccess);
    forge::CudaGraphRunner runner;
    runner.set_enabled(true);
    bool captured = runner.try_replay_or_capture(
        *graph, [&]() { graph->execute(cs); }, cs);
    cudaStreamSynchronize(cs);
    CHECK(captured);
    CHECK(to_host_u16(graph->last_output()) == codesA);
    bool replayed = runner.try_replay_or_capture(
        *graph, [&]() { graph->execute(cs); }, cs);
    cudaStreamSynchronize(cs);
    CHECK(replayed);
    CHECK(to_host_u16(graph->last_output()) == codesA);
    cudaStreamDestroy(cs);
    std::printf("  graph node bf16kv kv=%d nh=%d nkv=%d hd=%d : eager/capture/replay bitwise OK\n",
                kv_len, nh, nkv, hd);
}

// 6-B-6.6/6.7: decode benchmark, three arms -- all-BF16 QKV vs BF16-Q+FP32-KV
// (the 6-B-6 baseline) vs all-FP32. Protocol (per the 6-B-3 lessons):
// strictly alternating arms, best-of-N per arm. No speedup is preset; kv
// sweep covers launch-bound -> bandwidth-bound. The question under test:
// does BF16 KV cash the memory-traffic gain the FP32-KV slice proved absent?
void bench_decode(int kv_len, int nh, int nkv, int hd) {
    std::mt19937 g(0xB660 + kv_len);
    std::vector<uint16_t> qc, kc, vc;
    std::vector<float> qf, kf, vf;
    rand_bf16((int64_t)nh * hd, -1.0f, 1.0f, g, qc);
    qf.resize(qc.size());
    for (size_t i = 0; i < qc.size(); ++i)
        qf[i] = bf16_to_f32(qc[i]);
    rand_bf16((int64_t)kv_len * nkv * hd, -1.0f, 1.0f, g, kc);
    rand_bf16((int64_t)kv_len * nkv * hd, -1.0f, 1.0f, g, vc);
    kf.resize(kc.size());
    vf.resize(vc.size());
    // Same decoded values in the FP32 buffers: arms differ ONLY in storage.
    for (size_t i = 0; i < kc.size(); ++i)
        kf[i] = bf16_to_f32(kc[i]);
    for (size_t i = 0; i < vc.size(); ++i)
        vf[i] = bf16_to_f32(vc[i]);
    auto qb = make_dev_bf16(qc, 1, nh * hd);
    auto q32 = make_dev_fp32(qf, 1, nh * hd);
    auto kb = make_dev_bf16(kc, kv_len, nkv * hd);
    auto vb = make_dev_bf16(vc, kv_len, nkv * hd);
    auto k32 = make_dev_fp32(kf, kv_len, nkv * hd);
    auto v32 = make_dev_fp32(vf, kv_len, nkv * hd);
    auto ob = std::make_shared<Tensor>(DataType::BF16, std::vector<int64_t>{1, nh * hd},
                                       DeviceType::CUDA);
    auto obq = std::make_shared<Tensor>(DataType::BF16, std::vector<int64_t>{1, nh * hd},
                                        DeviceType::CUDA);
    auto o32 = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, nh * hd},
                                        DeviceType::CUDA);

    auto time_bfkv = [&](int iters) {
        cudaEvent_t es, ee;
        cudaEventCreate(&es);
        cudaEventCreate(&ee);
        cudaEventRecord(es);
        for (int i = 0; i < iters; ++i)
            forge::cuda::launch_flash_attention_gqa_decode_bf16(
                reinterpret_cast<const __nv_bfloat16*>(qb->data()),
                reinterpret_cast<const __nv_bfloat16*>(kb->data()),
                reinterpret_cast<const __nv_bfloat16*>(vb->data()),
                reinterpret_cast<__nv_bfloat16*>(ob->data()), kv_len, nh, nkv, hd, nullptr, 0);
        cudaEventRecord(ee);
        cudaEventSynchronize(ee);
        float ms = 0;
        cudaEventElapsedTime(&ms, es, ee);
        cudaEventDestroy(es);
        cudaEventDestroy(ee);
        return ms / iters;
    };
    auto time_bfq = [&](int iters) {
        cudaEvent_t es, ee;
        cudaEventCreate(&es);
        cudaEventCreate(&ee);
        cudaEventRecord(es);
        for (int i = 0; i < iters; ++i)
            forge::cuda::launch_flash_attention_gqa_decode_bf16q(
                reinterpret_cast<const __nv_bfloat16*>(qb->data()),
                static_cast<const float*>(k32->data()), static_cast<const float*>(v32->data()),
                reinterpret_cast<__nv_bfloat16*>(obq->data()), kv_len, nh, nkv, hd, nullptr, 0);
        cudaEventRecord(ee);
        cudaEventSynchronize(ee);
        float ms = 0;
        cudaEventElapsedTime(&ms, es, ee);
        cudaEventDestroy(es);
        cudaEventDestroy(ee);
        return ms / iters;
    };
    auto time_f32 = [&](int iters) {
        cudaEvent_t es, ee;
        cudaEventCreate(&es);
        cudaEventCreate(&ee);
        cudaEventRecord(es);
        for (int i = 0; i < iters; ++i)
            forge::cuda::launch_flash_attention_gqa_decode(
                static_cast<const float*>(q32->data()), static_cast<const float*>(k32->data()),
                static_cast<const float*>(v32->data()), static_cast<float*>(o32->data()), kv_len,
                nh, nkv, hd, nullptr, 0);
        cudaEventRecord(ee);
        cudaEventSynchronize(ee);
        float ms = 0;
        cudaEventElapsedTime(&ms, es, ee);
        cudaEventDestroy(es);
        cudaEventDestroy(ee);
        return ms / iters;
    };

    const int iters = (kv_len <= 512) ? 200 : 50;
    for (int i = 0; i < 2; ++i) {
        time_bfkv(5);
        time_bfq(5);
        time_f32(5);
    }
    cudaDeviceSynchronize();
    float best_bfkv = 1e30f, best_bfq = 1e30f, best_32 = 1e30f;
    for (int r = 0; r < 3; ++r) {
        best_bfkv = std::min(best_bfkv, time_bfkv(iters));
        best_bfq = std::min(best_bfq, time_bfq(iters));
        best_32 = std::min(best_32, time_f32(iters));
    }
    std::printf(
        "bench decode kv=%5d nh=%d nkv=%d hd=%d : BF16-KV %.5f ms/tok | BF16-Q %.5f ms/tok | FP32 "
        "%.5f ms/tok | kv-gain %.2fx\n",
        kv_len, nh, nkv, hd, best_bfkv, best_bfq, best_32, best_bfq / best_bfkv);
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("SKIP: no CUDA device available (test_attention_bf16)\n");
        return 0;
    }
    forge::register_builtin_op_kernels();

    if (!forge::op_supports_lowp(forge::OpType::FLASH_ATTN_GQA)) {
        std::printf("FAIL: op_supports_lowp(FLASH_ATTN_GQA) expected true (6-A-7 + 6-B-6)\n");
        return 1;
    }

    std::printf("[6-B-6] BF16-Q decode correctness vs FP32 reference\n");
    std::mt19937 rng(0xB616);

    for (int hd : {16, 32, 64, 128}) {
        check_decode(512, 8, 2, hd, rng);
    }
    std::printf("[6-B-6.7] all-BF16 decode correctness vs FP32 reference\n");
    for (int hd : {16, 32, 64, 128}) {
        check_decode_bf16(512, 8, 2, hd, rng);
    }
    std::printf("[6-B-7.0] all-BF16 prefill correctness vs causal FP32 reference\n");
    for (int hd : {16, 64, 128}) {
        check_prefill_bf16(4, 16, 8, 2, hd, rng);
    }
    std::printf("[6-B-6] FP32 self-consistency (same values, FP32 kernel vs BF16-Q kernel)\n");
    for (int hd : {16, 32, 64, 128}) {
        check_fp32_consistency(512, 8, 2, hd, rng);
    }
    std::printf("[6-B-6] selector + loud-rejection surface\n");
    check_selector();
    check_loud(128, 8, 2, 64, rng);
    std::printf("[6-B-6] graph node (BF16 Q + FP32 KV)\n");
    check_graph_node(128, 8, 2, 64, rng);
    std::printf("[6-B-6.7] graph node (all-BF16 QKV)\n");
    check_graph_node_bf16kv(128, 8, 2, 64, rng);

    if (failures == 0) {
        std::printf(
            "[6-B-6.6/6.7] decode benchmark (BF16-KV vs BF16-Q vs FP32, best-of-3 alternating)\n");
        bench_decode(128, 8, 2, 64);
        bench_decode(512, 8, 2, 64);
        bench_decode(2048, 8, 2, 64);
        std::printf("PASS attn-bf16 (Phase 6-B-6 + 6-B-6.7)\n");
        return 0;
    }
    std::printf("FAILURES: %d\n", failures);
    return 1;
}
