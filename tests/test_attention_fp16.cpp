// =========================================================================
// test_attention_fp16.cpp -- Phase 6-A-7: native FP16 attention
//
// 7.1 (this file, decode-first): FP16 Q/K/V storage, FP32 QK^T / online
//     softmax / PV, FP16 output. Structural mirror of the FP32 streaming
//     decode kernel; no half2, no tiling redesign, no TensorCore.
// 7.2 (prefill) lands separately.
//
// Acceptance (decode slice):
//   1. FP16 Q/K/V -> FP16 output (dtype hard asserts, not just numerics)
//   2. vs CPU FP32 reference (parity) + vs FP32 kernel self-consistency
//   3. FP32 accumulation/softmax by construction (kernel diff review; the
//      parity test catches scale/ordering regressions)
//   4. ordinary stream == capture stream (bitwise)
//   5. ComputeGraph FLASH_ATTN_GQA node capture/replay (bitwise)
//
// Requires CUDA; skips gracefully without a GPU.
// =========================================================================

#include <cuda_fp16.h>
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
#include "forge/op_dispatch.h"
#include "forge/op_enum.h"
#include "forge/tensor.h"
#include "forge/types.h"

namespace {

using forge::DataType;
using forge::DeviceType;
using forge::Tensor;
using forge::TensorPtr;

uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    const uint32_t man = x & 0x7FFFFFu;
    if (exp <= 0)
        return static_cast<uint16_t>(sign);
    if (exp >= 31)
        return static_cast<uint16_t>(sign | 0x7C00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (man >> 13));
}
float f16_to_f32(uint16_t h) {
    const uint32_t sign = (h >> 15) & 1u;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x3FFu;
    float s = sign ? -1.0f : 1.0f;
    if (exp == 0)
        return (mant == 0) ? 0.0f : std::ldexp(static_cast<float>(mant), -24) * s;
    if (exp == 31)
        return std::numeric_limits<float>::infinity() * s;
    return std::ldexp(static_cast<float>(mant + 1024), static_cast<int>(exp) - 25) * s;
}

TensorPtr make_dev_fp16(const std::vector<uint16_t>& h, int64_t r, int64_t c) {
    auto host = std::make_shared<Tensor>(DataType::FP16, std::vector<int64_t>{r, c},
                                         DeviceType::CPU);
    std::memcpy(host->data(), h.data(), h.size() * 2);
    auto dev = std::make_shared<Tensor>(DataType::FP16, std::vector<int64_t>{r, c},
                                        DeviceType::CUDA);
    dev->copy_from(*host);
    return dev;
}
TensorPtr make_dev_fp32(const std::vector<float>& h, int64_t r, int64_t c) {
    auto host = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{r, c},
                                         DeviceType::CPU);
    std::memcpy(host->data(), h.data(), h.size() * 4);
    auto dev = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{r, c},
                                        DeviceType::CUDA);
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
    if (t->dtype() == DataType::FP16) {
        std::vector<uint16_t> c(v.size());
        std::memcpy(c.data(), host->data(), c.size() * 2);
        for (size_t i = 0; i < v.size(); ++i) v[i] = f16_to_f32(c[i]);
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

void rand_fp16(int64_t n, float lo, float hi, std::mt19937& g, std::vector<uint16_t>& codes) {
    std::uniform_real_distribution<float> d(lo, hi);
    codes.resize(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) codes[i] = f32_to_f16(d(g));
}

// CPU FP32 GQA reference (2-pass softmax). score = dot(q,k) * rsqrt(head_dim)
// — the same scale convention as the CUDA kernels (cuda_flash_attn.cu).
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
            if (scores[j] > mx) mx = scores[j];
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

// 7.1 decode correctness: direct kernel vs CPU FP32 ref, dtype hard asserts,
// ordinary stream == capture stream (bitwise).
void check_decode(int kv_len, int nh, int nkv, int hd, std::mt19937& g) {
    std::vector<uint16_t> qc, kc, vc;
    rand_fp16((int64_t)nh * hd, -1.0f, 1.0f, g, qc);
    rand_fp16((int64_t)kv_len * nkv * hd, -1.0f, 1.0f, g, kc);
    rand_fp16((int64_t)kv_len * nkv * hd, -1.0f, 1.0f, g, vc);
    auto q = make_dev_fp16(qc, 1, nh * hd);
    auto k = make_dev_fp16(kc, kv_len, nkv * hd);
    auto v = make_dev_fp16(vc, kv_len, nkv * hd);
    auto o = std::make_shared<Tensor>(DataType::FP16, std::vector<int64_t>{1, nh * hd},
                                      DeviceType::CUDA);

    // A: legacy stream.
    forge::cuda::launch_flash_attention_gqa_decode_fp16(
        reinterpret_cast<const __half*>(q->data()), reinterpret_cast<const __half*>(k->data()),
        reinterpret_cast<const __half*>(v->data()), reinterpret_cast<__half*>(o->data()), kv_len,
        nh, nkv, hd, nullptr, 0);
    cudaDeviceSynchronize();
    auto codesA = to_host_u16(o);

    // B: explicit capture stream — must match A bitwise.
    cudaStream_t cs = nullptr;
    CHECK(cudaStreamCreate(&cs) == cudaSuccess);
    auto oB = std::make_shared<Tensor>(DataType::FP16, std::vector<int64_t>{1, nh * hd},
                                       DeviceType::CUDA);
    forge::cuda::launch_flash_attention_gqa_decode_fp16(
        reinterpret_cast<const __half*>(q->data()), reinterpret_cast<const __half*>(k->data()),
        reinterpret_cast<const __half*>(v->data()), reinterpret_cast<__half*>(oB->data()), kv_len,
        nh, nkv, hd, nullptr, cs);
    cudaStreamSynchronize(cs);
    auto codesB = to_host_u16(oB);
    CHECK(codesA == codesB);
    cudaStreamDestroy(cs);

    // Reference: upcast inputs, CPU FP32 GQA, compare in FP32.
    std::vector<float> qf(qc.size()), kf(kc.size()), vf(vc.size());
    for (size_t i = 0; i < qc.size(); ++i) qf[i] = f16_to_f32(qc[i]);
    for (size_t i = 0; i < kc.size(); ++i) kf[i] = f16_to_f32(kc[i]);
    for (size_t i = 0; i < vc.size(); ++i) vf[i] = f16_to_f32(vc[i]);
    std::vector<float> ref(static_cast<size_t>(nh * hd));
    ref_gqa_decode_fp32(qf.data(), kf.data(), vf.data(), ref.data(), kv_len, nh, nkv, hd);

    auto got = to_host_f32(o);
    float max_abs = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i)
        max_abs = std::max(max_abs, std::fabs(got[i] - ref[i]));
    std::printf("  decode kv=%d nh=%d nkv=%d hd=%d : max_abs=%.3e (A==B: %d)\n", kv_len, nh, nkv,
                hd, max_abs, codesA == codesB ? 1 : 0);
    // FP16-output rounding dominates (~1e-3); kernel math itself is FP32.
    CHECK(max_abs < 1e-2f);
}

// FP32 self-consistency: same values through the FP32 decode kernel must match
// the FP16 kernel to FP16 rounding (no structural deviation).
void check_fp32_consistency(int kv_len, int nh, int nkv, int hd, std::mt19937& g) {
    std::vector<uint16_t> qc, kc, vc;
    rand_fp16((int64_t)nh * hd, -1.0f, 1.0f, g, qc);
    rand_fp16((int64_t)kv_len * nkv * hd, -1.0f, 1.0f, g, kc);
    rand_fp16((int64_t)kv_len * nkv * hd, -1.0f, 1.0f, g, vc);
    std::vector<float> qf(qc.size()), kf(kc.size()), vf(vc.size());
    for (size_t i = 0; i < qc.size(); ++i) qf[i] = f16_to_f32(qc[i]);
    for (size_t i = 0; i < kc.size(); ++i) kf[i] = f16_to_f32(kc[i]);
    for (size_t i = 0; i < vc.size(); ++i) vf[i] = f16_to_f32(vc[i]);
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

    auto q16 = make_dev_fp16(qc, 1, nh * hd);
    auto k16 = make_dev_fp16(kc, kv_len, nkv * hd);
    auto v16 = make_dev_fp16(vc, kv_len, nkv * hd);
    auto o16 = std::make_shared<Tensor>(DataType::FP16, std::vector<int64_t>{1, nh * hd},
                                        DeviceType::CUDA);
    forge::cuda::launch_flash_attention_gqa_decode_fp16(
        reinterpret_cast<const __half*>(q16->data()), reinterpret_cast<const __half*>(k16->data()),
        reinterpret_cast<const __half*>(v16->data()), reinterpret_cast<__half*>(o16->data()),
        kv_len, nh, nkv, hd, nullptr, 0);
    cudaDeviceSynchronize();

    auto got32 = to_host_f32(o32);
    auto got16 = to_host_f32(o16);
    float max_abs = 0.0f;
    for (size_t i = 0; i < got32.size(); ++i)
        max_abs = std::max(max_abs, std::fabs(got16[i] - got32[i]));
    std::printf("  fp32-consistency kv=%d nh=%d nkv=%d hd=%d : max_abs=%.3e\n", kv_len, nh, nkv,
                hd, max_abs);
    CHECK(max_abs < 1e-2f);
}

// Phase 2 round 2: __half2 decode candidate. Gates:
//   1. stream A/B: candidate on legacy stream == candidate on capture stream,
//      BITWISE (same binary);
//   2. vs the 6-A-7 baseline: NOT bitwise. FMA-contraction decisions differ
//      between the two compilations (baseline reads K/V straight from global;
//      the candidate reads the hoisted v8[] register array), so products can
//      be rounded differently (e.g. 2xFMUL+FADD vs FFMA). The honest gate is
//      FP16-output equivalence: max_abs <= ~1 ULP (< 1e-3 for |out| <= 1);
//   3. parity vs the CPU FP32 reference (same bound as the baseline decode).
void check_decode_h2_candidate(int kv_len, int nh, int nkv, int hd, std::mt19937& g) {
    std::vector<uint16_t> qc, kc, vc;
    rand_fp16((int64_t)nh * hd, -1.0f, 1.0f, g, qc);
    rand_fp16((int64_t)kv_len * nkv * hd, -1.0f, 1.0f, g, kc);
    rand_fp16((int64_t)kv_len * nkv * hd, -1.0f, 1.0f, g, vc);
    auto q = make_dev_fp16(qc, 1, nh * hd);
    auto k = make_dev_fp16(kc, kv_len, nkv * hd);
    auto v = make_dev_fp16(vc, kv_len, nkv * hd);
    auto run = [&](TensorPtr& o, cudaStream_t st, bool candidate) {
        const __half* qh = reinterpret_cast<const __half*>(q->data());
        const __half* kh = reinterpret_cast<const __half*>(k->data());
        const __half* vh = reinterpret_cast<const __half*>(v->data());
        __half* oh = reinterpret_cast<__half*>(o->data());
        if (candidate)
            forge::cuda::launch_flash_attention_gqa_decode_fp16_h2(qh, kh, vh, oh, kv_len, nh,
                                                                   nkv, hd, nullptr, st);
        else
            forge::cuda::launch_flash_attention_gqa_decode_fp16(qh, kh, vh, oh, kv_len, nh, nkv,
                                                                hd, nullptr, st);
    };
    auto oA = std::make_shared<Tensor>(DataType::FP16, std::vector<int64_t>{1, nh * hd},
                                       DeviceType::CUDA);
    run(oA, 0, false);  // baseline
    auto oC = std::make_shared<Tensor>(DataType::FP16, std::vector<int64_t>{1, nh * hd},
                                       DeviceType::CUDA);
    run(oC, 0, true);  // candidate, legacy stream
    cudaStream_t cs = nullptr;
    CHECK(cudaStreamCreate(&cs) == cudaSuccess);
    auto oC2 = std::make_shared<Tensor>(DataType::FP16, std::vector<int64_t>{1, nh * hd},
                                        DeviceType::CUDA);
    run(oC2, cs, true);  // candidate, capture stream
    cudaStreamSynchronize(cs);
    cudaStreamDestroy(cs);

    auto codesA = to_host_u16(oA);
    auto codesC = to_host_u16(oC);
    auto codesC2 = to_host_u16(oC2);
    CHECK(codesC2 == codesC);  // candidate stream A/B, bitwise (same binary)

    // Candidate vs baseline: <= ~1 FP16 output ULP (FMA-contraction noise).
    auto got = to_host_f32(oC);
    auto gotA = to_host_f32(oA);
    float max_vs_base = 0.0f;
    for (size_t i = 0; i < gotA.size(); ++i)
        max_vs_base = std::max(max_vs_base, std::fabs(got[i] - gotA[i]));

    // Parity vs CPU FP32 reference (same inputs, same bound as baseline).
    std::vector<float> qf(qc.size()), kf(kc.size()), vf(vc.size());
    for (size_t i = 0; i < qc.size(); ++i) qf[i] = f16_to_f32(qc[i]);
    for (size_t i = 0; i < kc.size(); ++i) kf[i] = f16_to_f32(kc[i]);
    for (size_t i = 0; i < vc.size(); ++i) vf[i] = f16_to_f32(vc[i]);
    std::vector<float> ref(static_cast<size_t>(nh * hd));
    ref_gqa_decode_fp32(qf.data(), kf.data(), vf.data(), ref.data(), kv_len, nh, nkv, hd);
    float max_abs = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i)
        max_abs = std::max(max_abs, std::fabs(got[i] - ref[i]));
    std::printf("  decode-h2 kv=%d nh=%d nkv=%d hd=%d : max_abs=%.3e (vs baseline %.3e, "
                "streamA==B: %d)\n",
                kv_len, nh, nkv, hd, max_abs, max_vs_base, codesC2 == codesC ? 1 : 0);
    CHECK(max_vs_base < 1e-3f);  // <= ~1 FP16 output ULP
    CHECK(max_abs < 1e-2f);
}

// Selector unit test: pure function, q_dtype drives the F16 paths.
void check_selector() {
    using forge::AttentionPath;
    forge::AttentionProblem p;
    p.device = DeviceType::CUDA;
    p.num_heads = 8;
    p.num_kv_heads = 2;
    p.head_dim = 64;

    p.seq_len = 1;
    p.q_dtype = DataType::FP16;
    CHECK(forge::choose_attention_path(p) == AttentionPath::CUDA_F16_DECODE);
    p.q_dtype = DataType::FP32;
    CHECK(forge::choose_attention_path(p) == AttentionPath::CUDA_FP32_DECODE);

    p.seq_len = 16;
    p.q_dtype = DataType::FP16;
    CHECK(forge::choose_attention_path(p) == AttentionPath::CUDA_F16_PREFILL);
    p.q_dtype = DataType::FP32;
    CHECK(forge::choose_attention_path(p) == AttentionPath::CUDA_FP32_PREFILL);

    CHECK(std::string(forge::attention_path_name(AttentionPath::CUDA_F16_DECODE)) ==
          "CUDA_F16_DECODE");
    std::printf("  selector: F16/FP32 decode+prefill paths OK\n");
}

// Graph integration: FLASH_ATTN_GQA node (FP16) — eager vs dispatch vs
// capture vs replay, all bitwise; output dtype FP16.
void check_graph_node(int kv_len, int nh, int nkv, int hd, std::mt19937& g) {
    std::vector<uint16_t> qc, kc, vc;
    rand_fp16((int64_t)nh * hd, -1.0f, 1.0f, g, qc);
    rand_fp16((int64_t)kv_len * nkv * hd, -1.0f, 1.0f, g, kc);
    rand_fp16((int64_t)kv_len * nkv * hd, -1.0f, 1.0f, g, vc);
    auto q = make_dev_fp16(qc, 1, nh * hd);
    auto k = make_dev_fp16(kc, kv_len, nkv * hd);
    auto v = make_dev_fp16(vc, kv_len, nkv * hd);

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
    graph->add_node("attn", forge::OpType::FLASH_ATTN_GQA, {qi, ki, vi}, params,
                    DeviceType::CUDA);

    CHECK(forge::CudaGraphRunner::is_compatible(*graph));

    // A: implicit stream. Output dtype must be FP16 (contract, not just values).
    auto outA = graph->execute();
    CHECK(outA != nullptr);
    CHECK(outA->dtype() == DataType::FP16);
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
    bool captured = runner.try_replay_or_capture(*graph, [&]() { graph->execute(cs); }, cs);
    cudaStreamSynchronize(cs);
    CHECK(captured);
    CHECK(to_host_u16(graph->last_output()) == codesA);
    bool replayed = runner.try_replay_or_capture(*graph, [&]() { graph->execute(cs); }, cs);
    cudaStreamSynchronize(cs);
    CHECK(replayed);
    CHECK(to_host_u16(graph->last_output()) == codesA);
    cudaStreamDestroy(cs);
    std::printf("  graph node kv=%d nh=%d nkv=%d hd=%d : eager/capture/replay bitwise OK (FP16)\n",
                kv_len, nh, nkv, hd);
}

// CPU FP32 GQA prefill reference (causal). q_pos = kv_len - q_len + qi —
// the same convention as the vectorized CUDA kernels.
void ref_gqa_prefill_fp32(const float* Q, const float* K, const float* V, float* O, int q_len,
                          int kv_len, int nh, int nkv, int hd, bool causal) {
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
                if (causal && j > q_pos) {
                    scores[j] = -1e30f;
                    continue;
                }
                float dot = 0.0f;
                for (int d = 0; d < hd; ++d)
                    dot += qr[d] * K[(size_t)j * nkv * hd + kvh * hd + d];
                scores[j] = dot * scale;
                if (scores[j] > mx) mx = scores[j];
            }
            float sum = 0.0f;
            for (int j = 0; j < kv_len; ++j) {
                scores[j] = std::exp(scores[j] - mx);
                sum += scores[j];
            }
            float* orow = O + ((size_t)qi * nh + h) * hd;
            for (int d = 0; d < hd; ++d) {
                float acc = 0.0f;
                for (int j = 0; j < kv_len; ++j)
                    acc += scores[j] * V[(size_t)j * nkv * hd + kvh * hd + d];
                orow[d] = acc / sum;
            }
        }
    }
}

// 7.2 prefill correctness: direct kernel vs CPU ref, stream A/B, dtype hard.
void check_prefill(int q_len, int kv_len, int nh, int nkv, int hd, std::mt19937& g) {
    std::vector<uint16_t> qc, kc, vc;
    rand_fp16((int64_t)q_len * nh * hd, -1.0f, 1.0f, g, qc);
    rand_fp16((int64_t)kv_len * nkv * hd, -1.0f, 1.0f, g, kc);
    rand_fp16((int64_t)kv_len * nkv * hd, -1.0f, 1.0f, g, vc);
    auto q = make_dev_fp16(qc, q_len, nh * hd);
    auto k = make_dev_fp16(kc, kv_len, nkv * hd);
    auto v = make_dev_fp16(vc, kv_len, nkv * hd);
    auto o = std::make_shared<Tensor>(DataType::FP16, std::vector<int64_t>{q_len, nh * hd},
                                      DeviceType::CUDA);
    CHECK(o->dtype() == DataType::FP16);

    forge::cuda::launch_flash_attention_gqa_prefill_fp16(
        reinterpret_cast<const __half*>(q->data()), reinterpret_cast<const __half*>(k->data()),
        reinterpret_cast<const __half*>(v->data()), reinterpret_cast<__half*>(o->data()), q_len,
        kv_len, nh, nkv, hd, nullptr, true, 0);
    cudaDeviceSynchronize();
    auto codesA = to_host_u16(o);

    cudaStream_t cs = nullptr;
    CHECK(cudaStreamCreate(&cs) == cudaSuccess);
    auto oB = std::make_shared<Tensor>(DataType::FP16, std::vector<int64_t>{q_len, nh * hd},
                                       DeviceType::CUDA);
    forge::cuda::launch_flash_attention_gqa_prefill_fp16(
        reinterpret_cast<const __half*>(q->data()), reinterpret_cast<const __half*>(k->data()),
        reinterpret_cast<const __half*>(v->data()), reinterpret_cast<__half*>(oB->data()), q_len,
        kv_len, nh, nkv, hd, nullptr, true, cs);
    cudaStreamSynchronize(cs);
    CHECK(to_host_u16(oB) == codesA);
    cudaStreamDestroy(cs);

    std::vector<float> qf(qc.size()), kf(kc.size()), vf(vc.size());
    for (size_t i = 0; i < qc.size(); ++i) qf[i] = f16_to_f32(qc[i]);
    for (size_t i = 0; i < kc.size(); ++i) kf[i] = f16_to_f32(kc[i]);
    for (size_t i = 0; i < vc.size(); ++i) vf[i] = f16_to_f32(vc[i]);
    std::vector<float> ref((size_t)q_len * nh * hd);
    ref_gqa_prefill_fp32(qf.data(), kf.data(), vf.data(), ref.data(), q_len, kv_len, nh, nkv, hd,
                         true);
    auto got = to_host_f32(o);
    float max_abs = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i)
        max_abs = std::max(max_abs, std::fabs(got[i] - ref[i]));
    std::printf("  prefill q=%d kv=%d nh=%d nkv=%d hd=%d : max_abs=%.3e\n", q_len, kv_len, nh,
                nkv, hd, max_abs);
    CHECK(max_abs < 1e-2f);
}

// Phase 2 round 3: TensorCore (wmma) prefill candidate. Gates:
//   1. stream A/B: candidate legacy == candidate capture, BITWISE (same binary);
//   2. vs the 6-A-7 vectorized baseline: NOT bitwise (P operand stored f16 --
//      the documented contract amendment -- plus a different f32 accumulation
//      tree); honest gate: max_abs < 5e-3;
//   3. vs the CPU FP32 reference: same bound as the baseline prefill (1e-2).
// hd=64 only (candidate scope); covers q/kv tails (non-multiples of 32) and
// kv > q. Round 4 adds check_prefill_mma_cfg_scan: the same gates for every
// scanned tile config via the explicit-config entry.
void check_prefill_mma_candidate(int q_len, int kv_len, int nh, int nkv, std::mt19937& g) {
    const int hd = 64;
    std::vector<uint16_t> qc, kc, vc;
    rand_fp16((int64_t)q_len * nh * hd, -1.0f, 1.0f, g, qc);
    rand_fp16((int64_t)kv_len * nkv * hd, -1.0f, 1.0f, g, kc);
    rand_fp16((int64_t)kv_len * nkv * hd, -1.0f, 1.0f, g, vc);
    auto q = make_dev_fp16(qc, q_len, nh * hd);
    auto k = make_dev_fp16(kc, kv_len, nkv * hd);
    auto v = make_dev_fp16(vc, kv_len, nkv * hd);
    auto run = [&](TensorPtr& o, cudaStream_t st, bool candidate) {
        const __half* qh = reinterpret_cast<const __half*>(q->data());
        const __half* kh = reinterpret_cast<const __half*>(k->data());
        const __half* vh = reinterpret_cast<const __half*>(v->data());
        __half* oh = reinterpret_cast<__half*>(o->data());
        if (candidate)
            forge::cuda::launch_flash_attention_gqa_prefill_mma_fp16(qh, kh, vh, oh, q_len, kv_len,
                                                                     nh, nkv, hd, true, st);
        else  // frozen 6-A-7.2 baseline arm -- explicit vectorized entry
            forge::cuda::launch_flash_attention_gqa_prefill_fp16_vectorized(
                qh, kh, vh, oh, q_len, kv_len, nh, nkv, hd, nullptr, true, st);
    };
    auto oA = std::make_shared<Tensor>(DataType::FP16, std::vector<int64_t>{q_len, nh * hd},
                                       DeviceType::CUDA);
    run(oA, 0, false);  // baseline
    auto oC = std::make_shared<Tensor>(DataType::FP16, std::vector<int64_t>{q_len, nh * hd},
                                       DeviceType::CUDA);
    run(oC, 0, true);  // candidate, legacy stream
    cudaStream_t cs = nullptr;
    CHECK(cudaStreamCreate(&cs) == cudaSuccess);
    auto oC2 = std::make_shared<Tensor>(DataType::FP16, std::vector<int64_t>{q_len, nh * hd},
                                        DeviceType::CUDA);
    run(oC2, cs, true);  // candidate, capture stream
    cudaStreamSynchronize(cs);
    cudaStreamDestroy(cs);

    auto codesC = to_host_u16(oC);
    auto codesC2 = to_host_u16(oC2);
    CHECK(codesC2 == codesC);  // candidate stream A/B, bitwise

    // vs CPU FP32 reference
    std::vector<float> qf(qc.size()), kf(kc.size()), vf(vc.size());
    for (size_t i = 0; i < qc.size(); ++i) qf[i] = f16_to_f32(qc[i]);
    for (size_t i = 0; i < kc.size(); ++i) kf[i] = f16_to_f32(kc[i]);
    for (size_t i = 0; i < vc.size(); ++i) vf[i] = f16_to_f32(vc[i]);
    std::vector<float> ref((size_t)q_len * nh * hd);
    ref_gqa_prefill_fp32(qf.data(), kf.data(), vf.data(), ref.data(), q_len, kv_len, nh, nkv, hd,
                         true);

    auto got = to_host_f32(oC);
    auto gotA = to_host_f32(oA);
    float max_abs = 0.0f, max_vs_base = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
        max_abs = std::max(max_abs, std::fabs(got[i] - ref[i]));
        max_vs_base = std::max(max_vs_base, std::fabs(got[i] - gotA[i]));
    }
    std::printf("  prefill-mma q=%d kv=%d nh=%d nkv=%d hd=%d : max_abs=%.3e (vs baseline %.3e, "
                "streamA==B: %d)\n",
                q_len, kv_len, nh, nkv, hd, max_abs, max_vs_base, codesC2 == codesC ? 1 : 0);
    CHECK(max_vs_base < 5e-3f);
    CHECK(max_abs < 1e-2f);
}

// Phase 2 round 4: run the round-3 gates for every scanned tile config and
// report one aggregated line per config (worst-case errors over the shapes).
void check_prefill_mma_cfg_scan(std::mt19937& g) {
    struct Cfg {
        int br, bc, warps;
    };
    const Cfg cfgs[] = {{32, 16, 2}, {32, 32, 2}, {32, 64, 2}, {64, 16, 2}, {64, 32, 2},
                        {64, 64, 2}, {64, 16, 4}, {64, 32, 4}, {64, 64, 4}};
    struct Shape {
        int q, kv;
    };
    const Shape shapes[] = {{16, 64}, {64, 64}, {48, 80}, {64, 128}};
    for (const auto& c : cfgs) {
        float worst_ref = 0.0f, worst_base = 0.0f;
        int ab_ok = 1;
        for (const auto& s : shapes) {
            const int q_len = s.q, kv_len = s.kv, nh = 8, nkv = 2, hd = 64;
            std::vector<uint16_t> qc, kc, vc;
            rand_fp16((int64_t)q_len * nh * hd, -1.0f, 1.0f, g, qc);
            rand_fp16((int64_t)kv_len * nkv * hd, -1.0f, 1.0f, g, kc);
            rand_fp16((int64_t)kv_len * nkv * hd, -1.0f, 1.0f, g, vc);
            auto q = make_dev_fp16(qc, q_len, nh * hd);
            auto k = make_dev_fp16(kc, kv_len, nkv * hd);
            auto v = make_dev_fp16(vc, kv_len, nkv * hd);
            auto launch = [&](TensorPtr& o, cudaStream_t st, bool candidate) {
                const __half* qh = reinterpret_cast<const __half*>(q->data());
                const __half* kh = reinterpret_cast<const __half*>(k->data());
                const __half* vh = reinterpret_cast<const __half*>(v->data());
                __half* oh = reinterpret_cast<__half*>(o->data());
                if (candidate)
                    forge::cuda::launch_flash_attention_gqa_prefill_mma_cfg(qh, kh, vh, oh, q_len,
                                                                            kv_len, nh, nkv, hd,
                                                                            true, c.br, c.bc,
                                                                            c.warps, st);
                else  // frozen 6-A-7.2 baseline arm -- explicit vectorized entry
                    forge::cuda::launch_flash_attention_gqa_prefill_fp16_vectorized(
                        qh, kh, vh, oh, q_len, kv_len, nh, nkv, hd, nullptr, true, st);
            };
            auto oA = std::make_shared<Tensor>(DataType::FP16,
                                               std::vector<int64_t>{q_len, nh * hd},
                                               DeviceType::CUDA);
            launch(oA, 0, false);
            auto oC = std::make_shared<Tensor>(DataType::FP16,
                                               std::vector<int64_t>{q_len, nh * hd},
                                               DeviceType::CUDA);
            launch(oC, 0, true);
            cudaStream_t cs = nullptr;
            CHECK(cudaStreamCreate(&cs) == cudaSuccess);
            auto oC2 = std::make_shared<Tensor>(DataType::FP16,
                                                std::vector<int64_t>{q_len, nh * hd},
                                                DeviceType::CUDA);
            launch(oC2, cs, true);
            cudaStreamSynchronize(cs);
            cudaStreamDestroy(cs);
            if (!(to_host_u16(oC2) == to_host_u16(oC))) ab_ok = 0;

            std::vector<float> qf(qc.size()), kf(kc.size()), vf(vc.size());
            for (size_t i = 0; i < qc.size(); ++i) qf[i] = f16_to_f32(qc[i]);
            for (size_t i = 0; i < kc.size(); ++i) kf[i] = f16_to_f32(kc[i]);
            for (size_t i = 0; i < vc.size(); ++i) vf[i] = f16_to_f32(vc[i]);
            std::vector<float> ref((size_t)q_len * nh * hd);
            ref_gqa_prefill_fp32(qf.data(), kf.data(), vf.data(), ref.data(), q_len, kv_len, nh,
                                 nkv, hd, true);
            auto got = to_host_f32(oC);
            auto gotA = to_host_f32(oA);
            for (size_t i = 0; i < ref.size(); ++i) {
                worst_ref = std::max(worst_ref, std::fabs(got[i] - ref[i]));
                worst_base = std::max(worst_base, std::fabs(got[i] - gotA[i]));
            }
        }
        std::printf("  mma-cfg Br=%d Bc=%d w=%d : worst vs ref %.3e, worst vs baseline %.3e, "
                    "streamA==B: %d\n",
                    c.br, c.bc, c.warps, worst_ref, worst_base, ab_ok);
        CHECK(ab_ok == 1);
        CHECK(worst_base < 5e-3f);
        CHECK(worst_ref < 1e-2f);
    }
}

// Phase 2 round 5 (production wiring): the production prefill entry routes
// hd=64 causal no-mask through the promoted wmma kernel (must be BITWISE
// identical to the explicit cfg entry with the round-4 winner (64,16,4)),
// and every other scope (hd=128, non-causal, masked) to the frozen 6-A-7.2
// vectorized kernel (bitwise identical to the explicit vectorized entry).
void check_prefill_production_routing(std::mt19937& g) {
    struct Shape {
        int q, kv;
    };
    const Shape shapes[] = {{16, 64}, {48, 80}, {64, 128}};
    for (const auto& s : shapes) {
        const int q_len = s.q, kv_len = s.kv, nh = 8, nkv = 2;
        std::vector<uint16_t> qc, kc, vc;
        rand_fp16((int64_t)q_len * nh * 64, -1.0f, 1.0f, g, qc);
        rand_fp16((int64_t)kv_len * nkv * 64, -1.0f, 1.0f, g, kc);
        rand_fp16((int64_t)kv_len * nkv * 64, -1.0f, 1.0f, g, vc);
        auto q = make_dev_fp16(qc, q_len, nh * 64);
        auto k = make_dev_fp16(kc, kv_len, nkv * 64);
        auto v = make_dev_fp16(vc, kv_len, nkv * 64);
        auto o_prod = std::make_shared<Tensor>(DataType::FP16,
                                               std::vector<int64_t>{q_len, nh * 64},
                                               DeviceType::CUDA);
        auto o_ref = std::make_shared<Tensor>(DataType::FP16,
                                              std::vector<int64_t>{q_len, nh * 64},
                                              DeviceType::CUDA);
        const __half* qh = reinterpret_cast<const __half*>(q->data());
        const __half* kh = reinterpret_cast<const __half*>(k->data());
        const __half* vh = reinterpret_cast<const __half*>(v->data());
        // production entry (hd=64, causal, no mask) -> wmma (64,16,4)
        forge::cuda::launch_flash_attention_gqa_prefill_fp16(
            qh, kh, vh, reinterpret_cast<__half*>(o_prod->data()), q_len, kv_len, nh, nkv, 64,
            nullptr, true, 0);
        // explicit cfg entry, round-4 winner
        forge::cuda::launch_flash_attention_gqa_prefill_mma_cfg(
            qh, kh, vh, reinterpret_cast<__half*>(o_ref->data()), q_len, kv_len, nh, nkv, 64, true,
            64, 16, 4, 0);
        cudaDeviceSynchronize();
        CHECK(to_host_u16(o_prod) == to_host_u16(o_ref));
    }
    std::printf("  production routing: hd=64 causal no-mask == wmma(64,16,4) bitwise\n");

    // hd=128 must fall back to the vectorized kernel (bitwise).
    {
        const int q_len = 16, kv_len = 64, nh = 8, nkv = 2, hd = 128;
        std::vector<uint16_t> qc, kc, vc;
        rand_fp16((int64_t)q_len * nh * hd, -1.0f, 1.0f, g, qc);
        rand_fp16((int64_t)kv_len * nkv * hd, -1.0f, 1.0f, g, kc);
        rand_fp16((int64_t)kv_len * nkv * hd, -1.0f, 1.0f, g, vc);
        auto q = make_dev_fp16(qc, q_len, nh * hd);
        auto k = make_dev_fp16(kc, kv_len, nkv * hd);
        auto v = make_dev_fp16(vc, kv_len, nkv * hd);
        auto o_prod = std::make_shared<Tensor>(DataType::FP16,
                                               std::vector<int64_t>{q_len, nh * hd},
                                               DeviceType::CUDA);
        auto o_ref = std::make_shared<Tensor>(DataType::FP16,
                                              std::vector<int64_t>{q_len, nh * hd},
                                              DeviceType::CUDA);
        const __half* qh = reinterpret_cast<const __half*>(q->data());
        const __half* kh = reinterpret_cast<const __half*>(k->data());
        const __half* vh = reinterpret_cast<const __half*>(v->data());
        forge::cuda::launch_flash_attention_gqa_prefill_fp16(
            qh, kh, vh, reinterpret_cast<__half*>(o_prod->data()), q_len, kv_len, nh, nkv, hd,
            nullptr, true, 0);
        forge::cuda::launch_flash_attention_gqa_prefill_fp16_vectorized(
            qh, kh, vh, reinterpret_cast<__half*>(o_ref->data()), q_len, kv_len, nh, nkv, hd,
            nullptr, true, 0);
        cudaDeviceSynchronize();
        CHECK(to_host_u16(o_prod) == to_host_u16(o_ref));
    }
    std::printf("  production routing: hd=128 falls back to vectorized, bitwise\n");
}

// Graph integration for prefill (incl. capture/replay, bitwise).
void check_graph_node_prefill(int q_len, int kv_len, int nh, int nkv, int hd, std::mt19937& g) {
    std::vector<uint16_t> qc, kc, vc;
    rand_fp16((int64_t)q_len * nh * hd, -1.0f, 1.0f, g, qc);
    rand_fp16((int64_t)kv_len * nkv * hd, -1.0f, 1.0f, g, kc);
    rand_fp16((int64_t)kv_len * nkv * hd, -1.0f, 1.0f, g, vc);
    auto graph = std::make_unique<forge::ComputeGraph>();
    int qi = graph->add_input(make_dev_fp16(qc, q_len, nh * hd));
    int ki = graph->add_input(make_dev_fp16(kc, kv_len, nkv * hd));
    int vi = graph->add_input(make_dev_fp16(vc, kv_len, nkv * hd));
    int32_t params[forge::OP_PARAMS_MAX_SIZE / sizeof(int32_t)] = {0};
    params[0] = nh;
    params[1] = nkv;
    params[2] = hd;
    params[3] = 1;
    params[4] = 1;
    graph->add_node("attn", forge::OpType::FLASH_ATTN_GQA, {qi, ki, vi}, params,
                    DeviceType::CUDA);
    CHECK(forge::CudaGraphRunner::is_compatible(*graph));

    auto outA = graph->execute();
    CHECK(outA != nullptr);
    CHECK(outA->dtype() == DataType::FP16);
    cudaDeviceSynchronize();
    auto codesA = to_host_u16(graph->last_output());

    cudaStream_t cs = nullptr;
    CHECK(cudaStreamCreate(&cs) == cudaSuccess);
    forge::CudaGraphRunner runner;
    runner.set_enabled(true);
    CHECK(runner.try_replay_or_capture(*graph, [&]() { graph->execute(cs); }, cs));
    cudaStreamSynchronize(cs);
    CHECK(to_host_u16(graph->last_output()) == codesA);
    CHECK(runner.try_replay_or_capture(*graph, [&]() { graph->execute(cs); }, cs));
    cudaStreamSynchronize(cs);
    CHECK(to_host_u16(graph->last_output()) == codesA);
    cudaStreamDestroy(cs);
    std::printf("  graph prefill q=%d kv=%d nh=%d nkv=%d hd=%d : eager/capture/replay OK (FP16)\n",
                q_len, kv_len, nh, nkv, hd);
}

// Loud failure: HD 256 FP16 decode must throw, never reinterpret as float.
void check_unsupported_hd() {
    auto q = std::make_shared<Tensor>(DataType::FP16, std::vector<int64_t>{1, 8 * 256},
                                      DeviceType::CUDA);
    auto k = std::make_shared<Tensor>(DataType::FP16, std::vector<int64_t>{4, 2 * 256},
                                      DeviceType::CUDA);
    auto v = std::make_shared<Tensor>(DataType::FP16, std::vector<int64_t>{4, 2 * 256},
                                      DeviceType::CUDA);
    auto graph = std::make_unique<forge::ComputeGraph>();
    int qi = graph->add_input(q);
    int ki = graph->add_input(k);
    int vi = graph->add_input(v);
    int32_t params[forge::OP_PARAMS_MAX_SIZE / sizeof(int32_t)] = {0};
    params[0] = 8;
    params[1] = 2;
    params[2] = 256;
    params[3] = 1;
    params[4] = 1;
    graph->add_node("attn", forge::OpType::FLASH_ATTN_GQA, {qi, ki, vi}, params,
                    DeviceType::CUDA);
    bool threw = false;
    try {
        graph->execute();
        cudaDeviceSynchronize();
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
    std::printf("  unsupported hd=256: loud failure OK\n");
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("SKIP: no CUDA device available (test_attention_fp16)\n");
        return 0;
    }
    forge::register_builtin_op_kernels();

    std::printf("[6-A-7] Native FP16 attention decode (FP16 QKV, FP32 math, FP16 out)\n");
    std::mt19937 rng(0xA7716);

    check_selector();
    for (int hd : {16, 32, 64, 128}) {
        check_decode(128, 8, 2, hd, rng);
    }
    check_decode(512, 8, 2, 64, rng);
    check_fp32_consistency(128, 8, 2, 64, rng);
    check_graph_node(128, 8, 2, 64, rng);
    check_unsupported_hd();

    // Phase 2 round 2: __half2 decode candidate gates (bitwise==baseline +
    // FP32-ref parity) across template coverage and both workload regimes.
    std::printf("[Phase 2 round 2] __half2 decode candidate vs 6-A-7 baseline\n");
    for (int hd : {16, 32, 64, 128}) {
        check_decode_h2_candidate(512, 8, 2, hd, rng);
    }
    check_decode_h2_candidate(128, 8, 2, 64, rng);
    check_decode_h2_candidate(2048, 8, 2, 64, rng);

    // Phase 2 round 3: wmma prefill candidate gates (incl. q/kv tails, kv>q).
    std::printf("[Phase 2 round 3] TensorCore prefill candidate vs 6-A-7 baseline\n");
    check_prefill_mma_candidate(16, 64, 8, 2, rng);
    check_prefill_mma_candidate(64, 64, 8, 2, rng);
    check_prefill_mma_candidate(48, 80, 8, 2, rng);
    check_prefill_mma_candidate(64, 128, 8, 2, rng);

    // Phase 2 round 4: tile-config scan -- round-3 gates for every config.
    std::printf("[Phase 2 round 4] wmma prefill config scan (Br/Bc/warps, same gates)\n");
    check_prefill_mma_cfg_scan(rng);

    // Phase 2 round 5: production wiring -- routing + graph through the new
    // production entry.
    std::printf("[Phase 2 round 5] wmma prefill promoted into production entry\n");
    check_prefill_production_routing(rng);

    check_prefill(16, 64, 8, 2, 64, rng);
    check_prefill(16, 64, 8, 2, 128, rng);
    check_graph_node_prefill(16, 64, 8, 2, 64, rng);

    if (failures == 0) {
        std::printf("PASS attn-fp16 (Phase 6-A-7 decode+prefill)\n");
        return 0;
    }
    std::printf("FAILURES: %d\n", failures);
    return 1;
}
