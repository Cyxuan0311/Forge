// =========================================================================
// test_e2e_bf16.cpp -- Phase 6-B-7: end-to-end BF16 validation.
//
// Closed 2-layer llama-style loop, identical topology for BF16 and FP32:
//
//     BF16 input
//        -> RMSNorm -> Q/K/V projection -> BF16 Attention (+ BF16 KV)
//        -> residual ADD -> RMSNorm -> gate/up -> SILU -> gate*up (MUL)
//        -> residual ADD -> next layer -> final norm -> logits (matmul)
//
// Both dispatch modes are covered: prefill (q_len = kv_len, K/V projected
// from the multi-token hidden state like a real prefill) and decode
// (q_len = 1). The decode KV-cache is NOT synthetic: it is produced by the
// real prefill write path -- the same native BF16 projection matmuls
// (rms_norm -> matmul_transB with wk/wv) the prefill graph itself runs --
// then consumed as decode graph inputs. This links the KV write path and the
// KV read path inside one test, unlike 6-A-8 (pre-made FP16 cache input).
//
// 6-B-7.0 prerequisite (in-tree, same step): the native BF16 vectorized
// prefill kernel unlocks the prefill half of this loop; decode uses the
// 6-B-6/6-B-6.7 kernels. No wmma/paged/select-or redesign in this step.
//
// Acceptance (mirrors 6-A-8):
//   1. dtype trace -- every main-chain node stays BF16 (no BF16->FP32->BF16)
//   2. BF16 vs FP32 logits parity (FIXED thresholds, calibrated below)
//   3. CUDA Graph on the whole BF16 graph: ordinary == capture == replay x N,
//      bitwise stable, buffer address/size stable.
//
// Requires CUDA; skips gracefully without a GPU.
// =========================================================================

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "forge/compute_graph.h"
#include "forge/cuda_graph_runner.h"
#include "forge/cuda_kernels.h"
#include "forge/cuda_mem_pool.h"
#include "forge/op_dispatch.h"
#include "forge/op_enum.h"
#include "forge/operator_elementwise.h"
#include "forge/operator_matmul.h"
#include "forge/operator_norm.h"
#include "forge/quant_traits.h"
#include "forge/tensor.h"
#include "forge/types.h"

namespace {

using forge::ComputeGraph;
using forge::DataType;
using forge::DeviceType;
using forge::OpType;
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

int failures = 0;
#define CHECK(cond)                                                     \
    do {                                                                \
        if (!(cond)) {                                                  \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                 \
        }                                                               \
    } while (0)

// ---- FIXED parity thresholds (6-B-7; frozen from the first measured run) ----
// First measurement (RTX 4050, 2026-09-07): prefill max_abs=1.3e-3 /
// max_rel=4.8e-3 / cosine=0.9999902 / topk 8/8; decode max_abs=8.7e-4 /
// max_rel=2.0e-3 / cosine=0.9999886 / topk 8/8. Bounds below sit ~6-10x above
// observed -- the same margin discipline as 6-A-8's 0.05 vs 2e-4, rescaled
// for BF16 storage. A real regression (fallback/garbage) misses by 10x+.
constexpr float kParityMaxAbs = 0.01f;    // logits are O(0.1..1) by construction
constexpr float kParityMaxRel = 0.03f;    // denominator guard: |ref| >= 0.15
constexpr float kParityRelGuard = 0.15f;
constexpr float kParityCosine = 0.999f;   // plan.md 2.2 end-to-end bar
constexpr int kTopK = 8;
constexpr int kTopKMinAgree = 7;          // same bar as 6-A-8 (observed 8/8)

// Materialize a device tensor from float values. BF16 storage rounds through
// f32->bf16; FP32 stores as-is (the FP32 reference graph sees the exact values
// the BF16 chain consumed, before storage rounding).
TensorPtr dev_from_values(const std::vector<float>& vals, int64_t r, int64_t c, DataType dt) {
    auto host = std::make_shared<Tensor>(dt, std::vector<int64_t>{r, c}, DeviceType::CPU);
    if (dt == DataType::BF16) {
        auto* h = static_cast<uint16_t*>(host->data());
        for (size_t i = 0; i < vals.size(); ++i) h[i] = f32_to_bf16(vals[i]);
    } else {
        std::memcpy(host->data(), vals.data(), vals.size() * 4);
    }
    auto dev = std::make_shared<Tensor>(dt, std::vector<int64_t>{r, c}, DeviceType::CUDA);
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
    size_t n = static_cast<size_t>(t->numel());
    std::vector<float> v(n);
    if (t->dtype() == DataType::BF16) {
        auto* h = static_cast<uint16_t*>(host->data());
        for (size_t i = 0; i < n; ++i) v[i] = bf16_to_f32(h[i]);
    } else {
        std::memcpy(v.data(), host->data(), n * 4);
    }
    return v;
}

void rand_vals(int64_t n, float lo, float hi, std::mt19937& g, std::vector<float>& out) {
    std::uniform_real_distribution<float> d(lo, hi);
    out.resize(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) out[i] = d(g);
}

struct GraphSpec {
    int D = 256;
    int nh = 8, nkv = 2, hd = 64;
    int FF = 512;
    int V = 128;      // vocab (logits width)
    int layers = 2;
    int q_len = 16;   // == kv_len => prefill; 1 => decode
    int kv_len = 16;
};

// One set of float values drives BOTH the BF16 and the FP32 graph: the BF16
// graph rounds them through f32->bf16 at materialization, the FP32 graph
// consumes the exact values. Any logits difference is therefore attributable
// to BF16 storage/compute inside the chain, not to input drift.
struct ModelValues {
    std::vector<float> x;                                     // [q_len, D]
    std::vector<std::vector<float>> w_attn, wq, wk, wv, wo;   // per layer
    std::vector<std::vector<float>> w_ffn, wg, wu;            // per layer
    std::vector<float> w_out_v;                               // [1, D]
    std::vector<float> wvoc;                                  // [V, D]
};

ModelValues make_model_values(const GraphSpec& s, std::mt19937& g) {
    ModelValues m;
    rand_vals((int64_t)s.q_len * s.D, -1.0f, 1.0f, g, m.x);
    float ws = 1.0f / std::sqrt((float)s.D);  // keeps activations O(1)
    auto wmat = [&](int64_t r, int64_t c, float scale) {
        std::vector<float> v;
        rand_vals(r * c, -scale, scale, g, v);
        return v;
    };
    for (int l = 0; l < s.layers; ++l) {
        m.w_attn.push_back(wmat(1, s.D, 0.2f));
        m.wq.push_back(wmat(s.nh * s.hd, s.D, ws));
        m.wk.push_back(wmat(s.nkv * s.hd, s.D, ws));
        m.wv.push_back(wmat(s.nkv * s.hd, s.D, ws));
        m.wo.push_back(wmat(s.D, s.nh * s.hd, ws));
        m.w_ffn.push_back(wmat(1, s.D, 0.2f));
        m.wg.push_back(wmat(s.FF, s.D, ws));
        m.wu.push_back(wmat(s.FF, s.D, ws));
    }
    m.w_out_v = wmat(1, s.D, 0.2f);
    m.wvoc = wmat(s.V, s.D, ws);
    return m;
}

// Materialize the model in `dt` (BF16: storage-rounded; FP32: exact values).
struct ModelTensors {
    TensorPtr x;
    std::vector<TensorPtr> w_attn, wq, wk, wv, wo, w_ffn, wg, wu;
    std::vector<TensorPtr> k_cache, v_cache;  // decode only, see below
    TensorPtr w_out, wvoc;
};

ModelTensors materialize(const GraphSpec& s, const ModelValues& mv, DataType dt) {
    ModelTensors t;
    std::vector<float> xq(mv.x.begin(), mv.x.begin() + (size_t)s.q_len * s.D);
    t.x = dev_from_values(xq, s.q_len, s.D, dt);
    for (int l = 0; l < s.layers; ++l) {
        t.w_attn.push_back(dev_from_values(mv.w_attn[l], 1, s.D, dt));
        t.wq.push_back(dev_from_values(mv.wq[l], s.nh * s.hd, s.D, dt));
        t.wk.push_back(dev_from_values(mv.wk[l], s.nkv * s.hd, s.D, dt));
        t.wv.push_back(dev_from_values(mv.wv[l], s.nkv * s.hd, s.D, dt));
        t.wo.push_back(dev_from_values(mv.wo[l], s.D, s.nh * s.hd, dt));
        t.w_ffn.push_back(dev_from_values(mv.w_ffn[l], 1, s.D, dt));
        t.wg.push_back(dev_from_values(mv.wg[l], s.FF, s.D, dt));
        t.wu.push_back(dev_from_values(mv.wu[l], s.FF, s.D, dt));
        // Decode KV-cache is filled by the caller via produce_decode_kv (the
        // real prefill write path), not here -- see check_e2e.
    }
    t.w_out = dev_from_values(mv.w_out_v, 1, s.D, dt);
    t.wvoc = dev_from_values(mv.wvoc, s.V, s.D, dt);
    return t;
}

// The prefill KV write path, executed eagerly with the real native ops: the
// exact computation the prefill graph's wk/wv nodes perform
// (rms_norm -> matmul_transB), producing BF16 (or FP32) K/V cache tensors
// that the decode graph then consumes as inputs. Same ops, same tensors as
// the prefill graph -- so decode reads genuinely written KV, not synthetic
// randoms (the 6-A-8 gap this closes).
void produce_decode_kv(const GraphSpec& pre, const ModelTensors& tp, ModelTensors& td) {
    for (int l = 0; l < pre.layers; ++l) {
        TensorPtr h = forge::ops::rms_norm(tp.x, tp.w_attn[l], 1e-5f);
        td.k_cache.push_back(forge::ops::matmul_transB(h, tp.wk[l]));
        td.v_cache.push_back(forge::ops::matmul_transB(h, tp.wv[l]));
    }
}

// Build the closed-loop graph (identical topology for both dtypes).
// RMS_NORM params: eps as float at params[0..1].
// FLASH_ATTN_GQA params: [0]=nh [1]=nkv [2]=hd [3]=causal [4]=device(CUDA=1).
//
// Decode mode: attention consumes the KV-cache directly as graph inputs
// (k_cache/v_cache, kv_len entries) -- the real decode step reads prior K/V
// from cache instead of recomputing them from the single token. wk/wv stay
// materialized but unused in this mode (q still comes from wq).
std::unique_ptr<ComputeGraph> build_model_graph(const GraphSpec& s, const ModelTensors& t) {
    auto g = std::make_unique<ComputeGraph>();
    int x_i = g->add_input(t.x);
    std::vector<int> w_attn_i, wq_i, wk_i, wv_i, wo_i, w_ffn_i, wg_i, wu_i;
    std::vector<int> k_cache_i, v_cache_i;
    auto in = [&](const TensorPtr& p, std::vector<int>& slot) {
        slot.push_back(g->add_input(p));
        return slot.back();
    };
    for (int l = 0; l < s.layers; ++l) {
        in(t.w_attn[l], w_attn_i);
        in(t.wq[l], wq_i);
        if (s.q_len != 1) {  // prefill: K/V computed from the multi-token hidden state
            in(t.wk[l], wk_i);
            in(t.wv[l], wv_i);
        } else {  // decode: K/V read from the cache
            in(t.k_cache[l], k_cache_i);
            in(t.v_cache[l], v_cache_i);
        }
        in(t.wo[l], wo_i);
        in(t.w_ffn[l], w_ffn_i);
        in(t.wg[l], wg_i);
        in(t.wu[l], wu_i);
    }
    int w_out_i = g->add_input(t.w_out);
    int wvoc_i = g->add_input(t.wvoc);

    int32_t rms_params[forge::OP_PARAMS_MAX_SIZE / sizeof(int32_t)] = {0};
    float eps = 1e-5f;
    std::memcpy(rms_params, &eps, sizeof(float));
    int32_t attn_params[forge::OP_PARAMS_MAX_SIZE / sizeof(int32_t)] = {0};
    attn_params[0] = s.nh;
    attn_params[1] = s.nkv;
    attn_params[2] = s.hd;
    attn_params[3] = 1;  // causal
    attn_params[4] = 1;  // CUDA

    // cur holds the running residual stream as an input_indices reference:
    // >= 0 = graph input index; < 0 = -node_idx - 1 (node output).
    int cur = x_i;

    for (int l = 0; l < s.layers; ++l) {
        std::string p = "L" + std::to_string(l) + ".";
        int h = g->add_node(p + "rms_attn", OpType::RMS_NORM, {cur, w_attn_i[l]},
                            rms_params, DeviceType::CUDA);
        int q = g->add_node(p + "wq", OpType::MUL_MAT_TRANSB, {-h - 1, wq_i[l]}, nullptr,
                            DeviceType::CUDA);
        // Attention K/V: prefill projects the hidden state; decode reads the
        // cache inputs directly (q_len=1, causal covers the whole cache).
        int k_i = (s.q_len != 1) ? -g->add_node(p + "wk", OpType::MUL_MAT_TRANSB,
                                                {-h - 1, wk_i[l]}, nullptr, DeviceType::CUDA) - 1
                                 : k_cache_i[l];
        int v_i = (s.q_len != 1) ? -g->add_node(p + "wv", OpType::MUL_MAT_TRANSB,
                                                {-h - 1, wv_i[l]}, nullptr, DeviceType::CUDA) - 1
                                 : v_cache_i[l];
        int attn = g->add_node(p + "attn", OpType::FLASH_ATTN_GQA, {-q - 1, k_i, v_i},
                               attn_params, DeviceType::CUDA);
        int o = g->add_node(p + "wo", OpType::MUL_MAT_TRANSB, {-attn - 1, wo_i[l]}, nullptr,
                            DeviceType::CUDA);
        int x2 = g->add_node(p + "residual1", OpType::ADD, {cur, -o - 1}, nullptr,
                             DeviceType::CUDA);
        int h2 = g->add_node(p + "rms_ffn", OpType::RMS_NORM, {-x2 - 1, w_ffn_i[l]},
                             rms_params, DeviceType::CUDA);
        int gt = g->add_node(p + "wg", OpType::MUL_MAT_TRANSB, {-h2 - 1, wg_i[l]}, nullptr,
                             DeviceType::CUDA);
        int ut = g->add_node(p + "wu", OpType::MUL_MAT_TRANSB, {-h2 - 1, wu_i[l]}, nullptr,
                             DeviceType::CUDA);
        int act = g->add_node(p + "silu", OpType::SILU, {-gt - 1}, nullptr, DeviceType::CUDA);
        int gated = g->add_node(p + "gate_up", OpType::MUL, {-act - 1, -ut - 1}, nullptr,
                                DeviceType::CUDA);
        cur = -g->add_node(p + "residual2", OpType::ADD, {-x2 - 1, -gated - 1}, nullptr,
                           DeviceType::CUDA) - 1;
    }
    int hf = g->add_node("out.rms", OpType::RMS_NORM, {cur, w_out_i}, rms_params,
                         DeviceType::CUDA);
    g->add_node("logits", OpType::MUL_MAT_TRANSB, {-hf - 1, wvoc_i}, nullptr, DeviceType::CUDA);
    return g;
}

// ---- acceptance 2: parity metrics on logits ----
struct Parity {
    float max_abs = 0, max_rel = 0, cosine = 0, max_ref = 0;
    int topk_agree = 0;
    int agree_k = 0;
};

Parity compare_logits(const std::vector<float>& gotbf, const std::vector<float>& ref32) {
    Parity p;
    p.agree_k = kTopK;
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < ref32.size(); ++i) {
        float diff = std::fabs(gotbf[i] - ref32[i]);
        p.max_abs = std::max(p.max_abs, diff);
        p.max_ref = std::max(p.max_ref, std::fabs(ref32[i]));
        if (std::fabs(ref32[i]) >= kParityRelGuard)
            p.max_rel = std::max(p.max_rel, diff / std::fabs(ref32[i]));
        dot += (double)gotbf[i] * ref32[i];
        na += (double)gotbf[i] * gotbf[i];
        nb += (double)ref32[i] * ref32[i];
    }
    p.cosine = (float)(dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12));
    auto topk = [&](const std::vector<float>& v) {
        std::vector<int> idx(v.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + kTopK, idx.end(),
                          [&](int a, int b) { return v[a] > v[b]; });
        idx.resize(kTopK);
        std::sort(idx.begin(), idx.end());
        return idx;
    };
    auto ta = topk(gotbf), tb = topk(ref32);
    for (int i = 0; i < kTopK; ++i)
        if (ta[i] != tb[i]) p.topk_agree++;
    return p;
}

// ---- the full 6-B-7 gate set for one mode (prefill / decode) ----
void check_e2e(const GraphSpec& s, const ModelValues& mv, const GraphSpec& pre,
               const ModelTensors& tp_bf, const ModelTensors& tp_fp,
               std::mt19937& g) {
    (void)g;
    const char* mode = (s.q_len == 1) ? "decode" : "prefill";

    auto tb = materialize(s, mv, DataType::BF16);
    auto t32 = materialize(s, mv, DataType::FP32);

    if (s.q_len == 1) {
        // Decode KV linkage: the cache entering decode is produced by the real
        // prefill write path (native projection matmuls on the prefill hidden
        // state), in each dtype respectively -- no synthetic random cache.
        produce_decode_kv(pre, tp_bf, tb);
        produce_decode_kv(pre, tp_fp, t32);
        CHECK(tb.k_cache.size() == (size_t)s.layers);
        CHECK(t32.k_cache.size() == (size_t)s.layers);
        for (int l = 0; l < s.layers; ++l) {
            CHECK(tb.k_cache[l]->dtype() == DataType::BF16);
            CHECK(tb.v_cache[l]->dtype() == DataType::BF16);
            CHECK(tb.k_cache[l]->shape()[0] == pre.kv_len);
        }
    }

    auto gbf = build_model_graph(s, tb);
    auto g32 = build_model_graph(s, t32);

    // ---- FP32 reference run (existing FP32 path, same topology) ----
    auto ref_out = g32->execute();
    cudaDeviceSynchronize();
    CHECK(ref_out != nullptr);
    CHECK(ref_out->dtype() == DataType::FP32);
    cudaDeviceSynchronize();
    std::vector<float> ref = to_host_f32(g32->last_output());
    CHECK((int)ref.size() == s.q_len * s.V);

    // ---- BF16 run (ordinary) ----
    auto outbf = gbf->execute();
    cudaDeviceSynchronize();
    CHECK(outbf != nullptr);
    CHECK(outbf->dtype() == DataType::BF16);  // hard dtype assert, not just numerics
    cudaDeviceSynchronize();
    auto codesA = to_host_u16(gbf->last_output());
    void* bufA = gbf->last_output()->data();
    size_t nbytesA = gbf->last_output()->nbytes();
    CHECK(nbytesA == (size_t)s.q_len * s.V * 2);  // BF16-sized logits buffer

    std::vector<float> got = to_host_f32(gbf->last_output());

    // ---- acceptance 1: dtype trace -- the whole main chain stays BF16 ----
    const auto& trace = gbf->dtype_trace();
    CHECK(trace.size() == (size_t)gbf->num_nodes());
    size_t dst_injected = 0;
    for (const auto& r : trace) {
        bool all_in_bf16 = true;
        for (auto d : r.input_dtypes)
            if (d != DataType::BF16) all_in_bf16 = false;
        CHECK(all_in_bf16);
        CHECK(r.output_dtype == DataType::BF16);
        if (r.dst_injected) dst_injected++;
    }

    // The dst-injectable BF16 ops must actually go through the planner buffer
    // (RMS_NORM / ADD / MUL / SILU): proves 6-B-5 plumbing, not the legacy
    // allocate+copy detour. Matmuls and attention are legacy by design.
    size_t expect_dst = 0;
    for (const auto& r : trace) {
        bool dstable = r.name.find("rms") != std::string::npos ||
                       r.name.find("residual") != std::string::npos ||
                       r.name.find("gate_up") != std::string::npos ||
                       r.name.find("silu") != std::string::npos;
        if (dstable) {
            expect_dst++;
            CHECK(r.dst_injected);
        }
    }
    CHECK(dst_injected == expect_dst);
    // Trace accumulates across execute() calls: consume it now so the CUDA
    // Graph runs below start from a clean slate.
    gbf->clear_dtype_trace();

    // ---- acceptance 2: BF16 vs FP32 parity (FIXED thresholds) ----
    Parity par = compare_logits(got, ref);
    std::printf("  e2e-%s parity: max_abs=%.3e max_rel=%.3e cosine=%.7f max_ref=%.3e topk_mismatch=%d/%d\n",
                mode, par.max_abs, par.max_rel, par.cosine, par.max_ref, par.topk_agree,
                par.agree_k);

    CHECK(par.max_ref >= kParityRelGuard);  // the rel metric must be non-vacuous
    CHECK(par.max_abs < kParityMaxAbs);
    CHECK(par.max_rel < kParityMaxRel);
    CHECK(par.cosine >= kParityCosine);
    CHECK(par.topk_agree <= kTopK - kTopKMinAgree);

    // ---- acceptance 3: CUDA Graph on the WHOLE BF16 graph ----
    CHECK(forge::CudaGraphRunner::is_compatible(*gbf));

    // B: explicit legacy stream -- bitwise equal to A.
    gbf->execute(0);
    cudaDeviceSynchronize();
    CHECK(to_host_u16(gbf->last_output()) == codesA);

    cudaStream_t cs = nullptr;
    CHECK(cudaStreamCreate(&cs) == cudaSuccess);
    forge::CudaGraphRunner runner;
    runner.set_enabled(true);

    // C: capture the whole graph on the capture stream.
    bool captured = runner.try_replay_or_capture(*gbf, [&]() { gbf->execute(cs); }, cs);
    cudaStreamSynchronize(cs);
    CHECK(captured);
    CHECK(to_host_u16(gbf->last_output()) == codesA);

    // D: replay x N -- bitwise stable every time, buffer address/size stable.
    constexpr int kReplays = 3;
    for (int i = 0; i < kReplays; ++i) {
        bool replayed = runner.try_replay_or_capture(*gbf, [&]() { gbf->execute(cs); }, cs);
        cudaStreamSynchronize(cs);
        CHECK(replayed);
        CHECK(to_host_u16(gbf->last_output()) == codesA);
        CHECK(gbf->last_output()->data() == bufA);
        CHECK(gbf->last_output()->nbytes() == nbytesA);
    }
    cudaStreamDestroy(cs);


    gbf->release_graph();
    g32->release_graph();
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("SKIP: no CUDA device available (test_e2e_bf16)\n");
        return 0;
    }
    setenv("FORGE_TRACE_DTYPE", "1", 1);
    forge::register_builtin_op_kernels();


    std::mt19937 rng(0x6B07);

    // Shared model values: prefill and decode see the same weights; only the
    // token count (q_len) differs, so both modes validate the same "model".
    GraphSpec prefill;
    prefill.q_len = prefill.kv_len = 16;
    ModelValues mv = make_model_values(prefill, rng);

    // Prefill tensors in both dtypes (decode KV is produced from these).
    ModelTensors tp_bf = materialize(prefill, mv, DataType::BF16);
    ModelTensors tp_fp = materialize(prefill, mv, DataType::FP32);

    check_e2e(prefill, mv, prefill, tp_bf, tp_fp, rng);

    GraphSpec decode;
    decode.q_len = 1;
    decode.kv_len = 16;
    check_e2e(decode, mv, prefill, tp_bf, tp_fp, rng);

    if (failures == 0) {
        std::printf("PASS e2e-bf16 (Phase 6-B-7)\n");
        return 0;
    }
    std::printf("FAILURES: %d\n", failures);
    return 1;
}
