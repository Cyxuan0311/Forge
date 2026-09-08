// =========================================================================
// test_e2e_fp16.cpp -- Phase 6-A-8: end-to-end native FP16 validation
//
// Minimal closed loop (NOT a full model): a 2-layer llama-style graph with
// all-FP16 storage, executed through the real ComputeGraph machinery
// (MemoryPlanner + dst injection + OpDispatch + native FP16 kernels):
//
//   x --RMS_NORM--> h --MUL_MAT_TRANSB(Wq/Wk/Wv)--> q,k,v
//     --FLASH_ATTN_GQA--> attn --MUL_MAT_TRANSB(Wo)--> o
//     --ADD(x, .)--> x2                                    (residual)
//   x2 --RMS_NORM--> h2 --MUL_MAT_TRANSB(Wg/Wu)--> g,u
//     --SILU--> silu_g --MUL(silu_g, u)--> gated
//     --ADD(x2, .)--> x3                                   (residual)
//   ... next layer ... --RMS_NORM(w_out)--> MUL_MAT_TRANSB(Wvoc) --> logits
//
// Acceptance (the three 6-A-8 gates):
//   1. dtype trace: every node on the main chain keeps FP16 in/out -- no
//      implicit FP16 -> FP32 -> FP16 materialization (projection, residual,
//      attention output, next-layer input included).
//   2. FP16 vs FP32 end-to-end parity on logits: max_abs / max_rel /
//      cosine similarity / top-k agreement against the SAME topology run in
//      FP32 from exactly-upcast inputs. Thresholds are FIXED constants.
//   3. CUDA Graph real chain: ordinary -> capture -> replay x N on the whole
//      FP16 graph, output bitwise stable, buffer address/size stable.
//
// Both dispatch modes are covered: prefill (q_len = kv_len) and decode
// (q_len = 1) -- the selector picks CUDA_F16_PREFILL / CUDA_F16_DECODE.
//
// 8.0 section: chain validation of the two new 6-A-8.0 elementwise kernels
// (native FP16 ADD / MUL vs FP32 reference), per the 6-A discipline
// "native lowp <=> FP32 reference".
//
// Requires CUDA; skips gracefully without a GPU.
// =========================================================================

#include <cuda_fp16.h>
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

int failures = 0;
#define CHECK(cond)                                                     \
    do {                                                                \
        if (!(cond)) {                                                  \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                 \
        }                                                               \
    } while (0)

// ---- FIXED parity thresholds (6-A-8.1; frozen from the first measured run) ----
constexpr float kParityMaxAbs = 0.05f;   // logits are O(0.1..1) by construction
constexpr float kParityMaxRel = 0.05f;   // denominator guard: |ref| >= 0.15
constexpr float kParityRelGuard = 0.15f; // max|ref| = 0.207 prefill / 0.187 decode: engages
constexpr float kParityCosine = 0.999f;  // plan.md 2.2 end-to-end bar
constexpr int kTopK = 8;
constexpr int kTopKMinAgree = 7;         // of 8; random ties make 8/8 brittle

// Materialize a device tensor from float values. FP16 storage rounds through
// f32->f16; FP32 stores as-is (the FP32 reference graph sees the exact values
// the FP16 chain consumed, before storage rounding).
TensorPtr dev_from_values(const std::vector<float>& vals, int64_t r, int64_t c, DataType dt) {
    auto host = std::make_shared<Tensor>(dt, std::vector<int64_t>{r, c}, DeviceType::CPU);
    if (dt == DataType::FP16) {
        auto* h = static_cast<uint16_t*>(host->data());
        for (size_t i = 0; i < vals.size(); ++i) h[i] = f32_to_f16(vals[i]);
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
    if (t->dtype() == DataType::FP16) {
        auto* h = static_cast<uint16_t*>(host->data());
        for (size_t i = 0; i < n; ++i) v[i] = f16_to_f32(h[i]);
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

// =========================================================================
// 8.0: chain validation of the new native FP16 ADD / MUL kernels
// =========================================================================
void check_elementwise_fp16_kernels(std::mt19937& g) {
    const int n = 1000 + 256;  // not a multiple of the 256-thread block
    std::vector<float> av, bv;
    rand_vals(n, -2.0f, 2.0f, g, av);
    rand_vals(n, -2.0f, 2.0f, g, bv);
    std::vector<uint16_t> ac(n), bc(n);
    for (int i = 0; i < n; ++i) {
        ac[i] = f32_to_f16(av[i]);
        bc[i] = f32_to_f16(bv[i]);
    }
    auto mk16 = [&](const std::vector<uint16_t>& c) {
        auto host = std::make_shared<Tensor>(DataType::FP16, std::vector<int64_t>{1, n},
                                             DeviceType::CPU);
        std::memcpy(host->data(), c.data(), c.size() * 2);
        auto dev = std::make_shared<Tensor>(DataType::FP16, std::vector<int64_t>{1, n},
                                            DeviceType::CUDA);
        dev->copy_from(*host);
        return dev;
    };
    auto a = mk16(ac), b = mk16(bc);
    auto oa = mk16(std::vector<uint16_t>(n, 0));
    auto om = mk16(std::vector<uint16_t>(n, 0));

    // A: legacy stream.
    forge::cuda::launch_add_fp16(reinterpret_cast<const __half*>(a->data()),
                                 reinterpret_cast<const __half*>(b->data()),
                                 reinterpret_cast<__half*>(oa->data()), n, 0);
    forge::cuda::launch_multiply_fp16(reinterpret_cast<const __half*>(a->data()),
                                      reinterpret_cast<const __half*>(b->data()),
                                      reinterpret_cast<__half*>(om->data()), n, 0);
    cudaDeviceSynchronize();
    auto codesA = to_host_u16(oa);
    auto codesM = to_host_u16(om);

    // B: explicit capture stream -- must match A bitwise.
    cudaStream_t cs = nullptr;
    CHECK(cudaStreamCreate(&cs) == cudaSuccess);
    auto oaB = mk16(std::vector<uint16_t>(n, 0));
    auto omB = mk16(std::vector<uint16_t>(n, 0));
    forge::cuda::launch_add_fp16(reinterpret_cast<const __half*>(a->data()),
                                 reinterpret_cast<const __half*>(b->data()),
                                 reinterpret_cast<__half*>(oaB->data()), n, cs);
    forge::cuda::launch_multiply_fp16(reinterpret_cast<const __half*>(a->data()),
                                      reinterpret_cast<const __half*>(b->data()),
                                      reinterpret_cast<__half*>(omB->data()), n, cs);
    cudaStreamSynchronize(cs);
    CHECK(to_host_u16(oaB) == codesA);
    CHECK(to_host_u16(omB) == codesM);
    cudaStreamDestroy(cs);

    // Reference: CPU FP32 on the FP16-rounded inputs (kernel math is FP32 per
    // element; only the output storage rounds).
    float add_max = 0, mul_max = 0;
    for (int i = 0; i < n; ++i) {
        float x = f16_to_f32(ac[i]), y = f16_to_f32(bc[i]);
        add_max = std::max(add_max, std::fabs(f16_to_f32(codesA[i]) - (x + y)));
        mul_max = std::max(mul_max, std::fabs(f16_to_f32(codesM[i]) - (x * y)));
    }
    CHECK(add_max < 1e-3f);  // FP16 output rounding of |x+y| <= 4
    CHECK(mul_max < 1e-2f);  // FP16 output rounding of |x*y| <= 4
}

// =========================================================================
// 8.1: the end-to-end FP16 closed loop
// =========================================================================
struct GraphSpec {
    int D = 256;
    int nh = 8, nkv = 2, hd = 64;
    int FF = 512;
    int V = 128;      // vocab (logits width)
    int layers = 2;
    int q_len = 16;   // == kv_len => prefill; 1 => decode
    int kv_len = 16;
};

// One set of float values drives BOTH the FP16 and the FP32 graph: the FP16
// graph rounds them through f32->f16 at materialization, the FP32 graph
// consumes the exact values. Any logits difference is therefore attributable
// to FP16 storage/compute inside the chain, not to input drift.
//
// Decode mode models the real decode step: q comes from the current token,
// while K/V are the KV-cache contents (kv_len entries) consumed as graph
// inputs -- not recomputed from the single token. Prefill computes K/V from
// the (multi-token) hidden state like a real prefill does.
struct ModelValues {
    std::vector<float> x;                                          // [q_len, D]
    std::vector<std::vector<float>> w_attn, wq, wk, wv, wo;        // per layer
    std::vector<std::vector<float>> w_ffn, wg, wu;                 // per layer
    std::vector<std::vector<float>> k_cache, v_cache;              // per layer, decode only
    std::vector<float> w_out;                                      // [1, D]
    std::vector<float> wvoc;                                       // [V, D]
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
        // Always generate the KV-cache: prefill and decode share one ModelValues
        // ("same model, different token count"), only decode consumes it.
        m.k_cache.push_back(wmat(s.kv_len, s.nkv * s.hd, 0.5f));
        m.v_cache.push_back(wmat(s.kv_len, s.nkv * s.hd, 0.5f));
    }
    m.w_out = wmat(1, s.D, 0.2f);
    m.wvoc = wmat(s.V, s.D, ws);
    return m;
}

// Materialize the model in `dt` (FP16: storage-rounded; FP32: exact values).
struct ModelTensors {
    TensorPtr x;
    std::vector<TensorPtr> w_attn, wq, wk, wv, wo, w_ffn, wg, wu;
    std::vector<TensorPtr> k_cache, v_cache;
    TensorPtr w_out, wvoc;
};

ModelTensors materialize(const GraphSpec& s, const ModelValues& mv, DataType dt) {
    ModelTensors t;
    // x: use the FIRST q_len token rows of the shared model values (decode
    // reuses the same "model" with a single-token input).
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
        if (s.q_len == 1) {
            t.k_cache.push_back(dev_from_values(mv.k_cache[l], s.kv_len, s.nkv * s.hd, dt));
            t.v_cache.push_back(dev_from_values(mv.v_cache[l], s.kv_len, s.nkv * s.hd, dt));
        }
    }
    t.w_out = dev_from_values(mv.w_out, 1, s.D, dt);
    t.wvoc = dev_from_values(mv.wvoc, s.V, s.D, dt);
    return t;
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
        // cache inputs directly (q_len=1, causal covers the whole cache since
        // both FP32 and FP16 decode kernels attend all total_len positions).
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

Parity compare_logits(const std::vector<float>& got16, const std::vector<float>& ref32) {
    Parity p;
    p.agree_k = kTopK;
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < ref32.size(); ++i) {
        float diff = std::fabs(got16[i] - ref32[i]);
        p.max_abs = std::max(p.max_abs, diff);
        p.max_ref = std::max(p.max_ref, std::fabs(ref32[i]));
        if (std::fabs(ref32[i]) >= kParityRelGuard)
            p.max_rel = std::max(p.max_rel, diff / std::fabs(ref32[i]));
        dot += (double)got16[i] * ref32[i];
        na += (double)got16[i] * got16[i];
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
    auto ta = topk(got16), tb = topk(ref32);
    for (int i = 0; i < kTopK; ++i)
        if (ta[i] != tb[i]) p.topk_agree++;
    return p;
}

// ---- the full 6-A-8 gate set for one mode (prefill / decode) ----
void check_e2e(const GraphSpec& s, const ModelValues& mv, std::mt19937& g) {
    const char* mode = (s.q_len == 1) ? "decode" : "prefill";

    auto t16 = materialize(s, mv, DataType::FP16);
    auto t32 = materialize(s, mv, DataType::FP32);

    auto g16 = build_model_graph(s, t16);
    auto g32 = build_model_graph(s, t32);

    // ---- FP32 reference run (existing FP32 path, same topology) ----
    auto ref_out = g32->execute();
    cudaDeviceSynchronize();
    CHECK(ref_out != nullptr);
    CHECK(ref_out->dtype() == DataType::FP32);
    cudaDeviceSynchronize();
    std::vector<float> ref = to_host_f32(g32->last_output());
    CHECK((int)ref.size() == s.q_len * s.V);

    // ---- FP16 run (ordinary) ----
    auto out16 = g16->execute();
    cudaDeviceSynchronize();
    CHECK(out16 != nullptr);
    CHECK(out16->dtype() == DataType::FP16);  // hard dtype assert, not just numerics
    cudaDeviceSynchronize();
    auto codesA = to_host_u16(g16->last_output());
    void* bufA = g16->last_output()->data();
    size_t nbytesA = g16->last_output()->nbytes();
    CHECK(nbytesA == (size_t)s.q_len * s.V * 2);  // FP16-sized logits buffer

    std::vector<float> got = to_host_f32(g16->last_output());

    // ---- acceptance 1: dtype trace -- the whole main chain stays FP16 ----
    const auto& trace = g16->dtype_trace();
    CHECK(trace.size() == (size_t)g16->num_nodes());
    size_t dst_injected = 0;
    for (const auto& r : trace) {
        bool all_in_fp16 = true;
        for (auto d : r.input_dtypes)
            if (d != DataType::FP16) all_in_fp16 = false;
        CHECK(all_in_fp16);
        CHECK(r.output_dtype == DataType::FP16);
        if (r.dst_injected) dst_injected++;
    }

    // The dst-injectable FP16 ops must actually go through the planner buffer
    // (RMS_NORM / ADD / MUL / SILU): proves 6-A-6.2/6-A-8.0 plumbing, not the
    // legacy allocate+copy detour.
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
    g16->clear_dtype_trace();

    // ---- acceptance 2: FP16 vs FP32 parity (FIXED thresholds) ----
    Parity par = compare_logits(got, ref);

    CHECK(par.max_ref >= kParityRelGuard);  // the rel metric must be non-vacuous
    CHECK(par.max_abs < kParityMaxAbs);
    CHECK(par.max_rel < kParityMaxRel);
    CHECK(par.cosine >= kParityCosine);
    CHECK(par.topk_agree <= kTopK - kTopKMinAgree);

    // ---- acceptance 3: CUDA Graph on the WHOLE FP16 graph ----
    CHECK(forge::CudaGraphRunner::is_compatible(*g16));

    // B: explicit legacy stream -- bitwise equal to A.
    g16->execute(0);
    cudaDeviceSynchronize();
    CHECK(to_host_u16(g16->last_output()) == codesA);

    cudaStream_t cs = nullptr;
    CHECK(cudaStreamCreate(&cs) == cudaSuccess);
    forge::CudaGraphRunner runner;
    runner.set_enabled(true);

    // C: capture the whole graph on the capture stream.
    bool captured = runner.try_replay_or_capture(*g16, [&]() { g16->execute(cs); }, cs);
    cudaStreamSynchronize(cs);
    CHECK(captured);
    CHECK(to_host_u16(g16->last_output()) == codesA);

    // D: replay x N -- bitwise stable every time, buffer address/size stable.
    constexpr int kReplays = 3;
    for (int i = 0; i < kReplays; ++i) {
        bool replayed = runner.try_replay_or_capture(*g16, [&]() { g16->execute(cs); }, cs);
        cudaStreamSynchronize(cs);
        CHECK(replayed);
        CHECK(to_host_u16(g16->last_output()) == codesA);
        CHECK(g16->last_output()->data() == bufA);
        CHECK(g16->last_output()->nbytes() == nbytesA);
    }
    cudaStreamDestroy(cs);


    g16->release_graph();
    g32->release_graph();
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("SKIP: no CUDA device available (test_e2e_fp16)\n");
        return 0;
    }
    setenv("FORGE_TRACE_DTYPE", "1", 1);
    forge::register_builtin_op_kernels();


    std::mt19937 rng(0x6A08);

    check_elementwise_fp16_kernels(rng);

    // Shared model values: prefill and decode see the same weights; only the
    // token count (q_len) differs, so both modes validate the same "model".
    GraphSpec prefill;
    prefill.q_len = prefill.kv_len = 16;
    ModelValues mv = make_model_values(prefill, rng);
    check_e2e(prefill, mv, rng);

    GraphSpec decode;
    decode.q_len = 1;
    decode.kv_len = 16;
    check_e2e(decode, mv, rng);

    if (failures == 0) {
        std::printf("PASS e2e-fp16 (Phase 6-A-8)\n");
        return 0;
    }
    std::printf("FAILURES: %d\n", failures);
    return 1;
}
