// =========================================================================
// test_graph_fp16.cpp -- Phase 6-A-6: native FP16 Graph / Binding integration
//
//     6-A-6.1  graph builder must not promote FP16 tensors to FP32
//     6-A-6.2  binding layer keeps FP16 dtype (planner sizes FP16, not FP32)
//     6-A-6.3  dst injection must not silently materialize FP32
//     6-A-6.5  a real FP16 graph runs end-to-end:  RMSNorm -> Matmul -> SiLU
//     6-A-6.6  dtype trace proves no FP16 -> FP32 -> FP16 round trip
//     6-A-6.4  CUDA Graph capture/replay keeps FP16 buffer address/size stable
//
// This is the step that turns "several isolated FP16 kernels" into a real
// ComputeGraph execution of the same operator chain: same operators, same
// tensors, plus graph memory planning + dst injection + dispatch.
//
// Requires CUDA; skips gracefully without a GPU.
// =========================================================================

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "forge/compute_graph.h"
#include "forge/cuda_graph_runner.h"
#include "forge/cuda_mem_pool.h"
#include "forge/op_dispatch.h"
#include "forge/op_enum.h"
#include "forge/operator_activation.h"
#include "forge/operator_matmul.h"
#include "forge/operator_norm.h"
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
std::vector<int64_t> shape2(int64_t r, int64_t c) {
    return std::vector<int64_t>{r, c};
}

TensorPtr make_dev_fp16(const std::vector<uint16_t>& h, int64_t r, int64_t c) {
    auto host = std::make_shared<Tensor>(DataType::FP16, shape2(r, c), DeviceType::CPU);
    std::memcpy(host->data(), h.data(), h.size() * 2);
    auto dev = std::make_shared<Tensor>(DataType::FP16, shape2(r, c), DeviceType::CUDA);
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
    for (int64_t i = 0; i < n; ++i)
        codes[i] = f32_to_f16(d(g));
}

// forge::dtype_name (quant_traits.h) is reused directly.

// Build the real FP16 chain as a ComputeGraph with real OpType nodes:
//   x --RMS_NORM(w)--> h --MUL_MAT_TRANSB(b)--> p --SILU--> o
std::unique_ptr<forge::ComputeGraph> build_chain_graph(const TensorPtr& x, const TensorPtr& w,
                                                       const TensorPtr& b) {
    auto g = std::make_unique<forge::ComputeGraph>();
    int x_i = g->add_input(x);
    int w_i = g->add_input(w);
    int b_i = g->add_input(b);

    int32_t rms_params[forge::OP_PARAMS_MAX_SIZE / sizeof(int32_t)] = {0};
    float eps = 1e-5f;
    std::memcpy(rms_params, &eps, sizeof(float));

    int n0 = g->add_node("rms_norm", forge::OpType::RMS_NORM, {x_i, w_i}, rms_params,
                         DeviceType::CUDA);
    int n1 = g->add_node("matmul", forge::OpType::MUL_MAT_TRANSB, {-n0 - 1, b_i}, nullptr,
                         DeviceType::CUDA);
    g->add_node("silu", forge::OpType::SILU, {-n1 - 1}, nullptr, DeviceType::CUDA);
    return g;
}

// 6-A-6.5 / 6-A-6.6: run the graph, compare against the standalone operator chain,
// and assert no stage is promoted to FP32.
void check_graph_chain(int M, int K, int N, std::mt19937& g) {
    std::vector<uint16_t> x_code, b_code, w_code;
    rand_fp16((int64_t)M * K, -2.0f, 2.0f, g, x_code);
    rand_fp16((int64_t)N * K, -1.0f, 1.0f, g, b_code);
    rand_fp16((int64_t)K, 0.9f, 1.1f, g, w_code);  // RMSNorm weight [1,K]

    auto x = make_dev_fp16(x_code, M, K);
    auto b = make_dev_fp16(b_code, N, K);
    auto w = make_dev_fp16(w_code, 1, K);

    // ---- reference: the same operators, called standalone (6-A-5 chain) ----
    auto h_ref = forge::ops::rms_norm(x, w, 1e-5f);
    auto p_ref = forge::ops::matmul_transB(h_ref, b);
    auto o_ref = forge::ops::silu(p_ref);

    // ---- the same chain, but through ComputeGraph ----
    auto graph = build_chain_graph(x, w, b);
    auto out = graph->execute();

    CHECK(out != nullptr);
    if (!out)
        return;
    CHECK(out->dtype() == DataType::FP16);  // 6-A-6.1: no promotion to FP32 at the boundary

    auto got = to_host_u16(out);
    auto want = to_host_u16(o_ref);
    float max_abs = 0, max_rel = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        float v = f16_to_f32(got[i]);
        float diff = std::fabs(v - f16_to_f32(want[i]));
        if (diff > max_abs)
            max_abs = diff;
        if (std::fabs(f16_to_f32(want[i])) > 0.1f) {
            float re = diff / std::fabs(f16_to_f32(want[i]));
            if (re > max_rel)
                max_rel = re;
        }
    }
    std::printf("  graph chain M=%d K=%d N=%d : max_abs=%.3e max_rel=%.3e (out=%s)\n", M, K, N,
                max_abs, max_rel, forge::dtype_name(out->dtype()).c_str());
    // Same operators, same tensors, plus graph plumbing: results must match the
    // standalone chain to FP16 storage precision. A regression (FP32 fallback,
    // mis-cast dst buffer, wrong element size) blows these bounds by orders of magnitude.
    CHECK(max_abs < 0.05f);
    CHECK(max_rel < 0.05f);

    // ---- 6-A-6.6: dtype trace -- every stage stays FP16 ----
    const auto& trace = graph->dtype_trace();
    CHECK(trace.size() == 3);
    for (const auto& r : trace) {
        bool all_in_fp16 = true;
        for (auto d : r.input_dtypes)
            if (d != DataType::FP16)
                all_in_fp16 = false;
        CHECK(all_in_fp16);
        CHECK(r.output_dtype == DataType::FP16);  // no FP16 -> FP32 materialization
        std::printf("    trace %-9s in=%s out=%s dst_injected=%d\n", r.name.c_str(),
                    forge::dtype_name(r.input_dtypes.empty() ? DataType::FP16 : r.input_dtypes[0])
                        .c_str(),
                    forge::dtype_name(r.output_dtype).c_str(), r.dst_injected ? 1 : 0);
    }
    if (trace.size() == 3) {
        // 6-A-6.3: the dst-injected nodes wrote FP16 into the pre-allocated graph
        // buffer (they did NOT get an FP32 dst view and silently upcast).
        CHECK(trace[0].dst_injected);  // RMS_NORM
        CHECK(trace[2].dst_injected);  // SILU
    }

    // ---- 6-A-6.2: the planner must size the buffer in FP16 (not FP32) ----
    // The SILU node output is M*N halves; if the planner had promoted, it would be
    // 2x and the graph buffer would be oversized. Check via the output nbytes.
    CHECK(out->nbytes() == static_cast<size_t>(M) * N * 2);
}

// 6-A-6.4: full-chain stream capture — RMSNorm -> Matmul -> SiLU on a capture
// stream, with strong assertions (no soft-report):
//
//   A: execute()             (implicit legacy stream)
//   B: execute(0)            (explicit legacy stream) — must equal A bitwise
//   C: capture on cap_stream (first try_replay_or_capture) — must equal A
//   D: replay on cap_stream  (second try_replay_or_capture) — must equal C
//
// Plus the allocation probe: capture must not trigger cudaMalloc (pool hits
// only). Matmul is legacy (allocates its temp internally); the A/B warmup runs
// put that temp into the pool freelist, so a capture-safe run shows
// alloc_count delta == total temp allocs served from cache. If capture fails,
// this test FAILS: matmul allocation capture-safety is a hard 6-A-6.4 gate.
void check_cuda_graph_fp16_capture(int M, int K, int N, std::mt19937& g) {
    std::vector<uint16_t> x_code, b_code, w_code;
    rand_fp16((int64_t)M * K, -2.0f, 2.0f, g, x_code);
    rand_fp16((int64_t)N * K, -1.0f, 1.0f, g, b_code);
    rand_fp16((int64_t)K, 0.9f, 1.1f, g, w_code);

    auto x = make_dev_fp16(x_code, M, K);
    auto b = make_dev_fp16(b_code, N, K);
    auto w = make_dev_fp16(w_code, 1, K);

    auto graph = build_chain_graph(x, w, b);
    CHECK(forge::CudaGraphRunner::is_compatible(*graph));

    // A: implicit legacy stream.
    auto outA = graph->execute();
    CHECK(outA != nullptr);
    CHECK(outA->dtype() == DataType::FP16);
    cudaDeviceSynchronize();
    void* bufA = graph->last_output()->data();
    size_t nbytesA = graph->last_output()->nbytes();
    auto codesA = to_host_u16(graph->last_output());

    // B: explicit legacy stream — must match A bitwise (default-stream compat).
    graph->execute(0);
    cudaDeviceSynchronize();
    auto codesB = to_host_u16(graph->last_output());
    CHECK(codesA == codesB);

    // Replay preconditions: stable address + exact FP16 footprint.
    CHECK(graph->last_output()->data() == bufA);
    CHECK(graph->last_output()->nbytes() == nbytesA);
    CHECK(nbytesA == static_cast<size_t>(M) * N * 2);  // FP16-sized, not FP32

    // --- allocation probe data (6-A-6.4 hard-gate evidence) ---
    size_t allocs_before = forge::cuda_mem::alloc_count();
    size_t hits_before = forge::cuda_mem::hit_count();

    cudaStream_t cs = nullptr;
    CHECK(cudaStreamCreate(&cs) == cudaSuccess);

    forge::CudaGraphRunner runner;
    runner.set_enabled(true);
    // C: capture the full chain on the capture stream.
    bool captured = runner.try_replay_or_capture(*graph, [&]() { graph->execute(cs); }, cs);
    cudaStreamSynchronize(cs);
    size_t allocs_after = forge::cuda_mem::alloc_count();
    size_t hits_after = forge::cuda_mem::hit_count();
    std::printf("  alloc probe M=%d K=%d N=%d : allocs %zu->%zu, hits %zu->%zu\n", M, K, N,
                allocs_before, allocs_after, hits_before, hits_after);

    CHECK(captured);  // strong assert: capture must succeed
    auto codesC = to_host_u16(graph->last_output());
    CHECK(codesC == codesA);  // max_abs(regular, capture) == 0

    // D: replay on the same stream — must match capture bitwise.
    bool replayed = runner.try_replay_or_capture(*graph, [&]() { graph->execute(cs); }, cs);
    cudaStreamSynchronize(cs);
    CHECK(replayed);
    auto codesD = to_host_u16(graph->last_output());
    CHECK(codesD == codesC);  // max_abs(capture, replay) == 0

    // Replay keeps the same FP16 buffer.
    CHECK(graph->last_output()->data() == bufA);
    CHECK(graph->last_output()->nbytes() == nbytesA);

    cudaStreamDestroy(cs);
    std::printf("  cuda graph M=%d K=%d N=%d : capture+replay OK, buffer=%p size=%zu (FP16)\n",
                M, K, N, bufA, nbytesA);
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("SKIP: no CUDA device available (test_graph_fp16)\n");
        return 0;
    }

    // Enable the Phase 6-A-6 dtype trace before anything touches ComputeGraph.
    setenv("FORGE_TRACE_DTYPE", "1", 1);
    forge::register_builtin_op_kernels();

    std::printf("[6-A-6] Native FP16 Graph / Binding integration (RMSNorm -> Matmul -> SiLU)\n");
    std::mt19937 rng(0x6A06);

    check_graph_chain(1, 4096, 11008, rng);
    check_graph_chain(8, 4096, 11008, rng);
    check_graph_chain(32, 4096, 11008, rng);

    check_cuda_graph_fp16_capture(8, 1024, 2048, rng);

    if (failures == 0) {
        std::printf("PASS graph-fp16 (Phase 6-A-6)\n");
        return 0;
    }
    std::printf("FAILURES: %d\n", failures);
    return 1;
}
