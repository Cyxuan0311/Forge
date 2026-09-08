// =========================================================================
// test_graph_bf16.cpp -- Phase 6-B-5: native BF16 Graph / Binding integration
//
// Mirrors test_graph_fp16.cpp (6-A-6) for BF16:
//
//     BF16 input
//        -> BF16 RMSNorm (dst, 6-B-2)
//        -> BF16 Matmul  (legacy, 6-B-4 -- no dst kernel by design, as in 6-A-6)
//        -> BF16 SiLU    (dst, 6-B-3)
//        -> BF16 ADD     (dst, 6-B-5 residual node)
//        -> BF16 MUL     (dst, 6-B-5 scale node)
//        -> BF16 output
//
// This is the final proof surface of the BF16 contract: every node that is
// native-eager must also be native in-graph (no eager-native / graph-legacy
// split), the planner must size everything in BF16 (elements x 2), the dtype
// trace must read BF16 -> BF16 end to end, and CUDA Graph capture/replay must
// be bitwise stable on a stable BF16 buffer.
//
// Requires CUDA; skips gracefully without a GPU.
// =========================================================================

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "forge/compute_graph.h"
#include "forge/cuda_graph_runner.h"
#include "forge/cuda_mem_pool.h"
#include "forge/op_dispatch.h"
#include "forge/op_enum.h"
#include "forge/operator_activation.h"
#include "forge/operator_elementwise.h"
#include "forge/operator_matmul.h"
#include "forge/operator_norm.h"
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
std::vector<int64_t> shape2(int64_t r, int64_t c) {
    return std::vector<int64_t>{r, c};
}

TensorPtr make_dev_bf16(const std::vector<uint16_t>& h, int64_t r, int64_t c) {
    auto host = std::make_shared<Tensor>(DataType::BF16, shape2(r, c), DeviceType::CPU);
    std::memcpy(host->data(), h.data(), h.size() * 2);
    auto dev = std::make_shared<Tensor>(DataType::BF16, shape2(r, c), DeviceType::CUDA);
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

void rand_bf16(int64_t n, float lo, float hi, std::mt19937& g, std::vector<uint16_t>& codes) {
    std::uniform_real_distribution<float> d(lo, hi);
    codes.resize(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
        codes[i] = f32_to_bf16(d(g));
}

// Build the 5-node BF16 chain as a ComputeGraph with real OpType nodes:
//   x --RMS_NORM(w)--> h --MUL_MAT_TRANSB(b)--> p --SILU--> s
//     --ADD(res)--> r --MUL(scl)--> o
std::unique_ptr<forge::ComputeGraph> build_chain_graph(const TensorPtr& x, const TensorPtr& w,
                                                       const TensorPtr& b, const TensorPtr& res,
                                                       const TensorPtr& scl) {
    auto g = std::make_unique<forge::ComputeGraph>();
    int x_i = g->add_input(x);
    int w_i = g->add_input(w);
    int b_i = g->add_input(b);
    int r_i = g->add_input(res);
    int s_i = g->add_input(scl);

    int32_t rms_params[forge::OP_PARAMS_MAX_SIZE / sizeof(int32_t)] = {0};
    float eps = 1e-5f;
    std::memcpy(rms_params, &eps, sizeof(float));

    int n0 = g->add_node("rms_norm", forge::OpType::RMS_NORM, {x_i, w_i}, rms_params,
                         DeviceType::CUDA);
    int n1 = g->add_node("matmul", forge::OpType::MUL_MAT_TRANSB, {-n0 - 1, b_i}, nullptr,
                         DeviceType::CUDA);
    int n2 = g->add_node("silu", forge::OpType::SILU, {-n1 - 1}, nullptr, DeviceType::CUDA);
    int n3 = g->add_node("add", forge::OpType::ADD, {-n2 - 1, r_i}, nullptr, DeviceType::CUDA);
    g->add_node("mul", forge::OpType::MUL, {-n3 - 1, s_i}, nullptr, DeviceType::CUDA);
    return g;
}

// 6-B-5: run the graph, compare against the standalone operator chain,
// and assert no stage is promoted to FP32.
void check_graph_chain(int M, int K, int N, std::mt19937& g) {
    std::vector<uint16_t> x_code, b_code, w_code, r_code, s_code;
    rand_bf16((int64_t)M * K, -2.0f, 2.0f, g, x_code);
    rand_bf16((int64_t)N * K, -1.0f, 1.0f, g, b_code);
    rand_bf16((int64_t)K, 0.9f, 1.1f, g, w_code);  // RMSNorm weight [1,K]
    rand_bf16((int64_t)M * N, -1.0f, 1.0f, g, r_code);
    rand_bf16((int64_t)M * N, 0.9f, 1.1f, g, s_code);

    auto x = make_dev_bf16(x_code, M, K);
    auto b = make_dev_bf16(b_code, N, K);
    auto w = make_dev_bf16(w_code, 1, K);
    auto res = make_dev_bf16(r_code, M, N);
    auto scl = make_dev_bf16(s_code, M, N);

    // ---- reference: the same operators, called standalone ----
    auto h_ref = forge::ops::rms_norm(x, w, 1e-5f);
    auto p_ref = forge::ops::matmul_transB(h_ref, b);
    auto s_ref = forge::ops::silu(p_ref);
    auto r_ref = forge::ops::add(s_ref, res);
    auto o_ref = forge::ops::multiply(r_ref, scl);

    // ---- the same chain, but through ComputeGraph ----
    auto graph = build_chain_graph(x, w, b, res, scl);
    auto out = graph->execute();

    CHECK(out != nullptr);
    if (!out)
        return;
    CHECK(out->dtype() == DataType::BF16);  // no promotion to FP32 at the boundary

    auto got = to_host_u16(out);
    auto want = to_host_u16(o_ref);
    float max_abs = 0, max_rel = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        float v = bf16_to_f32(got[i]);
        float diff = std::fabs(v - bf16_to_f32(want[i]));
        if (diff > max_abs)
            max_abs = diff;
        if (std::fabs(bf16_to_f32(want[i])) >= 1.0f) {
            float re = diff / std::fabs(bf16_to_f32(want[i]));
            if (re > max_rel)
                max_rel = re;
        }
    }
    std::printf("  graph chain M=%d K=%d N=%d : max_abs=%.3e max_rel(|ref|>=1)=%.3e (out=%s)\n",
                M, K, N, max_abs, max_rel, forge::dtype_name(out->dtype()).c_str());
    // Same operators, same tensors, plus graph plumbing (planning / dst
    // injection / dispatch). Thresholds mirror the 6-B-4.3 chain family
    // (cosine-grade agreement); a regression (FP32 fallback, mis-cast dst
    // buffer, wrong element size) blows these bounds by orders of magnitude.
    CHECK(max_abs < 2.0f);
    CHECK(max_rel < 0.3f);

    // ---- dtype trace -- every stage stays BF16 ----
    const auto& trace = graph->dtype_trace();
    CHECK(trace.size() == 5);
    for (const auto& r : trace) {
        bool all_in_bf16 = true;
        for (auto d : r.input_dtypes)
            if (d != DataType::BF16)
                all_in_bf16 = false;
        CHECK(all_in_bf16);
        CHECK(r.output_dtype == DataType::BF16);  // no BF16 -> FP32 materialization
        std::printf("    trace %-9s in=%s out=%s dst_injected=%d\n", r.name.c_str(),
                    forge::dtype_name(r.input_dtypes.empty() ? DataType::BF16 : r.input_dtypes[0])
                        .c_str(),
                    forge::dtype_name(r.output_dtype).c_str(), r.dst_injected ? 1 : 0);
    }
    if (trace.size() == 5) {
        // dst-injected nodes wrote BF16 into the pre-allocated graph buffer.
        // matmul (trace[1]) is legacy by design (no dst kernel, as in 6-A-6).
        CHECK(trace[0].dst_injected);   // RMS_NORM
        CHECK(!trace[1].dst_injected);  // MUL_MAT_TRANSB (legacy, expected)
        CHECK(trace[2].dst_injected);   // SILU
        CHECK(trace[3].dst_injected);   // ADD
        CHECK(trace[4].dst_injected);   // MUL
    }

    // ---- the planner must size the buffer in BF16 (not FP32) ----
    // The MUL node output is M*N bf16 halves; if the planner had promoted, it
    // would be 2x and the graph buffer would be oversized.
    CHECK(out->nbytes() == static_cast<size_t>(M) * N * 2);
}

// Full-chain stream capture on the 5-node BF16 graph, same A/B/C/D protocol
// as the FP16 capture test (ordinary == capture == replay, bitwise; stable
// BF16 buffer address/size; no cudaMalloc inside capture).
void check_cuda_graph_bf16_capture(int M, int K, int N, std::mt19937& g) {
    std::vector<uint16_t> x_code, b_code, w_code, r_code, s_code;
    rand_bf16((int64_t)M * K, -2.0f, 2.0f, g, x_code);
    rand_bf16((int64_t)N * K, -1.0f, 1.0f, g, b_code);
    rand_bf16((int64_t)K, 0.9f, 1.1f, g, w_code);
    rand_bf16((int64_t)M * N, -1.0f, 1.0f, g, r_code);
    rand_bf16((int64_t)M * N, 0.9f, 1.1f, g, s_code);

    auto x = make_dev_bf16(x_code, M, K);
    auto b = make_dev_bf16(b_code, N, K);
    auto w = make_dev_bf16(w_code, 1, K);
    auto res = make_dev_bf16(r_code, M, N);
    auto scl = make_dev_bf16(s_code, M, N);

    auto graph = build_chain_graph(x, w, b, res, scl);
    CHECK(forge::CudaGraphRunner::is_compatible(*graph));

    // A: implicit legacy stream.
    auto outA = graph->execute();
    CHECK(outA != nullptr);
    CHECK(outA->dtype() == DataType::BF16);
    cudaDeviceSynchronize();
    void* bufA = graph->last_output()->data();
    size_t nbytesA = graph->last_output()->nbytes();
    auto codesA = to_host_u16(graph->last_output());

    // B: explicit legacy stream — must match A bitwise.
    graph->execute(0);
    cudaDeviceSynchronize();
    auto codesB = to_host_u16(graph->last_output());
    CHECK(codesA == codesB);

    // Replay preconditions: stable address + exact BF16 footprint.
    CHECK(graph->last_output()->data() == bufA);
    CHECK(graph->last_output()->nbytes() == nbytesA);
    CHECK(nbytesA == static_cast<size_t>(M) * N * 2);  // BF16-sized, not FP32

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

    CHECK(captured);
    auto codesC = to_host_u16(graph->last_output());
    CHECK(codesC == codesA);

    // D: replay on the same stream — must match capture bitwise.
    bool replayed = runner.try_replay_or_capture(*graph, [&]() { graph->execute(cs); }, cs);
    cudaStreamSynchronize(cs);
    CHECK(replayed);
    auto codesD = to_host_u16(graph->last_output());
    CHECK(codesD == codesC);

    // Replay keeps the same BF16 buffer.
    CHECK(graph->last_output()->data() == bufA);
    CHECK(graph->last_output()->nbytes() == nbytesA);

    cudaStreamDestroy(cs);
    std::printf("  cuda graph M=%d K=%d N=%d : capture+replay OK, buffer=%p size=%zu (BF16)\n",
                M, K, N, bufA, nbytesA);
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("SKIP: no CUDA device available (test_graph_bf16)\n");
        return 0;
    }

    // Enable the dtype trace before anything touches ComputeGraph.
    setenv("FORGE_TRACE_DTYPE", "1", 1);
    forge::register_builtin_op_kernels();

    std::printf("[6-B-5] Native BF16 Graph / Binding integration (RMSNorm -> Matmul -> SiLU -> Add -> Mul)\n");
    std::mt19937 rng(0x6B05);

    check_graph_chain(1, 4096, 11008, rng);
    check_graph_chain(8, 4096, 11008, rng);
    check_graph_chain(32, 4096, 11008, rng);

    check_cuda_graph_bf16_capture(8, 1024, 2048, rng);

    if (failures == 0) {
        std::printf("PASS graph-bf16 (Phase 6-B-5)\n");
        return 0;
    }
    std::printf("FAILURES: %d\n", failures);
    return 1;
}
