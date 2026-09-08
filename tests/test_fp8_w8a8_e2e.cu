// Phase 6-C-5.6: minimal deterministic W8A8 FP8 E2E chain.
//
// Drives the REAL production chain through the ops boundary (the same entry the
// ComputeGraph uses for a layer), NOT standalone launcher calls:
//
//   x_f32 --quantize_fp8_e4m3_w8a8--> A(fp8, device-authoritative scale sidecar)
//           (or, 5.6a: fused ReLU activation producer -> FP8 activation, no
//            materialized FP32 intermediate)
//   w_f32 --quantize_fp8_e4m3_w8a8--> B(fp8, device-authoritative scale sidecar)
//   Y = cuda_matmul_transB(A, B)          (M>1: CUTLASS GEMM;  M==1: FP8 GEMV)
//   Y dtype == FP32.  The chain STOPS at FP32 -- nothing is quantized back to
//   FP8 (no FP8 -> FP32 -> FP8 round-trip anywhere).
//
// The scale NEVER leaves the Tensor sidecar: the consumer resolves both scales
// via Tensor::fp8_scale_device() (device-authoritative slots written by the
// producer kernels), so the fp8_scale_source_counts() gate must show
// sidecar > 0 AND registry == 0.
//
// Gates:
//   1. FP32 output vs decode-based CPU golden  (cosine / max-abs)
//   2. determinism: two identical runs are bitwise equal (no atomics in path)
//   3. source counters: registry == 0, sidecar >= 2 per matmul
//   4. dtype/authority: producer out FP8 + fp8_scale_device_authoritative(),
//      matmul out FP32
//   5. 5.6a activation producer (ReLU fused) byte-and-scale equals a manual
//      ReLU-then-quantize reference, i.e. one fused scale contract, no FP32
//      activation materialized on the device path
//
// Skips gracefully without an Ada (sm_89+) GPU.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

#include "forge/fp8_quant.h"
#include "forge/fp8_scale_debug.h"
#include "forge/fp8_utils.h"
#include "forge/tensor.h"
#include "operators/cuda/matmul_cuda.h"

using namespace forge;

static int g_failures = 0;

static bool device_is_ada() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess)
        return false;
    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    return (major * 10 + minor) >= 89;
}

static TensorPtr make_cuda_fp32(const std::vector<float>& h, int64_t rows, int64_t cols) {
    auto t = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{rows, cols},
                                      DeviceType::CUDA);
    if (!t->data())
        throw std::runtime_error("make_cuda_fp32: CUDA alloc failed");
    cudaMemcpy(t->data(), h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);
    return t;
}

static TensorPtr make_out_fp32(int64_t rows, int64_t cols) {
    return std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{rows, cols},
                                    DeviceType::CUDA);
}

static float read_device_scale(const TensorPtr& t) {
    float s = 0.f;
    if (!t->fp8_scale_device_ptr())
        throw std::runtime_error("read_device_scale: no device scale slot");
    cudaMemcpy(&s, t->fp8_scale_device_ptr(), sizeof(float), cudaMemcpyDeviceToHost);
    return s;
}

static std::vector<uint8_t> fetch_bytes(const TensorPtr& t) {
    std::vector<uint8_t> h(t->numel());
    cudaMemcpy(h.data(), t->data(), h.size(), cudaMemcpyDeviceToHost);
    return h;
}

static void fetch_floats(const TensorPtr& t, std::vector<float>& h) {
    h.resize(static_cast<size_t>(t->numel()));
    cudaMemcpy(h.data(), t->data(), h.size() * sizeof(float), cudaMemcpyDeviceToHost);
}

// 6-C-5.1a producer + read-back of the device-computed scale. Asserts the result
// is a device-authoritative FP8 tensor (producer owns the scale write).
static TensorPtr quantize_and_check(const TensorPtr& x, const char* what) {
    auto q = quantize_fp8_e4m3_w8a8(x, 0);
    if (!q)
        throw std::runtime_error(std::string(what) + ": producer returned null");
    if (q->dtype() != DataType::FP8_E4M3)
        throw std::runtime_error(std::string(what) + ": producer output not FP8_E4M3");
    if (!q->has_fp8_scale() || !q->fp8_scale_device_ptr())
        throw std::runtime_error(std::string(what) + ": producer scale sidecar missing");
    if (!q->fp8_scale_device_authoritative())
        throw std::runtime_error(std::string(what) + ": scale not device-authoritative");
    return q;
}

// CPU golden: Y[m*N+n] = s_a*s_w * sum_k decode(A[m*K+k]) * decode(B[n*K+k]).
// B is the Forge transB weight in its native row-major [N,K] form (row n
// contiguous) -- the exact buffer layout consumed by BOTH the CUTLASS FP8 GEMM
// (column-major [K,N], ldb=K) and the row-warp FP8 GEMV production backend.
// A and B arrive pre-quantized (bytes fetched from the device producer), so the
// golden re-uses exactly the encoded payload the device path consumes.
static std::vector<float> ref_chain(const std::vector<uint8_t>& qA, float sa,
                                    const std::vector<uint8_t>& qB, float sb, int M, int K, int N) {
    std::vector<float> y((size_t)M * N, 0.f);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k)
                acc += fp8_e4m3_to_fp32(qA[(size_t)m * K + k]) *
                       fp8_e4m3_to_fp32(qB[(size_t)n * K + k]);
            y[(size_t)m * N + n] = acc * (sa * sb);
        }
    return y;
}

// One full chain case. `apply_relu` selects the 6-C-5.6a fused ReLU activation
// producer (act = quantize(relu(x))) instead of the plain input quantize.
static bool run_chain_case(const char* name, int M, int K, int N, bool apply_relu) {
    bool ok = true;
    std::mt19937 rng(20260908u + (unsigned)(M * 131 + K * 17 + N) + (apply_relu ? 1u : 0u));
    std::uniform_real_distribution<float> u(-0.8f, 0.8f);

    std::vector<float> hx((size_t)M * K), hw((size_t)N * K);
    for (float& v : hx)
        v = u(rng);
    // Row-major [N,K] Forge transB weight: output row n holds its K input weights
    // contiguously (W[n*K + k]) -- the production layout for FP8 GEMM and GEMV.
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k)
            hw[(size_t)n * K + k] = u(rng);
    auto x_dev = make_cuda_fp32(hx, M, K);
    auto w_dev = make_cuda_fp32(hw, N, K);

    // ---- Activation producer (A side) ----
    TensorPtr act;
    float sa = 0.f;
    if (apply_relu) {
        // 5.6a: fused ReLU producer, no FP32 relu tensor materialized on device.
        act = std::make_shared<Tensor>(DataType::FP8_E4M3, std::vector<int64_t>{M, K},
                                       DeviceType::CUDA);
        if (!act->fp8_scale_device_slot(0))
            throw std::runtime_error("chain: activation slot alloc failed");
        quantize_fp8_e4m3_output_into(x_dev, act, 0, /*apply_relu=*/true);
        sa = read_device_scale(act);
    } else {
        act = quantize_and_check(x_dev, "chain:act");
        sa = read_device_scale(act);
    }
    // ---- Weight producer (B side) ----
    auto qb = quantize_and_check(w_dev, "chain:w");
    const float sb = read_device_scale(qb);

    // Fetch the exact quantized payloads the device chain consumes.
    const std::vector<uint8_t> qA_h = fetch_bytes(act);
    const std::vector<uint8_t> qB_h = fetch_bytes(qb);
    const std::vector<float> golden = ref_chain(qA_h, sa, qB_h, sb, M, K, N);

    // ---- Real consumer (ops boundary; same route as ComputeGraph layers) ----
    fp8_scale_reset_counts();
    auto out1 = make_out_fp32(M, N);
    auto out2 = make_out_fp32(M, N);  // identical second run for the determinism gate
    try {
        ops::cuda_matmul_transB(act, qb, out1, M, K, N, 0);
        ops::cuda_matmul_transB(act, qb, out2, M, K, N, 0);
    } catch (const std::exception& e) {
        printf("  [FAIL] %s: consumer threw: %s\n", name, e.what());
        return false;
    }
    if (out1->dtype() != DataType::FP32) {
        printf("  [FAIL] %s: chain output dtype is %s, expected FP32 (no FP8 re-quantize)\n", name,
               dtype_name(out1->dtype()).c_str());
        ok = false;
    }

    std::vector<float> y1, y2;
    fetch_floats(out1, y1);
    fetch_floats(out2, y2);

    // Gate 2: determinism -- the two identical runs must be bitwise equal.
    int bdiff = 0;
    for (size_t i = 0; i < y1.size(); ++i)
        if (std::memcmp(&y1[i], &y2[i], sizeof(float)) != 0)
            ++bdiff;
    if (bdiff != 0) {
        printf("  [FAIL] %s: two identical runs differ on %d/%zu elements (not deterministic)\n",
               name, bdiff, y1.size());
        ok = false;
    }

    // Gate 1: parity vs decode-based CPU golden.
    double dot = 0, ng = 0, nt = 0;
    float maxerr = 0.f;
    for (size_t i = 0; i < y1.size(); ++i) {
        maxerr = std::max(maxerr, std::fabs(y1[i] - golden[i]));
        dot += (double)y1[i] * golden[i];
        ng += (double)y1[i] * y1[i];
        nt += (double)golden[i] * golden[i];
    }
    const float cosine = (ng > 0 && nt > 0) ? (float)(dot / (std::sqrt(ng) * std::sqrt(nt))) : 0.f;
    if (cosine < 0.9995f || maxerr > 0.05f) {
        printf("  [FAIL] %s: parity cosine=%.6f maxerr=%.3e\n", name, cosine, maxerr);
        ok = false;
    }

    // Gate 3: the consumer must have resolved BOTH scales from Tensor sidecars.
    const Fp8ScaleSourceCounts c = fp8_scale_source_counts();
    if (c.registry != 0) {
        printf("  [FAIL] %s: legacy registry used (%llu), sidecar=%llu\n", name, c.registry,
               c.sidecar);
        ok = false;
    }
    if (c.sidecar < 2) {
        printf("  [FAIL] %s: expected >=2 sidecar lookups, got %llu\n", name, c.sidecar);
        ok = false;
    }

    if (ok)
        printf(
            "  [OK] 6-C-5.6 %s (%dx%dx%d relu=%d): FP32 out, cosine=%.6f maxerr=%.3e "
            "deterministic(bitwise) registry=%llu sidecar=%llu\n",
            name, M, K, N, (int)apply_relu, cosine, maxerr, c.registry, c.sidecar);
    return ok;
}

// 6-C-5.6a: the fused ReLU producer must be byte-and-scale identical to a manual
// ReLU-then-quantize reference. This is what lets the E2E chain run an FP8
// activation without materializing an FP32 intermediate.
static bool test_activation_producer() {
    bool ok = true;
    const int n = 8192;
    std::mt19937 rng(777u);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    std::vector<float> hx(n);
    for (float& v : hx)
        v = u(rng);

    auto x_dev = make_cuda_fp32(hx, n, 1);

    // Reference: relu on host, then the plain (unfused) output producer.
    std::vector<float> hrelu(n);
    for (int i = 0; i < n; ++i)
        hrelu[i] = std::max(hx[i], 0.f);
    auto xr_dev = make_cuda_fp32(hrelu, n, 1);
    auto ref = quantize_fp8_e4m3_output(xr_dev, 0);
    const float ref_scale = read_device_scale(ref);
    const std::vector<uint8_t> ref_bytes = fetch_bytes(ref);

    // Fused: quantize(relu(x)) in one producer -- no FP32 relu tensor on device.
    auto act =
        std::make_shared<Tensor>(DataType::FP8_E4M3, std::vector<int64_t>{n, 1}, DeviceType::CUDA);
    if (!act->fp8_scale_device_slot(0)) {
        printf("  [FAIL] 6-C-5.6a: activation slot alloc failed\n");
        return false;
    }
    quantize_fp8_e4m3_output_into(x_dev, act, 0, /*apply_relu=*/true);
    const float act_scale = read_device_scale(act);
    const std::vector<uint8_t> act_bytes = fetch_bytes(act);

    int bdiff = 0;
    for (int i = 0; i < n; ++i)
        if (act_bytes[i] != ref_bytes[i])
            ++bdiff;
    const bool scale_eq = (std::memcmp(&act_scale, &ref_scale, sizeof(float)) == 0);
    if (bdiff != 0 || !scale_eq) {
        printf(
            "  [FAIL] 6-C-5.6a: fused relu producer != relu-then-quantize (bytes differ "
            "%d/%d, scale %g vs %g)\n",
            bdiff, n, act_scale, ref_scale);
        ok = false;
    }
    // Negative-heavy input must collapse to scale 1 / all-zero bytes (relu kills
    // every negative; amax == 0 contract from 5.1b).
    std::vector<float> hneg(n, -1.5f);
    auto xn_dev = make_cuda_fp32(hneg, n, 1);
    auto act_neg =
        std::make_shared<Tensor>(DataType::FP8_E4M3, std::vector<int64_t>{n, 1}, DeviceType::CUDA);
    act_neg->fp8_scale_device_slot(0);
    quantize_fp8_e4m3_output_into(xn_dev, act_neg, 0, true);
    const float neg_scale = read_device_scale(act_neg);
    const std::vector<uint8_t> neg_bytes = fetch_bytes(act_neg);
    int nz = 0;
    for (uint8_t b : neg_bytes)
        if (b != 0)
            ++nz;
    if (neg_scale != 1.f || nz != 0) {
        printf("  [FAIL] 6-C-5.6a: all-negative input -> scale %g, %d non-zero bytes\n", neg_scale,
               nz);
        ok = false;
    }

    if (ok)
        printf(
            "  [OK] 6-C-5.6a activation producer: fused ReLU == relu-then-quantize "
            "(bytewise + scale); all-negative -> scale 1 + zeros\n");
    return ok;
}

int main() {
    // The 5.4 source counters only accumulate while trace is enabled (trace_enabled
    // is cached lazily, so set this before the first matmul/consumer call).
    setenv("FORGE_TRACE_FP8_SCALE", "1", 0);

    if (!device_is_ada()) {
        printf("[fp8-w8a8-e2e] SKIP: no Ada (sm_89+) GPU present\n");
        return 0;
    }

    // 5.6: M == 1 decode chain routes through the FP8 GEMV consumer.
    if (!run_chain_case("decode-chain", 1, 4096, 1024, /*apply_relu=*/false))
        ++g_failures;
    if (!run_chain_case("decode-act-chain", 1, 4096, 1024, /*apply_relu=*/true))
        ++g_failures;
    // 5.6: M > 1 prefill chain routes through the CUTLASS FP8 GEMM consumer.
    if (!run_chain_case("prefill-chain", 32, 512, 256, /*apply_relu=*/false))
        ++g_failures;
    if (!run_chain_case("prefill-act-chain", 32, 512, 256, /*apply_relu=*/true))
        ++g_failures;

    // 5.6a: fused ReLU activation producer correctness (single scale contract).
    if (!test_activation_producer())
        ++g_failures;

    if (g_failures == 0) {
        printf("\n[fp8-w8a8-e2e] PASS\n");
        return 0;
    }
    printf("\n[fp8-w8a8-e2e] FAILURES=%d\n", g_failures);
    return 1;
}
