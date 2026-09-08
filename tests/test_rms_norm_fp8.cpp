// =========================================================================
// test_rms_norm_fp8.cpp -- Phase 6-C-2: native FP8 E4M3 RMSNorm correctness +
// chain validation + micro-benchmark (requires CUDA; skips gracefully when no
// GPU is present).
//
// Compute model under test (mirrors the 6-A-2 / 6-B-2 discipline):
//     FP8 x -> FP32 square / reduction -> rsqrt -> FP32 scale -> FP8 out
// i.e. a native FP8 *storage path* with an FP32 reduction (norm_compute stays
// FP32 while activation is FP8 E4M3), NOT "all intermediates in fp8".
// The weight is stored in FP8 and decoded per element.
//
// Correctness: the reference is computed on the *decoded E4M3 inputs*
// (encode -> decode round trip applied to the test vector first), so the only
// residual error is the FP8 output store (3-bit mantissa) -- the input
// quantization error must NOT leak into the kernel error metric. Structural
// defects (wrong axis, wrong weight, dropped reduction) produce O(1) garbage.
//
// Metrics: cosine is the PRIMARY gate (quantization noise is roughly
// sign-symmetric, so direction is preserved); max_abs / max_rel are
// diagnostics bounded by the E4M3 store step (~3.1% relative half-step above
// the subnormal range).
//
// Chain validation (the 6-A lesson): FP8 out feeds a second FP8 RMSNorm, and
// the reference applies an encode/decode round trip between hops -- single-
// kernel tests cannot see store/decode error compounding.
// =========================================================================

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "forge/compute_graph.h"
#include "forge/dtype_contract.h"
#include "forge/fp8_utils.h"
#include "forge/operators.h"
#include "forge/tensor.h"
#include "forge/types.h"
#include "operators/cuda/cuda_rms_norm.h"

namespace {

using forge::DataType;
using forge::DeviceType;
using forge::Tensor;
using forge::TensorPtr;

std::vector<int64_t> shape2(int64_t r, int64_t c) {
    return std::vector<int64_t>{r, c};
}

float urand(std::mt19937& g, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    return d(g);
}

// FP32 reference RMSNorm operating on already-decoded FP8 float values.
void rms_norm_ref(const std::vector<float>& x, const std::vector<float>& w, int rows, int cols,
                  float eps, std::vector<float>& out) {
    for (int r = 0; r < rows; ++r) {
        float ss = 0.0f;
        for (int c = 0; c < cols; ++c) {
            float v = x[static_cast<size_t>(r) * cols + c];
            ss += v * v;
        }
        float rms = 1.0f / std::sqrt(ss / static_cast<float>(cols) + eps);
        for (int c = 0; c < cols; ++c) {
            float v = x[static_cast<size_t>(r) * cols + c];
            out[static_cast<size_t>(r) * cols + c] = v * rms * w[c];
        }
    }
}

void encode_fp8(const std::vector<float>& src, std::vector<uint8_t>& dst) {
    dst.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i)
        dst[i] = forge::fp32_to_fp8_e4m3(src[i]);
}

void decode_fp8(const std::vector<uint8_t>& src, std::vector<float>& dst) {
    dst.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i)
        dst[i] = forge::fp8_e4m3_to_fp32(src[i]);
}

struct Metrics {
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    double cosine = 1.0;
};

Metrics compare(const std::vector<float>& got, const std::vector<float>& ref) {
    Metrics m;
    double dot = 0.0, ng = 0.0, nr = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        float diff = std::fabs(got[i] - ref[i]);
        m.max_abs = std::max(m.max_abs, diff);
        if (std::fabs(ref[i]) > 0.1f) {
            float re = diff / std::fabs(ref[i]);
            m.max_rel = std::max(m.max_rel, re);
        }
        dot += static_cast<double>(got[i]) * ref[i];
        ng += static_cast<double>(got[i]) * got[i];
        nr += static_cast<double>(ref[i]) * ref[i];
    }
    m.cosine = dot / (std::sqrt(ng) * std::sqrt(nr) + 1e-30);
    return m;
}

TensorPtr fp8_device_tensor(const std::vector<uint8_t>& bytes, int rows, int cols) {
    auto host = std::make_shared<Tensor>(DataType::FP8_E4M3, shape2(rows, cols), DeviceType::CPU);
    std::memcpy(host->data(), bytes.data(), bytes.size());
    auto dev = std::make_shared<Tensor>(DataType::FP8_E4M3, shape2(rows, cols), DeviceType::CUDA);
    dev->copy_from(*host);
    return dev;
}

std::vector<uint8_t> fp8_host_bytes(const TensorPtr& dev) {
    auto host = std::make_shared<Tensor>(DataType::FP8_E4M3, dev->shape(), DeviceType::CPU);
    host->copy_from(*dev);
    const uint8_t* p = static_cast<const uint8_t*>(host->data());
    return std::vector<uint8_t>(p, p + static_cast<size_t>(dev->numel()));
}

// ---- correctness: native FP8 RMSNorm vs decoded-input FP32 reference ------
int run_correctness() {
    int failures = 0;
    std::mt19937 rng(0xF00DC2u);

    // Retirement proof: 6-C-2 retired RMS_NORM from the op_supports_fp8 ledger.
    if (!forge::op_supports_fp8(forge::OpType::RMS_NORM)) {
        std::printf("FAIL: op_supports_fp8(RMS_NORM) expected true (6-C-2)\n");
        return 1;
    }

    const std::vector<int> hiddens = {4096, 8192};
    const std::vector<int> rows_list = {1, 512};

    for (int hidden : hiddens) {
        for (int rows : rows_list) {
            const int cols = hidden;
            std::vector<float> xf_raw(static_cast<size_t>(rows) * cols);
            std::vector<float> wf_raw(cols);
            for (auto& v : xf_raw)
                v = urand(rng, -2.0f, 2.0f);
            for (auto& v : wf_raw)
                v = urand(rng, 0.9f, 1.1f);

            // Encode to FP8, then decode -- the reference consumes decoded
            // values so input quantization error stays out of the metric.
            std::vector<uint8_t> xb, wb;
            encode_fp8(xf_raw, xb);
            encode_fp8(wf_raw, wb);
            std::vector<float> xf, wf;
            decode_fp8(xb, xf);
            decode_fp8(wb, wf);

            auto x_dev = fp8_device_tensor(xb, rows, cols);
            auto w_dev = fp8_device_tensor(wb, 1, cols);

            auto out_dev = forge::ops::rms_norm(x_dev, w_dev, 1e-5f);
            if (!out_dev || out_dev->dtype() != DataType::FP8_E4M3) {
                std::printf("FAIL: FP8 RMSNorm returned wrong dtype\n");
                return 1;
            }
            std::vector<uint8_t> ob = fp8_host_bytes(out_dev);
            std::vector<float> of;
            decode_fp8(ob, of);

            std::vector<float> ref(xf.size());
            rms_norm_ref(xf, wf, rows, cols, 1e-5f, ref);

            Metrics m = compare(of, ref);
            std::printf("  weighted hidden=%5d rows=%4d : max_abs=%.3e max_rel=%.3e cosine=%.6f\n",
                        hidden, rows, m.max_abs, m.max_rel, m.cosine);
            if (m.max_abs > 0.15f) {
                std::printf("    FAIL max_abs too large (fp8 store rounding only)\n");
                ++failures;
            }
            // E4M3 relative half-step upper bound is 1/16 = 6.25% (3-bit
            // mantissa, worst case at the low end of a binade). Measured
            // values sit at ~6.0% -- right at the theoretical store bound.
            if (m.max_rel > 0.07f) {
                std::printf("    FAIL max_rel too large (bound: 1/16 store step)\n");
                ++failures;
            }
            if (m.cosine < 0.999) {
                std::printf("    FAIL cosine below 0.999 (primary gate)\n");
                ++failures;
            }
        }
    }

    // Unweighted FP8 path.
    {
        const int rows = 64, cols = 4096;
        std::vector<float> xf_raw(static_cast<size_t>(rows) * cols);
        for (auto& v : xf_raw)
            v = urand(rng, -2.0f, 2.0f);
        std::vector<uint8_t> xb;
        encode_fp8(xf_raw, xb);
        std::vector<float> xf;
        decode_fp8(xb, xf);

        auto x_dev = fp8_device_tensor(xb, rows, cols);
        auto out_dev = forge::ops::rms_norm(x_dev, nullptr, 1e-5f);
        if (!out_dev || out_dev->dtype() != DataType::FP8_E4M3) {
            std::printf("FAIL: unweighted FP8 RMSNorm returned wrong dtype\n");
            return 1;
        }
        std::vector<uint8_t> ob = fp8_host_bytes(out_dev);
        std::vector<float> of;
        decode_fp8(ob, of);

        std::vector<float> wf(cols, 1.0f), ref(xf.size());
        rms_norm_ref(xf, wf, rows, cols, 1e-5f, ref);
        Metrics m = compare(of, ref);
        std::printf("  unweighted hidden=%5d rows=%4d : max_abs=%.3e cosine=%.6f\n", cols, rows,
                    m.max_abs, m.cosine);
        if (m.max_abs > 0.15f || m.cosine < 0.999) {
            std::printf("    FAIL unweighted accuracy\n");
            ++failures;
        }
    }

    // Mixed dtypes (FP8 x + FP32 weight) must fail loudly, never mis-cast.
    {
        auto x_dev =
            fp8_device_tensor(std::vector<uint8_t>(128, forge::fp32_to_fp8_e4m3(0.5f)), 1, 128);
        auto w_host = std::make_shared<Tensor>(DataType::FP32, shape2(1, 128), DeviceType::CPU);
        {
            std::vector<float> wb(128, 1.0f);
            std::memcpy(w_host->data(), wb.data(), wb.size() * 4);
        }
        auto w_dev = std::make_shared<Tensor>(DataType::FP32, shape2(1, 128), DeviceType::CUDA);
        w_dev->copy_from(*w_host);
        bool threw = false;
        try {
            (void)forge::ops::rms_norm(x_dev, w_dev, 1e-5f);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        if (!threw) {
            std::printf("FAIL: mixed FP8/FP32 RMSNorm did not throw\n");
            ++failures;
        } else {
            std::printf("  mixed FP8 x + FP32 weight: loud reject OK\n");
        }
    }

    return failures;
}

// ---- chain validation: FP8 -> RMSNorm -> FP8 -> RMSNorm -> FP8 ------------
// The reference applies an encode/decode round trip after each hop, so the
// comparison isolates kernel math + store/decode compounding, not the input
// quantization. Single-kernel tests cannot see this error path (6-A lesson).
int run_chain() {
    std::mt19937 rng(0xC411u);
    const int rows = 8, cols = 4096;
    std::vector<float> xf_raw(static_cast<size_t>(rows) * cols);
    for (auto& v : xf_raw)
        v = urand(rng, -2.0f, 2.0f);
    std::vector<uint8_t> xb, wb;
    encode_fp8(xf_raw, xb);
    {
        std::vector<float> wf_raw(cols, 1.0f);
        encode_fp8(wf_raw, wb);
    }
    std::vector<float> xf, wf;
    decode_fp8(xb, xf);
    decode_fp8(wb, wf);

    auto x_dev = fp8_device_tensor(xb, rows, cols);
    auto w_dev = fp8_device_tensor(wb, 1, cols);

    // Two chained native FP8 RMSNorm hops (real FP8 tensors feed the next hop).
    auto out1 = forge::ops::rms_norm(x_dev, w_dev, 1e-5f);
    auto out2 = forge::ops::rms_norm(out1, w_dev, 1e-5f);
    std::vector<uint8_t> ob = fp8_host_bytes(out2);
    std::vector<float> got;
    decode_fp8(ob, got);

    // Reference: FP32 math on decoded values with an FP8 round trip per hop
    // (store + decode = exactly what the next native hop consumes).
    std::vector<float> ref(xf.size());
    rms_norm_ref(xf, wf, rows, cols, 1e-5f, ref);
    for (size_t i = 0; i < ref.size(); ++i)
        ref[i] = forge::fp8_e4m3_to_fp32(forge::fp32_to_fp8_e4m3(ref[i]));
    std::vector<float> ref2(ref.size());
    rms_norm_ref(ref, wf, rows, cols, 1e-5f, ref2);
    for (size_t i = 0; i < ref2.size(); ++i)
        ref2[i] = forge::fp8_e4m3_to_fp32(forge::fp32_to_fp8_e4m3(ref2[i]));

    Metrics m = compare(got, ref2);
    std::printf("  chain 2-hop hidden=%5d rows=%4d : max_abs=%.3e max_rel=%.3e cosine=%.6f\n",
                cols, rows, m.max_abs, m.max_rel, m.cosine);
    if (m.cosine < 0.998 || m.max_rel > 0.09f) {
        std::printf("    FAIL chain drift beyond fp8 store compounding bound\n");
        return 1;
    }
    return 0;
}

// ---- stream A/B: default stream vs a side stream must agree bitwise ------
int run_stream_ab() {
    std::mt19937 rng(0x57EA4Bu);
    const int rows = 16, cols = 4096;
    std::vector<float> xf_raw(static_cast<size_t>(rows) * cols);
    std::vector<float> wf_raw(cols);
    for (auto& v : xf_raw)
        v = urand(rng, -2.0f, 2.0f);
    for (auto& v : wf_raw)
        v = urand(rng, 0.9f, 1.1f);
    std::vector<uint8_t> xb, wb;
    encode_fp8(xf_raw, xb);
    encode_fp8(wf_raw, wb);

    auto x_dev = fp8_device_tensor(xb, rows, cols);
    auto w_dev = fp8_device_tensor(wb, 1, cols);

    auto out_default = forge::ops::rms_norm(x_dev, w_dev, 1e-5f);

    cudaStream_t side = nullptr;
    cudaStreamCreate(&side);
    auto x_side =
        std::make_shared<Tensor>(DataType::FP8_E4M3, shape2(rows, cols), DeviceType::CUDA);
    auto w_side = std::make_shared<Tensor>(DataType::FP8_E4M3, shape2(1, cols), DeviceType::CUDA);
    x_side->copy_from(*x_dev);
    w_side->copy_from(*w_dev);
    cudaDeviceSynchronize();  // copies must land before the side-stream launch
    std::vector<uint8_t> ob_side(static_cast<size_t>(rows) * cols);
    {
        auto out_side = std::make_shared<Tensor>(DataType::FP8_E4M3, shape2(rows, cols),
                                                 DeviceType::CUDA);
        forge::cuda::launch_rms_norm_fp8(static_cast<const uint8_t*>(x_side->data()),
                                         static_cast<const uint8_t*>(w_side->data()),
                                         static_cast<uint8_t*>(out_side->data()), rows, cols,
                                         1e-5f, side);
        cudaStreamSynchronize(side);
        auto host = std::make_shared<Tensor>(DataType::FP8_E4M3, shape2(rows, cols),
                                             DeviceType::CPU);
        host->copy_from(*out_side);
        const uint8_t* p = static_cast<const uint8_t*>(host->data());
        ob_side.assign(p, p + static_cast<size_t>(rows) * cols);
    }
    cudaStreamDestroy(side);

    std::vector<uint8_t> ob_default = fp8_host_bytes(out_default);
    if (ob_default != ob_side) {
        std::printf("FAIL: stream A/B mismatch (kernel must be deterministic)\n");
        return 1;
    }
    std::printf("  stream A/B: bitwise identical OK\n");
    return 0;
}

// ---- single-node graph smoke: FP8 dtype follows through dst injection -----
int run_graph_smoke() {
    std::mt19937 rng(0xC2FBu);
    const int rows = 4, cols = 1024;
    std::vector<float> xf_raw(static_cast<size_t>(rows) * cols);
    std::vector<float> wf_raw(cols);
    for (auto& v : xf_raw)
        v = urand(rng, -2.0f, 2.0f);
    for (auto& v : wf_raw)
        v = urand(rng, 0.9f, 1.1f);
    std::vector<uint8_t> xb, wb;
    encode_fp8(xf_raw, xb);
    encode_fp8(wf_raw, wb);
    auto x = fp8_device_tensor(xb, rows, cols);
    auto w = fp8_device_tensor(wb, 1, cols);

    // Reference: the same op called standalone.
    auto ref = forge::ops::rms_norm(x, w, 1e-5f);

    // The same op through ComputeGraph (dst injection path).
    forge::ComputeGraph g;
    int xi = g.add_input(x);
    int wi = g.add_input(w);
    int32_t params[8] = {0};
    float eps = 1e-5f;
    std::memcpy(params, &eps, sizeof(float));
    g.add_node("rms_norm", forge::OpType::RMS_NORM, {xi, wi}, params, DeviceType::CUDA);
    TensorPtr out;
    try {
        out = g.execute();
    } catch (const std::exception& e) {
        std::printf("FAIL: graph FP8 RMSNorm threw: %s\n", e.what());
        return 1;
    }
    if (!out || out->dtype() != DataType::FP8_E4M3) {
        std::printf("FAIL: graph FP8 RMSNorm wrong dtype\n");
        return 1;
    }
    std::vector<uint8_t> got = fp8_host_bytes(out);
    std::vector<uint8_t> want = fp8_host_bytes(ref);
    if (got != want) {
        std::printf("FAIL: graph FP8 RMSNorm differs from eager (bitwise expected)\n");
        return 1;
    }
    std::printf("  graph smoke: dtype=%s bitwise identical OK\n",
                forge::dtype_name(out->dtype()).c_str());
    return 0;
}

// ---- micro-bench: FP8 vs BF16 vs FP16 vs FP32 ----------------------------
void fill_fp8_const(const TensorPtr& t, float v) {
    auto host = std::make_shared<Tensor>(t->dtype(), t->shape(), DeviceType::CPU);
    const size_t n = static_cast<size_t>(t->numel());
    std::vector<uint8_t> buf(n, forge::fp32_to_fp8_e4m3(v));
    std::memcpy(host->data(), buf.data(), n);
    t->copy_from(*host);
}

void fill_bf16_const(const TensorPtr& t, float v) {
    // BF16 bits: 1.0f = 0x3F80, 0.5f = 0x3E00 (both exact).
    uint16_t bits = (v == 1.0f) ? 0x3F80u : 0x3E00u;
    auto host = std::make_shared<Tensor>(t->dtype(), t->shape(), DeviceType::CPU);
    const size_t n = static_cast<size_t>(t->numel());
    std::vector<uint16_t> buf(n, bits);
    std::memcpy(host->data(), buf.data(), n * 2);
    t->copy_from(*host);
}

void run_bench(int rows, int cols, float eps, int iters) {
    auto x8 = std::make_shared<Tensor>(DataType::FP8_E4M3, shape2(rows, cols), DeviceType::CUDA);
    auto w8 = std::make_shared<Tensor>(DataType::FP8_E4M3, shape2(1, cols), DeviceType::CUDA);
    auto o8 = std::make_shared<Tensor>(DataType::FP8_E4M3, shape2(rows, cols), DeviceType::CUDA);
    auto xb = std::make_shared<Tensor>(DataType::BF16, shape2(rows, cols), DeviceType::CUDA);
    auto wb = std::make_shared<Tensor>(DataType::BF16, shape2(1, cols), DeviceType::CUDA);
    auto ob = std::make_shared<Tensor>(DataType::BF16, shape2(rows, cols), DeviceType::CUDA);
    auto x32 = std::make_shared<Tensor>(DataType::FP32, shape2(rows, cols), DeviceType::CUDA);
    auto w32 = std::make_shared<Tensor>(DataType::FP32, shape2(1, cols), DeviceType::CUDA);
    auto o32 = std::make_shared<Tensor>(DataType::FP32, shape2(rows, cols), DeviceType::CUDA);
    fill_fp8_const(x8, 0.5f);
    fill_fp8_const(w8, 1.0f);
    fill_bf16_const(xb, 0.5f);
    fill_bf16_const(wb, 1.0f);
    {
        auto h = std::make_shared<Tensor>(DataType::FP32, x32->shape(), DeviceType::CPU);
        std::vector<float> buf(static_cast<size_t>(x32->numel()), 0.5f);
        std::memcpy(h->data(), buf.data(), buf.size() * 4);
        x32->copy_from(*h);
    }
    {
        auto h = std::make_shared<Tensor>(DataType::FP32, w32->shape(), DeviceType::CPU);
        std::vector<float> buf(static_cast<size_t>(w32->numel()), 1.0f);
        std::memcpy(h->data(), buf.data(), buf.size() * 4);
        w32->copy_from(*h);
    }

    auto launch_fp8 = [&] {
        forge::cuda::launch_rms_norm_fp8(static_cast<const uint8_t*>(x8->data()),
                                         static_cast<const uint8_t*>(w8->data()),
                                         static_cast<uint8_t*>(o8->data()), rows, cols, eps);
    };
    auto launch_bf16 = [&] {
        forge::cuda::launch_rms_norm_bf16(static_cast<const __nv_bfloat16*>(xb->data()),
                                          static_cast<const __nv_bfloat16*>(wb->data()),
                                          static_cast<__nv_bfloat16*>(ob->data()), rows, cols,
                                          eps);
    };
    auto launch_fp32 = [&] {
        forge::cuda::launch_rms_norm(static_cast<const float*>(x32->data()),
                                     static_cast<const float*>(w32->data()),
                                     static_cast<float*>(o32->data()), rows, cols, eps);
    };

    for (int i = 0; i < 5; ++i) {
        launch_fp8();
        launch_bf16();
        launch_fp32();
    }
    cudaDeviceSynchronize();

    cudaEvent_t es, ee;
    cudaEventCreate(&es);
    cudaEventCreate(&ee);
    float ms8 = 0.0f, msb = 0.0f, ms32 = 0.0f;

    auto time = [&](float& out_ms, const std::function<void()>& fn) {
        cudaEventRecord(es);
        for (int i = 0; i < iters; ++i)
            fn();
        cudaEventRecord(ee);
        cudaEventSynchronize(ee);
        cudaEventElapsedTime(&out_ms, es, ee);
    };
    time(ms8, launch_fp8);
    time(msb, launch_bf16);
    time(ms32, launch_fp32);

    cudaEventDestroy(es);
    cudaEventDestroy(ee);

    const double bytes8 = static_cast<double>(rows) * cols * 2.0 + static_cast<double>(cols) * 1.0;
    const double bytes16 = static_cast<double>(rows) * cols * 4.0 + static_cast<double>(cols) * 2.0;
    const double bytes32 = static_cast<double>(rows) * cols * 8.0 + static_cast<double>(cols) * 4.0;
    std::printf("bench hidden=%d rows=%d iters=%d:\n", cols, rows, iters);
    std::printf("  FP8 : %.4f ms/iter  %.1f GB/s\n", ms8 / iters,
                bytes8 / (ms8 / 1000.0 / iters) / 1e9);
    std::printf("  BF16: %.4f ms/iter  %.1f GB/s\n", msb / iters,
                bytes16 / (msb / 1000.0 / iters) / 1e9);
    std::printf("  FP32: %.4f ms/iter  %.1f GB/s\n", ms32 / iters,
                bytes32 / (ms32 / 1000.0 / iters) / 1e9);
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("SKIP: no CUDA device available (test_rms_norm_fp8)\n");
        return 0;
    }

    std::printf("[6-C-2] FP8 RMSNorm correctness vs decoded-input FP32 reference\n");
    int failures = run_correctness();
    std::printf("[6-C-2] chain validation (FP8 -> RMSNorm -> FP8 -> RMSNorm -> FP8)\n");
    failures += run_chain();
    std::printf("[6-C-2] stream A/B determinism\n");
    failures += run_stream_ab();
    std::printf("[6-C-2] single-node graph smoke (FP8 dst injection)\n");
    failures += run_graph_smoke();
    if (failures == 0) {
        std::printf("[6-C-2] micro-bench (FP8 vs BF16 vs FP32, three regimes)\n");
        run_bench(1, 4096, 1e-5f, 50);    // launch-bound
        run_bench(512, 8192, 1e-5f, 50);  // L2-resident (fp8 working set ~8.4 MB)
        run_bench(4096, 8192, 1e-5f, 20); // DRAM-bound (fp8 ~67 MB > L2, honest bandwidth)
        std::printf("PASS rms-norm-fp8 (Phase 6-C-2)\n");
        return 0;
    }
    std::printf("FAILURES: %d\n", failures);
    return 1;
}
