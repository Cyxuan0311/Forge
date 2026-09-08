// =========================================================================
// test_rms_norm_bf16.cpp -- Phase 6-B-2: native BF16 RMSNorm correctness +
// micro-benchmark (requires CUDA; skips gracefully when no GPU is present).
//
// Compute model under test (mirrors the 6-A-2 FP16 discipline):
//     BF16 x -> FP32 square / reduction -> rsqrt -> FP16/FP32 scale -> BF16 out
// i.e. a native BF16 *storage/arithmetic path* with an FP32 reduction
// (norm_compute stays FP32 while activation is BF16), NOT "all intermediates
// in bfloat16". Numerical stability is preserved.
//
// Correctness: run the real CUDA kernel (via forge::ops::rms_norm, which now
// routes BF16 to launch_rms_norm_bf16) and compare against an FP32 CPU
// reference built from the *same decoded BF16 inputs* -- so the only residual
// error is the BF16 output store (7-bit mantissa, ~4-8x the FP16 rounding),
// which is exactly what we want to bound. Structural defects (wrong axis,
// wrong weight, dropped reduction) produce O(1) garbage and fail the tolerance.
//
// Micro-bench: kernel-only launch timing of BF16 vs FP32 (primary) and vs FP16
// (sanity: same byte count, expect ~parity -- gains come from bandwidth, not
// compute). Two regimes: small hidden (launch-bound) and large hidden
// (bandwidth-bound). No speedup is preset; measurement decides.
//
// Retirement proof: this test calls ops::rms_norm with BF16 and expects
// success -- it fails if any routing site (eager / dst / graph gate) still
// rejects BF16 for RMS_NORM (see test_bf16_retirement bookkeeping).
// =========================================================================

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "forge/compute_graph.h"
#include "forge/dtype_contract.h"
#include "forge/operator_norm.h"
#include "forge/tensor.h"
#include "forge/types.h"
#include "operators/cuda/cuda_rms_norm.h"

namespace {

using forge::DataType;
using forge::DeviceType;
using forge::Tensor;
using forge::TensorPtr;

// ---- bf16 codec (matches the kernel's storage convention) ----------------
// BF16 is the top 16 bits of FP32. Encode rounds to nearest-even on the
// discarded bits (so large-magnitude inputs don't bias); decode is exact
// (left shift). Both sides of every comparison below use the same decoded
// values, so the encode rounding never leaks into the error metric.
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

float urand(std::mt19937& g, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    return d(g);
}

// FP32 reference RMSNorm operating on already-decoded BF16 float values.
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

void fill_const(const TensorPtr& t, float v) {
    auto h = std::make_shared<Tensor>(t->dtype(), t->shape(), DeviceType::CPU);
    const size_t n = static_cast<size_t>(t->numel());
    if (t->dtype() == DataType::BF16) {
        std::vector<uint16_t> buf(n, f32_to_bf16(v));
        std::memcpy(h->data(), buf.data(), n * 2);
    } else if (t->dtype() == DataType::FP16) {
        // 0.5f is exactly representable in both FP16 and BF16; store directly.
        std::vector<uint16_t> buf(n, 0x3800u);
        (void)v;
        std::memcpy(h->data(), buf.data(), n * 2);
    } else {
        std::vector<float> buf(n, v);
        std::memcpy(h->data(), buf.data(), n * 4);
    }
    t->copy_from(*h);
}

// ---- correctness: native BF16 RMSNorm vs FP32 reference -----------------
int run_correctness() {
    int failures = 0;
    std::mt19937 rng(0xB16B00B5u);

    // Retirement bookkeeping ties to 6-B-1: RMS_NORM stays whitelisted and the
    // BF16 gate function itself is unchanged; the routing proof is this test.
    if (!forge::op_supports_lowp(forge::OpType::RMS_NORM)) {
        std::printf("FAIL: op_supports_lowp(RMS_NORM) expected true (6-A-2 + 6-B-2)\n");
        return 1;
    }

    const std::vector<int> hiddens = {4096, 8192};
    const std::vector<int> rows_list = {1, 32, 128, 512};

    for (int hidden : hiddens) {
        for (int rows : rows_list) {
            const int cols = hidden;
            std::vector<uint16_t> xb(static_cast<size_t>(rows) * cols);
            std::vector<uint16_t> wb(cols);
            for (auto& b : xb)
                b = f32_to_bf16(urand(rng, -2.0f, 2.0f));
            for (auto& b : wb)
                b = f32_to_bf16(urand(rng, 0.9f, 1.1f));

            auto x_host =
                std::make_shared<Tensor>(DataType::BF16, shape2(rows, cols), DeviceType::CPU);
            auto w_host =
                std::make_shared<Tensor>(DataType::BF16, shape2(1, cols), DeviceType::CPU);
            std::memcpy(x_host->data(), xb.data(), xb.size() * 2);
            std::memcpy(w_host->data(), wb.data(), wb.size() * 2);
            auto x_dev =
                std::make_shared<Tensor>(DataType::BF16, shape2(rows, cols), DeviceType::CUDA);
            auto w_dev =
                std::make_shared<Tensor>(DataType::BF16, shape2(1, cols), DeviceType::CUDA);
            x_dev->copy_from(*x_host);
            w_dev->copy_from(*w_host);

            // Native BF16 RMSNorm (weighted path).
            auto out_dev = forge::ops::rms_norm(x_dev, w_dev, 1e-5f);
            if (!out_dev || out_dev->dtype() != DataType::BF16) {
                std::printf("FAIL: BF16 RMSNorm returned wrong dtype\n");
                return 1;
            }
            auto out_host =
                std::make_shared<Tensor>(DataType::BF16, shape2(rows, cols), DeviceType::CPU);
            out_host->copy_from(*out_dev);

            // Reference from the same decoded BF16 inputs.
            std::vector<float> xf(static_cast<size_t>(rows) * cols);
            std::vector<float> wf(cols);
            std::vector<float> ref(xf.size());
            for (size_t i = 0; i < xf.size(); ++i)
                xf[i] = bf16_to_f32(xb[i]);
            for (int c = 0; c < cols; ++c)
                wf[c] = bf16_to_f32(wb[c]);
            rms_norm_ref(xf, wf, rows, cols, 1e-5f, ref);

            const uint16_t* out_bits = static_cast<const uint16_t*>(out_host->data());
            float max_abs = 0.0f, max_rel = 0.0f;
            for (size_t i = 0; i < ref.size(); ++i) {
                float got = bf16_to_f32(out_bits[i]);
                float diff = std::fabs(got - ref[i]);
                if (diff > max_abs)
                    max_abs = diff;
                if (std::fabs(ref[i]) > 0.1f) {
                    float re = diff / std::fabs(ref[i]);
                    if (re > max_rel)
                        max_rel = re;
                }
            }
            std::printf("  hidden=%5d rows=%4d : max_abs_err=%.3e max_rel_err=%.3e\n", hidden, rows,
                        max_abs, max_rel);
            if (max_abs > 0.02f) {
                std::printf("    FAIL max_abs_err too large (bf16 store rounding only)\n");
                ++failures;
            }
            if (max_rel > 0.05f) {
                std::printf("    FAIL max_rel_err too large\n");
                ++failures;
            }
        }
    }

    // Unweighted BF16 path (kernel uses an internal ones weight).
    {
        const int rows = 64, cols = 4096;
        std::vector<uint16_t> xb(static_cast<size_t>(rows) * cols);
        for (auto& b : xb)
            b = f32_to_bf16(urand(rng, -2.0f, 2.0f));
        auto x_host = std::make_shared<Tensor>(DataType::BF16, shape2(rows, cols), DeviceType::CPU);
        std::memcpy(x_host->data(), xb.data(), xb.size() * 2);
        auto x_dev = std::make_shared<Tensor>(DataType::BF16, shape2(rows, cols), DeviceType::CUDA);
        x_dev->copy_from(*x_host);

        auto out_dev = forge::ops::rms_norm(x_dev, nullptr, 1e-5f);
        if (!out_dev || out_dev->dtype() != DataType::BF16) {
            std::printf("FAIL: unweighted BF16 RMSNorm returned wrong dtype\n");
            return 1;
        }
        auto out_host =
            std::make_shared<Tensor>(DataType::BF16, shape2(rows, cols), DeviceType::CPU);
        out_host->copy_from(*out_dev);

        std::vector<float> xf(xb.size()), ref(xb.size());
        for (size_t i = 0; i < xf.size(); ++i)
            xf[i] = bf16_to_f32(xb[i]);
        std::vector<float> wf(cols, 1.0f);
        rms_norm_ref(xf, wf, rows, cols, 1e-5f, ref);

        const uint16_t* out_bits = static_cast<const uint16_t*>(out_host->data());
        float max_abs = 0.0f;
        for (size_t i = 0; i < ref.size(); ++i) {
            float diff = std::fabs(bf16_to_f32(out_bits[i]) - ref[i]);
            if (diff > max_abs)
                max_abs = diff;
        }
        std::printf("  unweighted hidden=%5d rows=%4d : max_abs_err=%.3e\n", cols, rows, max_abs);
        if (max_abs > 0.02f) {
            std::printf("    FAIL unweighted max_abs_err too large\n");
            ++failures;
        }
    }

    // Mixed dtypes (BF16 x + FP32 weight) must fail loudly, never mis-cast.
    {
        auto x_dev = std::make_shared<Tensor>(DataType::BF16, shape2(1, 64), DeviceType::CUDA);
        auto w_dev = std::make_shared<Tensor>(DataType::FP32, shape2(1, 64), DeviceType::CUDA);
        bool threw = false;
        try {
            (void)forge::ops::rms_norm(x_dev, w_dev, 1e-5f);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        if (!threw) {
            std::printf("FAIL: mixed BF16/FP32 RMSNorm did not throw\n");
            ++failures;
        } else {
            std::printf("  mixed BF16 x + FP32 weight: loud reject OK\n");
        }
    }

    return failures;
}

// ---- single-node graph smoke: BF16 dtype follows through dst injection ----
int run_graph_smoke() {
    std::mt19937 rng(0xBF16);
    const int rows = 4, cols = 1024;
    std::vector<uint16_t> xb(static_cast<size_t>(rows) * cols);
    std::vector<uint16_t> wb(cols);
    for (auto& b : xb)
        b = f32_to_bf16(urand(rng, -2.0f, 2.0f));
    for (auto& b : wb)
        b = f32_to_bf16(urand(rng, 0.9f, 1.1f));
    auto mk = [&](const std::vector<uint16_t>& codes, int r, int c) {
        auto host = std::make_shared<Tensor>(DataType::BF16, shape2(r, c), DeviceType::CPU);
        std::memcpy(host->data(), codes.data(), codes.size() * 2);
        auto dev = std::make_shared<Tensor>(DataType::BF16, shape2(r, c), DeviceType::CUDA);
        dev->copy_from(*host);
        return dev;
    };
    auto x = mk(xb, rows, cols);
    auto w = mk(wb, 1, cols);

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
        std::printf("FAIL: graph BF16 RMSNorm threw: %s\n", e.what());
        return 1;
    }
    if (!out || out->dtype() != DataType::BF16) {
        std::printf("FAIL: graph BF16 RMSNorm wrong dtype\n");
        return 1;
    }
    auto oh = std::make_shared<Tensor>(DataType::BF16, shape2(rows, cols), DeviceType::CPU);
    oh->copy_from(*out);
    auto rh = std::make_shared<Tensor>(DataType::BF16, shape2(rows, cols), DeviceType::CPU);
    rh->copy_from(*ref);
    const uint16_t* got = static_cast<const uint16_t*>(oh->data());
    const uint16_t* want = static_cast<const uint16_t*>(rh->data());
    float max_abs = 0.0f;
    for (size_t i = 0; i < static_cast<size_t>(rows) * cols; ++i)
        max_abs = std::max(max_abs, std::fabs(bf16_to_f32(got[i]) - bf16_to_f32(want[i])));
    std::printf("  graph smoke: dtype=%s max_abs_vs_eager=%.3e\n",
                forge::dtype_name(out->dtype()).c_str(), max_abs);
    if (!(max_abs == 0.0f)) {
        std::printf("FAIL: graph BF16 RMSNorm differs from eager (bitwise expected)\n");
        return 1;
    }
    return 0;
}

// ---- micro-bench: BF16 vs FP32 (primary) + vs FP16 (byte-parity sanity) ---
void run_bench(int rows, int cols, float eps, int iters) {
    auto xb = std::make_shared<Tensor>(DataType::BF16, shape2(rows, cols), DeviceType::CUDA);
    auto wb = std::make_shared<Tensor>(DataType::BF16, shape2(1, cols), DeviceType::CUDA);
    auto ob = std::make_shared<Tensor>(DataType::BF16, shape2(rows, cols), DeviceType::CUDA);
    auto x16 = std::make_shared<Tensor>(DataType::FP16, shape2(rows, cols), DeviceType::CUDA);
    auto w16 = std::make_shared<Tensor>(DataType::FP16, shape2(1, cols), DeviceType::CUDA);
    auto o16 = std::make_shared<Tensor>(DataType::FP16, shape2(rows, cols), DeviceType::CUDA);
    auto x32 = std::make_shared<Tensor>(DataType::FP32, shape2(rows, cols), DeviceType::CUDA);
    auto w32 = std::make_shared<Tensor>(DataType::FP32, shape2(1, cols), DeviceType::CUDA);
    auto o32 = std::make_shared<Tensor>(DataType::FP32, shape2(rows, cols), DeviceType::CUDA);
    fill_const(xb, 0.5f);
    fill_const(wb, 1.0f);
    fill_const(x16, 0.5f);
    fill_const(w16, 1.0f);
    fill_const(x32, 0.5f);
    fill_const(w32, 1.0f);

    for (int i = 0; i < 5; ++i)
        forge::cuda::launch_rms_norm_bf16(static_cast<const __nv_bfloat16*>(xb->data()),
                                          static_cast<const __nv_bfloat16*>(wb->data()),
                                          static_cast<__nv_bfloat16*>(ob->data()), rows, cols, eps);
    for (int i = 0; i < 5; ++i)
        forge::cuda::launch_rms_norm_fp16(static_cast<const __half*>(x16->data()),
                                          static_cast<const __half*>(w16->data()),
                                          static_cast<__half*>(o16->data()), rows, cols, eps);
    for (int i = 0; i < 5; ++i)
        forge::cuda::launch_rms_norm(static_cast<const float*>(x32->data()),
                                     static_cast<const float*>(w32->data()),
                                     static_cast<float*>(o32->data()), rows, cols, eps);
    cudaDeviceSynchronize();

    cudaEvent_t es, ee;
    cudaEventCreate(&es);
    cudaEventCreate(&ee);
    float msbf = 0.0f, ms16 = 0.0f, ms32 = 0.0f;

    cudaEventRecord(es);
    for (int i = 0; i < iters; ++i)
        forge::cuda::launch_rms_norm_bf16(static_cast<const __nv_bfloat16*>(xb->data()),
                                          static_cast<const __nv_bfloat16*>(wb->data()),
                                          static_cast<__nv_bfloat16*>(ob->data()), rows, cols, eps);
    cudaEventRecord(ee);
    cudaEventSynchronize(ee);
    cudaEventElapsedTime(&msbf, es, ee);

    cudaEventRecord(es);
    for (int i = 0; i < iters; ++i)
        forge::cuda::launch_rms_norm_fp16(static_cast<const __half*>(x16->data()),
                                          static_cast<const __half*>(w16->data()),
                                          static_cast<__half*>(o16->data()), rows, cols, eps);
    cudaEventRecord(ee);
    cudaEventSynchronize(ee);
    cudaEventElapsedTime(&ms16, es, ee);

    cudaEventRecord(es);
    for (int i = 0; i < iters; ++i)
        forge::cuda::launch_rms_norm(static_cast<const float*>(x32->data()),
                                     static_cast<const float*>(w32->data()),
                                     static_cast<float*>(o32->data()), rows, cols, eps);
    cudaEventRecord(ee);
    cudaEventSynchronize(ee);
    cudaEventElapsedTime(&ms32, es, ee);

    cudaEventDestroy(es);
    cudaEventDestroy(ee);

    const double bytes16 = static_cast<double>(rows) * cols * 4.0 + static_cast<double>(cols) * 2.0;
    const double bytes32 = static_cast<double>(rows) * cols * 8.0 + static_cast<double>(cols) * 4.0;
    std::printf("bench hidden=%d rows=%d iters=%d:\n", cols, rows, iters);
    std::printf("  BF16: %.4f ms/iter  %.1f GB/s\n", msbf / iters,
                bytes16 / (msbf / 1000.0 / iters) / 1e9);
    std::printf("  FP16: %.4f ms/iter  %.1f GB/s (byte-parity sanity)\n", ms16 / iters,
                bytes16 / (ms16 / 1000.0 / iters) / 1e9);
    std::printf("  FP32: %.4f ms/iter  %.1f GB/s\n", ms32 / iters,
                bytes32 / (ms32 / 1000.0 / iters) / 1e9);
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("SKIP: no CUDA device available (test_rms_norm_bf16)\n");
        return 0;
    }

    std::printf("[6-B-2] BF16 RMSNorm correctness vs FP32 reference\n");
    int failures = run_correctness();
    std::printf("[6-B-2] single-node graph smoke (BF16 dst injection)\n");
    failures += run_graph_smoke();
    if (failures == 0) {
        std::printf("[6-B-2] micro-bench (BF16 vs FP32 + FP16 sanity, two regimes)\n");
        run_bench(1, 4096, 1e-5f, 50);    // launch-bound
        run_bench(512, 8192, 1e-5f, 50);  // bandwidth-bound
        std::printf("PASS rms-norm-bf16 (Phase 6-B-2)\n");
        return 0;
    }
    std::printf("FAILURES: %d\n", failures);
    return 1;
}
