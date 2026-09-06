// =========================================================================
// test_rms_norm_fp16.cpp -- Phase 6-A-2: native FP16 RMSNorm correctness +
// micro-benchmark (requires CUDA; skips gracefully when no GPU is present).
//
// Compute model under test (from the 6-A-2 plan):
//     FP16 x -> FP32 square / reduction -> rsqrt -> FP16/FP32 scale -> FP16 out
// i.e. a native FP16 *storage/arithmetic path* with an FP32 reduction, NOT
// "all intermediates in half". Numerical stability is preserved.
//
// Correctness: run the real CUDA kernel (via forge::ops::rms_norm, which now
// routes FP16 to launch_rms_norm_fp16) and compare against an FP32 CPU
// reference built from the *same decoded FP16 inputs* -- so the only residual
// error is the FP16 output store (plus a 1-ulp rsqrt-vs-1/sqrt detail), which
// is exactly what we want to bound. Structural defects (wrong axis, wrong
// weight, dropped reduction) produce O(1) garbage and fail the tolerance.
//
// Micro-bench: kernel-only launch timing of FP16 vs the existing FP32 kernel,
// reported as ms/iter and achieved bandwidth.
// =========================================================================

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

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

// ---- fp16 codec (matches the kernel's storage convention) ----------------
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

float urand(std::mt19937& g, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    return d(g);
}

// FP32 reference RMSNorm operating on already-decoded FP16 float values.
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
    if (t->dtype() == DataType::FP16) {
        std::vector<uint16_t> buf(n, f32_to_f16(v));
        std::memcpy(h->data(), buf.data(), n * 2);
    } else {
        std::vector<float> buf(n, v);
        std::memcpy(h->data(), buf.data(), n * 4);
    }
    t->copy_from(*h);
}

// ---- correctness: native FP16 RMSNorm vs FP32 reference -----------------
int run_correctness() {
    int failures = 0;
    std::mt19937 rng(0x5EED);

    // Sanity: 6-A-2 whitelisted RMS_NORM in the contract (ties to 6-A-1).
    if (!forge::op_supports_lowp(forge::OpType::RMS_NORM)) {
        std::printf("FAIL: op_supports_lowp(RMS_NORM) expected true in 6-A-2\n");
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
                b = f32_to_f16(urand(rng, -2.0f, 2.0f));
            for (auto& b : wb)
                b = f32_to_f16(urand(rng, 0.9f, 1.1f));

            auto x_host = std::make_shared<Tensor>(DataType::FP16, shape2(rows, cols), DeviceType::CPU);
            auto w_host = std::make_shared<Tensor>(DataType::FP16, shape2(1, cols), DeviceType::CPU);
            std::memcpy(x_host->data(), xb.data(), xb.size() * 2);
            std::memcpy(w_host->data(), wb.data(), wb.size() * 2);
            auto x_dev = std::make_shared<Tensor>(DataType::FP16, shape2(rows, cols), DeviceType::CUDA);
            auto w_dev = std::make_shared<Tensor>(DataType::FP16, shape2(1, cols), DeviceType::CUDA);
            x_dev->copy_from(*x_host);
            w_dev->copy_from(*w_host);

            // Native FP16 RMSNorm (weighted path).
            auto out_dev = forge::ops::rms_norm(x_dev, w_dev, 1e-5f);
            if (!out_dev || out_dev->dtype() != DataType::FP16) {
                std::printf("FAIL: FP16 RMSNorm returned wrong dtype\n");
                return 1;
            }
            auto out_host = std::make_shared<Tensor>(DataType::FP16, shape2(rows, cols), DeviceType::CPU);
            out_host->copy_from(*out_dev);

            // Reference from the same decoded FP16 inputs.
            std::vector<float> xf(static_cast<size_t>(rows) * cols);
            std::vector<float> wf(cols);
            std::vector<float> ref(xf.size());
            for (size_t i = 0; i < xf.size(); ++i)
                xf[i] = f16_to_f32(xb[i]);
            for (int c = 0; c < cols; ++c)
                wf[c] = f16_to_f32(wb[c]);
            rms_norm_ref(xf, wf, rows, cols, 1e-5f, ref);

            const uint16_t* out_bits = static_cast<const uint16_t*>(out_host->data());
            float max_abs = 0.0f, max_rel = 0.0f;
            for (size_t i = 0; i < ref.size(); ++i) {
                float got = f16_to_f32(out_bits[i]);
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
                std::printf("    FAIL max_abs_err too large (fp16 store rounding only)\n");
                ++failures;
            }
            if (max_rel > 0.05f) {
                std::printf("    FAIL max_rel_err too large\n");
                ++failures;
            }
        }
    }

    // Unweighted FP16 path (kernel uses an internal ones weight).
    {
        const int rows = 64, cols = 4096;
        std::vector<uint16_t> xb(static_cast<size_t>(rows) * cols);
        for (auto& b : xb)
            b = f32_to_f16(urand(rng, -2.0f, 2.0f));
        auto x_host = std::make_shared<Tensor>(DataType::FP16, shape2(rows, cols), DeviceType::CPU);
        std::memcpy(x_host->data(), xb.data(), xb.size() * 2);
        auto x_dev = std::make_shared<Tensor>(DataType::FP16, shape2(rows, cols), DeviceType::CUDA);
        x_dev->copy_from(*x_host);

        auto out_dev = forge::ops::rms_norm(x_dev, nullptr, 1e-5f);
        if (!out_dev || out_dev->dtype() != DataType::FP16) {
            std::printf("FAIL: unweighted FP16 RMSNorm returned wrong dtype\n");
            return 1;
        }
        auto out_host = std::make_shared<Tensor>(DataType::FP16, shape2(rows, cols), DeviceType::CPU);
        out_host->copy_from(*out_dev);

        std::vector<float> xf(xb.size()), ref(xb.size());
        for (size_t i = 0; i < xf.size(); ++i)
            xf[i] = f16_to_f32(xb[i]);
        std::vector<float> wf(cols, 1.0f);
        rms_norm_ref(xf, wf, rows, cols, 1e-5f, ref);

        const uint16_t* out_bits = static_cast<const uint16_t*>(out_host->data());
        float max_abs = 0.0f;
        for (size_t i = 0; i < ref.size(); ++i) {
            float diff = std::fabs(f16_to_f32(out_bits[i]) - ref[i]);
            if (diff > max_abs)
                max_abs = diff;
        }
        std::printf("  unweighted hidden=%5d rows=%4d : max_abs_err=%.3e\n", cols, rows, max_abs);
        if (max_abs > 0.02f) {
            std::printf("    FAIL unweighted max_abs_err too large\n");
            ++failures;
        }
    }

    return failures;
}

// ---- micro-bench: FP16 kernel vs FP32 kernel (kernel-only timing) -------
void run_bench(int rows, int cols, float eps, int iters) {
    auto x16 = std::make_shared<Tensor>(DataType::FP16, shape2(rows, cols), DeviceType::CUDA);
    auto w16 = std::make_shared<Tensor>(DataType::FP16, shape2(1, cols), DeviceType::CUDA);
    auto o16 = std::make_shared<Tensor>(DataType::FP16, shape2(rows, cols), DeviceType::CUDA);
    auto x32 = std::make_shared<Tensor>(DataType::FP32, shape2(rows, cols), DeviceType::CUDA);
    auto w32 = std::make_shared<Tensor>(DataType::FP32, shape2(1, cols), DeviceType::CUDA);
    auto o32 = std::make_shared<Tensor>(DataType::FP32, shape2(rows, cols), DeviceType::CUDA);
    fill_const(x16, 0.5f);
    fill_const(w16, 1.0f);
    fill_const(x32, 0.5f);
    fill_const(w32, 1.0f);

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
    float ms16 = 0.0f, ms32 = 0.0f;

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
    std::printf("  FP16: %.4f ms/iter  %.1f GB/s\n", ms16 / iters,
                bytes16 / (ms16 / 1000.0 / iters) / 1e9);
    std::printf("  FP32: %.4f ms/iter  %.1f GB/s\n", ms32 / iters,
                bytes32 / (ms32 / 1000.0 / iters) / 1e9);
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("SKIP: no CUDA device available (test_rms_norm_fp16)\n");
        return 0;
    }

    std::printf("[6-A-2] FP16 RMSNorm correctness vs FP32 reference\n");
    int failures = run_correctness();
    if (failures == 0) {
        std::printf("[6-A-2] micro-bench (FP16 vs FP32)\n");
        run_bench(512, 4096, 1e-5f, 50);
        run_bench(512, 8192, 1e-5f, 50);
        std::printf("PASS rms-norm-fp16 (Phase 6-A-2)\n");
        return 0;
    }
    std::printf("FAILURES: %d\n", failures);
    return 1;
}
