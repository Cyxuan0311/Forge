// =========================================================================
// test_layer_norm_fp32.cpp -- device LayerNorm correctness + micro-benchmark
// (requires CUDA; skips gracefully when no GPU is present).
//
// Under test: forge::ops::layer_norm on a CUDA FP32 tensor, which now routes to
// cuda::launch_layer_norm instead of copying x to the host. The kernel does the
// same two-pass arithmetic as the CPU reference (mean, then variance around
// that mean, then (x-mean)*rsqrt(var/cols+eps)*w+b), so the only expected
// difference is summation order -- tree reduction on the device vs a serial
// loop on the host.
//
// Correctness gate: cosine similarity against a double-precision reference
// (primary, catches structural defects like a wrong axis or a dropped
// reduction) plus max-abs on the normalized output (catches scale errors).
// The last case deliberately runs x on the device with weight/bias left on the
// host, which is the staging path ops::layer_norm has to support.
//
// Shapes cover the vision-tower regime (many rows x 1024 cols) that motivated
// the kernel, plus a few odd widths so the strided/blocked loops are exercised.
// =========================================================================

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "forge/operator_norm.h"
#include "forge/tensor.h"
#include "forge/types.h"

namespace {

using forge::DataType;
using forge::DeviceType;
using forge::Tensor;
using forge::TensorPtr;

constexpr float kEps = 1e-6f;

// Double-precision reference: independent of both implementations.
void reference_layer_norm(const std::vector<float>& x, const std::vector<float>& w,
                          const std::vector<float>& b, int rows, int cols, float eps,
                          std::vector<float>& out) {
    out.assign(static_cast<size_t>(rows) * cols, 0.0f);
    for (int r = 0; r < rows; ++r) {
        const float* xr = x.data() + static_cast<size_t>(r) * cols;
        double mean = 0.0;
        for (int c = 0; c < cols; ++c)
            mean += xr[c];
        mean /= cols;
        double var = 0.0;
        for (int c = 0; c < cols; ++c) {
            const double d = xr[c] - mean;
            var += d * d;
        }
        var /= cols;
        const double inv = 1.0 / std::sqrt(var + eps);
        for (int c = 0; c < cols; ++c) {
            double v = (xr[c] - mean) * inv;
            if (!w.empty())
                v *= w[c];
            if (!b.empty())
                v += b[c];
            out[static_cast<size_t>(r) * cols + c] = static_cast<float>(v);
        }
    }
}

double cosine(const std::vector<float>& a, const std::vector<float>& b) {
    double da = 0.0, db = 0.0, dab = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        da += static_cast<double>(a[i]) * a[i];
        db += static_cast<double>(b[i]) * b[i];
        dab += static_cast<double>(a[i]) * b[i];
    }
    if (da <= 0.0 || db <= 0.0)
        return 0.0;
    return dab / (std::sqrt(da) * std::sqrt(db));
}

int failures = 0;

// bias_on_host selects the staging path: gamma/beta stay on the CPU while x is
// on the device.
void run_case(const char* tag, int rows, int cols, bool with_weight, bool with_bias,
              bool bias_on_host, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<float> x(static_cast<size_t>(rows) * cols);
    for (auto& v : x)
        v = dist(rng);
    std::vector<float> w, b;
    if (with_weight) {
        w.resize(cols);
        for (auto& v : w)
            v = dist(rng);
    }
    if (with_bias) {
        b.resize(cols);
        for (auto& v : b)
            v = dist(rng);
    }

    auto h_x = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{rows, cols},
                                        DeviceType::CPU);
    std::memcpy(h_x->data(), x.data(), x.size() * sizeof(float));
    auto d_x = std::make_shared<Tensor>(DataType::FP32, h_x->shape(), DeviceType::CUDA);
    d_x->copy_from(*h_x);

    auto make_param = [&](const std::vector<float>& v, DeviceType dev) -> TensorPtr {
        if (v.empty())
            return nullptr;
        auto h = std::make_shared<Tensor>(DataType::FP32,
                                          std::vector<int64_t>{static_cast<int64_t>(v.size())},
                                          DeviceType::CPU);
        std::memcpy(h->data(), v.data(), v.size() * sizeof(float));
        if (dev == DeviceType::CPU)
            return h;
        auto d = std::make_shared<Tensor>(DataType::FP32, h->shape(), DeviceType::CUDA);
        d->copy_from(*h);
        return d;
    };

    const DeviceType param_dev = bias_on_host ? DeviceType::CPU : DeviceType::CUDA;
    TensorPtr w_t = make_param(w, param_dev);
    TensorPtr b_t = make_param(b, param_dev);

    auto out = forge::ops::layer_norm(d_x, w_t, b_t, kEps);
    if (!out || out->device() != DeviceType::CUDA) {
        printf("  %-34s FAIL (output is not on the device)\n", tag);
        ++failures;
        return;
    }
    auto h_out = std::make_shared<Tensor>(DataType::FP32, out->shape(), DeviceType::CPU);
    h_out->copy_from(*out);
    const float* op = static_cast<const float*>(h_out->data());
    std::vector<float> got(op, op + static_cast<size_t>(rows) * cols);

    std::vector<float> want;
    reference_layer_norm(x, w, b, rows, cols, kEps, want);

    const double cos = cosine(got, want);
    double max_abs = 0.0;
    for (size_t i = 0; i < got.size(); ++i)
        max_abs = std::max(max_abs, std::fabs(static_cast<double>(got[i]) - want[i]));
    const bool ok = cos > 0.999999 && max_abs < 1e-4;
    printf("  %-34s cos=%.7f maxabs=%.3e  %s\n", tag, cos, max_abs, ok ? "ok" : "FAIL");
    if (!ok)
        ++failures;
}

void micro_bench(int rows, int cols) {
    std::vector<float> x(static_cast<size_t>(rows) * cols);
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = static_cast<float>((i % 97) - 48) * 0.01f;
    std::vector<float> w(cols, 1.0f), b(cols, 0.0f);

    auto h_x = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{rows, cols},
                                        DeviceType::CPU);
    std::memcpy(h_x->data(), x.data(), x.size() * sizeof(float));
    auto d_x = std::make_shared<Tensor>(DataType::FP32, h_x->shape(), DeviceType::CUDA);
    d_x->copy_from(*h_x);
    auto h_w = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{cols},
                                        DeviceType::CUDA);
    h_w->zero_();
    auto d_w = std::make_shared<Tensor>(DataType::FP32, h_w->shape(), DeviceType::CUDA);
    d_w->copy_from(*h_w);
    auto d_b = std::make_shared<Tensor>(DataType::FP32, h_w->shape(), DeviceType::CUDA);
    d_b->copy_from(*h_w);

    auto once = [&]() { forge::ops::layer_norm(d_x, d_w, d_b, kEps); };
    once();
    cudaDeviceSynchronize();
    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    const int iters = 50;
    cudaEventRecord(s);
    for (int i = 0; i < iters; ++i)
        once();
    cudaEventRecord(e);
    cudaDeviceSynchronize();
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, s, e);
    ms /= iters;
    // Three passes over x (mean, variance, write) plus the write of out.
    const double bytes = 4.0 * static_cast<double>(rows) * cols * 4.0;
    printf("  rows=%-5d cols=%-5d  %.3f ms/iter  (%.1f GB/s effective)\n", rows, cols, ms,
           bytes / (ms * 1e-3) / 1e9);
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        printf("SKIP: no CUDA device\n");
        return 0;
    }
    std::mt19937 rng(1234);

    printf("=== device layer_norm vs double-precision reference ===\n");
    run_case("ViT tower shape (2304x1024)", 2304, 1024, true, true, false, rng);
    run_case("small (8x16)", 8, 16, true, true, false, rng);
    run_case("odd width (37x131)", 37, 131, true, true, false, rng);
    run_case("no weight", 64, 256, false, false, false, rng);
    run_case("weight only", 64, 256, true, false, false, rng);
    run_case("bias only", 64, 256, false, true, false, rng);
    run_case("weight/bias left on the host", 128, 1024, true, true, true, rng);
    run_case("single row (1x4096)", 1, 4096, true, true, false, rng);
    run_case("wide single row (1x8192)", 1, 8192, true, true, false, rng);

    printf("\n=== kernel throughput ===\n");
    micro_bench(2304, 1024);
    micro_bench(2304, 2560);

    if (failures) {
        printf("layer-norm-fp32: FAIL (%d case(s))\n", failures);
        return 1;
    }
    printf("layer-norm-fp32: PASS\n");
    return 0;
}