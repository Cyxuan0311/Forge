// =========================================================================
// test_matmul_fp16.cpp -- Phase 6-A-5: Native FP16 Matmul
// correctness + micro-benchmark (requires CUDA; skips gracefully w/o GPU).
//
// 6-A-5.1 (direct launch): FP16 A + FP16 B -> cuBLAS -> FP16 C, FP32 accumulate.
// 6-A-5.2 (route):        ops::matmul_transB(FP16, FP16) -> FP16 out, routed to
//                         the native kernel; FP32 / quantized paths untouched.
//
// "No implicit FP32" guarantee is checked structurally: the native route returns
// an FP16 tensor (out dtype == FP16) and feeds __half buffers straight into cuBLAS
// with no FP32 staging tensor. Correctness is checked against the verified FP32
// matmul on the *same decoded FP16 inputs* -- the only residual is the FP16 output
// store rounding (native accumulates in FP32, CUBLAS_COMPUTE_32F).
// =========================================================================

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <cstring>
#include <random>
#include <vector>

#include "forge/cuda_kernels.h"
#include "forge/dtype_contract.h"
#include "forge/operator_matmul.h"
#include "forge/tensor.h"
#include "forge/types.h"

namespace {

using forge::DataType;
using forge::DeviceType;
using forge::Tensor;
using forge::TensorPtr;
using namespace forge::cuda;  // launch_cublas_gemm_fp16 / launch_cublas_sgemm

// ---- fp16 codec (matches the kernel storage convention) ----------------
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

// Host fp16 buffer -> CUDA FP16 tensor.
TensorPtr make_dev_fp16(const std::vector<uint16_t>& h, int64_t r, int64_t c) {
    auto host = std::make_shared<Tensor>(DataType::FP16, shape2(r, c), DeviceType::CPU);
    std::memcpy(host->data(), h.data(), h.size() * 2);
    auto dev = std::make_shared<Tensor>(DataType::FP16, shape2(r, c), DeviceType::CUDA);
    dev->copy_from(*host);
    return dev;
}
// Host fp32 buffer -> CPU FP32 tensor (used for the verified reference).
TensorPtr make_host_fp32(const std::vector<float>& h, int64_t r, int64_t c) {
    auto t = std::make_shared<Tensor>(DataType::FP32, shape2(r, c), DeviceType::CPU);
    std::memcpy(t->data(), h.data(), h.size() * 4);
    return t;
}

// FP16 uniform random in [-1,1], returns the fp16 codes and the fp32-decoded
// values (so the reference reflects exactly the fp16 input rounding).
void rand_fp16_matrix(int64_t r, int64_t c, std::mt19937& g, std::vector<uint16_t>& codes,
                      std::vector<float>& decoded) {
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    int64_t n = r * c;
    codes.resize(n);
    decoded.resize(n);
    for (int64_t i = 0; i < n; ++i) {
        float v = d(g);
        codes[i] = f32_to_f16(v);
        decoded[i] = f16_to_f32(codes[i]);
    }
}

int failures = 0;
#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);   \
            ++failures;                                                    \
        }                                                                 \
    } while (0)

// Compare a native FP16 result (decoded) to the FP32 reference; the only
// residual is the FP16 output store rounding (native accumulates in FP32).
void compare_to_fp32_ref(const char* name, const std::vector<uint16_t>& got_fp16,
                         const std::vector<float>& ref_fp32) {
    float max_abs = 0, max_rel = 0;
    for (size_t i = 0; i < ref_fp32.size(); ++i) {
        float got = f16_to_f32(got_fp16[i]);
        float diff = std::fabs(got - ref_fp32[i]);
        if (diff > max_abs)
            max_abs = diff;
        if (std::fabs(ref_fp32[i]) > 0.1f) {
            float re = diff / std::fabs(ref_fp32[i]);
            if (re > max_rel)
                max_rel = re;
        }
    }
    std::printf("  %-22s : max_abs=%.3e max_rel=%.3e\n", name, max_abs, max_rel);
    CHECK(max_abs < 0.05f);
    CHECK(max_rel < 0.03f);
}

// ---- 6-A-5.1: direct native FP16 GEMM launch --------------------------
void check_native_launch(int M, int K, int N, std::mt19937& g) {
    std::vector<uint16_t> a_code, b_code;
    std::vector<float> a_dec, b_dec;
    rand_fp16_matrix(M, K, g, a_code, a_dec);
    rand_fp16_matrix(N, K, g, b_code, b_dec);

    auto a_dev = make_dev_fp16(a_code, M, K);
    auto b_dev = make_dev_fp16(b_code, N, K);

    auto c_dev = std::make_shared<Tensor>(DataType::FP16, shape2(M, N), DeviceType::CUDA);
    launch_cublas_gemm_fp16(static_cast<const __half*>(a_dev->data()),
                                  static_cast<const __half*>(b_dev->data()),
                                  static_cast<__half*>(c_dev->data()), M, K, N, /*transB=*/true, 0);

    auto c_host = std::make_shared<Tensor>(DataType::FP16, shape2(M, N), DeviceType::CPU);
    c_host->copy_from(*c_dev);
    const uint16_t* c_out = static_cast<const uint16_t*>(c_host->data());
    std::vector<uint16_t> got(c_out, c_out + (size_t)M * N);

    // Verified FP32 reference on the same decoded FP16 inputs (CPU sgemm).
    auto a_ref = make_host_fp32(a_dec, M, K);
    auto b_ref = make_host_fp32(b_dec, N, K);
    auto c_ref = forge::ops::matmul_transB(a_ref, b_ref);
    const float* ref = static_cast<const float*>(c_ref->data());
    std::vector<float> ref_vec(ref, ref + (size_t)M * N);

    char tag[64];
    std::snprintf(tag, sizeof(tag), "native_launch M=%d", M);
    compare_to_fp32_ref(tag, got, ref_vec);
}

// ---- 6-A-5.2: ops::matmul_transB route (FP16 in -> FP16 out) ----------
void check_route(int M, int K, int N, std::mt19937& g) {
    std::vector<uint16_t> a_code, b_code;
    std::vector<float> a_dec, b_dec;
    rand_fp16_matrix(M, K, g, a_code, a_dec);
    rand_fp16_matrix(N, K, g, b_code, b_dec);

    auto a_dev = make_dev_fp16(a_code, M, K);
    auto b_dev = make_dev_fp16(b_code, N, K);

    // The whole point: input + weight are FP16, and the op must stay FP16.
    CHECK(a_dev->dtype() == DataType::FP16);
    CHECK(b_dev->dtype() == DataType::FP16);

    auto out = forge::ops::matmul_transB(a_dev, b_dev);

    // "No implicit FP32" structural guarantee: the result is a real FP16 tensor.
    CHECK(out->dtype() == DataType::FP16);
    if (out->dtype() != DataType::FP16) {
        std::printf("  FAIL route M=%d: output dtype is not FP16\n", M);
        return;
    }

    auto out_host = std::make_shared<Tensor>(DataType::FP16, shape2(M, N), DeviceType::CPU);
    out_host->copy_from(*out);
    const uint16_t* c_out = static_cast<const uint16_t*>(out_host->data());
    std::vector<uint16_t> got(c_out, c_out + (size_t)M * N);

    auto a_ref = make_host_fp32(a_dec, M, K);
    auto b_ref = make_host_fp32(b_dec, N, K);
    auto c_ref = forge::ops::matmul_transB(a_ref, b_ref);
    const float* ref = static_cast<const float*>(c_ref->data());
    std::vector<float> ref_vec(ref, ref + (size_t)M * N);

    char tag[64];
    std::snprintf(tag, sizeof(tag), "route M=%d", M);
    compare_to_fp32_ref(tag, got, ref_vec);
}

// ---- micro-bench: native FP16 cuBLAS vs FP32 cuBLAS, M sweep ----------
void bench_matmul(int M, int K, int N, int iters) {
    auto a16 = std::make_shared<Tensor>(DataType::FP16, shape2(M, K), DeviceType::CUDA);
    auto b16 = std::make_shared<Tensor>(DataType::FP16, shape2(N, K), DeviceType::CUDA);
    auto c16 = std::make_shared<Tensor>(DataType::FP16, shape2(M, N), DeviceType::CUDA);
    auto a32 = std::make_shared<Tensor>(DataType::FP32, shape2(M, K), DeviceType::CUDA);
    auto b32 = std::make_shared<Tensor>(DataType::FP32, shape2(N, K), DeviceType::CUDA);
    auto c32 = std::make_shared<Tensor>(DataType::FP32, shape2(M, N), DeviceType::CUDA);

    std::vector<uint16_t> fill16((size_t)M * K, f32_to_f16(0.5f));
    std::vector<uint16_t> fillb16((size_t)N * K, f32_to_f16(0.5f));
    std::vector<float> fill32((size_t)M * K, 0.5f);
    std::vector<float> fillb32((size_t)N * K, 0.5f);
    auto ha16 = std::make_shared<Tensor>(DataType::FP16, shape2(M, K), DeviceType::CPU);
    auto hb16 = std::make_shared<Tensor>(DataType::FP16, shape2(N, K), DeviceType::CPU);
    auto ha32 = std::make_shared<Tensor>(DataType::FP32, shape2(M, K), DeviceType::CPU);
    auto hb32 = std::make_shared<Tensor>(DataType::FP32, shape2(N, K), DeviceType::CPU);
    std::memcpy(ha16->data(), fill16.data(), fill16.size() * 2);
    std::memcpy(hb16->data(), fillb16.data(), fillb16.size() * 2);
    std::memcpy(ha32->data(), fill32.data(), fill32.size() * 4);
    std::memcpy(hb32->data(), fillb32.data(), fillb32.size() * 4);
    a16->copy_from(*ha16);
    b16->copy_from(*hb16);
    a32->copy_from(*ha32);
    b32->copy_from(*hb32);

    for (int i = 0; i < 3; ++i) {
        launch_cublas_gemm_fp16(static_cast<const __half*>(a16->data()),
                                      static_cast<const __half*>(b16->data()),
                                      static_cast<__half*>(c16->data()), M, K, N, true);
        launch_cublas_sgemm(static_cast<const float*>(a32->data()),
                                  static_cast<const float*>(b32->data()),
                                  static_cast<float*>(c32->data()), M, K, N, true);
    }
    cudaDeviceSynchronize();

    cudaEvent_t es, ee;
    cudaEventCreate(&es);
    cudaEventCreate(&ee);
    float ms16 = 0, ms32 = 0;
    cudaEventRecord(es);
    for (int i = 0; i < iters; ++i)
        launch_cublas_gemm_fp16(static_cast<const __half*>(a16->data()),
                                      static_cast<const __half*>(b16->data()),
                                      static_cast<__half*>(c16->data()), M, K, N, true);
    cudaEventRecord(ee);
    cudaEventSynchronize(ee);
    cudaEventElapsedTime(&ms16, es, ee);
    cudaEventRecord(es);
    for (int i = 0; i < iters; ++i)
        launch_cublas_sgemm(static_cast<const float*>(a32->data()),
                                  static_cast<const float*>(b32->data()),
                                  static_cast<float*>(c32->data()), M, K, N, true);
    cudaEventRecord(ee);
    cudaEventSynchronize(ee);
    cudaEventElapsedTime(&ms32, es, ee);
    cudaEventDestroy(es);
    cudaEventDestroy(ee);

    // Traffic: A,B read + C written. FP16 => 3 * 2B/elt; FP32 => 3 * 4B/elt.
    const double bytes16 = static_cast<double>(M) * (K + K + N) * 2.0;
    const double bytes32 = static_cast<double>(M) * (K + K + N) * 4.0;
    std::printf("bench M=%4d K=%d N=%d iters=%d : FP16 %.5f ms/iter %.1f GB/s | FP32 %.5f ms/iter %.1f GB/s\n",
                M, K, N, iters, ms16 / iters, bytes16 / (ms16 / 1000.0 / iters) / 1e9, ms32 / iters,
                bytes32 / (ms32 / 1000.0 / iters) / 1e9);
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("SKIP: no CUDA device available (test_matmul_fp16)\n");
        return 0;
    }

    std::printf("[6-A-5] Native FP16 Matmul correctness (vs FP32 reference)\n");
    std::mt19937 rng(0x6A51);

    std::printf("  6-A-5.1 native launch (FP16 A x FP16 B -> FP16 C, FP32 accumulate):\n");
    check_native_launch(1, 4096, 4096, rng);
    check_native_launch(8, 4096, 4096, rng);
    check_native_launch(64, 4096, 4096, rng);

    std::printf("  6-A-5.2 ops::matmul_transB route (FP16 in -> FP16 out):\n");
    check_route(1, 4096, 4096, rng);
    check_route(1, 256, 512, rng);  // 6-A-8 decode-graph dims (M=1, small K/N)
    check_route(8, 4096, 4096, rng);
    check_route(64, 4096, 4096, rng);

    if (failures == 0) {
        std::printf("[6-A-5] micro-bench (FP16 vs FP32 cuBLAS, M sweep; K=N=4096)\n");
        bench_matmul(1, 4096, 4096, 500);
        bench_matmul(8, 4096, 4096, 200);
        bench_matmul(32, 4096, 4096, 100);
        bench_matmul(128, 4096, 4096, 50);
        bench_matmul(512, 4096, 4096, 20);
        std::printf("PASS matmul-fp16 (Phase 6-A-5)\n");
        return 0;
    }
    std::printf("FAILURES: %d\n", failures);
    return 1;
}
