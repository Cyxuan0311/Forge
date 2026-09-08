// Phase 6-C-4.2: M==1 FP8 E4M3 GEMV — standalone benchmark + correctness audit.
//
// Independent of the 6-C-4.1 CUTLASS GEMM path (hand-written kernel). Does NOT
// wire the KernelSelector (that happens only after this audit passes).
//
// Contract under test:
//   M == 1, A: 1xK fp8_e4m3, W: transB weight, D[n] = sum_k decode(A[k]) *
//   decode(W[n,k]) * (s_a * s_w), FP32 accumulate.
//
// Audited:
//   * correctness   : golden = decode(A_fp8) x decode(W_fp8); cosine + max abs err
//   * scale equiv   : (s_a*s_w) fused inside vs applied after the sum (ULP check)
//   * no OOB        : run under compute-sanitizer --tool memcheck
//   * M=1 latency   : us/token, effective bandwidth, at decode model shapes
//   * FP8 vs FP16   : same kernel structure, 1:2 element width
//
// LAYOUT NOTE (6-C-5.6 correctness fix): the INLINE audit kernels benchmark VEC
// widths on an ARTIFICIAL k-major weight layout (W[k*N + j]) -- that VEC sweep was
// the 6-C-4.2 bandwidth experiment. The PRODUCTION backend
// (forge::cuda::launch_fp8_gemv_e4m3_fp32) consumes the REAL Forge transB layout --
// row-major [N, K] with row n contiguous, W[n*K + k], the exact buffer used by the
// FP32 GEMV, the CUTLASS FP8 GEMM and the CPU reference -- so the "PROD" runs in
// main() feed a row-major weight and its own golden. A single weight Tensor must
// stay correct for both M == 1 (GEMV) and M > 1 (GEMM), which only holds in the
// row-major form.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <string>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "forge/fp8_utils.h"
#include "forge/fp8_gemv.h"
#include "forge/kernel_selector.h"
#include "forge/tensor.h"
#include "forge/fp8_quant.h"

using namespace forge;

// ---------------------------------------------------------------------------
// Device decode + vectorized load traits
// ---------------------------------------------------------------------------
template <typename T>
struct Decode {
    __device__ static float f(T x);
};
template <>
__device__ inline float Decode<uint8_t>::f(uint8_t x) {
    return fp8_e4m3_to_fp32(x);
}
template <>
__device__ inline float Decode<__half>::f(__half x) {
    return __half2float(x);
}
template <>
__device__ inline float Decode<float>::f(float x) {
    return x;
}

// Vector load: for a fixed k, a thread owns VEC consecutive output columns, so a
// single vector load of VEC elements (16 bytes) is naturally coalesced across
// the warp. VEC = 16 (fp8), 8 (fp16), 4 (fp32); VEC == 1 is the scalar path.
template <typename T, int V>
struct VecLoad {
    using type = T;
    __device__ static T load(const T* p) { return p[0]; }
};
template <>
struct VecLoad<uint8_t, 16> {
    using type = uint4;
    __device__ static uint4 load(const uint8_t* p) { return *reinterpret_cast<const uint4*>(p); }
};
template <>
struct VecLoad<__half, 8> {
    using type = uint4;
    __device__ static uint4 load(const __half* p) { return *reinterpret_cast<const uint4*>(p); }
};
template <>
struct VecLoad<float, 4> {
    using type = uint4;
    __device__ static uint4 load(const float* p) { return *reinterpret_cast<const uint4*>(p); }
};

// FP8 E4M3 GEMV with a shared-memory 256-entry decode LUT (value = fp8_e4m3_to_fp32
// byte, so it is bit-identical to the host golden). The LUT read is a single
// branchless shared load, removing the decode from the critical path; the kernel
// then becomes memory-bound and the 1-byte payload's bandwidth advantage shows.
// VELEM = number of consecutive output columns handled per thread (vectorized W
// load). 1 == scalar, 16 == 16-byte vector load.
template <int VELEM>
__global__ void gemv_fp8(float* __restrict__ D, const uint8_t* __restrict__ A,
                         const uint8_t* __restrict__ W, int K, int N, float s, int mode,
                         const float* __restrict__ lut) {
    extern __shared__ __align__(16) char smem[];
    float* LUT = reinterpret_cast<float*>(smem);
    uint8_t* Ash = reinterpret_cast<uint8_t*>(smem + 256 * sizeof(float));
    const int tid = threadIdx.x;

    for (int i = tid; i < 256; i += blockDim.x) LUT[i] = lut[i];
    for (int k = tid; k < K; k += blockDim.x) Ash[k] = A[k];
    __syncthreads();

    const int base = blockIdx.x * blockDim.x * VELEM + tid * VELEM;
    if (base >= N) return;

    float acc[VELEM];
#pragma unroll
    for (int v = 0; v < VELEM; ++v) acc[v] = 0.f;

    for (int k = 0; k < K; ++k) {
        const float af = LUT[Ash[k]];
        const uint8_t* wp = W + (size_t)k * N + base;
        uint8_t buf[VELEM];
        if (VELEM == 16) {
            *reinterpret_cast<uint4*>(buf) = *reinterpret_cast<const uint4*>(wp);
        } else if (VELEM == 4) {
            *reinterpret_cast<uint32_t*>(buf) = *reinterpret_cast<const uint32_t*>(wp);
        } else {
            buf[0] = wp[0];
        }
#pragma unroll
        for (int v = 0; v < VELEM; ++v) {
            const float wf = LUT[buf[v]];
            const float prod = af * wf;
            acc[v] += (mode == 1) ? prod * s : prod;
        }
    }
    if (mode == 0) {
#pragma unroll
        for (int v = 0; v < VELEM; ++v) acc[v] *= s;
    }
#pragma unroll
    for (int v = 0; v < VELEM; ++v) {
        const int j = base + v;
        if (j < N) D[j] = acc[v];
    }
}

template <typename InT, int VEC>
__global__ void gemv_tmpl(float* __restrict__ D, const InT* __restrict__ A,
                          const InT* __restrict__ W, int K, int N, float s, int mode) {
    extern __shared__ __align__(16) char smem[];
    InT* Ash = reinterpret_cast<InT*>(smem);
    const int tid = threadIdx.x;

    // Cooperative load of A (length K) into shared memory (reused over K steps).
    for (int k = tid; k < K; k += blockDim.x) Ash[k] = A[k];
    __syncthreads();

    const int base = blockIdx.x * blockDim.x * VEC + tid * VEC;
    if (base >= N) return;

    float acc[VEC];
#pragma unroll
    for (int v = 0; v < VEC; ++v) acc[v] = 0.f;

    for (int k = 0; k < K; ++k) {
        const float af = Decode<InT>::f(Ash[k]);
        const InT* wp = W + (size_t)k * N + base;
        typename VecLoad<InT, VEC>::type packed = VecLoad<InT, VEC>::load(wp);
        InT wvec[VEC];
        *reinterpret_cast<typename VecLoad<InT, VEC>::type*>(wvec) = packed;
#pragma unroll
        for (int v = 0; v < VEC; ++v) {
            const float wf = Decode<InT>::f(wvec[v]);
            const float prod = af * wf;
            acc[v] += (mode == 1) ? prod * s : prod;
        }
    }
    if (mode == 0) {
#pragma unroll
        for (int v = 0; v < VEC; ++v) acc[v] *= s;
    }
#pragma unroll
    for (int v = 0; v < VEC; ++v) {
        const int j = base + v;
        if (j < N) D[j] = acc[v];
    }
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------
static bool device_ok() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return false;
    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);
    return (major * 10 + minor) >= 89;
}

struct RunResult {
    double us_per_token;
    double bandwidth_gbs;
    float max_abs_err;
    float cosine;
    float scale_max_rel;
};

template <typename InT, int VEC>
static RunResult run_one(int K, int N, const std::vector<InT>& hA, const std::vector<InT>& hW,
                         const std::vector<float>& golden, float s, bool check) {
    RunResult r{0, 0, 0, 0, 0};
    const size_t Wbytes = (size_t)K * N * sizeof(InT);
    const size_t Abytes = (size_t)K * sizeof(InT);
    const size_t Dbytes = (size_t)N * sizeof(float);

    InT *dA = nullptr, *dW = nullptr;
    float* dD = nullptr;
    cudaMalloc(&dA, Abytes);
    cudaMalloc(&dW, Wbytes);
    cudaMalloc(&dD, Dbytes);
    cudaMemcpy(dA, hA.data(), Abytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dW, hW.data(), Wbytes, cudaMemcpyHostToDevice);

    const int BDIM = 256;
    const int grid = (N + BDIM * VEC - 1) / (BDIM * VEC);
    const size_t shm = (size_t)K * sizeof(InT);

    // --- correctness: both scale modes ---
    std::vector<float> hD(N);
    if (check) {
        gemv_tmpl<InT, VEC><<<grid, BDIM, shm>>>(dD, dA, dW, K, N, s, 0);
        cudaDeviceSynchronize();
        cudaMemcpy(hD.data(), dD, Dbytes, cudaMemcpyDeviceToHost);

        double dot = 0, ng = 0, nt = 0;
        float maxerr = 0;
        for (int j = 0; j < N; ++j) {
            const float d = hD[j], g = golden[j];
            maxerr = std::max(maxerr, std::fabs(d - g));
            dot += d * g;
            ng += d * d;
            nt += g * g;
        }
        r.max_abs_err = maxerr;
        r.cosine = (ng > 0 && nt > 0) ? (float)(dot / (std::sqrt(ng) * std::sqrt(nt))) : 0.f;

        // scale equivalence: mode 1 (fused inside) vs mode 0 (after)
        std::vector<float> hD2(N);
        gemv_tmpl<InT, VEC><<<grid, BDIM, shm>>>(dD, dA, dW, K, N, s, 1);
        cudaDeviceSynchronize();
        cudaMemcpy(hD2.data(), dD, Dbytes, cudaMemcpyDeviceToHost);
        float maxrel = 0;
        for (int j = 0; j < N; ++j) {
            const float denom = std::fabs(golden[j]) + 1e-6f;
            maxrel = std::max(maxrel, std::fabs(hD2[j] - hD[j]) / denom);
        }
        r.scale_max_rel = maxrel;
    }

    // --- timing (one launch = one token = one M==1 GEMV) ---
    const int warm = 20, iters = 200;
    for (int i = 0; i < warm; ++i)
        gemv_tmpl<InT, VEC><<<grid, BDIM, shm>>>(dD, dA, dW, K, N, s, 0);
    cudaDeviceSynchronize();

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0);
    for (int i = 0; i < iters; ++i)
        gemv_tmpl<InT, VEC><<<grid, BDIM, shm>>>(dD, dA, dW, K, N, s, 0);
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    float ms = 0;
    cudaEventElapsedTime(&ms, ev0, ev1);
    r.us_per_token = (double)ms * 1000.0 / iters;
    const double bytes = (double)Wbytes + (double)Abytes + (double)Dbytes;
    r.bandwidth_gbs = bytes / (r.us_per_token * 1e-6) / 1e9;

    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaFree(dA);
    cudaFree(dW);
    cudaFree(dD);
    return r;
}

// FP8 E4M3 runner with the shared-LUT kernel. `dLUT` is the device 256-entry
// decode table (value = fp8_e4m3_to_fp32(byte)). VELEM sweeps the vector width.
template <int VELEM>
static RunResult run_fp8(int K, int N, const std::vector<uint8_t>& hA, const std::vector<uint8_t>& hW,
                        const std::vector<float>& golden, float s, bool check, const float* dLUT) {
    RunResult r{0, 0, 0, 0, 0};
    const size_t Wbytes = (size_t)K * N;
    const size_t Abytes = (size_t)K;
    const size_t Dbytes = (size_t)N * sizeof(float);

    uint8_t *dA = nullptr, *dW = nullptr;
    float* dD = nullptr;
    cudaMalloc(&dA, Abytes);
    cudaMalloc(&dW, Wbytes);
    cudaMalloc(&dD, Dbytes);
    cudaMemcpy(dA, hA.data(), Abytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dW, hW.data(), Wbytes, cudaMemcpyHostToDevice);

    const int BDIM = 256;
    const int grid = (N + BDIM * VELEM - 1) / (BDIM * VELEM);
    const size_t shm = 256 * sizeof(float) + (size_t)K;  // LUT + A

    std::vector<float> hD(N);
    if (check) {
        gemv_fp8<VELEM><<<grid, BDIM, shm>>>(dD, dA, dW, K, N, s, 0, dLUT);
        cudaDeviceSynchronize();
        cudaMemcpy(hD.data(), dD, Dbytes, cudaMemcpyDeviceToHost);

        double dot = 0, ng = 0, nt = 0;
        float maxerr = 0;
        for (int j = 0; j < N; ++j) {
            const float d = hD[j], g = golden[j];
            maxerr = std::max(maxerr, std::fabs(d - g));
            dot += d * g;
            ng += d * d;
            nt += g * g;
        }
        r.max_abs_err = maxerr;
        r.cosine = (ng > 0 && nt > 0) ? (float)(dot / (std::sqrt(ng) * std::sqrt(nt))) : 0.f;

        std::vector<float> hD2(N);
        gemv_fp8<VELEM><<<grid, BDIM, shm>>>(dD, dA, dW, K, N, s, 1, dLUT);
        cudaDeviceSynchronize();
        cudaMemcpy(hD2.data(), dD, Dbytes, cudaMemcpyDeviceToHost);
        float maxrel = 0;
        for (int j = 0; j < N; ++j) {
            const float denom = std::fabs(golden[j]) + 1e-6f;
            maxrel = std::max(maxrel, std::fabs(hD2[j] - hD[j]) / denom);
        }
        r.scale_max_rel = maxrel;
    }

    const int warm = 20, iters = 200;
    for (int i = 0; i < warm; ++i)
        gemv_fp8<VELEM><<<grid, BDIM, shm>>>(dD, dA, dW, K, N, s, 0, dLUT);
    cudaDeviceSynchronize();

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0);
    for (int i = 0; i < iters; ++i)
        gemv_fp8<VELEM><<<grid, BDIM, shm>>>(dD, dA, dW, K, N, s, 0, dLUT);
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    float ms = 0;
    cudaEventElapsedTime(&ms, ev0, ev1);
    r.us_per_token = (double)ms * 1000.0 / iters;
    const double bytes = (double)Wbytes + (double)Abytes + (double)Dbytes;
    r.bandwidth_gbs = bytes / (r.us_per_token * 1e-6) / 1e9;

    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaFree(dA);
    cudaFree(dW);
    cudaFree(dD);
    return r;
}

// ---------------------------------------------------------------------------
// 6-C-5.2b: device scalar LIFETIME. The single most important property for graph
// reuse is: changing the host sidecar must NOT change the device address; and the
// address must survive move. (copy_from/view/slice create new Tensors, so they get
// their own address -- but the sidecar VALUE must propagate.)
// ---------------------------------------------------------------------------
static bool test_device_scale_lifetime() {
    bool ok = true;

    // (1) host value change keeps the same device address
    auto a = std::make_shared<Tensor>(DataType::FP8_E4M3, std::vector<int64_t>{1024},
                                      DeviceType::CUDA);
    a->set_fp8_scale(0.02f);
    const float* p0 = a->fp8_scale_device();
    a->set_fp8_scale(0.04f);
    const float* p1 = a->fp8_scale_device();
    if (!p0 || p0 != p1) {
        printf("  [FAIL] host scale change moved the device address (%p -> %p)\n", (void*)p0,
               (void*)p1);
        ok = false;
    }

    // (2) move transfers ownership: the SAME address is served by the moved-to tensor
    auto b = std::make_shared<Tensor>(DataType::FP8_E4M3, std::vector<int64_t>{1024},
                                      DeviceType::CUDA);
    b->set_fp8_scale(0.03f);
    const float* pb = b->fp8_scale_device();
    Tensor moved(std::move(*b));
    const float* pm = moved.fp8_scale_device();
    if (!pb || pm != pb) {
        printf("  [FAIL] move did not preserve the device address (%p -> %p)\n", (void*)pb,
               (void*)pm);
        ok = false;
    }

    // (3) copy_from / view / slice propagate the sidecar VALUE (new tensor, own address)
    auto src = std::make_shared<Tensor>(DataType::FP8_E4M3, std::vector<int64_t>{1024},
                                        DeviceType::CUDA);
    src->set_fp8_scale(0.05f);
    auto dst = std::make_shared<Tensor>(DataType::FP8_E4M3, std::vector<int64_t>{1024},
                                        DeviceType::CUDA);
    dst->copy_from(*src);
    auto v = src->view(std::vector<int64_t>{32, 32});
    auto sl = src->slice(0, 0, 16);
    if (!dst->has_fp8_scale() || dst->fp8_scale() != 0.05f) {
        printf("  [FAIL] copy_from did not propagate the sidecar (has=%d val=%g)\n",
               (int)dst->has_fp8_scale(), dst->fp8_scale());
        ok = false;
    }
    if (!v.has_fp8_scale() || v.fp8_scale() != 0.05f) {
        printf("  [FAIL] view did not propagate the sidecar\n");
        ok = false;
    }
    if (!sl.has_fp8_scale() || sl.fp8_scale() != 0.05f) {
        printf("  [FAIL] slice did not propagate the sidecar\n");
        ok = false;
    }

    if (ok)
        printf("  [OK] 6-C-5.2b lifetime: addr stable across mutation+move, sidecar propagates "
               "via copy_from/view/slice\n");
    return ok;
}

// ---------------------------------------------------------------------------
// 6-C-5.1a: W8A8 quantize producer. Device-side amax -> scale -> encode; the scale
// must land in the output Tensor's own device scalar (device-authoritative).
// ---------------------------------------------------------------------------
static bool test_quant_producer() {
    const int n = 4096;
    std::vector<float> hx(n);
    float amax = 0.f;
    for (int i = 0; i < n; ++i) {
        hx[i] = (i % 251) * 0.0137f - 1.5f;  // deterministic, signed, non-degenerate
        amax = std::max(amax, std::fabs(hx[i]));
    }
    auto src = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{n}, DeviceType::CUDA);
    cudaMemcpy(src->data(), hx.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    auto q = quantize_fp8_e4m3_w8a8(src, 0);
    if (!q) {
        printf("  [FAIL] quantize_fp8_e4m3_w8a8 returned null\n");
        return false;
    }
    if (q->dtype() != DataType::FP8_E4M3) {
        printf("  [FAIL] output dtype is not FP8_E4M3\n");
        return false;
    }
    if (!q->has_fp8_scale()) {
        printf("  [FAIL] output has no FP8 scale sidecar\n");
        return false;
    }
    if (!q->fp8_scale_device_ptr()) {
        printf("  [FAIL] output has no device scale\n");
        return false;
    }
    if (!q->fp8_scale_device_authoritative()) {
        printf("  [FAIL] device scale is not marked authoritative (scale not computed on "
               "device)\n");
        return false;
    }

    // The device-computed scale must equal amax / 240 exactly as the host contract says.
    float dev_scale = 0.f;
    cudaMemcpy(&dev_scale, q->fp8_scale_device_ptr(), sizeof(float), cudaMemcpyDeviceToHost);
    const float want = amax / 240.0f;
    if (std::fabs(dev_scale - want) > 1e-6f * std::max(1.f, want)) {
        printf("  [FAIL] device scale %g != amax/240 %g (amax=%g)\n", dev_scale, want, amax);
        return false;
    }

    // Round-trip: decode(q) * scale must match the source within FP8 resolution.
    std::vector<uint8_t> hq(n);
    cudaMemcpy(hq.data(), q->data(), n, cudaMemcpyDeviceToHost);
    float maxerr = 0.f;
    for (int i = 0; i < n; ++i) {
        const float got = fp8_e4m3_to_fp32(hq[i]) * dev_scale;
        maxerr = std::max(maxerr, std::fabs(got - hx[i]));
    }
    // E4M3 has 3 mantissa bits: relative error <= 2^-4 of the local magnitude.
    const float tol = 0.07f * amax;
    if (maxerr > tol) {
        printf("  [FAIL] quantize round-trip maxerr=%g > tol=%g (amax=%g)\n", maxerr, tol, amax);
        return false;
    }

    printf("  [OK] 6-C-5.1a producer: dtype=FP8_E4M3 scale=%g (amax/240=%g) maxerr=%g\n", dev_scale,
           want, maxerr);
    return true;
}

// ---------------------------------------------------------------------------
// 6-C-5.1b: amax contract -- amax>0 -> amax/240; amax==0 -> scale 1 and all-zero
// output; NaN/Inf -> rejected (scale NaN), NOT silently swallowed.
// ---------------------------------------------------------------------------
static bool test_quant_edge_cases() {
    bool ok = true;

    // Host-side contract reference.
    struct Case {
        float amax;
        Fp8AmaxStatus want;
    };
    const Case cases[] = {{2.5f, Fp8AmaxStatus::Ok},
                          {0.0f, Fp8AmaxStatus::Zero},
                          {std::numeric_limits<float>::quiet_NaN(), Fp8AmaxStatus::NonFinite},
                          {std::numeric_limits<float>::infinity(), Fp8AmaxStatus::NonFinite}};
    for (const auto& c : cases) {
        if (fp8_amax_status(c.amax) != c.want) {
            printf("  [FAIL] fp8_amax_status(%g) wrong\n", c.amax);
            ok = false;
        }
        const float s = fp8_w8a8_scale_from_amax(c.amax);
        if (c.want == Fp8AmaxStatus::Ok && std::fabs(s - c.amax / 240.f) > 1e-9f) {
            printf("  [FAIL] scale(%g) = %g, expected %g\n", c.amax, s, c.amax / 240.f);
            ok = false;
        }
        if (c.want == Fp8AmaxStatus::Zero && s != 1.f) {
            printf("  [FAIL] scale(0) = %g, expected 1\n", s);
            ok = false;
        }
        if (c.want == Fp8AmaxStatus::NonFinite && !std::isnan(s)) {
            printf("  [FAIL] scale(%g) = %g, expected NaN (reject)\n", c.amax, s);
            ok = false;
        }
    }

    // Device path, amax == 0: scale must be 1 and the FP8 payload all zeros.
    {
        const int n = 1024;
        std::vector<float> z(n, 0.f);
        auto src =
            std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{n}, DeviceType::CUDA);
        cudaMemcpy(src->data(), z.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        auto q = quantize_fp8_e4m3_w8a8(src, 0);
        float s = 0.f;
        cudaMemcpy(&s, q->fp8_scale_device_ptr(), sizeof(float), cudaMemcpyDeviceToHost);
        std::vector<uint8_t> hq(n, 0xAB);
        cudaMemcpy(hq.data(), q->data(), n, cudaMemcpyDeviceToHost);
        bool all_zero = true;
        for (int i = 0; i < n; ++i)
            if (hq[i] != 0) all_zero = false;
        if (s != 1.f || !all_zero) {
            printf("  [FAIL] amax==0: scale=%g all_zero=%d\n", s, (int)all_zero);
            ok = false;
        }
    }

    // Device path, NaN present: must be REJECTED (scale NaN), not silently ignored.
    // (fmaxf(NaN,x) == x, so this is exactly the case a naive reduction gets wrong.)
    {
        const int n = 1024;
        std::vector<float> v(n, 1.f);
        v[n / 2] = std::numeric_limits<float>::quiet_NaN();
        auto src =
            std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{n}, DeviceType::CUDA);
        cudaMemcpy(src->data(), v.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        auto q = quantize_fp8_e4m3_w8a8(src, 0);
        float s = 0.f;
        cudaMemcpy(&s, q->fp8_scale_device_ptr(), sizeof(float), cudaMemcpyDeviceToHost);
        if (!std::isnan(s)) {
            printf("  [FAIL] NaN input not rejected: scale=%g (expected NaN)\n", s);
            ok = false;
        }
    }

    if (ok) printf("  [OK] 6-C-5.1b edge cases: amax=0 -> scale 1 + zeros; NaN/Inf -> rejected\n");
    return ok;
}

// ---------------------------------------------------------------------------
// 6-C-5.3: THE decisive experiment. A captured CUDA graph must observe a NEW scale
// written into the SAME Tensor-owned device scalar -- no re-capture, no
// re-instantiation -- and must match an eager launch bitwise.
//   5.3a capture baseline (uses the Tensor-owned device slot, no host scalar)
//   5.3b mutate the same slot, replay the SAME graphExec, compare to eager
//   5.3c reverse check: unchanged -> same output; mutated -> different output
//   5.3d ownership move after capture must not invalidate the captured pointer
// ---------------------------------------------------------------------------
static bool test_graph_scale_mutation() {
    const int K = 4096;
    const int N = 4096;

    // deterministic pseudo-random FP8 payloads
    std::vector<uint8_t> hA(K);
    std::vector<uint8_t> hW((size_t)K * N);
    uint64_t seed = 0x9E3779B97F4A7C15ull;
    auto rnd = [&seed]() {
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<uint8_t>((seed >> 33) & 0xFF);
    };
    for (int i = 0; i < K; ++i) hA[i] = rnd();
    for (size_t i = 0; i < hW.size(); ++i) hW[i] = rnd();

    auto A = std::make_shared<Tensor>(DataType::FP8_E4M3, std::vector<int64_t>{K},
                                      DeviceType::CUDA);
    auto W = std::make_shared<Tensor>(DataType::FP8_E4M3, std::vector<int64_t>{N, K},
                                      DeviceType::CUDA);
    auto D = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{N}, DeviceType::CUDA);
    cudaMemcpy(A->data(), hA.data(), K, cudaMemcpyHostToDevice);
    cudaMemcpy(W->data(), hW.data(), hW.size(), cudaMemcpyHostToDevice);

    A->set_fp8_scale(0.02f);
    W->set_fp8_scale(0.03f);
    const float* pa0 = A->fp8_scale_device();
    const float* pw0 = W->fp8_scale_device();
    if (!pa0 || !pw0) {
        printf("  [FAIL] device scale slot is null\n");
        return false;
    }

    cuda::Fp8W8A8Scales sc;
    sc.scale_a = pa0;
    sc.scale_w = pw0;

    cudaStream_t s = nullptr;
    if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) {
        printf("  [FAIL] could not create a capture stream\n");
        return false;
    }

    const uint8_t* dA = static_cast<const uint8_t*>(A->data());
    const uint8_t* dW = static_cast<const uint8_t*>(W->data());
    float* dD = static_cast<float*>(D->data());

    // Warm-up OUTSIDE capture: this is what allocates the decoder LUT, so nothing
    // allocates during capture (the launcher itself is allocation-free, 6-C-5.2).
    if (!cuda::launch_fp8_gemv_e4m3_fp32(dA, dW, dD, K, N, sc, s)) {
        printf("  [FAIL] warm-up GEMV launch failed\n");
        cudaStreamDestroy(s);
        return false;
    }
    cudaStreamSynchronize(s);

    // ---- 5.3a: capture. Only the GEMV launch is captured; no H2D/D2H inside. ----
    cudaGraph_t graph = nullptr;
    cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed);
    cuda::launch_fp8_gemv_e4m3_fp32(dA, dW, dD, K, N, sc, s);
    cudaStreamEndCapture(s, &graph);
    if (!graph) {
        printf("  [FAIL] cudaStreamEndCapture produced no graph\n");
        cudaStreamDestroy(s);
        return false;
    }
    cudaGraphExec_t gexec = nullptr;
    if (cudaGraphInstantiate(&gexec, graph, 0) != cudaSuccess || !gexec) {
        printf("  [FAIL] cudaGraphInstantiate failed\n");
        cudaGraphDestroy(graph);
        cudaStreamDestroy(s);
        return false;
    }

    auto replay = [&](std::vector<float>& out) {
        cudaGraphLaunch(gexec, s);
        cudaStreamSynchronize(s);
        out.assign(N, 0.f);
        cudaMemcpy(out.data(), dD, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost);
    };

    std::vector<float> out1;
    replay(out1);

    // ---- 5.3b: mutate the SAME device slot. No re-capture, no re-instantiate. ----
    A->set_fp8_scale(0.04f);
    W->set_fp8_scale(0.05f);
    const float* pa1 = A->fp8_scale_device(s);  // H2D into the SAME address
    const float* pw1 = W->fp8_scale_device(s);
    cudaStreamSynchronize(s);                   // ensure the upload precedes the replay

    std::vector<float> out2;
    replay(out2);  // SAME gexec

    bool ok = true;
    if (pa1 != pa0 || pw1 != pw0) {
        printf("  [FAIL] device scale ptr changed after mutation (a %p->%p, w %p->%p)\n",
               (void*)pa0, (void*)pa1, (void*)pw0, (void*)pw1);
        ok = false;
    }

    // Eager reference at the MUTATED scales (same tensors, same device scalars).
    auto De = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{N}, DeviceType::CUDA);
    cuda::launch_fp8_gemv_e4m3_fp32(dA, dW, static_cast<float*>(De->data()), K, N, sc, s);
    cudaStreamSynchronize(s);
    std::vector<float> eager2(N);
    cudaMemcpy(eager2.data(), De->data(), (size_t)N * sizeof(float), cudaMemcpyDeviceToHost);

    int diff2 = 0;
    float maxd2 = 0.f;
    for (int i = 0; i < N; ++i) {
        if (out2[i] != eager2[i]) ++diff2;
        maxd2 = std::max(maxd2, std::fabs(out2[i] - eager2[i]));
    }
    if (diff2 != 0) {
        printf("  [FAIL] replay(mutated) != eager bitwise: %d/%d differ (max %g)\n", diff2, N,
               maxd2);
        ok = false;
    }

    // ---- 5.3c: reverse check -- the kernel must really read the device scale ----
    int same12 = 0;
    for (int i = 0; i < N; ++i)
        if (out1[i] == out2[i]) ++same12;
    if (same12 == N) {
        printf("  [FAIL] output identical after mutation: the replay is NOT reading the "
               "device scalar\n");
        ok = false;
    }
    std::vector<float> out3;
    replay(out3);  // no mutation in between -> must be bit-stable
    int diff23 = 0;
    for (int i = 0; i < N; ++i)
        if (out2[i] != out3[i]) ++diff23;
    if (diff23 != 0) {
        printf("  [FAIL] replay not deterministic without mutation (%d differ)\n", diff23);
        ok = false;
    }

    // ---- 5.3d: ownership move AFTER capture must not invalidate the captured ptr ----
    // (shared_ptr move: ownership transfer, the Tensor object itself is untouched, so
    // this exercises the ownership model without manufacturing a use-after-free.)
    auto A_moved = std::move(A);
    if (A_moved->fp8_scale_device() != pa0) {
        printf("  [FAIL] move invalidated the captured device scale ptr\n");
        ok = false;
    }
    A_moved->set_fp8_scale(0.08f);
    if (A_moved->fp8_scale_device(s) != pa0) {
        printf("  [FAIL] mutation after move changed the device scale ptr\n");
        ok = false;
    }
    cudaStreamSynchronize(s);
    std::vector<float> out4;
    replay(out4);
    // diff34 counts DIFFERING elements: the move+mutation must be observed, so a
    // full match (diff34 == 0) is the failure, not the success.
    int diff34 = 0;
    for (int i = 0; i < N; ++i)
        if (out3[i] != out4[i]) ++diff34;
    if (diff34 == 0) {
        printf("  [FAIL] replay after move+mutation did not observe the new scale\n");
        ok = false;
    }

    if (ok) {
        printf("  [OK] 6-C-5.3 graph mutation:\n");
        printf("       graphExec=%p (never re-instantiated)\n", (void*)gexec);
        printf("       scale_a_ptr=%p scale_w_ptr=%p (stable)\n", (void*)pa0, (void*)pw0);
        printf("       replay(0.02,0.03) captured; replay(0.04,0.05)==eager bitwise\n");
        printf("       mutation changes output (%d/%d unchanged), no-mutation replay stable\n",
               same12, N);
    }

    cudaGraphExecDestroy(gexec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(s);
    return ok;
}

// ---------------------------------------------------------------------------
// 6-C-5.5c: FP8 output numerical correctness across the value classes, plus the
// requirement that has_fp8_scale() / fp8_scale() / fp8_scale_device() agree.
// ---------------------------------------------------------------------------
static bool test_fp8_output_correctness() {
    bool ok = true;

    struct Case {
        const char* name;
        std::vector<float> data;
    };
    const int n = 512;
    std::vector<float> pos(n), neg(n), zeros(n, 0.f), sat(n), nans(n, 1.f), pinf(n, 1.f);
    for (int i = 0; i < n; ++i) {
        pos[i] = 0.001f * (i + 1);
        neg[i] = -0.001f * (i + 1);
        // saturation boundary: one value far above the rest forces heavy clipping of
        // the small ones, exercising the E4M3 max-normal (240) region.
        sat[i] = (i == 0) ? 1.0e6f : 0.5f;
    }
    nans[n / 2] = std::numeric_limits<float>::quiet_NaN();
    pinf[n / 2] = std::numeric_limits<float>::infinity();

    const Case cases[] = {{"normal+", pos},      {"normal-", neg}, {"all-zero", zeros},
                          {"saturation", sat},   {"NaN", nans},    {"+Inf", pinf}};

    for (const auto& c : cases) {
        auto src = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{n},
                                            DeviceType::CUDA);
        cudaMemcpy(src->data(), c.data.data(), (size_t)n * sizeof(float),
                   cudaMemcpyHostToDevice);

        auto out = quantize_fp8_e4m3_output(src, 0);
        if (out->dtype() != DataType::FP8_E4M3 || !out->has_fp8_scale() ||
            !out->fp8_scale_device_ptr()) {
            printf("  [FAIL] %s: output sidecar incomplete\n", c.name);
            ok = false;
            continue;
        }

        // Host mirror must agree with the device scalar after an explicit sync.
        if (!out->sync_fp8_scale_from_device()) {
            printf("  [FAIL] %s: sync_fp8_scale_from_device failed\n", c.name);
            ok = false;
            continue;
        }
        float dev_scale = 0.f;
        cudaMemcpy(&dev_scale, out->fp8_scale_device_ptr(), sizeof(float),
                   cudaMemcpyDeviceToHost);
        // NaN-aware: `NaN != NaN` is true, so two NaNs must be treated as agreeing.
        const bool both_nan = std::isnan(out->fp8_scale()) && std::isnan(dev_scale);
        if (!both_nan && out->fp8_scale() != dev_scale) {
            printf("  [FAIL] %s: host scale %g != device scale %g\n", c.name, out->fp8_scale(),
                   dev_scale);
            ok = false;
        }

        const bool expect_reject = (std::string(c.name) == "NaN" ||
                                    std::string(c.name) == "+Inf");
        if (expect_reject) {
            if (!std::isnan(dev_scale)) {
                printf("  [FAIL] %s: not rejected (scale=%g, expected NaN)\n", c.name, dev_scale);
                ok = false;
            }
            continue;  // no numeric check for rejected inputs
        }

        // decode(fp8) * scale must reproduce the FP32 source within E4M3 resolution.
        std::vector<uint8_t> hq(n);
        cudaMemcpy(hq.data(), out->data(), n, cudaMemcpyDeviceToHost);

        // finite amax of the source (NaN/Inf already handled above)
        float amax = 0.f;
        for (float v : c.data) amax = std::max(amax, std::fabs(v));
        const float want_scale = (amax > 0.f) ? amax / 240.0f : 1.f;
        if (std::fabs(dev_scale - want_scale) > 1e-6f * std::max(1.f, want_scale)) {
            printf("  [FAIL] %s: scale %g != amax/240 %g\n", c.name, dev_scale, want_scale);
            ok = false;
        }
        float maxerr = 0.f;
        for (int i = 0; i < n; ++i) {
            const float got = fp8_e4m3_to_fp32(hq[i]) * dev_scale;
            maxerr = std::max(maxerr, std::fabs(got - c.data[i]));
        }
        // E4M3 has 3 mantissa bits; allow 2^-3 relative to the local magnitude, and note
        // the saturation case intentionally clips (tol is relative to amax).
        const float tol = 0.07f * std::max(amax, 1e-6f);
        if (maxerr > tol) {
            printf("  [FAIL] %s: round-trip maxerr=%g > tol=%g\n", c.name, maxerr, tol);
            ok = false;
        }
        if (std::string(c.name) == "all-zero") {
            bool all_zero = true;
            for (int i = 0; i < n; ++i)
                if (hq[i] != 0) all_zero = false;
            if (dev_scale != 1.f || !all_zero) {
                printf("  [FAIL] all-zero: scale=%g all_zero=%d\n", dev_scale, (int)all_zero);
                ok = false;
            }
        }
    }

    if (ok)
        printf("  [OK] 6-C-5.5c output correctness: +/-, zero(->scale 1 + zeros), "
               "saturation, NaN/Inf rejected; host/device scale agree\n");
    return ok;
}

// ---------------------------------------------------------------------------
// 6-C-5.5d: THE decisive experiment for the OUTPUT side. The scale slot is written by
// the PRODUCER INSIDE the captured graph, not by the host. Changing the FP32 input must
// update the scale on replay, with no re-capture / re-instantiation.
// ---------------------------------------------------------------------------
static bool test_fp8_output_graph_mutation() {
    const int n = 8192;

    auto src = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{n},
                                        DeviceType::CUDA);
    auto dst = std::make_shared<Tensor>(DataType::FP8_E4M3, std::vector<int64_t>{n},
                                        DeviceType::CUDA);

    std::vector<float> d1(n), d2(n);
    for (int i = 0; i < n; ++i) {
        d1[i] = 0.001f * ((i % 97) + 1);        // amax ~ 0.097
        d2[i] = 0.050f * ((i % 97) + 1);        // amax ~ 4.85  -> very different scale
    }
    cudaMemcpy(src->data(), d1.data(), (size_t)n * sizeof(float), cudaMemcpyHostToDevice);

    cudaStream_t s = nullptr;
    if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) {
        printf("  [FAIL] could not create a capture stream\n");
        return false;
    }

    // Allocate the dst scale slot OUTSIDE capture (warm-up also sizes the amax scratch).
    float* slot0 = dst->fp8_scale_device_slot(s);
    if (!slot0) {
        printf("  [FAIL] could not allocate dst device scale slot\n");
        cudaStreamDestroy(s);
        return false;
    }
    quantize_fp8_e4m3_output_into(src, dst, s);  // warm-up outside capture
    cudaStreamSynchronize(s);

    // ---- capture the whole amax -> scale -> encode producer ----
    cudaGraph_t graph = nullptr;
    cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed);
    quantize_fp8_e4m3_output_into(src, dst, s);
    cudaStreamEndCapture(s, &graph);
    if (!graph) {
        printf("  [FAIL] output producer is not capturable\n");
        cudaStreamDestroy(s);
        return false;
    }
    cudaGraphExec_t gexec = nullptr;
    if (cudaGraphInstantiate(&gexec, graph, 0) != cudaSuccess) {
        printf("  [FAIL] cudaGraphInstantiate failed\n");
        cudaGraphDestroy(graph);
        cudaStreamDestroy(s);
        return false;
    }

    auto replay_and_read = [&](std::vector<uint8_t>& q, float& scale) {
        cudaGraphLaunch(gexec, s);
        cudaStreamSynchronize(s);
        q.assign(n, 0);
        cudaMemcpy(q.data(), dst->data(), n, cudaMemcpyDeviceToHost);
        cudaMemcpy(&scale, dst->fp8_scale_device_ptr(), sizeof(float), cudaMemcpyDeviceToHost);
    };

    std::vector<uint8_t> q1;
    float sc1 = 0.f;
    replay_and_read(q1, sc1);

    // ---- change D so amax (and therefore the scale) MUST change; no re-capture ----
    cudaMemcpy(src->data(), d2.data(), (size_t)n * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<uint8_t> q2;
    float sc2 = 0.f;
    replay_and_read(q2, sc2);  // SAME gexec

    bool ok = true;
    if (dst->fp8_scale_device_ptr() != slot0) {
        printf("  [FAIL] dst scale ptr changed (%p -> %p)\n", (void*)slot0,
               (void*)dst->fp8_scale_device_ptr());
        ok = false;
    }
    if (!(sc2 > sc1 * 2.0f)) {
        printf("  [FAIL] producer did not update the scale on replay (sc1=%g sc2=%g)\n", sc1,
               sc2);
        ok = false;
    }

    // eager reference at D2 must match the replay bitwise (bytes AND scale).
    auto dst_e = std::make_shared<Tensor>(DataType::FP8_E4M3, std::vector<int64_t>{n},
                                          DeviceType::CUDA);
    dst_e->fp8_scale_device_slot(s);
    quantize_fp8_e4m3_output_into(src, dst_e, s);
    cudaStreamSynchronize(s);
    std::vector<uint8_t> qe(n);
    cudaMemcpy(qe.data(), dst_e->data(), n, cudaMemcpyDeviceToHost);
    float sce = 0.f;
    cudaMemcpy(&sce, dst_e->fp8_scale_device_ptr(), sizeof(float), cudaMemcpyDeviceToHost);

    int bdiff = 0;
    for (int i = 0; i < n; ++i)
        if (q2[i] != qe[i]) ++bdiff;
    if (bdiff != 0 || sc2 != sce) {
        printf("  [FAIL] replay != eager (bytes differ %d/%d, scale %g vs %g)\n", bdiff, n, sc2,
               sce);
        ok = false;
    }

    if (ok) {
        printf("  [OK] 6-C-5.5d output graph mutation:\n");
        printf("       graphExec=%p (never re-instantiated), scale_ptr=%p (stable)\n",
               (void*)gexec, (void*)slot0);
        printf("       scale updated BY THE GRAPH: %g -> %g; replay == eager bitwise\n", sc1,
               sc2);
    }

    cudaGraphExecDestroy(gexec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(s);
    return ok;
}

// 6-C-5.2: the Tensor-owned device scale must be (a) non-null, (b) STABLE across calls
// (so a captured graph keeps reading one address), (c) equal to the host sidecar, and
// (d) updated in place at the SAME address after a mutation. (d) is the exact primitive
// 6-C-5.3 relies on: replay sees the new scale without rebuilding the graphExec.
static bool test_tensor_device_scale() {
    auto t = std::make_shared<Tensor>(DataType::FP8_E4M3, std::vector<int64_t>{4096},
                                      DeviceType::CUDA);
    t->set_fp8_scale(0.03125f);

    const float* p1 = t->fp8_scale_device();
    if (!p1) {
        printf("  [FAIL] fp8_scale_device() returned null\n");
        return false;
    }
    const float* p2 = t->fp8_scale_device();
    if (p1 != p2) {
        printf("  [FAIL] device scale address not stable (%p vs %p)\n", (void*)p1, (void*)p2);
        return false;
    }
    float got = 0.f;
    cudaMemcpy(&got, p1, sizeof(float), cudaMemcpyDeviceToHost);
    if (std::fabs(got - 0.03125f) > 1e-9f) {
        printf("  [FAIL] device scale value %g, expected 0.03125\n", got);
        return false;
    }

    t->set_fp8_scale(0.125f);
    const float* p3 = t->fp8_scale_device();
    if (p3 != p1) {
        printf("  [FAIL] device scale address changed after mutation (%p -> %p)\n", (void*)p1,
               (void*)p3);
        return false;
    }
    cudaMemcpy(&got, p1, sizeof(float), cudaMemcpyDeviceToHost);
    if (std::fabs(got - 0.125f) > 1e-9f) {
        printf("  [FAIL] mutation not visible at stable address (got %g, expected 0.125)\n", got);
        return false;
    }

    printf("  [OK] 6-C-5.2 device scale: addr=%p stable, value 0.03125 -> %g (in place)\n",
           (void*)p1, got);
    return true;
}

// Production-backend regression: drive the REAL forge::cuda::launch_fp8_gemv_e4m3_fp32
// (the one wired into the KernelSelector via MATMUL_FP8_E4M3_GEMV), not the inline
// audit kernel above. Verifies the production launcher matches the same golden.
static RunResult run_prod_gemv(int K, int N, const std::vector<uint8_t>& hA,
                               const std::vector<uint8_t>& hW, const std::vector<float>& golden,
                               float sa, float sw, bool check) {
    RunResult r{0, 0, 0, 0, 0};
    const size_t Wbytes = (size_t)K * N;
    const size_t Abytes = (size_t)K;
    const size_t Dbytes = (size_t)N * sizeof(float);

    uint8_t *dA = nullptr, *dW = nullptr;
    float* dD = nullptr;
    cudaMalloc(&dA, Abytes);
    cudaMalloc(&dW, Wbytes);
    cudaMalloc(&dD, Dbytes);
    cudaMemcpy(dA, hA.data(), Abytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dW, hW.data(), Wbytes, cudaMemcpyHostToDevice);

    float *d_sa = nullptr, *d_sw = nullptr;
    cudaMalloc(&d_sa, sizeof(float));
    cudaMalloc(&d_sw, sizeof(float));
    cudaMemcpy(d_sa, &sa, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sw, &sw, sizeof(float), cudaMemcpyHostToDevice);

    cuda::Fp8W8A8Scales sc;
    sc.scale_a = d_sa;
    sc.scale_w = d_sw;

    const bool ok = cuda::launch_fp8_gemv_e4m3_fp32(dA, dW, dD, K, N, sc, 0);

    std::vector<float> hD(N);
    if (ok && check) {
        cudaMemcpy(hD.data(), dD, Dbytes, cudaMemcpyDeviceToHost);
        double dot = 0, ng = 0, nt = 0;
        float maxerr = 0;
        for (int j = 0; j < N; ++j) {
            const float d = hD[j], g = golden[j];
            maxerr = std::max(maxerr, std::fabs(d - g));
            dot += d * g;
            ng += d * d;
            nt += g * g;
        }
        r.max_abs_err = maxerr;
        r.cosine = (ng > 0 && nt > 0) ? (float)(dot / (std::sqrt(ng) * std::sqrt(nt))) : 0.f;
    } else if (!ok) {
        r.cosine = 0.f;
    }

    cudaFree(dA);
    cudaFree(dW);
    cudaFree(dD);
    cudaFree(d_sa);
    cudaFree(d_sw);
    return r;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (!device_ok()) {
        printf("[fp8-gemv] SKIP: no Ada (sm_89+) GPU present\n");
        return 0;
    }

    const bool sanitize = (argc > 1 && std::string(argv[1]) == "--sanitize");

    // Decode-size shapes. K=4096 fixed; N varies across FFN/intermediate widths.
    std::vector<std::pair<int, int>> shapes;
    if (sanitize) {
        shapes.push_back({256, 256});
    } else {
        shapes.push_back({4096, 4096});
        shapes.push_back({4096, 11008});
        shapes.push_back({4096, 14336});
    }

    // Device decode LUT (value = fp8_e4m3_to_fp32(byte)) — shared with the golden.
    float hLUT[256];
    for (int i = 0; i < 256; ++i) hLUT[i] = fp8_e4m3_to_fp32((uint8_t)i);
    float* dLUT = nullptr;
    cudaMalloc(&dLUT, sizeof(hLUT));
    cudaMemcpy(dLUT, hLUT, sizeof(hLUT), cudaMemcpyHostToDevice);

    int failures = 0;

    // 6-C-5.2: Tensor-owned device scale (stable address + in-place mutation).
    if (!test_tensor_device_scale()) {
        printf("[fp8-gemv] FAIL: Tensor device scale contract\n");
        ++failures;
    }
    // 6-C-5.2b: device scalar lifetime (mutation/move keep the address; sidecar
    // propagates through copy_from/view/slice).
    if (!test_device_scale_lifetime()) {
        printf("[fp8-gemv] FAIL: 6-C-5.2b device scale lifetime\n");
        ++failures;
    }
    // 6-C-5.1a: W8A8 quantize producer writes a device-computed scale sidecar.
    if (!test_quant_producer()) {
        printf("[fp8-gemv] FAIL: 6-C-5.1a quantize producer\n");
        ++failures;
    }
    // 6-C-5.1b: amax contract (0 -> scale 1 + zeros, NaN/Inf -> rejected).
    if (!test_quant_edge_cases()) {
        printf("[fp8-gemv] FAIL: 6-C-5.1b amax contract\n");
        ++failures;
    }
    // 6-C-5.3: decisive experiment -- a captured graph observes a mutated
    // Tensor-owned device scale with no re-capture / re-instantiation.
    if (!test_graph_scale_mutation()) {
        printf("[fp8-gemv] FAIL: 6-C-5.3 graph scale mutation\n");
        ++failures;
    }
    // 6-C-5.5c: FP8 output numerical correctness (value classes + host/device agreement).
    if (!test_fp8_output_correctness()) {
        printf("[fp8-gemv] FAIL: 6-C-5.5c output correctness\n");
        ++failures;
    }
    // 6-C-5.5d: the producer itself updates the scale slot INSIDE the graph.
    if (!test_fp8_output_graph_mutation()) {
        printf("[fp8-gemv] FAIL: 6-C-5.5d output graph mutation\n");
        ++failures;
    }

    for (auto& [K, N] : shapes) {
        printf("\n=== K=%d N=%d ===\n", K, N);

        std::mt19937 rng(1234 + K * 7 + N);
        std::uniform_real_distribution<float> u(-1.f, 1.f);

        std::vector<float> hAr(K), hWr((size_t)K * N);
        for (int i = 0; i < K; ++i) hAr[i] = u(rng);
        for (size_t i = 0; i < (size_t)K * N; ++i) hWr[i] = u(rng);

        // ---- FP8 E4M3 (with real scale s = sa*sw != 1) ----
        std::vector<uint8_t> hAf(K), hWf((size_t)K * N);
        for (int i = 0; i < K; ++i) hAf[i] = fp32_to_fp8_e4m3(hAr[i]);
        for (size_t i = 0; i < (size_t)K * N; ++i) hWf[i] = fp32_to_fp8_e4m3(hWr[i]);
        float amaxA = 0, amaxW = 0;
        for (int i = 0; i < K; ++i) amaxA = std::max(amaxA, std::fabs(hAr[i]));
        for (size_t i = 0; i < (size_t)K * N; ++i) amaxW = std::max(amaxW, std::fabs(hWr[i]));
        const float sa = amaxA / 240.f, sw = amaxW / 240.f, s = sa * sw;

        std::vector<float> gFp8(N, 0.f);
        for (int j = 0; j < N; ++j) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k)
                acc += fp8_e4m3_to_fp32(hAf[k]) * fp8_e4m3_to_fp32(hWf[(size_t)k * N + j]);
            gFp8[j] = acc * s;
        }

        auto r1 = run_fp8<1>(K, N, hAf, hWf, gFp8, s, !sanitize, dLUT);
        auto r4 = run_fp8<4>(K, N, hAf, hWf, gFp8, s, !sanitize, dLUT);
        auto r16 = run_fp8<16>(K, N, hAf, hWf, gFp8, s, !sanitize, dLUT);
        printf("  fp8 VEC=1 : us/tok=%8.3f  BW=%6.1f GB/s  maxerr=%9.3e  cos=%f  scaleMaxRel=%9.3e\n",
               r1.us_per_token, r1.bandwidth_gbs, r1.max_abs_err, r1.cosine, r1.scale_max_rel);
        printf("  fp8 VEC=4 : us/tok=%8.3f  BW=%6.1f GB/s  maxerr=%9.3e  cos=%f\n",
               r4.us_per_token, r4.bandwidth_gbs, r4.max_abs_err, r4.cosine);
        printf("  fp8 VEC=16: us/tok=%8.3f  BW=%6.1f GB/s  maxerr=%9.3e  cos=%f\n",
               r16.us_per_token, r16.bandwidth_gbs, r16.max_abs_err, r16.cosine);

        // ---- production backend: REAL forge::cuda::launch_fp8_gemv_e4m3_fp32 ----
        // (the launcher wired into the KernelSelector for FP8 M==1). This exercises
        // the same path a real matmul_transB(M==1, FP8) call takes.
        // 6-C-5.6: the production kernel consumes the REAL Forge transB weight
        // layout (row-major [N,K], W[n*K + k] -- the buffer the CUTLASS FP8 GEMM
        // and the FP32 GEMV read), so we feed it a row-major weight and its own
        // golden rather than the k-major audit arrays above.
        std::vector<uint8_t> hWf_prod((size_t)K * N);
        std::vector<float> gFp8_prod(N, 0.f);
        for (int n = 0; n < N; ++n)
            for (int k = 0; k < K; ++k) hWf_prod[(size_t)n * K + k] = hWf[(size_t)k * N + n];
        for (int n = 0; n < N; ++n) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k)
                acc += fp8_e4m3_to_fp32(hAf[k]) * fp8_e4m3_to_fp32(hWf_prod[(size_t)n * K + k]);
            gFp8_prod[n] = acc * s;
        }
        auto rp = run_prod_gemv(K, N, hAf, hWf_prod, gFp8_prod, sa, sw, !sanitize);
        printf("  fp8 PROD(row-major W):    maxerr=%9.3e  cos=%f\n", rp.max_abs_err,
               rp.cosine);

        // selector routing check: FP8 + M==1 + sm89+ -> MATMUL_FP8_E4M3_GEMV
        KernelProblem kp;
        kp.domain = KernelDomain::MatMul;
        kp.dtype = DataType::FP8_E4M3;
        kp.M = 1;
        kp.K = K;
        kp.N = N;
        const KernelVariant vr = select_kernel(kp);
        printf("  selector(M==1,FP8) -> %s\n", kernel_variant_name(vr));

        // ---- FP16 baseline ----
        std::vector<__half> hAh(K), hWh((size_t)K * N);
        std::vector<float> gFp16(N, 0.f);
        for (int i = 0; i < K; ++i) hAh[i] = __float2half(hAr[i]);
        for (size_t i = 0; i < (size_t)K * N; ++i) hWh[i] = __float2half(hWr[i]);
        for (int j = 0; j < N; ++j) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k)
                acc += __half2float(hAh[k]) * __half2float(hWh[(size_t)k * N + j]);
            gFp16[j] = acc;
        }
        auto rf16 = run_one<__half, 8>(K, N, hAh, hWh, gFp16, 1.f, !sanitize);
        printf("  fp16 VEC=8: us/tok=%8.3f  BW=%6.1f GB/s  maxerr=%9.3e  cos=%f\n",
               rf16.us_per_token, rf16.bandwidth_gbs, rf16.max_abs_err, rf16.cosine);

        // ---- FP32 baseline ----
        std::vector<float> hAf32(K), hWf32((size_t)K * N);
        std::vector<float> gFp32(N, 0.f);
        for (int i = 0; i < K; ++i) hAf32[i] = hAr[i];
        for (size_t i = 0; i < (size_t)K * N; ++i) hWf32[i] = hWr[i];
        for (int j = 0; j < N; ++j) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k) acc += hAf32[k] * hWf32[(size_t)k * N + j];
            gFp32[j] = acc;
        }
        auto rf32 = run_one<float, 4>(K, N, hAf32, hWf32, gFp32, 1.f, !sanitize);
        printf("  fp32 VEC=4: us/tok=%8.3f  BW=%6.1f GB/s  maxerr=%9.3e  cos=%f\n",
               rf32.us_per_token, rf32.bandwidth_gbs, rf32.max_abs_err, rf32.cosine);

        // ---- gates ----
        if (!sanitize) {
            bool ok = true;
            if (r16.cosine < 0.9995f) { printf("  [FAIL] fp8 cosine too low\n"); ok = false; }
            if (r16.max_abs_err > 0.05f) { printf("  [FAIL] fp8 max_abs_err too high\n"); ok = false; }
            // scale fusion equivalence: s*sum(a*w) (mode 0) vs sum(s*a*w) (mode 1) differ
            // only by FP32 summation reordering (~K*eps relative); a real divergence would be >> 1%.
            if (r16.scale_max_rel > 1e-2f) { printf("  [FAIL] scale fusion not equivalent\n"); ok = false; }
            if (rp.cosine < 0.9995f) { printf("  [FAIL] fp8 PROD backend cosine too low\n"); ok = false; }
            if (vr != KernelVariant::MATMUL_FP8_E4M3_GEMV) {
                printf("  [FAIL] selector picked %s (expected MATMUL_FP8_E4M3_GEMV)\n",
                       kernel_variant_name(vr));
                ok = false;
            }
            if (rf16.cosine < 0.9995f) { printf("  [FAIL] fp16 cosine too low\n"); ok = false; }
            if (rf32.cosine < 0.9995f) { printf("  [FAIL] fp32 cosine too low\n"); ok = false; }
            if (!ok) ++failures;
            double fp8best = r1.us_per_token;
            if (r4.us_per_token < fp8best) fp8best = r4.us_per_token;
            if (r16.us_per_token < fp8best) fp8best = r16.us_per_token;
            double fp8best_bw = r1.bandwidth_gbs;
            if (r4.bandwidth_gbs > fp8best_bw) fp8best_bw = r4.bandwidth_gbs;
            if (r16.bandwidth_gbs > fp8best_bw) fp8best_bw = r16.bandwidth_gbs;
            printf("  -> FP8 best vs FP16: speedup=%.2fx ; FP8 best BW=%.1f GB/s\n",
                   rf16.us_per_token / fp8best, fp8best_bw);
        }
    }

    cudaFree(dLUT);

    if (sanitize) {
        printf("[fp8-gemv] sanitizer run OK (use compute-sanitizer for the real OOB check)\n");
        return 0;
    }
    if (failures == 0) {
        printf("\n[fp8-gemv] PASS\n");
        return 0;
    }
    printf("\n[fp8-gemv] FAILURES=%d\n", failures);
    return 1;
}
