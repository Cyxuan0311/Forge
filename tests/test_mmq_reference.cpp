// Reference tests for the CUDA MMQ (dp4a) kernels in
// src/operators/cuda/cuda_mmq.cu.
//
// Golden model: the quantized weight rows are decoded on the CPU with the
// llama.cpp-faithful scalar dequantizers (src/operators/cpu/common/dequant.cpp)
// and multiplied against FP32 activations in double precision. The MMQ kernels
// quantize activations to int8 blocks internally, so their outputs carry
// activation-quantization noise (~4e-3 relative on random data). Structural
// defects -- K-stream misalignment, wrong scale/min unpacking, dropped high
// bits -- produce O(1) garbage that fails the 5%-of-max tolerance decisively.
//
// Test data: synthetic blocks with valid fp16 scale fields and random payload
// bytes. Any bit pattern decodes deterministically, so no host quantizer is
// required; agreement between the CPU decoder and the GPU kernel is exactly
// what is under test.
//
// Build & run:
//   cmake --build build -j --target forge-mmq-ref-test && \
//   ./build/tests/forge-mmq-ref-test

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "forge/cuda_kernels.h"
#include "operators/cpu/common/dequant.h"

namespace {

using forge::cuda::MmqFn;

// ---- tiny fp16 codec (only needs to produce valid positive normals) ------

uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    const uint32_t man = x & 0x7FFFFFu;
    if (exp <= 0) return static_cast<uint16_t>(sign);
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (man >> 13));
}

void put_f16(uint8_t* p, float v) {
    const uint16_t b = f32_to_f16(v);
    std::memcpy(p, &b, 2);
}

// ---- synthetic block generators ------------------------------------------
//
// Every generator fills ONE block with valid fp16 scale fields plus random
// payload bytes drawn from rng. Byte layouts mirror dequant.cpp.

constexpr float kDMin = 0.02f, kDMax = 1.0f;
constexpr float kDminLo = 0.01f, kDminHi = 0.3f;

float urand(std::mt19937& g, float lo, float hi) {
    return lo + (hi - lo) * (static_cast<float>(g() % 100000) / 100000.0f);
}

void fill_random(uint8_t* p, size_t n, std::mt19937& g) {
    for (size_t i = 0; i < n; ++i) p[i] = static_cast<uint8_t>(g() & 0xFF);
}

void gen_q4_k(uint8_t* p, std::mt19937& g) {          // 144 B: d dmin sc[12] qs[128]
    put_f16(p + 0, urand(g, kDMin, kDMax));
    put_f16(p + 2, urand(g, kDminLo, kDminHi));
    fill_random(p + 4, 12, g);
    fill_random(p + 16, 128, g);
}

void gen_q5_k(uint8_t* p, std::mt19937& g) {          // 176 B: d dmin sc[12] qh[32] qs[128]
    put_f16(p + 0, urand(g, kDMin, kDMax));
    put_f16(p + 2, urand(g, kDminLo, kDminHi));
    fill_random(p + 4, 12, g);
    fill_random(p + 16, 32, g);
    fill_random(p + 48, 128, g);
}

void gen_q6_k(uint8_t* p, std::mt19937& g) {          // 210 B: ql[128] qh[64] sc(i8)[16] d
    fill_random(p + 0, 128, g);
    fill_random(p + 128, 64, g);
    for (int i = 0; i < 16; ++i)
        p[192 + i] = static_cast<uint8_t>(static_cast<int>(g() % 127) - 63);  // int8 scale
    put_f16(p + 208, urand(g, kDMin, kDMax));
}

void gen_q2_k(uint8_t* p, std::mt19937& g) {          // 84 B: sc[16] qs[64] d dmin
    fill_random(p + 0, 16, g);                        // lo nibble=scale, hi=min
    fill_random(p + 16, 64, g);
    put_f16(p + 80, urand(g, kDMin, kDMax));
    put_f16(p + 82, urand(g, kDminLo, kDminHi));
}

void gen_q3_k(uint8_t* p, std::mt19937& g) {          // 110 B: hmask[32] qs[64] sc[12] d
    // Component-isolation switches for debugging the residual deviation:
    // FORGE_Q3K_ZERO_HMASK / FORGE_Q3K_CONST_SCALES / FORGE_Q3K_NIB1_QS
    const bool z_hm = std::getenv("FORGE_Q3K_ZERO_HMASK") != nullptr;
    const bool o_hm = std::getenv("FORGE_Q3K_ONES_HMASK") != nullptr;
    const bool c_sc = std::getenv("FORGE_Q3K_CONST_SCALES") != nullptr;
    const bool n_qs = std::getenv("FORGE_Q3K_NIB1_QS") != nullptr;
    if (z_hm) {
        std::memset(p, 0, 32);
    } else if (o_hm) {
        std::memset(p, 0xFF, 32);
    } else {
        fill_random(p + 0, 32, g);
    }
    if (n_qs) {
        // only low nibble varies per byte position pattern 0..7
        for (int i = 0; i < 64; ++i) p[32 + i] = static_cast<uint8_t>(i % 4);
    } else {
        fill_random(p + 32, 64, g);
    }
    if (c_sc) {
        // all-zero scale bytes decode to a uniform -32 on BOTH the kernel
        // (__vsubss4(0, 0x20202020)) and the CPU aux-shuffle path (0-32),
        // isolating qs/hmask handling from scale handling entirely.
        std::memset(p + 96, 0, 12);
    } else {
        fill_random(p + 96, 12, g);
    }
    put_f16(p + 108, urand(g, kDMin, kDMax));
}

void gen_q4_0(uint8_t* p, std::mt19937& g) {          // 18 B: d qs[16]
    put_f16(p + 0, urand(g, kDMin, kDMax));
    fill_random(p + 2, 16, g);
}

void gen_q8_0(uint8_t* p, std::mt19937& g) {          // 34 B: d qs(int8)[32]
    put_f16(p + 0, urand(g, kDMin, kDMax));
    fill_random(p + 2, 32, g);  // read back as signed int8 by dequantizer and kernel
}

// Phase 3 (Tensor Core) adapters: MmqFn-shaped wrappers around the
// whole-matrix dequant-to-FP16 + cuBLAS FP16 GEMM launcher.
static void launch_tc_q4_0(const float* a, const void* w, float* o, int M, int K, int N,
                           cudaStream_t s) {
    forge::cuda::launch_cublas_gemm_quant_fp16(forge::DataType::Q4_0, a, w, o, M, K, N,
                                               /*transB=*/true, s);
}
static void launch_tc_q4_k(const float* a, const void* w, float* o, int M, int K, int N,
                           cudaStream_t s) {
    forge::cuda::launch_cublas_gemm_quant_fp16(forge::DataType::Q4_K, a, w, o, M, K, N,
                                               /*transB=*/true, s);
}
static void launch_tc_q8_0(const float* a, const void* w, float* o, int M, int K, int N,
                           cudaStream_t s) {
    forge::cuda::launch_cublas_gemm_quant_fp16(forge::DataType::Q8_0, a, w, o, M, K, N,
                                               /*transB=*/true, s);
}
static void launch_tc_q6_k(const float* a, const void* w, float* o, int M, int K, int N,
                           cudaStream_t s) {
    forge::cuda::launch_cublas_gemm_quant_fp16(forge::DataType::Q6_K, a, w, o, M, K, N,
                                               /*transB=*/true, s);
}
static void launch_tc_q5_k(const float* a, const void* w, float* o, int M, int K, int N,
                           cudaStream_t s) {
    forge::cuda::launch_cublas_gemm_quant_fp16(forge::DataType::Q5_K, a, w, o, M, K, N,
                                               /*transB=*/true, s);
}

// ---- per-dtype descriptors -------------------------------------------------

struct DtypeSpec {
    const char* name;
    int block_bytes;
    int block_elems;   // elements per quant super-block
    void (*gen)(uint8_t*, std::mt19937&);
    void (*dequant_row)(const uint8_t*, float*, int, int);   // forge::ops golden
    MmqFn launch;                                            // forge::cuda kernel
    double tol;          // normalized (over global_max) tolerance
};

const DtypeSpec kSpecs[] = {
    {"Q4_K", 144, 256, gen_q4_k, forge::ops::dequantize_q4_k_row, forge::cuda::launch_mmq_q4_k,
     0.05},
    {"Q6_K", 210, 256, gen_q6_k, forge::ops::dequantize_q6_k_row, forge::cuda::launch_mmq_q6_k,
     0.05},
    {"Q5_K", 176, 256, gen_q5_k, forge::ops::dequantize_q5_k_row, forge::cuda::launch_mmq_q5_k,
     0.05},
    {"Q2_K", 84, 256, gen_q2_k, forge::ops::dequantize_q2_k_row, forge::cuda::launch_mmq_q2_k,
     0.05},
    {"Q3_K", 110, 256, gen_q3_k, forge::ops::dequantize_q3_k_row, forge::cuda::launch_mmq_q3_k,
     0.05},
    {"Q4_0", 18, 32, gen_q4_0, forge::ops::dequantize_q4_0_row, forge::cuda::launch_mmq_q4_0,
     0.05},
};

// Phase 3/5 Tensor Core path (dequant-to-FP16 + cuBLAS FP16 GEMM, FP32
// accumulate). FP16 carries ~2^-11 per-element rounding vs the FP64 golden, so
// the tolerance is tighter than MMQ's (activation-int8 noise) but still far
// below the O(1) signature of structural defects. Q8_0's real MMQ arrived with
// Phase 3.5 (it is covered in the MMQ suite via cuda_mmq); this TC suite covers
// every dtype with an FP16 whole-matrix dequant (Phase 5 folded Q6_K in).
const DtypeSpec kTcSpecs[] = {
    {"Q4_0/T", 18, 32, gen_q4_0, forge::ops::dequantize_q4_0_row, launch_tc_q4_0, 0.01},
    {"Q4_K/T", 144, 256, gen_q4_k, forge::ops::dequantize_q4_k_row, launch_tc_q4_k, 0.01},
    {"Q6_K/T", 210, 256, gen_q6_k, forge::ops::dequantize_q6_k_row, launch_tc_q6_k, 0.01},
    {"Q5_K/T", 176, 256, gen_q5_k, forge::ops::dequantize_q5_k_row, launch_tc_q5_k, 0.01},
    {"Q8_0/T", 34, 32, gen_q8_0, forge::ops::dequantize_q8_0_row, launch_tc_q8_0, 0.01},
};

// ---- helpers ----------------------------------------------------------------

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t e_ = (call);                                               \
        if (e_ != cudaSuccess) {                                               \
            std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_),    \
                        __FILE__, __LINE__);                                   \
            return false;                                                      \
        }                                                                      \
    } while (0)

struct DeviceBuffers {
    float* d_a = nullptr;
    void* d_w = nullptr;
    float* d_out = nullptr;
    float* d_out2 = nullptr;
    ~DeviceBuffers() {
        cudaFree(d_a);
        cudaFree(d_w);
        cudaFree(d_out);
        cudaFree(d_out2);
    }
};

// Runs one (dtype, M, K, N) case. Returns pass/fail; reports worst normalized
// deviation and determinism through the out-params.
bool run_case(const DtypeSpec& spec, int M, int K, int N, std::mt19937& rng,
              double* worst_rel, bool* deterministic, std::string* err) {
    // The per-weight FP16 dequant cache keys on the raw device pointer with no
    // content validation. run_case hands freshly-randomized weights to
    // cudaMalloc'd buffers that the CUDA allocator frequently recycles at the
    // SAME address across cases, so a stale entry would poison the next case.
    // Real inference keeps weights immutable for the process lifetime; tests
    // must invalidate explicitly.
    forge::cuda::dequant_cache_clear();
    const size_t blocks_per_row =
        (static_cast<size_t>(K) + spec.block_elems - 1) / spec.block_elems;
    const size_t row_bytes = blocks_per_row * spec.block_bytes;

    // Host data -----------------------------------------------------------
    std::vector<uint8_t> w_q(row_bytes * N);
    for (size_t n = 0; n < w_q.size(); n += spec.block_bytes)
        spec.gen(w_q.data() + n, rng);

    std::vector<float> a(static_cast<size_t>(M) * K);
    {
        // FORGE_MMQ_CONST_ACT=1: uniform activations collapse per-32 amax
        // blocks to a single scale, isolating activation-side indexing bugs
        // from weight-side decode bugs.
        if (std::getenv("FORGE_MMQ_CONST_ACT")) {
            for (int m = 0; m < M; ++m)
                for (int k = 0; k < K; ++k)
                    a[m * K + k] = 0.5f;
        } else {
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (auto& v : a) v = dist(rng);
        }
    }

    // Golden: CPU dequant + double-precision GEMM --------------------------
    std::vector<double> w_f32(static_cast<size_t>(N) * K);
    for (int n = 0; n < N; ++n) {
        std::vector<float> row(K);
        spec.dequant_row(w_q.data() + static_cast<size_t>(n) * row_bytes, row.data(), K,
                         /*row=*/0);
        for (int k = 0; k < K; ++k) w_f32[static_cast<size_t>(n) * K + k] = row[k];
    }
    double global_max = 0.0;
    std::vector<double> gold(static_cast<size_t>(M) * N, 0.0);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            const double* wr = &w_f32[static_cast<size_t>(n) * K];
            for (int k = 0; k < K; ++k) acc += static_cast<double>(a[m * K + k]) * wr[k];
            gold[static_cast<size_t>(m) * N + n] = acc;
            global_max = std::max(global_max, std::fabs(acc));
        }
    }
    if (global_max <= 0.0) {
        *err = "degenerate golden matrix";
        return false;
    }

    // Device run ------------------------------------------------------------
    DeviceBuffers bufs;
    CUDA_CHECK(cudaMalloc(&bufs.d_a, a.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bufs.d_w, w_q.size()));
    CUDA_CHECK(cudaMalloc(&bufs.d_out, static_cast<size_t>(M) * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bufs.d_out2, static_cast<size_t>(M) * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(bufs.d_a, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(bufs.d_w, w_q.data(), w_q.size(), cudaMemcpyHostToDevice));

    spec.launch(bufs.d_a, bufs.d_w, bufs.d_out, M, K, N, 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    spec.launch(bufs.d_a, bufs.d_w, bufs.d_out2, M, K, N, 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> out(static_cast<size_t>(M) * N);
    CUDA_CHECK(cudaMemcpy(out.data(), bufs.d_out, out.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    std::vector<float> out2(out.size());
    CUDA_CHECK(cudaMemcpy(out2.data(), bufs.d_out2, out2.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    *deterministic =
        std::memcmp(out.data(), out2.data(), out.size() * sizeof(float)) == 0;

    // Tolerance (normalized over the matrix max): MMQ activation-int8 noise and
    // FP16 TC rounding both stay far below their tolerances on random data while
    // structural corruption lands at O(1). spec.tol encodes per-path slack.
    const double kRelTol = spec.tol;
    *worst_rel = 0.0;
    size_t worst_idx = 0;
    for (size_t i = 0; i < out.size(); ++i) {
        const double rel = std::fabs(static_cast<double>(out[i]) - gold[i]) / global_max;
        if (rel > *worst_rel) {
            *worst_rel = rel;
            worst_idx = i;
        }
    }
    if (*worst_rel > kRelTol) {
        char buf[160];
        std::snprintf(buf, sizeof buf, "rel err %.3e > tol %.0e", *worst_rel, kRelTol);
        *err = buf;

        return false;
    }
    if (!*deterministic) {
        *err = "non-deterministic output between runs";
        return false;
    }
    return true;
}

}  // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count < 1) {
        std::printf("NOTE: no CUDA device available; skipping MMQ reference tests\n");
        return 0;
    }

    struct Shape {
        int M, K, N;
    };
    const Shape shapes[] = {
        {2, 256, 64},    {6, 256, 512},    {17, 256, 64},
        {33, 3584, 512}, {64, 3584, 64},
    };

    int checks = 0, failures = 0;
    unsigned seed = 20260825;
    if (const char* s = std::getenv("FORGE_MMQ_SEED")) seed = static_cast<unsigned>(std::atoi(s));
    std::mt19937 rng(seed);

    // MMQ (dp4a) suite, then the Phase 3 TensorCore suite (dequant-to-FP16 +
    // cuBLAS FP16 GEMM) over the same shapes.
    const size_t nshapes = sizeof(shapes) / sizeof(shapes[0]);
    auto run_suite = [&](const char* label, const DtypeSpec* specs, size_t nspecs) {
        for (size_t si = 0; si < nspecs; ++si) {
            const DtypeSpec& spec = specs[si];
            int dtype_fails = 0;
            double dtype_worst = 0.0;
            for (const auto& sh : shapes) {
                ++checks;
                double rel = 0.0;
                bool det = false;
                std::string err;
                const bool ok = run_case(spec, sh.M, sh.K, sh.N, rng, &rel, &det, &err);
                dtype_worst = std::max(dtype_worst, rel);
                if (!ok) {
                    ++failures;
                    ++dtype_fails;
                    std::printf("FAIL %s %-7s M=%-3d K=%-5d N=%-4d : %s\n", label, spec.name,
                                sh.M, sh.K, sh.N, err.c_str());
                }
            }
            std::printf("%s %-7s %s (%zu/%zu cases, worst rel %.3e)\n", label, spec.name,
                        dtype_fails == 0 ? "ok   " : "FAILED", nshapes - dtype_fails, nshapes,
                        dtype_worst);
        }
    };

    run_suite("MMQ", kSpecs, sizeof(kSpecs) / sizeof(kSpecs[0]));
    run_suite("TC ", kTcSpecs, sizeof(kTcSpecs) / sizeof(kTcSpecs[0]));

    std::printf("\n=== mmq/TC ref: %d/%d cases passed ===\n", checks - failures, checks);
    return failures > 0 ? 1 : 0;
}
