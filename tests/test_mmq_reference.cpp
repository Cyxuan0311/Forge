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
#include "operators/cpu/common/quant_helpers.h"

namespace {

using forge::cuda::MmqFn;

// Golden flavour for run_case:
//   kFp32 -- dot the dequantized weights against the RAW fp32 activations.
//            The kernels quantize activations to int8 internally, so the
//            residual is that quantization noise (~4e-3 on random data); the
//            loose per-dtype tolerance only catches O(1) structural defects.
//   kQ8K  -- dot against the activations AFTER CPU quantize_row_q8_K (the exact
//            per-256 semantics quantize_q8_1_mmq_kernel targets). Both sides now
//            see the same integers, so the residual collapses to fp32
//            accumulation order (~1e-6) and any scale/sum (DS4 / D2S6) decode
//            defect -- wrong group width, wrong `s = d*bsum` term, stale index
//            -- lands orders of magnitude above the tolerance.
enum class Golden { kFp32, kQ8K };

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

// FORGE_MMQ_ZERO_DMIN: kill the weight min term (and with it the whole
// DS4 / D2S6 `s = d*bsum` path) so a strict-golden residual can be attributed
// to the sum term or to the plain (d, qs) stream.
const bool kZeroDMin = std::getenv("FORGE_MMQ_ZERO_DMIN") != nullptr;

void gen_q4_k(uint8_t* p, std::mt19937& g) {          // 144 B: d dmin sc[12] qs[128]
    put_f16(p + 0, urand(g, kDMin, kDMax));
    put_f16(p + 2, kZeroDMin ? 0.0f : urand(g, kDminLo, kDminHi));
    fill_random(p + 4, 12, g);
    fill_random(p + 16, 128, g);
}

void gen_q5_k(uint8_t* p, std::mt19937& g) {          // 176 B: d dmin sc[12] qh[32] qs[128]
    put_f16(p + 0, urand(g, kDMin, kDMax));
    put_f16(p + 2, kZeroDMin ? 0.0f : urand(g, kDminLo, kDminHi));
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
              Golden golden, double* worst_rel, bool* deterministic, std::string* err) {
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

    // FORGE_MMQ_PROBE_SUM (Q4_0 only): replace the weights by indicators of
    // each 32-element group -- row n is 1 on group (n % blocks_per_row) and 0
    // elsewhere. out[m][n] then IS the activation's per-32 dequantized sum,
    // i.e. exactly the DS4 `s = d*bsum` value the dot kernels consume, so the
    // whole scale/sum stream can be diffed against the CPU Q8_K reference
    // element by element.
    // Mode "1": indicators -> out == the per-32 activation sum (the DS4 `s`).
    // Mode "ramp": weight k = (k%16)-8 inside the target group -> the result is
    // order-sensitive, so it sees the INDIVIDUAL quants rather than their sum.
    //   ramp clean + sum wrong  => the qsum reduction drops/adds a value;
    //   ramp wrong              => the quantized values themselves differ.
    const char* probe_env = std::getenv("FORGE_MMQ_PROBE_SUM");
    const bool probe_sum = probe_env && spec.block_bytes == 18 && K % 32 == 0;
    const bool probe_ramp = probe_sum && std::strcmp(probe_env, "ramp") == 0;

    if (probe_sum) {
        const int bpb = K / 32;
        for (int n = 0; n < N; ++n) {
            for (int b = 0; b < bpb; ++b) {
                uint8_t* p = w_q.data() + (static_cast<size_t>(n) * bpb + b) * 18;
                put_f16(p, 1.0f);  // weight dequant = d*(q-8): 0x88 -> 0, 0x99 -> 1
                if (b != n % bpb) {
                    std::memset(p + 2, 0x88, 16);
                } else if (probe_ramp) {
                    for (int j = 0; j < 16; ++j)
                        p[2 + j] = static_cast<uint8_t>(j | (j << 4));
                } else {
                    std::memset(p + 2, 0x99, 16);
                }
            }
        }
    }

    std::vector<float> a(static_cast<size_t>(M) * K);
    {
        // FORGE_MMQ_CONST_ACT=1: uniform activations collapse per-32 amax
        // blocks to a single scale, isolating activation-side indexing bugs
        // from weight-side decode bugs.
        if (std::getenv("FORGE_MMQ_CONST_ACT")) {
            for (int m = 0; m < M; ++m)
                for (int k = 0; k < K; ++k)
                    a[m * K + k] = 0.5f;
        } else if (std::getenv("FORGE_MMQ_EXACT_ACT")) {
            // Every activation lands on an exact int8 grid point: with
            // amax = 127/254 the scale is iscale = -254 and i*0.5*-254 is an
            // exact integer, so GPU (__float2int_rn) and CPU (nearest_int_f)
            // CANNOT disagree on any element. Any residual is then purely
            // scale/d/s decode, not activation rounding.
            // Grid is i/256: every activation, the 1/iscale scale AND every
            // q*d product are exact in fp32 (power-of-two denominator), so the
            // GPU and CPU must agree bit-for-bit -- any residual is structural.
            std::uniform_int_distribution<int> idist(-127, 127);
            for (auto& v : a) v = static_cast<float>(idist(rng)) / 256.0f;
        } else {
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (auto& v : a) v = dist(rng);
        }
    }

    // Activation side of the golden: raw fp32, or the CPU Q8_K quantization of
    // it when the strict (Golden::kQ8K) flavour is requested.
    std::vector<double> a_eff(static_cast<size_t>(M) * K);
    if (golden == Golden::kQ8K) {
        const int nb = (K + 255) / 256;
        std::vector<forge::cpu::block_q8_K> q8(static_cast<size_t>(M) * nb);
        for (int m = 0; m < M; ++m) {
            forge::cpu::quantize_row_q8_K(&a[static_cast<size_t>(m) * K],
                                          &q8[static_cast<size_t>(m) * nb], K);
            for (int k = 0; k < K; ++k) {
                const forge::cpu::block_q8_K& b = q8[static_cast<size_t>(m) * nb + k / 256];
                a_eff[static_cast<size_t>(m) * K + k] =
                    static_cast<double>(b.d) * static_cast<double>(b.qs[k % 256]);
            }
        }
    } else {
        for (size_t i = 0; i < a.size(); ++i) a_eff[i] = static_cast<double>(a[i]);
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
            for (int k = 0; k < K; ++k) acc += a_eff[static_cast<size_t>(m) * K + k] * wr[k];
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

    if (probe_sum) {
        double worst = 0.0;
        int ndiff = 0, mw = -1, nw = -1;
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n) {
                const double d = std::fabs(static_cast<double>(out[(size_t)m * N + n]) -
                                           gold[(size_t)m * N + n]);
                if (d > 1e-12) ++ndiff;
                if (d > worst) {
                    worst = d;
                    mw = m;
                    nw = n;
                }
            }
        int nrun = 0;
        for (size_t i = 0; i < out.size(); ++i)
            if (out[i] != out2[i]) ++nrun;
        std::printf("    probe s-stream M=%d K=%d N=%d: %d/%d groups differ, max |diff| %.3e"
                    " (m=%d grp=%d) run-to-run diffs %d\n",
                    M, K, N, ndiff, M * N, worst, mw, nw, nrun);
        if (mw >= 0)
            std::printf("      gpu %14.7f  cpu %14.7f\n",
                        static_cast<double>(out[(size_t)mw * N + nw]),
                        gold[(size_t)mw * N + nw]);
        // Per-256-block aggregation for the worst row: is a wrong `s`
        // block-local (K-stream / block indexing) or scattered (quant rounding)?
        if (ndiff > 0 && mw >= 0 && K % 32 == 0) {
            const int gpb = 8;  // 32-element groups per 256-element block
            const int nb = K / 256;
            std::printf("      worst row %d, per-256-block max |diff| (8 grps each):\n", mw);
            for (int b = 0; b < nb; ++b) {
                double bd = 0.0;
                for (int g = 0; g < gpb; ++g) {
                    const int n = b * gpb + g;
                    if (n >= N) break;
                    bd = std::max(bd, std::fabs(static_cast<double>(out[(size_t)mw * N + n]) -
                                                gold[(size_t)mw * N + n]));
                }
                std::printf("        blk%-3d %.3e%s\n", b, bd, bd > 1e-9 ? "   <== " : "");
            }
        }
        // For the worst row, recover the block scale the GPU must have used:
        // s = d*qsum, so d_implied = s_gpu / qsum_cpu. A constant d_implied !=
        // d_cpu across the block's groups means the amax/iscale reduce picked a
        // different max; a scattered one means individual quants differ.
        if (mw >= 0 && K % 256 == 0) {
            const int nbk = K / 256;
            std::vector<forge::cpu::block_q8_K> rq(static_cast<size_t>(nbk));
            forge::cpu::quantize_row_q8_K(&a[static_cast<size_t>(mw) * K], rq.data(), K);
            int wb = 0;
            double wbd = 0.0;
            for (int b = 0; b < nbk; ++b)
                for (int g = 0; g < 8; ++g) {
                    const int n = b * 8 + g;
                    if (n >= N) break;
                    const double d = std::fabs(static_cast<double>(out[(size_t)mw * N + n]) -
                                               gold[(size_t)mw * N + n]);
                    if (d > wbd) {
                        wbd = d;
                        wb = b;
                    }
                }
            std::printf("      row %d worst blk %d (|diff| %.3e): d_cpu=%.9f\n", mw, wb, wbd,
                        static_cast<double>(rq[wb].d));
            // Distance of each product to the .5 rounding boundary inside the
            // most-suspect group: an EXACT tie is where a magic-number
            // (round-half-even) host rounder and __float2int_rn can diverge.
            {
                float mx = 0.0f, amx = 0.0f;
                for (int k = 0; k < 256; ++k) {
                    const float v = a[(size_t)mw * K + wb * 256 + k];
                    if (std::fabs(v) > amx) {
                        amx = std::fabs(v);
                        mx = v;
                    }
                }
                const float isc = -127.0f / mx;
                int gsel = 0;
                double bestr = 0.0;
                for (int g = 0; g < 8; ++g) {
                    const int n = wb * 8 + g;
                    if (n >= N) break;
                    const int qs = rq[wb].bsums[2 * g] + rq[wb].bsums[2 * g + 1];
                    if (qs != 0) {
                        const double r = std::fabs((out[(size_t)mw * N + n] / qs) / rq[wb].d - 1.0);
                        if (r > bestr) {
                            bestr = r;
                            gsel = g;
                        }
                    }
                }
                double mind = 1.0;
                int kmin = -1;
                for (int k = 0; k < 32; ++k) {
                    const float prod = isc * a[(size_t)mw * K + wb * 256 + gsel * 32 + k];
                    const double d = std::fabs(std::fabs((double)prod) -
                                               std::floor(std::fabs((double)prod)) - 0.5);
                    if (d < mind) {
                        mind = d;
                        kmin = k;
                    }
                }
                const float tp = isc * a[(size_t)mw * K + wb * 256 + gsel * 32 + kmin];
                std::printf("      blk %d grp %d: closest element to a .5 tie: k=%d dist %.3e%s\n",
                            wb, gsel, kmin, mind, mind < 1e-9 ? "  <== EXACT TIE" : "");
                std::printf("        tie value %.9f -> nearest_int_f %d, rint %d, gpu-implied %d\n",
                            static_cast<double>(tp), forge::cpu::nearest_int_f(tp),
                            (int)std::rint(tp),
                            (int)std::llround((out[(size_t)mw * N + wb * 8 + gsel] / rq[wb].d) -
                                              (rq[wb].bsums[2 * gsel] + rq[wb].bsums[2 * gsel + 1] -
                                               rq[wb].qs[gsel * 32 + kmin])));
            }
            for (int g = 0; g < 8; ++g) {
                const int n = wb * 8 + g;
                if (n >= N) break;
                const int qs = rq[wb].bsums[2 * g] + rq[wb].bsums[2 * g + 1];
                const double sg = out[(size_t)mw * N + n];
                std::printf("        g%d qsum %6d  s_gpu %12.7f  d_implied %+.9f  ratio %+.7f\n",
                            g, qs, sg, qs != 0 ? sg / qs : 0.0,
                            (qs != 0 && rq[wb].d != 0.0f) ? (sg / qs) / rq[wb].d : 0.0);
            }
        }
        // Per-group ratio over block 0: s = d*qsum, so a CONSTANT ratio across
        // the 8 groups means the quants agree and only the block scale `d`
        // differs; a varying ratio means the quantized values themselves differ.
        const int nblk = (K + 255) / 256;
        std::vector<forge::cpu::block_q8_K> pq(static_cast<size_t>(nblk));
        for (int m = 0; m < std::min(M, 2); ++m) {
            forge::cpu::quantize_row_q8_K(&a[static_cast<size_t>(m) * K], pq.data(), K);
            std::printf("      row %d cpu d=%.9f\n", m, static_cast<double>(pq[0].d));
            for (int g = 0; g < 8 && g < N; ++g) {
                const double s_cpu = static_cast<double>(pq[0].d) *
                                     (pq[0].bsums[2 * g] + pq[0].bsums[2 * g + 1]);
                const double s_gpu = out[(size_t)m * N + g];
                std::printf("        grp%-2d gpu %12.7f cpu %12.7f ratio %+.6f  qsum cpu %6d\n",
                            g, s_gpu, s_cpu, s_cpu != 0.0 ? s_gpu / s_cpu : 0.0,
                            (int)(pq[0].bsums[2 * g] + pq[0].bsums[2 * g + 1]));
            }
        }
    }

    // Tolerance (normalized over the matrix max): MMQ activation-int8 noise and
    // FP16 TC rounding both stay far below their tolerances on random data while
    // structural corruption lands at O(1). spec.tol encodes per-path slack.
    // Golden::kQ8K removes the activation-quantization noise from the comparison
    // entirely, so it can demand fp32-accumulation-level agreement.
    //
    // Measured floor of the strict mode (see debug-dflash-cuda-device.md,
    // "Round 4"): ~2e-4 for the DS4/D4 dtypes and ~5e-4 for Q2_K. Neither is a
    // scale-decode defect:
    //   * the DS4/D4 floor is a 1-LSB divergence of the ACTIVATION quantizer --
    //     ~1 element in 1e5 lands exactly on a .5 rounding boundary and the GPU
    //     (__float2int_rn) and the CPU (magic-number rounder) resolve it
    //     differently. One flipped quantum moves a single dot product by
    //     ~|w|*d_a, i.e. ~2e-4 of the matrix max.
    //   * Q2_K's floor is the D2S6 layout itself: partial sums are stored as
    //     fp16, so they carry 2^-11 ~ 4.9e-4 relative rounding against the
    //     CPU's exact int16 bsums. llama.cpp shares the layout.
    // Structural defects (wrong scale/sum slot, dropped high bits, K-stream
    // misalignment) land at >=1e-2, i.e. >10x above this floor.
    const double kRelTol = (golden == Golden::kQ8K) ? 1e-3 : spec.tol;
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

        // FORGE_MMQ_DIAG=1: localize a strict-golden failure.
        //   (a) row/column concentration -- activation-side vs weight-side;
        //   (b) K-prefix bisect -- which 256-element block first diverges.
        //     A defect confined to the activation stream lands on ONE prefix
        //     step; a weight-decode defect shows up on every prefix.
        if (std::getenv("FORGE_MMQ_DIAG")) {
            std::vector<double> row_worst(M, 0.0), col_worst(N, 0.0);
            for (int m = 0; m < M; ++m)
                for (int n = 0; n < N; ++n) {
                    const double r = std::fabs(static_cast<double>(out[(size_t)m * N + n]) -
                                               gold[(size_t)m * N + n]) /
                                     global_max;
                    row_worst[m] = std::max(row_worst[m], r);
                    col_worst[n] = std::max(col_worst[n], r);
                }
            auto count_above = [&](const std::vector<double>& v, double t) {
                int c = 0;
                for (double x : v)
                    if (x > t) ++c;
                return c;
            };
            const double t = 0.1 * *worst_rel;
            int mw = 0;
            for (int m = 1; m < M; ++m)
                if (row_worst[m] > row_worst[mw]) mw = m;
            std::printf("    diag rows>10%%: %d/%d  cols>10%%: %d/%d  worst row m=%d\n",
                        count_above(row_worst, t), M, count_above(col_worst, t), N, mw);

            // Is the worst row a UNIFORM rescale of the golden (=> its
            // activation block scale d differs) or does it carry an
            // element-level mismatch (=> quants / grouping / indexing)?
            double s_og = 0.0, s_gg = 0.0, s_oo = 0.0;
            for (int n = 0; n < N; ++n) {
                const double o = out[(size_t)mw * N + n];
                const double g = gold[(size_t)mw * N + n];
                s_og += o * g;
                s_gg += g * g;
                s_oo += o * o;
            }
            if (s_gg > 0.0) {
                const double alpha = s_og / s_gg;
                double res2 = 0.0;
                for (int n = 0; n < N; ++n) {
                    const double d2 = out[(size_t)mw * N + n] - alpha * gold[(size_t)mw * N + n];
                    res2 += d2 * d2;
                }
                std::printf("    diag row %d: best-fit alpha=%.9f  residual=%.3e  "
                            "|gold|=%.3e |out|=%.3e\n",
                            mw, alpha, std::sqrt(res2 / s_gg), std::sqrt(s_gg), std::sqrt(s_oo));
            }

            float* d_ap = nullptr;
            float* d_op = nullptr;
            if (cudaMalloc(&d_ap, a.size() * sizeof(float)) == cudaSuccess &&
                cudaMalloc(&d_op, static_cast<size_t>(M) * N * sizeof(float)) == cudaSuccess) {
                std::vector<float> a_p(static_cast<size_t>(M) * K);
                std::vector<float> o_p(static_cast<size_t>(M) * N);
                for (int kb = 1; kb <= K / 256; ++kb) {
                    const int Kp = kb * 256;
                    for (int m = 0; m < M; ++m)
                        for (int k = 0; k < Kp; ++k)
                            a_p[(size_t)m * Kp + k] = a[(size_t)m * K + k];
                    cudaMemcpy(d_ap, a_p.data(), (size_t)M * Kp * sizeof(float),
                               cudaMemcpyHostToDevice);
                    spec.launch(d_ap, bufs.d_w, d_op, M, Kp, N, 0);
                    cudaDeviceSynchronize();
                    cudaMemcpy(o_p.data(), d_op, (size_t)M * N * sizeof(float),
                               cudaMemcpyDeviceToHost);
                    double gmax_p = 0.0, worst_p = 0.0;
                    for (int m = 0; m < M; ++m)
                        for (int n = 0; n < N; ++n) {
                            double acc = 0.0;
                            for (int k = 0; k < Kp; ++k)
                                acc += a_eff[(size_t)m * K + k] * w_f32[(size_t)n * K + k];
                            gmax_p = std::max(gmax_p, std::fabs(acc));
                            worst_p = std::max(
                                worst_p, std::fabs(static_cast<double>(o_p[(size_t)m * N + n]) - acc));
                        }
                    std::printf("    prefix K=%-5d rel %.3e\n", Kp,
                                worst_p / (gmax_p > 0 ? gmax_p : 1.0));
                }
                cudaFree(d_ap);
                cudaFree(d_op);
            }
        }
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
    // Default sweep; FORGE_MMQ_SHAPE="M,K,N" narrows it to one shape for
    // bisecting a defect (M/K/N independence checks).
    Shape one_shape{0, 0, 0};
    const Shape* shape_list = shapes;
    size_t nshape = sizeof(shapes) / sizeof(shapes[0]);
    if (const char* s = std::getenv("FORGE_MMQ_SHAPE")) {
        if (std::sscanf(s, "%d,%d,%d", &one_shape.M, &one_shape.K, &one_shape.N) == 3) {
            shape_list = &one_shape;
            nshape = 1;
        }
    }
    (void)shape_list;
    (void)nshape;

    int checks = 0, failures = 0;
    unsigned seed = 20260825;
    if (const char* s = std::getenv("FORGE_MMQ_SEED")) seed = static_cast<unsigned>(std::atoi(s));
    std::mt19937 rng(seed);

    // MMQ (dp4a) suite, then the Phase 3 TensorCore suite (dequant-to-FP16 +
    // cuBLAS FP16 GEMM) over the same shapes.
    const size_t nshapes = nshape;
    auto run_suite = [&](const char* label, const DtypeSpec* specs, size_t nspecs, Golden golden) {
        for (size_t si = 0; si < nspecs; ++si) {
            const DtypeSpec& spec = specs[si];
            int dtype_fails = 0;
            double dtype_worst = 0.0;
            for (size_t si2 = 0; si2 < nshapes; ++si2) {
                const Shape& sh = shape_list[si2];
                ++checks;
                double rel = 0.0;
                bool det = false;
                std::string err;
                const bool ok = run_case(spec, sh.M, sh.K, sh.N, rng, golden, &rel, &det, &err);
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

    run_suite("MMQ", kSpecs, sizeof(kSpecs) / sizeof(kSpecs[0]), Golden::kFp32);
    // Strict pass: same kernels, golden fed the CPU-Q8_K-quantized activations.
    // This is the assertion that pins the activation side of the dot (DS4 d/s
    // pairs, D2S6 partial sums) to the CPU reference semantics.
    run_suite("Q8K", kSpecs, sizeof(kSpecs) / sizeof(kSpecs[0]), Golden::kQ8K);
    run_suite("TC ", kTcSpecs, sizeof(kTcSpecs) / sizeof(kTcSpecs[0]), Golden::kFp32);

    std::printf("\n=== mmq/TC ref: %d/%d cases passed ===\n", checks - failures, checks);
    return failures > 0 ? 1 : 0;
}
