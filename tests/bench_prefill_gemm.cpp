// Benchmark the prefill (M>1) quantized GEMM path against the M=1 decode path
// and verify M>1 correctness against an independent host-side fp64 golden
// (reference-suite semantics: global-max normalization over a sampled grid,
// tolerance 5e-2).
//
// Real prefill dims: K=4096, N=128256 (llama-3). M varies with prompt length;
// a ~200-token prefill reaches M=26 at the first llama-3 layer (26 rows).
//
// Built with CUDA (USE_CUDA, see tests/CMakeLists.txt + forge_link_native):
// tensors live on the GPU and the bench sweeps three passes for the Phase 5-C
// auto-rule dtypes (Q4_0 / Q4_K / Q5_K / Q6_K / Q8_0):
//   FORGE_MATMUL_PATH=mmq  -> dp4a (MMQ tiled kernel; Q8_0 gained its real MMQ
//                              instance in Phase 3.5)
//   FORGE_MATMUL_PATH=tc   -> dequant-to-FP16 + cuBLAS FP16 GEMM (Tensor Core)
//   FORGE_MATMUL_PATH=auto -> default dispatch (the Phase 5-C landing state):
//                              per-dtype conservative thresholds
//                              (Q8_0/Q5_K/Q6_K M>=96, Q4_0/Q4_K M>=128) -> TC,
//                              everything else dp4a (see tc_min_m below).
// The auto pass double-checks the default rule against the forced passes.
// The summary prints the tc/dp4a speedup and flags any auto/pass mismatch.
//
// 5-D methodology: every per-M cell reports min(best-of-N)/median/max of the N
// runs (never a single latency), so clock/thermal drift shows up as a visible
// spread instead of a silent artifact; the crossover band rows get a dedicated
// dispersion table. Single-latency numbers are not treated as evidence.
//
// Benchmark methodology: paired/interleaved measurement is preferred for
// backend comparisons; sequential backend sweeps may be affected by GPU
// clock/thermal drift (a whole-pass Q6_K scan once showed a 2.7x artifact that
// disappeared once dp4a/TC were interleaved per M). Use the q6k/q5k scan modes
// for per-M backend crossover characterization.
//
// Without CUDA the CPU (AVX2 decode) variant runs, as before.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <random>
#include <vector>

#include <omp.h>

#include "forge/operator_matmul.h"
#include "forge/quant_traits.h"
#include "forge/types.h"
#include "forge/tensor.h"
#include "forge/kernel_selector.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

using forge::DataType;
using forge::DeviceType;
using forge::Tensor;
using forge::TensorPtr;

static constexpr int K = 4096;
static constexpr int N = 128256;  // llama-3 vocab

// ---- quant block layout -----------------------------------------------------

static int blk_elems(DataType t) {
    switch (t) {
        case DataType::Q4_0:
        case DataType::Q8_0: return 32;
        default: return 256;   // Q2_K / Q3_K / Q4_K / Q5_K / Q6_K
    }
}

static int blk_bytes(DataType t) {
    switch (t) {
        case DataType::Q2_K: return 84;
        case DataType::Q3_K: return 110;
        case DataType::Q4_0: return 18;
        case DataType::Q4_K: return 144;
        case DataType::Q5_K: return 176;
        case DataType::Q6_K: return 210;
        case DataType::Q8_0: return 34;
        default: return 0;
    }
}

// dtypes with an FP16 whole-matrix dequant (Phase 3 + Phase 5-A Q6_K + Phase
// 5-B Q5_K; mirrors kernel_selector's tensorcore_eligible()).
static bool tc_eligible(DataType t) {
    return t == DataType::Q4_0 || t == DataType::Q4_K || t == DataType::Q5_K ||
           t == DataType::Q6_K || t == DataType::Q8_0;
}

// Phase 5-C per-dtype conservative static thresholds -- local mirror of
// kernel_selector.cpp kTcMinM; MUST stay in sync. Auto rule: TensorCore when
// M >= tc_min_m(dtype). Thresholds keep a deliberate safety margin over the
// measured MMQ->TC crossover bands (Q8_0 ~48-64, Q5_K ~64-80, Q6_K ~80,
// Q4_K/Q4_0 ~96-110) and are not the crossover points themselves.
static int tc_min_m(DataType t) {
    switch (t) {
    case DataType::Q8_0:
    case DataType::Q5_K:
    case DataType::Q6_K:
        return 96;
    case DataType::Q4_K:
    case DataType::Q4_0:
    default:
        return 128;
    }
}

static size_t weight_bytes(DataType t) {
    const int be = blk_elems(t);
    const size_t nb = (K + be - 1) / be;
    return (size_t)N * nb * (size_t)blk_bytes(t);
}

// ---- valid-weight generators (finite fp16 scales; random payload) -----------
// Byte layouts mirror the dequantizers; kept identical to test_mmq_reference.

static uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    const uint32_t man = x & 0x7FFFFFu;
    if (exp <= 0) return static_cast<uint16_t>(sign);
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (man >> 13));
}

static void put_f16(uint8_t* p, float v) {
    const uint16_t b = f32_to_f16(v);
    std::memcpy(p, &b, 2);
}

static float urand(std::mt19937& g, float lo, float hi) {
    return lo + (hi - lo) * (static_cast<float>(g() % 100000) / 100000.0f);
}

// Fills ONE quant block with valid fp16 scale(s) + random payload bytes.
static void gen_quant_block(uint8_t* p, DataType t, std::mt19937& rng) {
    const auto randb = [&](uint8_t* q, int n) {
        for (int i = 0; i < n; ++i) q[i] = static_cast<uint8_t>(rng() & 0xFF);
    };
    switch (t) {
        case DataType::Q4_0:  // 18 B: d qs[16]
            put_f16(p, urand(rng, 0.02f, 1.0f));
            randb(p + 2, 16);
            break;
        case DataType::Q8_0:  // 34 B: d qs(int8)[32]
            put_f16(p, urand(rng, 0.02f, 1.0f));
            randb(p + 2, 32);
            break;
        case DataType::Q4_K:  // 144 B: d dmin sc[12] qs[128]
            put_f16(p + 0, urand(rng, 0.02f, 1.0f));
            put_f16(p + 2, urand(rng, 0.01f, 0.3f));
            randb(p + 4, 12);
            randb(p + 16, 128);
            break;
        case DataType::Q5_K:  // 176 B: d dmin sc[12] qh[32] ql[128]
            put_f16(p + 0, urand(rng, 0.02f, 1.0f));
            put_f16(p + 2, urand(rng, 0.01f, 0.3f));
            randb(p + 4, 12);
            randb(p + 16, 32);
            randb(p + 48, 128);
            break;
        case DataType::Q6_K:  // 210 B: ql[128] qh[64] sc(i8)[16] d (mirrors test_mmq_reference)
            randb(p, 128);
            randb(p + 128, 64);
            for (int i = 0; i < 16; ++i) p[192 + i] = (uint8_t)((int)(rng() % 127) - 63);
            put_f16(p + 208, urand(rng, 0.02f, 1.0f));
            break;
        default:
            randb(p, blk_bytes(t));  // legacy K-types: raw bytes
            break;
    }
}

static void fill_weight(std::vector<uint8_t>& w, DataType t, std::mt19937& rng) {
    const int be = blk_elems(t);
    const size_t nb = (K + be - 1) / be;
    const size_t row_bytes = nb * (size_t)blk_bytes(t);
    w.resize((size_t)N * row_bytes);
    for (size_t n = 0; n < (size_t)N; ++n)
        for (size_t b = 0; b < nb; ++b)
            gen_quant_block(w.data() + n * row_bytes + b * (size_t)blk_bytes(t), t, rng);
}

static void fill_input(std::vector<float>& a, int M) {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    a.resize((size_t)M * K);
    for (auto& v : a) v = dist(rng);
}

// Sample ~S columns of the [M,N] output (spread evenly), every row.
static std::vector<int64_t> sample_cols(int S = 2048) {
    const int64_t stride = std::max<int64_t>(1, N / S);
    std::vector<int64_t> cols;
    cols.reserve((size_t)(N / stride));
    for (int64_t n = 0; n < N; n += stride) cols.push_back(n);
    return cols;
}

// Reference-suite semantics: the fp64 golden is compared against the device
// output normalized by the max |value| over the same sampled grid, and noise
// (activation-int8 / FP16) must stay below tol (5e-2, as in the test suite).
static bool verify_fp64(DataType t, const uint8_t* wbase, const float* a, const float* om,
                        int M, double tol) {
    auto deq = forge::get_dequant_row_fn(t);
    if (!deq) return false;
    const size_t row_bytes = weight_bytes(t) / (size_t)N;
    const std::vector<int64_t> cols = sample_cols();
    std::vector<float> row_buf(K);
    double gmax = 0.0, worst = 0.0;
    for (int m = 0; m < M; ++m) {
        const float* ar = a + (size_t)m * K;
        for (int64_t n : cols) {
            deq(wbase + (size_t)n * row_bytes, row_buf.data(), K, /*row=*/0);
            double acc = 0.0;
            for (int k = 0; k < K; ++k) acc += (double)ar[k] * (double)row_buf[k];
            gmax = std::max(gmax, std::fabs(acc));
            worst = std::max(worst, std::fabs((double)om[(size_t)m * N + n] - acc));
        }
    }
    const double rel = gmax > 0.0 ? worst / gmax : 0.0;
    printf("    fp64 ref (M=%d, %zu cols): max rel = %.3e  %s\n", M, cols.size(), rel,
           rel < tol ? "OK" : "MISMATCH");
    return rel < tol;
}

// All M input rows are the M=1 row, so every output row must match the M=1
// decode row on the same sampled columns (suite-style max normalization).
static bool verify_rows_vs_decode(const float* om, int M, const float* ref, double tol) {
    const std::vector<int64_t> cols = sample_cols();
    double gmax = 0.0, worst = 0.0;
    for (int64_t n : cols) gmax = std::max(gmax, std::fabs((double)ref[n]));
    for (int m = 0; m < M; ++m)
        for (int64_t n : cols)
            worst = std::max(worst, std::fabs((double)om[(size_t)m * N + n] - (double)ref[n]));
    const double rel = gmax > 0.0 ? worst / gmax : 0.0;
    printf("    rows vs decode (M=%d, %zu cols): max rel = %.3e  %s\n", M, cols.size(), rel,
           rel < tol ? "OK" : "MISMATCH");
    return rel < tol;
}

#ifdef USE_CUDA

static bool has_cuda_device() {
    int dev = 0;
    return cudaGetDeviceCount(&dev) == cudaSuccess && dev > 0;
}

static bool device_sm80_or_later() {
    int dev = 0, major = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess)
        return false;
    return major >= 8;
}

// Allocate a device tensor and upload `bytes` of host data.
static TensorPtr make_device(const void* host, size_t bytes, DataType t,
                             const std::vector<int64_t>& shape) {
    auto tn = std::make_shared<Tensor>(t, shape, DeviceType::CUDA);
    if (bytes > 0) cudaMemcpy(tn->data(), host, bytes, cudaMemcpyHostToDevice);
    return tn;
}

struct GpuTimer {
    cudaEvent_t t0, t1;
    GpuTimer() {
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);
    }
    ~GpuTimer() {
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
    }
    double run(const TensorPtr& a, const TensorPtr& w, TensorPtr& out) {
        cudaEventRecord(t0, 0);
        out = forge::ops::matmul_transB(a, w);
        cudaEventRecord(t1, 0);
        cudaEventSynchronize(t1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, t0, t1);
        return (double)ms / 1e3;  // seconds
    }
};

static int run_gpu(int iters, bool large) {
    if (!has_cuda_device()) {
        printf("CUDA built but no device found -- skipping GPU sweep.\n");
        return 0;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const bool sm80 = device_sm80_or_later();
    printf("prefill GEMM (CUDA %s, cc %d.%d): K=%d N=%d iters=%d\n", prop.name,
           prop.major, prop.minor, K, N, iters);

    // TC-eligible dtypes (Q4_0/Q4_K/Q8_0 since Phase 3, Q6_K since Phase 5-A,
    // Q5_K since Phase 5-B) carry the TC path; the other K-types (Q2_K/Q3_K)
    // stay on dp4a/MMQ and only run in the CPU variant. `large` mode (arg 3 ==
    // "large") traces the TC-vs-dp4a curve past the threshold plateau for all
    // five auto-rule dtypes.
    static constexpr int MAX_M = 16;
    const DataType types_def[] = {DataType::Q4_K, DataType::Q4_0, DataType::Q8_0,
                                  DataType::Q6_K, DataType::Q5_K};
    const char* names_def[] = {"Q4_K", "Q4_0", "Q8_0", "Q6_K", "Q5_K"};
    const int Ms_def[] = {8, 26, 32, 64, 96, 112, 128, 144, 160, 200};
    const DataType types_lg[] = {DataType::Q4_K, DataType::Q4_0, DataType::Q8_0,
                                 DataType::Q6_K, DataType::Q5_K};
    const char* names_lg[] = {"Q4_K", "Q4_0", "Q8_0", "Q6_K", "Q5_K"};
    const int Ms_lg[] = {128, 256, 384, 512, 768, 1024};
    const DataType* types = large ? types_lg : types_def;
    const char* const* names = large ? names_lg : names_def;
    const int* Ms = large ? Ms_lg : Ms_def;
    const int ntypes = (int)(sizeof(types_def) / sizeof(types_def[0]));
    const int nMs =
        large ? (int)(sizeof(Ms_lg) / sizeof(Ms_lg[0])) : (int)(sizeof(Ms_def) / sizeof(Ms_def[0]));
    constexpr int NT_MAX = 8;

    // dp4a-time / tc-time / auto-time [NT_MAX][nMs], ms. `best` cell = min over
    // `iters` runs (best-of-N); the matching median / max cells expose drift
    // (5-D: report best-of-N + median, never single latencies).
    double dp4a_ms[NT_MAX][MAX_M], tc_ms[NT_MAX][MAX_M], aut_ms[NT_MAX][MAX_M];
    double dp4a_med[NT_MAX][MAX_M], tc_med[NT_MAX][MAX_M], aut_med[NT_MAX][MAX_M];
    double dp4a_max[NT_MAX][MAX_M], tc_max[NT_MAX][MAX_M], aut_max[NT_MAX][MAX_M];
    bool have_tc[NT_MAX];
    for (int i = 0; i < NT_MAX; ++i)
        for (int m = 0; m < nMs; ++m) {
            dp4a_ms[i][m] = tc_ms[i][m] = aut_ms[i][m] = -1.0;
            dp4a_med[i][m] = tc_med[i][m] = aut_med[i][m] = -1.0;
            dp4a_max[i][m] = tc_max[i][m] = aut_max[i][m] = -1.0;
        }
    for (int i = 0; i < NT_MAX; ++i) have_tc[i] = false;

    GpuTimer timer;
    std::mt19937 rng(42);

    for (int ti = 0; ti < ntypes; ++ti) {
        const DataType t = types[ti];
        const char* nm = names[ti];
        const size_t wbytes = weight_bytes(t);
        const bool eligible = tc_eligible(t) && sm80;

        std::vector<uint8_t> out_w;
        fill_weight(out_w, t, rng);
        TensorPtr t_w = make_device(out_w.data(), wbytes, t, {(int64_t)N, (int64_t)K});
        const uint8_t* wbase = out_w.data();

        // Host copies used for verification + input uploads.
        std::vector<float> a1;
        fill_input(a1, 1);
        TensorPtr t_in1 = make_device(a1.data(), a1.size() * sizeof(float), DataType::FP32,
                                      {1, (int64_t)K});

        for (int pass = 0; pass < (eligible ? 3 : 1); ++pass) {
            // pass 0: forced dp4a; pass 1: forced TC; pass 2: default dispatch
            // (unset override -> FORGE_MATMUL_PATH=auto falls through to the
            // per-dtype Phase 5-C rule: TC when M>=tc_min_m(dtype), else dp4a).
            // The auto pass regression-checks the default rule: timings must
            // equal the forced-pass numbers at each M.
            const bool tc = (pass == 1);
            const bool aut = (pass == 2);
            const char* paths[] = {"mmq", "tc", "auto"};
            const char* labels[] = {"dp4a", "TC   ", "auto"};
            setenv("FORGE_MATMUL_PATH", paths[pass], 1);

            printf("--- %s / %s (row=%zu B, total=%.1f MB) ---\n", nm, labels[pass],
                   wbytes / (size_t)N, wbytes / 1e6);
            fflush(stdout);

            // Reference row (M=1) for correctness comparison (decode never uses
            // the TC path, so this is identical across passes).
            TensorPtr ref_out;
            double d_ref = timer.run(t_in1, t_w, ref_out);
            std::vector<float> ref(N);
            cudaMemcpy(ref.data(), ref_out->data(), N * sizeof(float), cudaMemcpyDeviceToHost);
            (void)d_ref;
            if (!tc) verify_fp64(t, wbase, a1.data(), ref.data(), 1, 5e-2);

            // Decode (M=1) timing -- machine streaming ceiling reference.
            // (timer.run syncs the stream; output tensors are dropped.)
            double best_d1 = 1e30;
            for (int r = 0; r < 5; ++r) {
                TensorPtr o;
                double dt = timer.run(t_in1, t_w, o);
                if (dt < best_d1) best_d1 = dt;
            }
            printf("    decode M=1: %7.2f ms  %7.1f MB/s (weight read once)\n", best_d1 * 1e3,
                   (double)wbytes / (best_d1 * 1e6));

            for (int mi = 0; mi < nMs; ++mi) {
                const int M = Ms[mi];
                std::vector<float> aM;
                fill_input(aM, M);
                for (int r = 1; r < M; ++r)
                    memcpy(&aM[(size_t)r * K], &aM[0], (size_t)K * sizeof(float));
                TensorPtr t_inM = make_device(aM.data(), aM.size() * sizeof(float),
                                              DataType::FP32, {M, (int64_t)K});

                TensorPtr out;
                double dt = timer.run(t_inM, t_w, out);
                std::vector<float> om((size_t)M * N);
                cudaMemcpy(om.data(), out->data(), om.size() * sizeof(float),
                           cudaMemcpyDeviceToHost);

                if (M == 32 || M == 128)
                    verify_fp64(t, wbase, aM.data(), om.data(), M, 5e-2);

                // All input rows are the M=1 row, so every output row must match
                // the M=1 decode row on the sampled columns.
                verify_rows_vs_decode(om.data(), M, ref.data(), 5e-2);

                double best = 1e30, sum = 0.0;
                std::vector<double> samples;
                samples.reserve((size_t)iters);
                for (int r = 0; r < iters; ++r) {
                    TensorPtr o;
                    const double dt2 = timer.run(t_inM, t_w, o) * 1e3;  // ms
                    samples.push_back(dt2);
                    if (dt2 < best) best = dt2;
                    sum += dt2;
                }
                std::vector<double> s2 = samples;  // median (sort a copy)
                std::nth_element(s2.begin(), s2.begin() + s2.size() / 2, s2.end());
                const double med = s2[s2.size() / 2];
                const double mx = *std::max_element(samples.begin(), samples.end());
                const double gflops = 2.0 * (double)M * K * N / (best * 1e-3) / 1e9;
                printf("    M=%3d  best=%7.2f  med=%7.2f  max=%7.2f  avg=%7.2f ms"
                       "  %7.1f GFLOPS\n",
                       M, best, med, mx, sum / iters, gflops);

                if (aut) {
                    aut_ms[ti][mi] = best;
                    aut_med[ti][mi] = med;
                    aut_max[ti][mi] = mx;
                } else if (tc) {
                    tc_ms[ti][mi] = best;
                    tc_med[ti][mi] = med;
                    tc_max[ti][mi] = mx;
                } else {
                    dp4a_ms[ti][mi] = best;
                    dp4a_med[ti][mi] = med;
                    dp4a_max[ti][mi] = mx;
                }
            }
            if (tc) have_tc[ti] = true;
        }
    }
    setenv("FORGE_MATMUL_PATH", "mmq", 1);

    // ---- Phase 3.5 summary: forced dp4a / forced TC / default(auto) -------
    // speedup = dp4a/tc. The `auto` column must match the per-dtype Phase 5-C
    // rule: tc at M>=tc_min_m(dtype), dp4a below it.
    printf("\n=== Phase 3.5: TensorCore vs dp4a vs default-auto (ms, best of %d) ===\n", iters);
    printf("dtype   M       dp4a(ms)   tc(ms)    auto(ms)   tc/dp4a\n");
    for (int ti = 0; ti < ntypes; ++ti) {
        if (!have_tc[ti]) continue;
        for (int mi = 0; mi < nMs; ++mi) {
            if (dp4a_ms[ti][mi] < 0 || tc_ms[ti][mi] < 0 || aut_ms[ti][mi] < 0) continue;
            const char* tag = "";
            // auto must sit on the forced path the per-dtype rule predicts
            // (Phase 5-C: Q8_0/Q5_K/Q6_K at M>=96 and Q4_0/Q4_K at M>=128 ->
            // TC, dp4a below). Short kernels on best-of-N runs jitter by
            // ~30-40%, so classify auto by which forced pass it is closer to
            // rather than a fixed % tolerance.
            const bool on_tc_rule = Ms[mi] >= tc_min_m(types[ti]);
            const double dd = std::fabs(aut_ms[ti][mi] - dp4a_ms[ti][mi]);
            const double dt = std::fabs(aut_ms[ti][mi] - tc_ms[ti][mi]);
            const bool auto_on_tc = dt < dd;
            if (auto_on_tc != on_tc_rule) tag = "   <-- auto mismatch";
            printf("%-6s  %-5d   %8.2f   %7.2f   %8.2f   %6.2fx%s\n", names[ti], Ms[mi],
                   dp4a_ms[ti][mi], tc_ms[ti][mi], aut_ms[ti][mi],
                   dp4a_ms[ti][mi] / tc_ms[ti][mi], tag);
        }
    }

    // 5-D dispersion near the crossover band: min(best-of-N)/median/max per
    // backend. Where the auto rule sits between the two forced numbers (64 for
    // Q8_0/Q5_K/Q6_K, 96-112 for Q4_0/Q4_K) is exactly where a single sample is
    // worthless -- this table makes the spread explicit instead.
    const int band[] = {64, 96, 112, 128};
    printf("\n=== 5-D stability near crossover (ms; min(best-of-%d)/median/max) ==="
           "\ndtype  M     dp4a min/med/max           tc min/med/max             "
           "auto min/med/max\n",
           iters);
    for (int ti = 0; ti < ntypes; ++ti) {
        if (!have_tc[ti]) continue;
        for (int b = 0; b < (int)(sizeof(band) / sizeof(band[0])); ++b) {
            int mi = -1;
            for (int j = 0; j < nMs; ++j)
                if (Ms[j] == band[b]) {
                    mi = j;
                    break;
                }
            if (mi < 0 || dp4a_ms[ti][mi] < 0) continue;
            printf("%-6s  %-5d   %7.2f/%-7.2f/%-7.2f  %7.2f/%-7.2f/%-7.2f  "
                   "%7.2f/%-7.2f/%-7.2f\n",
                   names[ti], Ms[mi], dp4a_ms[ti][mi], dp4a_med[ti][mi], dp4a_max[ti][mi],
                   tc_ms[ti][mi], tc_med[ti][mi], tc_max[ti][mi], aut_ms[ti][mi],
                   aut_med[ti][mi], aut_max[ti][mi]);
        }
    }
    return 0;
}

// ---- Phase 5-A/5-B: Q5_K / Q6_K MMQ(dp4a) vs FP16 TensorCore crossover scan ----
// Interleaved-per-M crossover characterization of a K-super-block dtype whose
// two backends both exist (dp4a MMQ kernel + whole-weight FP16 dequant +
// cuBLAS); forced env paths bypass the auto rule. Paired/interleaved
// measurement is the standard method here -- sequential whole-pass sweeps skew
// under GPU clock/thermal drift.
static int run_k_scan(int iters, DataType t, const char* tag, const char* phase) {
    if (!has_cuda_device()) {
        printf("CUDA built but no device found -- skipping %s crossover scan.\n", tag);
        return 0;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (!device_sm80_or_later()) {
        printf("%s TC scan requires an sm80+ device (TensorCore slice).\n", tag);
        return 0;
    }
    printf("%s TC-vs-MMQ scan (CUDA %s, cc %d.%d): K=%d N=%d iters=%d\n", tag, prop.name,
           prop.major, prop.minor, K, N, iters);
    if (iters < 1) iters = 1;
    const int Ms[] = {8, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 200};
    const int nMs = (int)(sizeof(Ms) / sizeof(Ms[0]));

    GpuTimer timer;
    std::mt19937 rng(42);
    std::vector<uint8_t> out_w;
    fill_weight(out_w, t, rng);
    const size_t wbytes = out_w.size();
    TensorPtr t_w = make_device(out_w.data(), wbytes, t, {(int64_t)N, (int64_t)K});
    const uint8_t* wbase = out_w.data();

    std::vector<float> a1;
    fill_input(a1, 1);
    TensorPtr t_in1 = make_device(a1.data(), a1.size() * sizeof(float), DataType::FP32,
                                  {1, (int64_t)K});
    TensorPtr ref_out;
    (void)timer.run(t_in1, t_w, ref_out);
    std::vector<float> ref(N);
    cudaMemcpy(ref.data(), ref_out->data(), N * sizeof(float), cudaMemcpyDeviceToHost);

    double mmq_ms[16], tc_ms[16];
    int fails = 0;
    // Interleave both passes per M (mmq then tc, back to back) so a long-run
    // GPU clock/thermal drift cannot skew one whole pass column against the
    // other; best-of-N still jitters, so cross-check with the Phase 3.5 parity
    // sweep (`forge-bench-prefill_gemm 14 3`) before setting a default rule.
    printf("--- %s (row=%zu B, total=%.1f MB), dp4a & TC interleaved per M ---\n", tag,
           wbytes / (size_t)N, wbytes / 1e6);
    fflush(stdout);
    for (int mi = 0; mi < nMs; ++mi) {
        const int M = Ms[mi];
        std::vector<float> aM;
        fill_input(aM, M);
        for (int r = 1; r < M; ++r)
            memcpy(&aM[(size_t)r * K], &aM[0], (size_t)K * sizeof(float));
        TensorPtr t_inM = make_device(aM.data(), aM.size() * sizeof(float), DataType::FP32,
                                      {M, (int64_t)K});
        for (int pass = 0; pass < 2; ++pass) {
            setenv("FORGE_MATMUL_PATH", pass == 0 ? "mmq" : "tc", 1);
            TensorPtr out;
            (void)timer.run(t_inM, t_w, out);
            std::vector<float> om((size_t)M * N);
            cudaMemcpy(om.data(), out->data(), om.size() * sizeof(float), cudaMemcpyDeviceToHost);
            if (!verify_fp64(t, wbase, aM.data(), om.data(), M, 5e-2)) ++fails;
            if (!verify_rows_vs_decode(om.data(), M, ref.data(), 5e-2)) ++fails;
            double best = 1e30;
            for (int r = 0; r < iters; ++r) {
                TensorPtr o;
                double dt = timer.run(t_inM, t_w, o);
                if (dt < best) best = dt;
            }
            if (pass == 0)
                mmq_ms[mi] = best * 1e3;
            else
                tc_ms[mi] = best * 1e3;
        }
    }
    setenv("FORGE_MATMUL_PATH", "auto", 1);

    printf("\n=== %s: %s MMQ(dp4a) vs FP16 TensorCore (ms, best of %d) ===\n", phase, tag, iters);
    printf("M        dp4a(ms)    tc(ms)     tc/dp4a\n");
    int crossover = -1;
    bool crossed = false;
    for (int mi = 0; mi < nMs; ++mi) {
        const double r = mmq_ms[mi] / tc_ms[mi];
        const bool tc_wins = r > 1.0;
        if (!crossed && tc_wins) {
            crossed = true;
            crossover = Ms[mi];
        }
        printf("  %-5d   %8.2f   %8.2f   %7.2fx%s\n", Ms[mi], mmq_ms[mi], tc_ms[mi], r,
               tc_wins ? "  <- TC wins" : "");
    }
    if (crossover > 0)
        printf("crossover: MMQ->TC at M=%d (auto threshold candidate)\n", crossover);
    else
        printf("crossover: none measured in this range\n");
    return fails > 0 ? 1 : 0;
}

// ---- 4-B convergence validation ---------------------------------------------
// Phase 4-B autotune is opt-in; this measures what it costs and what it gives
// back on a realistic repeated-prefill workload (per-layer projection shapes
// of a 200-token prompt; two sequences per pass so every signature recurs):
//   1) autotune OFF            -> default rule baseline (M>=128: TC, else dp4a)
//   2) autotune ON, first run  -> cold cache: the first encounter of each of
//                                 the 30 signatures pays the 6-probe (3x MMQ +
//                                 3x TC) measurement before returning
//   3) autotune ON, second run -> warm cache: cache hits only (steady state)
//   4) forced DP4A / TC        -> pure-kernel references (bypass the cache)
// Then it lists the measured winner of every signature from the process cache
// and re-verifies the below-threshold TensorCore flips against the fp64 golden.
static int run_conv_validation(int iters) {
    if (!has_cuda_device()) {
        printf("CUDA built but no device found -- skipping 4-B conv validation.\n");
        return 0;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (!device_sm80_or_later()) {
        printf("4-B conv validation requires an sm80+ device (TensorCore slice).\n");
        return 0;
    }
    printf("4-B/5-D convergence validation (CUDA %s, cc %d.%d): K=%d N=%d passes=%d\n",
           prop.name, prop.major, prop.minor, K, N, iters);
    if (iters < 1) iters = 1;

    // Phase 4-B slice + Phase 5-A Q6_K + Phase 5-B Q5_K: 5 dtypes, 10 shapes ->
    // 50 signatures.
    const DataType types[] = {DataType::Q4_0, DataType::Q4_K, DataType::Q8_0, DataType::Q6_K,
                              DataType::Q5_K};
    const char* names[] = {"Q4_0", "Q4_K", "Q8_0", "Q6_K", "Q5_K"};
    const int ntypes = (int)(sizeof(types) / sizeof(types[0]));
    const int Ms[] = {8, 26, 32, 64, 96, 112, 128, 144, 160, 200};
    const int nMs = (int)(sizeof(Ms) / sizeof(Ms[0]));
    const int nseq = 2;  // two sequences per pass -> signatures recur
    GpuTimer timer;
    std::mt19937 rng(42);

    std::vector<TensorPtr> t_w(ntypes);
    std::vector<std::vector<uint8_t>> whost(ntypes);
    std::vector<std::vector<TensorPtr>> t_in(ntypes, std::vector<TensorPtr>(nMs));
    std::vector<std::vector<std::vector<float>>> ahost(ntypes, std::vector<std::vector<float>>(nMs));

    for (int ti = 0; ti < ntypes; ++ti) {
        fill_weight(whost[ti], types[ti], rng);
        t_w[ti] = make_device(whost[ti].data(), whost[ti].size(), types[ti],
                              {(int64_t)N, (int64_t)K});
        for (int mi = 0; mi < nMs; ++mi) {
            const int M = Ms[mi];
            std::vector<float> aM;
            fill_input(aM, M);
            for (int r = 1; r < M; ++r)
                memcpy(&aM[(size_t)r * K], &aM[0], (size_t)K * sizeof(float));
            t_in[ti][mi] = make_device(aM.data(), aM.size() * sizeof(float), DataType::FP32,
                                       {M, (int64_t)K});
            ahost[ti][mi] = std::move(aM);
        }
    }

    const auto run_workload = [&]() -> double {  // ms, one full pass
        double ms = 0.0;
        for (int ti = 0; ti < ntypes; ++ti)
            for (int sq = 0; sq < nseq; ++sq)
                for (int mi = 0; mi < nMs; ++mi) {
                    TensorPtr out;
                    ms += timer.run(t_in[ti][mi], t_w[ti], out) * 1e3;
                }
        return ms;
    };
    const auto best_of = [&](int n) -> double {
        double b = 1e30;
        for (int r = 0; r < n; ++r) {
            const double d = run_workload();
            if (d < b) b = d;
        }
        return b;
    };
    // 5-D: whole-pass sampling + min/median/max/avg (report best-of-N and
    // median, never single latencies, to expose clock/thermal drift).
    const auto sample_n = [&](int n, std::vector<double>& out) {
        out.clear();
        out.reserve((size_t)n);
        for (int r = 0; r < n; ++r) out.push_back(run_workload());
    };
    const auto vec_stats = [](const std::vector<double>& v) {
        std::vector<double> c = v;
        std::nth_element(c.begin(), c.begin() + c.size() / 2, c.end());
        double sum = 0.0;
        for (double x : v) sum += x;
        return std::array<double, 4>{*std::min_element(v.begin(), v.end()),
                                     c[c.size() / 2],
                                     *std::max_element(v.begin(), v.end()),
                                     sum / (double)v.size()};  // min, median, max, avg
    };
    const auto env_rule = [] {
        setenv("FORGE_MATMUL_PATH", "auto", 1);
        unsetenv("FORGE_AUTOTUNE");
    };
    const auto env_autotune = [] {
        setenv("FORGE_MATMUL_PATH", "auto", 1);
        setenv("FORGE_AUTOTUNE", "1", 1);
    };
    const auto env_forced = [](const char* path) {
        setenv("FORGE_MATMUL_PATH", path, 1);
        unsetenv("FORGE_AUTOTUNE");
    };

    // (1) autotune OFF: default rule, cold cache (rule results get cached too).
    env_rule();
    forge::kernel_cache_reset();
    (void)best_of(1);  // warm-up (GPU clocks)
    std::vector<double> v_rule;
    sample_n(iters, v_rule);
    const double t_rule = *std::min_element(v_rule.begin(), v_rule.end());

    // (2) autotune ON, first run: cold cache -> each of the 50 signatures is
    // measured exactly once. Timed directly (best-of would hide the probes).
    env_autotune();
    forge::kernel_cache_reset();
    const double t_first = run_workload();
    // (3) autotune ON, second run: warm cache, steady state.
    std::vector<double> v_steady;
    sample_n(iters, v_steady);
    const double t_steady = *std::min_element(v_steady.begin(), v_steady.end());
    const forge::KernelCacheStats st = forge::kernel_cache_stats();

    // (4) Read the dispatch decision of every signature back from the warm
    // process cache while autotune is still active (every row is a cache hit,
    // so the read-back cannot change anything) and split it into the *static*
    // per-dtype policy vs the *effective* measured winner (5-D): a reader can
    // see exactly which layer produced each decision. matmul keys signatures
    // with the default sm_arch (0 = detect at selection time) -- mirrored below.
    std::vector<std::array<char, 128>> winners;
    for (int ti = 0; ti < ntypes; ++ti) {
        for (int mi = 0; mi < nMs; ++mi) {
            forge::KernelProblem p;
            p.domain = forge::KernelDomain::MatMul;
            p.dtype = types[ti];
            p.M = Ms[mi];
            p.N = N;
            p.K = K;
            bool hit = false;
            const forge::KernelVariant v = forge::select_kernel_cached(p, &hit);
            const bool meas_tc = (v == forge::KernelVariant::MMQ_TENSORCORE);
            const bool rule_tc = Ms[mi] >= tc_min_m(types[ti]);  // Phase 5-C static
            const char* stag = rule_tc ? "TC" : "MMQ";           // static policy
            const char* etag = meas_tc ? "TC" : "MMQ";           // effective
            const char* src = hit ? "autotune-cache" : "static";
            const char* tag = meas_tc == rule_tc
                                  ? "matches static"
                                  : (meas_tc ? "** overrides static (measured TC)"
                                             : "** static TC, measured MMQ");
            std::array<char, 128> row;
            snprintf(row.data(), row.size(), "%-6s %-5d  %-4s   %-4s  %-15s %s", names[ti],
                     Ms[mi], stag, etag, src, tag);
            winners.push_back(std::move(row));
        }
    }

    // (5) Forced DP4A / TC references (bypass the cache entirely): this is the
    // 5-D highest-priority invariant -- a forced path wins over static rule and
    // autotune cache by construction (it short-circuits before any cache
    // consult), and these two passes time out the same whether the cache holds
    // measured TC or MMQ records or is empty.
    env_forced("mmq");
    (void)best_of(1);  // discard first pass (kernel/handle warm-up)
    std::vector<double> v_mmq;
    sample_n(iters, v_mmq);
    const double t_mmq = *std::min_element(v_mmq.begin(), v_mmq.end());
    env_forced("tc");
    (void)best_of(1);
    std::vector<double> v_tc;
    sample_n(iters, v_tc);
    const double t_tc = *std::min_element(v_tc.begin(), v_tc.end());

    const double overhead = t_first - t_steady;
    const double per_sig = st.measured > 0 ? overhead / (double)st.measured : 0.0;
    const auto st_rule = vec_stats(v_rule);
    const auto st_steady = vec_stats(v_steady);
    const auto st_mmq = vec_stats(v_mmq);
    const auto st_tc = vec_stats(v_tc);
    printf("\n=== 4-B/5-D convergence: aggregate ms over one workload pass "
           "(min/median/max over %d whole passes each; not single latencies) ===\n",
           iters);
    const char* hdr = "%-28s min %9.1f  median %9.1f  max %9.1f ms   %5.2fx   %s\n";
    printf(hdr, "1 rule (autotune OFF)", st_rule[0], st_rule[1], st_rule[2], 1.0,
           "(per-dtype Phase 5-C static policy)");
    printf("%-28s %9.1f ms (one cold pass)                       %5.2fx   %s\n",
           "2 autotune ON, first run", t_first, t_first / t_rule,
           "(cold cache: every signature measured)");
    printf(hdr, "3 autotune ON, steady", st_steady[0], st_steady[1], st_steady[2],
           st_steady[0] / t_rule, "(warm cache: hits only)");
    printf(hdr, "4 forced DP4A", st_mmq[0], st_mmq[1], st_mmq[2], st_mmq[0] / t_rule,
           "(forced, bypasses cache)");
    printf(hdr, "5 forced TC", st_tc[0], st_tc[1], st_tc[2], st_tc[0] / t_rule,
           "(forced, bypasses cache)");
    printf("\nautotune first-run overhead : %8.1f ms over %ld measured signatures"
           " (%.2f ms/signature)\n", overhead, st.measured, per_sig);
    printf("steady-state speedup       : %8.1f ms -> %.1f ms vs rule (+%.1f%%)\n",
           t_rule, t_steady, (t_rule / t_steady - 1.0) * 100.0);
    printf("\nKernel Cache (process-local)\n");
    printf("--------------------------------------------\n");
    printf("lookups             %7ld\n", st.lookups);
    printf("hits                %7ld   (hit rate %.1f%%)\n", st.hits,
           100.0 * st.hits / (double)(st.lookups > 0 ? st.lookups : 1));
    printf("misses              %7ld\n", st.misses);
    printf("autotunes measured  %7ld\n", st.measured);
    printf("  TC selected       %7ld\n", st.tc_selected);
    printf("  DP4A selected     %7ld\n", st.dp4a_selected);

    printf("\n--- per-signature dispatch decision (static rule vs effective) ---\n");
    printf("dtype  M    static effective source          tag\n");
    for (const auto& r : winners) printf("%s\n", r.data());

    // fp64 re-verification under the *measured* dispatch (cache is still warm,
    // autotune re-enabled so hits drive the kernel): an autotune flip changes
    // which kernel runs, never the correctness bar.
    env_autotune();
    printf("\n--- fp64 re-verification of below-threshold measured winners (tol 5e-2) ---\n");
    const int flips[] = {64, 96, 112};
    bool all_ok = true;
    std::vector<float> om((size_t)112 * N);
    for (int ti = 0; ti < ntypes; ++ti) {
        for (int f : flips) {
            int mi = -1;
            for (int j = 0; j < nMs; ++j)
                if (Ms[j] == f) mi = j;
            TensorPtr out;
            (void)timer.run(t_in[ti][mi], t_w[ti], out);
            const int M = f;
            cudaMemcpy(om.data(), out->data(), (size_t)M * N * sizeof(float),
                       cudaMemcpyDeviceToHost);
            if (!verify_fp64(types[ti], whost[ti].data(), ahost[ti][mi].data(), om.data(), M, 5e-2))
                all_ok = false;
        }
    }
    env_rule();
    forge::kernel_cache_reset();
    return all_ok ? 0 : 1;
}

#else  // !USE_CUDA -- original CPU (AVX2) sweep

static int run_cpu(int nt, int iters, const char* dumpdir) {
    omp_set_num_threads(nt);
    const DataType types[] = {DataType::Q2_K, DataType::Q3_K, DataType::Q4_K,
                              DataType::Q5_K, DataType::Q6_K};
    const char* names[] = {"Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"};
    const int ntypes = (int)(sizeof(types) / sizeof(types[0]));
    const int Ms[] = {8, 26, 64, 128};
    std::mt19937 rng(42);

    for (int ti = 0; ti < ntypes; ++ti) {
        const DataType t = types[ti];
        std::vector<uint8_t> out_w;
        fill_weight(out_w, t, rng);
        TensorPtr t_out = std::make_shared<Tensor>(
            Tensor::from_buffer(out_w.data(), t, std::vector<int64_t>{(int64_t)N, (int64_t)K}));

        std::vector<float> a1;
        fill_input(a1, 1);
        TensorPtr t_in1 = std::make_shared<Tensor>(
            Tensor::from_buffer(a1.data(), DataType::FP32, std::vector<int64_t>{1, (int64_t)K}));
        auto ref_out = forge::ops::matmul_transB(t_in1, t_out);
        const float* ref = (const float*)ref_out->data();

        double best_d1 = 1e30;
        for (int r = 0; r < 5; ++r) {
            double t0 = omp_get_wtime();
            auto o = forge::ops::matmul_transB(t_in1, t_out);
            double dt = omp_get_wtime() - t0;
            if (dt < best_d1) best_d1 = dt;
            volatile float sink = *(const float*)o->data();
            (void)sink;
        }
        printf("  decode M=1: %7.2f ms  %7.1f MB/s (weight read once)\n", best_d1 * 1e3,
               out_w.size() / (best_d1 * 1e6));
        printf("--- %s (row=%zu B, total=%.1f MB) ---\n", names[ti], out_w.size() / (size_t)N,
               out_w.size() / 1e6);

        for (int mi = 0; mi < (int)(sizeof(Ms) / sizeof(Ms[0])); ++mi) {
            const int M = Ms[mi];
            std::vector<float> aM;
            fill_input(aM, M);
            for (int r = 1; r < M; ++r)
                memcpy(&aM[(size_t)r * K], &aM[0], (size_t)K * sizeof(float));
            TensorPtr t_inM = std::make_shared<Tensor>(
                Tensor::from_buffer(aM.data(), DataType::FP32,
                                    std::vector<int64_t>{M, (int64_t)K}));

            auto out = forge::ops::matmul_transB(t_inM, t_out);
            const float* om = (const float*)out->data();

            if (dumpdir && M == 26) {
                char fn[256];
                snprintf(fn, sizeof(fn), "%s/%s_M26.bin", dumpdir, names[ti]);
                FILE* f = fopen(fn, "wb");
                fwrite(om, sizeof(float), (size_t)N, f);
                fclose(f);
                snprintf(fn, sizeof(fn), "%s/%s_REF.bin", dumpdir, names[ti]);
                f = fopen(fn, "wb");
                fwrite(ref, sizeof(float), (size_t)N, f);
                fclose(f);
                snprintf(fn, sizeof(fn), "%s/%s_ROW1.bin", dumpdir, names[ti]);
                f = fopen(fn, "wb");
                fwrite(om + N, sizeof(float), (size_t)N, f);
                fclose(f);
                printf("  dumped %s\n", fn);
            }

            if (M == 26)
                verify_fp64(t, out_w.data(), aM.data(), om, M, 5e-2);

            // All M input rows are the M=1 row, so every output row must match.
            verify_rows_vs_decode(om, M, ref, 5e-2);

            double best = 1e30, sum = 0.0;
            for (int r = 0; r < iters; ++r) {
                double t0 = omp_get_wtime();
                auto o = forge::ops::matmul_transB(t_inM, t_out);
                double dt = omp_get_wtime() - t0;
                if (dt < best) best = dt;
                sum += dt;
                volatile float sink = *(const float*)o->data();
                (void)sink;
            }
            double gflops = 2.0 * (double)M * K * N / (best * 1e9);
            double gbps = out_w.size() / (best * 1e6);
            printf("  M=%3d  best=%7.2f ms  avg=%7.2f ms  %7.1f GFLOPS  %7.1f MB/s\n", M,
                   best * 1e3, (sum / iters) * 1e3, gflops, gbps);
        }
    }
    return 0;
}

#endif  // USE_CUDA

int main(int argc, char** argv) {
    const int nt = (argc > 1) ? atoi(argv[1]) : 14;
    const int iters = (argc > 2) ? atoi(argv[2]) : 10;
    const char* dumpdir = (argc > 3) ? argv[3] : nullptr;
#ifdef USE_CUDA
    (void)nt;
    (void)dumpdir;
    if ((argc > 3) && strcmp(argv[3], "conv") == 0) {
        printf("prefill GEMM bench: 4-B convergence validation (passes=%d)\n", iters);
        return run_conv_validation(iters);
    }
    if ((argc > 3) && strcmp(argv[3], "q6k") == 0) {
        printf("prefill GEMM bench: Phase 5-A Q6_K TC-vs-MMQ scan (iters=%d)\n", iters);
        return run_k_scan(iters, DataType::Q6_K, "Q6_K", "Phase 5-A");
    }
    if ((argc > 3) && strcmp(argv[3], "q5k") == 0) {
        printf("prefill GEMM bench: Phase 5-B Q5_K TC-vs-MMQ scan (iters=%d)\n", iters);
        return run_k_scan(iters, DataType::Q5_K, "Q5_K", "Phase 5-B");
    }
    const bool large = (argc > 3) && strcmp(argv[3], "large") == 0;
    printf("prefill GEMM bench: K=%d N=%d iters=%d mode=%s\n", K, N, iters,
           large ? "large-M plateau" : "crossover fine-sweep");
    return run_gpu(iters, large);
#else
    printf("prefill GEMM bench: K=%d N=%d threads=%d dump=%s\n", K, N, nt,
           dumpdir ? dumpdir : "-");
    return run_cpu(nt, iters, dumpdir);
#endif
}
