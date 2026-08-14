// Phase 1 diagnostic: isolate where the ~1.2ms/call overhead goes in the
// matmul_transB_shared_input / matmul_transB_batched_pairs MoE paths.
//
// Faithful replicas of the real loops (src/operators/cpu/matmul.cpp) with
// component variants:
//   full        - exact replica incl. scratch allocs + quantize + dot region
//   dots_static - dot region only (pre-quantized input), flat idx + div/mod
//   dots_nested - dot region using collapse(2) nested loops (no div/mod)
//   dots_pf     - dots_nested + software prefetch of the next row
//   dots_inline - dots_nested + direct (inlined) dot call instead of fn ptr
//   quantize    - quantize_row_q8_K cost alone (serial)
//   empty_omp   - bare OpenMP region overhead over the same iteration count
//
// Uses IQ2_XS (74-byte blocks), the real MoE expert dtype.
//
// Build:
//   g++ -std=c++17 -O3 -march=native -fopenmp -DFORGE_ARCH_X86 -DUSE_AVX2 \
//       -I. -Iinclude -Isrc/operators/cpu -Isrc/operators/cpu/common \
//       tests/bench_moe_multigemv.cpp src/operators/cpu/common/quant_tables.cpp \
//       -o /tmp/bench_moe
//   /tmp/bench_moe                (hot: same experts, threads 1 & 16)
//   OMP_NUM_THREADS=14 /tmp/bench_moe --rotate

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <omp.h>

#include "operators/cpu/common/quant_helpers.h"
#include "operators/cpu/common/quant_tables.h"
#include "simd.h"

using forge::cpu::block_q8_K;

// ---------------------------------------------------------------- data gen ---

static constexpr int IQ2XS_BLK = 74;
static constexpr int QK_K = 256;

static void fill_iq2xs(std::vector<uint8_t>& w, uint64_t seed) {
    std::mt19937 rng((uint32_t)seed);
    for (size_t bi = 0; bi + IQ2XS_BLK <= w.size(); bi += IQ2XS_BLK) {
        uint16_t d16 = 0x3C00;  // 1.0 fp16
        std::memcpy(w.data() + bi, &d16, 2);
        for (int j = 0; j < 32; ++j) {
            uint16_t v = (uint16_t)(rng() % 65536);
            std::memcpy(w.data() + bi + 2 + 2 * j, &v, 2);
        }
        for (int j = 0; j < 8; ++j) w[bi + 66 + j] = (uint8_t)(rng() % 256);
    }
}

// ---------------------------------------------------------------- variants ---

struct Cfg {
    const char* name;
    int n_mat;    // n_w for shared_input / n_pairs for batched_pairs
    int N;        // rows per matrix
    int K;        // cols per matrix
    int nb;
    size_t row_stride;
    bool pairs;   // batched_pairs-style (M inputs, quantize region) vs shared
};

template <typename DotFn>
static double run_dots(const Cfg& cfg, const std::vector<const uint8_t*>& wptrs,
                       const std::vector<block_q8_K>& q8, const block_q8_K* q8s,
                       float* out, DotFn dot, int variant, int iters) {
    const int total_N = cfg.n_mat * cfg.N;
    double best = 1e30;
    std::vector<float> sink(total_N, 0.0f);
    for (int rep = 0; rep < iters; ++rep) {
        double t0 = omp_get_wtime();
        if (variant == 0) {         // flat idx + div/mod
            #pragma omp parallel for schedule(static)
            for (int idx = 0; idx < total_N; ++idx) {
                int m = idx / cfg.N;
                int n = idx % cfg.N;
                const uint8_t* row = wptrs[m] + (size_t)n * cfg.row_stride;
                out[m * cfg.N + n] = dot(row, q8s + (size_t)m * cfg.nb, cfg.nb);
            }
        } else if (variant == 1) {  // collapse(2), no div/mod
            #pragma omp parallel for schedule(static) collapse(2)
            for (int m = 0; m < cfg.n_mat; ++m)
                for (int n = 0; n < cfg.N; ++n) {
                    const uint8_t* row = wptrs[m] + (size_t)n * cfg.row_stride;
                    out[m * cfg.N + n] = dot(row, q8s + (size_t)m * cfg.nb, cfg.nb);
                }
        } else if (variant == 2) {  // collapse(2) + prefetch next row
            #pragma omp parallel for schedule(static) collapse(2)
            for (int m = 0; m < cfg.n_mat; ++m)
                for (int n = 0; n < cfg.N; ++n) {
                    const uint8_t* row = wptrs[m] + (size_t)n * cfg.row_stride;
                    if (n + 1 < cfg.N) {
                        const uint8_t* next = wptrs[m] + (size_t)(n + 1) * cfg.row_stride;
                        _mm_prefetch((const char*)next, _MM_HINT_T0);
                        _mm_prefetch((const char*)next + 64, _MM_HINT_T0);
                    }
                    out[m * cfg.N + n] = dot(row, q8s + (size_t)m * cfg.nb, cfg.nb);
                }
        } else if (variant == 3) {  // empty region overhead
            #pragma omp parallel for schedule(static)
            for (int idx = 0; idx < total_N; ++idx) {
                sink[idx] = 0.0f;
            }
        } else if (variant == 4) {  // flat idx + prefetch ALL cache lines of next row
            const size_t row_lines = (cfg.row_stride + 63) / 64;
            #pragma omp parallel for schedule(static)
            for (int idx = 0; idx < total_N; ++idx) {
                int m = idx / cfg.N;
                int n = idx - m * cfg.N;
                const uint8_t* row = wptrs[m] + (size_t)n * cfg.row_stride;
                if (n + 1 < cfg.N) {
                    const uint8_t* next = wptrs[m] + (size_t)(n + 1) * cfg.row_stride;
                    for (size_t k = 0; k < row_lines; k += 2)
                        _mm_prefetch((const char*)next + k * 64, _MM_HINT_T0);
                }
                out[m * cfg.N + n] = dot(row, q8s + (size_t)m * cfg.nb, cfg.nb);
            }
        } else if (variant == 5) {  // pure streaming bandwidth: read rows, no dot
            #pragma omp parallel for schedule(static)
            for (int idx = 0; idx < total_N; ++idx) {
                int m = idx / cfg.N;
                int n = idx - m * cfg.N;
                const uint8_t* row = wptrs[m] + (size_t)n * cfg.row_stride;
                uint32_t acc = 0;
                for (size_t k = 0; k < cfg.row_stride; ++k) acc += row[k];
                out[m * cfg.N + n] = (float)acc;
            }
        }
        double dt = omp_get_wtime() - t0;
        if (dt < best) best = dt;
        out[0] = sink[0];
    }
    return best * 1e3;  // ms
}

// Full faithful replica of matmul_transB_batched_pairs (incl. quantize region).
static double run_full_pairs(const Cfg& cfg, const std::vector<const uint8_t*>& wptrs,
                             const float* in_data, float* out, int iters) {
    double best = 1e30;
    for (int rep = 0; rep < iters; ++rep) {
        double t0 = omp_get_wtime();
        std::vector<std::vector<block_q8_K>> q8_bufs(cfg.n_mat,
            std::vector<block_q8_K>(cfg.nb));
        #pragma omp parallel for schedule(static)
        for (int p = 0; p < cfg.n_mat; ++p)
            forge::cpu::quantize_row_q8_K(in_data + (size_t)p * cfg.K, q8_bufs[p].data(), cfg.K);
        int total_N = cfg.n_mat * cfg.N;
        #pragma omp parallel for schedule(static)
        for (int idx = 0; idx < total_N; ++idx) {
            int p = idx / cfg.N;
            int n = idx % cfg.N;
            const uint8_t* row = wptrs[p] + (size_t)n * cfg.row_stride;
            out[p * cfg.N + n] = forge::ops::get_dot_q8k_fn(forge::DataType::IQ2_XS)(
                row, q8_bufs[p].data(), cfg.nb);
        }
        double dt = omp_get_wtime() - t0;
        if (dt < best) best = dt;
    }
    return best * 1e3;
}

// Full faithful replica of matmul_transB_shared_input (serial quantize + region).
static double run_full_shared(const Cfg& cfg, const std::vector<const uint8_t*>& wptrs,
                              const float* in_data, float* out, int iters) {
    double best = 1e30;
    for (int rep = 0; rep < iters; ++rep) {
        double t0 = omp_get_wtime();
        std::vector<block_q8_K> q8_buf(cfg.nb);
        forge::cpu::quantize_row_q8_K(in_data, q8_buf.data(), cfg.K);
        int total_N = cfg.n_mat * cfg.N;
        #pragma omp parallel for schedule(static)
        for (int idx = 0; idx < total_N; ++idx) {
            int w = idx / cfg.N;
            int n = idx % cfg.N;
            const uint8_t* row = wptrs[w] + (size_t)n * cfg.row_stride;
            out[w * cfg.N + n] = forge::ops::get_dot_q8k_fn(forge::DataType::IQ2_XS)(
                row, q8_buf.data(), cfg.nb);
        }
        double dt = omp_get_wtime() - t0;
        if (dt < best) best = dt;
    }
    return best * 1e3;
}

// ---------------------------------------------------------------- driver -----

int main(int argc, char** argv) {
    bool rotate = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--rotate") rotate = true;

    const int iters = 120;

    // Shared-input case: input [1,4096], n_w=4 weights [960,4096] (gate0,up0,gate1,up1)
    const int K_s = 4096, N_s = 960, nb_s = 16;
    const size_t stride_s = (size_t)nb_s * IQ2XS_BLK;

    // Batched-pairs case: input [2,960], n_pairs=2 weights [4096,960]
    const int K_p = 960, N_p = 4096, nb_p = 4;
    const size_t stride_p = (size_t)nb_p * IQ2XS_BLK;

    // Expert pool: 16 experts for cache-rotation realism.
    const int n_exp = 16;
    std::vector<uint8_t> gate_pool((size_t)n_exp * N_s * stride_s);
    std::vector<uint8_t> up_pool((size_t)n_exp * N_s * stride_s);
    std::vector<uint8_t> down_pool((size_t)n_exp * N_p * stride_p);
    fill_iq2xs(gate_pool, 1);
    fill_iq2xs(up_pool, 2);
    fill_iq2xs(down_pool, 3);

    // Fixed experts (hot-cache baseline) = expert 0 + expert 1.
    auto wptr = [&](std::vector<uint8_t>& pool, int e, size_t stride) {
        return pool.data() + (size_t)e * N_s * stride;
    };
    auto wptr_p = [&](std::vector<uint8_t>& pool, int e, size_t stride) {
        return pool.data() + (size_t)e * N_p * stride;
    };

    std::vector<float> in_s(K_s);
    std::vector<float> in_p(2 * K_p);
    std::mt19937 rng(42);
    for (auto& v : in_s) v = (float)((int)(rng() % 2000) - 1000) / 100.0f;
    for (auto& v : in_p) v = (float)((int)(rng() % 2000) - 1000) / 100.0f;

    std::vector<block_q8_K> q8_s(nb_s);
    forge::cpu::quantize_row_q8_K(in_s.data(), q8_s.data(), K_s);
    std::vector<block_q8_K> q8_p(2 * nb_p);
    forge::cpu::quantize_row_q8_K(in_p.data(), q8_p.data(), K_p);
    forge::cpu::quantize_row_q8_K(in_p.data() + K_p, q8_p.data() + nb_p, K_p);

    std::vector<float> out_s((size_t)4 * N_s);
    std::vector<float> out_p((size_t)2 * N_p);

    auto dot = forge::ops::get_dot_q8k_fn(forge::DataType::IQ2_XS);
    auto dot_inline = forge::ops::dot_iq2_xs_q8_K_avx2;

    auto run = [&](const char* label, const std::vector<const uint8_t*>& wptrs,
                   const Cfg& cfg, const block_q8_K* q8s, float* out, int iters) {
        // warm
        run_dots(cfg, wptrs, std::vector<block_q8_K>{}, q8s, out, dot, 0, 4);
        double full = cfg.pairs ? run_full_pairs(cfg, wptrs, in_p.data(), out, iters)
                                : run_full_shared(cfg, wptrs, in_s.data(), out, iters);
        double ds = run_dots(cfg, wptrs, std::vector<block_q8_K>{}, q8s, out, dot, 0, iters);
        double dn = run_dots(cfg, wptrs, std::vector<block_q8_K>{}, q8s, out, dot, 1, iters);
        double dp = run_dots(cfg, wptrs, std::vector<block_q8_K>{}, q8s, out, dot, 2, iters);
        double dr = run_dots(cfg, wptrs, std::vector<block_q8_K>{}, q8s, out, dot, 4, iters);
        double bw = run_dots(cfg, wptrs, std::vector<block_q8_K>{}, q8s, out, dot, 5, iters);
        double di = run_dots(cfg, wptrs, std::vector<block_q8_K>{}, q8s, out, dot_inline, 1, iters);
        double em = run_dots(cfg, wptrs, std::vector<block_q8_K>{}, q8s, out, dot, 3, iters);
        printf("  %-22s full=%8.3f  dots_divmod=%7.3f  nested=%7.3f  nested_pf=%7.3f  "
               "pf_rowfull=%6.3f  bw=%6.3f  inline=%7.3f  omp_empty=%7.3f  ms\n",
               label, full, ds, dn, dp, dr, di, em);
    };

    for (int nt : {1, 16}) {
        omp_set_num_threads(nt);
        printf("\n=== OMP_NUM_THREADS=%d ===\n", nt);

        // quantize cost (serial, 1 thread)
        {
            double best = 1e30;
            for (int rep = 0; rep < 200; ++rep) {
                double t0 = omp_get_wtime();
                forge::cpu::quantize_row_q8_K(in_s.data(), q8_s.data(), K_s);
                forge::cpu::quantize_row_q8_K(in_p.data(), q8_p.data(), K_p);
                forge::cpu::quantize_row_q8_K(in_p.data() + K_p, q8_p.data() + nb_p, K_p);
                double dt = omp_get_wtime() - t0;
                if (dt < best) best = dt;
            }
            printf("  quantize_row_q8_K: K=4096=%.3fms  (2xK=960=%.3fms)\n",
                   best * 0.5e3, best * 0.5e3);
        }

        Cfg cfg_s{"shared_input 4x[960,4096]", 4, N_s, K_s, nb_s, stride_s, false};
        Cfg cfg_p{"batched_pairs 2x[4096,960]", 2, N_p, K_p, nb_p, stride_p, true};

        // Hot: same 2 experts every iteration (L3-resident)
        {
            std::vector<const uint8_t*> wptrs_s = {wptr(gate_pool, 0, stride_s),
                                                    wptr(up_pool, 0, stride_s),
                                                    wptr(gate_pool, 1, stride_s),
                                                    wptr(up_pool, 1, stride_s)};
            std::vector<const uint8_t*> wptrs_p = {wptr_p(down_pool, 0, stride_p),
                                                    wptr_p(down_pool, 1, stride_p)};
            printf("\n  [hot: same 2 experts]\n");
            run("shared_input", wptrs_s, cfg_s, q8_s.data(), out_s.data(), iters);
            run("batched_pairs", wptrs_p, cfg_p, q8_p.data(), out_p.data(), iters);
        }

        // Rotating: random 2 of 16 experts per iteration (DRAM-cold, realistic)
        if (rotate) {
            double best_s = 1e30, best_p = 1e30;
            std::vector<const uint8_t*> wptrs_s(4), wptrs_p(2);
            for (int rep = 0; rep < iters; ++rep) {
                int e0 = (int)(rng() % n_exp);
                int e1 = (int)(rng() % n_exp);
                if (e1 == e0) e1 = (e1 + 1) % n_exp;
                wptrs_s[0] = wptr(gate_pool, e0, stride_s);
                wptrs_s[1] = wptr(up_pool, e0, stride_s);
                wptrs_s[2] = wptr(gate_pool, e1, stride_s);
                wptrs_s[3] = wptr(up_pool, e1, stride_s);
                wptrs_p[0] = wptr_p(down_pool, e0, stride_p);
                wptrs_p[1] = wptr_p(down_pool, e1, stride_p);
                double t0 = omp_get_wtime();
                run_full_shared(cfg_s, wptrs_s, in_s.data(), out_s.data(), 1);
                double dt = omp_get_wtime() - t0;
                if (dt < best_s) best_s = dt;
                t0 = omp_get_wtime();
                run_full_pairs(cfg_p, wptrs_p, in_p.data(), out_p.data(), 1);
                dt = omp_get_wtime() - t0;
                if (dt < best_p) best_p = dt;
            }
            printf("\n  [rotating: random 2 of 16 experts, full replica]\n");
            printf("  shared_input full=%8.3f ms   batched_pairs full=%8.3f ms\n",
                   best_s * 1e3, best_p * 1e3);
        }
    }

    volatile float sink = out_s[0] + out_p[0];
    (void)sink;
    return 0;
}
