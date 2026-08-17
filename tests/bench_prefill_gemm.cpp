// Benchmark the prefill (M>1) quantized GEMM path against the M=1 decode path
// and verify M>1 correctness (all rows of input identical => output rows must
// match the M=1 gemv result within fp tolerance).
//
// Real prefill dims: K=4096, N=128256 (llama-3). M varies with prompt length;
// a ~200-token prefill uses M=26 for the first layer (26 rows).
//
// Build (match bench_output_proj_q6k):
//   g++ -O3 -march=native -flto -fno-fat-lto-objects -fPIE -fopenmp \
//       -std=gnu++20 -DFORGE_ARCH_X86 -DUSE_AVX2 \
//       -Iinclude tests/bench_prefill_gemm.cpp \
//       build/libforge_model.a build/libforge_core.a build/src/operators/libforge_ops.a \
//       build/libforge_infer.a build/libforge_model.a build/src/operators/libforge_ops.a \
//       build/libforge_core.a \
//       /usr/local/cuda-12.8/lib64/libcudart.so /usr/local/cuda-12.8/lib64/libcublas.so \
//       /usr/lib/gcc/x86_64-linux-gnu/11/libgomp.so -lpthread -ldl -o /tmp/bench_prefill_gemm

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include <omp.h>

#include "forge/operator_matmul.h"
#include "forge/quant_traits.h"

using forge::DataType;
using forge::Tensor;
using forge::TensorPtr;

static constexpr int QK = 256;
static constexpr int K = 4096;
static constexpr int N = 128256;  // llama-3 vocab

static int blk_bytes(DataType t) {
    switch (t) {
        case DataType::Q2_K: return 84;
        case DataType::Q3_K: return 110;
        case DataType::Q4_K: return 144;
        case DataType::Q5_K: return 176;
        case DataType::Q6_K: return 210;
        default: return 0;
    }
}

static void fill_weight(std::vector<uint8_t>& w, DataType t, std::mt19937& rng) {
    const int nb = (K + QK - 1) / QK;
    size_t total = (size_t)N * nb * (size_t)blk_bytes(t);
    w.resize(total);
    for (auto& b : w) b = (uint8_t)(rng() >> 16);
}

static void fill_input(std::vector<float>& a, int M) {
    std::mt19937 rng(7);
    a.resize((size_t)M * K);
    for (auto& v : a) v = (float)((int)(rng() % 2000) - 1000) / 100.0f;
}

int main(int argc, char** argv) {
    const int nt = (argc > 1) ? atoi(argv[1]) : 14;
    const int iters = (argc > 2) ? atoi(argv[2]) : 10;
    const char* dumpdir = (argc > 3) ? argv[3] : nullptr;
    omp_set_num_threads(nt);
    printf("prefill GEMM bench: K=%d N=%d threads=%d dump=%s\n", K, N, nt, dumpdir ? dumpdir : "-");

    const DataType types[] = {DataType::Q2_K, DataType::Q3_K, DataType::Q4_K, DataType::Q5_K, DataType::Q6_K};
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

        // Independent fp64 ground truth via forge's own row dequantizer
        auto verify_fp64 = [&](const uint8_t* wrow0, const float* a_row,
                               const float* om, int M) {
            auto deq = forge::get_dequant_row_fn(t);
            if (!deq) return false;
            const size_t row_bytes = out_w.size() / (size_t)N;
            const int64_t stride = std::max<int64_t>(1, N / 512);
            std::vector<float> row_buf(K);
            double max_rel = 0.0;
            for (int64_t n = 0; n < N; n += stride) {
                deq(wrow0, row_buf.data(), K, (int)n);
                double acc = 0.0;
                for (int k = 0; k < K; ++k) acc += (double)a_row[k] * (double)row_buf[k];
                double scale = std::max(1.0, std::fabs(acc));
                max_rel = std::max(max_rel, std::fabs((double)om[n] - acc) / scale);
            }
            printf("  fp64 ref (512 rows): M=%d max rel = %.3e  %s\n", M, max_rel,
                   max_rel < 1e-4 ? "OK" : "MISMATCH");
            return true;
        };

        // Reference row (M=1) for correctness comparison
        std::vector<float> a1;
        fill_input(a1, 1);
        TensorPtr t_in1 = std::make_shared<Tensor>(
            Tensor::from_buffer(a1.data(), DataType::FP32, std::vector<int64_t>{1, (int64_t)K}));
        auto ref_out = forge::ops::matmul_transB(t_in1, t_out);
        const float* ref = (const float*)ref_out->data();

        // Decode (M=1) timing for the same matrix — machine streaming ceiling reference
        double best_d1 = 1e30;
        for (int r = 0; r < 5; ++r) {
            double t0 = omp_get_wtime();
            auto o = forge::ops::matmul_transB(t_in1, t_out);
            double dt = omp_get_wtime() - t0;
            if (dt < best_d1) best_d1 = dt;
            volatile float sink = *(const float*)o->data();
            (void)sink;
        }
        printf("  decode M=1: %7.2f ms  %7.1f MB/s (weight read once)\n",
               best_d1 * 1e3, out_w.size() / (best_d1 * 1e6));

        printf("--- %s (row=%zu B, total=%.1f MB) ---\n", names[ti], out_w.size() / (size_t)N,
               out_w.size() / 1e6);

        for (int mi = 0; mi < (int)(sizeof(Ms) / sizeof(Ms[0])); ++mi) {
            const int M = Ms[mi];

            // Input rows identical to the reference row => output must match
            std::vector<float> aM;
            fill_input(aM, M);
            for (int r = 1; r < M; ++r) memcpy(&aM[(size_t)r * K], &aM[0], (size_t)K * sizeof(float));
            TensorPtr t_inM = std::make_shared<Tensor>(
                Tensor::from_buffer(aM.data(), DataType::FP32, std::vector<int64_t>{M, (int64_t)K}));

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

            if (M == 26) verify_fp64(out_w.data(), aM.data(), om, M);

            double max_diff = 0.0;
            double max_rel = 0.0;
            for (int64_t n = 0; n < N; ++n) {
                double d = std::fabs((double)om[n] - (double)ref[n]);
                max_diff = std::max(max_diff, d);
                double scale = std::max(1.0, std::fabs((double)ref[n]));
                max_rel = std::max(max_rel, d / scale);
            }
            printf("  M=%3d  max|M1-ref| = %.3e  max rel = %.3e  %s\n", M, max_diff, max_rel,
                   max_rel < 1e-3 ? "OK" : "MISMATCH");

            double best = 1e30;
            double sum = 0;
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
            printf("  M=%3d  best=%7.2f ms  avg=%7.2f ms  %7.1f GFLOPS  %7.1f MB/s\n",
                   M, best * 1e3, (sum / iters) * 1e3, gflops, gbps);
        }
    }
    return 0;
}
