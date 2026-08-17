// Benchmark the decode output_proj Q6_K path (real llama vocab dims) vs pure-read
// bandwidth references. M=1, K=4096, N=128256 (llama-3 vocab). The output head is
// ~431 MB, read once per token, so decode is dominated by streaming bandwidth.
//
// Build (match bench_ffn_q2k):
//   g++ -O3 -march=native -flto -fno-fat-lto-objects -fPIE -fopenmp \
//       -std=gnu++20 -DFORGE_ARCH_X86 -DUSE_AVX2 \
//       -Iinclude tests/bench_output_proj_q6k.cpp \
//       build/libforge_model.a build/libforge_core.a build/src/operators/libforge_ops.a \
//       build/libforge_infer.a build/libforge_model.a build/src/operators/libforge_ops.a \
//       build/libforge_core.a \
//       /usr/local/cuda-12.8/lib64/libcudart.so /usr/local/cuda-12.8/lib64/libcublas.so \
//       /usr/lib/gcc/x86_64-linux-gnu/11/libgomp.so -lpthread -ldl -o /tmp/bench_output_proj_q6k

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <immintrin.h>
#include <memory>
#include <random>
#include <vector>

#include <omp.h>

#include "forge/operator_matmul.h"

using forge::DataType;
using forge::DeviceType;
using forge::Tensor;
using forge::TensorPtr;

static constexpr int Q6K_BLK = 210;
static constexpr int QK = 256;
static constexpr int K = 4096;
static constexpr int N = 128256;  // llama-3 vocab

int main(int argc, char** argv) {
    const int nt = (argc > 1) ? atoi(argv[1]) : 16;
    const int iters = (argc > 2) ? atoi(argv[2]) : 20;
    omp_set_num_threads(nt);
    printf("Q6_K output_proj decode bench: [%d, %d], threads=%d\n", N, K, nt);

    std::mt19937 rng(42);
    int nb = (K + QK - 1) / QK;
    size_t row_bytes = (size_t)nb * Q6K_BLK;       // 3360 B/row
    size_t total_bytes = (size_t)N * row_bytes;     // ~431 MB

    std::vector<uint8_t> out_w(total_bytes);
    for (auto& b : out_w) b = (uint8_t)(rng() >> 16);

    std::vector<float> in(K);
    for (auto& v : in) v = (float)((int)(rng() % 2000) - 1000) / 100.0f;

    TensorPtr t_in = std::make_shared<Tensor>(
        Tensor::from_buffer(in.data(), DataType::FP32, std::vector<int64_t>{1, K}));
    TensorPtr t_out = std::make_shared<Tensor>(Tensor::from_buffer(
        out_w.data(), DataType::Q6_K, std::vector<int64_t>{(int64_t)N, (int64_t)K}));

    // Reference 1: memcpy of the same byte volume (machine streaming ceiling)
    std::vector<uint8_t> dst(total_bytes);
    double best_memcpy = 1e30;
    for (int r = 0; r < iters; ++r) {
        double t0 = omp_get_wtime();
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < (int64_t)total_bytes / 4096; ++i)
            memcpy(dst.data() + i * 4096, out_w.data() + (size_t)i * 4096, 4096);
        double dt = omp_get_wtime() - t0;
        if (dt < best_memcpy) best_memcpy = dt;
    }

    // Reference 2: pure vectorized read of the output rows (no dot compute)
    double best_read = 1e30;
    uint64_t read_sum = 0;
    for (int r = 0; r < iters; ++r) {
        double t0 = omp_get_wtime();
        uint64_t acc = 0;
        #pragma omp parallel for schedule(static) reduction(+:acc)
        for (int n = 0; n < N; ++n) {
            const uint8_t* gr = out_w.data() + (size_t)n * row_bytes;
            for (size_t off = 0; off < row_bytes; off += 32) {
                __m256i a = _mm256_loadu_si256((const __m256i*)(gr + off));
                __m256i b = _mm256_loadu_si256((const __m256i*)(gr + off + 16));
                __m256i s = _mm256_add_epi8(a, b);
                acc += _mm256_extract_epi64(_mm256_sad_epu8(s, _mm256_setzero_si256()), 0);
            }
        }
        read_sum += acc;
        double dt = omp_get_wtime() - t0;
        if (dt < best_read) best_read = dt;
    }

    // Reference 3: real output_proj (decode path, routes to gemm_q6_K_avx2 M=1)
    double best_real = 1e30;
    for (int r = 0; r < iters; ++r) {
        double t0 = omp_get_wtime();
        auto out = forge::ops::matmul_transB(t_in, t_out);
        double dt = omp_get_wtime() - t0;
        if (dt < best_real) best_real = dt;
        volatile float sink = *(float*)out->data();
        (void)sink;
    }

    auto gbps = [&](double s) { return total_bytes / (s * 1e9); };
    printf("memcpy ceiling : %8.3f ms   %7.2f GB/s\n", best_memcpy * 1e3, gbps(best_memcpy));
    printf("pure vectorized read : %8.3f ms   %7.2f GB/s\n", best_read * 1e3, gbps(best_read));
    printf("REAL output_proj: %8.3f ms   %7.2f GB/s\n", best_real * 1e3, gbps(best_real));
    printf("real/memcpy = %.3f\n", best_real / best_memcpy);
    return (int)(read_sum & 0xFF);
}
