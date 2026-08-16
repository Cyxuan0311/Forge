// Benchmark the decode FFN-up fused Q2_K path (real layer dims) vs pure-read
// bandwidth references. Links against the built forge libs.
//
// Build (match bench_moe_real):
//   g++ -O3 -march=native -flto -fno-fat-lto-objects -fPIE -fopenmp \
//       -std=gnu++20 -DFORGE_ARCH_X86 -DUSE_AVX2 \
//       -Iinclude tests/bench_ffn_q2k.cpp \
//       build/libforge_model.a build/libforge_core.a build/src/operators/libforge_ops.a \
//       build/libforge_infer.a build/libforge_model.a build/src/operators/libforge_ops.a \
//       build/libforge_core.a \
//       /usr/local/cuda-12.8/lib64/libcudart.so /usr/local/cuda-12.8/lib64/libcublas.so \
//       /usr/lib/gcc/x86_64-linux-gnu/11/libgomp.so -lpthread -ldl -o /tmp/bench_ffn_q2k

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

static constexpr int Q2K_BLK = 84;
static constexpr int QK = 256;
static constexpr int K = 4096;
static constexpr int N = 14336;  // 8B intermediate

int main(int argc, char** argv) {
    const int nt = (argc > 1) ? atoi(argv[1]) : 16;
    const int iters = (argc > 2) ? atoi(argv[2]) : 100;
    omp_set_num_threads(nt);
    printf("Q2_K FFN-up decode bench: gate/up [%d, %d], threads=%d\n", N, K, nt);

    std::mt19937 rng(42);
    int nb = (K + QK - 1) / QK;
    size_t row_bytes = (size_t)nb * Q2K_BLK;       // 1344 B/row
    size_t mat_bytes = (size_t)N * row_bytes;       // 19.3 MB per matrix
    size_t total_bytes = 2 * mat_bytes;             // 38.5 MB gate+up

    std::vector<uint8_t> gate(mat_bytes), up(mat_bytes);
    for (auto& b : gate) b = (uint8_t)(rng() >> 16);
    for (auto& b : up) b = (uint8_t)(rng() >> 16);

    std::vector<float> in(K);
    for (auto& v : in) v = (float)((int)(rng() % 2000) - 1000) / 100.0f;

    TensorPtr t_in = std::make_shared<Tensor>(
        Tensor::from_buffer(in.data(), DataType::FP32, std::vector<int64_t>{1, K}));
    TensorPtr t_gate = std::make_shared<Tensor>(Tensor::from_buffer(
        gate.data(), DataType::Q2_K, std::vector<int64_t>{(int64_t)N, (int64_t)K}));
    TensorPtr t_up = std::make_shared<Tensor>(Tensor::from_buffer(
        up.data(), DataType::Q2_K, std::vector<int64_t>{(int64_t)N, (int64_t)K}));

    // Reference 1: memcpy of the same byte volume (machine streaming ceiling)
    std::vector<uint8_t> dst(total_bytes);
    size_t blocks_per_mat = mat_bytes / 4096;  // 4704 blocks per 19.3MB matrix
    double best_memcpy = 1e30;
    for (int r = 0; r < iters; ++r) {
        double t0 = omp_get_wtime();
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < (int64_t)total_bytes / 4096; ++i) {
            const uint8_t* src = (i < (int64_t)blocks_per_mat) ? gate.data() : up.data();
            memcpy(dst.data() + i * 4096, src + (size_t)(i % blocks_per_mat) * 4096, 4096);
        }
        double dt = omp_get_wtime() - t0;
        if (dt < best_memcpy) best_memcpy = dt;
    }

    // Reference 2: pure vectorized read of gate+up rows (no dot compute), same pattern
    double best_read = 1e30;
    uint64_t read_sum = 0;
    for (int r = 0; r < iters; ++r) {
        double t0 = omp_get_wtime();
        uint64_t acc = 0;
        #pragma omp parallel for schedule(static) reduction(+:acc)
        for (int n = 0; n < N; ++n) {
            const uint8_t* gr = gate.data() + (size_t)n * row_bytes;
            const uint8_t* ur = up.data() + (size_t)n * row_bytes;
            for (size_t off = 0; off < row_bytes; off += 64) {
                __m256i a = _mm256_loadu_si256((const __m256i*)(gr + off));
                __m256i b = _mm256_loadu_si256((const __m256i*)(gr + off + 32));
                __m256i c = _mm256_loadu_si256((const __m256i*)(ur + off));
                __m256i d = _mm256_loadu_si256((const __m256i*)(ur + off + 32));
                __m256i s = _mm256_add_epi8(a, _mm256_add_epi8(b, _mm256_add_epi8(c, d)));
                acc += _mm256_extract_epi64(_mm256_sad_epu8(s, _mm256_setzero_si256()), 0);
            }
        }
        read_sum += acc;
        double dt = omp_get_wtime() - t0;
        if (dt < best_read) best_read = dt;
    }

    // Reference 3: real fused FFN-up (decode path)
    double best_real = 1e30;
    for (int r = 0; r < iters; ++r) {
        double t0 = omp_get_wtime();
        auto out = forge::ops::matmul_transB_fused_ffn_up_q2_k(t_in, t_gate, t_up);
        double dt = omp_get_wtime() - t0;
        if (dt < best_real) best_real = dt;
        volatile float sink = *(float*)out->data();
        (void)sink;
    }

    auto gbps = [&](double s) { return total_bytes / (s * 1e9); };
    printf("memcpy ceiling : %8.3f ms   %7.2f GB/s\n", best_memcpy * 1e3, gbps(best_memcpy));
    printf("pure vectorized read : %8.3f ms   %7.2f GB/s\n", best_read * 1e3, gbps(best_read));
    printf("REAL fused FFN-up: %8.3f ms   %7.2f GB/s\n", best_real * 1e3, gbps(best_real));
    printf("real/memcpy = %.3f\n", best_real / best_memcpy);
    return (int)(read_sum & 0xFF);
}
