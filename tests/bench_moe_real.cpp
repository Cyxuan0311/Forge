// Phase 1: measure the REAL matmul_transB_shared_input / batched_pairs from
// the built forge library (libforge_ops.a) with real-weight dimensions.
//
// Build (match forge-inspect link):
//   g++ -O3 -march=native -flto -fno-fat-lto-objects -fPIE -fopenmp \
//       -std=gnu++20 -DFORGE_ARCH_X86 -DUSE_AVX2 \
//       -Iinclude tests/bench_moe_real.cpp \
//       build/libforge_model.a build/libforge_core.a build/src/operators/libforge_ops.a \
//       build/libforge_infer.a build/libforge_model.a build/src/operators/libforge_ops.a \
//       build/libforge_core.a \
//       /usr/local/cuda-12.8/lib64/libcudart.so /usr/local/cuda-12.8/lib64/libcublas.so \
//       /usr/lib/gcc/x86_64-linux-gnu/11/libgomp.so -lpthread -ldl -o /tmp/bench_moe_real

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include <omp.h>

#include "forge/operator_matmul.h"

using forge::DataType;
using forge::DeviceType;
using forge::Tensor;
using forge::TensorPtr;

static constexpr int IQ2XS_BLK = 74;
static constexpr int QK = 256;

int main(int argc, char** argv) {
    const int nt = (argc > 1) ? atoi(argv[1]) : 16;
    const int iters = 150;
    omp_set_num_threads(nt);

    std::mt19937 rng(42);

    auto make_iq2xs = [&](int rows, int K) {
        int nb = (K + QK - 1) / QK;
        size_t bytes = (size_t)rows * nb * IQ2XS_BLK;
        std::vector<uint8_t> buf(bytes);
        for (auto& b : buf) b = (uint8_t)(rng() >> 16);
        return buf;
    };

    // shared_input: n_w=4 weights [960, 4096] (gate0,up0,gate1,up1) IQ2_XS
    const int K_s = 4096, N_s = 960, nb_s = 16;
    std::vector<std::vector<uint8_t>> wbuf_s(4);
    std::vector<TensorPtr> w_s(4);
    for (int i = 0; i < 4; ++i) {
        wbuf_s[i] = make_iq2xs(N_s, K_s);
        w_s[i] = std::make_shared<Tensor>(Tensor::from_buffer(
            wbuf_s[i].data(), DataType::IQ2_XS, std::vector<int64_t>{(int64_t)N_s, (int64_t)K_s}));
    }

    // batched_pairs: n_pairs=2 weights [4096, 960] IQ2_XS
    const int K_p = 960, N_p = 4096, nb_p = 4;
    std::vector<std::vector<uint8_t>> wbuf_p(2);
    std::vector<TensorPtr> w_p(2);
    for (int i = 0; i < 2; ++i) {
        wbuf_p[i] = make_iq2xs(N_p, K_p);
        w_p[i] = std::make_shared<Tensor>(Tensor::from_buffer(
            wbuf_p[i].data(), DataType::IQ2_XS, std::vector<int64_t>{(int64_t)N_p, (int64_t)K_p}));
    }

    // inputs
    std::vector<float> in_s(K_s);
    std::vector<float> in_p(2 * K_p);
    for (auto& v : in_s) v = (float)((int)(rng() % 2000) - 1000) / 100.0f;
    for (auto& v : in_p) v = (float)((int)(rng() % 2000) - 1000) / 100.0f;
    TensorPtr t_s = std::make_shared<Tensor>(
        Tensor::from_buffer(in_s.data(), DataType::FP32, std::vector<int64_t>{1, K_s}));
    TensorPtr t_p = std::make_shared<Tensor>(
        Tensor::from_buffer(in_p.data(), DataType::FP32, std::vector<int64_t>{2, K_p}));

    // warm
    for (int i = 0; i < 5; ++i) {
        forge::ops::matmul_transB_shared_input(t_s, w_s);
        forge::ops::matmul_transB_batched_pairs(t_p, w_p);
    }

    double best_s = 1e30, best_p = 1e30;
    for (int rep = 0; rep < iters; ++rep) {
        double t0 = omp_get_wtime();
        auto out_s = forge::ops::matmul_transB_shared_input(t_s, w_s);
        double dt = omp_get_wtime() - t0;
        if (dt < best_s) best_s = dt;

        t0 = omp_get_wtime();
        auto out_p = forge::ops::matmul_transB_batched_pairs(t_p, w_p);
        dt = omp_get_wtime() - t0;
        if (dt < best_p) best_p = dt;
        volatile float sink = *(float*)out_s->data() + *(float*)out_p->data();
        (void)sink;
    }

    printf("OMP_NUM_THREADS=%d  REAL shared_input=%8.3f ms  REAL batched_pairs=%8.3f ms\n",
           nt, best_s * 1e3, best_p * 1e3);
    return 0;
}