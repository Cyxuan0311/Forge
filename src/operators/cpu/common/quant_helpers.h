#pragma once
// Architecture-independent quantization block structures and scalar helpers.
// The SIMD-specific scale-decode helpers (decode_q4_k_scales, get_scale_shuffle*)
// live in arch/<arch>/kernels.h instead — see src/operators/cpu/simd.h.

#include <algorithm>
#include <cmath>
#include <cstdint>
#ifdef _OPENMP
#    include <omp.h>
#endif

namespace forge {
namespace cpu {

// Q8_K block structure (256 elements)
struct block_q8_K {
    float d;
    int8_t qs[256];
    int16_t bsums[16];
};

// Q2_K block structure (256 elements, 84 bytes)
// Layout: scales[16] + qs[64] + d[2] + dmin[2]
struct block_q2_K {
    uint8_t scales[16];
    uint8_t qs[64];
    uint16_t d;
    uint16_t dmin;
};
static_assert(sizeof(block_q2_K) == 84, "block_q2_K must be 84 bytes");

// Q3_K block structure (256 elements, 110 bytes)
// Layout: hmask[32] + qs[64] + scales[12] + d[2]
struct block_q3_K {
    uint8_t hmask[32];   // high bit mask
    uint8_t qs[64];      // low 2-bit quants
    uint8_t scales[12];  // scales, quantized with 6 bits
    uint16_t d;          // super-block scale (fp16)
};
static_assert(sizeof(block_q3_K) == 110, "block_q3_K must be 110 bytes");

// Q4_K block structure (256 elements, 144 bytes)
// Layout: d[2](fp16) + dmin[2](fp16) + scales[12] + qs[128]
struct block_q4_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};
static_assert(sizeof(block_q4_K) == 144, "block_q4_K must be 144 bytes");

// Q5_K block structure (256 elements, 176 bytes)
// Layout: d[2](fp16) + dmin[2](fp16) + scales[12] + qh[32] + ql[128]
struct block_q5_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t ql[128];
};
static_assert(sizeof(block_q5_K) == 176, "block_q5_K must be 176 bytes");

// Q6_K block structure (256 elements, 210 bytes)
struct block_q6_K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    uint16_t d;
};

// Quantize one FP32 row to Q8_K format
static void quantize_row_q8_K(const float* src, block_q8_K* dst, int k) {
    constexpr int QK_K = 256;
    const int nb = (k + QK_K - 1) / QK_K;
    for (int bi = 0; bi < nb; ++bi) {
        int base = bi * QK_K;
        int n_el = std::min(QK_K, k - base);
        float amax = 0.0f;
        for (int j = 0; j < n_el; ++j) {
            float v = std::abs(src[base + j]);
            if (v > amax)
                amax = v;
        }
        float d = amax / 127.0f;
        float id = d > 0.0f ? 1.0f / d : 0.0f;
        dst[bi].d = d;
        int sum[16] = {0};
        for (int j = 0; j < n_el; ++j) {
            int q = (int)(src[base + j] * id + (src[base + j] >= 0 ? 0.5f : -0.5f));
            if (q < -128)
                q = -128;
            if (q > 127)
                q = 127;
            dst[bi].qs[j] = (int8_t)q;
            sum[j / 16] += q;
        }
        for (int j = n_el; j < QK_K; ++j)
            dst[bi].qs[j] = 0;
        for (int j = 0; j < 16; ++j)
            dst[bi].bsums[j] = (int16_t)sum[j];
    }
}

// Round-robin parallel_for
template <typename Fn>
static void parallel_for_steal(int total, int chunk_size, Fn&& fn) {
    if (total <= 0)
        return;
    int n_chunks = (total + chunk_size - 1) / chunk_size;
#ifdef _OPENMP
#    pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int n_threads = omp_get_num_threads();
        for (int c = tid; c < n_chunks; c += n_threads) {
            int start = c * chunk_size;
            int end = start + chunk_size;
            if (end > total)
                end = total;
            fn(start, end);
        }
    }
#else
    fn(0, total);
#endif
}

}  // namespace cpu
}  // namespace forge
