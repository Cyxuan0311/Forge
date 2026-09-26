#pragma once
#include "cpu/common/thread_pool.h"
// Architecture-independent quantization block structures and scalar helpers.
// The SIMD-specific scale-decode helpers (decode_q4_k_scales, get_scale_shuffle*)
// live in arch/<arch>/kernels.h instead — see src/operators/cpu/simd.h.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include "forge/host_mem_pool.h"

namespace forge {
namespace cpu {

// std::vector variant backed by the host memory caching pool. Hot kernels use
// this for their per-call GEMV scratch buffers so blocks are reused (and stay
// 64-byte aligned for SIMD) instead of churning malloc/free every token.
template <typename T>
using scratch_vec = std::vector<T, forge::host_mem::host_allocator<T>>;

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

// llama.cpp nearest_int (round-to-nearest via magic number, ties to even).
static inline int nearest_int_f(float fval) {
    float val = fval + 12582912.0f;
    int i;
    std::memcpy(&i, &val, sizeof(int));
    return (int)(val - 12582912.0f);
}

// Quantize one FP32 row to Q8_K format.
// Mirrors llama.cpp quantize_row_q8_K_ref exactly: iscale = -127/max_signed
// (so d carries the sign of the max-magnitude element), integer range
// [-127, 127], magic-number rounding. Parity with llama.cpp matters because
// the fused dot kernels feed these same integers into identical AVX2 paths —
// with the old Forge convention (d = amax/127, round-half-away) element
// products differed by ±1 on rounding boundaries, which flips near-tie logits
// on low-bpw IQ models.
static void quantize_row_q8_K(const float* src, block_q8_K* dst, int k) {
    constexpr int QK_K = 256;
    const int nb = (k + QK_K - 1) / QK_K;
    for (int bi = 0; bi < nb; ++bi) {
        int base = bi * QK_K;
        int n_el = std::min(QK_K, k - base);
        float max = 0.0f;
        float amax = 0.0f;
        for (int j = 0; j < n_el; ++j) {
            const float ax = std::fabs(src[base + j]);
            if (ax > amax) {
                amax = ax;
                max = src[base + j];
            }
        }
        int sum[16] = {0};
        if (amax == 0.0f) {
            dst[bi].d = 0.0f;
            for (int j = 0; j < QK_K; ++j)
                dst[bi].qs[j] = 0;
        } else {
            const float iscale = -127.0f / max;
            for (int j = 0; j < n_el; ++j) {
                int v = nearest_int_f(iscale * src[base + j]);
                if (v > 127)
                    v = 127;
                dst[bi].qs[j] = (int8_t)v;
                sum[j / 16] += v;
            }
            // Tail (K not a multiple of 256): zeros contribute nothing to the
            // dot products; llama.cpp asserts full blocks.
            for (int j = n_el; j < QK_K; ++j)
                dst[bi].qs[j] = 0;
            dst[bi].d = 1.0f / iscale;
        }
        for (int j = 0; j < 16; ++j)
            dst[bi].bsums[j] = (int16_t)sum[j];
    }
}

// Round-robin parallel_for: chunks are handed out by work stealing, so an
// uneven `fn` cost per chunk still balances across threads.
template <typename Fn>
static void parallel_for_steal(int total, int chunk_size, Fn&& fn) {
    if (total <= 0)
        return;
    int n_chunks = (total + chunk_size - 1) / chunk_size;
cpu::parallel_for_chunks_dynamic(n_chunks, 1, [&](int c0, int c1) {
        for (int c = c0; c < c1; ++c) {
            int start = c * chunk_size;
            int end = start + chunk_size;
            if (end > total)
                end = total;
            fn(start, end);
        }
    });
}

}  // namespace cpu
}  // namespace forge
