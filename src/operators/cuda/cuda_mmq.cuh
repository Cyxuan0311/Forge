#pragma once

// Phase 6: CUDA MMQ (Matrix-Matrix Quantized multiplication) kernel.
// For large M (>32), replaces dequantize-to-FP32 + cuBLAS with on-the-fly
// quantized dot products using dp4a, inspired by llama.cpp's mmq.cu.
//
// Design:
//   - Pre-quantize FP32 activations (src1) to block_q8_1_mmq format
//   - Tile over weight rows (I) and activation rows (J)
//   - Load weight + activation tiles into shared memory
//   - Compute dp4a dot products per tile
//   - Stage A: Q3_K, Q4_K, Q5_K support with dp4a only

#include "cuda_common.h"
#include "cuda_gemv_common.cuh"
#include "forge/types.h"

namespace forge {
namespace cuda {

// ============================================================================
// MMQ Constants
// ============================================================================

#define MMQ_TILE_NE_K  32    // K tile size in elements (same as llama.cpp)
#define MMQ_ITER_K     256   // K iteration step (one Q_K block = 256 elements)
#define MMQ_NWARPS     8     // Warps per CUDA block

// Tile sizes for Ampere+ dp4a path
#define MMQ_I_Q3K  64
#define MMQ_J_Q3K  32
#define MMQ_I_Q4K  64
#define MMQ_J_Q4K  32
#define MMQ_I_Q5K  64
#define MMQ_J_Q5K  32
#define MMQ_I_Q4_0 64
#define MMQ_J_Q4_0 32
#define MMQ_I_Q6_K 64
#define MMQ_J_Q6_K 32
#define MMQ_I_Q2_K 64
#define MMQ_J_Q2_K 32

// Q8_1_mmq block: 4 standard Q8_1 blocks grouped together
// 128 int8 values + 4 half2 (or 4 float) scales = 144 bytes per block
// Covers 128 K-elements per block
// Note: GEMV_QI8_1 = 8 is "ints per scale", not elements per block.
// Standard block_q8_1_gemv has 32 int8 elements + 1 half2 scale = 36 bytes.
// MMQ groups 4 such blocks: 4*32=128 int8 + 4 scales = 144 bytes.
#define MMQ_QK8_1_MMQ  (4 * 32)  // 128

// Q4_0-specific MMQ constants (32-element blocks, not 256 like K-types)
#define MMQ_QR4_0  2   // QR = QK / (4*QI), Q4_0: 32/(4*4) = 2
#define MMQ_QI4_0  4   // ints per Q4_0 scale group (32 elements / 4 bits per value per int = 4)

// tile_y stride: padded to avoid bank conflicts
// Each tile_y row = MMQ_TILE_NE_K + MMQ_TILE_NE_K/QI8_1 = 32 + 4 = 36 ints
#define MMQ_TILE_Y_K   (MMQ_TILE_NE_K + MMQ_TILE_NE_K / GEMV_QI8_1)

// ============================================================================
// block_q8_1_mmq layout
// ============================================================================

// DS layout: determines how scales/sums are stored in block_q8_1_mmq
enum mmq_q8_1_ds_layout {
    MMQ_Q8_1_DS_LAYOUT_D4,    // 4 float32 scales (one per 32 int8 values)
    MMQ_Q8_1_DS_LAYOUT_DS4,   // 4 half2 (d, s) pairs (one per 32 int8 values)
    MMQ_Q8_1_DS_LAYOUT_D2S6,  // 2 half (d) + 6 half (partial sums) — for Q2_K
};

struct block_q8_1_mmq {
    union {
        float  d4[4];    // D4 layout: d0, d1, d2, d3
        half2  ds4[4];   // DS4 layout: (d0,s0), (d1,s1), (d2,s2), (d3,s3)
        half   d2s6[8];  // D2S6 layout: d0, d1, s0, s1, s2, s3, s4, s5
    };
    int8_t qs[MMQ_QK8_1_MMQ];  // 128 quantized int8 values
};
static_assert(sizeof(block_q8_1_mmq) == 144, "Unexpected block_q8_1_mmq size");

// ============================================================================
// tile_x_sizes: shared memory layout per weight type
// ============================================================================

struct tile_x_sizes {
    int qs;  // int32 elements for quantized values
    int dm;  // half2 elements for scale/min (or float for Q3_K)
    int sc;  // int32 elements for scales
};

// Per-type tile_x_sizes for dp4a (from llama.cpp, simplified)
// I = number of weight rows per tile

// Q3_K: qs = I*(2*MMQ_TILE_NE_K + 1), dm = I, sc = I*(MMQ_TILE_NE_K/8 + 1/8)
//   (2* because Q3_K has 2-bit packed values = 2 ints per 32 elements)
template <int I>
static __host__ __device__ constexpr tile_x_sizes mmq_txs_q3_k() {
    return tile_x_sizes{I * (2 * MMQ_TILE_NE_K + 1), I, I * (MMQ_TILE_NE_K / 8) + I / 8};
}

// Q4_K: qs = I*(MMQ_TILE_NE_K + 1), dm = I*MMQ_TILE_NE_K/QI4_K, sc = I*(MMQ_TILE_NE_K/8) + I/8
template <int I>
static __host__ __device__ constexpr tile_x_sizes mmq_txs_q4_k() {
    return tile_x_sizes{I * (MMQ_TILE_NE_K + 1), I * MMQ_TILE_NE_K / GEMV_QI4_K,
                        I * (MMQ_TILE_NE_K / 8) + I / 8};
}

// Q5_K: qs = I*(2*MMQ_TILE_NE_K + 1), dm = I*MMQ_TILE_NE_K/QI5_K, sc = I*(MMQ_TILE_NE_K/8) + I/8
template <int I>
static __host__ __device__ constexpr tile_x_sizes mmq_txs_q5_k() {
    return tile_x_sizes{I * (2 * MMQ_TILE_NE_K + 1), I * MMQ_TILE_NE_K / GEMV_QI5_K,
                        I * (MMQ_TILE_NE_K / 8) + I / 8};
}

// Q4_0: qs = I*(MMQ_TILE_NE_K + 1), dm = I*(MMQ_TILE_NE_K/QI4_0) + I/2, sc = 0
//   (4-bit packed values, one float scale per Q4_0 block of 32 elements;
//    padded so index i*8 + i/4 + kbxd (max 8I - 1) never leaves the region)
template <int I>
static __host__ __device__ constexpr tile_x_sizes mmq_txs_q4_0() {
    return tile_x_sizes{I * (MMQ_TILE_NE_K + 1),
                        I * (MMQ_TILE_NE_K / MMQ_QI4_0) + I / 2,
                        0};
}

// Q6_K: qs = I*(2*MMQ_TILE_NE_K + 1), dm = I*MMQ_TILE_NE_K/QI6_K, sc = I*(MMQ_TILE_NE_K/8) + I/8
//   (6-bit unpacked values in two arrays, float d, int8 scales)
template <int I>
static __host__ __device__ constexpr tile_x_sizes mmq_txs_q6_k() {
    return tile_x_sizes{I * (2 * MMQ_TILE_NE_K + 1), I * MMQ_TILE_NE_K / GEMV_QI6_K,
                        I * (MMQ_TILE_NE_K / 8) + I / 8};
}

// Q2_K: qs = I*(2*MMQ_TILE_NE_K + 1), dm = I*(MMQ_TILE_NE_K + 1), sc = 0
//   (2-bit packed values, half2 pre-multiplied d*scale pairs, no separate scales)
template <int I>
static __host__ __device__ constexpr tile_x_sizes mmq_txs_q2_k() {
    return tile_x_sizes{I * (2 * MMQ_TILE_NE_K + 1),
                        I * (MMQ_TILE_NE_K + 1),
                        0};
}

// Total shared memory for tile_x in bytes
template <int I>
static constexpr int mmq_tile_x_bytes_q3_k() {
    constexpr auto txs = mmq_txs_q3_k<I>();
    return (txs.qs + txs.dm + txs.sc) * 4;
}
template <int I>
static constexpr int mmq_tile_x_bytes_q4_k() {
    constexpr auto txs = mmq_txs_q4_k<I>();
    return (txs.qs + txs.dm + txs.sc) * 4;
}
template <int I>
static constexpr int mmq_tile_x_bytes_q5_k() {
    constexpr auto txs = mmq_txs_q5_k<I>();
    return (txs.qs + txs.dm + txs.sc) * 4;
}
template <int I>
static constexpr int mmq_tile_x_bytes_q4_0() {
    constexpr auto txs = mmq_txs_q4_0<I>();
    return (txs.qs + txs.dm + txs.sc) * 4;
}
template <int I>
static constexpr int mmq_tile_x_bytes_q6_k() {
    constexpr auto txs = mmq_txs_q6_k<I>();
    return (txs.qs + txs.dm + txs.sc) * 4;
}
template <int I>
static constexpr int mmq_tile_x_bytes_q2_k() {
    constexpr auto txs = mmq_txs_q2_k<I>();
    return (txs.qs + txs.dm + txs.sc) * 4;
}

// ============================================================================
// VDR (Values per Dot Round) — how many K-elements are processed per
// inner dp4a call.  Must match llama.cpp definitions.
// ============================================================================

// Q3_K × Q8_1 MMQ: VDR = 8 (process 8 sub-groups of 4 K-elements)
#define VDR_Q3_K_Q8_1_MMQ  8

// Q4_K × Q8_1 MMQ: VDR = 8
#define VDR_Q4_K_Q8_1_MMQ  8

// Q5_K × Q8_1 MMQ: VDR = 8
#define VDR_Q5_K_Q8_1_MMQ  8

// Q4_0 × Q8_1 MMQ: VDR = 4
#define VDR_Q4_0_Q8_1_MMQ  4

// Q6_K × Q8_1 MMQ: VDR = 8
#define VDR_Q6_K_Q8_1_MMQ  8

// Q2_K × Q8_1 MMQ: VDR = 4
#define VDR_Q2_K_Q8_1_MMQ  4

}  // namespace cuda
}  // namespace forge
