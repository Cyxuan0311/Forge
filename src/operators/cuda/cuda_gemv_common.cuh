#pragma once

#include "cuda_common.h"

// ============================================================================
// IQ-type lookup tables — uploaded lazily by ensure_iq*_tables() in cuda_quant.cu
// Definitions live in cuda_quant.cu; extern declarations in each .cu consumer.
// Do NOT put extern __constant__ in headers — NVCC treats them as per-TU
// static definitions, generating warning #20044-D and wasting constant memory.
// ============================================================================

// ============================================================================
// Shared constants (matching llama.cpp)
// QK_K = 256
// ============================================================================
#define GEMV_QK_K 256
#define GEMV_QI8_1 8

// Q2_K
#define GEMV_QR2_K 4
#define GEMV_QI2_K (GEMV_QK_K / (4 * GEMV_QR2_K))  // 16

// Q3_K
#define GEMV_QR3_K 4
#define GEMV_QI3_K (GEMV_QK_K / (4 * GEMV_QR3_K))  // 16

// Q4_K
#define GEMV_QR4_K 2
#define GEMV_QI4_K (GEMV_QK_K / (4 * GEMV_QR4_K))  // 32

// Q5_K
#define GEMV_QR5_K 2
#define GEMV_QI5_K (GEMV_QK_K / (4 * GEMV_QR5_K))  // 32

// Q6_K
#define GEMV_QR6_K 2
#define GEMV_QI6_K (GEMV_QK_K / (4 * GEMV_QR6_K))  // 32

// ============================================================================
// Q8_1 block layout: half2 ds (d + s) + int8_t qs[32] = 36 bytes
// Compatible with llama.cpp's block_q8_1
// ============================================================================
struct block_q8_1_gemv {
    half2 ds;         // ds.x = d (scale), ds.y = s (sum, unused in GEMV)
    int8_t qs[32];   // quantized values
};

// ============================================================================
// Device helpers for packed reads (matching llama.cpp)
// ============================================================================

// Read 4 consecutive int8 values as a 32-bit int (4-byte aligned)
static __device__ __forceinline__ int get_int_b4(const void* x, const int& i32) {
    return ((const int*)x)[i32];
}

// Read 4 consecutive 2-bit values as a 32-bit int (2-byte aligned)
static __device__ __forceinline__ int get_int_b2(const void* x, const int& i32) {
    const uint16_t* x16 = (const uint16_t*)x;
    int x32 = x16[2 * i32 + 0] << 0;
    x32 |= x16[2 * i32 + 1] << 16;
    return x32;
}

// Lookup 4 4-bit values from a 16-entry int8 table, packing results into 2 ints.
// CUDA-optimized path using __byte_perm PTX instruction.
static __device__ __forceinline__ int2 get_int_from_table_16(const int& q4, const int8_t* table) {
    const uint32_t* table32 = (const uint32_t*)table;
    uint32_t tmp[2];
    const uint32_t sel = (0x32103210 | ((q4 & 0x88888888) >> 1));
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low  = __byte_perm(table32[0], table32[1], q4 >> shift);
        const uint32_t high = __byte_perm(table32[2], table32[3], q4 >> shift);
        tmp[i] = __byte_perm(low, high, sel >> shift);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}

// Unpack 7-bit sign information into 4 byte-wide sign bits (0x80 per byte).
static __device__ __forceinline__ uint32_t unpack_ksigns(const uint32_t packed) {
    return ((packed << 24) & 0x80000000) | ((packed << 15) & 0x00800000) |
           ((packed <<  6) & 0x00008000) | ((packed >>  3) & 0x00000080);
}