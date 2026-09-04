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

// ============================================================================
// Shared Q8_1 × quantized-weight dot products.
//
// These device primitives back the Q8_1 + dp4a GEMV path. They are defined
// here (header-only) so both the decode GEMV kernels (cuda_gemv.cu) and the
// MoE expert GEMV kernel (cuda_moe.cu) can reuse the exact same numerics —
// the MoE path historically re-implemented the dot in scalar form and trailed
// the dp4a main path.
//
// Layout contract: `bq8_1` holds one block_q8_1_gemv per 32 activation
// elements, laid out contiguously as the activation tensor (so block index
// == activation block index). This matches quantize_q8_1_kernel.
// ============================================================================

// Q4_0 block (32 elems, 18 bytes) × one Q8_1 block (32 elems) — 1:1 mapping.
static __device__ __forceinline__ float vec_dot_q4_0_q8_1(
    const uint8_t* __restrict__ block_ptr,
    const block_q8_1_gemv* __restrict__ q8_blk)
{
    uint16_t scale_bits;
    memcpy(&scale_bits, block_ptr, sizeof(uint16_t));
    float scale = __half2float(reinterpret_cast<const __half&>(scale_bits));
    float d8 = __low2float(q8_blk->ds);

    const uint8_t* qs = block_ptr + 2;
    const int* q8_int = (const int*)q8_blk->qs;

    float block_sum = 0.0f;
    int bsum = 0;

#pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint32_t qs4;
        memcpy(&qs4, qs + j * 4, 4);

        int q4_low = qs4 & 0x0F0F0F0F;
        int q4_high = (qs4 >> 4) & 0x0F0F0F0F;

        int u_low = q8_int[j];
        int u_high = q8_int[j + 4];

        block_sum += (float)(forge::cuda::forge_dp4a(q4_low, u_low, 0) + forge::cuda::forge_dp4a(q4_high, u_high, 0));
        bsum += forge::cuda::forge_dp4a(0x01010101, u_low, 0) + forge::cuda::forge_dp4a(0x01010101, u_high, 0);
    }

    block_sum -= 8.0f * (float)bsum;
    return scale * d8 * block_sum;
}

// Q8_0 block (32 elems, 34 bytes) × one Q8_1 block — 1:1 mapping.
static __device__ __forceinline__ float vec_dot_q8_0_q8_1(
    const uint8_t* __restrict__ block_ptr,
    const block_q8_1_gemv* __restrict__ q8_blk)
{
    uint16_t d_bits;
    memcpy(&d_bits, block_ptr, sizeof(uint16_t));
    float d = __half2float(reinterpret_cast<const __half&>(d_bits));
    float d8 = __low2float(q8_blk->ds);

    const int8_t* qs = reinterpret_cast<const int8_t*>(block_ptr + 2);
    const int* q8_int = (const int*)q8_blk->qs;

    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        int qs_int;
        memcpy(&qs_int, qs + j * 4, 4);
        sumi = forge::cuda::forge_dp4a(qs_int, q8_int[j], sumi);
    }

    return d * d8 * (float)sumi;
}

// Q4_K block (256 elems, 144 bytes) × 8 Q8_1 blocks. One call covers one iqs
// step (iqs in 0,2,..,30); the caller iterates the iqs steps within a block.
static __device__ __forceinline__ float vec_dot_q4_K_q8_1(
    const void* __restrict__ vbq, const block_q8_1_gemv* __restrict__ bq8_1,
    const int& kbx, const int& iqs)
{
    const uint8_t* bq4_K = (const uint8_t*)vbq + kbx * 144;

    int v[2];
    int u[2 * GEMV_QR4_K];
    float d8[GEMV_QR4_K];

    const int bq8_offset = GEMV_QR4_K * ((iqs / 2) / (GEMV_QI8_1 / 2));

    const int* q4 = (const int*)(bq4_K + 16 + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    v[0] = q4[0];
    v[1] = q4[4];

    const uint16_t* scales = (const uint16_t*)(bq4_K + 4);
    uint16_t aux[2];
    const int j = bq8_offset / 2;
    if (j < 2) {
        aux[0] = scales[j + 0] & 0x3f3f;
        aux[1] = scales[j + 2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
    }
    const uint8_t* sc = (const uint8_t*)aux;
    const uint8_t* m  = sc + 2;

#pragma unroll
    for (int i = 0; i < GEMV_QR4_K; ++i) {
        const block_q8_1_gemv* bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int* q8 = (const int*)bq8i->qs + ((iqs / 2) % 4);
        u[2 * i + 0] = q8[0];
        u[2 * i + 1] = q8[4];
    }

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < GEMV_QR4_K; ++i) {
        const int v0i = (v[0] >> (4 * i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4 * i)) & 0x0F0F0F0F;

        const int dot1 = forge::cuda::forge_dp4a(v1i, u[2 * i + 1], forge::cuda::forge_dp4a(v0i, u[2 * i + 0], 0));
        const int dot2 = forge::cuda::forge_dp4a(0x01010101, u[2 * i + 1], forge::cuda::forge_dp4a(0x01010101, u[2 * i + 0], 0));

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }

    const float2 dm4f = __half22float2(*(const half2*)bq4_K);
    return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

// Q6_K block (256 elems, 210 bytes) × 8 Q8_1 blocks. One call covers one iqs
// step (iqs in 0..31); the caller iterates the iqs steps within a block.
static __device__ __forceinline__ float vec_dot_q6_K_q8_1(
    const void* __restrict__ vbq, const block_q8_1_gemv* __restrict__ bq8_1,
    const int& kbx, const int& iqs)
{
    const uint8_t* bq6_K = (const uint8_t*)vbq + kbx * 210;

    const int bq8_offset = 2 * GEMV_QR6_K * (iqs / (GEMV_QI6_K / 2)) + (iqs % (GEMV_QI6_K / 2)) / (GEMV_QI6_K / 4);
    const int scale_offset = (GEMV_QI6_K / 4) * (iqs / (GEMV_QI6_K / 2)) + (iqs % (GEMV_QI6_K / 2)) / (GEMV_QI6_K / 8);
    const int vh_shift = 2 * ((iqs % (GEMV_QI6_K / 2)) / (GEMV_QI6_K / 4));

    const int vl = get_int_b2(bq6_K, iqs);
    const int vh = get_int_b2(bq6_K + 128, (GEMV_QI6_K / 4) * (iqs / (GEMV_QI6_K / 2)) + iqs % (GEMV_QI6_K / 4)) >> vh_shift;

    uint16_t d_bits;
    memcpy(&d_bits, bq6_K + 208, 2);
    float d = __half2float(reinterpret_cast<const __half&>(d_bits));

    const int8_t* scales = reinterpret_cast<const int8_t*>(bq6_K + 192) + scale_offset;

    int u[GEMV_QR6_K];
    float d8[GEMV_QR6_K];

#pragma unroll
    for (int i = 0; i < GEMV_QR6_K; ++i) {
        u[i] = get_int_b4(bq8_1[bq8_offset + 2 * i].qs, iqs % GEMV_QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + 2 * i].ds);
    }

    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < GEMV_QR6_K; ++i) {
        const int sc = scales[4 * i];
        const int vil = (vl >> (4 * i)) & 0x0F0F0F0F;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int vi = __vsubss4((vil | vih), 0x20202020);
        sumf += d8[i] * (forge::cuda::forge_dp4a(vi, u[i], 0) * sc);
    }

    return d * sumf;
}