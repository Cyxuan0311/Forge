// Phase 6: CUDA MMQ kernel implementation.
// Quantized matrix-matrix multiplication for large M (>32).
// Replaces dequantize-to-FP32 + cuBLAS with on-the-fly dp4a dot products.
//
// Supported types (Stage A): Q3_K, Q4_K, Q5_K
//
// Flow:
//   1. Pre-quantize FP32 activations (src1) → block_q8_1_mmq format
//   2. MMQ kernel: tile over weight rows (I) × activation rows (J)
//   3. For each K-tile: load weight+activation into shared memory, dp4a dot product
//   4. Write accumulated results to output

#include "cuda_mmq.cuh"

namespace forge {
namespace cuda {

// ============================================================================
// Quantize FP32 → block_q8_1_mmq
// ============================================================================

// Each thread quantizes 4 consecutive FP32 values into 4 int8 values.
// A group of vals_per_scale/4 threads cooperates to compute the scale
// for one block of 32 int8 values (standard Q8_1).
// For D4 layout: 4 float scales (one per 32 values).
// For DS4 layout: 4 half2 (d, s) pairs (one per 32 values).

template <mmq_q8_1_ds_layout ds_layout>
static __global__ void quantize_q8_1_mmq_kernel(
    const float* __restrict__ x, void* __restrict__ vy,
    int K, int M, int K_padded)
{
    // Each thread processes 4 consecutive values
    const int i0 = ((int64_t)blockDim.x * blockIdx.y + threadIdx.x) * 4;

    if (i0 >= K_padded) return;

    const int m = blockIdx.x;  // Row index

    const float4* x4 = (const float4*)x;
    block_q8_1_mmq* y = (block_q8_1_mmq*)vy;

    const int64_t k_block = i0 / MMQ_QK8_1_MMQ;  // Which MMQ block
    const int64_t iqs     = i0 % MMQ_QK8_1_MMQ;  // Offset within MMQ block

    // Load 4 floats
    float4 xi;
    if (i0 < K) {
        xi = x4[(int64_t)m * K / 4 + i0 / 4];
    } else {
        xi = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // Compute max abs value across 32 values (8 threads cooperate)
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, 32));
    }

    // Compute sum for DS4 layout
    float sum = 0.0f;
    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4 || ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        sum = xi.x + xi.y + xi.z + xi.w;
#pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, 32);
        }
    }

    // Quantize
    const float d_inv = 127.0f / fmaxf(amax, 1e-8f);
    char4 q;
    q.x = roundf(xi.x * d_inv);
    q.y = roundf(xi.y * d_inv);
    q.z = roundf(xi.z * d_inv);
    q.w = roundf(xi.w * d_inv);
    const float d = 1.0f / d_inv;

    // Write int8 values
    const int64_t ib = (int64_t)m * (K_padded / MMQ_QK8_1_MMQ) + k_block;
    char4* yqs4 = (char4*)y[ib].qs;
    yqs4[iqs / 4] = q;

    // Write scale/sum
    if (iqs % 32 == 0) {
        if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
            // D2S6 layout: vals_per_scale=64 (two sub-blocks share one d)
            // d2s6[0] = d for first 64 values (iqs 0..63)
            // d2s6[1] = d for second 64 values (iqs 64..127)
            // d2s6[2..7] = partial sums for 16-value groups covering first 96 values
            if (iqs == 0) {
                y[ib].d2s6[0] = __float2half(d);
            } else if (iqs == 64) {
                y[ib].d2s6[1] = __float2half(d);
            }
            // Partial sums: 6 groups of 16 values covering first 96 values
            // Each 16-value group spans threads iqs 0-15, 16-31, 32-47, 48-63, 64-79, 80-95
            if (iqs < 96) {
                // Compute sum for this group of 16 values (4 threads cooperate)
                float group_sum = 0.0f;
                group_sum = xi.x + xi.y + xi.z + xi.w;
#pragma unroll
                for (int offset = 1; offset > 0; offset >>= 1) {
                    group_sum += __shfl_xor_sync(0xFFFFFFFF, group_sum, offset, 4);
                }
                if ((iqs % 16) == 0) {
                    y[ib].d2s6[2 + iqs / 16] = __float2half(group_sum);
                }
            }
        } else if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
            y[ib].ds4[iqs / 32] = make_half2(__float2half(d), __float2half(sum));
        } else {
            y[ib].d4[iqs / 32] = d;
        }
    }
}

// ============================================================================
// Q8_1_mmq quantization launch wrapper
// ============================================================================

static void launch_quantize_q8_1_mmq(
    const float* x, void* vy, int K, int M, mmq_q8_1_ds_layout ds_layout,
    cudaStream_t stream)
{
    // Pad K to multiple of MMQ_QK8_1_MMQ (128)
    int K_padded = (K + MMQ_QK8_1_MMQ - 1) / MMQ_QK8_1_MMQ * MMQ_QK8_1_MMQ;

    const int block_size = 32;  // 32 threads per warp, each processes 4 values
    const int blocks_per_row = K_padded / 4;  // number of 4-value groups
    const int block_num_y = (blocks_per_row + block_size - 1) / block_size;
    const dim3 num_blocks(M, block_num_y);

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D4) {
        quantize_q8_1_mmq_kernel<MMQ_Q8_1_DS_LAYOUT_D4>
            <<<num_blocks, block_size, 0, stream>>>(x, vy, K, M, K_padded);
    } else if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
        quantize_q8_1_mmq_kernel<MMQ_Q8_1_DS_LAYOUT_DS4>
            <<<num_blocks, block_size, 0, stream>>>(x, vy, K, M, K_padded);
    } else {
        quantize_q8_1_mmq_kernel<MMQ_Q8_1_DS_LAYOUT_D2S6>
            <<<num_blocks, block_size, 0, stream>>>(x, vy, K, M, K_padded);
    }
}

// ============================================================================
// MMQ dot product implementations (dp4a)
// ============================================================================

// ---- Q3_K × Q8_1_mmq (D4 layout) ----
// Q3_K: 2-bit packed values, 4 values per int32
// Q8_1_mmq D4: 4 float scales
static __device__ __forceinline__ float vec_dot_q3_K_q8_1_mmq(
    const int* __restrict__ v, const int* __restrict__ u,
    const int8_t* __restrict__ scales,
    const float& d3, const float& d8)
{
    int sumi = 0;

#pragma unroll
    for (int i0 = 0; i0 < GEMV_QR3_K * VDR_Q3_K_Q8_1_MMQ; i0 += GEMV_QI8_1 / 2) {
        int sumi_sc = 0;
#pragma unroll
        for (int i = i0; i < i0 + GEMV_QI8_1 / 2; ++i) {
            sumi_sc = forge_dp4a(v[i], u[i], sumi_sc);
        }
        sumi += sumi_sc * scales[i0 / (GEMV_QI8_1 / 2)];
    }

    return d3 * d8 * sumi;
}

// ---- Q4_K × Q8_1_mmq (DS4 layout) ----
// Q4_K: 4-bit values, extracted via shift+mask
// Q8_1_mmq DS4: 4 half2 (d, s) pairs
static __device__ __forceinline__ float vec_dot_q4_K_q8_1_mmq(
    const int* __restrict__ v, const int* __restrict__ u,
    const uint8_t* __restrict__ sc, const uint8_t* __restrict__ m,
    const half2& dm4, const half2* __restrict__ ds8)
{
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < GEMV_QR4_K * VDR_Q4_K_Q8_1_MMQ / GEMV_QI8_1; ++i) {
        int sumi_d = 0;
#pragma unroll
        for (int j = 0; j < GEMV_QI8_1; ++j) {
            sumi_d = forge_dp4a((v[j] >> (4 * i)) & 0x0F0F0F0F, u[i * GEMV_QI8_1 + j], sumi_d);
        }
        const float2 ds8f = __half22float2(ds8[i]);
        sumf_d += ds8f.x * (sc[i] * sumi_d);
        sumf_m += ds8f.y * m[i];
    }

    const float2 dm4f = __half22float2(dm4);
    return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

// ---- Q5_K × Q8_1_mmq (DS4 layout) ----
// Q5_K: 5-bit values stored as separate high/low nibbles
// Q8_1_mmq DS4: 4 half2 (d, s) pairs
static __device__ __forceinline__ float vec_dot_q5_K_q8_1_mmq(
    const int* __restrict__ v, const int* __restrict__ u,
    const uint8_t* __restrict__ sc, const uint8_t* __restrict__ m,
    const half2& dm5, const half2* __restrict__ ds8)
{
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < GEMV_QR5_K * VDR_Q5_K_Q8_1_MMQ / GEMV_QI8_1; ++i) {
        int sumi_d = 0;
#pragma unroll
        for (int j = 0; j < GEMV_QI8_1; ++j) {
            sumi_d = forge_dp4a(v[i * GEMV_QI8_1 + j], u[i * GEMV_QI8_1 + j], sumi_d);
        }
        const float2 ds8f = __half22float2(ds8[i]);
        sumf_d += ds8f.x * (sc[i] * sumi_d);
        sumf_m += ds8f.y * m[i];
    }

    const float2 dm5f = __half22float2(dm5);
    return dm5f.x * sumf_d - dm5f.y * sumf_m;
}

// ---- Q4_0 × Q8_1_mmq (DS4 layout) ----
// Q4_0: raw packed 4-bit values (NOT split into lo/hi — vec_dot extracts inline)
// Q8_1_mmq DS4: 4 half2 (d, s) pairs
static __device__ __forceinline__ float vec_dot_q4_0_q8_1_mmq(
    const int* __restrict__ v, const int* __restrict__ u,
    const float& d4, const half2* __restrict__ ds8)
{
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < VDR_Q4_0_Q8_1_MMQ; ++i) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;
        sumi = forge_dp4a(vi0, u[2 * i + 0], sumi);
        sumi = forge_dp4a(vi1, u[2 * i + 1], sumi);
    }
    const float2 ds8f = __half22float2(*ds8);
    return d4 * (sumi * ds8f.x - (8 * VDR_Q4_0_Q8_1_MMQ / MMQ_QI4_0) * ds8f.y);
}

// ---- Q6_K × Q8_1_mmq (D4 layout) ----
// Q6_K: unpacked 6-bit values (via __vsubss4), int8 scales, float d
// Q8_1_mmq D4: 4 float scales
static __device__ __forceinline__ float vec_dot_q6_K_q8_1_mmq(
    const int* __restrict__ v, const int* __restrict__ u,
    const int8_t* __restrict__ sc, const float& d6, const float* __restrict__ d8)
{
    float sumf_d = 0.0f;
    const int sc_packed = get_int_b4(sc, 0);
    const int8_t* sc_reg = (const int8_t*)&sc_packed;
#pragma unroll
    for (int i0 = 0; i0 < VDR_Q6_K_Q8_1_MMQ; i0 += 4) {
        int2 sumi_d = {0, 0};
        for (int i = i0; i < i0 + 2; ++i) {
            sumi_d.x = forge_dp4a(v[2 * i + 0], u[2 * i + 0], sumi_d.x);
            sumi_d.x = forge_dp4a(v[2 * i + 1], u[2 * i + 1], sumi_d.x);
            sumi_d.y = forge_dp4a(v[2 * i + 4], u[2 * i + 4], sumi_d.y);
            sumi_d.y = forge_dp4a(v[2 * i + 5], u[2 * i + 5], sumi_d.y);
        }
        sumf_d += d8[i0 / 4] * (sc_reg[i0 / 2 + 0] * sumi_d.x + sc_reg[i0 / 2 + 1] * sumi_d.y);
    }
    return d6 * sumf_d;
}

// ---- Q2_K × Q8_1_mmq (D2S6 layout) ----
// Q2_K: 2-bit packed values, half2 pre-multiplied d*scale pairs
// Q8_1_mmq D2S6: 2 half d + 6 half partial sums
static __device__ __forceinline__ float vec_dot_q2_K_q8_1_impl_mmq(
    const int* __restrict__ v, const int* __restrict__ u,
    const half2* __restrict__ dm2, const float& d8, const half2* __restrict__ s8)
{
    (void)s8;  // exact integer path: the min term uses sum(u) computed here
               // rather than llama.cpp's pre-quant float partial sums (D2S6),
               // which our quantizer does not need to reproduce.
    float sumf = 0.0f;
    float sumf_d8 = 0.0f;
#pragma unroll
    for (int i0 = 0; i0 < GEMV_QR2_K * VDR_Q2_K_Q8_1_MMQ; i0 += GEMV_QI8_1) {
        const float2 dm2f0 = __half22float2(dm2[i0 / (GEMV_QI8_1 / 2) + 0]);
        const float2 dm2f1 = __half22float2(dm2[i0 / (GEMV_QI8_1 / 2) + 1]);
        int sumi_d0 = 0;
        int sumi_m0 = 0;
        for (int i = i0; i < i0 + GEMV_QI8_1 / 2; ++i) {
            sumi_d0 = forge_dp4a(v[i], u[i], sumi_d0);
            sumi_m0 = forge_dp4a(0x01010101, u[i], sumi_m0);
        }
        sumf_d8 += dm2f0.x * sumi_d0;
        sumf_d8 -= dm2f0.y * sumi_m0;
        int sumi_d1 = 0;
        int sumi_m1 = 0;
        for (int i = i0 + GEMV_QI8_1 / 2; i < i0 + GEMV_QI8_1; ++i) {
            sumi_d1 = forge_dp4a(v[i], u[i], sumi_d1);
            sumi_m1 = forge_dp4a(0x01010101, u[i], sumi_m1);
        }
        sumf_d8 += dm2f1.x * sumi_d1;
        sumf_d8 -= dm2f1.y * sumi_m1;
    }
    return sumf + d8 * sumf_d8;
}

// ============================================================================
// Weight tile loading into shared memory (dp4a layout)
// ============================================================================

// Unpack one 6-bit scale/min word for Q4_K/Q5_K (ported from llama.cpp
// mmq-load-tiles.cuh). The 12 GGUF scale bytes, viewed as three ints, hold:
//   int0 = sc0..sc3, int1 = m0..m3, int2 = low nibbles of sc4..7 and m4..7,
// with each value's two high bits stored at bits 6-7 of its byte in int0/int1.
// ksc selects which word to produce: 0 -> [sc0-3], 1 -> [sc4-7],
// 2 -> [m0-3], 3 -> [m4-7]; every output byte holds one clean 6-bit value.
static __device__ __forceinline__ int unpack_scales_q45_K(const int* scales, int ksc) {
    return ((scales[(ksc % 2) + (ksc != 0)] >> (4 * (ksc & (ksc / 2)))) & 0x0F0F0F0F) |
           ((scales[ksc / 2] >> (2 * (ksc % 2))) & 0x30303030);
}

// ---- Q3_K load tiles ----
// Q3_K block (110 bytes): hmask[32] + qs[64] + scales[12] + d[2]
// Shared memory layout (dp4a):
//   x_qs: I*(2*MMQ_TILE_NE_K + 1) ints — packed 2-bit values
//   x_df: I floats — dequant scales (d * scale - 32)
//   x_sc: I*(MMQ_TILE_NE_K/8) + I/8 ints — sign masks
template <int I, int nwarps>
static __device__ __forceinline__ void load_tiles_q3_K(
    const uint8_t* __restrict__ q_weight, int* __restrict__ x_tile,
    int n_start, int kbx0, int K, int N, int num_blocks_row)
{
    constexpr auto txs = mmq_txs_q3_k<I>();
    int*   x_qs = x_tile;
    float* x_df = (float*)(x_qs + txs.qs);
    int*   x_sc = (int*)(x_df + txs.dm);

    constexpr int warp_size = 32;
    const int lane = threadIdx.x;

    // Ported from llama.cpp load_tiles_q3_K (dp4a path). Row stride is 110
    // bytes, so all block reads go through byte-assembled get_int_b2 -- a
    // plain int* dereference here is a misaligned-address fault.
    // Layout produced:
    //   x_qs[i*65 + k] : pre-biased 3-bit values, k = (kqsx/8)*32 + l*8 + kqsx%8
    //   x_df[i]        : super-block scale d
    //   x_sc[i*4+i/8+ksc]: four words of per-32-group int8 scales (bias -32
    //                      baked in), consumed as scales[k0/4 + byte]
    constexpr int threads_per_row = MMQ_ITER_K / (4 * GEMV_QR3_K);  // 256/16 = 16
    constexpr int nrows = warp_size / threads_per_row;              // 2
    const int kqsx = lane % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows * nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y * nrows + lane / threads_per_row);
        if (i >= I) break;
        int n = n_start + i;
        if (n >= N) i = min(i, N - n_start - 1);  // clamp for bounds

        const uint8_t* w_row = q_weight + (size_t)n * num_blocks_row * 110;
        const uint8_t* block_ptr = w_row + kbx0 * 110;

        const int x_ql_0 = get_int_b2(block_ptr + 32, kqsx);
        const int x_qh_0 =
            get_int_b2(block_ptr, kqsx % (GEMV_QI3_K / 2)) >> (4 * (kqsx / (GEMV_QI3_K / 2)));

#pragma unroll
        for (int l = 0; l < GEMV_QR3_K; ++l) {
            const int k = (kqsx / 8) * 32 + l * 8 + kqsx % 8;

            const int x_ql_k = (x_ql_0 >> (2 * l)) & 0x03030303;
            const int x_qh_k = ((x_qh_0 >> l) << 2) & 0x04040404;

            const int x_qs_k = __vsubss4(x_ql_k | x_qh_k, 0x04040404);
            x_qs[i * (2 * MMQ_TILE_NE_K + 1) + k] = x_qs_k;
        }
    }

    // Scales: 12 packed GGUF bytes -> 16 signed 6-bit values via the SAME
    // aux-shuffle as dequantize_row_q3_K / forge::ops::dequantize_q3_k_row.
    // (The previously ported nibble-unpack dropped the weight-4 bit plane,
    // biasing every scale by -4 and skewing all outputs.)
    {
        const int lane0 = 0;
#pragma unroll
        for (int i0 = 0; i0 < I; i0 += nwarps) {
            const int i = (i0 + threadIdx.y) % I;
            if (lane != lane0) continue;
            const int n = n_start + min(i, N - n_start - 1);

            const uint8_t* w_row = q_weight + (size_t)n * num_blocks_row * 110;

            uint32_t aux[4] = {0u, 0u, 0u, 0u};
            const uint8_t* sc_src = w_row + kbx0 * 110 + 96;
#pragma unroll
            for (int b = 0; b < 12; ++b) {
                reinterpret_cast<uint8_t*>(aux)[b] = sc_src[b];
            }
            const uint32_t tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & 0x0F0F0F0Fu) | (((tmp >> 4) & 0x03030303u) << 4);
            aux[3] = ((aux[1] >> 4) & 0x0F0F0F0Fu) | (((tmp >> 6) & 0x03030303u) << 4);
            aux[0] = (aux[0] & 0x0F0F0F0Fu) | (((tmp >> 0) & 0x03030303u) << 4);
            aux[1] = (aux[1] & 0x0F0F0F0Fu) | (((tmp >> 2) & 0x03030303u) << 4);
            uint8_t* s8 = reinterpret_cast<uint8_t*>(aux);
#pragma unroll
            for (int b = 0; b < 16; ++b) {
                s8[b] = static_cast<uint8_t>(s8[b] - 32);
            }
            int* dst = x_sc + i * (MMQ_TILE_NE_K / 8) + i / 8;
            dst[0] = static_cast<int>(aux[0]);
            dst[1] = static_cast<int>(aux[1]);
            dst[2] = static_cast<int>(aux[2]);
            dst[3] = static_cast<int>(aux[3]);
        }
    }

    // Super-block scale d (fp16 at offset 108)
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * warp_size) {
        int i = (i0 + threadIdx.y * warp_size + lane) % I;
        int n = n_start + min(i, N - n_start - 1);

        const uint8_t* w_row = q_weight + (size_t)n * num_blocks_row * 110;
        const uint8_t* block_ptr = w_row + kbx0 * 110;

        uint16_t d_bits;
        memcpy(&d_bits, block_ptr + 108, 2);
        x_df[i] = __half2float(reinterpret_cast<const __half&>(d_bits));
    }
}

// ---- Q4_K load tiles ----
// Q4_K block (144 bytes): d(f16,2B) + dmin(f16,2B) + scales[12] + qs[128]
// Shared memory layout (dp4a):
//   x_qs: I*(MMQ_TILE_NE_K + 1) ints — packed 4-bit values
//   x_dm: I half2 — scale/min pairs
//   x_sc: I*(MMQ_TILE_NE_K/8) + I/8 ints — unpacked scales
template <int I, int nwarps>
static __device__ __forceinline__ void load_tiles_q4_K(
    const uint8_t* __restrict__ q_weight, int* __restrict__ x_tile,
    int n_start, int kbx0, int K, int N, int num_blocks_row)
{
    constexpr auto txs = mmq_txs_q4_k<I>();
    int*   x_qs = x_tile;
    half2* x_dm = (half2*)(x_qs + txs.qs);
    int*   x_sc = (int*)(x_dm + txs.dm);

    constexpr int warp_size = 32;
    const int lane = threadIdx.x;

    // Load quantized values (4-bit packed)
    constexpr int threads_per_row = MMQ_ITER_K / (4 * GEMV_QR4_K);  // 256/(4*2) = 32
    constexpr int nrows = warp_size / threads_per_row;  // 32/32 = 1

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows * nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y * nrows + lane / threads_per_row);
        if (i >= I) break;
        int n = n_start + i;
        if (n >= N) i = min(i, N - n_start - 1);

        const uint8_t* w_row = q_weight + (size_t)n * num_blocks_row * 144;
        const uint8_t* block_ptr = w_row + kbx0 * 144;

        // Load 4-bit packed qs values
        // qs[128] at offset 16 in the block
        // Each thread loads 1 int32 (8 nibbles = 8 values)
        const int txi = lane % threads_per_row;
        const int* qs_i32 = (const int*)(block_ptr + 16);
        x_qs[i * (MMQ_TILE_NE_K + 1) + txi] = qs_i32[txi];
    }

    // Load dm (d and dmin)
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * warp_size) {
        int i = (i0 + threadIdx.y * warp_size + lane) % I;
        int n = n_start + min(i, N - n_start - 1);

        const uint8_t* w_row = q_weight + (size_t)n * num_blocks_row * 144;
        const uint8_t* block_ptr = w_row + kbx0 * 144;

        // dm is half2 at offset 0
        half2 dm;
        memcpy(&dm, block_ptr, sizeof(half2));
        x_dm[i] = dm;
    }

    // Load scales (12 bytes at offset 4): repack into four words per row,
    // [sc0-3][sc4-7][m0-3][m4-7], one clean 6-bit value per byte — exactly
    // what the compute side's sc/m indexing expects (vec_dot reads sc bytes
    // [0..2) or [2..4) of a word and the matching mins two words away).
    constexpr int rows_per_warp = warp_size / 4;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * rows_per_warp) {
        int i = (i0 + threadIdx.y * rows_per_warp + lane / (MMQ_TILE_NE_K / 8)) % I;
        int n = n_start + min(i, N - n_start - 1);

        const uint8_t* w_row = q_weight + (size_t)n * num_blocks_row * 144;
        const uint8_t* block_ptr = w_row + kbx0 * 144;

        const int* scales = (const int*)(block_ptr + 4);
        const int ksc = lane % (MMQ_TILE_NE_K / 8);

        x_sc[i * (MMQ_TILE_NE_K / 8) + i / 8 + ksc] = unpack_scales_q45_K(scales, ksc);
    }
}

// ---- Q5_K load tiles ----
// Q5_K block (176 bytes): d(f16,2B) + dmin(f16,2B) + scales[12] + qh[32] + ql[128]
// Shared memory layout (dp4a):
//   x_qs: I*(2*MMQ_TILE_NE_K + 1) ints — 5-bit values (lo+hi packed)
//   x_dm: I half2 — scale/min pairs
//   x_sc: I*(MMQ_TILE_NE_K/8) + I/8 ints — unpacked scales
template <int I, int nwarps>
static __device__ __forceinline__ void load_tiles_q5_K(
    const uint8_t* __restrict__ q_weight, int* __restrict__ x_tile,
    int n_start, int kbx0, int K, int N, int num_blocks_row)
{
    constexpr auto txs = mmq_txs_q5_k<I>();
    int*   x_qs = x_tile;
    half2* x_dm = (half2*)(x_qs + txs.qs);
    int*   x_sc = (int*)(x_dm + txs.dm);

    constexpr int warp_size = 32;
    const int lane = threadIdx.x;

    // Load quantized values (5-bit: lo nibbles from ql, hi bits from qh)
    constexpr int threads_per_row = MMQ_ITER_K / (4 * GEMV_QR5_K);  // 256/(4*2) = 32
    constexpr int nrows = warp_size / threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows * nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y * nrows + lane / threads_per_row);
        if (i >= I) break;
        int n = n_start + i;
        if (n >= N) i = min(i, N - n_start - 1);

        const uint8_t* w_row = q_weight + (size_t)n * num_blocks_row * 176;
        const uint8_t* block_ptr = w_row + kbx0 * 176;

        // ql[128] at offset 48, qh[32] at offset 16
        // Each thread loads 1 int32 from ql and merges the matching qh bits
        // into bit 4 of every byte (ported from llama.cpp load_tiles_q5_K):
        // the shared-memory row then holds contiguous 5-bit values, which
        // vec_dot_q5_K_q8_1_mmq consumes with plain dp4a.
        const int txi = lane % threads_per_row;
        const int* ql_i32 = (const int*)(block_ptr + 48);
        const int* qh_i32 = (const int*)(block_ptr + 16);

        const int ky = GEMV_QR5_K * txi;
        const int ql = ql_i32[txi];
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;

        const int qh = qh_i32[txi % (GEMV_QI5_K / 4)];
        const int qh0 = ((qh >> (2 * (txi / (GEMV_QI5_K / 4)) + 0)) << 4) & 0x10101010;
        const int qh1 = ((qh >> (2 * (txi / (GEMV_QI5_K / 4)) + 1)) << 4) & 0x10101010;

        const int kq0 = ky - ky % (GEMV_QI5_K / 2) + txi % (GEMV_QI5_K / 4);
        const int kq1 = kq0 + GEMV_QI5_K / 4;

        x_qs[i * (2 * MMQ_TILE_NE_K + 1) + kq0] = ql0 | qh0;
        x_qs[i * (2 * MMQ_TILE_NE_K + 1) + kq1] = ql1 | qh1;
    }

    // Load dm (d and dmin)
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * warp_size) {
        int i = (i0 + threadIdx.y * warp_size + lane) % I;
        int n = n_start + min(i, N - n_start - 1);

        const uint8_t* w_row = q_weight + (size_t)n * num_blocks_row * 176;
        const uint8_t* block_ptr = w_row + kbx0 * 176;

        half2 dm;
        memcpy(&dm, block_ptr, sizeof(half2));
        x_dm[i] = dm;
    }

    // Load scales (same 12-byte layout as Q4_K): canonical unpack into
    // [sc0-3][sc4-7][m0-3][m4-7] words, one clean 6-bit value per byte.
    constexpr int rows_per_warp = warp_size / 4;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * rows_per_warp) {
        int i = (i0 + threadIdx.y * rows_per_warp + lane / (MMQ_TILE_NE_K / 8)) % I;
        int n = n_start + min(i, N - n_start - 1);

        const uint8_t* w_row = q_weight + (size_t)n * num_blocks_row * 176;
        const uint8_t* block_ptr = w_row + kbx0 * 176;

        const int* scales = (const int*)(block_ptr + 4);
        const int ksc = lane % (MMQ_TILE_NE_K / 8);
        x_sc[i * (MMQ_TILE_NE_K / 8) + i / 8 + ksc] = unpack_scales_q45_K(scales, ksc);
    }
}

// ---- Q4_0 load tiles ----
// Q4_0 block (18 bytes): half d (2B) + int8_t qs[16]
// Shared memory layout (dp4a):
//   x_qs: I*(MMQ_TILE_NE_K + 1) ints — raw packed 4-bit values
//   x_df: I*(MMQ_TILE_NE_K/MMQ_QI4_0) + I/MMQ_QI4_0 floats — one scale per Q4_0 block
template <int I, int nwarps>
static __device__ __forceinline__ void load_tiles_q4_0(
    const uint8_t* __restrict__ q_weight, int* __restrict__ x_tile,
    int n_start, int kbx0, int K, int N, int num_blocks_row)
{
    constexpr auto txs = mmq_txs_q4_0<I>();
    int*   x_qs = x_tile;
    float* x_df = (float*)(x_qs + txs.qs);

    constexpr int warp_size = 32;
    const int lane = threadIdx.x;

    // Ported from llama.cpp load_tiles_q4_0 (dp4a path). One kb iteration
    // covers a 256-element window = EIGHT consecutive 18-byte q4_0 blocks,
    // pairing with activation chunks 2*kb / 2*kb+1 like the K-types.
    // Row stride is 18 bytes, so qs reads go through byte-assembled
    // get_int_b2 (a plain int* load here is a misaligned-address fault).
    // Layout produced:
    //   x_qs[i*33 + txi]          : raw packed nibbles, txi = kbx*4 + kqsx
    //   x_df[i*8 + i/4 + kbxd]    : one fp32 d per block (padded dm region)
    constexpr int threads_per_row = MMQ_ITER_K / (4 * MMQ_QR4_0);   // 32
    constexpr int nrows = warp_size / threads_per_row;              // 1

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows * nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y * nrows + lane / threads_per_row);
        if (i >= I) break;
        int n = n_start + i;
        if (n >= N) i = min(i, N - n_start - 1);

        const uint8_t* w_row = q_weight + (size_t)n * num_blocks_row * 18;

        const int txi = lane % threads_per_row;
        const int kbx = txi / MMQ_QI4_0;    // block within the window (0..7)
        const int kqsx = txi % MMQ_QI4_0;   // int within the block's qs[16]

        const uint8_t* blk = w_row + (size_t)(kbx0 * 8 + kbx) * 18 + 2;
        x_qs[i * (MMQ_TILE_NE_K + 1) + txi] = get_int_b2(blk, kqsx);
    }

    // One float scale per q4_0 block
    constexpr int blocks_per_tile_x_row = MMQ_TILE_NE_K / MMQ_QI4_0;   // 8
    constexpr int rows_per_warp = warp_size / blocks_per_tile_x_row;   // 4
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * rows_per_warp) {
        int i = (i0 + threadIdx.y * rows_per_warp + lane / blocks_per_tile_x_row) % I;
        int n = n_start + min(i, N - n_start - 1);

        const uint8_t* w_row = q_weight + (size_t)n * num_blocks_row * 18;

        const int kbxd = lane % blocks_per_tile_x_row;
        const uint8_t* blk = w_row + (size_t)(kbx0 * 8 + kbxd) * 18;
        uint16_t d_bits;
        memcpy(&d_bits, blk, sizeof(uint16_t));
        const float d = __half2float(reinterpret_cast<const __half&>(d_bits));
        x_df[i * (MMQ_TILE_NE_K / MMQ_QI4_0) + i / MMQ_QI4_0 + kbxd] = d;
    }
}

// ---- Q6_K load tiles ----
// Q6_K block (210 bytes): ql[128] + qh[64] + scales[16] + d[2]
// Shared memory layout (dp4a):
//   x_qs: I*(2*MMQ_TILE_NE_K + 1) ints — unpacked 6-bit values (via __vsubss4)
//   x_df: I*MMQ_TILE_NE_K/QI6_K floats — one float d per Q6_K block
//   x_sc: I*(MMQ_TILE_NE_K/8) + I/8 ints — scale bytes
template <int I, int nwarps>
static __device__ __forceinline__ void load_tiles_q6_K(
    const uint8_t* __restrict__ q_weight, int* __restrict__ x_tile,
    int n_start, int kbx0, int K, int N, int num_blocks_row)
{
    constexpr auto txs = mmq_txs_q6_k<I>();
    int*   x_qs = x_tile;
    float* x_df = (float*)(x_qs + txs.qs);
    int*   x_sc = (int*)(x_df + txs.dm);

    constexpr int warp_size = 32;
    const int lane = threadIdx.x;

    constexpr int threads_per_row = MMQ_ITER_K / (4 * GEMV_QR6_K);  // 256/(4*2) = 32
    constexpr int nrows = warp_size / threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows * nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y * nrows + lane / threads_per_row);
        if (i >= I) break;
        int n = n_start + i;
        if (n >= N) i = min(i, N - n_start - 1);

        const uint8_t* w_row = q_weight + (size_t)n * num_blocks_row * 210;
        const uint8_t* block_ptr = w_row + kbx0 * 210;

        const int txi = lane % threads_per_row;
        // Unpack 6-bit values using __vsubss4 (from llama.cpp dp4a path)
        const int ql = get_int_b2(block_ptr, txi);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;
        const int qh = get_int_b2(block_ptr + 128, (GEMV_QI6_K / 4) * (txi / (GEMV_QI6_K / 2)) + txi % (GEMV_QI6_K / 4));
        const int qh0 = ((qh >> ((txi & 0x08) >> 2)) << 4) & 0x30303030;
        const int qh1 = (qh >> ((txi & 0x08) >> 2)) & 0x30303030;
        const int kq0 = 2 * txi - txi % (GEMV_QI6_K / 2) + 0;
        const int kq1 = 2 * txi - txi % (GEMV_QI6_K / 2) + GEMV_QI6_K / 2;
        x_qs[i * (2 * MMQ_TILE_NE_K + 1) + kq0] = __vsubss4(ql0 | qh0, 0x20202020);
        x_qs[i * (2 * MMQ_TILE_NE_K + 1) + kq1] = __vsubss4(ql1 | qh1, 0x20202020);
    }

    // Load d (float at offset 208)
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * warp_size) {
        int i = (i0 + threadIdx.y * warp_size + lane) % I;
        int n = n_start + min(i, N - n_start - 1);

        const uint8_t* w_row = q_weight + (size_t)n * num_blocks_row * 210;
        const uint8_t* block_ptr = w_row + kbx0 * 210;

        uint16_t d_bits;
        memcpy(&d_bits, block_ptr + 208, 2);
        x_df[i * (MMQ_TILE_NE_K / GEMV_QI6_K)] = __half2float(reinterpret_cast<const __half&>(d_bits));
    }

    // Load scales (16 bytes at offset 192)
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * (warp_size / 4)) {
        int i = (i0 + threadIdx.y * (warp_size / 4) + lane / 4) % I;
        int n = n_start + min(i, N - n_start - 1);

        const uint8_t* w_row = q_weight + (size_t)n * num_blocks_row * 210;
        const uint8_t* block_ptr = w_row + kbx0 * 210;

        x_sc[i * (MMQ_TILE_NE_K / 8) + i / 8 + threadIdx.x % (MMQ_TILE_NE_K / 8)] =
            get_int_b2(block_ptr + 192, threadIdx.x % (GEMV_QI6_K / 8));
    }
}

// ---- Q2_K load tiles ----
// Q2_K block (84 bytes): dm(half2, 4B) + qs[64] + scales[16]
// Shared memory layout (dp4a):
//   x_qs: I*(2*MMQ_TILE_NE_K + 1) ints — 2-bit packed values
//   x_dm: I*(MMQ_TILE_NE_K + 1) half2 — pre-multiplied d*scale and dmin*(sc>>4)
template <int I, int nwarps>
static __device__ __forceinline__ void load_tiles_q2_K(
    const uint8_t* __restrict__ q_weight, int* __restrict__ x_tile,
    int n_start, int kbx0, int K, int N, int num_blocks_row)
{
    constexpr auto txs = mmq_txs_q2_k<I>();
    int*   x_qs = x_tile;
    half2* x_dm = (half2*)(x_qs + txs.qs);

    constexpr int warp_size = 32;
    const int lane = threadIdx.x;

    // Q2_K: threads_per_row = MMQ_ITER_K / (4 * GEMV_QR2_K) = 256/(4*4) = 16
    constexpr int threads_per_row = MMQ_ITER_K / (4 * GEMV_QR2_K);
    constexpr int nrows = warp_size / threads_per_row;  // 32/16 = 2

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows * nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y * nrows + lane / threads_per_row);
        if (i >= I) break;
        int n = n_start + i;
        if (n >= N) i = min(i, N - n_start - 1);

        const uint8_t* w_row = q_weight + (size_t)n * num_blocks_row * 84;
        const uint8_t* block_ptr = w_row + kbx0 * 84;

        const int kqsx = lane % threads_per_row;
        // Block layout (84 B): scales[16] @0, qs[64] @16, d @80, dmin @82.
        // Ported from llama.cpp load_tiles_q2_K: the old offsets here read
        // qs from the scale bytes and dm from packed scales -> inf/nan.
        const int x_ql_0 = get_int_b2(block_ptr + 16, kqsx);
        for (int l = 0; l < GEMV_QR2_K; ++l) {
            const int k = (kqsx / 8) * 32 + l * 8 + kqsx % 8;
            const int x_qs_k = (x_ql_0 >> (2 * l)) & 0x03030303;
            x_qs[i * (2 * MMQ_TILE_NE_K + 1) + k] = x_qs_k;
        }

        // Pre-multiply scales into half2 pairs
        const int sc_m = block_ptr[kqsx];  // scales at offset 0
        const float2 bxi_dmf =
            __half22float2(*reinterpret_cast<const half2*>(block_ptr + 80));  // d,dmin @80
        const half2 x_dm_ik = make_half2(
            __float2half(bxi_dmf.x * (sc_m & 0x0F)),
            __float2half(bxi_dmf.y * (sc_m >> 4)));
        x_dm[i * (MMQ_TILE_NE_K + 1) + kqsx] = x_dm_ik;
    }
}

// ============================================================================
// Main MMQ kernel
// ============================================================================

// The kernel computes C = A @ B^T where:
//   B is quantized weights [N, K]
//   A is FP32 activations [M, K] (pre-quantized to Q8_1_mmq)
//   C is FP32 output [M, N]
//
// Tile structure:
//   I = weight rows per tile (N direction)
//   J = activation rows per tile (M direction)
//   Each CUDA block processes one (I, J) tile
//   Thread block: (32, nwarps) = 256 threads

template <DataType DT, int I, int J>
__global__ void mul_mat_q_kernel(
    const uint8_t* __restrict__ q_weight,   // [N, K] quantized weight
    const int*     __restrict__ q_act_mmq,  // [M, K/128] block_q8_1_mmq as int*
    float*         __restrict__ out,         // [M, N] output
    int M, int K, int N, int num_blocks_row, int K_padded)
{
    constexpr int warp_size = 32;
    constexpr int nwarps = MMQ_NWARPS;

    // Block tile indices
    const int n_tile = blockIdx.x;  // Which I-weight-row tile
    const int m_tile = blockIdx.y;  // Which J-activation-row tile

    const int n_start = n_tile * I;
    const int m_start = m_tile * J;

    if (n_start >= N || m_start >= M) return;

    // Compute shared memory sizes
    constexpr auto txs = (DT == DataType::Q3_K)  ? mmq_txs_q3_k<I>() :
                         (DT == DataType::Q4_K)  ? mmq_txs_q4_k<I>() :
                         (DT == DataType::Q5_K)  ? mmq_txs_q5_k<I>() :
                         (DT == DataType::Q4_0)  ? mmq_txs_q4_0<I>() :
                         (DT == DataType::Q6_K)  ? mmq_txs_q6_k<I>() :
                                                    mmq_txs_q2_k<I>();
    constexpr int tile_x_ints = txs.qs + txs.dm + txs.sc;

    // Shared memory
    extern __shared__ int smem[];
    int* tile_x = smem;
    int* tile_y = tile_x + tile_x_ints;

    // Thread mapping
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;

    // Number of K-blocks (each MMQ_ITER_K = 256 K-elements)
    constexpr int BE = (DT == DataType::Q3_K) ? 256 :
                       (DT == DataType::Q4_K) ? 256 : 256;  // All K-types use 256
    const int num_k_blocks = (K + BE - 1) / BE;

    // Accumulator: each warp accumulates for J/nwarps activation rows
    constexpr int j_per_warp = (J + nwarps - 1) / nwarps;
    float sum[j_per_warp * I / warp_size] = {0.0f};

    // Iterate over K blocks
    for (int kb = 0; kb < num_k_blocks; ++kb) {
        // Load weight tile into shared memory
        if (DT == DataType::Q3_K) {
            load_tiles_q3_K<I, nwarps>(q_weight, tile_x, n_start, kb, K, N, num_blocks_row);
        } else if (DT == DataType::Q4_K) {
            load_tiles_q4_K<I, nwarps>(q_weight, tile_x, n_start, kb, K, N, num_blocks_row);
        } else if (DT == DataType::Q5_K) {
            load_tiles_q5_K<I, nwarps>(q_weight, tile_x, n_start, kb, K, N, num_blocks_row);
        } else if (DT == DataType::Q4_0) {
            load_tiles_q4_0<I, nwarps>(q_weight, tile_x, n_start, kb, K, N, num_blocks_row);
        } else if (DT == DataType::Q6_K) {
            load_tiles_q6_K<I, nwarps>(q_weight, tile_x, n_start, kb, K, N, num_blocks_row);
        } else {
            load_tiles_q2_K<I, nwarps>(q_weight, tile_x, n_start, kb, K, N, num_blocks_row);
        }

        // Load activation tile into shared memory
        // block_q8_1_mmq is 144 bytes = 36 ints per block
        // Each MMQ block covers 128 K-elements
        // For kb, the activation data starts at q_act_mmq + m*M_blocks + kb
        // But the layout is: for row m, block k, the data is at
        //   q_act_mmq[m * (K_padded/MMQ_QK8_1_MMQ) + kb] as block_q8_1_mmq
        constexpr int sz = sizeof(block_q8_1_mmq) / sizeof(int);  // 36
        const int num_mmq_blocks_per_row = K_padded / MMQ_QK8_1_MMQ;

        // Each block_q8_1_mmq contains 4 sub-blocks of 32 int8 values
        // We load 2 sub-blocks per syncthreads (matching llama.cpp's approach)
        // Sub-block 0: first 32 int8 values + scale (offset 0..36 in ints)
        // Sub-block 1: next 32 int8 values + scale (offset 36..72 in ints)

#pragma unroll
        for (int l0 = 0; l0 < J * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
            int l = l0 + warp * warp_size + lane;
            if (l < J * MMQ_TILE_Y_K) {
                int j = l / MMQ_TILE_Y_K;
                int k = l % MMQ_TILE_Y_K;
                int m = m_start + j;
                if (m < M) {
                    // One kb iteration covers a full 256-element weight
                    // super-block, which pairs with TWO consecutive 128-value
                    // activation blocks: 2*kb here (elements [256kb, 256kb+128))
                    // and 2*kb+1 in the reload below. The old `kb` indexing
                    // advanced the activation stream at half the weight rate,
                    // so everything past the first super-block was garbage.
                    const int* by0 = q_act_mmq +
                        (int64_t)m * num_mmq_blocks_per_row * sz +
                        (int64_t)(2 * kb) * sz;
                    tile_y[l] = by0[k];
                } else {
                    tile_y[l] = 0;
                }
            }
        }

        __syncthreads();

        // Compute dot products
        // For Q3_K (D4 layout):
        if (DT == DataType::Q3_K) {
            constexpr auto txs_q3k = mmq_txs_q3_k<I>();
            const int*   x_qs = tile_x;
            const float* x_df = (const float*)(x_qs + txs_q3k.qs);
            const int*   x_sc = (const int*)(x_df + txs_q3k.dm);

            // tile_y access: y_df = first 4 floats, y_qs = next 32 ints
            const float* y_df = (const float*)tile_y;
            const int*   y_qs = (const int*)(y_df + 4);

            // Process VDR sub-tiles within the 32 K-elements
            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += GEMV_QR3_K * VDR_Q3_K_Q8_1_MMQ) {
                int k0 = k01;

#pragma unroll
                for (int j0 = 0; j0 < J; j0 += nwarps) {
                    int j = j0 + warp;
                    if (m_start + j >= M) continue;

#pragma unroll
                    for (int i0 = 0; i0 < I; i0 += warp_size) {
                        int i = i0 + lane;
                        if (n_start + i >= N) continue;

                        const int8_t* scales_p = (const int8_t*)(x_sc + i * (MMQ_TILE_NE_K / 8) + i / 8) + k0 / 4;
            const int8_t* scales = scales_p;

                        sum[(j0 / nwarps) * (I / warp_size) + i0 / warp_size] +=
                            vec_dot_q3_K_q8_1_mmq(
                                &x_qs[i * (2 * MMQ_TILE_NE_K + 1) + k0],
                                &y_qs[j * MMQ_TILE_Y_K + k01],
                                scales,
                                x_df[i],
                                y_df[j * MMQ_TILE_Y_K + k01 / GEMV_QI8_1]);
                    }
                }
            }
        }
        // For Q4_K (DS4 layout):
        else if (DT == DataType::Q4_K) {
            constexpr auto txs_q4k = mmq_txs_q4_k<I>();
            const int*   x_qs = tile_x;
            const half2* x_dm = (const half2*)(x_qs + txs_q4k.qs);
            const int*   x_sc = (const int*)(x_dm + txs_q4k.dm);

            const half2* y_ds = (const half2*)tile_y;
            const int*   y_qs = (const int*)(y_ds + 4);

            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += GEMV_QR4_K * VDR_Q4_K_Q8_1_MMQ) {
                int k0 = k01;

#pragma unroll
                for (int j0 = 0; j0 < J; j0 += nwarps) {
                    int j = j0 + warp;
                    if (m_start + j >= M) continue;

#pragma unroll
                    for (int i0 = 0; i0 < I; i0 += warp_size) {
                        int i = i0 + lane;
                        if (n_start + i >= N) continue;

                        const uint8_t* sc = (const uint8_t*)(&x_sc[i * (MMQ_TILE_NE_K / 8) + i / 8 + k0 / 32]) + 2 * (k01 / 16);

                        sum[(j0 / nwarps) * (I / warp_size) + i0 / warp_size] +=
                            vec_dot_q4_K_q8_1_mmq(
                                &x_qs[i * (MMQ_TILE_NE_K + 1) + k0 / 2],
                                &y_qs[j * MMQ_TILE_Y_K + k01],
                                sc, sc + 8,
                                x_dm[i],
                                &y_ds[j * MMQ_TILE_Y_K + k01 / GEMV_QI8_1]);
                    }
                }
            }
        }
        // For Q5_K (DS4 layout):
        else if (DT == DataType::Q5_K) {
            constexpr auto txs_q5k = mmq_txs_q5_k<I>();
            const int*   x_qs = tile_x;
            const half2* x_dm = (const half2*)(x_qs + txs_q5k.qs);
            const int*   x_sc = (const int*)(x_dm + txs_q5k.dm);

            const half2* y_ds = (const half2*)tile_y;
            const int*   y_qs = (const int*)(y_ds + 4);

            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += GEMV_QR5_K * VDR_Q5_K_Q8_1_MMQ) {
                int k0 = k01;

#pragma unroll
                for (int j0 = 0; j0 < J; j0 += nwarps) {
                    int j = j0 + warp;
                    if (m_start + j >= M) continue;

#pragma unroll
                    for (int i0 = 0; i0 < I; i0 += warp_size) {
                        int i = i0 + lane;
                        if (n_start + i >= N) continue;

                        const uint8_t* sc = ((const uint8_t*)(&x_sc[i * (MMQ_TILE_NE_K / 8) + i / 8 + k0 / 32])) + 2 * (k01 / 16);

                        sum[(j0 / nwarps) * (I / warp_size) + i0 / warp_size] +=
                            vec_dot_q5_K_q8_1_mmq(
                                &x_qs[i * (2 * MMQ_TILE_NE_K + 1) + k0],
                                &y_qs[j * MMQ_TILE_Y_K + k01],
                                sc, sc + 8,
                                x_dm[i],
                                &y_ds[j * MMQ_TILE_Y_K + k01 / GEMV_QI8_1]);
                    }
                }
            }
        }
        // For Q4_0 (DS4 layout):
        else if (DT == DataType::Q4_0) {
            constexpr auto txs_q40 = mmq_txs_q4_0<I>();
            const int*   x_qs = tile_x;
            const float* x_df = (const float*)(x_qs + txs_q40.qs);

            const half2* y_ds = (const half2*)tile_y;
            const int*   y_qs = (const int*)(y_ds + 4);

            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += MMQ_QR4_0 * VDR_Q4_0_Q8_1_MMQ) {
                // q4_0 packs e_l | e_{l+16}<<4 per byte, so the 32-element
                // group splits into two 16-value halves in tile_y. Gather
                // them interleaved (ported from llama.cpp's wrapper): the
                // dot pairs lo nibbles with u[2i], hi nibbles with u[2i+1].
                const int kyqs =
                    GEMV_QI8_1 * ((k01 / 2) / (GEMV_QI8_1 / 2)) + (k01 / 2) % (GEMV_QI8_1 / 2);
                const int k0 = k01;
#pragma unroll
                for (int j0 = 0; j0 < J; j0 += nwarps) {
                    int j = j0 + warp;
                    if (m_start + j >= M) continue;

#pragma unroll
                    for (int i0 = 0; i0 < I; i0 += warp_size) {
                        int i = i0 + lane;
                        if (n_start + i >= N) continue;

                        int u[2 * VDR_Q4_0_Q8_1_MMQ];
#pragma unroll
                        for (int l = 0; l < MMQ_QI4_0; ++l) {
                            u[2 * l + 0] = y_qs[j * MMQ_TILE_Y_K + kyqs + l];
                            u[2 * l + 1] = y_qs[j * MMQ_TILE_Y_K + kyqs + MMQ_QI4_0 + l];
                        }

                        sum[(j0 / nwarps) * (I / warp_size) + i0 / warp_size] +=
                            vec_dot_q4_0_q8_1_mmq(
                                &x_qs[i * (MMQ_TILE_NE_K + 1) + k0 / MMQ_QR4_0],
                                u,
                                x_df[i * (MMQ_TILE_NE_K / MMQ_QI4_0) + i / MMQ_QI4_0 +
                                     k0 / (MMQ_QR4_0 * MMQ_QI4_0)],
                                &y_ds[j * MMQ_TILE_Y_K + k01 / GEMV_QI8_1]);
                    }
                }
            }
        }
        // For Q6_K (D4 layout):
        else if (DT == DataType::Q6_K) {
            constexpr auto txs_q6k = mmq_txs_q6_k<I>();
            const int*   x_qs = tile_x;
            const float* x_df = (const float*)(x_qs + txs_q6k.qs);
            const int*   x_sc = (const int*)(x_df + txs_q6k.dm);

            const float* y_df = (const float*)tile_y;
            const int*   y_qs = (const int*)(y_df + 4);

            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += GEMV_QR6_K * VDR_Q6_K_Q8_1_MMQ) {
#pragma unroll
                for (int j0 = 0; j0 < J; j0 += nwarps) {
                    int j = j0 + warp;
                    if (m_start + j >= M) continue;

#pragma unroll
                    for (int i0 = 0; i0 < I; i0 += warp_size) {
                        int i = i0 + lane;
                        if (n_start + i >= N) continue;

                        const int8_t* sc = (const int8_t*)(x_sc + i * (MMQ_TILE_NE_K / 8) + i / 8 + k01 / 16);

                        sum[(j0 / nwarps) * (I / warp_size) + i0 / warp_size] +=
                            vec_dot_q6_K_q8_1_mmq(
                                &x_qs[i * (2 * MMQ_TILE_NE_K + 1) + k01],
                                &y_qs[j * MMQ_TILE_Y_K + k01],
                                sc,
                                x_df[i * (MMQ_TILE_NE_K / GEMV_QI6_K)],
                                &y_df[j * MMQ_TILE_Y_K + k01 / GEMV_QI8_1]);
                    }
                }
            }
        }
        // For Q2_K (D2S6 layout):
        else {  // Q2_K
            constexpr auto txs_q2k = mmq_txs_q2_k<I>();
            const int*   x_qs = tile_x;
            const half2* x_dm = (const half2*)(x_qs + txs_q2k.qs);

            // tile_y: D2S6 layout
            const half*  y_d2s6 = (const half*)tile_y;
            const int*   y_qs = (const int*)(y_d2s6 + 8);

            // First loop: k01 = 0..MMQ_TILE_NE_K/2-1, step QR2_K*VDR, ns=2
            for (int k01 = 0; k01 < MMQ_TILE_NE_K / 2; k01 += GEMV_QR2_K * VDR_Q2_K_Q8_1_MMQ) {
#pragma unroll
                for (int j0 = 0; j0 < J; j0 += nwarps) {
                    int j = j0 + warp;
                    if (m_start + j >= M) continue;

#pragma unroll
                    for (int i0 = 0; i0 < I; i0 += warp_size) {
                        int i = i0 + lane;
                        if (n_start + i >= N) continue;

                        const float d8 = __half2float(y_d2s6[j * MMQ_TILE_Y_K * 2]);  // y_df.x = d for first half
                        const half2* s8 = (const half2*)(y_d2s6 + j * MMQ_TILE_Y_K * 2 + (1 + k01 / GEMV_QI8_1) * 2);

                        sum[(j0 / nwarps) * (I / warp_size) + i0 / warp_size] +=
                            vec_dot_q2_K_q8_1_impl_mmq(
                                &x_qs[i * (2 * MMQ_TILE_NE_K + 1) + k01],
                                &y_qs[j * MMQ_TILE_Y_K + k01],
                                &x_dm[i * (MMQ_TILE_NE_K + 1) + k01 / 4],
                                d8, s8);
                    }
                }
            }

            // Second loop: k01 = MMQ_TILE_NE_K/2..MMQ_TILE_NE_K-1, step QR2_K*VDR, ns=1
            for (int k01 = MMQ_TILE_NE_K / 2; k01 < MMQ_TILE_NE_K; k01 += GEMV_QR2_K * VDR_Q2_K_Q8_1_MMQ) {
#pragma unroll
                for (int j0 = 0; j0 < J; j0 += nwarps) {
                    int j = j0 + warp;
                    if (m_start + j >= M) continue;

#pragma unroll
                    for (int i0 = 0; i0 < I; i0 += warp_size) {
                        int i = i0 + lane;
                        if (n_start + i >= N) continue;

                        const float d8 = __half2float(y_d2s6[j * MMQ_TILE_Y_K * 2 + 1]);  // y_df.y = d for second half
                        const half2* s8 = (const half2*)(y_d2s6 + j * MMQ_TILE_Y_K * 2 + (1 + k01 / GEMV_QI8_1) * 2);

                        sum[(j0 / nwarps) * (I / warp_size) + i0 / warp_size] +=
                            vec_dot_q2_K_q8_1_impl_mmq(
                                &x_qs[i * (2 * MMQ_TILE_NE_K + 1) + k01],
                                &y_qs[j * MMQ_TILE_Y_K + k01],
                                &x_dm[i * (MMQ_TILE_NE_K + 1) + k01 / 4],
                                d8, s8);
                    }
                }
            }
        }

        __syncthreads();

        // Load second sub-block of activation (next 32 int8 values + scale)
        // This handles the second half of the 64-element tile
        // (following llama.cpp's approach of loading 2 sub-blocks per iteration)
#pragma unroll
        for (int l0 = 0; l0 < J * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
            int l = l0 + warp * warp_size + lane;
            if (l < J * MMQ_TILE_Y_K) {
                int j = l / MMQ_TILE_Y_K;
                int k = l % MMQ_TILE_Y_K;
                int m = m_start + j;
                if (m < M) {
                    const int* by0 = q_act_mmq +
                        (int64_t)m * num_mmq_blocks_per_row * sz +
                        (int64_t)(2 * kb + 1) * sz;  // second 128-value chunk
                    tile_y[l] = by0[k];
                } else {
                    tile_y[l] = 0;
                }
            }
        }

        __syncthreads();

        // Process second sub-block (k0 + MMQ_TILE_NE_K)
        if (DT == DataType::Q3_K) {
            constexpr auto txs_q3k = mmq_txs_q3_k<I>();
            const int*   x_qs = tile_x;
            const float* x_df = (const float*)(x_qs + txs_q3k.qs);
            const int*   x_sc = (const int*)(x_df + txs_q3k.dm);
            const float* y_df = (const float*)tile_y;
            const int*   y_qs = (const int*)(y_df + 4);

            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += GEMV_QR3_K * VDR_Q3_K_Q8_1_MMQ) {
                int k0 = MMQ_TILE_NE_K + k01;

#pragma unroll
                for (int j0 = 0; j0 < J; j0 += nwarps) {
                    int j = j0 + warp;
                    if (m_start + j >= M) continue;

#pragma unroll
                    for (int i0 = 0; i0 < I; i0 += warp_size) {
                        int i = i0 + lane;
                        if (n_start + i >= N) continue;

                        const int8_t* scales = (const int8_t*)(x_sc + i * (MMQ_TILE_NE_K / 8) + i / 8) + k0 / 4;

                        sum[(j0 / nwarps) * (I / warp_size) + i0 / warp_size] +=
                            vec_dot_q3_K_q8_1_mmq(
                                &x_qs[i * (2 * MMQ_TILE_NE_K + 1) + k0],
                                &y_qs[j * MMQ_TILE_Y_K + k01],
                                scales,
                                x_df[i],
                                y_df[j * MMQ_TILE_Y_K + k01 / GEMV_QI8_1]);
                    }
                }
            }
        } else if (DT == DataType::Q4_K) {
            constexpr auto txs_q4k = mmq_txs_q4_k<I>();
            const int*   x_qs = tile_x;
            const half2* x_dm = (const half2*)(x_qs + txs_q4k.qs);
            const int*   x_sc = (const int*)(x_dm + txs_q4k.dm);
            const half2* y_ds = (const half2*)tile_y;
            const int*   y_qs = (const int*)(y_ds + 4);

            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += GEMV_QR4_K * VDR_Q4_K_Q8_1_MMQ) {
                int k0 = MMQ_TILE_NE_K + k01;

#pragma unroll
                for (int j0 = 0; j0 < J; j0 += nwarps) {
                    int j = j0 + warp;
                    if (m_start + j >= M) continue;

#pragma unroll
                    for (int i0 = 0; i0 < I; i0 += warp_size) {
                        int i = i0 + lane;
                        if (n_start + i >= N) continue;

                        const uint8_t* sc = (const uint8_t*)(&x_sc[i * (MMQ_TILE_NE_K / 8) + i / 8 + k0 / 32]) + 2 * (k01 / 16);

                        sum[(j0 / nwarps) * (I / warp_size) + i0 / warp_size] +=
                            vec_dot_q4_K_q8_1_mmq(
                                &x_qs[i * (MMQ_TILE_NE_K + 1) + k0 / 2],
                                &y_qs[j * MMQ_TILE_Y_K + k01],
                                sc, sc + 8,
                                x_dm[i],
                                &y_ds[j * MMQ_TILE_Y_K + k01 / GEMV_QI8_1]);
                    }
                }
            }
        } else if (DT == DataType::Q5_K) {
            constexpr auto txs_q5k = mmq_txs_q5_k<I>();
            const int*   x_qs = tile_x;
            const half2* x_dm = (const half2*)(x_qs + txs_q5k.qs);
            const int*   x_sc = (const int*)(x_dm + txs_q5k.dm);
            const half2* y_ds = (const half2*)tile_y;
            const int*   y_qs = (const int*)(y_ds + 4);

            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += GEMV_QR5_K * VDR_Q5_K_Q8_1_MMQ) {
                int k0 = MMQ_TILE_NE_K + k01;

#pragma unroll
                for (int j0 = 0; j0 < J; j0 += nwarps) {
                    int j = j0 + warp;
                    if (m_start + j >= M) continue;

#pragma unroll
                    for (int i0 = 0; i0 < I; i0 += warp_size) {
                        int i = i0 + lane;
                        if (n_start + i >= N) continue;

                        const uint8_t* sc = ((const uint8_t*)(&x_sc[i * (MMQ_TILE_NE_K / 8) + i / 8 + k0 / 32])) + 2 * (k01 / 16);

                        sum[(j0 / nwarps) * (I / warp_size) + i0 / warp_size] +=
                            vec_dot_q5_K_q8_1_mmq(
                                &x_qs[i * (2 * MMQ_TILE_NE_K + 1) + k0],
                                &y_qs[j * MMQ_TILE_Y_K + k01],
                                sc, sc + 8,
                                x_dm[i],
                                &y_ds[j * MMQ_TILE_Y_K + k01 / GEMV_QI8_1]);
                    }
                }
            }
        } else if (DT == DataType::Q4_0) {
            constexpr auto txs_q40 = mmq_txs_q4_0<I>();
            const int*   x_qs = tile_x;
            const float* x_df = (const float*)(x_qs + txs_q40.qs);

            const half2* y_ds = (const half2*)tile_y;
            const int*   y_qs = (const int*)(y_ds + 4);

            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += MMQ_QR4_0 * VDR_Q4_0_Q8_1_MMQ) {
                const int kyqs =
                    GEMV_QI8_1 * ((k01 / 2) / (GEMV_QI8_1 / 2)) + (k01 / 2) % (GEMV_QI8_1 / 2);
                const int k0 = MMQ_TILE_NE_K + k01;
#pragma unroll
                for (int j0 = 0; j0 < J; j0 += nwarps) {
                    int j = j0 + warp;
                    if (m_start + j >= M) continue;

#pragma unroll
                    for (int i0 = 0; i0 < I; i0 += warp_size) {
                        int i = i0 + lane;
                        if (n_start + i >= N) continue;

                        int u[2 * VDR_Q4_0_Q8_1_MMQ];
#pragma unroll
                        for (int l = 0; l < MMQ_QI4_0; ++l) {
                            u[2 * l + 0] = y_qs[j * MMQ_TILE_Y_K + kyqs + l];
                            u[2 * l + 1] = y_qs[j * MMQ_TILE_Y_K + kyqs + MMQ_QI4_0 + l];
                        }

                        sum[(j0 / nwarps) * (I / warp_size) + i0 / warp_size] +=
                            vec_dot_q4_0_q8_1_mmq(
                                &x_qs[i * (MMQ_TILE_NE_K + 1) + k0 / MMQ_QR4_0],
                                u,
                                x_df[i * (MMQ_TILE_NE_K / MMQ_QI4_0) + i / MMQ_QI4_0 +
                                     k0 / (MMQ_QR4_0 * MMQ_QI4_0)],
                                &y_ds[j * MMQ_TILE_Y_K + k01 / GEMV_QI8_1]);
                    }
                }
            }
        } else if (DT == DataType::Q6_K) {
            constexpr auto txs_q6k = mmq_txs_q6_k<I>();
            const int*   x_qs = tile_x;
            const float* x_df = (const float*)(x_qs + txs_q6k.qs);
            const int*   x_sc = (const int*)(x_df + txs_q6k.dm);

            const float* y_df = (const float*)tile_y;
            const int*   y_qs = (const int*)(y_df + 4);

            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += GEMV_QR6_K * VDR_Q6_K_Q8_1_MMQ) {
                const int k0 = MMQ_TILE_NE_K + k01;
#pragma unroll
                for (int j0 = 0; j0 < J; j0 += nwarps) {
                    int j = j0 + warp;
                    if (m_start + j >= M) continue;

#pragma unroll
                    for (int i0 = 0; i0 < I; i0 += warp_size) {
                        int i = i0 + lane;
                        if (n_start + i >= N) continue;

                        const int8_t* sc = (const int8_t*)(x_sc + i * (MMQ_TILE_NE_K / 8) + i / 8 + k0 / 16);

                        sum[(j0 / nwarps) * (I / warp_size) + i0 / warp_size] +=
                            vec_dot_q6_K_q8_1_mmq(
                                &x_qs[i * (2 * MMQ_TILE_NE_K + 1) + k0],
                                &y_qs[j * MMQ_TILE_Y_K + k01],
                                sc,
                                x_df[i * (MMQ_TILE_NE_K / GEMV_QI6_K)],
                                &y_df[j * MMQ_TILE_Y_K + k01 / GEMV_QI8_1]);
                    }
                }
            }
        } else {  // Q2_K
            constexpr auto txs_q2k = mmq_txs_q2_k<I>();
            const int*   x_qs = tile_x;
            const half2* x_dm = (const half2*)(x_qs + txs_q2k.qs);

            const half*  y_d2s6 = (const half*)tile_y;
            const int*   y_qs = (const int*)(y_d2s6 + 8);

            // Second sub-block: k01 = 0..MMQ_TILE_NE_K/2-1, step QR2_K*VDR, ns=2
            for (int k01 = 0; k01 < MMQ_TILE_NE_K / 2; k01 += GEMV_QR2_K * VDR_Q2_K_Q8_1_MMQ) {
#pragma unroll
                for (int j0 = 0; j0 < J; j0 += nwarps) {
                    int j = j0 + warp;
                    if (m_start + j >= M) continue;

#pragma unroll
                    for (int i0 = 0; i0 < I; i0 += warp_size) {
                        int i = i0 + lane;
                        if (n_start + i >= N) continue;

                        const float d8 = __half2float(y_d2s6[j * MMQ_TILE_Y_K * 2]);
                        const half2* s8 = (const half2*)(y_d2s6 + j * MMQ_TILE_Y_K * 2 + (1 + k01 / GEMV_QI8_1) * 2);
                        const int k0 = MMQ_TILE_NE_K + k01;

                        sum[(j0 / nwarps) * (I / warp_size) + i0 / warp_size] +=
                            vec_dot_q2_K_q8_1_impl_mmq(
                                &x_qs[i * (2 * MMQ_TILE_NE_K + 1) + k0],
                                &y_qs[j * MMQ_TILE_Y_K + k01],
                                &x_dm[i * (MMQ_TILE_NE_K + 1) + k0 / 4],
                                d8, s8);
                    }
                }
            }

            // Second sub-block: k01 = MMQ_TILE_NE_K/2..MMQ_TILE_NE_K-1, step QR2_K*VDR, ns=1
            for (int k01 = MMQ_TILE_NE_K / 2; k01 < MMQ_TILE_NE_K; k01 += GEMV_QR2_K * VDR_Q2_K_Q8_1_MMQ) {
#pragma unroll
                for (int j0 = 0; j0 < J; j0 += nwarps) {
                    int j = j0 + warp;
                    if (m_start + j >= M) continue;

#pragma unroll
                    for (int i0 = 0; i0 < I; i0 += warp_size) {
                        int i = i0 + lane;
                        if (n_start + i >= N) continue;

                        const float d8 = __half2float(y_d2s6[j * MMQ_TILE_Y_K * 2 + 1]);
                        const half2* s8 = (const half2*)(y_d2s6 + j * MMQ_TILE_Y_K * 2 + (1 + k01 / GEMV_QI8_1) * 2);
                        const int k0 = MMQ_TILE_NE_K + k01;

                        sum[(j0 / nwarps) * (I / warp_size) + i0 / warp_size] +=
                            vec_dot_q2_K_q8_1_impl_mmq(
                                &x_qs[i * (2 * MMQ_TILE_NE_K + 1) + k0],
                                &y_qs[j * MMQ_TILE_Y_K + k01],
                                &x_dm[i * (MMQ_TILE_NE_K + 1) + k0 / 4],
                                d8, s8);
                    }
                }
            }
        }

        __syncthreads();
    }

    // Write back results
    // Each warp handles J/nwarps activation rows
#pragma unroll
    for (int j0 = 0; j0 < J; j0 += nwarps) {
        int j = j0 + warp;
        int m = m_start + j;
        if (m >= M) continue;

#pragma unroll
        for (int i0 = 0; i0 < I; i0 += warp_size) {
            int i = i0 + lane;
            int n = n_start + i;
            if (n >= N) continue;

            out[m * N + n] = sum[(j0 / nwarps) * (I / warp_size) + i0 / warp_size];
        }
    }
}

// ============================================================================
// Launch wrappers
// ============================================================================

// Scratch buffer for Q8_1_mmq quantized activations (thread-local)
static CudaScratchPool& mmq_scratch() {
    static thread_local CudaScratchPool pool;
    return pool;
}

void launch_mmq_q3_k(const float* x, const void* q_weight, float* out,
                      int M, int K, int N, cudaStream_t stream) {
    constexpr int I = MMQ_I_Q3K;
    constexpr int J = MMQ_J_Q3K;

    int K_padded = (K + MMQ_QK8_1_MMQ - 1) / MMQ_QK8_1_MMQ * MMQ_QK8_1_MMQ;
    int num_blocks_row = (K + 255) / 256;
    int num_mmq_blocks_per_row = K_padded / MMQ_QK8_1_MMQ;

    // Pre-quantize activations
    size_t q8_bytes = (size_t)M * num_mmq_blocks_per_row * sizeof(block_q8_1_mmq);
    void* q8_buf = mmq_scratch().ensure(q8_bytes);
    launch_quantize_q8_1_mmq(x, q8_buf, K, M, MMQ_Q8_1_DS_LAYOUT_D4, stream);

    // Compute shared memory size
    constexpr auto txs = mmq_txs_q3_k<I>();
    constexpr int tile_x_ints = txs.qs + txs.dm + txs.sc;
    constexpr int tile_y_ints = J * MMQ_TILE_Y_K;
    constexpr int smem_ints = tile_x_ints + tile_y_ints;
    constexpr int smem_bytes = smem_ints * sizeof(int);

    // Grid dimensions
    int n_tiles = (N + I - 1) / I;
    int m_tiles = (M + J - 1) / J;
    dim3 grid(n_tiles, m_tiles);
    dim3 block(32, MMQ_NWARPS);

    mul_mat_q_kernel<DataType::Q3_K, I, J>
        <<<grid, block, smem_bytes, stream>>>(
            static_cast<const uint8_t*>(q_weight),
            static_cast<const int*>(q8_buf),
            out, M, K, N, num_blocks_row, K_padded);
}

void launch_mmq_q4_k(const float* x, const void* q_weight, float* out,
                      int M, int K, int N, cudaStream_t stream) {
    constexpr int I = MMQ_I_Q4K;
    constexpr int J = MMQ_J_Q4K;

    int K_padded = (K + MMQ_QK8_1_MMQ - 1) / MMQ_QK8_1_MMQ * MMQ_QK8_1_MMQ;
    int num_blocks_row = (K + 255) / 256;
    int num_mmq_blocks_per_row = K_padded / MMQ_QK8_1_MMQ;

    // Pre-quantize activations (DS4 layout for Q4_K)
    size_t q8_bytes = (size_t)M * num_mmq_blocks_per_row * sizeof(block_q8_1_mmq);
    void* q8_buf = mmq_scratch().ensure(q8_bytes);
    launch_quantize_q8_1_mmq(x, q8_buf, K, M, MMQ_Q8_1_DS_LAYOUT_DS4, stream);

    constexpr auto txs = mmq_txs_q4_k<I>();
    constexpr int tile_x_ints = txs.qs + txs.dm + txs.sc;
    constexpr int tile_y_ints = J * MMQ_TILE_Y_K;
    constexpr int smem_ints = tile_x_ints + tile_y_ints;
    constexpr int smem_bytes = smem_ints * sizeof(int);

    int n_tiles = (N + I - 1) / I;
    int m_tiles = (M + J - 1) / J;
    dim3 grid(n_tiles, m_tiles);
    dim3 block(32, MMQ_NWARPS);

    mul_mat_q_kernel<DataType::Q4_K, I, J>
        <<<grid, block, smem_bytes, stream>>>(
            static_cast<const uint8_t*>(q_weight),
            static_cast<const int*>(q8_buf),
            out, M, K, N, num_blocks_row, K_padded);
}

void launch_mmq_q5_k(const float* x, const void* q_weight, float* out,
                      int M, int K, int N, cudaStream_t stream) {
    constexpr int I = MMQ_I_Q5K;
    constexpr int J = MMQ_J_Q5K;

    int K_padded = (K + MMQ_QK8_1_MMQ - 1) / MMQ_QK8_1_MMQ * MMQ_QK8_1_MMQ;
    int num_blocks_row = (K + 255) / 256;
    int num_mmq_blocks_per_row = K_padded / MMQ_QK8_1_MMQ;

    // Pre-quantize activations (DS4 layout for Q5_K)
    size_t q8_bytes = (size_t)M * num_mmq_blocks_per_row * sizeof(block_q8_1_mmq);
    void* q8_buf = mmq_scratch().ensure(q8_bytes);
    launch_quantize_q8_1_mmq(x, q8_buf, K, M, MMQ_Q8_1_DS_LAYOUT_DS4, stream);

    constexpr auto txs = mmq_txs_q5_k<I>();
    constexpr int tile_x_ints = txs.qs + txs.dm + txs.sc;
    constexpr int tile_y_ints = J * MMQ_TILE_Y_K;
    constexpr int smem_ints = tile_x_ints + tile_y_ints;
    constexpr int smem_bytes = smem_ints * sizeof(int);

    int n_tiles = (N + I - 1) / I;
    int m_tiles = (M + J - 1) / J;
    dim3 grid(n_tiles, m_tiles);
    dim3 block(32, MMQ_NWARPS);

    mul_mat_q_kernel<DataType::Q5_K, I, J>
        <<<grid, block, smem_bytes, stream>>>(
            static_cast<const uint8_t*>(q_weight),
            static_cast<const int*>(q8_buf),
            out, M, K, N, num_blocks_row, K_padded);
}

void launch_mmq_q4_0(const float* x, const void* q_weight, float* out,
                      int M, int K, int N, cudaStream_t stream) {
    constexpr int I = MMQ_I_Q4_0;
    constexpr int J = MMQ_J_Q4_0;

    int K_padded = (K + MMQ_QK8_1_MMQ - 1) / MMQ_QK8_1_MMQ * MMQ_QK8_1_MMQ;
    // q4_0 rows are strided by 32-element blocks (18 B each), not by
    // super-blocks: the row stride must count true blocks or every weight
    // row overlaps its neighbours.
    int num_blocks_row = (K + 31) / 32;
    int num_mmq_blocks_per_row = K_padded / MMQ_QK8_1_MMQ;

    // Pre-quantize activations (DS4 layout for Q4_0)
    size_t q8_bytes = (size_t)M * num_mmq_blocks_per_row * sizeof(block_q8_1_mmq);
    void* q8_buf = mmq_scratch().ensure(q8_bytes);
    launch_quantize_q8_1_mmq(x, q8_buf, K, M, MMQ_Q8_1_DS_LAYOUT_DS4, stream);

    constexpr auto txs = mmq_txs_q4_0<I>();
    constexpr int tile_x_ints = txs.qs + txs.dm + txs.sc;
    constexpr int tile_y_ints = J * MMQ_TILE_Y_K;
    constexpr int smem_ints = tile_x_ints + tile_y_ints;
    constexpr int smem_bytes = smem_ints * sizeof(int);

    int n_tiles = (N + I - 1) / I;
    int m_tiles = (M + J - 1) / J;
    dim3 grid(n_tiles, m_tiles);
    dim3 block(32, MMQ_NWARPS);

    mul_mat_q_kernel<DataType::Q4_0, I, J>
        <<<grid, block, smem_bytes, stream>>>(
            static_cast<const uint8_t*>(q_weight),
            static_cast<const int*>(q8_buf),
            out, M, K, N, num_blocks_row, K_padded);
}

void launch_mmq_q6_k(const float* x, const void* q_weight, float* out,
                      int M, int K, int N, cudaStream_t stream) {
    constexpr int I = MMQ_I_Q6_K;
    constexpr int J = MMQ_J_Q6_K;

    int K_padded = (K + MMQ_QK8_1_MMQ - 1) / MMQ_QK8_1_MMQ * MMQ_QK8_1_MMQ;
    int num_blocks_row = (K + 255) / 256;
    int num_mmq_blocks_per_row = K_padded / MMQ_QK8_1_MMQ;

    // Pre-quantize activations (D4 layout for Q6_K)
    size_t q8_bytes = (size_t)M * num_mmq_blocks_per_row * sizeof(block_q8_1_mmq);
    void* q8_buf = mmq_scratch().ensure(q8_bytes);
    launch_quantize_q8_1_mmq(x, q8_buf, K, M, MMQ_Q8_1_DS_LAYOUT_D4, stream);

    constexpr auto txs = mmq_txs_q6_k<I>();
    constexpr int tile_x_ints = txs.qs + txs.dm + txs.sc;
    constexpr int tile_y_ints = J * MMQ_TILE_Y_K;
    constexpr int smem_ints = tile_x_ints + tile_y_ints;
    constexpr int smem_bytes = smem_ints * sizeof(int);

    int n_tiles = (N + I - 1) / I;
    int m_tiles = (M + J - 1) / J;
    dim3 grid(n_tiles, m_tiles);
    dim3 block(32, MMQ_NWARPS);

    mul_mat_q_kernel<DataType::Q6_K, I, J>
        <<<grid, block, smem_bytes, stream>>>(
            static_cast<const uint8_t*>(q_weight),
            static_cast<const int*>(q8_buf),
            out, M, K, N, num_blocks_row, K_padded);
}

void launch_mmq_q2_k(const float* x, const void* q_weight, float* out,
                      int M, int K, int N, cudaStream_t stream) {
    constexpr int I = MMQ_I_Q2_K;
    constexpr int J = MMQ_J_Q2_K;

    int K_padded = (K + MMQ_QK8_1_MMQ - 1) / MMQ_QK8_1_MMQ * MMQ_QK8_1_MMQ;
    int num_blocks_row = (K + 255) / 256;
    int num_mmq_blocks_per_row = K_padded / MMQ_QK8_1_MMQ;

    // Pre-quantize activations (D2S6 layout for Q2_K)
    size_t q8_bytes = (size_t)M * num_mmq_blocks_per_row * sizeof(block_q8_1_mmq);
    void* q8_buf = mmq_scratch().ensure(q8_bytes);
    launch_quantize_q8_1_mmq(x, q8_buf, K, M, MMQ_Q8_1_DS_LAYOUT_D2S6, stream);

    constexpr auto txs = mmq_txs_q2_k<I>();
    constexpr int tile_x_ints = txs.qs + txs.dm + txs.sc;
    constexpr int tile_y_ints = J * MMQ_TILE_Y_K;
    constexpr int smem_ints = tile_x_ints + tile_y_ints;
    constexpr int smem_bytes = smem_ints * sizeof(int);

    int n_tiles = (N + I - 1) / I;
    int m_tiles = (M + J - 1) / J;
    dim3 grid(n_tiles, m_tiles);
    dim3 block(32, MMQ_NWARPS);

    mul_mat_q_kernel<DataType::Q2_K, I, J>
        <<<grid, block, smem_bytes, stream>>>(
            static_cast<const uint8_t*>(q_weight),
            static_cast<const int*>(q8_buf),
            out, M, K, N, num_blocks_row, K_padded);
}

// ============================================================================
// MMQ dispatch table
// ============================================================================

using MmqFn = void (*)(const float*, const void*, float*, int, int, int, cudaStream_t);

extern const MmqFn mmq_dispatch[20] = {
    /* FP32=0  */ nullptr,
    /* FP16=1  */ nullptr,
    /* Q4_0=2  */ launch_mmq_q4_0,
    /* Q4_1=3  */ nullptr,
    /* Q4_K=4  */ launch_mmq_q4_k,
    /* INT8=5  */ nullptr,
    /* INT32=6 */ nullptr,
    /* Q8_0=7  */ nullptr,
    /* Q5_0=8  */ nullptr,
    /* Q5_1=9  */ nullptr,
    /* Q2_K=10 */ launch_mmq_q2_k,
    /* Q3_K=11 */ launch_mmq_q3_k,
    /* Q5_K=12 */ launch_mmq_q5_k,
    /* Q6_K=13 */ launch_mmq_q6_k,
    /* IQ2_S=14  */ nullptr,
    /* BF16=15  */ nullptr,
    /* IQ2_XXS=16 */ nullptr,
    /* IQ4_NL=17  */ nullptr,
    /* IQ2_XS=18 */ nullptr,
    /* IQ3_S=19  */ nullptr,
};

}  // namespace cuda
}  // namespace forge
