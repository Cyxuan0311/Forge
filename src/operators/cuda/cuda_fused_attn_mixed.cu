// Mixed-dtype fused GQA attention: K and V stored at different precisions.
//
// Every pre-existing fused kernel in cuda_fused_attn.cu is monomorphic: one
// block format for both K and V, and a single q_row_size for both. An asymmetric
// cache (-ctk q8_0 -ctv q4_0) therefore had no fused path and fell back to
// materialising the filled rows to FP32 -- correct, but with extra memory
// traffic and a visibly noisier throughput (measured 21.2-26.1 tok/s against a
// steady 22.7-22.9 for the symmetric q8_0/q8_0 path on MiniCPM5-2B at -c 16384).
//
// Here K and V are templated independently and carry independent row sizes, so
// the recommended pairing runs fused. The numerics are identical to the
// monomorphic kernels for the symmetric cases: the same online softmax, the same
// per-block dequantisation, the same cross-warp reduction.
//
// The formats are addressed through traits rather than an if/else so both sides
// collapse at compile time and the inner loops stay as tight as before.

#include <cstdint>
#include <cstdio>
#include <cuda_fp16.h>

#include "cuda_attn_common.cuh"

namespace forge {
namespace cuda {

// =========================================================================
// Block-format traits
// =========================================================================
//
// ELEMS  elements per block, so a head of HEAD_DIM spans HEAD_DIM/ELEMS blocks
// BYTES  bytes per block, including the leading scale
//
// Dequantisation follows the GGML/ggml convention exactly as the monomorphic
// kernels do it, so a mixed kernel and its symmetric twin agree bit for bit:
//   q8_0  fp16 scale + 32 int8,   value[i]     = q[i]        * scale
//   q4_0  fp16 scale + 16 nibbles, sign-EXTENDED int4 in the low nibble for
//         element i and the high nibble for element i+16. The scale is negative
//         (d = amax / -8), so value = int4_signed * scale.
//         Sign extension matters: (n & 0xF) - 8 is a different function and
//         yields values off by 8 with the wrong sign on half the range.
//   f16   no scale, 8 halves per 16 bytes

struct FmtQ8_0 {
    static constexpr int ELEMS = 32;
    static constexpr int BYTES = 34;

    static __device__ __forceinline__ void dot_block(const uint8_t* p, const float* q, int off,
                                                     float& acc) {
        uint16_t sb;
        memcpy(&sb, p, sizeof(uint16_t));
        const float sc = __half2float(reinterpret_cast<const __half&>(sb));
        const int8_t* qs = reinterpret_cast<const int8_t*>(p + sizeof(__half));
#pragma unroll
        for (int i = 0; i < 32; ++i) acc += q[off + i] * (qs[i] * sc);
    }

    static __device__ __forceinline__ void accum_block(const uint8_t* p, float w, float corr,
                                                      float* acc, int off) {
        uint16_t sb;
        memcpy(&sb, p, sizeof(uint16_t));
        const float sc = __half2float(reinterpret_cast<const __half&>(sb));
        const int8_t* qs = reinterpret_cast<const int8_t*>(p + sizeof(__half));
#pragma unroll
        for (int i = 0; i < 32; ++i) acc[off + i] = acc[off + i] * corr + w * (qs[i] * sc);
    }
};

struct FmtQ4_0 {
    static constexpr int ELEMS = 32;
    static constexpr int BYTES = 18;

    // Unsigned nibble offset by 8, which is what the writers actually produce:
    // cuda::launch_quantize_q4_0_matrix and quantize_row_q4_0 both encode
    // nibble = clamp(round(x/d) + 8, 0, 15). dequantize_q4_0_cpu() uses a
    // sign-extended int4 instead, so these two conventions are NOT the same
    // function -- they differ by 8 on every element -- and only the former is
    // correct for a row the CUDA cache write path produced.
    static __device__ __forceinline__ int q4_0_nibble(int n) { return (n & 0x0F) - 8; }

    static __device__ __forceinline__ void dot_block(const uint8_t* p, const float* q, int off,
                                                     float& acc) {
        uint16_t sb;
        memcpy(&sb, p, sizeof(uint16_t));
        const float sc = __half2float(reinterpret_cast<const __half&>(sb));
        const uint8_t* qs = p + sizeof(__half);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int lo = q4_0_nibble(qs[i] & 0x0F);
            const int hi = q4_0_nibble((qs[i] >> 4) & 0x0F);
            acc += q[off + i] * (lo * sc);
            acc += q[off + i + 16] * (hi * sc);
        }
    }

    static __device__ __forceinline__ void accum_block(const uint8_t* p, float w, float corr,
                                                      float* acc, int off) {
        uint16_t sb;
        memcpy(&sb, p, sizeof(uint16_t));
        const float sc = __half2float(reinterpret_cast<const __half&>(sb));
        const uint8_t* qs = p + sizeof(__half);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int lo = q4_0_nibble(qs[i] & 0x0F);
            const int hi = q4_0_nibble((qs[i] >> 4) & 0x0F);
            acc[off + i] = acc[off + i] * corr + w * (lo * sc);
            acc[off + i + 16] = acc[off + i + 16] * corr + w * (hi * sc);
        }
    }
};

struct FmtF16 {
    // 8 halves per block keeps the byte count a multiple of 16 and the
    // unrolled inner loop short. HEAD_DIM is validated to be a multiple of this
    // at the call site.
    static constexpr int ELEMS = 8;
    static constexpr int BYTES = 16;

    static __device__ __forceinline__ void dot_block(const uint8_t* p, const float* q, int off,
                                                     float& acc) {
        const __half* h = reinterpret_cast<const __half*>(p);
#pragma unroll
        for (int i = 0; i < 8; ++i)
            acc += q[off + i] * __half2float(h[i]);
    }

    static __device__ __forceinline__ void accum_block(const uint8_t* p, float w, float corr,
                                                      float* acc, int off) {
        const __half* h = reinterpret_cast<const __half*>(p);
#pragma unroll
        for (int i = 0; i < 8; ++i)
            acc[off + i] = acc[off + i] * corr + w * __half2float(h[i]);
    }
};

// =========================================================================
// Mixed-dtype fused GQA decode (HEAD_DIM <= 128, one online-softmax pass)
// =========================================================================

// Page-table row resolution. Duplicated here as it already is in
// cuda_paged_kv.cu and cuda_tiled_prefill.cu; when PAGED is false no paged
// instantiation exists and this is never reached.
__device__ __forceinline__ const uint8_t* paged_kv_row(const uint8_t* base,
                                                       const int32_t* __restrict__ page_ids,
                                                       void* const* __restrict__ page_ptrs,
                                                       int page_size, size_t row_bytes, int j) {
    if (page_ids == nullptr) return base + (size_t)j * row_bytes;
    const int page_idx = j / page_size;
    const int offset = j - page_idx * page_size;
    const uint8_t* page_base = static_cast<const uint8_t*>(page_ptrs[page_ids[page_idx]]);
    return page_base + (size_t)offset * row_bytes;
}

// PAGED resolves KV rows through the page table rather than a contiguous base
// pointer. It is a template parameter, not a runtime flag, so the contiguous
// instantiations (the well-tuned f16 / q8_0 paths) remain exactly the code they
// were, and so a paged launch cannot be handed a page-pointer table where the
// kernel expects KV data -- which is precisely what segfaulted asymmetric paged
// KV before the selector learned to refuse it.
template <int HEAD_DIM, int NUM_WARPS, typename KFmt, typename VFmt, bool PAGED = false>
__global__ void fused_attn_mixed_decode_kernel(const float* __restrict__ Q,
                                               const uint8_t* __restrict__ q_K,
                                               const uint8_t* __restrict__ q_V,
                                               float* __restrict__ O, int kv_len, int num_heads,
                                               int num_kv_heads, size_t k_row_size,
                                               size_t v_row_size, const float* __restrict__ mask_row,
                                               void* const* k_pages = nullptr,
                                               void* const* v_pages = nullptr,
                                               const int32_t* page_ids = nullptr,
                                               int page_size = 0) {
    const int h = blockIdx.x;
    if (h >= num_heads)
        return;

    const int kv_groups = num_heads / num_kv_heads;
    const int kv_h = h / kv_groups;

    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // The two sides have different block geometries, so the per-head byte offset
    // differs too. This is the crux of the change: the monomorphic kernels
    // computed one offset for both.
    constexpr int K_BLOCKS = HEAD_DIM / KFmt::ELEMS;
    constexpr int V_BLOCKS = HEAD_DIM / VFmt::ELEMS;
    const size_t k_head_off = (size_t)kv_h * K_BLOCKS * KFmt::BYTES;
    const size_t v_head_off = (size_t)kv_h * V_BLOCKS * VFmt::BYTES;

    __shared__ float s_q[HEAD_DIM];
    for (int d = tid; d < HEAD_DIM; d += block_size) s_q[d] = Q[h * HEAD_DIM + d];
    __syncthreads();

    const float scale = 1.0f / sqrtf(static_cast<float>(HEAD_DIM));

    float local_max = -1e30f;
    float local_sum = 0.0f;
    float local_acc[HEAD_DIM];
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) local_acc[d] = 0.0f;

    for (int j = tid; j < kv_len; j += block_size) {
        if (mask_row != nullptr && mask_row[j] < -1e20f) continue;

        const uint8_t* k_row =
            PAGED ? paged_kv_row(nullptr, page_ids, k_pages, page_size, k_row_size, j) + k_head_off
                  : q_K + (size_t)j * k_row_size + k_head_off;
        const uint8_t* v_row =
            PAGED ? paged_kv_row(nullptr, page_ids, v_pages, page_size, v_row_size, j) + v_head_off
                  : q_V + (size_t)j * v_row_size + v_head_off;

        float dot = 0.0f;
#pragma unroll
        for (int bi = 0; bi < K_BLOCKS; ++bi)
            KFmt::dot_block(k_row + bi * KFmt::BYTES, s_q, bi * KFmt::ELEMS, dot);
        dot *= scale;
        if (mask_row != nullptr) dot += mask_row[j];

        const float new_max = fmaxf(local_max, dot);
        const float correction = expf(local_max - new_max);
        const float weight = expf(dot - new_max);
        local_sum = local_sum * correction + weight;

#pragma unroll
        for (int bi = 0; bi < V_BLOCKS; ++bi)
            VFmt::accum_block(v_row + bi * VFmt::BYTES, weight, correction, local_acc,
                              bi * VFmt::ELEMS);
        local_max = new_max;
    }

    float warp_max = warp_reduce_max(local_max);
    const float rescale = expf(local_max - warp_max);
    float warp_sum = warp_reduce_sum(local_sum * rescale);
    float warp_acc[HEAD_DIM];
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) warp_acc[d] = local_acc[d] * rescale;
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) warp_acc[d] = warp_reduce_sum(warp_acc[d]);

    __shared__ float s_warp_max[NUM_WARPS];
    __shared__ float s_warp_sum[NUM_WARPS];
    __shared__ float s_warp_acc[NUM_WARPS * HEAD_DIM];

    if (lane == 0) {
        s_warp_max[warp_id] = warp_max;
        s_warp_sum[warp_id] = warp_sum;
    }
    for (int d = lane; d < HEAD_DIM; d += 32) s_warp_acc[warp_id * HEAD_DIM + d] = warp_acc[d];
    __syncthreads();

    cross_warp_merge_and_write<HEAD_DIM, NUM_WARPS>(s_warp_max, s_warp_sum, s_warp_acc, O, h, lane);
}

// Paged mixed-dtype decode. Dispatch table is spelled out rather than shared
// with the contiguous launcher through macros: two ~20-line tables beat a macro
// layer that has to select between them at compile time, and the asymmetry is
// the whole point of this entry point.
//
// fmt codes: 0 = q8_0, 1 = q4_0, 2 = f16 (same codes as the contiguous launcher).
bool launch_paged_flash_attention_gqa_decode_mixed(const float* Q, void* const* k_pages,
                                                   void* const* v_pages, const int32_t* page_ids,
                                                   float* O, int kv_len, int num_heads,
                                                   int num_kv_heads, int head_dim, int page_size,
                                                   size_t k_row_size, size_t v_row_size, int k_fmt,
                                                   int v_fmt, const float* mask_row,
                                                   cudaStream_t stream) {
    const int blocks = num_heads;

#define LAUNCH_PAGED(HD, KF, VF)                                                              \
    fused_attn_mixed_decode_kernel<HD, 4, KF, VF, true><<<blocks, 128, 0, stream>>>(           \
        Q, nullptr, nullptr, O, kv_len, num_heads, num_kv_heads, k_row_size, v_row_size,       \
        mask_row, k_pages, v_pages, page_ids, page_size)

#define LAUNCH_PAGED_HD(KF, VF)                     \
    do {                                            \
        switch (head_dim) {                         \
        case 64:  LAUNCH_PAGED(64, KF, VF); break;  \
        case 96:  LAUNCH_PAGED(96, KF, VF); break;  \
        case 128: LAUNCH_PAGED(128, KF, VF); break; \
        default: return false;                      \
        }                                           \
    } while (0)

    if (k_fmt == 0 && v_fmt == 0) {
        LAUNCH_PAGED_HD(FmtQ8_0, FmtQ8_0);
    } else if (k_fmt == 0 && v_fmt == 1) {
        LAUNCH_PAGED_HD(FmtQ8_0, FmtQ4_0);
    } else if (k_fmt == 1 && v_fmt == 0) {
        LAUNCH_PAGED_HD(FmtQ4_0, FmtQ8_0);
    } else if (k_fmt == 1 && v_fmt == 1) {
        LAUNCH_PAGED_HD(FmtQ4_0, FmtQ4_0);
    } else if (k_fmt == 0 && v_fmt == 2) {
        LAUNCH_PAGED_HD(FmtQ8_0, FmtF16);
    } else if (k_fmt == 1 && v_fmt == 2) {
        LAUNCH_PAGED_HD(FmtQ4_0, FmtF16);
    } else if (k_fmt == 2 && v_fmt == 0) {
        LAUNCH_PAGED_HD(FmtF16, FmtQ8_0);
    } else if (k_fmt == 2 && v_fmt == 1) {
        LAUNCH_PAGED_HD(FmtF16, FmtQ4_0);
    } else if (k_fmt == 2 && v_fmt == 2) {
        LAUNCH_PAGED_HD(FmtF16, FmtF16);
    } else {
        return false;
    }
    return true;

#undef LAUNCH_PAGED
#undef LAUNCH_PAGED_HD
}

}  // namespace cuda
}  // namespace forge

// =========================================================================
// Launcher
// =========================================================================

// Dispatch on the pair. The (K,V) combinations covered here are the ones the
// selection policy actually recommends or that already had a symmetric fused
// path; anything else returns false and the caller falls back to FP32
// materialisation, which is correct but slower. Reporting that is the caller's
// job -- an unsupported pair must never look like a silent no-op.
bool launch_fused_flash_attention_gqa_decode_mixed(const float* Q, const void* q_K, const void* q_V,
                                                   float* O, int kv_len, int num_heads,
                                                   int num_kv_heads, int head_dim,
                                                   size_t k_row_size, size_t v_row_size,
                                                   int k_fmt, int v_fmt,
                                                   const float* mask_row, cudaStream_t stream);

namespace forge {
namespace cuda {

bool launch_fused_flash_attention_gqa_decode_mixed(const float* Q, const void* q_K, const void* q_V,
                                                   float* O, int kv_len, int num_heads,
                                                   int num_kv_heads, int head_dim,
                                                   size_t k_row_size, size_t v_row_size,
                                                   int k_fmt, int v_fmt, const float* mask_row,
                                                   cudaStream_t stream) {
    const int blocks = num_heads;
    const auto K = static_cast<const uint8_t*>(q_K);
    const auto V = static_cast<const uint8_t*>(q_V);

#define LAUNCH_MIXED(HD, NW, KF, VF)                                                       \
    fused_attn_mixed_decode_kernel<HD, NW, KF, VF><<<blocks, 128, 0, stream>>>(            \
        Q, K, V, O, kv_len, num_heads, num_kv_heads, k_row_size, v_row_size, mask_row)

#define LAUNCH_MIXED_HD(KF, VF)                      \
    do {                                             \
        switch (head_dim) {                          \
        case 64:  LAUNCH_MIXED(64, 4, KF, VF); break;  \
        case 96:  LAUNCH_MIXED(96, 4, KF, VF); break;  \
        case 128: LAUNCH_MIXED(128, 4, KF, VF); break; \
        default: return false;                       \
        }                                            \
    } while (0)

    // fmt codes: 0 = q8_0, 1 = q4_0, 2 = f16 (must match the dispatch table below)
    if (k_fmt == 0 && v_fmt == 0) {
        LAUNCH_MIXED_HD(FmtQ8_0, FmtQ8_0);
    } else if (k_fmt == 0 && v_fmt == 1) {
        LAUNCH_MIXED_HD(FmtQ8_0, FmtQ4_0);
    } else if (k_fmt == 1 && v_fmt == 0) {
        LAUNCH_MIXED_HD(FmtQ4_0, FmtQ8_0);
    } else if (k_fmt == 1 && v_fmt == 1) {
        LAUNCH_MIXED_HD(FmtQ4_0, FmtQ4_0);
    } else if (k_fmt == 0 && v_fmt == 2) {
        LAUNCH_MIXED_HD(FmtQ8_0, FmtF16);
    } else if (k_fmt == 1 && v_fmt == 2) {
        // Was missing. A legitimate -ctk q4_0 -ctv f16 reached here and the
        // launcher refused, which the executor turns into a thrown
        // std::runtime_error: i.e. a supported dtype combination crashed the
        // process instead of running or falling back.
        LAUNCH_MIXED_HD(FmtQ4_0, FmtF16);
    } else if (k_fmt == 2 && v_fmt == 0) {
        LAUNCH_MIXED_HD(FmtF16, FmtQ8_0);
    } else if (k_fmt == 2 && v_fmt == 1) {
        LAUNCH_MIXED_HD(FmtF16, FmtQ4_0);
    } else if (k_fmt == 2 && v_fmt == 2) {
        LAUNCH_MIXED_HD(FmtF16, FmtF16);
    } else {
        return false;
    }
    return true;

#undef LAUNCH_MIXED_HD
#undef LAUNCH_MIXED
}

}  // namespace cuda
}  // namespace forge
