// =========================================================================
// cuda_tiled_prefill.cu — KV-tiled flash-attention PREFILL (Phase 2-C + 2-B)
//
// Motivation (2-C "full tiling"):
//   The original prefill kernels keep `local_acc[HEAD_DIM]` live in registers
//   across the whole KV loop. For HEAD_DIM=128 that is 128 registers per thread
//   purely for the accumulator, which collapses occupancy. This kernel instead
//   processes KV in tiles and keeps only HEAD_DIM/threads elements of the
//   accumulator per thread (1 element for HEAD_DIM=128 at 128 threads/block).
//
// Algorithm (per block = one (qi, h)):
//   For each KV tile:
//     1. Each thread owns one KV position of the tile and computes its score
//        s_j (one scalar — no per-thread accumulator).
//     2. Block-reduce max over the tile -> m_t; w_j = exp(s_j - m_t).
//     3. Block-reduce sum of w_j -> l_t.
//     4. Merge into the running (m_g, l_g, acc) with the standard flash
//        rescaling:  acc *= exp(m_g - m_new);  acc += exp(m_t - m_new) * P·V
//     5. P·V accumulation is distributed over the head dim: thread t owns
//        d = t, t+threads, ... so the per-thread accumulator stays tiny and the
//        V reads stay coalesced.
//   Final: O[d] = acc[d] / l_g.
//   Fully-masked tiles are skipped without perturbing (m_g, l_g).
//
// KV format is a compile-time policy (Dequant), so the SAME template serves
// FP32, F16, Q8_0 and FP8 (E4M3/E5M2), contiguous or paged (Phase 2-B).
// =========================================================================

#include <cstdio>

#include <cuda_fp16.h>

#include "cuda_attn_common.cuh"
#include "forge/cuda_kernels.h"
#include "forge/fp8_utils.h"

namespace forge {
namespace cuda {

// =========================================================================
// KV format policies.
//   head_offset(): byte offset of this kv_head's slice inside a KV row.
//   deq():         dequantized element value (already includes FP8 row scale).
// `scales` is the per-(row, kv_head) FP8 scale buffer, or nullptr otherwise.
// =========================================================================

struct DqFp32 {
    static constexpr int kMinHeadDim = 16;
    template <int HEAD_DIM>
    __device__ __forceinline__ static size_t head_offset(int kv_h) {
        return (size_t)kv_h * HEAD_DIM * sizeof(float);
    }
    template <int HEAD_DIM>
    __device__ __forceinline__ static float deq(const uint8_t* row, int d, int j, int kv_h,
                                                int num_kv_heads, const float* scales) {
        (void)j; (void)kv_h; (void)num_kv_heads; (void)scales;
        return reinterpret_cast<const float*>(row)[d];
    }
};

struct DqF16 {
    static constexpr int kMinHeadDim = 16;
    template <int HEAD_DIM>
    __device__ __forceinline__ static size_t head_offset(int kv_h) {
        return (size_t)kv_h * HEAD_DIM * sizeof(__half);
    }
    template <int HEAD_DIM>
    __device__ __forceinline__ static float deq(const uint8_t* row, int d, int j, int kv_h,
                                                int num_kv_heads, const float* scales) {
        (void)j; (void)kv_h; (void)num_kv_heads; (void)scales;
        return __half2float(reinterpret_cast<const __half*>(row)[d]);
    }
};

// Q8_0: 32-element blocks, each = fp16 d (2B) + int8 qs[32] (32B) = 34B.
// Requires HEAD_DIM >= 32 (a head must hold at least one whole block).
struct DqQ8_0 {
    static constexpr int kMinHeadDim = 32;
    template <int HEAD_DIM>
    __device__ __forceinline__ static size_t head_offset(int kv_h) {
        return (size_t)kv_h * (HEAD_DIM / 32) * 34;
    }
    template <int HEAD_DIM>
    __device__ __forceinline__ static float deq(const uint8_t* row, int d, int j, int kv_h,
                                                int num_kv_heads, const float* scales) {
        (void)j; (void)kv_h; (void)num_kv_heads; (void)scales;
        const int blk = d >> 5;
        const int i = d - (blk << 5);
        const uint8_t* bp = row + blk * 34;
        uint16_t sb;
        memcpy(&sb, bp, sizeof(uint16_t));
        const float s = __half2float(reinterpret_cast<const __half&>(sb));
        const int8_t* qs = reinterpret_cast<const int8_t*>(bp + 2);
        return static_cast<float>(qs[i]) * s;
    }
};

// FP8 (E4M3 / E5M2): 1 byte/element, no block; per-(row, kv_head) fp32 scale.
template <typename FMT>
struct DqFp8 {
    static constexpr int kMinHeadDim = 16;
    template <int HEAD_DIM>
    __device__ __forceinline__ static size_t head_offset(int kv_h) {
        return (size_t)kv_h * HEAD_DIM;
    }
    template <int HEAD_DIM>
    __device__ __forceinline__ static float deq(const uint8_t* row, int d, int j, int kv_h,
                                                int num_kv_heads, const float* scales) {
        const float s = (scales != nullptr) ? scales[j * num_kv_heads + kv_h] : 1.0f;
        return fp8_load<FMT>(row, d) * s;
    }
};

// =========================================================================
// Row addressing: contiguous (base + j*row_bytes) or paged
// (page_ptrs[page_ids[j/page_size]] + (j%page_size)*row_bytes).
// Mirrors paged_kv_row() in cuda_paged_kv.cu.
// =========================================================================

template <typename Dequant, bool PAGED, int HEAD_DIM>
struct KvView {
    const uint8_t* k_base;
    const uint8_t* v_base;
    void* const* k_pages;
    void* const* v_pages;
    const int32_t* page_ids;
    int page_size;
    size_t row_bytes;
    size_t head_off;
    int kv_h;
    int num_kv_heads;
    const float* k_scales;
    const float* v_scales;

    __device__ __forceinline__ const uint8_t* krow(int j) const {
        const uint8_t* base;
        if (PAGED) {
            const int page_idx = j / page_size;
            const int off = j - page_idx * page_size;
            base = static_cast<const uint8_t*>(k_pages[page_ids[page_idx]]) +
                   (size_t)off * row_bytes;
        } else {
            base = k_base + (size_t)j * row_bytes;
        }
        return base + head_off;
    }

    __device__ __forceinline__ const uint8_t* vrow(int j) const {
        const uint8_t* base;
        if (PAGED) {
            const int page_idx = j / page_size;
            const int off = j - page_idx * page_size;
            base = static_cast<const uint8_t*>(v_pages[page_ids[page_idx]]) +
                   (size_t)off * row_bytes;
        } else {
            base = v_base + (size_t)j * row_bytes;
        }
        return base + head_off;
    }

    __device__ __forceinline__ float kd(const uint8_t* row, int d, int j) const {
        return Dequant::template deq<HEAD_DIM>(row, d, j, kv_h, num_kv_heads, k_scales);
    }
    __device__ __forceinline__ float vd(const uint8_t* row, int d, int j) const {
        return Dequant::template deq<HEAD_DIM>(row, d, j, kv_h, num_kv_heads, v_scales);
    }
};

// =========================================================================
// The tiled prefill kernel.
// =========================================================================

template <typename Dequant, bool PAGED, int HEAD_DIM, int BLOCK_SIZE>
__global__ void fused_attn_tiled_prefill_kernel(
    const float* __restrict__ Q, const uint8_t* __restrict__ K, const uint8_t* __restrict__ V,
    float* __restrict__ O, int q_len, int kv_len, int num_heads, int num_kv_heads,
    size_t row_bytes, const float* __restrict__ mask, bool causal, void* const* k_pages,
    void* const* v_pages, const int32_t* page_ids, int page_size, const float* k_scales,
    const float* v_scales) {
    const int h = blockIdx.y;
    const int qi = blockIdx.x;
    if (h >= num_heads || qi >= q_len) return;

    const int q_pos = kv_len - q_len + qi;
    const int kv_groups = num_heads / num_kv_heads;
    const int kv_h = h / kv_groups;

    const float* q_row = Q + qi * num_heads * HEAD_DIM + h * HEAD_DIM;
    float* o_row = O + qi * num_heads * HEAD_DIM + h * HEAD_DIM;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    constexpr int TILE = BLOCK_SIZE;  // one KV position per thread per tile
    // Output elements owned by this thread — the whole point of the tiling.
    constexpr int DPT = (HEAD_DIM + BLOCK_SIZE - 1) / BLOCK_SIZE;

    const KvView<Dequant, PAGED, HEAD_DIM> kv{
        K, V, k_pages, v_pages, page_ids, page_size, row_bytes,
        Dequant::template head_offset<HEAD_DIM>(kv_h), kv_h, num_kv_heads, k_scales, v_scales};

    __shared__ float s_q[HEAD_DIM];
    for (int d = tid; d < HEAD_DIM; d += BLOCK_SIZE) s_q[d] = q_row[d];
    __syncthreads();

    __shared__ float s_w[TILE];  // tile softmax weights w_j
    __shared__ float s_wmax[NUM_WARPS];
    __shared__ float s_wsum[NUM_WARPS];

    const float attn_scale = 1.0f / sqrtf(static_cast<float>(HEAD_DIM));

    // Running (max, sum) — identical value in every thread, kept in registers.
    float m_g = -1e30f;
    float l_g = 0.0f;
    float acc[DPT];
#pragma unroll
    for (int i = 0; i < DPT; ++i) acc[i] = 0.0f;

    for (int tile_start = 0; tile_start < kv_len; tile_start += TILE) {
        const int j = tile_start + tid;

        // ---- Phase A: score for this thread's KV position (one scalar) ----
        float s = -1e30f;
        if (j < kv_len) {
            const float m_val = (mask != nullptr) ? mask[qi * kv_len + j] : 0.0f;
            const bool skip = (m_val < -1e20f) || (!mask && causal && j > q_pos);
            if (!skip) {
                const uint8_t* krow = kv.krow(j);
                float dot = 0.0f;
#pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) dot += s_q[d] * kv.kd(krow, d, j);
                dot *= attn_scale;
                s = dot + m_val;
            }
        }

        // ---- Tile max (block reduce) ----
        const float wmax = warp_reduce_max(s);
        if (lane == 0) s_wmax[warp_id] = wmax;
        __syncthreads();
        float m_tile = -1e30f;
#pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w) m_tile = fmaxf(m_tile, s_wmax[w]);
        if (m_tile <= -1e29f) {  // whole tile masked -> contribute nothing
            __syncthreads();
            continue;
        }

        const float w = (s > -1e29f) ? expf(s - m_tile) : 0.0f;
        s_w[tid] = w;

        // ---- Tile sum (block reduce) ----
        const float wsum = warp_reduce_sum(w);
        if (lane == 0) s_wsum[warp_id] = wsum;
        __syncthreads();
        float l_tile = 0.0f;
#pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w) l_tile += s_wsum[w];

        // ---- Merge into the running softmax state ----
        const float m_new = fmaxf(m_g, m_tile);
        const float rg = expf(m_g - m_new);
        const float rt = expf(m_tile - m_new);
        l_g = l_g * rg + l_tile * rt;
        m_g = m_new;
#pragma unroll
        for (int i = 0; i < DPT; ++i) acc[i] *= rg;

        // ---- Phase B: P·V over the tile (accumulator distributed over d) ----
        const int tile_end = (tile_start + TILE < kv_len) ? TILE : (kv_len - tile_start);
#pragma unroll 1
        for (int t = 0; t < tile_end; ++t) {
            const float wt = s_w[t] * rt;
            if (wt == 0.0f) continue;
            const uint8_t* vrow = kv.vrow(tile_start + t);
#pragma unroll
            for (int i = 0; i < DPT; ++i) {
                const int d = tid + i * BLOCK_SIZE;
                if (d < HEAD_DIM) acc[i] += wt * kv.vd(vrow, d, tile_start + t);
            }
        }
        __syncthreads();
    }

    const float inv = (l_g > 0.0f) ? (1.0f / l_g) : 0.0f;
#pragma unroll
    for (int i = 0; i < DPT; ++i) {
        const int d = tid + i * BLOCK_SIZE;
        if (d < HEAD_DIM) o_row[d] = acc[i] * inv;
    }
}

// =========================================================================
// Launch helpers
// =========================================================================

#define TILED_PREFILL_LAUNCH(DEQ, PAGED, HD)                                                   \
    fused_attn_tiled_prefill_kernel<DEQ, PAGED, HD, 128><<<grid, 128, 0, stream>>>(            \
        Q, K, V, O, q_len, kv_len, num_heads, num_kv_heads, row_bytes, mask, causal, k_pages,  \
        v_pages, page_ids, page_size, k_scales, v_scales)

// Contiguous FP32 KV — no quantization, row_bytes = num_kv_heads*head_dim*4.
static inline void launch_fp32(const float* Q, const float* Kf, const float* Vf, float* O,
                               int q_len, int kv_len, int num_heads, int num_kv_heads,
                               int head_dim, const float* mask, bool causal,
                               cudaStream_t stream) {
    dim3 grid(q_len, num_heads);
    const uint8_t* K = reinterpret_cast<const uint8_t*>(Kf);
    const uint8_t* V = reinterpret_cast<const uint8_t*>(Vf);
    const size_t row_bytes = (size_t)num_kv_heads * head_dim * sizeof(float);
    void* const* k_pages = nullptr;
    void* const* v_pages = nullptr;
    const int32_t* page_ids = nullptr;
    const float* k_scales = nullptr;
    const float* v_scales = nullptr;
    const int page_size = 0;
    switch (head_dim) {
    case 16:  TILED_PREFILL_LAUNCH(DqFp32, false, 16);  break;
    case 32:  TILED_PREFILL_LAUNCH(DqFp32, false, 32);  break;
    case 64:  TILED_PREFILL_LAUNCH(DqFp32, false, 64);  break;
    case 72:  TILED_PREFILL_LAUNCH(DqFp32, false, 72);  break;
    case 96:  TILED_PREFILL_LAUNCH(DqFp32, false, 96);  break;
    case 128: TILED_PREFILL_LAUNCH(DqFp32, false, 128); break;
    case 256: TILED_PREFILL_LAUNCH(DqFp32, false, 256); break;
    case 512: TILED_PREFILL_LAUNCH(DqFp32, false, 512); break;
    default:  fprintf(stderr, "[ERROR] tiled_prefill_fp32: unsupported head_dim=%d\n", head_dim); break;
    }
}

// Contiguous quantized KV (F16 / Q8_0 / FP8).
template <typename DEQ>
static inline void launch_quant(const float* Q, const void* q_K, const void* q_V, float* O,
                                int q_len, int kv_len, int num_heads, int num_kv_heads,
                                int head_dim, size_t row_bytes, const float* mask, bool causal,
                                const float* k_scales, const float* v_scales,
                                cudaStream_t stream, const char* tag) {
    dim3 grid(q_len, num_heads);
    const uint8_t* K = static_cast<const uint8_t*>(q_K);
    const uint8_t* V = static_cast<const uint8_t*>(q_V);
    void* const* k_pages = nullptr;
    void* const* v_pages = nullptr;
    const int32_t* page_ids = nullptr;
    const int page_size = 0;
    if (head_dim < DEQ::kMinHeadDim) {
        fprintf(stderr, "[ERROR] tiled_prefill %s: head_dim=%d below minimum %d\n", tag, head_dim,
                DEQ::kMinHeadDim);
        return;
    }
    switch (head_dim) {
    case 16:  TILED_PREFILL_LAUNCH(DEQ, false, 16);  break;
    case 32:  TILED_PREFILL_LAUNCH(DEQ, false, 32);  break;
    case 64:  TILED_PREFILL_LAUNCH(DEQ, false, 64);  break;
    case 72:  TILED_PREFILL_LAUNCH(DEQ, false, 72);  break;
    case 96:  TILED_PREFILL_LAUNCH(DEQ, false, 96);  break;
    case 128: TILED_PREFILL_LAUNCH(DEQ, false, 128); break;
    case 256: TILED_PREFILL_LAUNCH(DEQ, false, 256); break;
    case 512: TILED_PREFILL_LAUNCH(DEQ, false, 512); break;
    default:  fprintf(stderr, "[ERROR] tiled_prefill %s: unsupported head_dim=%d\n", tag, head_dim); break;
    }
}

// Paged quantized KV (F16 / Q8_0 / FP8).
template <typename DEQ>
static inline void launch_paged(const float* Q, void* const* k_pages, void* const* v_pages,
                                const int32_t* page_ids, float* O, int q_len, int kv_len,
                                int num_heads, int num_kv_heads, int head_dim, int page_size,
                                size_t row_bytes, const float* mask, bool causal,
                                const float* k_scales, const float* v_scales, cudaStream_t stream,
                                const char* tag) {
    dim3 grid(q_len, num_heads);
    const uint8_t* K = nullptr;
    const uint8_t* V = nullptr;
    if (head_dim < DEQ::kMinHeadDim) {
        fprintf(stderr, "[ERROR] tiled_paged_prefill %s: head_dim=%d below minimum %d\n", tag,
                head_dim, DEQ::kMinHeadDim);
        return;
    }
    switch (head_dim) {
    case 16:  TILED_PREFILL_LAUNCH(DEQ, true, 16);  break;
    case 32:  TILED_PREFILL_LAUNCH(DEQ, true, 32);  break;
    case 64:  TILED_PREFILL_LAUNCH(DEQ, true, 64);  break;
    case 72:  TILED_PREFILL_LAUNCH(DEQ, true, 72);  break;
    case 96:  TILED_PREFILL_LAUNCH(DEQ, true, 96);  break;
    case 128: TILED_PREFILL_LAUNCH(DEQ, true, 128); break;
    case 256: TILED_PREFILL_LAUNCH(DEQ, true, 256); break;
    case 512: TILED_PREFILL_LAUNCH(DEQ, true, 512); break;
    default:  fprintf(stderr, "[ERROR] tiled_paged_prefill %s: unsupported head_dim=%d\n", tag, head_dim); break;
    }
}

// ---- Public launch entry points ----

void launch_fused_flash_attention_gqa_prefill_fp32(const float* Q, const float* K, const float* V,
                                                    float* O, int q_len, int kv_len, int num_heads,
                                                    int num_kv_heads, int head_dim,
                                                    const float* mask, bool causal,
                                                    cudaStream_t stream) {
    launch_fp32(Q, K, V, O, q_len, kv_len, num_heads, num_kv_heads, head_dim, mask, causal,
                stream);
}

#define DEFINE_QUANT_PREFILL(NAME, DEQ, TAG)                                                   \
    void NAME(const float* Q, const void* q_K, const void* q_V, float* O, int q_len, int kv_len, \
              int num_heads, int num_kv_heads, int head_dim, size_t row_bytes,                 \
              const float* mask, bool causal, const float* k_scales, const float* v_scales,    \
              cudaStream_t stream) {                                                           \
        launch_quant<DEQ>(Q, q_K, q_V, O, q_len, kv_len, num_heads, num_kv_heads, head_dim,    \
                          row_bytes, mask, causal, k_scales, v_scales, stream, TAG);           \
    }

#define DEFINE_PAGED_PREFILL(NAME, DEQ, TAG)                                                   \
    void NAME(const float* Q, void* const* k_pages, void* const* v_pages,                      \
              const int32_t* page_ids, float* O, int q_len, int kv_len, int num_heads,          \
              int num_kv_heads, int head_dim, int page_size, size_t row_bytes,                 \
              const float* mask, bool causal, const float* k_scales, const float* v_scales,    \
              cudaStream_t stream) {                                                           \
        launch_paged<DEQ>(Q, k_pages, v_pages, page_ids, O, q_len, kv_len, num_heads,          \
                          num_kv_heads, head_dim, page_size, row_bytes, mask, causal,          \
                          k_scales, v_scales, stream, TAG);                                    \
    }

DEFINE_QUANT_PREFILL(launch_fused_flash_attention_gqa_prefill_f16, DqF16, "f16")
DEFINE_QUANT_PREFILL(launch_fused_flash_attention_gqa_prefill_q8_0, DqQ8_0, "q8_0")
DEFINE_QUANT_PREFILL(launch_fused_flash_attention_gqa_prefill_fp8_e4m3, DqFp8<Fp8E4M3>, "fp8_e4m3")
DEFINE_QUANT_PREFILL(launch_fused_flash_attention_gqa_prefill_fp8_e5m2, DqFp8<Fp8E5M2>, "fp8_e5m2")

DEFINE_PAGED_PREFILL(launch_paged_flash_attention_gqa_prefill_f16, DqF16, "f16")
DEFINE_PAGED_PREFILL(launch_paged_flash_attention_gqa_prefill_q8_0, DqQ8_0, "q8_0")
DEFINE_PAGED_PREFILL(launch_paged_flash_attention_gqa_prefill_fp8_e4m3, DqFp8<Fp8E4M3>, "fp8_e4m3")
DEFINE_PAGED_PREFILL(launch_paged_flash_attention_gqa_prefill_fp8_e5m2, DqFp8<Fp8E5M2>, "fp8_e5m2")

#undef TILED_PREFILL_LAUNCH

}  // namespace cuda
}  // namespace forge
