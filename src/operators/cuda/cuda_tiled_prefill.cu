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
//
// ===== STATUS (2026-09-13) =====
// The *per-(qi,h)* kernel below (fused_attn_tiled_prefill_kernel) is the
// original Phase 2-B design: one thread-block per (query position, head),
// looping over the entire KV. It has no K/V reuse across query positions, so
// KV traffic is O(q_len * num_heads) and it dominated prefill latency on
// consumer GPUs (bench: 8.9 s for a 1702-tok prefill on RTX 4050).
//
// fused_attn_qtiled_v2_kernel (QUERY-TILED, the active FUSED prefill path)
// flips the loop order: one block covers BLOCK_M query rows of ONE head and
// stages each K/V tile into shared memory ONCE, reused by all BLOCK_M rows.
// KV traffic drops by ~num_heads*BLOCK_M (=~256x for 32 heads, BLOCK_M=8),
// and each warp owns HEAD_DIM/32 accumulator elements (not 128), so occupancy
// stays high. This is the llama.cpp fattn-tile design. See rewrite notes in
// the analysis. The KV-tiled kernel is retained for fallback / paged paths.
// =========================================================================

#include <cstdio>
#include <cstdlib>
#include <type_traits>

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

// Staging policy for the query-tiled v2 kernel: tiles are staged into shared
// memory once per block. Quantized KV formats (<=16-bit storage) stage as
// __half — lossless w.r.t. their own storage precision. FP32 KV MUST stage as
// float: routing FP32 through __half truncates the 24-bit mantissa to 11 bits
// (~5e-4 relative) and violates the 1e-6 tiled-vs-reference contract
// (test tiled-attn-formats, FP32 paged cases).
// kMaxQtiledHD = largest HEAD_DIM whose float-staged tiles still fit the 48KB
// static shared limit ((s_k+s_v) BLOCK_N*HD*4*2 + s_q BLOCK_M*HD*4).
struct StageHalf {
    using StageT = __half;
    static constexpr int kMaxQtiledHD = 512;
    __device__ __forceinline__ static float stage(float v) { return __float2half(v); }
    __device__ __forceinline__ static float load(StageT v) { return __half2float(v); }
};

struct StageFull {
    using StageT = float;
    static constexpr int kMaxQtiledHD = 128;
    __device__ __forceinline__ static float stage(float v) { return v; }
    __device__ __forceinline__ static float load(StageT v) { return v; }
};

struct DqFp32 : StageFull {
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

struct DqF16 : StageHalf {
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
struct DqQ8_0 : StageHalf {
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

// Q4_0: 32-element blocks, each = fp16 d (2B) + 16B nibble-packed (4b/element),
// values stored as unsigned 4-bit minus 8. Requires HEAD_DIM >= 32.
struct DqQ4_0 : StageHalf {
    static constexpr int kMinHeadDim = 32;
    template <int HEAD_DIM>
    __device__ __forceinline__ static size_t head_offset(int kv_h) {
        return (size_t)kv_h * (HEAD_DIM / 32) * 18;
    }
    template <int HEAD_DIM>
    __device__ __forceinline__ static float deq(const uint8_t* row, int d, int j, int kv_h,
                                                int num_kv_heads, const float* scales) {
        (void)j; (void)kv_h; (void)num_kv_heads; (void)scales;
        const int blk = d >> 5;
        const int i = d - (blk << 5);  // 0..31
        const uint8_t* bp = row + blk * 18;
        uint16_t sb;
        memcpy(&sb, bp, sizeof(uint16_t));
        const float s = __half2float(reinterpret_cast<const __half&>(sb));
        // Engine-canonical Q4_0 layout (cuda_common.h::q4_unpack): byte i packs
        // element i (low nibble) and element i+16 (high nibble).
        const uint8_t nib = (i < 16) ? (bp[2 + i] & 0x0F) : (bp[2 + (i - 16)] >> 4);
        return static_cast<float>(nib - 8) * s;
    }
};

// FP8 (E4M3 / E5M2): 1 byte/element, no block; per-(row, kv_head) fp32 scale.
template <typename FMT>
struct DqFp8 : StageHalf {
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

template <typename Dequant, bool PAGED, int HEAD_DIM, typename DequantV = Dequant>
struct KvView {
    const uint8_t* k_base;
    const uint8_t* v_base;
    void* const* k_pages;
    void* const* v_pages;
    const int32_t* page_ids;
    int page_size;
    // K and V can have different cache dtypes (-ctk/-ctv), which means different
    // block geometries: q8_0 is 32 elems / 34 B, q4_0 is 32 / 18, f16 is 8 / 16.
    // So the row stride AND the per-head byte offset differ per side. This used to
    // be one row_bytes and one head_off for both, which is why an asymmetric pair
    // had no path here. DequantV defaults to Dequant so every existing
    // instantiation -- the tuned f16 / q8_0 prefill -- compiles to exactly what it
    // did before.
    size_t row_bytes;   // K's row stride
    size_t v_row_bytes;
    size_t head_off;    // K's per-head offset
    size_t v_head_off;
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
                   (size_t)off * v_row_bytes;
        } else {
            base = v_base + (size_t)j * v_row_bytes;
        }
        return base + v_head_off;
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

// DequantV (defaulting to Dequant) is the V-side dequantiser and v_row_bytes the
// V-side row stride; both only matter when K and V dtypes differ. Defaulting
// keeps every existing instantiation byte-identical in behaviour.
template <typename Dequant, bool PAGED, int HEAD_DIM, int BLOCK_SIZE, typename DequantV = Dequant>
__global__ void fused_attn_tiled_prefill_kernel(
    const float* __restrict__ Q, const uint8_t* __restrict__ K, const uint8_t* __restrict__ V,
    float* __restrict__ O, int q_len, int kv_len, int num_heads, int num_kv_heads,
    size_t row_bytes, const float* __restrict__ mask, bool causal, void* const* k_pages,
    void* const* v_pages, const int32_t* page_ids, int page_size, const float* k_scales,
    const float* v_scales, size_t v_row_bytes = 0) {
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

    // Callers that predate asymmetric support pass 0 for v_row_bytes; treat that
    // as "same as K" so every existing instantiation is unchanged.
    const size_t v_row_stride = v_row_bytes ? v_row_bytes : row_bytes;
    const KvView<Dequant, PAGED, HEAD_DIM, DequantV> kv{
        K, V, k_pages, v_pages, page_ids, page_size, row_bytes, v_row_stride,
        Dequant::template head_offset<HEAD_DIM>(kv_h),
        DequantV::template head_offset<HEAD_DIM>(kv_h), kv_h, num_kv_heads, k_scales, v_scales};

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
// Query-TILED flash-attention PREFILL (v2) — the active high-perf FUSED path.
//
// Block = one head h x BLOCK_M query rows. Each K/V tile is staged into shared
// memory ONCE and reused by all BLOCK_M query rows, removing the O(q_len *
// num_heads) redundant KV re-reads that made the per-(qi,h) kernel the prefill
// bottleneck. One warp = one query row; the 32 lanes hold HEAD_DIM/32 elements
// of Q and of the FP32 running accumulator (occupancy stays high). K/V tiles
// are dequantized (via the Dequant policy) and staged as __half. Causal and
// explicit (non-causal) masks are both supported.
// =========================================================================

// DequantV (defaulting to Dequant) is the V-side dequantiser, and v_row_bytes
// the V-side row stride; both are only meaningful when K and V dtypes differ.
template <typename Dequant, bool PAGED, int HEAD_DIM, int BLOCK_M, int BLOCK_N,
          typename DequantV = Dequant>
__global__ void fused_attn_qtiled_v2_kernel(
    const float* __restrict__ Q, const uint8_t* __restrict__ K, const uint8_t* __restrict__ V,
    float* __restrict__ O, int q_len, int kv_len, int num_heads, int num_kv_heads,
    size_t row_bytes, const float* __restrict__ mask, bool causal,
    void* const* k_pages, void* const* v_pages, const int32_t* page_ids, int page_size,
    const float* k_scales, const float* v_scales, size_t v_row_bytes = 0) {
    const int h = blockIdx.y;
    if (h >= num_heads) return;
    const int kv_groups = num_heads / num_kv_heads;
    const int kv_h = h / kv_groups;

    const int m0 = blockIdx.x * BLOCK_M;
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int qi = m0 + warp_id;
    const bool active = (qi < q_len);

    // Absolute KV position of this query (prefix-cache aware).
    const int q_pos = kv_len - q_len + qi;
    const float attn_scale = 1.0f / sqrtf((float)HEAD_DIM);

    constexpr int DPT = HEAD_DIM / 32;   // dims per lane (HEAD_DIM multiple of 32)
    const int base_d = lane * DPT;

    __shared__ typename Dequant::StageT s_k[BLOCK_N * HEAD_DIM];
    // V may stage as a different width than K when the two dtypes differ
    // (e.g. FP32 K needs float tiles, V can be __half).
    __shared__ typename DequantV::StageT s_v[BLOCK_N * HEAD_DIM];
    __shared__ float s_q[BLOCK_M * HEAD_DIM];

    // Cooperative Q load for this block's query tile.
    for (int i = threadIdx.x; i < BLOCK_M * HEAD_DIM; i += blockDim.x) {
        const int qrow = i / HEAD_DIM;
        const int d = i % HEAD_DIM;
        float val = 0.0f;
        if (m0 + qrow < q_len) val = Q[(m0 + qrow) * num_heads * HEAD_DIM + h * HEAD_DIM + d];
        s_q[i] = val;
    }
    __syncthreads();

    float acc[DPT];
#pragma unroll
    for (int i = 0; i < DPT; ++i) acc[i] = 0.0f;

    float m_g = -1e30f;
    float l_g = 0.0f;

    // v_row_bytes == 0 means "same as K", so callers that predate asymmetric
    // support need no change.
    const size_t v_row_stride = v_row_bytes ? v_row_bytes : row_bytes;
    const KvView<Dequant, PAGED, HEAD_DIM, DequantV> kv{
        K, V, k_pages, v_pages, page_ids, page_size, row_bytes, v_row_stride,
        Dequant::template head_offset<HEAD_DIM>(kv_h),
        DequantV::template head_offset<HEAD_DIM>(kv_h), kv_h, num_kv_heads, k_scales, v_scales};

    for (int t0 = 0; t0 < kv_len; t0 += BLOCK_N) {
        const int tile_end = (t0 + BLOCK_N <= kv_len) ? BLOCK_N : (kv_len - t0);

        // Stage K/V tile into shared memory (dequant -> half) once for the block.
        for (int i = threadIdx.x; i < BLOCK_N * HEAD_DIM; i += blockDim.x) {
            const int jj = i / HEAD_DIM;
            const int d = i % HEAD_DIM;
            if (jj < tile_end) {
                const int j = t0 + jj;
                const uint8_t* krow = kv.krow(j);
                const uint8_t* vrow = kv.vrow(j);
                s_k[jj * HEAD_DIM + d] = Dequant::stage(Dequant::deq<HEAD_DIM>(krow, d, j, kv_h, num_kv_heads, k_scales));
                s_v[jj * HEAD_DIM + d] = DequantV::stage(DequantV::deq<HEAD_DIM>(vrow, d, j, kv_h, num_kv_heads, v_scales));
            }
        }
        __syncthreads();

        // Score for this lane's KV position.
        float s_self = -1e30f;
        if (active && lane < tile_end) {
            const int j = t0 + lane;
            bool skip = false;
            float mval = 0.0f;
            if (mask) {
                mval = mask[qi * kv_len + j];
                if (mval < -1e20f) skip = true;
            } else if (causal && j > q_pos) {
                skip = true;
            }
            if (!skip) {
                // Full dot product Q[qi] · K[j] over all HEAD_DIM dims (Q and K
                // both resident in shared). Each lane owns one KV position jj=lane.
                float dot = 0.0f;
                for (int d = 0; d < HEAD_DIM; ++d)
                    dot += s_q[warp_id * HEAD_DIM + d] * Dequant::load(s_k[lane * HEAD_DIM + d]);
                dot *= attn_scale;
                if (mask) dot += mval;
                s_self = dot;
            }
        }

        const float m_tile = warp_reduce_max(s_self);
        // Fully-masked tile: no contribution, but keep the SAME sync path (no
        // early continue) to avoid cross-warp __syncthreads divergence.
        const float w_self = (s_self > -1e29f && m_tile > -1e29f) ? expf(s_self - m_tile) : 0.0f;
        const float l_tile = warp_reduce_sum(w_self);

        const float m_new = fmaxf(m_g, m_tile);
        const float rg = expf(m_g - m_new);
        const float rt = (m_tile > -1e29f) ? expf(m_tile - m_new) : 0.0f;
        l_g = l_g * rg + l_tile * rt;
        m_g = m_new;
#pragma unroll
        for (int i = 0; i < DPT; ++i) acc[i] *= rg;

        // P·V: gather every weight via warp shuffle (no shared-weight collision).
        // Scale by rt = exp(m_tile - m_new) to fold this tile into the running
        // max m_g (online softmax). Omitting rt over-weights later tiles.
        for (int j = 0; j < tile_end; ++j) {
            const float wj = rt * __shfl_sync(0xffffffff, w_self, j);
            if (wj > 0.0f) {
                const typename DequantV::StageT* vrow = &s_v[j * HEAD_DIM];
#pragma unroll
                for (int i = 0; i < DPT; ++i) acc[i] += wj * DequantV::load(vrow[base_d + i]);
            }
        }

        __syncthreads();
    }

    if (active) {
        const float inv = (l_g > 0.0f) ? (1.0f / l_g) : 0.0f;
        float* o_row = O + qi * num_heads * HEAD_DIM + h * HEAD_DIM;
#pragma unroll
        for (int i = 0; i < DPT; ++i) o_row[base_d + i] = acc[i] * inv;
    }
}

// Per-case qtiled_v2 launch. enable_if keeps HD > DEQ::kMaxQtiledHD
// instantiations (e.g. FP32-staged HD=256/512, over the 48KB static shared
// limit) from compiling the kernel at all; the launcher's runtime guard never
// reaches those cases.
template <typename DEQ, bool PAGED, int HD, int BM, int BN, typename DEQV = DEQ,
          typename std::enable_if<(HD <= DEQ::kMaxQtiledHD), int>::type = 0>
static inline void qtiled_v2_case(const float* Q, const uint8_t* K, const uint8_t* V, float* O,
                                  int q_len, int kv_len, int num_heads, int num_kv_heads,
                                  size_t row_bytes, const float* mask, bool causal,
                                  void* const* k_pages, void* const* v_pages,
                                  const int32_t* page_ids, int page_size,
                                  const float* k_scales, const float* v_scales, dim3 grid,
                                  dim3 block, cudaStream_t stream, size_t v_row_bytes = 0) {
    fused_attn_qtiled_v2_kernel<DEQ, PAGED, HD, BM, BN, DEQV><<<grid, block, 0, stream>>>(
        Q, K, V, O, q_len, kv_len, num_heads, num_kv_heads, row_bytes, mask, causal, k_pages,
        v_pages, page_ids, page_size, k_scales, v_scales, v_row_bytes);
}

template <typename DEQ, bool PAGED, int HD, int BM, int BN,
          typename std::enable_if<(HD > DEQ::kMaxQtiledHD), int>::type = 0>
static inline void qtiled_v2_case(const float*, const uint8_t*, const uint8_t*, float*, int, int,
                                  int, int, size_t, const float*, bool, void* const*,
                                  void* const*, const int32_t*, int, const float*, const float*,
                                  dim3, dim3, cudaStream_t) {
    fprintf(stderr, "[ERROR] qtiled_v2: hd=%d exceeds staged shared budget (max %d)\n", HD,
            DEQ::kMaxQtiledHD);
}

#define QTILED_V2_CASE(DEQ, PAGED, HD, BM, BN)                                                 \
    qtiled_v2_case<DEQ, PAGED, HD, BM, BN>(                                                    \
        Q, K, V, O, q_len, kv_len, num_heads, num_kv_heads, row_bytes, mask, causal, k_pages,  \
        v_pages, page_ids, page_size, k_scales, v_scales, grid, block, stream)

// Forward declaration so launch_qtiled can fall back to the original kernel.
template <typename DEQ>
static inline void launch_quant(const float* Q, const void* q_K, const void* q_V, float* O,
                                int q_len, int kv_len, int num_heads, int num_kv_heads,
                                int head_dim, size_t row_bytes, const float* mask, bool causal,
                                const float* k_scales, const float* v_scales, cudaStream_t stream,
                                const char* tag);

// ---- Query-tiled (v2) contiguous launch (FUSED paths) ----
template <typename DEQ>
static inline void launch_qtiled(const float* Q, const void* q_K, const void* q_V, float* O,
                                 int q_len, int kv_len, int num_heads, int num_kv_heads,
                                 int head_dim, size_t row_bytes, const float* mask, bool causal,
                                 const float* k_scales, const float* v_scales, cudaStream_t stream,
                                 const char* tag) {
    // HEAD_DIM must be a multiple of 32 and fit the staged shared-memory budget;
    // otherwise fall back.
    if (head_dim % 32 != 0 || head_dim > DEQ::kMaxQtiledHD) {
        launch_quant<DEQ>(Q, q_K, q_V, O, q_len, kv_len, num_heads, num_kv_heads, head_dim,
                          row_bytes, mask, causal, k_scales, v_scales, stream, tag);
        return;
    }
    const uint8_t* K = reinterpret_cast<const uint8_t*>(q_K);
    const uint8_t* V = reinterpret_cast<const uint8_t*>(q_V);
    void* const* k_pages = nullptr;
    void* const* v_pages = nullptr;
    const int32_t* page_ids = nullptr;
    const int page_size = 0;
    int bm, bn;
    if (head_dim <= 128) {
        bm = 8;
        bn = 32;
    } else if (head_dim <= 256) {
        bm = 4;
        bn = 32;
    } else {
        bm = 2;
        bn = 16;
    }
    dim3 grid((q_len + bm - 1) / bm, num_heads);
    dim3 block(bm * 32);

    switch (head_dim) {
    case 32:  QTILED_V2_CASE(DEQ, false, 32,  8, 32); break;
    case 64:  QTILED_V2_CASE(DEQ, false, 64,  8, 32); break;
    case 96:  QTILED_V2_CASE(DEQ, false, 96,  8, 32); break;
    case 128: QTILED_V2_CASE(DEQ, false, 128, 8, 32); break;
    case 256: QTILED_V2_CASE(DEQ, false, 256, 4, 32); break;
    case 512: QTILED_V2_CASE(DEQ, false, 512, 2, 16); break;
    default:  launch_quant<DEQ>(Q, q_K, q_V, O, q_len, kv_len, num_heads, num_kv_heads, head_dim, row_bytes, mask, causal, k_scales, v_scales, stream, tag); break;
    }

    // Optional numeric self-check against the original (per-(qi,h)) kernel.
    // O now holds the new kernel's output; we run the old kernel into Oref and
    // compare. (This is purely a verification aid; it does not affect O.)
    if (getenv("FORGE_QTILED_CHECK")) {
        float* Oref = nullptr;
        cudaMalloc(&Oref, (size_t)q_len * num_heads * head_dim * sizeof(float));
        launch_quant<DEQ>(Q, q_K, q_V, Oref, q_len, kv_len, num_heads, num_kv_heads, head_dim,
                          row_bytes, mask, causal, k_scales, v_scales, stream, tag);
        float* h_new = (float*)malloc((size_t)q_len * num_heads * head_dim * sizeof(float));
        float* h_ref = (float*)malloc((size_t)q_len * num_heads * head_dim * sizeof(float));
        cudaMemcpy(h_new, O, (size_t)q_len * num_heads * head_dim * sizeof(float),
                  cudaMemcpyDeviceToHost);
        cudaMemcpy(h_ref, Oref, (size_t)q_len * num_heads * head_dim * sizeof(float),
                  cudaMemcpyDeviceToHost);
        float maxerr = 0.0f, maxref = 0.0f;
        const size_t n = (size_t)q_len * num_heads * head_dim;
        for (size_t i = 0; i < n; ++i) {
            maxerr = fmaxf(maxerr, fabsf(h_new[i] - h_ref[i]));
            maxref = fmaxf(maxref, fabsf(h_ref[i]));
        }
        fprintf(stderr,
                "[qtiled-check] %s hd=%d q=%d kv=%d max|new-ref|=%.3e max|ref|=%.3e rel=%.3e\n",
                tag, head_dim, q_len, kv_len, maxerr, maxref,
                maxref > 0.0f ? maxerr / maxref : 0.0f);
        free(h_new);
        free(h_ref);
        cudaFree(Oref);
    }
}

// Forward declaration (defined later, below launch_paged).
template <typename DEQ>
static inline void launch_paged(const float* Q, void* const* k_pages, void* const* v_pages,
                                const int32_t* page_ids, float* O, int q_len, int kv_len,
                                int num_heads, int num_kv_heads, int head_dim, int page_size,
                                size_t row_bytes, const float* mask, bool causal,
                                const float* k_scales, const float* v_scales,
                                cudaStream_t stream, const char* tag);

// =========================================================================
// Query-tiled (v2) PAGED launch (FUSED paths, paged KV).
// Mirrors launch_qtiled but resolves K/V through the page table: K/V base
// pointers are unused (KvView::paged reads via k_pages/v_pages/page_ids).
// This is the fix that makes the server's paged-KV prefill use the
// query-tiled v2 kernel instead of the per-(qi,h) tiled kernel.
// =========================================================================
template <typename DEQ>
static inline void launch_qtiled_paged(const float* Q, void* const* k_pages,
                                       void* const* v_pages, const int32_t* page_ids,
                                       float* O, int q_len, int kv_len, int num_heads,
                                       int num_kv_heads, int head_dim, int page_size,
                                       size_t row_bytes, const float* mask, bool causal,
                                       const float* k_scales, const float* v_scales,
                                       cudaStream_t stream, const char* tag) {
    // HEAD_DIM must be a multiple of 32 and fit the staged shared-memory budget
    // (float-staged FP32 KV caps at 128); otherwise fall back.
    if (head_dim % 32 != 0 || head_dim > DEQ::kMaxQtiledHD) {
        launch_paged<DEQ>(Q, k_pages, v_pages, page_ids, O, q_len, kv_len, num_heads,
                          num_kv_heads, head_dim, page_size, row_bytes, mask, causal,
                          k_scales, v_scales, stream, tag);
        return;
    }
    const uint8_t* K = nullptr;  // unused when PAGED; KvView resolves via page table
    const uint8_t* V = nullptr;
    int bm, bn;
    if (head_dim <= 128) {
        bm = 8;
        bn = 32;
    } else if (head_dim <= 256) {
        bm = 4;
        bn = 32;
    } else {
        bm = 2;
        bn = 16;
    }
    dim3 grid((q_len + bm - 1) / bm, num_heads);
    dim3 block(bm * 32);

    switch (head_dim) {
    case 32:  QTILED_V2_CASE(DEQ, true, 32,  8, 32); break;
    case 64:  QTILED_V2_CASE(DEQ, true, 64,  8, 32); break;
    case 96:  QTILED_V2_CASE(DEQ, true, 96,  8, 32); break;
    case 128: QTILED_V2_CASE(DEQ, true, 128, 8, 32); break;
    case 256: QTILED_V2_CASE(DEQ, true, 256, 4, 32); break;
    case 512: QTILED_V2_CASE(DEQ, true, 512, 2, 16); break;
    default:  launch_paged<DEQ>(Q, k_pages, v_pages, page_ids, O, q_len, kv_len, num_heads, num_kv_heads, head_dim, page_size, row_bytes, mask, causal, k_scales, v_scales, stream, tag); break;
    }

    // Optional numeric self-check against the original paged (per-(qi,h)) kernel.
    if (getenv("FORGE_QTILED_CHECK")) {
        float* Oref = nullptr;
        cudaMalloc(&Oref, (size_t)q_len * num_heads * head_dim * sizeof(float));
        launch_paged<DEQ>(Q, k_pages, v_pages, page_ids, Oref, q_len, kv_len, num_heads,
                          num_kv_heads, head_dim, page_size, row_bytes, mask, causal,
                          k_scales, v_scales, stream, tag);
        float* h_new = (float*)malloc((size_t)q_len * num_heads * head_dim * sizeof(float));
        float* h_ref = (float*)malloc((size_t)q_len * num_heads * head_dim * sizeof(float));
        cudaMemcpy(h_new, O, (size_t)q_len * num_heads * head_dim * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_ref, Oref, (size_t)q_len * num_heads * head_dim * sizeof(float), cudaMemcpyDeviceToHost);
        float maxerr = 0.0f, maxref = 0.0f;
        const size_t n = (size_t)q_len * num_heads * head_dim;
        for (size_t i = 0; i < n; ++i) {
            float d = h_new[i] - h_ref[i]; if (d < 0.0f) d = -d; if (d > maxerr) maxerr = d;
            float r = h_ref[i]; if (r < 0.0f) r = -r; if (r > maxref) maxref = r;
        }
        fprintf(stderr,
                "[qtiled-check-paged] %s hd=%d q=%d kv=%d max|new-ref|=%.3e max|ref|=%.3e rel=%.3e\n",
                tag, head_dim, q_len, kv_len, maxerr, maxref,
                maxref > 0.0f ? maxerr / maxref : 0.0f);
        free(h_new);
        free(h_ref);
        cudaFree(Oref);
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
        launch_qtiled_paged<DEQ>(Q, k_pages, v_pages, page_ids, O, q_len, kv_len, num_heads,   \
                                 num_kv_heads, head_dim, page_size, row_bytes, mask, causal,    \
                                 k_scales, v_scales, stream, TAG);                               \
    }

// FUSED (contiguous) prefill paths now route to the query-tiled v2 kernel.
#define DEFINE_QUANT_PREFILL_QTILED(NAME, DEQ, TAG)                                             \
    void NAME(const float* Q, const void* q_K, const void* q_V, float* O, int q_len, int kv_len, \
              int num_heads, int num_kv_heads, int head_dim, size_t row_bytes,                 \
              const float* mask, bool causal, const float* k_scales, const float* v_scales,    \
              cudaStream_t stream) {                                                              \
        launch_qtiled<DEQ>(Q, q_K, q_V, O, q_len, kv_len, num_heads, num_kv_heads, head_dim,   \
                           row_bytes, mask, causal, k_scales, v_scales, stream, TAG);            \
    }

DEFINE_QUANT_PREFILL_QTILED(launch_fused_flash_attention_gqa_prefill_q4_0, DqQ4_0, "q4_0")
DEFINE_QUANT_PREFILL_QTILED(launch_fused_flash_attention_gqa_prefill_f16, DqF16, "f16")
DEFINE_QUANT_PREFILL_QTILED(launch_fused_flash_attention_gqa_prefill_q8_0, DqQ8_0, "q8_0")
DEFINE_QUANT_PREFILL_QTILED(launch_fused_flash_attention_gqa_prefill_fp8_e4m3, DqFp8<Fp8E4M3>, "fp8_e4m3")
DEFINE_QUANT_PREFILL_QTILED(launch_fused_flash_attention_gqa_prefill_fp8_e5m2, DqFp8<Fp8E5M2>, "fp8_e5m2")

// Paged FP32: previously the only paged KV layout that materialized; the tiled
// kernel reads FP32 pages directly through the page table.
// ---- Paged MIXED prefill (asymmetric K/V) ----
//
// -ctk/-ctv can give K and V different dtypes, which means different block
// geometries and therefore different row strides AND different per-head byte
// offsets. Every prefill kernel above is monomorphic (one Dequant, one
// row_bytes), which is why asymmetric paged KV had no path at all and had to be
// refused outright rather than served.
//
// Same Dequant policy types as the monomorphic kernels, so K/V dequantisation is
// literally the same code; only the row/offset arithmetic differs per side.
template <typename DEQK, typename DEQV>
static inline void launch_qtiled_paged_mixed(const float* Q, void* const* k_pages,
                                             void* const* v_pages, const int32_t* page_ids,
                                             float* O, int q_len, int kv_len, int num_heads,
                                             int num_kv_heads, int head_dim, int page_size,
                                             size_t k_row_bytes, size_t v_row_bytes,
                                             const float* mask, bool causal, const float* k_scales,
                                             const float* v_scales, cudaStream_t stream,
                                             const char* tag) {
    if (head_dim % 32 != 0 || head_dim > DEQK::kMaxQtiledHD ||
        head_dim > DEQV::kMaxQtiledHD) {
        fprintf(stderr,
                "[ERROR] tiled_paged_mixed_prefill %s: unsupported head_dim=%d (need %%32==0 and "
                "<=%d)\n",
                tag, head_dim, DEQK::kMaxQtiledHD);
        return;
    }
    const uint8_t* K = nullptr;  // unused when PAGED; KvView resolves via the page table
    const uint8_t* V = nullptr;
    int bm, bn;
    if (head_dim <= 128) {
        bm = 8;
        bn = 32;
    } else if (head_dim <= 256) {
        bm = 4;
        bn = 32;
    } else {
        bm = 2;
        bn = 16;
    }
    dim3 grid((q_len + bm - 1) / bm, num_heads);
    dim3 block(bm * 32);

#define PAGED_MIXED_CASE(HD, BM, BN)                                                          \
    qtiled_v2_case<DEQK, true, HD, BM, BN, DEQV>(                                            \
        Q, K, V, O, q_len, kv_len, num_heads, num_kv_heads, k_row_bytes, mask, causal,         \
        k_pages, v_pages, page_ids, page_size, k_scales, v_scales, grid, block, stream,         \
        v_row_bytes)
    switch (head_dim) {
    case 32:  PAGED_MIXED_CASE(32,  8, 32); break;
    case 64:  PAGED_MIXED_CASE(64,  8, 32); break;
    case 96:  PAGED_MIXED_CASE(96,  8, 32); break;
    case 128: PAGED_MIXED_CASE(128, 8, 32); break;
    case 256: PAGED_MIXED_CASE(256, 4, 32); break;
    case 512: PAGED_MIXED_CASE(512, 2, 16); break;
    default: break;
    }
#undef PAGED_MIXED_CASE
}

#define PAGED_MIXED_ENTRY(NAME, K, V)                                                           \
    void NAME(const float* Q, void* const* k_pages, void* const* v_pages,                      \
              const int32_t* page_ids, float* O, int q_len, int kv_len, int num_heads,          \
              int num_kv_heads, int head_dim, int page_size, size_t k_row_bytes,                \
              size_t v_row_bytes, const float* mask, bool causal, const float* k_scales,        \
              const float* v_scales, cudaStream_t stream) {                                     \
        launch_qtiled_paged_mixed<K, V>(Q, k_pages, v_pages, page_ids, O, q_len, kv_len,        \
                                        num_heads, num_kv_heads, head_dim, page_size,           \
                                        k_row_bytes, v_row_bytes, mask, causal, k_scales,        \
                                        v_scales, stream, "mixed");                             \
    }

PAGED_MIXED_ENTRY(launch_paged_flash_attention_gqa_prefill_q8_0_q4_0, DqQ8_0, DqQ4_0)
PAGED_MIXED_ENTRY(launch_paged_flash_attention_gqa_prefill_q4_0_q8_0, DqQ4_0, DqQ8_0)
PAGED_MIXED_ENTRY(launch_paged_flash_attention_gqa_prefill_q8_0_f16, DqQ8_0, DqF16)
PAGED_MIXED_ENTRY(launch_paged_flash_attention_gqa_prefill_f16_q8_0, DqF16, DqQ8_0)
PAGED_MIXED_ENTRY(launch_paged_flash_attention_gqa_prefill_q4_0_f16, DqQ4_0, DqF16)
PAGED_MIXED_ENTRY(launch_paged_flash_attention_gqa_prefill_f16_q4_0, DqF16, DqQ4_0)
#undef PAGED_MIXED_ENTRY

DEFINE_PAGED_PREFILL(launch_paged_flash_attention_gqa_prefill_fp32, DqFp32, "fp32")
DEFINE_PAGED_PREFILL(launch_paged_flash_attention_gqa_prefill_q4_0, DqQ4_0, "q4_0")
DEFINE_PAGED_PREFILL(launch_paged_flash_attention_gqa_prefill_f16, DqF16, "f16")
DEFINE_PAGED_PREFILL(launch_paged_flash_attention_gqa_prefill_q8_0, DqQ8_0, "q8_0")
DEFINE_PAGED_PREFILL(launch_paged_flash_attention_gqa_prefill_fp8_e4m3, DqFp8<Fp8E4M3>, "fp8_e4m3")
DEFINE_PAGED_PREFILL(launch_paged_flash_attention_gqa_prefill_fp8_e5m2, DqFp8<Fp8E5M2>, "fp8_e5m2")

#undef TILED_PREFILL_LAUNCH

}  // namespace cuda
}  // namespace forge
