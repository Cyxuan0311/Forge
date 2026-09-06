// =========================================================================
// attention_selector.h — CUDA Attention Path Selection
//
// Centralizes the dispatch logic for choosing the optimal attention kernel
// based on problem parameters (device, GQA, KV dtype, seq_len, head_dim).
//
// Design follows llama.cpp's fattn.cu kernel selection pattern:
//   - Encode problem constraints into AttentionProblem
//   - choose_attention_path() returns the optimal path
//   - The caller dispatches to the corresponding kernel launch function
// =========================================================================

#pragma once

#include "forge/kv_cache.h"
#include "forge/types.h"

namespace forge {

// ---- Attention path enumeration ----

enum class AttentionPath {
    // CPU paths
    CPU_MHA,  // CPU multi-head attention (no GQA)
    CPU_GQA,  // CPU grouped-query attention

    // CUDA FP32 paths (K/V in FP32, no fused dequant)
    CUDA_FP32_PREFILL,  // CUDA GQA prefill (q_len > 1)
    CUDA_FP32_DECODE,   // CUDA GQA decode  (q_len == 1, FP32 KV)
    CUDA_GENERIC_MHA,   // CUDA non-GQA (uses generic ops)

    // CUDA native-FP16 paths (6-A-7): Q/K/V all FP16 contiguous tensors,
    // FP32 QK^T/softmax/PV, FP16 output. Distinct from CUDA_FUSED_F16_*
    // (those are Q-FP32 + KV-F16 with on-the-fly dequant).
    CUDA_F16_PREFILL,  // CUDA GQA prefill, all-FP16 QKV (q_len > 1)
    CUDA_F16_DECODE,   // CUDA GQA decode, all-FP16 QKV  (q_len == 1)

    // CUDA fused paths (K/V in quantized format, dequantized on-the-fly)
    CUDA_FUSED_Q4_0_DECODE,      // CUDA GQA decode with Q4_0 KV
    CUDA_FUSED_F16_DECODE,       // CUDA GQA decode with F16 KV
    CUDA_FUSED_Q8_0_DECODE,      // CUDA GQA decode with Q8_0 KV
    CUDA_FUSED_FP8_E4M3_DECODE,  // CUDA GQA decode with FP8 E4M3 KV
    CUDA_FUSED_FP8_E5M2_DECODE,  // CUDA GQA decode with FP8 E5M2 KV

    // CUDA fused quantized prefill (Phase 2-B): KV stays in its quantized form
    // and is dequantized on the fly inside the kernel. Long-context win = no
    // FP32 materialization of the whole KV cache during prefill.
    // Q4_K has no fused policy: its 256-element super-blocks do not align with
    // per-head boundaries for HEAD_DIM < 256, and its decode path materializes
    // too, so it keeps the FP32 materialize route.
    CUDA_FUSED_Q4_0_PREFILL,      // CUDA GQA prefill with Q4_0 KV
    CUDA_FUSED_F16_PREFILL,       // CUDA GQA prefill with F16 KV
    CUDA_FUSED_Q8_0_PREFILL,      // CUDA GQA prefill with Q8_0 KV
    CUDA_FUSED_FP8_E4M3_PREFILL,  // CUDA GQA prefill with FP8 E4M3 KV
    CUDA_FUSED_FP8_E5M2_PREFILL,  // CUDA GQA prefill with FP8 E5M2 KV

    // CUDA paged paths (Phase 4): KV held in fixed-size pages; the decode kernel
    // traverses the per-sequence page table directly. No materialization.
    // FP32 paged KV falls back to CUDA_FP32_DECODE (materialize then attend).
    CUDA_PAGED_Q4_0_DECODE,      // CUDA paged GQA decode with Q4_0 KV
    CUDA_PAGED_F16_DECODE,       // CUDA paged GQA decode with F16 KV
    CUDA_PAGED_Q8_0_DECODE,      // CUDA paged GQA decode with Q8_0 KV
    CUDA_PAGED_FP8_E4M3_DECODE,  // CUDA paged GQA decode with FP8 E4M3 KV
    CUDA_PAGED_FP8_E5M2_DECODE,  // CUDA paged GQA decode with FP8 E5M2 KV

    // CUDA paged quantized prefill (Phase 2-B). FP32 paged KV now has a direct
    // paged policy as well, so nothing paged needs to materialize anymore.
    CUDA_PAGED_FP32_PREFILL,      // CUDA paged GQA prefill with FP32 KV
    CUDA_PAGED_Q4_0_PREFILL,      // CUDA paged GQA prefill with Q4_0 KV
    CUDA_PAGED_F16_PREFILL,       // CUDA paged GQA prefill with F16 KV
    CUDA_PAGED_Q8_0_PREFILL,      // CUDA paged GQA prefill with Q8_0 KV
    CUDA_PAGED_FP8_E4M3_PREFILL,  // CUDA paged GQA prefill with FP8 E4M3 KV
    CUDA_PAGED_FP8_E5M2_PREFILL,  // CUDA paged GQA prefill with FP8 E5M2 KV

    // Fallback
    UNSUPPORTED,
};

// ---- Attention backend enumeration ----
//
// Selects the *implementation style* for the chosen path. This is orthogonal to
// AttentionPath (which selects the dtype/dispatch family). The pipeline intent:
//   - Scalar     : thread-centric kernel (Q re-read from global per KV position)
//   - Vectorized : Q loaded once into shared memory (Q tile) + float4 K/V loads
//   - TensorCore : reserved for a future MMA-tiled backend
// Phase 2-C introduces Vectorized for the CUDA FP32 prefill path.

enum class AttentionBackend {
    Scalar,      // thread-centric kernel (Q re-read from global per KV position)
    Vectorized,  // Q tile in shared mem + float4 K/V loads
    TensorCore,  // reserved for a future MMA-tiled backend
};

// ---- Problem descriptor ----

struct AttentionProblem {
    DeviceType device = DeviceType::CPU;
    int seq_len = 1;  // q_len (1 for decode, >1 for prefill)
    int num_heads = 0;
    int num_kv_heads = 0;
    int head_dim = 0;

    // KV cache state (relevant for CUDA decode path selection)
    bool has_quantized_kv = false;  // d_q_K && d_q_V available
    KVCacheDType kv_type_k = KVCacheDType::FP32;
    KVCacheDType kv_type_v = KVCacheDType::FP32;

    // Paged KV cache (Phase 4). When true and device==CUDA, the decode path
    // resolves K/V rows through the per-sequence page table instead of reading
    // a contiguous tensor. seq_id selects which sequence's page table to use.
    bool paged = false;
    int seq_id = -1;

    // 6-A-7: Q dtype drives native-FP16 path selection. FP16 Q (+FP16 K/V,
    // enforced by the executor/dispatch layer) selects CUDA_F16_*; FP32 keeps
    // the existing paths. Pure input to choose_attention_path (testable).
    DataType q_dtype = DataType::FP32;
};

// ---- Selector function ----

AttentionPath choose_attention_path(const AttentionProblem& problem);

// ---- Human-readable path name (for logging/debugging) ----

const char* attention_path_name(AttentionPath path);

// ---- Human-readable backend name (for logging/debugging) ----

const char* attention_backend_name(AttentionBackend backend);

}  // namespace forge
