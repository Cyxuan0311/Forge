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
    CPU_MHA,          // CPU multi-head attention (no GQA)
    CPU_GQA,          // CPU grouped-query attention

    // CUDA FP32 paths (K/V in FP32, no fused dequant)
    CUDA_FP32_PREFILL,    // CUDA GQA prefill (q_len > 1)
    CUDA_FP32_DECODE,     // CUDA GQA decode  (q_len == 1, FP32 KV)
    CUDA_GENERIC_MHA,     // CUDA non-GQA (uses generic ops)

    // CUDA fused paths (K/V in quantized format, dequantized on-the-fly)
    CUDA_FUSED_Q4_0_DECODE,  // CUDA GQA decode with Q4_0 KV
    CUDA_FUSED_F16_DECODE,   // CUDA GQA decode with F16 KV
    CUDA_FUSED_Q8_0_DECODE,  // CUDA GQA decode with Q8_0 KV

    // Fallback
    UNSUPPORTED,
};

// ---- Problem descriptor ----

struct AttentionProblem {
    DeviceType device = DeviceType::CPU;
    int seq_len = 1;           // q_len (1 for decode, >1 for prefill)
    int num_heads = 0;
    int num_kv_heads = 0;
    int head_dim = 0;

    // KV cache state (relevant for CUDA decode path selection)
    bool has_quantized_kv = false;        // d_q_K && d_q_V available
    KVCacheDType kv_type_k = KVCacheDType::FP32;
    KVCacheDType kv_type_v = KVCacheDType::FP32;
};

// ---- Selector function ----

AttentionPath choose_attention_path(const AttentionProblem& problem);

// ---- Human-readable path name (for logging/debugging) ----

const char* attention_path_name(AttentionPath path);

}  // namespace forge
