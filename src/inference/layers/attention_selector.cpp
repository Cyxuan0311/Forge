#include "forge/attention_selector.h"

#include <cstdio>

namespace forge {

AttentionPath choose_attention_path(const AttentionProblem& p) {
    // ---- CPU paths ----
    if (p.device == DeviceType::CPU) {
        bool is_gqa = (p.num_kv_heads > 0 && p.num_kv_heads < p.num_heads);
        return is_gqa ? AttentionPath::CPU_GQA : AttentionPath::CPU_MHA;
    }

    // ---- CUDA paths ----
    bool is_gqa = (p.num_kv_heads > 0 && p.num_kv_heads < p.num_heads);

    // Non-GQA CUDA: use generic ops path
    if (!is_gqa) {
        return AttentionPath::CUDA_GENERIC_MHA;
    }

    // GQA decode (seq_len == 1): prefer fused quantized KV if available
    if (p.seq_len == 1) {
        // Paged KV cache (Phase 4): traverse the page table directly.
        // Only quantized paged KV has a dedicated kernel; FP32 paged falls
        // through to the materialize-then-attend FP32 decode path.
        if (p.paged && p.has_quantized_kv && p.kv_type_k == p.kv_type_v) {
            switch (p.kv_type_k) {
            case KVCacheDType::Q4_0:
                return AttentionPath::CUDA_PAGED_Q4_0_DECODE;
            case KVCacheDType::F16:
                return AttentionPath::CUDA_PAGED_F16_DECODE;
            case KVCacheDType::Q8_0:
                return AttentionPath::CUDA_PAGED_Q8_0_DECODE;
            case KVCacheDType::FP8_E4M3:
                return AttentionPath::CUDA_PAGED_FP8_E4M3_DECODE;
            case KVCacheDType::FP8_E5M2:
                return AttentionPath::CUDA_PAGED_FP8_E5M2_DECODE;
            default:
                break;  // Unsupported symmetric type → FP32 fallback
            }
        }
        if (p.has_quantized_kv && p.kv_type_k == p.kv_type_v) {
            switch (p.kv_type_k) {
            case KVCacheDType::Q4_0:
                return AttentionPath::CUDA_FUSED_Q4_0_DECODE;
            case KVCacheDType::F16:
                return AttentionPath::CUDA_FUSED_F16_DECODE;
            case KVCacheDType::Q8_0:
                return AttentionPath::CUDA_FUSED_Q8_0_DECODE;
            case KVCacheDType::FP8_E4M3:
                return AttentionPath::CUDA_FUSED_FP8_E4M3_DECODE;
            case KVCacheDType::FP8_E5M2:
                return AttentionPath::CUDA_FUSED_FP8_E5M2_DECODE;
            default:
                break;  // Unsupported symmetric type → FP32 fallback
            }
        }
        // FP32 or asymmetric KV → standard FP32 decode kernel
        return AttentionPath::CUDA_FP32_DECODE;
    }

    // GQA prefill (seq_len > 1)
    return AttentionPath::CUDA_FP32_PREFILL;
}

const char* attention_path_name(AttentionPath path) {
    switch (path) {
    case AttentionPath::CPU_MHA:
        return "CPU_MHA";
    case AttentionPath::CPU_GQA:
        return "CPU_GQA";
    case AttentionPath::CUDA_FP32_PREFILL:
        return "CUDA_FP32_PREFILL";
    case AttentionPath::CUDA_FP32_DECODE:
        return "CUDA_FP32_DECODE";
    case AttentionPath::CUDA_GENERIC_MHA:
        return "CUDA_GENERIC_MHA";
    case AttentionPath::CUDA_FUSED_Q4_0_DECODE:
        return "CUDA_FUSED_Q4_0_DECODE";
    case AttentionPath::CUDA_FUSED_F16_DECODE:
        return "CUDA_FUSED_F16_DECODE";
    case AttentionPath::CUDA_FUSED_Q8_0_DECODE:
        return "CUDA_FUSED_Q8_0_DECODE";
    case AttentionPath::CUDA_FUSED_FP8_E4M3_DECODE:
        return "CUDA_FUSED_FP8_E4M3_DECODE";
    case AttentionPath::CUDA_FUSED_FP8_E5M2_DECODE:
        return "CUDA_FUSED_FP8_E5M2_DECODE";
    case AttentionPath::CUDA_PAGED_Q4_0_DECODE:
        return "CUDA_PAGED_Q4_0_DECODE";
    case AttentionPath::CUDA_PAGED_F16_DECODE:
        return "CUDA_PAGED_F16_DECODE";
    case AttentionPath::CUDA_PAGED_Q8_0_DECODE:
        return "CUDA_PAGED_Q8_0_DECODE";
    case AttentionPath::CUDA_PAGED_FP8_E4M3_DECODE:
        return "CUDA_PAGED_FP8_E4M3_DECODE";
    case AttentionPath::CUDA_PAGED_FP8_E5M2_DECODE:
        return "CUDA_PAGED_FP8_E5M2_DECODE";
    case AttentionPath::UNSUPPORTED:
        return "UNSUPPORTED";
    default:
        return "UNKNOWN";
    }
}

}  // namespace forge
