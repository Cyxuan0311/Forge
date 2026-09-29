#include "forge/attention_selector.h"

#include <cstdio>

#include "forge/kv_cache_dtype.h"

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

    // 6-A-7: native-FP16 QKV (contiguous FP16 tensors, no paged/quantized KV).
    // Locked rule: q FP16 selects the F16 path by seq_len; q FP32 keeps every
    // existing branch below untouched (no dtype-if inside CUDA_FP32_* cases).
    if (p.q_dtype == DataType::FP16) {
        return (p.seq_len == 1) ? AttentionPath::CUDA_F16_DECODE
                                : AttentionPath::CUDA_F16_PREFILL;
    }

    // 6-B-6: native BF16-Q decode (decode-first). q BF16 + seq_len == 1 selects
    // the BF16 path; K/V legitimacy sinks to the executor and op_kernels
    // layers, mirroring the F16 rule above.
    // 6-B-7.0: BF16 prefill unlocked (all-BF16 QKV vectorized kernel) so the
    // E2E closed loop can run prefill natively. K/V legitimacy still sinks
    // downstream (FP32 or BF16 accepted in decode; prefill requires all-BF16).
    if (p.q_dtype == DataType::BF16) {
        return (p.seq_len == 1) ? AttentionPath::CUDA_BF16_DECODE
                                : AttentionPath::CUDA_BF16_PREFILL;
    }

    // GQA decode (seq_len == 1): prefer fused quantized KV if available
    if (p.seq_len == 1) {
        // Paged KV cache (Phase 4): traverse the page table directly.
        // Quantized paged KV has a dedicated decode kernel. FP32 paged has no
        // dedicated decode kernel AND the CUDA paged storage cannot materialize
        // rows (read_key* return null by design) — so route it to the paged
        // prefill kernel, which handles seq_len==1 + causal correctly (the
        // single query attends to all kv_len rows). Previously FP32 paged fell
        // through to CUDA_FP32_DECODE, which materializes from the contiguous
        // cache and crashed (null K/V) with a paged CUDA storage.
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
            case KVCacheDType::FP32:
                return AttentionPath::CUDA_PAGED_FP32_PREFILL;
            default:
                break;  // Unsupported symmetric type → FP32 fallback
            }
        }
        // Asymmetric K/V (-ctk/-ctv). The mixed kernel is templated on the pair,
        // so it handles the symmetric cases too -- but the symmetric ones keep
        // their existing dedicated paths so a refactor here cannot perturb the
        // well-tuned f16 / q8_0 / q4_0 decode throughput.
        if (p.has_quantized_kv && p.kv_type_k != p.kv_type_v) {
            const auto fmt = [](KVCacheDType d) {
                switch (d) {
                case KVCacheDType::Q8_0: return 0;
                case KVCacheDType::Q4_0: return 1;
                case KVCacheDType::F16:  return 2;
                default: return -1;
                }
            };
            // The kernel covers head_dim 64/96/128 and the {q8_0,q4_0,f16}
            // formats; anything else must fall through to materialisation.
            const bool hd_ok = p.head_dim == 64 || p.head_dim == 96 || p.head_dim == 128;
            if (hd_ok && fmt(p.kv_type_k) >= 0 && fmt(p.kv_type_v) >= 0) {
                return AttentionPath::CUDA_FUSED_MIXED_DECODE;
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
        // FP32, or an ASYMMETRIC K/V pair.
        //
        // Every fused quantized kernel is monomorphic: it takes one q_row_size
        // and hard-codes one block layout for both K and V, so a mixed pair has
        // no fused path and materializes the filled rows to FP32 instead. That
        // is correct -- dequantize_layer() only re-materializes rows
        // [dequantized_filled, filled) -- but it costs memory traffic and it
        // measured noticeably variable (21.2-26.1 tok/s across runs where the
        // symmetric q8_0/q8_0 path held 22.7-22.9), so it must not read as a
        // silent no-op. Log it once per process; the mixed-dtype fused kernels
        // are the follow-up that removes this branch.
        if (p.has_quantized_kv && p.kv_type_k != p.kv_type_v) {
            static bool warned_mixed = false;
            if (!warned_mixed) {
                warned_mixed = true;
                std::fprintf(stderr,
                             "[Forge WARN] asymmetric KV cache (K=%s, V=%s) with head_dim=%d: no "
                             "fused mixed-dtype kernel for this combination, falling back to FP32 "
                             "materialization. Output is correct; throughput is not optimal.\n",
                             kv_cache_dtype_name(p.kv_type_k), kv_cache_dtype_name(p.kv_type_v),
                             p.head_dim);
            }
        }
        return AttentionPath::CUDA_FP32_DECODE;
    }

    // GQA prefill (seq_len > 1): prefer the fused quantized prefill kernel when
    // the KV cache holds a symmetric quantized type we have a policy for.
    // Paged KV resolves rows through the page table (no materialization).
    if (p.has_quantized_kv && p.kv_type_k == p.kv_type_v) {
        if (p.paged) {
            switch (p.kv_type_k) {
            case KVCacheDType::FP32:
                return AttentionPath::CUDA_PAGED_FP32_PREFILL;
            case KVCacheDType::Q4_0:
                return AttentionPath::CUDA_PAGED_Q4_0_PREFILL;
            case KVCacheDType::F16:
                return AttentionPath::CUDA_PAGED_F16_PREFILL;
            case KVCacheDType::Q8_0:
                return AttentionPath::CUDA_PAGED_Q8_0_PREFILL;
            case KVCacheDType::FP8_E4M3:
                return AttentionPath::CUDA_PAGED_FP8_E4M3_PREFILL;
            case KVCacheDType::FP8_E5M2:
                return AttentionPath::CUDA_PAGED_FP8_E5M2_PREFILL;
            default:
                break;  // Q4_K → FP32 materialize prefill
            }
        } else {
            switch (p.kv_type_k) {
            case KVCacheDType::Q4_0:
                return AttentionPath::CUDA_FUSED_Q4_0_PREFILL;
            case KVCacheDType::F16:
                return AttentionPath::CUDA_FUSED_F16_PREFILL;
            case KVCacheDType::Q8_0:
                return AttentionPath::CUDA_FUSED_Q8_0_PREFILL;
            case KVCacheDType::FP8_E4M3:
                return AttentionPath::CUDA_FUSED_FP8_E4M3_PREFILL;
            case KVCacheDType::FP8_E5M2:
                return AttentionPath::CUDA_FUSED_FP8_E5M2_PREFILL;
            default:
                break;  // FP32/Q4_K → FP32 materialize prefill
            }
        }
    }
    return AttentionPath::CUDA_FP32_PREFILL;
}

const char* attention_backend_name(AttentionBackend backend) {
    switch (backend) {
    case AttentionBackend::Scalar:
        return "Scalar";
    case AttentionBackend::Vectorized:
        return "Vectorized";
    case AttentionBackend::TensorCore:
        return "TensorCore";
    default:
        return "UNKNOWN";
    }
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
    case AttentionPath::CUDA_F16_PREFILL:
        return "CUDA_F16_PREFILL";
    case AttentionPath::CUDA_F16_DECODE:
        return "CUDA_F16_DECODE";
    case AttentionPath::CUDA_BF16_DECODE:
        return "CUDA_BF16_DECODE";
    case AttentionPath::CUDA_BF16_PREFILL:
        return "CUDA_BF16_PREFILL";
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
    case AttentionPath::CUDA_FUSED_Q4_0_PREFILL:
        return "CUDA_FUSED_Q4_0_PREFILL";
    case AttentionPath::CUDA_FUSED_F16_PREFILL:
        return "CUDA_FUSED_F16_PREFILL";
    case AttentionPath::CUDA_FUSED_Q8_0_PREFILL:
        return "CUDA_FUSED_Q8_0_PREFILL";
    case AttentionPath::CUDA_FUSED_FP8_E4M3_PREFILL:
        return "CUDA_FUSED_FP8_E4M3_PREFILL";
    case AttentionPath::CUDA_FUSED_FP8_E5M2_PREFILL:
        return "CUDA_FUSED_FP8_E5M2_PREFILL";
    case AttentionPath::CUDA_PAGED_FP32_PREFILL:
        return "CUDA_PAGED_FP32_PREFILL";
    case AttentionPath::CUDA_PAGED_Q4_0_PREFILL:
        return "CUDA_PAGED_Q4_0_PREFILL";
    case AttentionPath::CUDA_PAGED_F16_PREFILL:
        return "CUDA_PAGED_F16_PREFILL";
    case AttentionPath::CUDA_PAGED_Q8_0_PREFILL:
        return "CUDA_PAGED_Q8_0_PREFILL";
    case AttentionPath::CUDA_PAGED_FP8_E4M3_PREFILL:
        return "CUDA_PAGED_FP8_E4M3_PREFILL";
    case AttentionPath::CUDA_PAGED_FP8_E5M2_PREFILL:
        return "CUDA_PAGED_FP8_E5M2_PREFILL";
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
