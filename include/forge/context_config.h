#pragma once
#include <string>

#include "kv_cache.h"

namespace forge {

// =========================================================================
// KV_CACHE_PRECISION
// -------------------------------------------------------------------------
// High-level KV-cache precision *intent* profiles. A single string selects a
// concrete KV-cache dtype so users don't have to remember the dtype names or
// the accuracy/throughput trade-off. When set to a non-AUTO value this takes
// precedence over `kv_cache_dtype`/`kv_cache_type_k`/`kv_cache_type_v`.
//
//   AUTO            -> honor kv_cache_dtype verbatim (backward compatible)
//   HIGH_ACCURACY   -> "q8_0"      (4x vs fp32)   cosine >= 0.999 when healthy
//   HIGH_THROUGHPUT -> "fp8_e4m3"  (8x vs fp32)   cosine ~= 0.988, needs sign-off
//
// Measured on MiMo-7B / RTX 4050, greedy 32-token decode vs FP32 baseline
// (see benchmarks/verify_kv_cache_accuracy.py and the test report):
//   dtype        mean cosine   top-1 match
//   f16          0.9994        90.6%
//   q8_0         0.999341      90.6%   <-- FIXED 2026-08-31 (was 0.9285); see below
//   fp8_e4m3     0.9885        87.5%
//   fp8_e5m2     0.9771        75.0%
//
// 2026-08-31 Q8_0 fix (TODO #5): the per-block fp16 scale was stored via
// memcpy/two byte-stores into a uint8_t* and got clobbered by the subsequent
// aliased int8_t qs writes (low byte zeroed) on some toolchains/drivers,
// yielding ~8%-off scales and cosine 0.9285. The scale is now stored as a
// single 16-bit store in both quantize paths (contiguous + paged), matching
// how dequantize/fused-attention read it back; see
// src/operators/cuda/cuda_quant.cu quantize_q8_0_matrix_kernel.
// Re-measured mean cosine = 0.999341 (acceptance bar >= 0.999) => PASS.
// HIGH_ACCURACY is therefore now the approved production default for
// high-accuracy deployments (select via kv_cache_precision="high_accuracy").
// =========================================================================
enum class KVCachePrecision { AUTO = 0, HIGH_ACCURACY = 1, HIGH_THROUGHPUT = 2 };

// Returns the concrete KV-cache dtype string for a profile, or nullptr for
// AUTO (caller keeps the explicitly-passed kv_cache_dtype).
inline const char* kv_cache_precision_to_dtype(KVCachePrecision p) {
    switch (p) {
    case KVCachePrecision::HIGH_ACCURACY:
        return "q8_0";
    case KVCachePrecision::HIGH_THROUGHPUT:
        return "fp8_e4m3";
    case KVCachePrecision::AUTO:
    default:
        return nullptr;
    }
}

inline bool parse_kv_cache_precision(const std::string& s, KVCachePrecision& out) {
    if (s == "auto") {
        out = KVCachePrecision::AUTO;
        return true;
    }
    if (s == "high_accuracy") {
        out = KVCachePrecision::HIGH_ACCURACY;
        return true;
    }
    if (s == "high_throughput") {
        out = KVCachePrecision::HIGH_THROUGHPUT;
        return true;
    }
    return false;
}

// Phase 1 (Python API): Python-friendly context configuration.
// Maps to InferenceContext::ContextParams but uses string-based enums
// so Python users don't need to import C++ enum types.
struct ContextConfig {
    std::string kv_cache_dtype = "fp32";      // "fp32"|"f16"|"q8_0"|"q4_0"|"q4_k"
    std::string kv_cache_precision = "auto";  // "auto"|"high_accuracy"|"high_throughput"
    std::string kv_cache_type_k = "";         // if empty, use kv_cache_dtype
    std::string kv_cache_type_v = "";         // if empty, use kv_cache_dtype
    std::string kv_storage = "contiguous";    // "auto"|"contiguous"|"paged"
    int page_size = 16;
    int max_seq_len = 4096;
    int max_num_seqs = 4;
    int gpu_layers = -1;
    int n_batch = 512;
    int n_ubatch = 256;
    int n_threads = 4;
    int n_threads_batch = 8;
    bool prefix_cache = false;
    int64_t prefix_cache_bytes = 0;
    int swa_window = 0;
    std::string device = "cuda";  // "cuda"|"cpu"
};

}  // namespace forge
