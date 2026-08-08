#pragma once
#include <string>
#include "kv_cache.h"

namespace forge {

// Phase 1 (Python API): Python-friendly context configuration.
// Maps to InferenceContext::ContextParams but uses string-based enums
// so Python users don't need to import C++ enum types.
struct ContextConfig {
    std::string kv_cache_dtype = "fp32";   // "fp32"|"f16"|"q8_0"|"q4_0"|"q4_k"
    std::string kv_cache_type_k = "";      // if empty, use kv_cache_dtype
    std::string kv_cache_type_v = "";      // if empty, use kv_cache_dtype
    std::string kv_storage = "contiguous"; // "auto"|"contiguous"|"paged"
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
    std::string device = "cuda";           // "cuda"|"cpu"
};

}  // namespace forge
