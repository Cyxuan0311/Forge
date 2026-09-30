#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "forge/gguf_model.h"
#include "perf/thread_pool.h"

namespace forge::inspect {

// A fully-parsed GGUF header snapshot (metadata KV pairs + tensor index).
// This mirrors the data GgufModel exposes, but the tensor index is decoded
// with a two-pass parallel algorithm (see gguf_scanner.cpp).
struct GgufSnapshot {
    bool ok = false;
    std::string error;
    uint32_t version = 0;
    uint64_t tensor_count = 0;
    uint64_t metadata_count = 0;
    uint64_t file_size = 0;

    std::unordered_map<std::string, std::string> meta_str;
    std::unordered_map<std::string, int64_t> meta_int;
    std::unordered_map<std::string, double> meta_float;
    std::unordered_map<std::string, std::vector<int32_t>> meta_int_array;
    // Element count of a STRING array, keyed by the array's own name. The strings
    // themselves are skipped rather than stored: a 128k-entry vocabulary would cost
    // megabytes and nothing here reads them, but the length is what tells you the
    // vocabulary size and whether token_embd matches it.
    std::unordered_map<std::string, uint64_t> meta_str_array_len;

    std::vector<GgufLoadedTensor> tensors;
};

// Load a GGUF header from `path`. When threads > 1 the metadata KV section and
// the tensor index are decoded in parallel across `threads` workers; threads
// <= 1 runs serially. If `pool` is null, a process-wide cached pool is reused.
GgufSnapshot load_gguf(const std::string& path, int threads, ThreadPool* pool = nullptr);

// Number of on-disk bytes for `numel` elements of a GGML type, block padding
// included. Exposed so the tests build a file whose tensor offsets agree with
// the sizes the scanner will derive, instead of inventing placeholder strides.
int64_t packed_bytes(GgmlDType dt, int64_t numel);

// Human-readable name for a GGML tensor type, e.g. "Q4_K".
std::string dtype_name(GgmlDType dt);

// Parallel aggregate statistics over the loaded tensors.
struct TensorStats {
    struct DtypeStat {
        GgmlDType dtype = GgmlDType::F32;
        std::string name;   // e.g. "Q4_K"
        int64_t count = 0;  // number of tensors
        int64_t bytes = 0;  // packed bytes on disk
        int64_t elements = 0;
    };
    std::vector<DtypeStat> by_dtype;  // sorted by bytes (desc)
    int64_t total_bytes = 0;
    int64_t total_elements = 0;
    std::vector<std::pair<std::string, int64_t>> layer_bytes;  // "model.layers.N" -> bytes
    int64_t peak_tensor_bytes = 0;                             // largest single tensor
    std::string largest_tensor_name;
};

TensorStats compute_stats(const GgufSnapshot& snap, int threads, ThreadPool* pool = nullptr);

}  // namespace forge::inspect
