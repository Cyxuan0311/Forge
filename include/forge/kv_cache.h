#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "tensor.h"

namespace forge {

enum class KVCacheDType : int {
    FP32 = 0,
    Q4_0 = 1,
};

// Per-cell metadata for sequence-aware KV cache.
// Inspired by llama.cpp's llama_kv_cells — tracks which sequences own each cell.
struct KVCellMeta {
    int64_t pos = -1;            // token position, -1 = free cell
    uint32_t seq_id_mask = 0;    // bitmask of owning sequences (bit i → seq_id i)

    bool is_free() const { return pos == -1; }
    bool has_seq(int seq_id) const { return (seq_id_mask >> seq_id) & 1u; }
    void add_seq(int seq_id) { seq_id_mask |= (1u << seq_id); }
    void rm_seq(int seq_id) { seq_id_mask &= ~(1u << seq_id); }
    bool no_seqs() const { return seq_id_mask == 0; }
};

struct KVCacheLayer {
    TensorPtr key_cache;
    TensorPtr value_cache;
    std::vector<uint8_t> q_key_cache;
    std::vector<uint8_t> q_value_cache;
    // CUDA-side quantized KV cache buffers
    void* d_q_key_cache = nullptr;
    void* d_q_value_cache = nullptr;
    size_t d_q_cache_bytes = 0;
    int filled = 0;              // retained for backward compat and PagedKVCache migration

    // Cell metadata track — size max_seq_len, parallel to key_cache/value_cache
    std::vector<KVCellMeta> cells;
};

class KVCache {
public:
    KVCache() = default;
    ~KVCache();

    bool init(int num_layers, int num_kv_heads, int head_dim, int max_seq_len, DeviceType device);

    bool init_quantized(int num_layers, int num_kv_heads, int head_dim, int max_seq_len,
                        DeviceType device, KVCacheDType kv_dtype);

    // Per-layer KV cache init (for mixed-attention architectures like Gemma 4)
    bool init_per_layer(int num_layers, const std::vector<int>& kv_dims, int max_seq_len,
                        DeviceType device);

    void reset();

    // --- Legacy update (single-seq, seq_id=0, pos=filled) ---
    int update(int layer, const TensorPtr& new_key, const TensorPtr& new_value, int seq_len);

    // --- Sequence-aware update (explicit seq_id and start position) ---
    int update(int layer, int seq_id, int64_t pos,
               const TensorPtr& new_key, const TensorPtr& new_value, int seq_len);

    // --- Sequence operations ---

    // Free positions [p0, p1) from seq_id. If a cell loses all owners, it becomes free.
    void seq_rm(int seq_id, int64_t p0, int64_t p1);

    // Zero-copy metadata transfer: add dst_seq ownership for cells in [p0, p1) owned by src_seq.
    // No KV data is copied — both sequences share the same physical cell.
    void seq_cp(int src_seq, int dst_seq, int64_t p0, int64_t p1);

    // Remove all cells NOT owned by seq_id. Cells exclusively owned by other sequences are freed.
    void seq_keep(int seq_id);

    // Find first free cell slot (linear scan with wraparound).
    // Returns -1 if cache is full.
    int find_slot(int layer) const;

    // How many filled cells belong to a given sequence (across all layers).
    int seq_filled(int layer, int seq_id) const;

    // --- Accessors ---

    TensorPtr get_key(int layer) const;
    TensorPtr get_value(int layer) const;

    TensorPtr get_key_filled(int layer) const;
    TensorPtr get_value_filled(int layer) const;

    int filled(int layer) const;
    int max_seq_len() const { return max_seq_len_; }
    int num_layers() const { return static_cast<int>(layers_.size()); }
    DeviceType device() const { return device_; }
    KVCacheDType kv_dtype() const { return kv_dtype_; }
    int max_seqs() const { return max_seqs_; }
    int kv_dim(int layer) const {
        if (!kv_dim_per_layer_.empty() && layer >= 0 && layer < (int)kv_dim_per_layer_.size())
            return kv_dim_per_layer_[layer];
        return num_kv_heads_ * head_dim_;
    }

    size_t nbytes() const;

    static size_t q4_0_block_nbytes(int n);

    void dequantize_layer(int layer);

private:
    int update_fp32(int layer, int64_t start_pos,
                    const TensorPtr& new_key, const TensorPtr& new_value, int seq_len);
    int update_quantized(int layer, int64_t start_pos,
                         const TensorPtr& new_key, const TensorPtr& new_value, int seq_len);
    int update_quantized_cuda(int layer, int64_t start_pos,
                              const TensorPtr& new_key, const TensorPtr& new_value, int seq_len);
    void dequantize_layer_cuda(int layer);
    void alloc_cuda_q_cache(int layer, size_t bytes);
    void init_cells(int layer);  // allocate cells vector for a layer

    std::vector<KVCacheLayer> layers_;
    int num_kv_heads_ = 0;
    int head_dim_ = 0;
    std::vector<int> kv_dim_per_layer_;  // per-layer kv_dim for mixed-attention archs
    int max_seq_len_ = 0;
    DeviceType device_ = DeviceType::CPU;
    KVCacheDType kv_dtype_ = KVCacheDType::FP32;
    int max_seqs_ = 32;  // max concurrent sequences (uint32_t bitmask supports up to 32)
};

}  // namespace forge
