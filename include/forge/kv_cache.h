#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "tensor.h"

namespace forge {

enum class KVCacheDType : int {
    FP32 = 0,
    F16  = 1,
    Q8_0 = 2,
    Q4_0 = 3,  // renumbered, was 1
    Q4_K = 4,
};

// Internal feature flag: selects the KV storage backend.
// Default is Contiguous (existing KVCache). Paged is reserved for future phases.
enum class KVStorageMode : int {
    Contiguous = 0,
    Paged      = 1,
};

// Phase 6: per-layer memory policy.
// Inspired by llama.cpp's llama_kv_cache_iswa — different layers can use
// different eviction/memory strategies. This formalizes the previously
// implicit use_ring_buffer_ bool into a first-class concept.
enum class KVLayerPolicy : int {
    None           = 0,  // unset / default
    Full           = 1,  // full attention, linear KV growth
    SlidingWindow  = 2,  // SWA: ring buffer, window_size eviction
    Recurrent      = 3,  // recurrent state (future: SSM/Mamba) — stub
};

// K/V can have different quantization types (asymmetric KV cache).
struct KVCacheTypeConfig {
    KVCacheDType type_k = KVCacheDType::FP32;
    KVCacheDType type_v = KVCacheDType::FP32;
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

// =========================================================================
// KVCacheStorage — unified per-layer storage bound to a device.
//
// Replaces the old dual-track (host std::vector<uint8_t> + bare CUDA void*)
// and the FP32 shadow cache for quantized modes.
//
// For FP32 mode: holds a TensorPtr (FP32, device-bound).
// For quantized modes (F16/Q8_0/Q4_0/Q4_K):
//   - CPU layers: h_data vector
//   - CUDA layers: d_data + d_bytes (RAII-managed)
// =========================================================================

struct KVCacheStorage {
    KVCacheDType dtype = KVCacheDType::FP32;
    DeviceType device = DeviceType::CPU;

    // --- FP32 path: TensorPtr (used when dtype == FP32) ---
    TensorPtr tensor;   // FP32 Tensor, device-bound

    // --- Quantized path: raw buffers ---
    std::vector<uint8_t> h_data;     // CPU-side quantized buffer
    void* d_data = nullptr;          // CUDA-side quantized buffer
    size_t d_bytes = 0;              // allocated size of d_data

    // --- Common ---
    int max_rows = 0;                // max_seq_len
    size_t row_bytes = 0;            // bytes per row (block_nbytes for quantized, kv_dim*4 for FP32)

    KVCacheStorage() = default;
    ~KVCacheStorage();

    // Non-copyable, movable
    KVCacheStorage(const KVCacheStorage&) = delete;
    KVCacheStorage& operator=(const KVCacheStorage&) = delete;
    KVCacheStorage(KVCacheStorage&& o) noexcept;
    KVCacheStorage& operator=(KVCacheStorage&& o) noexcept;

    // Allocate storage for given dimensions.
    // For FP32: creates a Tensor(max_rows, row_bytes/4) on `device`.
    // For quantized: allocates h_data (CPU) or d_data (CUDA).
    void alloc(KVCacheDType dt, DeviceType dev, int max_rows, size_t row_bytes);

    // Zero-fill all allocated storage
    void zero_fill();

    // Total allocated bytes
    size_t capacity_bytes() const;

    // ---- Data access ----

    // FP32 data pointer (only valid when dtype == FP32)
    float* fp32_data();
    const float* fp32_data() const;

    // Quantized host data pointer (only valid when device == CPU && dtype != FP32)
    uint8_t* q_data();
    const uint8_t* q_data() const;

    // Quantized device data pointer (only valid when device == CUDA && dtype != FP32)
    void* d_q_data();
    const void* d_q_data() const;

    // Row pointer (host) — for CPU quantized mode
    uint8_t* q_row(int row);
    const uint8_t* q_row(int row) const;

    // FP32 row pointer — for FP32 mode
    float* fp32_row(int row);
    const float* fp32_row(int row) const;
};

// =========================================================================
// KVCacheLayer — per-layer KV storage + metadata
// =========================================================================

struct KVCacheLayer {
    KVCacheStorage key_store;
    KVCacheStorage value_store;
    int filled = 0;             // physical write cursor for non-ring; logical pos_max for ring
    int logical_filled = 0;     // monotonic logical position max (all layers, for rollback/seq_rm)
    int dequantized_filled = 0; // how many rows have been dequantized (for incremental dequant)

    // Cell metadata — size max_seq_len, parallel to KV rows
    std::vector<KVCellMeta> cells;
};

class KVCache {
public:
    KVCache() = default;
    ~KVCache();

    bool init(int num_layers, int num_kv_heads, int head_dim, int max_seq_len, DeviceType device);

    bool init_quantized(int num_layers, int num_kv_heads, int head_dim, int max_seq_len,
                        DeviceType device, KVCacheDType kv_dtype);

    // Per-K/V-type init (asymmetric KV cache)
    bool init_quantized(int num_layers, int num_kv_heads, int head_dim, int max_seq_len,
                        DeviceType device, const KVCacheTypeConfig& kv_config);

    // Per-layer KV cache init (for mixed-attention architectures like Gemma 4)
    bool init_per_layer(int num_layers, const std::vector<int>& kv_dims, int max_seq_len,
                        DeviceType device);

    // Per-layer device init: call after init_quantized/init_per_layer to place
    // each layer's KV cache on its target device (CPU or CUDA).
    // layer_devices[i] = device for layer i. Empty = all layers stay on `device`.
    void set_layer_devices(const std::vector<DeviceType>& layer_devices);

    // Per-layer device query
    DeviceType layer_device(int layer) const;

    // --- Phase 6: per-layer memory policy ---
    // Set per-layer policies. SlidingWindow layers will use ring buffer
    // eviction with window_size = swa_window (typically cfg.n_swa).
    // Full layers grow linearly. Recurrent is a stub (treated as Full).
    // This replaces the older set_ring_buffer() calls with a unified API.
    void set_layer_policies(const std::vector<KVLayerPolicy>& policies, int swa_window = 0);
    KVLayerPolicy layer_policy(int layer) const;

    // Set the CUDA stream for all KV cache operations (default: stream 0)
    void set_cuda_stream(void* stream);

    void reset();

    // Roll back KV cache filled position to `to_pos` (for speculative decoding rejection)
    void rollback(int64_t to_pos);

    // --- Ring buffer mode ---
    // Enable ring buffer per-layer: KV entries wrap around at window_size,
    // so SWA attention automatically sees only the latest window_size positions.
    // Call after init/init_per_layer. Pass the layer index to enable ring buffer
    // for specific (SWA) layers only, or -1 to enable for all layers.
    void set_ring_buffer(int window_size, int layer = -1);
    bool use_ring_buffer(int layer) const;
    int window_size() const { return window_size_; }

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

    // Returns FP32 key/value tensor for the layer.
    // For FP32 mode: returns the storage tensor directly.
    // For quantized mode: triggers dequantization into a temporary FP32 buffer.
    TensorPtr get_key(int layer) const;
    TensorPtr get_value(int layer) const;

    // Returns the "active" key/value tensor for attention.
    // In ring buffer mode, returns a contiguous view of the last min(filled, window_size)
    // entries, reordered so that older entries come first.
    // In normal mode, returns slice [0, filled).
    TensorPtr get_key_filled(int layer) const;
    TensorPtr get_value_filled(int layer) const;

    // Effective KV length for attention:
    // - Normal mode: layers_[layer].filled
    // - Ring buffer mode: min(layers_[layer].filled, window_size_)
    int filled(int layer) const;
    int max_seq_len() const { return max_seq_len_; }
    int num_layers() const { return static_cast<int>(layers_.size()); }
    const std::vector<KVCacheLayer>& layers() const { return layers_; }

    // Access quantized CUDA cache pointers for fused attention kernels
    void* d_q_key_cache(int layer) const;
    void* d_q_value_cache(int layer) const;

    DeviceType device() const { return device_; }
    KVCacheDType kv_dtype() const { return kv_dtype_; }
    KVCacheDType type_k() const { return kv_config_.type_k; }
    KVCacheDType type_v() const { return kv_config_.type_v; }
    const KVCacheTypeConfig& kv_config() const { return kv_config_; }
    int max_seqs() const { return max_seqs_; }
    int kv_dim(int layer) const {
        if (!kv_dim_per_layer_.empty() && layer >= 0 && layer < (int)kv_dim_per_layer_.size())
            return kv_dim_per_layer_[layer];
        return num_kv_heads_ * head_dim_;
    }

    size_t nbytes() const;

    // Number of bytes actively occupied by filled cells (sum across all layers, K+V).
    size_t active_bytes() const;

    // Number of free cell slots across all layers.
    int num_free_slots() const;

    static size_t q4_0_block_nbytes(int n);
    static size_t block_nbytes(KVCacheDType dtype, int n);

    // Quantize/dequantize a single row (public for PagedKVStorage reuse).
    // For FP32: memcpy. For F16/Q8_0/Q4_0/Q4_K: calls the appropriate CPU quantizer.
    static void quantize_row(KVCacheDType dtype, const float* src, uint8_t* dst, int n);
    static void dequantize_row(KVCacheDType dtype, const uint8_t* src, float* dst, int n);

    // Dequantize a layer's KV data into FP32 (on-demand, for non-fused attention paths).
    // Returns the FP32 key/value tensor pair. Results are cached per-layer.
    void dequantize_layer(int layer);

    // Get the quantized row size for key/value at a given layer
    size_t key_row_bytes(int layer) const;
    size_t value_row_bytes(int layer) const;

private:
    int update_fp32(int layer, int64_t start_pos,
                    const TensorPtr& new_key, const TensorPtr& new_value, int seq_len);
    int update_quantized(int layer, int64_t start_pos,
                         const TensorPtr& new_key, const TensorPtr& new_value, int seq_len);
    int update_quantized_cuda(int layer, int64_t start_pos,
                              const TensorPtr& new_key, const TensorPtr& new_value, int seq_len);
    void dequantize_layer_cuda(int layer);
    void init_cells(int layer);  // allocate cells vector for a layer

    // Get the CUDA stream (returns nullptr for default stream)
    void* cuda_stream() const;

    std::vector<KVCacheLayer> layers_;
    int num_kv_heads_ = 0;
    int head_dim_ = 0;
    std::vector<int> kv_dim_per_layer_;  // per-layer kv_dim for mixed-attention archs
    int max_seq_len_ = 0;
    DeviceType device_ = DeviceType::CPU;               // fallback / primary device
    std::vector<DeviceType> layer_devices_;             // per-layer device (empty = all on device_)
    KVCacheDType kv_dtype_ = KVCacheDType::FP32;       // legacy: max(type_k, type_v)
    KVCacheTypeConfig kv_config_;                        // per-K/V type config
    int max_seqs_ = 32;  // max concurrent sequences (uint32_t bitmask supports up to 32)

    // Ring buffer mode (SWA sliding window)
    int window_size_ = 0;       // maximum KV entries visible to attention
    // Per-layer: whether ring buffer is active for this layer
    std::vector<bool> use_ring_buffer_;
    // Per-layer ring cursors: next slot to write (wraps at window_size_).
    // Only meaningful for SWA layers; non-SWA layers use normal linear filling.
    std::vector<int> ring_cursor_;

    // Phase 6: per-layer memory policy (formalizes use_ring_buffer_).
    // SlidingWindow → ring buffer; Full → linear; Recurrent → stub (linear).
    std::vector<KVLayerPolicy> layer_policies_;

    // CUDA stream for KV cache operations (default: nullptr = default stream)
    void* cuda_stream_ = nullptr;
};

}  // namespace forge
