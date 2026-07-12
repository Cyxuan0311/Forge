#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "tensor.h"

namespace forge {

struct KVBlock {
    TensorPtr key_block;
    TensorPtr value_block;
    int ref_count = 0;
    bool in_use = false;
};

struct SeqBlockTable {
    std::vector<int> block_ids;
    int num_filled_in_last = 0;
};

class PagedKVCache {
public:
    PagedKVCache() = default;

    bool init(int num_layers, int num_kv_heads, int head_dim, int max_seq_len, int block_size,
              int max_num_seqs, DeviceType device);

    int allocate_seq(int seq_id);
    void release_seq(int seq_id);

    int append(int seq_id, int layer, const TensorPtr& new_key, const TensorPtr& new_value,
               int seq_len);
    TensorPtr get_key(int seq_id, int layer) const;
    TensorPtr get_value(int seq_id, int layer) const;

    int filled(int seq_id, int layer) const;
    int num_seqs() const;
    bool has_seq(int seq_id) const;

    int block_size() const { return block_size_; }
    int num_layers() const { return num_layers_; }
    int num_kv_heads() const { return num_kv_heads_; }
    int head_dim() const { return head_dim_; }
    DeviceType device() const { return device_; }

    size_t nbytes() const;
    int num_free_blocks() const;
    int num_total_blocks() const;

private:
    int alloc_block();
    void free_block(int block_id);
    void ensure_seq_capacity(int seq_id, int layer, int needed_slots);

    int num_layers_ = 0;
    int num_kv_heads_ = 0;
    int head_dim_ = 0;
    int max_seq_len_ = 0;
    int block_size_ = 16;
    int max_num_seqs_ = 0;
    DeviceType device_ = DeviceType::CPU;

    std::vector<std::vector<KVBlock>> layer_blocks_;
    std::vector<int> free_block_ids_;

    std::unordered_map<int, std::vector<SeqBlockTable>> seq_tables_;

    mutable std::mutex mutex_;
};

}  // namespace forge
