#include "nanoinfer/paged_kv_cache.h"
#include "nanoinfer/logger.h"
#include <cstring>
#include <algorithm>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace nanoinfer {

bool PagedKVCache::init(int num_layers, int num_kv_heads, int head_dim,
                         int max_seq_len, int block_size, int max_num_seqs,
                         DeviceType device) {
    num_layers_ = num_layers;
    num_kv_heads_ = num_kv_heads;
    head_dim_ = head_dim;
    max_seq_len_ = max_seq_len;
    block_size_ = block_size;
    max_num_seqs_ = max_num_seqs;
    device_ = device;

    int kv_dim = num_kv_heads * head_dim;
    int num_blocks = (max_seq_len * max_num_seqs + block_size - 1) / block_size;
    num_blocks = std::max(num_blocks, 1);

    layer_blocks_.resize(num_layers);
    for (int l = 0; l < num_layers; ++l) {
        layer_blocks_[l].resize(num_blocks);
        for (int b = 0; b < num_blocks; ++b) {
            auto& blk = layer_blocks_[l][b];
            blk.key_block = std::make_shared<Tensor>(DataType::FP32,
                std::vector<int64_t>{block_size, kv_dim}, device);
            blk.value_block = std::make_shared<Tensor>(DataType::FP32,
                std::vector<int64_t>{block_size, kv_dim}, device);
            blk.key_block->zero_();
            blk.value_block->zero_();
            blk.ref_count = 0;
            blk.in_use = false;
        }
    }

    free_block_ids_.resize(num_blocks);
    for (int i = 0; i < num_blocks; ++i) {
        free_block_ids_[i] = i;
    }

    LOG_INFO("PagedKVCache initialized: " + std::to_string(num_layers) + " layers, " +
             std::to_string(num_blocks) + " blocks, block_size=" + std::to_string(block_size) +
             ", kv_dim=" + std::to_string(kv_dim));

    return true;
}

int PagedKVCache::allocate_seq(int seq_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (seq_tables_.count(seq_id) > 0) {
        LOG_WARN("PagedKVCache: seq_id " + std::to_string(seq_id) + " already exists");
        return -1;
    }

    seq_tables_[seq_id].resize(num_layers_);
    for (int l = 0; l < num_layers_; ++l) {
        seq_tables_[seq_id][l].block_ids.clear();
        seq_tables_[seq_id][l].num_filled_in_last = 0;
    }

    return 0;
}

void PagedKVCache::release_seq(int seq_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = seq_tables_.find(seq_id);
    if (it == seq_tables_.end()) return;

    for (int l = 0; l < num_layers_; ++l) {
        for (int bid : it->second[l].block_ids) {
            free_block(bid);
        }
    }

    seq_tables_.erase(it);
}

int PagedKVCache::alloc_block() {
    if (free_block_ids_.empty()) {
        LOG_ERROR("PagedKVCache: out of blocks");
        return -1;
    }
    int bid = free_block_ids_.back();
    free_block_ids_.pop_back();
    for (int l = 0; l < num_layers_; ++l) {
        layer_blocks_[l][bid].in_use = true;
        layer_blocks_[l][bid].ref_count = 1;
    }
    return bid;
}

void PagedKVCache::free_block(int block_id) {
    for (int l = 0; l < num_layers_; ++l) {
        auto& blk = layer_blocks_[l][block_id];
        blk.ref_count--;
        if (blk.ref_count <= 0) {
            blk.in_use = false;
            blk.ref_count = 0;
        }
    }
    free_block_ids_.push_back(block_id);
}

int PagedKVCache::append(int seq_id, int layer, const TensorPtr& new_key, const TensorPtr& new_value, int seq_len) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = seq_tables_.find(seq_id);
    if (it == seq_tables_.end()) return -1;

    auto& table = it->second[layer];
    int kv_dim = num_kv_heads_ * head_dim_;

    int total_filled = static_cast<int>(table.block_ids.size()) * block_size_
                       - block_size_ + table.num_filled_in_last;
    if (!table.block_ids.empty()) {
        total_filled = (static_cast<int>(table.block_ids.size()) - 1) * block_size_ + table.num_filled_in_last;
    } else {
        total_filled = 0;
    }

    int slots_needed = total_filled + seq_len;
    int slots_available = table.block_ids.empty() ? 0 :
                          (static_cast<int>(table.block_ids.size()) * block_size_);

    ensure_seq_capacity(seq_id, layer, slots_needed);

    const float* k_src = static_cast<const float*>(new_key->data());
    const float* v_src = static_cast<const float*>(new_value->data());

    std::vector<float> h_key, h_val;
    if (device_ == DeviceType::CUDA) {
#ifdef USE_CUDA
        h_key.resize(seq_len * kv_dim);
        h_val.resize(seq_len * kv_dim);
        cudaMemcpy(h_key.data(), k_src, seq_len * kv_dim * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_val.data(), v_src, seq_len * kv_dim * sizeof(float), cudaMemcpyDeviceToHost);
        k_src = h_key.data();
        v_src = h_val.data();
#endif
    }

    int write_offset = total_filled;
    for (int s = 0; s < seq_len; ++s) {
        int global_slot = write_offset + s;
        int block_idx = global_slot / block_size_;
        int slot_in_block = global_slot % block_size_;

        while (block_idx >= static_cast<int>(table.block_ids.size())) {
            int bid = alloc_block();
            if (bid < 0) return -1;
            table.block_ids.push_back(bid);
            table.num_filled_in_last = 0;
        }

        int bid = table.block_ids[block_idx];
        auto& blk = layer_blocks_[layer][bid];

        float* k_dst = static_cast<float*>(blk.key_block->data()) + slot_in_block * kv_dim;
        float* v_dst = static_cast<float*>(blk.value_block->data()) + slot_in_block * kv_dim;

        if (device_ == DeviceType::CUDA) {
#ifdef USE_CUDA
            std::vector<float> row(kv_dim);
            std::memcpy(row.data(), k_src + s * kv_dim, kv_dim * sizeof(float));
            cudaMemcpy(k_dst, row.data(), kv_dim * sizeof(float), cudaMemcpyHostToDevice);
            std::memcpy(row.data(), v_src + s * kv_dim, kv_dim * sizeof(float));
            cudaMemcpy(v_dst, row.data(), kv_dim * sizeof(float), cudaMemcpyHostToDevice);
#endif
        } else {
            std::memcpy(k_dst, k_src + s * kv_dim, kv_dim * sizeof(float));
            std::memcpy(v_dst, v_src + s * kv_dim, kv_dim * sizeof(float));
        }

        table.num_filled_in_last = slot_in_block + 1;
    }

    int new_filled = write_offset + seq_len;
    return new_filled;
}

void PagedKVCache::ensure_seq_capacity(int seq_id, int layer, int needed_slots) {
    auto& table = seq_tables_[seq_id][layer];
    int current_capacity = static_cast<int>(table.block_ids.size()) * block_size_;
    while (current_capacity < needed_slots) {
        int bid = alloc_block();
        if (bid < 0) return;
        table.block_ids.push_back(bid);
        table.num_filled_in_last = 0;
        current_capacity += block_size_;
    }
}

TensorPtr PagedKVCache::get_key(int seq_id, int layer) const {
    auto it = seq_tables_.find(seq_id);
    if (it == seq_tables_.end()) return nullptr;

    const auto& table = it->second[layer];
    int total_filled = 0;
    if (!table.block_ids.empty()) {
        total_filled = (static_cast<int>(table.block_ids.size()) - 1) * block_size_ + table.num_filled_in_last;
    }
    if (total_filled == 0) return nullptr;

    int kv_dim = num_kv_heads_ * head_dim_;
    auto result = std::make_shared<Tensor>(DataType::FP32,
        std::vector<int64_t>{total_filled, kv_dim}, device_);
    result->zero_();

    float* dst = static_cast<float*>(result->data());

    int rows_to_copy = total_filled;
    for (int b = 0; b < static_cast<int>(table.block_ids.size()) && rows_to_copy > 0; ++b) {
        int bid = table.block_ids[b];
        const auto& blk = layer_blocks_[layer][bid];
        int rows_in_this_block = std::min(block_size_, rows_to_copy);

        if (device_ == DeviceType::CUDA) {
#ifdef USE_CUDA
            cudaMemcpy(dst, blk.key_block->data(),
                       rows_in_this_block * kv_dim * sizeof(float),
                       cudaMemcpyDeviceToDevice);
#endif
        } else {
            std::memcpy(dst, blk.key_block->data(),
                        rows_in_this_block * kv_dim * sizeof(float));
        }

        dst += rows_in_this_block * kv_dim;
        rows_to_copy -= rows_in_this_block;
    }

    return result;
}

TensorPtr PagedKVCache::get_value(int seq_id, int layer) const {
    auto it = seq_tables_.find(seq_id);
    if (it == seq_tables_.end()) return nullptr;

    const auto& table = it->second[layer];
    int total_filled = 0;
    if (!table.block_ids.empty()) {
        total_filled = (static_cast<int>(table.block_ids.size()) - 1) * block_size_ + table.num_filled_in_last;
    }
    if (total_filled == 0) return nullptr;

    int kv_dim = num_kv_heads_ * head_dim_;
    auto result = std::make_shared<Tensor>(DataType::FP32,
        std::vector<int64_t>{total_filled, kv_dim}, device_);
    result->zero_();

    float* dst = static_cast<float*>(result->data());

    int rows_to_copy = total_filled;
    for (int b = 0; b < static_cast<int>(table.block_ids.size()) && rows_to_copy > 0; ++b) {
        int bid = table.block_ids[b];
        const auto& blk = layer_blocks_[layer][bid];
        int rows_in_this_block = std::min(block_size_, rows_to_copy);

        if (device_ == DeviceType::CUDA) {
#ifdef USE_CUDA
            cudaMemcpy(dst, blk.value_block->data(),
                       rows_in_this_block * kv_dim * sizeof(float),
                       cudaMemcpyDeviceToDevice);
#endif
        } else {
            std::memcpy(dst, blk.value_block->data(),
                        rows_in_this_block * kv_dim * sizeof(float));
        }

        dst += rows_in_this_block * kv_dim;
        rows_to_copy -= rows_in_this_block;
    }

    return result;
}

int PagedKVCache::filled(int seq_id, int layer) const {
    auto it = seq_tables_.find(seq_id);
    if (it == seq_tables_.end()) return 0;

    const auto& table = it->second[layer];
    if (table.block_ids.empty()) return 0;
    return (static_cast<int>(table.block_ids.size()) - 1) * block_size_ + table.num_filled_in_last;
}

int PagedKVCache::num_seqs() const {
    return static_cast<int>(seq_tables_.size());
}

bool PagedKVCache::has_seq(int seq_id) const {
    return seq_tables_.count(seq_id) > 0;
}

size_t PagedKVCache::nbytes() const {
    if (layer_blocks_.empty()) return 0;
    int num_blocks = static_cast<int>(layer_blocks_[0].size());
    int kv_dim = num_kv_heads_ * head_dim_;
    size_t per_block = block_size_ * kv_dim * sizeof(float) * 2;
    return per_block * num_blocks * num_layers_;
}

int PagedKVCache::num_free_blocks() const {
    return static_cast<int>(free_block_ids_.size());
}

int PagedKVCache::num_total_blocks() const {
    if (layer_blocks_.empty()) return 0;
    return static_cast<int>(layer_blocks_[0].size());
}

} // namespace nanoinfer
