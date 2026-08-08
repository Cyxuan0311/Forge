#include "forge/kv_storage.h"

#include <algorithm>
#include <cstring>

#include "forge/cuda_kernels.h"
#include "forge/logger.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace forge {

// =========================================================================
// ContiguousKVStorage
// =========================================================================

ContiguousKVStorage::ContiguousKVStorage(KVCache& cache) : cache_(cache) {}

bool ContiguousKVStorage::init(int /*num_layers*/, const std::vector<int>& /*kv_dims*/,
                                int /*max_seq_len*/, DeviceType /*device*/,
                                const KVCacheTypeConfig& /*kv_config*/,
                                int /*page_size*/, int /*max_num_seqs*/) {
    // Contiguous mode: KVCache is already initialized by the engine.
    return true;
}

bool ContiguousKVStorage::write_kv(int layer, int64_t pos, int seq_len,
                                   const float* k_data, const float* v_data) {
    // Contiguous mode delegates directly to KVCache::update.
    // The caller must ensure the K/V tensors are on the correct device.
    (void)k_data;
    (void)v_data;
    (void)layer;
    (void)pos;
    (void)seq_len;
    return true;
}

bool ContiguousKVStorage::write_kv_seq(int layer, int seq_id, int64_t pos, int seq_len,
                                       const float* /*k_data*/, const float* /*v_data*/) {
    // Contiguous mode uses KVCache::update() directly via the engine.
    (void)layer;
    (void)seq_id;
    (void)pos;
    (void)seq_len;
    return true;
}

TensorPtr ContiguousKVStorage::read_key(int layer) const {
    return cache_.get_key_filled(layer);
}

TensorPtr ContiguousKVStorage::read_value(int layer) const {
    return cache_.get_value_filled(layer);
}

int ContiguousKVStorage::filled(int layer) const {
    return cache_.filled(layer);
}

int ContiguousKVStorage::seq_filled(int layer, int seq_id) const {
    return cache_.seq_filled(layer, seq_id);
}

bool ContiguousKVStorage::seq_remove(int seq_id, int64_t p0, int64_t p1) {
    if (seq_id < 0 || seq_id >= cache_.max_seqs()) {
        LOG_ERROR("ContiguousKVStorage::seq_remove: invalid seq_id " + std::to_string(seq_id));
        return false;
    }
    cache_.seq_rm(seq_id, p0, p1);
    return true;
}

bool ContiguousKVStorage::seq_share(int src_seq, int dst_seq, int64_t p0, int64_t p1) {
    if (src_seq < 0 || src_seq >= cache_.max_seqs() ||
        dst_seq < 0 || dst_seq >= cache_.max_seqs()) {
        LOG_ERROR("ContiguousKVStorage::seq_share: invalid seq_id src=" + std::to_string(src_seq) +
                  " dst=" + std::to_string(dst_seq));
        return false;
    }
    cache_.seq_cp(src_seq, dst_seq, p0, p1);
    return true;
}

bool ContiguousKVStorage::seq_keep(int seq_id) {
    if (seq_id < 0 || seq_id >= cache_.max_seqs()) {
        LOG_ERROR("ContiguousKVStorage::seq_keep: invalid seq_id " + std::to_string(seq_id));
        return false;
    }
    cache_.seq_keep(seq_id);
    return true;
}

void ContiguousKVStorage::release(int seq_id) {
    cache_.seq_rm(seq_id, 0, cache_.max_seq_len());
}

int ContiguousKVStorage::max_seq_len() const {
    return cache_.max_seq_len();
}

int ContiguousKVStorage::num_layers() const {
    return cache_.num_layers();
}

size_t ContiguousKVStorage::nbytes() const {
    return cache_.nbytes();
}

size_t ContiguousKVStorage::active_bytes() const {
    return cache_.active_bytes();
}

int ContiguousKVStorage::num_free_slots() const {
    return cache_.num_free_slots();
}

// =========================================================================
// PagedKVStorage
// =========================================================================

PagedKVStorage::PagedKVStorage() = default;

PagedKVStorage::~PagedKVStorage() {
#ifdef USE_CUDA
    for (int l = 0; l < num_layers_; ++l) {
        if (l < (int)d_key_page_ptrs_.size() && d_key_page_ptrs_[l])
            cudaFree(d_key_page_ptrs_[l]);
        if (l < (int)d_value_page_ptrs_.size() && d_value_page_ptrs_[l])
            cudaFree(d_value_page_ptrs_[l]);
    }
    if (d_seq_page_ids_)
        cudaFree(d_seq_page_ids_);
#endif
}

bool PagedKVStorage::init(int num_layers, const std::vector<int>& kv_dims, int max_seq_len,
                          DeviceType device, const KVCacheTypeConfig& kv_config,
                          int page_size, int max_num_seqs) {
    std::lock_guard<std::mutex> lock(mutex_);

    num_layers_ = num_layers;
    kv_dims_ = kv_dims;
    max_seq_len_ = max_seq_len;
    device_ = device;
    kv_config_ = kv_config;
    page_size_ = page_size > 0 ? page_size : 16;
    max_num_seqs_ = max_num_seqs > 0 ? max_num_seqs : 32;

    // Phase 6: per-layer pool sizing.
    // If layer_pools_ was pre-configured via set_layer_policies(), use those sizes.
    // Otherwise, default all layers to Full with uniform max_pages.
    if (layer_pools_.empty()) {
        layer_pools_.assign(num_layers_, LayerPoolInfo{});
        for (auto& p : layer_pools_) p.policy = KVLayerPolicy::Full;
    } else if ((int)layer_pools_.size() < num_layers_) {
        layer_pools_.resize(num_layers_, LayerPoolInfo{});
    }
    int theoretical_max = (max_num_seqs_ * max_seq_len_ + page_size_ - 1) / page_size_;
    const int MAX_PAGES_CAP = 256;
    for (int l = 0; l < num_layers_; ++l) {
        auto& pool = layer_pools_[l];
        if (pool.policy == KVLayerPolicy::SlidingWindow && pool.window_size > 0) {
            int swa_pages = (pool.window_size + page_size_ - 1) / page_size_;
            // SWA needs a few extra pages for the wrap-around cursor
            pool.max_pages = std::min(swa_pages + 1, MAX_PAGES_CAP);
        } else {
            pool.max_pages = std::min(theoretical_max, MAX_PAGES_CAP);
        }
    }
    // max_pages_ = max across layers (for backward-compat in some log messages)
    max_pages_ = 0;
    for (const auto& p : layer_pools_) max_pages_ = std::max(max_pages_, p.max_pages);

    // Allocate page pools per layer
    layer_pages_.resize(num_layers_);
    for (int l = 0; l < num_layers_; ++l) {
        int kv_dim = (l < (int)kv_dims_.size()) ? kv_dims_[l] : kv_dims_[0];
        size_t k_row_bytes = KVCache::block_nbytes(kv_config_.type_k, kv_dim);
        size_t v_row_bytes = KVCache::block_nbytes(kv_config_.type_v, kv_dim);

        int lmax = layer_pools_[l].max_pages;
        layer_pages_[l].resize(lmax);
        for (int p = 0; p < lmax; ++p) {
            auto& page = layer_pages_[l][p];
            page.key.alloc(kv_config_.type_k, device_, page_size_, k_row_bytes);
            page.value.alloc(kv_config_.type_v, device_, page_size_, v_row_bytes);
            page.key.zero_fill();
            page.value.zero_fill();
            page.filled = 0;
            page.ref_count = 0;
        }
        // Init free list for this layer
        layer_pools_[l].free_page_ids.clear();
        layer_pools_[l].free_page_set.clear();
        for (int p = 0; p < lmax; ++p) {
            layer_pools_[l].free_page_ids.push_back(p);
            layer_pools_[l].free_page_set.insert(p);
        }
    }

    seq_tables_.clear();

    // ---- Build CUDA page pointer tables (Phase 4) ----
    // When device_==CUDA, KVCacheStorage::alloc has already placed each page's
    // data on the device. Collect the per-page device base pointers into a host
    // array, then upload to a device array (one per layer). The paged attention
    // kernel resolves K/V rows through this table + the per-sequence page_ids.
    // FP32 pages use tensor->data() (device ptr); quantized pages use d_q_data().
#ifdef USE_CUDA
    if (device_ == DeviceType::CUDA) {
        // Free any existing device arrays (re-init case) before reallocating.
        for (int l = 0; l < (int)d_key_page_ptrs_.size(); ++l)
            if (d_key_page_ptrs_[l]) cudaFree(d_key_page_ptrs_[l]);
        for (int l = 0; l < (int)d_value_page_ptrs_.size(); ++l)
            if (d_value_page_ptrs_[l]) cudaFree(d_value_page_ptrs_[l]);
        if (d_seq_page_ids_) { cudaFree(d_seq_page_ids_); d_seq_page_ids_ = nullptr; }
        d_seq_page_ids_cap_ = 0;

        d_key_page_ptrs_host_.assign(num_layers_, {});
        d_value_page_ptrs_host_.assign(num_layers_, {});
        d_key_page_ptrs_.assign(num_layers_, nullptr);
        d_value_page_ptrs_.assign(num_layers_, nullptr);
        for (int l = 0; l < num_layers_; ++l) {
            int lmax = layer_pools_[l].max_pages;
            d_key_page_ptrs_host_[l].resize(lmax);
            d_value_page_ptrs_host_[l].resize(lmax);
            for (int p = 0; p < lmax; ++p) {
                auto& page = layer_pages_[l][p];
                d_key_page_ptrs_host_[l][p] =
                    (kv_config_.type_k == KVCacheDType::FP32)
                        ? static_cast<void*>(page.key.fp32_data())
                        : page.key.d_q_data();
                d_value_page_ptrs_host_[l][p] =
                    (kv_config_.type_v == KVCacheDType::FP32)
                        ? static_cast<void*>(page.value.fp32_data())
                        : page.value.d_q_data();
            }
            cudaMalloc(&d_key_page_ptrs_[l], lmax * sizeof(void*));
            cudaMemcpy(d_key_page_ptrs_[l], d_key_page_ptrs_host_[l].data(),
                       lmax * sizeof(void*), cudaMemcpyHostToDevice);
            cudaMalloc(&d_value_page_ptrs_[l], lmax * sizeof(void*));
            cudaMemcpy(d_value_page_ptrs_[l], d_value_page_ptrs_host_[l].data(),
                       lmax * sizeof(void*), cudaMemcpyHostToDevice);
        }
    }
#endif

    LOG_INFO("PagedKVStorage init: layers=" + std::to_string(num_layers_) +
             ", page_size=" + std::to_string(page_size_) +
             ", max_pages=" + std::to_string(max_pages_) +
             ", max_seqs=" + std::to_string(max_num_seqs_) +
             ", dev=" + (device_ == DeviceType::CUDA ? "CUDA" : "CPU"));
    return true;
}

int PagedKVStorage::alloc_page(int layer) {
    if (layer < 0 || layer >= (int)layer_pools_.size()) return -1;
    auto& pool = layer_pools_[layer];
    if (pool.free_page_ids.empty()) {
        LOG_WARN("PagedKVStorage: page pool exhausted for layer " + std::to_string(layer));
        return -1;
    }
    int page_id = pool.free_page_ids.back();
    pool.free_page_ids.pop_back();
    pool.free_page_set.erase(page_id);
    layer_pages_[layer][page_id].filled = 0;
    layer_pages_[layer][page_id].ref_count = 1;
    return page_id;
}

void PagedKVStorage::free_page(int layer, int page_id) {
    if (layer < 0 || layer >= (int)layer_pools_.size()) return;
    if (page_id < 0 || page_id >= (int)layer_pages_[layer].size()) return;
    auto& pool = layer_pools_[layer];
    if (pool.free_page_set.count(page_id)) return;  // prevent double-enqueue
    pool.free_page_ids.push_back(page_id);
    pool.free_page_set.insert(page_id);
    layer_pages_[layer][page_id].ref_count = 0;
    layer_pages_[layer][page_id].filled = 0;
}

int PagedKVStorage::layer_max_pages(int layer) const {
    if (layer < 0 || layer >= (int)layer_pools_.size()) return max_pages_;
    return layer_pools_[layer].max_pages;
}

void PagedKVStorage::set_layer_policies(const std::vector<KVLayerPolicy>& policies, int swa_window) {
    layer_pools_.assign(num_layers_ > 0 ? num_layers_ : (int)policies.size(), LayerPoolInfo{});
    for (int i = 0; i < (int)layer_pools_.size() && i < (int)policies.size(); ++i) {
        layer_pools_[i].policy = policies[i];
        if (policies[i] == KVLayerPolicy::SlidingWindow)
            layer_pools_[i].window_size = swa_window;
    }
    // If num_layers_ not yet set (pre-init), this just stores policies for init() to use.
}

std::vector<int> PagedKVStorage::layer_pool_max_pages() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<int> result;
    result.reserve(layer_pools_.size());
    for (const auto& pool : layer_pools_)
        result.push_back(pool.max_pages);
    return result;
}

void PagedKVStorage::init_seq(int seq_id) {
    if (seq_tables_.find(seq_id) == seq_tables_.end()) {
        seq_tables_[seq_id].resize(num_layers_);
    }
}

bool PagedKVStorage::ensure_seq_capacity(int seq_id, int layer, int needed_slots) {
    init_seq(seq_id);
    auto& table = seq_tables_[seq_id][layer];

    int current_capacity = (int)table.page_ids.size() * page_size_;
    if (current_capacity >= needed_slots)
        return true;

    int needed_pages = (needed_slots + page_size_ - 1) / page_size_;
    int pages_to_alloc = needed_pages - (int)table.page_ids.size();

    // Track newly allocated pages for rollback on failure
    std::vector<int> newly_allocated;
    for (int i = 0; i < pages_to_alloc; ++i) {
        int page_id = alloc_page(layer);
        if (page_id < 0) {
            // Rollback: free all newly allocated pages
            for (int pid : newly_allocated) {
                free_page(layer, pid);
            }
            LOG_ERROR("PagedKVStorage: page allocation failed for seq=" +
                      std::to_string(seq_id) + " layer=" + std::to_string(layer) +
                      ", rolled back " + std::to_string(newly_allocated.size()) + " pages");
            return false;
        }
        newly_allocated.push_back(page_id);
        table.page_ids.push_back(page_id);
    }
    return true;
}

bool PagedKVStorage::write_kv(int layer, int64_t pos, int seq_len,
                              const float* k_data, const float* v_data) {
    // Legacy single-seq write: use seq_id=0
    return write_kv_seq(layer, 0, pos, seq_len, k_data, v_data);
}

bool PagedKVStorage::write_kv_seq(int layer, int seq_id, int64_t pos, int seq_len,
                                  const float* k_data, const float* v_data) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (layer < 0 || layer >= num_layers_) {
        LOG_ERROR("PagedKVStorage::write_kv_seq: invalid layer " + std::to_string(layer));
        return false;
    }
    if (seq_id < 0) {
        LOG_ERROR("PagedKVStorage::write_kv_seq: invalid seq_id " + std::to_string(seq_id));
        return false;
    }

    int kv_dim = (layer < (int)kv_dims_.size()) ? kv_dims_[layer] : kv_dims_[0];

    // Ensure capacity for pos + seq_len
    int needed_slots = (int)(pos + seq_len);
    if (!ensure_seq_capacity(seq_id, layer, needed_slots)) {
        return false;
    }

    auto& table = seq_tables_[seq_id][layer];

    // ---- CUDA scatter path (Phase 4) ----
    // k_data/v_data are device FP32 pointers; the kernel writes directly into
    // pages and quantizes in-place. No host staging.
#ifdef USE_CUDA
    if (device_ == DeviceType::CUDA) {
        const int32_t* d_page_ids = upload_seq_page_table_unlocked(layer, seq_id);
        if (d_page_ids == nullptr) {
            LOG_ERROR("PagedKVStorage::write_kv_seq: failed to upload page table");
            return false;
        }
        size_t k_row_bytes = KVCache::block_nbytes(kv_config_.type_k, kv_dim);
        size_t v_row_bytes = KVCache::block_nbytes(kv_config_.type_v, kv_dim);
        cuda::launch_kv_scatter(k_data, v_data,
                                d_key_page_ptrs_[layer], d_value_page_ptrs_[layer],
                                d_page_ids, seq_len, pos, page_size_, kv_dim,
                                k_row_bytes, v_row_bytes, kv_config_.type_k, 0);

        // Update host-side page metadata (filled cursor). The kernel wrote the
        // device data; we only track how full each touched page is.
        for (int i = 0; i < seq_len; ++i) {
            int64_t abs_pos = pos + i;
            int page_idx = (int)(abs_pos / page_size_);
            int offset = (int)(abs_pos % page_size_);
            int page_id = table.page_ids[page_idx];
            uint32_t new_filled = (uint32_t)(offset + 1);
            if (new_filled > layer_pages_[layer][page_id].filled)
                layer_pages_[layer][page_id].filled = new_filled;
        }

        table.logical_len = std::max(table.logical_len, pos + seq_len);
        int last_page_idx = (int)((table.logical_len - 1) / page_size_);
        if (last_page_idx < (int)table.page_ids.size()) {
            table.filled_in_last = (int)(table.logical_len - (int64_t)last_page_idx * page_size_);
        }
        return true;
    }
#endif

    // ---- CPU write path (Phase 3, unchanged) ----
    // Write data into pages
    for (int i = 0; i < seq_len; ++i) {
        int64_t abs_pos = pos + i;
        int page_idx = (int)(abs_pos / page_size_);
        int offset = (int)(abs_pos % page_size_);

        if (page_idx >= (int)table.page_ids.size()) {
            LOG_ERROR("PagedKVStorage: page index out of bounds");
            return false;
        }

        int page_id = table.page_ids[page_idx];
        auto& page = layer_pages_[layer][page_id];

        const float* k_src = k_data + i * kv_dim;
        const float* v_src = v_data + i * kv_dim;

        if (kv_config_.type_k == KVCacheDType::FP32) {
            float* k_dst = page.key.fp32_row(offset);
            std::memcpy(k_dst, k_src, kv_dim * sizeof(float));
        } else {
            uint8_t* k_dst = page.key.q_row(offset);
            KVCache::quantize_row(kv_config_.type_k, k_src, k_dst, kv_dim);
        }

        if (kv_config_.type_v == KVCacheDType::FP32) {
            float* v_dst = page.value.fp32_row(offset);
            std::memcpy(v_dst, v_src, kv_dim * sizeof(float));
        } else {
            uint8_t* v_dst = page.value.q_row(offset);
            KVCache::quantize_row(kv_config_.type_v, v_src, v_dst, kv_dim);
        }

        // Update filled count
        if (offset + 1 > (int)page.filled) {
            page.filled = offset + 1;
        }
    }

    // Update sequence table
    table.logical_len = std::max(table.logical_len, pos + seq_len);
    int last_page_idx = (int)((table.logical_len - 1) / page_size_);
    if (last_page_idx < (int)table.page_ids.size()) {
        table.filled_in_last = (int)(table.logical_len - (int64_t)last_page_idx * page_size_);
    }

    return true;
}

TensorPtr PagedKVStorage::read_key(int layer) const {
    std::lock_guard<std::mutex> lock(mutex_);

    // CUDA paged decode reads KV through the page table directly (no materialize).
    if (device_ == DeviceType::CUDA)
        return nullptr;

    if (layer < 0 || layer >= num_layers_)
        return nullptr;

    int kv_dim = (layer < (int)kv_dims_.size()) ? kv_dims_[layer] : kv_dims_[0];

    // Calculate total filled length across all sequences
    int total_filled = 0;
    for (const auto& [seq_id, tables] : seq_tables_) {
        if (layer < (int)tables.size()) {
            total_filled += (int)tables[layer].logical_len;
        }
    }

    if (total_filled == 0)
        return nullptr;

    // Gather all sequences' K data into a contiguous FP32 tensor
    auto out = std::make_shared<Tensor>(DataType::FP32,
                                         std::vector<int64_t>{total_filled, kv_dim},
                                         DeviceType::CPU);
    float* dst = static_cast<float*>(out->data());

    int row = 0;
    for (const auto& [seq_id, tables] : seq_tables_) {
        if (layer >= (int)tables.size())
            continue;
        const auto& table = tables[layer];
        for (int p = 0; p < (int)table.page_ids.size(); ++p) {
            int page_id = table.page_ids[p];
            const auto& page = layer_pages_[layer][page_id];
            int rows_in_page = (p == (int)table.page_ids.size() - 1)
                                 ? table.filled_in_last
                                 : page_size_;
            for (int r = 0; r < rows_in_page; ++r) {
                if (kv_config_.type_k == KVCacheDType::FP32) {
                    const float* src = page.key.fp32_row(r);
                    std::memcpy(dst + row * kv_dim, src, kv_dim * sizeof(float));
                } else {
                    const uint8_t* src = page.key.q_row(r);
                    KVCache::dequantize_row(kv_config_.type_k, src, dst + row * kv_dim, kv_dim);
                }
                ++row;
            }
        }
    }

    return out;
}

TensorPtr PagedKVStorage::read_value(int layer) const {
    std::lock_guard<std::mutex> lock(mutex_);

    // CUDA paged decode reads KV through the page table directly (no materialize).
    if (device_ == DeviceType::CUDA)
        return nullptr;

    if (layer < 0 || layer >= num_layers_)
        return nullptr;

    int kv_dim = (layer < (int)kv_dims_.size()) ? kv_dims_[layer] : kv_dims_[0];

    // Calculate total filled length across all sequences
    int total_filled = 0;
    for (const auto& [seq_id, tables] : seq_tables_) {
        if (layer < (int)tables.size()) {
            total_filled += (int)tables[layer].logical_len;
        }
    }

    if (total_filled == 0)
        return nullptr;

    // Gather all sequences' V data into a contiguous FP32 tensor
    auto out = std::make_shared<Tensor>(DataType::FP32,
                                         std::vector<int64_t>{total_filled, kv_dim},
                                         DeviceType::CPU);
    float* dst = static_cast<float*>(out->data());

    int row = 0;
    for (const auto& [seq_id, tables] : seq_tables_) {
        if (layer >= (int)tables.size())
            continue;
        const auto& table = tables[layer];
        for (int p = 0; p < (int)table.page_ids.size(); ++p) {
            int page_id = table.page_ids[p];
            const auto& page = layer_pages_[layer][page_id];
            int rows_in_page = (p == (int)table.page_ids.size() - 1)
                                 ? table.filled_in_last
                                 : page_size_;
            for (int r = 0; r < rows_in_page; ++r) {
                if (kv_config_.type_v == KVCacheDType::FP32) {
                    const float* src = page.value.fp32_row(r);
                    std::memcpy(dst + row * kv_dim, src, kv_dim * sizeof(float));
                } else {
                    const uint8_t* src = page.value.q_row(r);
                    KVCache::dequantize_row(kv_config_.type_v, src, dst + row * kv_dim, kv_dim);
                }
                ++row;
            }
        }
    }

    return out;
}

int PagedKVStorage::filled(int layer) const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (layer < 0 || layer >= num_layers_)
        return 0;

    int total = 0;
    for (const auto& [seq_id, tables] : seq_tables_) {
        if (layer < (int)tables.size()) {
            total += (int)tables[layer].logical_len;
        }
    }
    return total;
}

// Per-sequence read: gather only the specified sequence's KV pages.
// This avoids duplicating data when pages are shared across sequences
// (e.g., prefix cache sharing via seq_share).
TensorPtr PagedKVStorage::read_key_seq(int layer, int seq_id) const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (layer < 0 || layer >= num_layers_)
        return nullptr;

    auto it = seq_tables_.find(seq_id);
    if (it == seq_tables_.end() || layer >= (int)it->second.size())
        return nullptr;

    const auto& table = it->second[layer];
    int logical_len = (int)table.logical_len;
    if (logical_len == 0)
        return nullptr;

    int kv_dim = (layer < (int)kv_dims_.size()) ? kv_dims_[layer] : kv_dims_[0];
    auto out = std::make_shared<Tensor>(DataType::FP32,
                                         std::vector<int64_t>{logical_len, kv_dim},
                                         DeviceType::CPU);
    float* dst = static_cast<float*>(out->data());

    int row = 0;
    for (int p = 0; p < (int)table.page_ids.size(); ++p) {
        int page_id = table.page_ids[p];
        const auto& page = layer_pages_[layer][page_id];
        int rows_in_page = (p == (int)table.page_ids.size() - 1)
                             ? table.filled_in_last
                             : page_size_;
        for (int r = 0; r < rows_in_page; ++r) {
            if (kv_config_.type_k == KVCacheDType::FP32) {
                const float* src = page.key.fp32_row(r);
                std::memcpy(dst + row * kv_dim, src, kv_dim * sizeof(float));
            } else {
                const uint8_t* src = page.key.q_row(r);
                KVCache::dequantize_row(kv_config_.type_k, src, dst + row * kv_dim, kv_dim);
            }
            ++row;
        }
    }

    return out;
}

TensorPtr PagedKVStorage::read_value_seq(int layer, int seq_id) const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (layer < 0 || layer >= num_layers_)
        return nullptr;

    auto it = seq_tables_.find(seq_id);
    if (it == seq_tables_.end() || layer >= (int)it->second.size())
        return nullptr;

    const auto& table = it->second[layer];
    int logical_len = (int)table.logical_len;
    if (logical_len == 0)
        return nullptr;

    int kv_dim = (layer < (int)kv_dims_.size()) ? kv_dims_[layer] : kv_dims_[0];
    auto out = std::make_shared<Tensor>(DataType::FP32,
                                         std::vector<int64_t>{logical_len, kv_dim},
                                         DeviceType::CPU);
    float* dst = static_cast<float*>(out->data());

    int row = 0;
    for (int p = 0; p < (int)table.page_ids.size(); ++p) {
        int page_id = table.page_ids[p];
        const auto& page = layer_pages_[layer][page_id];
        int rows_in_page = (p == (int)table.page_ids.size() - 1)
                             ? table.filled_in_last
                             : page_size_;
        for (int r = 0; r < rows_in_page; ++r) {
            if (kv_config_.type_v == KVCacheDType::FP32) {
                const float* src = page.value.fp32_row(r);
                std::memcpy(dst + row * kv_dim, src, kv_dim * sizeof(float));
            } else {
                const uint8_t* src = page.value.q_row(r);
                KVCache::dequantize_row(kv_config_.type_v, src, dst + row * kv_dim, kv_dim);
            }
            ++row;
        }
    }

    return out;
}

int PagedKVStorage::seq_filled(int layer, int seq_id) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = seq_tables_.find(seq_id);
    if (it == seq_tables_.end())
        return 0;
    if (layer < 0 || layer >= (int)it->second.size())
        return 0;
    return (int)it->second[layer].logical_len;
}

bool PagedKVStorage::seq_remove(int seq_id, int64_t p0, int64_t p1) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = seq_tables_.find(seq_id);
    if (it == seq_tables_.end())
        return false;

    // For paged storage, seq_remove removes the entire sequence's pages
    // in the [p0, p1) range. Since pages are per-sequence, we remove pages
    // that fall entirely within [p0, p1) and decrement refcount.
    // Pages shared with other sequences (via seq_share) are only freed when
    // all references are gone.

    for (int l = 0; l < num_layers_; ++l) {
        auto& table = it->second[l];

        // Calculate which pages to remove
        int first_page = (int)(p0 / page_size_);
        int last_page = (int)((p1 - 1) / page_size_);

        std::vector<int32_t> remaining_pages;
        for (int p = 0; p < (int)table.page_ids.size(); ++p) {
            int page_id = table.page_ids[p];
            if (p >= first_page && p <= last_page) {
                // This page is in the removal range — decrement refcount
                auto& page = layer_pages_[l][page_id];
                if (page.ref_count > 0) {
                    page.ref_count--;
                    if (page.ref_count == 0) {
                        free_page(l, page_id);
                    }
                }
            } else {
                remaining_pages.push_back(page_id);
            }
        }

        table.page_ids = remaining_pages;
        // Recalculate logical_len
        if (table.page_ids.empty()) {
            table.logical_len = 0;
            table.filled_in_last = 0;
        } else {
            // Approximate: logical_len = page_ids.size() * page_size
            // (precise recalculation is complex; this is sufficient for removal)
            table.logical_len = std::min(table.logical_len, p0);
            int last_idx = (int)table.page_ids.size() - 1;
            table.filled_in_last = (int)(table.logical_len - (int64_t)last_idx * page_size_);
            if (table.filled_in_last <= 0) {
                table.filled_in_last = page_size_;
            }
        }
    }

    return true;
}

bool PagedKVStorage::seq_share(int src_seq, int dst_seq, int64_t p0, int64_t p1) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto src_it = seq_tables_.find(src_seq);
    if (src_it == seq_tables_.end()) {
        LOG_ERROR("PagedKVStorage::seq_share: src_seq " + std::to_string(src_seq) + " not found");
        return false;
    }

    init_seq(dst_seq);

    // Copy page table entries from src to dst for the [p0, p1) range.
    // No KV data is copied — only page references are shared.
    // ref_count is incremented on shared pages.

    int first_page = (int)(p0 / page_size_);
    int last_page = (int)((p1 - 1) / page_size_);

    for (int l = 0; l < num_layers_; ++l) {
        const auto& src_table = src_it->second[l];
        auto& dst_table = seq_tables_[dst_seq][l];

        for (int p = first_page; p <= last_page && p < (int)src_table.page_ids.size(); ++p) {
            int page_id = src_table.page_ids[p];
            dst_table.page_ids.push_back(page_id);
            // Increment refcount — page is now shared
            layer_pages_[l][page_id].ref_count++;
        }

        dst_table.logical_len = std::max(dst_table.logical_len, p1 - p0);
        int last_idx = (int)dst_table.page_ids.size() - 1;
        if (last_idx >= 0) {
            dst_table.filled_in_last = (int)(dst_table.logical_len - (int64_t)last_idx * page_size_);
            if (dst_table.filled_in_last <= 0 || dst_table.filled_in_last > page_size_) {
                dst_table.filled_in_last = page_size_;
            }
        }
    }

    return true;
}

bool PagedKVStorage::seq_keep(int seq_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    // In paged mode, each sequence has its own pages, so seq_keep is essentially a no-op.
    // Pages are per-sequence; there's no cross-sequence contamination to clean up.
    // Only shared pages (via seq_share) need refcount management, which is handled
    // by seq_remove/release.
    (void)seq_id;
    return true;
}

void PagedKVStorage::release(int seq_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = seq_tables_.find(seq_id);
    if (it == seq_tables_.end())
        return;

    // Free all pages owned by this sequence
    for (int l = 0; l < num_layers_; ++l) {
        auto& table = it->second[l];
        for (int page_id : table.page_ids) {
            auto& page = layer_pages_[l][page_id];
            if (page.ref_count > 0) {
                page.ref_count--;
                if (page.ref_count == 0) {
                    free_page(l, page_id);
                }
            }
        }
    }

    seq_tables_.erase(it);
}

bool PagedKVStorage::has_seq(int seq_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return seq_tables_.find(seq_id) != seq_tables_.end();
}

size_t PagedKVStorage::nbytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t total = 0;
    for (int l = 0; l < num_layers_; ++l) {
        for (int p = 0; p < layer_pools_[l].max_pages; ++p) {
            total += layer_pages_[l][p].key.capacity_bytes();
            total += layer_pages_[l][p].value.capacity_bytes();
        }
    }
    return total;
}

size_t PagedKVStorage::active_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t total = 0;
    for (int l = 0; l < num_layers_; ++l) {
        for (const auto& [seq_id, tables] : seq_tables_) {
            if (l < (int)tables.size()) {
                for (int page_id : tables[l].page_ids) {
                    total += layer_pages_[l][page_id].key.capacity_bytes();
                    total += layer_pages_[l][page_id].value.capacity_bytes();
                }
            }
        }
    }
    return total;
}

int PagedKVStorage::num_free_slots() const {
    std::lock_guard<std::mutex> lock(mutex_);
    int total = 0;
    for (const auto& pool : layer_pools_)
        total += (int)pool.free_page_ids.size() * page_size_;
    return total;
}

int PagedKVStorage::num_free_pages() const {
    std::lock_guard<std::mutex> lock(mutex_);
    int total = 0;
    for (const auto& pool : layer_pools_)
        total += (int)pool.free_page_ids.size();
    return total;
}

void PagedKVStorage::reset() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Clear all sequence tables
    seq_tables_.clear();

    // Phase 6: return all pages to each layer's free list and reset metadata
    for (int l = 0; l < num_layers_; ++l) {
        auto& pool = layer_pools_[l];
        pool.free_page_ids.clear();
        pool.free_page_set.clear();
        for (int p = 0; p < pool.max_pages; ++p) {
            pool.free_page_ids.push_back(p);
            pool.free_page_set.insert(p);
            layer_pages_[l][p].filled = 0;
            layer_pages_[l][p].ref_count = 0;
        }
    }
    // Note: device page pointer tables are stable across reset() (the page
    // pool itself is not freed); only the per-sequence page_ids are cleared.
    // d_seq_page_ids_ is reused as-is (overwritten on next upload).
}

// =========================================================================
// Paged CUDA accessors (Phase 4)
// =========================================================================

void* const* PagedKVStorage::d_key_page_ptrs(int layer) const {
    if (device_ != DeviceType::CUDA || layer < 0 || layer >= (int)d_key_page_ptrs_.size())
        return nullptr;
    return d_key_page_ptrs_[layer];
}

void* const* PagedKVStorage::d_value_page_ptrs(int layer) const {
    if (device_ != DeviceType::CUDA || layer < 0 || layer >= (int)d_value_page_ptrs_.size())
        return nullptr;
    return d_value_page_ptrs_[layer];
}

const int32_t* PagedKVStorage::upload_seq_page_table(int layer, int seq_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    return upload_seq_page_table_unlocked(layer, seq_id);
}

const int32_t* PagedKVStorage::upload_seq_page_table_unlocked(int layer, int seq_id) {
    auto it = seq_tables_.find(seq_id);
    if (it == seq_tables_.end() || layer < 0 || layer >= (int)it->second.size())
        return nullptr;
    const auto& page_ids = it->second[layer].page_ids;
    int n = (int)page_ids.size();
    if (n == 0)
        return nullptr;
#ifdef USE_CUDA
    if (device_ != DeviceType::CUDA)
        return nullptr;
    if (n > d_seq_page_ids_cap_) {
        if (d_seq_page_ids_)
            cudaFree(d_seq_page_ids_);
        d_seq_page_ids_cap_ = n * 2;  // grow-only
        cudaMalloc(&d_seq_page_ids_, (size_t)d_seq_page_ids_cap_ * sizeof(int32_t));
    }
    cudaMemcpy(d_seq_page_ids_, page_ids.data(), (size_t)n * sizeof(int32_t),
               cudaMemcpyHostToDevice);
    return d_seq_page_ids_;
#else
    return nullptr;
#endif
}

}  // namespace forge
