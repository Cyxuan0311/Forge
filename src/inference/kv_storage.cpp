#include "forge/kv_storage.h"

#include <algorithm>
#include <cstring>

#include "forge/cuda_kernels.h"
#include "forge/kv_sizing.h"
#include "forge/logger.h"

#ifdef USE_CUDA
#    include <cuda_runtime.h>
#endif

namespace forge {

namespace {
#ifdef USE_CUDA
// Phase 2-D: free the persistent device page-table mirrors of one sequence.
// Mirrors are owned by PagedKVStorage and never outlive a host table entry.
void free_page_table_mirrors(std::vector<SequencePageTable>& tables) {
    for (auto& t : tables) {
        if (t.d_mirror) {
            cudaFree(t.d_mirror);
            t.d_mirror = nullptr;
            t.d_cap = 0;
            t.d_synced = 0;
            t.d_dirty = true;
        }
    }
}
#endif
}  // namespace

// =========================================================================
// ContiguousKVStorage
// =========================================================================

ContiguousKVStorage::ContiguousKVStorage(KVCache& cache) : cache_(cache) {}

bool ContiguousKVStorage::init(int /*num_layers*/, const std::vector<int>& /*kv_dims*/,
                               int /*max_seq_len*/, DeviceType /*device*/,
                               const KVCacheTypeConfig& /*kv_config*/, int /*page_size*/,
                               int /*max_num_seqs*/) {
    // Contiguous mode: KVCache is already initialized by the engine.
    return true;
}

bool ContiguousKVStorage::write_kv(int layer, int64_t pos, int seq_len, const float* k_data,
                                   const float* v_data) {
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
    defrag_if_needed();  // Roadmap 2.2: compact holes left by the removal
    return true;
}

void ContiguousKVStorage::defrag() {
    cache_.defrag();
}

void ContiguousKVStorage::defrag_if_needed() {
    cache_.defrag_if_needed();
}

bool ContiguousKVStorage::seq_share(int src_seq, int dst_seq, int64_t p0, int64_t p1) {
    if (src_seq < 0 || src_seq >= cache_.max_seqs() || dst_seq < 0 ||
        dst_seq >= cache_.max_seqs()) {
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
    defrag_if_needed();  // Roadmap 2.2: compact after a sequence exits
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
    free_pinned_blocks_();
#ifdef USE_CUDA
    for (int l = 0; l < num_layers_; ++l) {
        if (l < (int)d_key_page_ptrs_.size() && d_key_page_ptrs_[l])
            cudaFree(d_key_page_ptrs_[l]);
        if (l < (int)d_value_page_ptrs_.size() && d_value_page_ptrs_[l])
            cudaFree(d_value_page_ptrs_[l]);
    }
    for (auto& kv : seq_tables_)
        free_page_table_mirrors(kv.second);
    // The slabs own every page's bytes. The per-page KVCacheStorage views do not
    // free them (owns_buffer() == false), so this is the only place they are
    // released -- without it the pool leaks ~1.9 GB per process.
    free_layer_slabs_();
#endif
}

bool PagedKVStorage::init(int num_layers, const std::vector<int>& kv_dims, int max_seq_len,
                          DeviceType device, const KVCacheTypeConfig& kv_config, int page_size,
                          int max_num_seqs) {
    std::lock_guard<std::mutex> lock(mutex_);

    // ---- Re-init guard (vLLM: size once, never shrink) ----
    // A second init() happens when warmup's reset_kv_cache() clears the
    // initialized flag and the first real request rebuilds the memory. At that
    // point the freshly measured free memory is an artifact of the EXISTING
    // pages still being allocated (observed: budget 264 MB -> 221 pages while
    // the live pool held 1929), so shrinking here would cripple the pool for
    // every subsequent long prompt. If the geometry is unchanged, keep the
    // larger of old and new page caps; the pages themselves are still
    // allocated (reset() returned them to the free lists), so an unchanged
    // target needs no reallocation at all.
    const bool reinit_same_geometry =
        !layer_pages_.empty() && layer_pages_[0].size() > 0 && num_layers_ == num_layers &&
        max_seq_len_ == max_seq_len && device_ == device && page_size_ == (page_size > 0 ? page_size : 16) &&
        kv_config_.type_k == kv_config.type_k && kv_config_.type_v == kv_config.type_v;
    const int old_max_seqs = max_num_seqs_;

    num_layers_ = num_layers;
    kv_dims_ = kv_dims;
    max_seq_len_ = max_seq_len;
    device_ = device;
    kv_config_ = kv_config;
    page_size_ = page_size > 0 ? page_size : 16;

    // ---- Roadmap 1.2: auto-capacity from the device memory budget ----
    // max_num_seqs <= 0 means "auto": derive both the concurrency limit and the
    // per-layer page count from the free device memory. A positive value is an
    // explicit override and keeps the historical behaviour exactly.
    int auto_max_pages = 0;
    if (max_num_seqs > 0) {
        max_num_seqs_ = max_num_seqs;
    } else {
        KVSizingParams sp;
        sp.num_layers = num_layers;
        sp.kv_dims = kv_dims;
        sp.max_seq_len = max_seq_len;
        sp.page_size = page_size_;
        sp.device = device;
        sp.type_k = kv_config.type_k;
        sp.type_v = kv_config.type_v;
        sp.policies.resize(num_layers);
        for (int l = 0; l < num_layers; ++l) {
            sp.policies[l] = (l < static_cast<int>(layer_pools_.size())) ? layer_pools_[l].policy
                                                                         : KVLayerPolicy::Full;
            if (sp.policies[l] == KVLayerPolicy::SlidingWindow)
                sp.swa_window = std::max(sp.swa_window, layer_pools_[l].window_size);
        }

        // Same conservative share of free memory as resize_to_budget() (see the
        // comment there). Without it init takes ~90% of free VRAM -- 2.28 GB on
        // a 6 GB card -- leaving vram_free=0: warmup can then no longer grow the
        // pool ("no device budget available") and every activation allocation in
        // the forward pass has to fight the allocator, which showed up as decode
        // steps degrading 18 ms -> 541 ms while the context only grew 7%.
        const KVSizingResult sizing = auto_size_kv_for_device(sp, 0.65);
        if (sizing.auto_sized) {
            max_num_seqs_ = sizing.max_num_seqs;
            auto_max_pages = sizing.max_pages_per_layer;
            LOG_INFO("PagedKVStorage auto-sizing: per_seq=" +
                     std::to_string(sizing.per_seq_bytes / (1024 * 1024)) +
                     " MB, budget=" + std::to_string(sizing.budget_bytes / (1024 * 1024)) +
                     " MB -> max_seqs=" + std::to_string(max_num_seqs_) +
                     ", max_pages=" + std::to_string(auto_max_pages) +
                     (reinit_same_geometry ? " (re-init; existing pool may be larger)" : ""));
        } else {
            // No usable device free-memory information (CPU backend, or the query
            // returned nothing e.g. a headless/background CUDA context). Size for a
            // small number of sequences so the pool can still hold real prompts;
            // the small init pool keeps VRAM free for warmup, and warmup's
            // resize_to_budget() grows the pools once the real budget is known.
            max_num_seqs_ = 2;
        }
    }

    // Phase 6: per-layer pool sizing.
    // If layer_pools_ was pre-configured via set_layer_policies(), use those sizes.
    // Otherwise, default all layers to Full with uniform max_pages.
    if (layer_pools_.empty()) {
        layer_pools_.assign(num_layers_, LayerPoolInfo{});
        for (auto& p : layer_pools_)
            p.policy = KVLayerPolicy::Full;
    } else if ((int)layer_pools_.size() < num_layers_) {
        layer_pools_.resize(num_layers_, LayerPoolInfo{});
    }
    // vLLM-style pool sizing. The page pool below is FULLY allocated up front
    // (one cudaMalloc per page, see allocate_page_pools_locked), so the cap
    // must come from the device memory budget — never from max_seq_len.
    // vLLM does the same: num_gpu_blocks derives from post-profile free
    // memory, while max_model_len only validates individual requests.
    // Sizing the pool to max_num_seqs * max_seq_len (the historical
    // "fallback_pages") made a 256K-native model try to reserve ~16384 pages
    // per layer and OOM on a 6 GB card.
    //
    // The budget query runs before warmup, when free VRAM is transiently low
    // (observed: 330 MB vs 2742 MB after warmup), so auto_max_pages can be 0
    // or tiny. init therefore reserves only a small floor (enough for warmup
    // and short prompts); InferenceContext::warmup() calls resize_to_budget()
    // afterwards — vLLM's profile_run → determine_num_available_blocks
    // sequence — to grow the pools to the real budget.
    constexpr int kMinPagedPages = 64;  // 1024 tokens @ page_size 16
    int page_cap;
    if (device_ == DeviceType::CPU) {
        // Host pages are cheap and the auto-sizing query returns no budget for
        // CPU (query_free_device_bytes is CUDA-only). Size the pool to hold
        // each admitted sequence at a practical 8K-token length instead of the
        // model's full (possibly 256K) native context — no artificial floor.
        const int cpu_floor = (max_num_seqs_ * std::min(max_seq_len_, 8192) + page_size_ - 1) /
                              page_size_;
        page_cap = std::max(auto_max_pages, cpu_floor);
    } else {
        page_cap = std::max(auto_max_pages, kMinPagedPages);
    }
    // KVMem pool cap: the GPU pool only has to hold the working set
    // (sink + budget + generation reserve), never the logical max_seq_len.
    if (pool_cap_tokens_ > 0) {
        int cap_pages = (pool_cap_tokens_ + page_size_ - 1) / page_size_;
        if (cap_pages < page_cap)
            page_cap = cap_pages;
    }
    // Re-init: never shrink below the existing pool (see comment at function top).
    if (reinit_same_geometry) {
        max_num_seqs_ = std::max(max_num_seqs_, old_max_seqs);
        int old_cap = 0;
        for (const auto& p : layer_pools_)
            old_cap = std::max(old_cap, p.max_pages);
        page_cap = std::max(page_cap, old_cap);
    }
    bool unchanged = reinit_same_geometry;
    for (int l = 0; l < num_layers_; ++l) {
        auto& pool = layer_pools_[l];
        if (pool.policy == KVLayerPolicy::SlidingWindow && pool.window_size > 0) {
            int swa_pages = (pool.window_size + page_size_ - 1) / page_size_;
            // SWA needs a few extra pages for the wrap-around cursor
            pool.max_pages = std::min(swa_pages + 1, page_cap);
        } else {
            pool.max_pages = page_cap;
        }
        if (unchanged && pool.max_pages != layer_pages_[l].size())
            unchanged = false;
    }
    if (unchanged) {
        // Same geometry and same per-layer pool sizes: the pages are still
        // allocated and reset() has returned them all to the free lists —
        // skip the (multi-GB) zero_fill re-allocation entirely.
        LOG_INFO("PagedKVStorage re-init: geometry unchanged, keeping " +
                 std::to_string(max_pages_) + " pages/layer without reallocation");
        return true;
    }
    // max_pages_ = max across layers (for backward-compat in some log messages)
    max_pages_ = 0;
    for (const auto& p : layer_pools_)
        max_pages_ = std::max(max_pages_, p.max_pages);

    if (!allocate_page_pools_locked())
        return false;

    LOG_INFO("PagedKVStorage init: layers=" + std::to_string(num_layers_) + ", page_size=" +
             std::to_string(page_size_) + ", max_pages=" + std::to_string(max_pages_) +
             ", max_seqs=" + std::to_string(max_num_seqs_) +
             ", dev=" + (device_ == DeviceType::CUDA ? "CUDA" : "CPU"));
    return true;
}

void PagedKVStorage::free_layer_slabs_() {
#ifdef USE_CUDA
    for (void* p : layer_slab_key_) cudaFree(p);
    for (void* p : layer_slab_value_) cudaFree(p);
#endif
    layer_slab_key_.clear();
    layer_slab_value_.clear();
}

bool PagedKVStorage::allocate_page_pools_locked() {
    // Re-entrant: resize_to_budget() calls this again after warmup, when free
    // VRAM is known. Release the previous slabs first, otherwise each resize
    // strands a full pool's worth of device memory and the next budget query sees
    // less free memory than it should.
    free_layer_slabs_();

    // Allocate page pools per layer
    layer_pages_.resize(num_layers_);
    for (int l = 0; l < num_layers_; ++l) {
        int kv_dim = (l < (int)kv_dims_.size()) ? kv_dims_[l] : kv_dims_[0];
        size_t k_row_bytes = KVCache::block_nbytes(kv_config_.type_k, kv_dim);
        size_t v_row_bytes = KVCache::block_nbytes(kv_config_.type_v, kv_dim);

        int lmax = layer_pools_[l].max_pages;
        layer_pages_[l].resize(lmax);

        // One slab per (layer, side) instead of one allocation per page.
        //
        // The page count is budget-derived, not context-derived: at -c 4096 on a
        // 6 GB card the q4_0 pool came out at 10,496 pages/layer, so 42 layers x
        // 2 sides = 881,664 separate cudaMalloc calls. That cost ~60 s of startup
        // (ctx=60865 ms) at roughly 68 us apiece -- the single largest source of
        // paged startup latency, and it scales with free VRAM rather than with
        // the work being done.
        //
        // The pages are already addressed indirectly through
        // d_key_page_ptrs_/d_value_page_ptrs_, so a slab needs no change to the
        // page-table indirection at all: each page just points at its slice.
        // 881,664 calls become 2 per layer.
        void* k_slab = nullptr;
        void* v_slab = nullptr;
        size_t k_slab_bytes = 0, v_slab_bytes = 0;
        if (device_ == DeviceType::CUDA && lmax > 0) {
#ifdef USE_CUDA
            k_slab_bytes = static_cast<size_t>(lmax) * page_size_ * k_row_bytes;
            v_slab_bytes = static_cast<size_t>(lmax) * page_size_ * v_row_bytes;
            if (cudaMalloc(&k_slab, k_slab_bytes) != cudaSuccess) k_slab = nullptr;
            if (cudaMalloc(&v_slab, v_slab_bytes) != cudaSuccess) v_slab = nullptr;
            if (k_slab) cudaMemset(k_slab, 0, k_slab_bytes);
            if (v_slab) cudaMemset(v_slab, 0, v_slab_bytes);
#endif
        }

        for (int p = 0; p < lmax; ++p) {
            auto& page = layer_pages_[l][p];
            const size_t slice_k = static_cast<size_t>(page_size_) * k_row_bytes;
            const size_t slice_v = static_cast<size_t>(page_size_) * v_row_bytes;
            if (k_slab && v_slab) {
                page.key.adopt_device_buffer(kv_config_.type_k,
                                             static_cast<uint8_t*>(k_slab) +
                                                 static_cast<size_t>(p) * slice_k,
                                             slice_k, k_row_bytes);
                page.value.adopt_device_buffer(kv_config_.type_v,
                                               static_cast<uint8_t*>(v_slab) +
                                                   static_cast<size_t>(p) * slice_v,
                                               slice_v, v_row_bytes);
            } else {
                // CPU pools, or a CUDA slab that failed to allocate: fall back to
                // per-page storage so behaviour degrades rather than breaks.
                page.key.alloc(kv_config_.type_k, device_, page_size_, k_row_bytes);
                page.value.alloc(kv_config_.type_v, device_, page_size_, v_row_bytes);
                page.key.zero_fill();
                page.value.zero_fill();
            }
            page.filled = 0;
            page.ref_count = 0;
        }
        if (k_slab) {
            layer_slab_key_.push_back(k_slab);
            layer_slab_value_.push_back(v_slab);
        }
        // Init free list for this layer
        layer_pools_[l].free_page_ids.clear();
        layer_pools_[l].free_page_set.clear();
        for (int p = 0; p < lmax; ++p) {
            layer_pools_[l].free_page_ids.push_back(p);
            layer_pools_[l].free_page_set.insert(p);
        }
    }

#ifdef USE_CUDA
    for (auto& kv : seq_tables_)
        free_page_table_mirrors(kv.second);
#endif
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
            if (d_key_page_ptrs_[l])
                cudaFree(d_key_page_ptrs_[l]);
        for (int l = 0; l < (int)d_value_page_ptrs_.size(); ++l)
            if (d_value_page_ptrs_[l])
                cudaFree(d_value_page_ptrs_[l]);

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
                d_key_page_ptrs_host_[l][p] = (kv_config_.type_k == KVCacheDType::FP32)
                                                  ? static_cast<void*>(page.key.fp32_data())
                                                  : page.key.d_q_data();
                d_value_page_ptrs_host_[l][p] = (kv_config_.type_v == KVCacheDType::FP32)
                                                    ? static_cast<void*>(page.value.fp32_data())
                                                    : page.value.d_q_data();
            }
            cudaMalloc(&d_key_page_ptrs_[l], lmax * sizeof(void*));
            cudaMemcpy(d_key_page_ptrs_[l], d_key_page_ptrs_host_[l].data(), lmax * sizeof(void*),
                       cudaMemcpyHostToDevice);
            cudaMalloc(&d_value_page_ptrs_[l], lmax * sizeof(void*));
            cudaMemcpy(d_value_page_ptrs_[l], d_value_page_ptrs_host_[l].data(),
                       lmax * sizeof(void*), cudaMemcpyHostToDevice);
        }
    }
#endif

    LOG_INFO("PagedKVStorage page pools allocated: layers=" + std::to_string(num_layers_) +
             ", page_size=" + std::to_string(page_size_) + ", max_pages=" +
             std::to_string(max_pages_) + ", max_seqs=" + std::to_string(max_num_seqs_) +
             ", dev=" + (device_ == DeviceType::CUDA ? "CUDA" : "CPU"));
    return true;
}

bool PagedKVStorage::resize_to_budget() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (num_layers_ <= 0 || layer_pools_.empty())
        return false;

    // Re-derive the per-layer page budget from the CURRENT device free memory.
    // Same params as init(); layer_pools_ (policies/windows) is unchanged, so
    // SWA pools keep their window-derived sizes below.
    KVSizingParams sp;
    sp.num_layers = num_layers_;
    sp.kv_dims = kv_dims_;
    sp.max_seq_len = max_seq_len_;
    sp.page_size = page_size_;
    sp.device = device_;
    sp.type_k = kv_config_.type_k;
    sp.type_v = kv_config_.type_v;
    sp.policies.resize(num_layers_);
    for (int l = 0; l < num_layers_; ++l)
        sp.policies[l] = layer_pools_[l].policy;
    for (int l = 0; l < num_layers_ && l < (int)sp.policies.size(); ++l) {
        if (sp.policies[l] == KVLayerPolicy::SlidingWindow)
            sp.swa_window = std::max(sp.swa_window, layer_pools_[l].window_size);
    }

    // Fraction of the post-warmup free memory handed to the page pool.
    // vLLM's gpu_memory_utilization is a share of TOTAL memory and the KV gets
    // only what remains after weights + activations; here the queried "free"
    // already excludes the weights, so taking 0.9 of it left just ~10%
    // (≈505 MB on this 6 GB card) for activations and workspace — while the
    // execution plan needs a 741 MB scratch — and the request died mid-flight
    // with "CUDA malloc failed". Reserve a third of free memory for the
    // forward pass instead.
    constexpr double kKvShareOfFree = 0.65;
    const KVSizingResult sizing = auto_size_kv_for_device(sp, kKvShareOfFree);
    if (!sizing.auto_sized) {
        LOG_INFO("PagedKVStorage resize_to_budget: no device budget available, keeping " +
                 std::to_string(max_pages_) + " pages/layer");
        return true;
    }

    int new_cap = sizing.max_pages_per_layer;
    // KVMem pool cap (see init()).
    if (pool_cap_tokens_ > 0) {
        int cap_pages = (pool_cap_tokens_ + page_size_ - 1) / page_size_;
        new_cap = std::min(new_cap, cap_pages);
    }
    // Never shrink: the pool may already back live traffic, and growing is the
    // only post-init adjustment this path makes (vLLM sizes once, we grow once).
    int cur_cap = 0;
    for (const auto& p : layer_pools_)
        cur_cap = std::max(cur_cap, p.max_pages);
    if (new_cap <= cur_cap) {
        LOG_INFO("PagedKVStorage resize_to_budget: budget " + std::to_string(new_cap) +
                 " pages <= current " + std::to_string(cur_cap) + ", keeping current");
        return true;
    }

    for (int l = 0; l < num_layers_; ++l) {
        auto& pool = layer_pools_[l];
        if (pool.policy == KVLayerPolicy::SlidingWindow && pool.window_size > 0) {
            int swa_pages = (pool.window_size + page_size_ - 1) / page_size_;
            pool.max_pages = std::min(swa_pages + 1, new_cap);
        } else {
            pool.max_pages = new_cap;
        }
    }
    max_pages_ = 0;
    for (const auto& p : layer_pools_)
        max_pages_ = std::max(max_pages_, p.max_pages);

    if (!allocate_page_pools_locked())
        return false;

    LOG_INFO("PagedKVStorage resize_to_budget: " + std::to_string(cur_cap) + " -> " +
             std::to_string(new_cap) + " pages/layer (" +
             std::to_string((long long)max_pages_ * page_size_) + " tokens capacity, budget " +
             std::to_string(sizing.budget_bytes / (1024 * 1024)) + " MB)");
    return true;
}

int PagedKVStorage::alloc_page(int layer) {
    if (layer < 0 || layer >= (int)layer_pools_.size())
        return -1;
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
    if (layer < 0 || layer >= (int)layer_pools_.size())
        return;
    if (page_id < 0 || page_id >= (int)layer_pages_[layer].size())
        return;
    auto& pool = layer_pools_[layer];
    if (pool.free_page_set.count(page_id))
        return;  // prevent double-enqueue
    auto& page = layer_pages_[layer][page_id];
    // Roadmap 1.1: a freed page gives up any host copy it still holds. An
    // evicted page's data lives in the swap pool; a restored page keeps its
    // host copy so a later re-eviction is cheap — but once the page is freed
    // for good, that copy must not leak. drop() is a no-op for pages that were
    // never offloaded.
    swapper_.drop(layer, page_id);
    page.evicted = false;
    pool.free_page_ids.push_back(page_id);
    pool.free_page_set.insert(page_id);
    page.ref_count = 0;
    page.filled = 0;
}

int PagedKVStorage::layer_max_pages(int layer) const {
    if (layer < 0 || layer >= (int)layer_pools_.size())
        return max_pages_;
    return layer_pools_[layer].max_pages;
}

void PagedKVStorage::set_layer_policies(const std::vector<KVLayerPolicy>& policies,
                                        int swa_window) {
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
        // Sticky arming: the working set is still attached to this sequence, so
        // a table re-created after reset()/release() must come back
        // window-active or the per-turn shrink in the generator silently skips.
        if (ws_armed_seqs_.count(seq_id)) {
            for (auto& t : seq_tables_[seq_id])
                t.window_active = true;
        }
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
            LOG_ERROR("PagedKVStorage: page allocation failed for seq=" + std::to_string(seq_id) +
                      " layer=" + std::to_string(layer) + ", rolled back " +
                      std::to_string(newly_allocated.size()) + " pages");
            return false;
        }
        newly_allocated.push_back(page_id);
        table.page_ids.push_back(page_id);
    }
    return true;
}

bool PagedKVStorage::write_kv(int layer, int64_t pos, int seq_len, const float* k_data,
                              const float* v_data) {
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

    // Phase 4-B/4-D: rows at/above pos may be rewritten by this call, so both
    // row mirrors must re-upload them before their next read. Rows below pos
    // are untouched; in particular an appending write (pos == logical_len)
    // leaves an up-to-date mirror alone, so decode only ever uploads its one
    // new row (on the device) or dequantizes it (on the host).
    if (table.cuda_mirror_synced > pos)
        table.cuda_mirror_synced = pos;
    if (table.host_mirror_synced > pos)
        table.host_mirror_synced = pos;

    // seq_window_archive_page releases a page's buffers and keeps the
    // slot->page mapping intact, so a write that lands on an archived slot (a
    // KVMem clip re-forwards the sink and tail over exactly those slots) must
    // bring the page's storage back first -- writing through the mapping would
    // otherwise dereference released memory. The restored bytes are immediately
    // overwritten; this only re-arms the allocation the row writes go through.
    {
        const int64_t first_page = pos / page_size_;
        const int64_t last_page = (pos + seq_len - 1) / page_size_;
        for (int64_t p = first_page; p <= last_page; ++p) {
            if (p >= static_cast<int64_t>(table.page_ids.size()))
                break;
            const int page_id = table.page_ids[p];
            if (page_id >= 0 && layer_pages_[layer][page_id].evicted &&
                !bring_back_unlocked(layer, page_id)) {
                LOG_ERROR("PagedKVStorage::write_kv_seq: failed to bring back "
                          "archived page " + std::to_string(page_id) +
                          " for seq=" + std::to_string(seq_id) +
                          " layer=" + std::to_string(layer));
                return false;
            }
        }
    }

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
        cuda::launch_kv_scatter(k_data, v_data, d_key_page_ptrs_[layer], d_value_page_ptrs_[layer],
                                d_page_ids, seq_len, pos, page_size_, kv_dim, k_row_bytes,
                                v_row_bytes, kv_config_.type_k, kv_config_.type_v, 0);

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
    auto out = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{total_filled, kv_dim},
                                        DeviceType::CPU);
    float* dst = static_cast<float*>(out->data());

    int row = 0;
    for (const auto& [seq_id, tables] : seq_tables_) {
        if (layer >= (int)tables.size())
            continue;
        const auto& table = tables[layer];
        // Bound the walk by logical_len: page_ids can cover more pages than the
        // visible length (a KVMem clip shrinks logical_len while archived pages
        // stay in the slot map), and materialising rows past the end would
        // overrun the output tensor -- heap corruption, not just stale data.
        int64_t remaining = table.logical_len;
        for (int p = 0; p < (int)table.page_ids.size() && remaining > 0; ++p) {
            int page_id = table.page_ids[p];
            const auto& page = layer_pages_[layer][page_id];
            const int rows_in_page =
                static_cast<int>(std::min<int64_t>(page_size_, remaining));
            remaining -= rows_in_page;
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
    auto out = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{total_filled, kv_dim},
                                        DeviceType::CPU);
    float* dst = static_cast<float*>(out->data());

    int row = 0;
    for (const auto& [seq_id, tables] : seq_tables_) {
        if (layer >= (int)tables.size())
            continue;
        const auto& table = tables[layer];
        // Bound the walk by logical_len: page_ids can cover more pages than the
        // visible length (a KVMem clip shrinks logical_len while archived pages
        // stay in the slot map), and materialising rows past the end would
        // overrun the output tensor -- heap corruption, not just stale data.
        int64_t remaining = table.logical_len;
        for (int p = 0; p < (int)table.page_ids.size() && remaining > 0; ++p) {
            int page_id = table.page_ids[p];
            const auto& page = layer_pages_[layer][page_id];
            const int rows_in_page =
                static_cast<int>(std::min<int64_t>(page_size_, remaining));
            remaining -= rows_in_page;
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

    // Host-side materialisation is only possible when the page data actually
    // lives on the host. For a CUDA-backed paged pool, q_row()/fp32_row() return
    // DEVICE pointers, and dequantize_row() below is a host function: it would
    // dereference device memory on the CPU and segfault. The rest of this class
    // already documents that CUDA paged storage cannot materialise rows
    // (read_key*/read_value* return null by design) -- this returns null so that
    // contract is actually honoured, and the caller picks a paged kernel or
    // reports that the combination is unsupported, instead of crashing.
    if (device_ == DeviceType::CUDA) return nullptr;

    int kv_dim = (layer < (int)kv_dims_.size()) ? kv_dims_[layer] : kv_dims_[0];
    auto out = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{logical_len, kv_dim},
                                        DeviceType::CPU);
    float* dst = static_cast<float*>(out->data());

    int row = 0;
    // Bound the walk by logical_len: page_ids can cover more pages than the
    // visible length (a KVMem clip shrinks logical_len while archived pages
    // stay in the slot map), and materialising rows past the end would
    // overrun the output tensor -- heap corruption, not just stale data.
    int64_t remaining = table.logical_len;
    for (int p = 0; p < (int)table.page_ids.size() && remaining > 0; ++p) {
        int page_id = table.page_ids[p];
        const auto& page = layer_pages_[layer][page_id];
        const int rows_in_page =
            static_cast<int>(std::min<int64_t>(page_size_, remaining));
        remaining -= rows_in_page;
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

    // Host-side materialisation is only possible when the page data actually
    // lives on the host. For a CUDA-backed paged pool, q_row()/fp32_row() return
    // DEVICE pointers, and dequantize_row() below is a host function: it would
    // dereference device memory on the CPU and segfault. The rest of this class
    // already documents that CUDA paged storage cannot materialise rows
    // (read_key*/read_value* return null by design) -- this returns null so that
    // contract is actually honoured, and the caller picks a paged kernel or
    // reports that the combination is unsupported, instead of crashing.
    if (device_ == DeviceType::CUDA) return nullptr;

    int kv_dim = (layer < (int)kv_dims_.size()) ? kv_dims_[layer] : kv_dims_[0];
    auto out = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{logical_len, kv_dim},
                                        DeviceType::CPU);
    float* dst = static_cast<float*>(out->data());

    int row = 0;
    // Bound the walk by logical_len: page_ids can cover more pages than the
    // visible length (a KVMem clip shrinks logical_len while archived pages
    // stay in the slot map), and materialising rows past the end would
    // overrun the output tensor -- heap corruption, not just stale data.
    int64_t remaining = table.logical_len;
    for (int p = 0; p < (int)table.page_ids.size() && remaining > 0; ++p) {
        int page_id = table.page_ids[p];
        const auto& page = layer_pages_[layer][page_id];
        const int rows_in_page =
            static_cast<int>(std::min<int64_t>(page_size_, remaining));
        remaining -= rows_in_page;
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

bool PagedKVStorage::seq_set_logical_len(int seq_id, int64_t len) {
    if (len < 0)
        return false;
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = seq_tables_.find(seq_id);
    if (it == seq_tables_.end())
        return false;

    bool any = false;
    for (int l = 0; l < (int)it->second.size(); ++l) {
        auto& table = it->second[l];
        if (len > table.logical_len)
            return false;  // never grows
        if (table.logical_len == len)
            continue;
        table.logical_len = len;
        // Keep filled_in_last consistent with the new length: it is how much of
        // the final page is valid, and a stale value would make the last-page
        // bound wrong for the next write.
        if (len == 0) {
            table.filled_in_last = 0;
        } else {
            const int last_page_idx = (int)((len - 1) / page_size_);
            table.filled_in_last = (int)(len - (int64_t)last_page_idx * page_size_);
        }
        any = true;
    }
    return any;
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
        // Phase 4-B/4-D: pages after the removed range compact to earlier
        // slots, so mirrored rows at/above p0 now live at different logical
        // positions. Both the device and the host mirror must re-upload them.
        if (table.cuda_mirror_synced > p0)
            table.cuda_mirror_synced = p0;
        if (table.host_mirror_synced > p0)
            table.host_mirror_synced = p0;
        table.d_dirty = true;  // Phase 2-D: indexes shifted — full re-upload.
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
        dst_table.d_dirty = true;  // Phase 2-D: table mutated — full re-upload.
        // Phase 4-B/4-D: dst's tail rows now alias src pages; force the dst row
        // mirrors to re-upload everything on their next read (rare operation).
        dst_table.cuda_mirror_synced = 0;
        dst_table.host_mirror_synced = 0;

        for (int p = first_page; p <= last_page && p < (int)src_table.page_ids.size(); ++p) {
            int page_id = src_table.page_ids[p];
            dst_table.page_ids.push_back(page_id);
            // Increment refcount — page is now shared
            layer_pages_[l][page_id].ref_count++;
        }

        dst_table.logical_len = std::max(dst_table.logical_len, p1 - p0);
        int last_idx = (int)dst_table.page_ids.size() - 1;
        if (last_idx >= 0) {
            dst_table.filled_in_last =
                (int)(dst_table.logical_len - (int64_t)last_idx * page_size_);
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

    // Defensive: drop any deferred row writes for this sequence (the queue is
    // normally empty between forwards; this guards an aborted forward).
    pending_row_writes_.erase(std::remove_if(pending_row_writes_.begin(),
                                             pending_row_writes_.end(),
                                             [seq_id](const PendingRowWrite& r) {
                                                 return r.seq_id == seq_id;
                                             }),
                              pending_row_writes_.end());

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
        // KVMem: archived pages are still reserved (device data evicted, host
        // copy alive). free_page drops the host copy via swapper_.drop().
        for (const auto& ap : table.archived_pages) {
            auto& page = layer_pages_[l][ap.page_id];
            if (page.ref_count > 0) {
                page.ref_count--;
                if (page.ref_count == 0) {
                    free_page(l, ap.page_id);
                }
            }
        }
        table.archived_pages.clear();
        table.window_active = false;
    }

    // Release the persistent mirrors, then drop the host tables.
#ifdef USE_CUDA
    free_page_table_mirrors(it->second);
#endif
    // Phase 4-D: the FP32 host row mirrors are the largest of the per-sequence
    // allocations; drop them explicitly instead of waiting for the table's
    // destructor, so a released sequence's working set is reclaimable even if
    // some caller still holds a slice view of the storage.
    for (auto& t : it->second) {
        t.host_key_mirror.reset();
        t.host_value_mirror.reset();
        t.host_mirror_rows = 0;
        t.host_mirror_synced = 0;
    }
    seq_tables_.erase(it);
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
                    const auto& page = layer_pages_[l][page_id];
                    // Roadmap 1.1: evicted pages no longer occupy primary memory.
                    if (page.evicted)
                        continue;
                    total += page.key.capacity_bytes();
                    total += page.value.capacity_bytes();
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

int PagedKVStorage::free_pages_full() const {
    std::lock_guard<std::mutex> lock(mutex_);
    int min_free = -1;
    for (const auto& pool : layer_pools_) {
        if (pool.max_pages < max_pages_)
            continue;  // SWA pool: recycles a fixed window, never binds
        const int f = (int)pool.free_page_ids.size();
        if (min_free < 0 || f < min_free)
            min_free = f;
    }
    return min_free < 0 ? 0 : min_free;
}

void PagedKVStorage::reset() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Defensive: drop deferred row writes (reset discards all sequence state).
    pending_row_writes_.clear();

    // Free the persistent mirrors before dropping the host tables.
#ifdef USE_CUDA
    for (auto& kv : seq_tables_)
        free_page_table_mirrors(kv.second);
#endif
    for (auto& kv : seq_tables_)
        for (auto& t : kv.second) {
            t.host_key_mirror.reset();
            t.host_value_mirror.reset();
            t.host_mirror_rows = 0;
            t.host_mirror_synced = 0;
        }
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
            // Roadmap 1.1: a reset must not leave pages in the evicted state —
            // they are all returning to the free list. Any host copies are
            // dropped (drop is a no-op for pages never evicted).
            swapper_.drop(l, p);
            layer_pages_[l][p].evicted = false;
        }
    }
    offloaded_pages_ = 0;
    brought_back_pages_ = 0;
    // Prefix blob pool: blobs are unreachable after the sequence tables are
    // cleared (every cache seq is gone), so the pool is dropped too. The blob
    // id counter is intentionally NOT reset — see the member comment.
    prefix_swapper_.reset();
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

// =========================================================================
// Roadmap 1.1: host offload (swap) helpers
// =========================================================================

namespace {

// Raw data pointer of a page's KVCacheStorage (source for a host copy).
const uint8_t* storage_raw_ptr(const KVCacheStorage& s) {
    if (s.dtype == KVCacheDType::FP32)
        return s.tensor ? reinterpret_cast<const uint8_t*>(s.tensor->data()) : nullptr;
    if (s.device == DeviceType::CPU)
        return s.h_data.empty() ? nullptr : s.h_data.data();
#ifdef USE_CUDA
    return static_cast<const uint8_t*>(s.d_data);
#else
    return nullptr;
#endif
}

// Mutable raw data pointer (destination for a blob restore into the page's
// own already-allocated buffers).
uint8_t* storage_raw_ptr(KVCacheStorage& s) {
    return const_cast<uint8_t*>(storage_raw_ptr(const_cast<const KVCacheStorage&>(s)));
}

// Release the page's primary-memory data (keeps dtype/device/rows/row_bytes so
// the storage can be re-allocated with the same layout by bring_back).
void storage_release_data(KVCacheStorage& s) {
    if (s.dtype == KVCacheDType::FP32) {
        s.tensor.reset();
    } else if (s.device == DeviceType::CPU) {
        s.h_data.clear();
        s.h_data.shrink_to_fit();
    } else {
#ifdef USE_CUDA
        if (s.d_data) {
            cudaFree(s.d_data);
            s.d_data = nullptr;
        }
        s.d_bytes = 0;
#endif
    }
}

// Re-allocate the page's storage (same layout) and copy the host copy back in.
bool storage_restore(KVCacheStorage& s, const uint8_t* src, size_t bytes) {
    if (!src || bytes == 0)
        return false;
    s.alloc(s.dtype, s.device, s.max_rows, s.row_bytes);
    if (s.dtype == KVCacheDType::FP32) {
        void* dst = s.tensor ? s.tensor->data() : nullptr;
        if (!dst)
            return false;
#ifdef USE_CUDA
        if (s.device == DeviceType::CUDA) {
            cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
        } else
#endif
        {
            std::memcpy(dst, src, bytes);
        }
        return true;
    }
    if (s.device == DeviceType::CPU) {
        if (s.h_data.size() < bytes)
            return false;
        std::memcpy(s.h_data.data(), src, bytes);
        return true;
    }
#ifdef USE_CUDA
    if (s.d_data && s.d_bytes >= bytes) {
        cudaMemcpy(s.d_data, src, bytes, cudaMemcpyHostToDevice);
        return true;
    }
#endif
    return false;
}

}  // namespace

// Rebuild the per-layer CUDA pointer tables after a page's storage pointer
// changed (evicted -> nullptr, brought back -> new device pointer).
void PagedKVStorage::refresh_cuda_ptr_tables_unlocked(int layer) {
#ifdef USE_CUDA
    if (device_ != DeviceType::CUDA || layer < 0 || layer >= (int)layer_pools_.size())
        return;
    int lmax = layer_pools_[layer].max_pages;
    d_key_page_ptrs_host_[layer].resize(lmax);
    d_value_page_ptrs_host_[layer].resize(lmax);
    for (int p = 0; p < lmax; ++p) {
        auto& page = layer_pages_[layer][p];
        if (page.evicted) {
            d_key_page_ptrs_host_[layer][p] = nullptr;
            d_value_page_ptrs_host_[layer][p] = nullptr;
        } else {
            d_key_page_ptrs_host_[layer][p] = (kv_config_.type_k == KVCacheDType::FP32)
                                                  ? static_cast<void*>(page.key.fp32_data())
                                                  : page.key.d_q_data();
            d_value_page_ptrs_host_[layer][p] = (kv_config_.type_v == KVCacheDType::FP32)
                                                    ? static_cast<void*>(page.value.fp32_data())
                                                    : page.value.d_q_data();
        }
    }
    if (d_key_page_ptrs_[layer])
        cudaMemcpy(d_key_page_ptrs_[layer], d_key_page_ptrs_host_[layer].data(),
                   lmax * sizeof(void*), cudaMemcpyHostToDevice);
    if (d_value_page_ptrs_[layer])
        cudaMemcpy(d_value_page_ptrs_[layer], d_value_page_ptrs_host_[layer].data(),
                   lmax * sizeof(void*), cudaMemcpyHostToDevice);
#else
    (void)layer;
#endif
}

bool PagedKVStorage::offload_to_host(int layer, int page_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    return offload_to_host_unlocked(layer, page_id);
}

bool PagedKVStorage::bring_back(int layer, int page_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    return bring_back_unlocked(layer, page_id);
}

bool PagedKVStorage::offload_seq(int seq_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = seq_tables_.find(seq_id);
    if (it == seq_tables_.end())
        return false;
    bool any = false;
    for (int l = 0; l < num_layers_; ++l) {
        const auto& table = it->second[l];
        for (int page_id : table.page_ids) {
            // Already-evicted pages have nothing new to move; skip them so the
            // caller can tell "this sequence was newly evicted" from "its pages
            // were already on the host".
            if (layer_pages_[l][page_id].evicted)
                continue;
            if (offload_to_host_unlocked(l, page_id))
                any = true;
        }
    }
    return any;
}

int PagedKVStorage::num_device_pages_in_use() const {
    std::lock_guard<std::mutex> lock(mutex_);
    int n = 0;
    for (int l = 0; l < num_layers_; ++l)
        for (const auto& page : layer_pages_[l])
            if (page.ref_count > 0 && !page.evicted)
                n++;
    return n;
}

bool PagedKVStorage::bring_back_seq(int seq_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = seq_tables_.find(seq_id);
    if (it == seq_tables_.end())
        return false;
    bool any = false;
    for (int l = 0; l < num_layers_; ++l) {
        const auto& table = it->second[l];
        for (int page_id : table.page_ids) {
            if (bring_back_unlocked(l, page_id))
                any = true;
        }
    }
    return any;
}

bool PagedKVStorage::offload_to_host_unlocked(int layer, int page_id) {
    if (layer < 0 || layer >= (int)layer_pages_.size())
        return false;
    auto& page = layer_pages_[layer][page_id];
    if (page.evicted)
        return true;  // already offloaded
    if (page.ref_count == 0)
        return false;  // free page: nothing to evict

    size_t k_bytes = page.key.capacity_bytes();
    size_t v_bytes = page.value.capacity_bytes();
    const uint8_t* k_src = storage_raw_ptr(page.key);
    const uint8_t* v_src = storage_raw_ptr(page.value);
    if (k_bytes == 0 && v_bytes == 0)
        return false;

    // Device or FP32 storage's bytes live in device memory; std::memcpy cannot
    // read them, so the swapper must be told which side is which.
    const KVBlockSwapper::CopyKind kind =
        (device_ == DeviceType::CUDA) ? KVBlockSwapper::CopyKind::Device
                                       : KVBlockSwapper::CopyKind::Host;
    if (!swapper_.offload(layer, page_id, k_src, k_bytes, v_src, v_bytes, kind))
        return false;

    storage_release_data(page.key);
    storage_release_data(page.value);
    page.evicted = true;
    offloaded_pages_++;
    refresh_cuda_ptr_tables_unlocked(layer);
    return true;
}

bool PagedKVStorage::bring_back_unlocked(int layer, int page_id) {
    if (layer < 0 || layer >= (int)layer_pages_.size())
        return false;
    auto& page = layer_pages_[layer][page_id];
    if (!page.evicted)
        return true;  // already on device

    size_t k_bytes = 0, v_bytes = 0;
    if (!swapper_.size(layer, page_id, &k_bytes, &v_bytes))
        return false;

    // Fetch the host copy into temporary buffers first, so a failed re-alloc
    // does not leave the page half-restored.
    std::vector<uint8_t> tmp_k(k_bytes), tmp_v(v_bytes);
    if (!swapper_.bring_back(layer, page_id, tmp_k.data(), k_bytes, tmp_v.data(), v_bytes))
        return false;

    if (k_bytes && !storage_restore(page.key, tmp_k.data(), k_bytes))
        return false;
    if (v_bytes && !storage_restore(page.value, tmp_v.data(), v_bytes))
        return false;

    page.evicted = false;
    brought_back_pages_++;
    refresh_cuda_ptr_tables_unlocked(layer);
    return true;
}

// =========================================================================
// Multi-level prefix cache: node-seg backup/restore blob pool
// =========================================================================

bool PagedKVStorage::backup_prefix_pages(int seq_id, int64_t p0, int64_t p1,
                                         std::vector<std::vector<int32_t>>* blob_keys) {
    std::lock_guard<std::mutex> lock(mutex_);
    return backup_prefix_pages_unlocked(seq_id, p0, p1, blob_keys);
}

bool PagedKVStorage::backup_prefix_pages_unlocked(int seq_id, int64_t p0, int64_t p1,
                                                  std::vector<std::vector<int32_t>>* blob_keys) {
    auto it = seq_tables_.find(seq_id);
    if (it == seq_tables_.end()) {
        LOG_ERROR("backup_prefix_pages: seq " + std::to_string(seq_id) + " not found");
        return false;
    }
    if (!blob_keys || p0 < 0 || p1 <= p0) {
        LOG_ERROR("backup_prefix_pages: invalid arguments (p0=" + std::to_string(p0) +
                  ", p1=" + std::to_string(p1) + ")");
        return false;
    }
    const int64_t n = p1 - p0;
    if ((int)blob_keys->size() != num_layers_) {
        LOG_ERROR("backup_prefix_pages: blob_keys has " + std::to_string(blob_keys->size()) +
                  " layer rows, expected " + std::to_string(num_layers_));
        return false;
    }

    const KVBlockSwapper::CopyKind kind = (device_ == DeviceType::CUDA)
                                              ? KVBlockSwapper::CopyKind::Device
                                              : KVBlockSwapper::CopyKind::Host;

    for (int l = 0; l < num_layers_; ++l) {
        auto& keys = (*blob_keys)[l];
        if ((int64_t)keys.size() != n) {
            LOG_ERROR("backup_prefix_pages: blob_keys[layer=" + std::to_string(l) +
                      "] has " + std::to_string(keys.size()) + " entries, expected " +
                      std::to_string(n));
            return false;
        }
        const auto& table = it->second[l];
        if (p1 > (int64_t)table.page_ids.size()) {
            LOG_ERROR("backup_prefix_pages: page range [" + std::to_string(p0) + ", " +
                      std::to_string(p1) + ") exceeds seq " + std::to_string(seq_id) + " layer " +
                      std::to_string(l) + " table (" + std::to_string(table.page_ids.size()) +
                      " pages)");
            return false;
        }
        for (int64_t i = 0; i < n; ++i) {
            const int page_id = table.page_ids[p0 + i];
            const auto& page = layer_pages_[l][page_id];
            // A prefix seg page must be complete and resident. A released
            // buffer (raw ptr nullptr) covers KVMem-archived pages defensively.
            if (page.evicted || page.filled != (uint32_t)page_size_) {
                LOG_ERROR("backup_prefix_pages: page " + std::to_string(page_id) + " not usable " +
                          "(evicted=" + std::to_string(page.evicted ? 1 : 0) + ", filled=" +
                          std::to_string(page.filled) + "/" + std::to_string(page_size_) + ")");
                return false;
            }
            const size_t k_bytes = page.key.capacity_bytes();
            const size_t v_bytes = page.value.capacity_bytes();
            const uint8_t* k_src = storage_raw_ptr(page.key);
            const uint8_t* v_src = storage_raw_ptr(page.value);
            if (!k_src || !v_src || (k_bytes == 0 && v_bytes == 0)) {
                LOG_ERROR("backup_prefix_pages: page " + std::to_string(page_id) +
                          " has no resident data");
                return false;
            }
            int32_t blob_id = keys[i];
            if (blob_id < 0)
                blob_id = next_prefix_blob_id_++;
            // Phase 3: a full host pool is not a dead end when L3 is backing it
            // — demote the coldest blobs and retry instead of losing the prefix.
            const size_t need = k_bytes + v_bytes;
            if (prefix_swapper_.max_bytes() && !prefix_swapper_.has(l, blob_id) &&
                prefix_swapper_.nbytes() + need > prefix_swapper_.max_bytes()) {
                spill_prefix_pool_unlocked(need);
            }
            if (!prefix_swapper_.offload(l, blob_id, k_src, k_bytes, v_src, v_bytes, kind))
                return false;  // e.g. host pool over budget — caller keeps the node on device
            keys[i] = blob_id;
        }
    }
    return true;
}

bool PagedKVStorage::restore_prefix_pages(int seq_id, int64_t p0,
                                          const std::vector<std::vector<int32_t>>& blob_keys) {
    std::lock_guard<std::mutex> lock(mutex_);
    return restore_prefix_pages_unlocked(seq_id, p0, blob_keys);
}

bool PagedKVStorage::restore_prefix_pages_unlocked(
    int seq_id, int64_t p0, const std::vector<std::vector<int32_t>>& blob_keys) {
    if (seq_id < 0 || p0 < 0 || (int)blob_keys.size() != num_layers_) {
        LOG_ERROR("restore_prefix_pages: invalid arguments (seq=" + std::to_string(seq_id) +
                  ", p0=" + std::to_string(p0) + ", layers=" + std::to_string(blob_keys.size()) +
                  "/" + std::to_string(num_layers_) + ")");
        return false;
    }
    int64_t n = -1;
    for (int l = 0; l < num_layers_; ++l) {
        if (n < 0)
            n = (int64_t)blob_keys[l].size();
        else if ((int64_t)blob_keys[l].size() != n) {
            LOG_ERROR("restore_prefix_pages: ragged blob_keys (layer 0 has " + std::to_string(n) +
                      ", layer " + std::to_string(l) + " has " +
                      std::to_string(blob_keys[l].size()) + ")");
            return false;
        }
    }
    if (n <= 0) {
        LOG_ERROR("restore_prefix_pages: empty page range");
        return false;
    }

    const KVBlockSwapper::CopyKind kind = (device_ == DeviceType::CUDA)
                                              ? KVBlockSwapper::CopyKind::Device
                                              : KVBlockSwapper::CopyKind::Host;
    const int64_t first_row = p0 * page_size_;

    for (int l = 0; l < num_layers_; ++l) {
        auto& keys = blob_keys[l];
        // Appends the [p0, p0+n) page slots. On alloc failure the pages taken
        // so far are rolled back inside ensure_seq_capacity for this layer;
        // the caller must release the half-built sequence.
        if (!ensure_seq_capacity(seq_id, l, (int)(first_row + n * page_size_)))
            return false;
        auto& table = seq_tables_[seq_id][l];
        if (p0 + n > (int64_t)table.page_ids.size()) {
            LOG_ERROR("restore_prefix_pages: page table too small after capacity ensure");
            return false;
        }
        // The restored rows were never in this sequence's mirrors: force
        // re-upload (device page table) and re-dequantize (host row mirror).
        table.d_dirty = true;
        if (table.cuda_mirror_synced > first_row)
            table.cuda_mirror_synced = (int)first_row;
        if (table.host_mirror_synced > first_row)
            table.host_mirror_synced = (int)first_row;

        for (int64_t i = 0; i < n; ++i) {
            const int page_id = table.page_ids[p0 + i];
            auto& page = layer_pages_[l][page_id];
            if (page.evicted && !bring_back_unlocked(l, page_id)) {
                LOG_ERROR("restore_prefix_pages: failed to bring back page " +
                          std::to_string(page_id));
                return false;
            }
            const int32_t blob_id = keys[i];
            if (blob_id < 0) {
                LOG_ERROR("restore_prefix_pages: invalid blob id at layer " + std::to_string(l) +
                          " page " + std::to_string(p0 + i));
                return false;
            }
            const size_t k_bytes = page.key.capacity_bytes();
            const size_t v_bytes = page.value.capacity_bytes();
            // Phase 3: the host copy may have been spilled to L3 — pull it back
            // into the pool so the (host -> device) copy below can proceed.
            if (!ensure_prefix_blob_unlocked(l, blob_id, k_bytes, v_bytes)) {
                LOG_ERROR("restore_prefix_pages: blob (" + std::to_string(l) + "," +
                          std::to_string(blob_id) + ") not found in host pool or L3");
                return false;
            }
            // Copy straight into the page's own buffers: they were just
            // allocated (or brought back) with the right layout, so no host
            // staging is needed. A mid-way failure leaves this sequence's
            // fresh pages inconsistent — the caller releases the sequence.
            if (!prefix_swapper_.bring_back(l, blob_id, storage_raw_ptr(page.key), k_bytes,
                                            storage_raw_ptr(page.value), v_bytes, kind)) {
                LOG_ERROR("restore_prefix_pages: blob (" + std::to_string(l) + "," +
                          std::to_string(blob_id) + ") missing or size mismatch");
                return false;
            }
            page.filled = (uint32_t)page_size_;

        }

        table.logical_len = std::max<int64_t>(table.logical_len, first_row + n * page_size_);
        if (p0 + n == (int64_t)table.page_ids.size())
            table.filled_in_last = page_size_;  // restored pages are always complete
    }
    return true;
}

bool PagedKVStorage::drop_prefix_blob(int layer, int32_t blob_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (layer < 0 || layer >= num_layers_)
        return false;
    if (!prefix_swapper_.has(layer, blob_id))
        return false;
    prefix_swapper_.drop(layer, blob_id);
    return true;
}

bool PagedKVStorage::spill_prefix_pool_unlocked(size_t need) {
    if (!l3_)
        return false;

    const size_t cap = prefix_swapper_.max_bytes();
    int guard = 0;
    while ((!cap || prefix_swapper_.nbytes() + need > cap) && guard++ < 100000) {
        int l = -1;
        int32_t id = -1;
        if (!prefix_swapper_.lru_oldest(&l, &id))
            return false;  // pool is empty and `need` still does not fit
        const uint8_t* k = nullptr;
        const uint8_t* v = nullptr;
        size_t k_bytes = 0;
        size_t v_bytes = 0;
        if (!prefix_swapper_.peek(l, id, &k, &v, &k_bytes, &v_bytes)) {
            prefix_swapper_.drop(l, id);  // entry without data: just drop it
            continue;
        }
        if (!l3_->put(SwapperKVBackend::make_key(l, id), k, k_bytes, v, v_bytes))
            return false;
        prefix_swapper_.drop(l, id);
        ++l3_spills_;
    }
    return true;
}

bool PagedKVStorage::ensure_prefix_blob_unlocked(int layer, int32_t blob_id, size_t k_bytes,
                                                  size_t v_bytes) {
    if (prefix_swapper_.has(layer, blob_id))
        return true;
    if (!l3_)
        return false;

    const uint64_t key = SwapperKVBackend::make_key(layer, blob_id);
    std::vector<uint8_t> k_buf(k_bytes);
    std::vector<uint8_t> v_buf(v_bytes);
    if (!l3_->get(key, k_buf.data(), k_bytes, v_buf.data(), v_bytes))
        return false;
    // Park the blob back in the host pool: the restore path copies from there
    // (host -> device), which keeps a single copy path for both tiers. The re-park
    // is transient (consumed by bring_back immediately below), so it bypasses the
    // host-pool cap — a momentary over-budget park is re-balanced on the next
    // host-tier eviction.
    if (!prefix_swapper_.offload(layer, blob_id, k_buf.data(), k_bytes, v_buf.data(), v_bytes,
                                 KVBlockSwapper::CopyKind::Host, /*force=*/true))
        return false;
    ++l3_fetches_;
    return true;
}

bool PagedKVStorage::spill_prefix_blobs(const std::vector<std::vector<int32_t>>& blob_keys) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!l3_)
        return false;

    bool ok = true;
    size_t spilled = 0;
    for (int l = 0; l < num_layers_ && ok; ++l) {
        if (l >= static_cast<int>(blob_keys.size()))
            break;
        for (int32_t id : blob_keys[l]) {
            if (id < 0)
                continue;
            const uint8_t* k = nullptr;
            const uint8_t* v = nullptr;
            size_t k_bytes = 0;
            size_t v_bytes = 0;
            if (!prefix_swapper_.peek(l, id, &k, &v, &k_bytes, &v_bytes))
                continue;  // already spilled (or dropped) — nothing to move
            if (!l3_->put(SwapperKVBackend::make_key(l, id), k, k_bytes, v, v_bytes)) {
                ok = false;
                break;
            }
            prefix_swapper_.drop(l, id);
            ++spilled;
        }
    }
    // A partial spill is still progress (the pool shrank), but the caller needs
    // to know it is incomplete so it can decide whether to keep the node.
    l3_spills_ += static_cast<int>(spilled);
    return ok && spilled > 0;
}

size_t PagedKVStorage::prefix_pool_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return prefix_swapper_.nbytes();
}

bool PagedKVStorage::prefix_pool_lru_oldest(int* layer, int32_t* blob_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    int l = -1, id = -1;
    if (!prefix_swapper_.lru_oldest(&l, &id))
        return false;
    if (layer)
        *layer = l;
    if (blob_id)
        *blob_id = id;
    return true;
}

void PagedKVStorage::set_prefix_pool_max_bytes(size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    prefix_swapper_.set_max_bytes(bytes);
}

int PagedKVStorage::seq_num_pages(int seq_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = seq_tables_.find(seq_id);
    if (it == seq_tables_.end())
        return 0;
    int n = 0;
    for (int l = 0; l < num_layers_; ++l)
        n += (int)it->second[l].page_ids.size();
    return n;
}

size_t PagedKVStorage::host_pool_bytes() const {
    return swapper_.nbytes();
}

int PagedKVStorage::total_page_capacity() const {
    std::lock_guard<std::mutex> lock(mutex_);
    int total = 0;
    for (const auto& pool : layer_pools_)
        total += pool.max_pages;
    return total;
}

const int32_t* PagedKVStorage::upload_seq_page_table_unlocked(int layer, int seq_id) {
    auto it = seq_tables_.find(seq_id);
    if (it == seq_tables_.end() || layer < 0 || layer >= (int)it->second.size())
        return nullptr;
    SequencePageTable& table = it->second[layer];
    const auto& page_ids = table.page_ids;
    int n = (int)page_ids.size();
    if (n == 0)
        return nullptr;
#ifdef USE_CUDA
    if (device_ != DeviceType::CUDA)
        return nullptr;
    // Phase 2-D: keep a persistent per-(seq, layer) device mirror. Decode grows
    // the table one page at a time, so the hot path copies only the appended
    // tail (O(new pages)) instead of the whole table (O(total pages) per
    // token). A full rewrite happens when the host table changed mid-array
    // (d_dirty, e.g. seq_remove/seq_share) or the capacity must grow.
    bool need_full = table.d_dirty || n > table.d_cap || n < table.d_synced;
    if (need_full) {
        if (n > table.d_cap) {
            if (table.d_mirror)
                cudaFree(table.d_mirror);
            table.d_cap = std::max(16, n * 2);  // grow-only amortized O(1)
            cudaMalloc(&table.d_mirror, (size_t)table.d_cap * sizeof(int32_t));
        }
        cudaMemcpy(table.d_mirror, page_ids.data(), (size_t)n * sizeof(int32_t),
                   cudaMemcpyHostToDevice);
        table.d_synced = n;
        table.d_dirty = false;
        return table.d_mirror;
    }
    // Pure append: copy only the pages added since the last sync.
    int start = table.d_synced;
    if (start < n) {
        cudaMemcpy(table.d_mirror + start, page_ids.data() + start,
                   (size_t)(n - start) * sizeof(int32_t), cudaMemcpyHostToDevice);
        table.d_synced = n;
    }
    return table.d_mirror;
#else
    return nullptr;
#endif
}

// =========================================================================
// Phase 4-B: device row mirror for CPU-backed pools serving CUDA attention.
//
// Without this, a partial-offload deployment (-ngl < num_layers) pays, on
// every attention call, for a host materialization of the whole window
// (read_key_seq/read_value_seq) plus a full H2D copy of both tensors —
// O(context) PCIe traffic per layer per decode step, which is what made
// paged decode ~2x slower than the contiguous KV cache at long context.
// The mirror keeps rows [0, synced) of each (seq, layer) resident on the
// device; writes only ever append (decode) or truncate (new turn), so the
// steady-state cost is one row uploaded per generated token. Invalidation
// lives entirely on the write side (write_kv_seq / seq_remove / seq_share);
// every other mutation either preserves rows [0, p) or erases the table
// (release/reset free the mirrors with it).
// =========================================================================
bool PagedKVStorage::read_window_cuda(int layer, int seq_id, TensorPtr* k_out,
                                      TensorPtr* v_out) {
#ifdef USE_CUDA
    // Mirror serves CPU-backed pools only: a CUDA-backed pool already answers
    // from its device pages. The mirror itself always stores FP32 rows -- the
    // materialized attention paths consume FP32 regardless of the storage
    // dtype -- so quantized pages are dequantized row-by-row on upload.
    if (device_ != DeviceType::CPU)
        return false;

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = seq_tables_.find(seq_id);
    if (it == seq_tables_.end() || layer < 0 || layer >= (int)it->second.size())
        return false;
    SequencePageTable& table = it->second[layer];
    const int64_t n = table.logical_len;
    if (n <= 0)
        return false;

    const int kv_dim = (layer < (int)kv_dims_.size()) ? kv_dims_[layer] : kv_dims_[0];
    const size_t row_bytes = (size_t)kv_dim * sizeof(float);

    // (Re)allocate the mirror when missing or too small. Capacity doubles so
    // decode's one-row appends amortize to O(1) H2D traffic; a realloc
    // re-uploads everything once.
    if (!table.cuda_key_mirror || table.cuda_mirror_rows < n) {
        const int64_t new_rows =
            table.cuda_mirror_rows > 0
                ? std::max<int64_t>(table.cuda_mirror_rows * 2, n)
                : std::max<int64_t>(n, 256);
        table.cuda_key_mirror = std::make_shared<Tensor>(
            DataType::FP32, std::vector<int64_t>{new_rows, kv_dim}, DeviceType::CUDA);
        table.cuda_value_mirror = std::make_shared<Tensor>(
            DataType::FP32, std::vector<int64_t>{new_rows, kv_dim}, DeviceType::CUDA);
        table.cuda_mirror_rows = new_rows;
        table.cuda_mirror_synced = 0;
    }

    // Upload rows [cuda_mirror_synced, n). Same page walk as read_key_seq:
    // pages fill sequentially, the last bounded by logical_len (page_ids may
    // cover more pages than there are visible rows after a KVMem clip).
    // Rows at/above pending_pos are visible via logical_len (the deferred-write
    // enqueue advanced it) but their page cells have not been written yet — the
    // page upload must stop short of them; the pending row itself is filled
    // from its device tensors after the loop below.
    const int64_t page_cap =
        (table.pending_pos >= 0 && table.pending_pos < n) ? table.pending_pos : n;
    int64_t row_base = 0;
    for (int p = 0; p < (int)table.page_ids.size() && row_base < n; ++p) {
        const int64_t rows_in_page = std::min<int64_t>(page_size_, n - row_base);
        const int64_t span_lo = std::max(row_base, table.cuda_mirror_synced);
        const int64_t span_hi = std::min(row_base + rows_in_page, page_cap);  // exclusive
        if (span_lo < span_hi) {
            const int page_id = table.page_ids[p];
            const auto& page = layer_pages_[layer][page_id];
            const int r0 = static_cast<int>(span_lo - row_base);
            const int cnt = static_cast<int>(span_hi - span_lo);
            const size_t span_bytes = static_cast<size_t>(cnt) * row_bytes;
            const size_t span_rows = static_cast<size_t>(cnt) * kv_dim;
            float* k_dst =
                static_cast<float*>(table.cuda_key_mirror->data()) +
                static_cast<size_t>(span_lo) * kv_dim;
            float* v_dst =
                static_cast<float*>(table.cuda_value_mirror->data()) +
                static_cast<size_t>(span_lo) * kv_dim;
            const bool k_fp32 = kv_config_.type_k == KVCacheDType::FP32;
            const bool v_fp32 = kv_config_.type_v == KVCacheDType::FP32;
            // Host staging for dequantized spans (K first half, V second half so
            // one upload's source is never overwritten while another is queued).
            float* stage = nullptr;
            if (!k_fp32 || !v_fp32) {
                table.mirror_stage_.resize(span_rows * ((k_fp32 != v_fp32) ? 1 : 2));
                stage = table.mirror_stage_.data();
            }
            // Uploads are asynchronous on the default stream: the attention
            // kernel that consumes these rows is launched on the same stream,
            // so stream order alone guarantees visibility -- a synchronous
            // cudaMemcpy here would stall the host until the whole queued GPU
            // pipeline drains (observed as ~3x slower decode).
            if (k_fp32) {
                // Fast path: FP32 pages are contiguous -- copy the span as-is.
                cudaMemcpyAsync(k_dst, page.key.fp32_row(r0), span_bytes,
                                cudaMemcpyHostToDevice);
            } else {
                // Quantized pages: dequantize the span into host staging, then
                // upload it as one block. Decode appends a single row per step,
                // so the steady-state cost is one row dequant + 2 KB H2D.
                for (int r = 0; r < cnt; ++r) {
                    KVCache::dequantize_row(kv_config_.type_k, page.key.q_row(r0 + r),
                                            stage + static_cast<size_t>(r) * kv_dim, kv_dim);
                }
                cudaMemcpyAsync(k_dst, stage, span_bytes, cudaMemcpyHostToDevice);
            }
            if (v_fp32) {
                cudaMemcpyAsync(v_dst, page.value.fp32_row(r0), span_bytes,
                                cudaMemcpyHostToDevice);
            } else {
                float* v_stage = k_fp32 ? stage : stage + span_rows;
                for (int r = 0; r < cnt; ++r) {
                    KVCache::dequantize_row(kv_config_.type_v, page.value.q_row(r0 + r),
                                            v_stage + static_cast<size_t>(r) * kv_dim, kv_dim);
                }
                cudaMemcpyAsync(v_dst, v_stage, span_bytes, cudaMemcpyHostToDevice);
            }
        }
        row_base += rows_in_page;
    }

    // Serve the pending (not yet flushed) decode row straight from its device
    // tensors. This also covers a mirror realloc above: the page upload stops
    // at page_cap, so without this the newest row would be missing entirely.
    if (table.pending_pos >= 0 && table.pending_pos >= table.cuda_mirror_synced &&
        table.pending_pos < n && table.pending_pos < table.cuda_mirror_rows &&
        table.pending_k_dev && table.pending_v_dev) {
        float* k_dst = static_cast<float*>(table.cuda_key_mirror->data()) +
                       (size_t)table.pending_pos * kv_dim;
        float* v_dst = static_cast<float*>(table.cuda_value_mirror->data()) +
                       (size_t)table.pending_pos * kv_dim;
        cudaMemcpyAsync(k_dst, table.pending_k_dev->data(), row_bytes,
                        cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(v_dst, table.pending_v_dev->data(), row_bytes,
                        cudaMemcpyDeviceToDevice);
    }
    table.cuda_mirror_synced = n;

    // Contiguous [n, kv_dim] views into the mirror storage: no copy, and the
    // next read re-uploads only rows appended after this call.
    *k_out = std::make_shared<Tensor>(table.cuda_key_mirror->slice(0, 0, n));
    *v_out = std::make_shared<Tensor>(table.cuda_value_mirror->slice(0, 0, n));
    return true;
#else
    (void)layer;
    (void)seq_id;
    (void)k_out;
    (void)v_out;
    return false;
#endif
}

// =========================================================================
// Phase 4-D: host row mirror for CPU-backed pools serving CPU attention.
//
// read_window_cuda fixed the CUDA side of long-context paged decode, but the
// layers whose attention runs on the host (partial offload: -ngl < num_layers)
// still went through read_key_seq/read_value_seq, which materialize the WHOLE
// window into two freshly allocated FP32 tensors on every call — O(context)
// host dequant + copy per layer per decode step, and the fresh allocations
// also throw away the cache locality a persistent buffer would have. Measured
// on Qwen2.5-7B Q4_0 / RTX 4050 at 5.1k context with -ngl 16: the 12
// non-offloaded layers cost ~22 ms/step, i.e. essentially the whole remaining
// gap to the contiguous cache (which hands out a zero-copy view into a
// persistent per-layer KV buffer).
// This provides that view: rows [0, synced) stay resident on the host, decode
// appends one row per step, and every write at position p truncates the
// watermark to p so rewritten rows are re-uploaded. Invalidation is handled
// entirely by the same write-side hooks as the CUDA mirror (write_kv_seq,
// seq_remove, seq_share); release/reset drop the tables that own the tensors.
// =========================================================================
bool PagedKVStorage::read_window_host(int layer, int seq_id, TensorPtr* k_out,
                                      TensorPtr* v_out) {
    // Only a CPU-backed pool can be materialized on the host: page buffers are
    // host pointers, and dequantize_row() below is a host function. A CUDA
    // pool must use its paged kernels instead (read_key* return null there).
    if (device_ != DeviceType::CPU)
        return false;

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = seq_tables_.find(seq_id);
    if (it == seq_tables_.end() || layer < 0 || layer >= (int)it->second.size())
        return false;
    SequencePageTable& table = it->second[layer];
    const int64_t n = table.logical_len;
    if (n <= 0)
        return false;
    // A deferred (device-only) row inside the visible window cannot be served
    // here: its page cell still holds the previous contents and its data lives
    // on the device, so the upload below would read a stale row. Unreachable
    // in practice — deferral only applies to layers whose attention runs on
    // CUDA, and those read through read_window_cuda — so fall back instead of
    // silently serving stale KV.
    if (table.pending_pos >= 0 && table.pending_pos < n)
        return false;

    const int kv_dim = (layer < (int)kv_dims_.size()) ? kv_dims_[layer] : kv_dims_[0];

    // (Re)allocate when missing or too small; capacity doubles so decode's
    // one-row appends amortize to O(1) host work. A realloc re-uploads once.
    if (!table.host_key_mirror || table.host_mirror_rows < n) {
        const int64_t new_rows =
            table.host_mirror_rows > 0
                ? std::max<int64_t>(table.host_mirror_rows * 2, n)
                : std::max<int64_t>(n, 256);
        table.host_key_mirror = std::make_shared<Tensor>(
            DataType::FP32, std::vector<int64_t>{new_rows, kv_dim}, DeviceType::CPU);
        table.host_value_mirror = std::make_shared<Tensor>(
            DataType::FP32, std::vector<int64_t>{new_rows, kv_dim}, DeviceType::CPU);
        table.host_mirror_rows = new_rows;
        table.host_mirror_synced = 0;
    }

    float* k_base = static_cast<float*>(table.host_key_mirror->data());
    float* v_base = static_cast<float*>(table.host_value_mirror->data());
    const bool k_fp32 = kv_config_.type_k == KVCacheDType::FP32;
    const bool v_fp32 = kv_config_.type_v == KVCacheDType::FP32;
    const size_t row_bytes = static_cast<size_t>(kv_dim) * sizeof(float);

    // Upload rows [host_mirror_synced, n). Same page walk as read_key_seq:
    // pages fill sequentially, the last bounded by logical_len (page_ids may
    // cover more pages than there are visible rows after a KVMem clip).
    int64_t row_base = 0;
    for (int p = 0; p < (int)table.page_ids.size() && row_base < n; ++p) {
        const int64_t rows_in_page = std::min<int64_t>(page_size_, n - row_base);
        const int64_t span_lo = std::max(row_base, table.host_mirror_synced);
        const int64_t span_hi = std::min(row_base + rows_in_page, n);  // exclusive
        if (span_lo < span_hi) {
            const auto& page = layer_pages_[layer][table.page_ids[p]];
            const int r0 = static_cast<int>(span_lo - row_base);
            const int cnt = static_cast<int>(span_hi - span_lo);
            float* k_dst = k_base + static_cast<size_t>(span_lo) * kv_dim;
            float* v_dst = v_base + static_cast<size_t>(span_lo) * kv_dim;
            for (int r = 0; r < cnt; ++r) {
                if (k_fp32) {
                    std::memcpy(k_dst + static_cast<size_t>(r) * kv_dim,
                                page.key.fp32_row(r0 + r), row_bytes);
                } else {
                    KVCache::dequantize_row(kv_config_.type_k, page.key.q_row(r0 + r),
                                            k_dst + static_cast<size_t>(r) * kv_dim, kv_dim);
                }
                if (v_fp32) {
                    std::memcpy(v_dst + static_cast<size_t>(r) * kv_dim,
                                page.value.fp32_row(r0 + r), row_bytes);
                } else {
                    KVCache::dequantize_row(kv_config_.type_v, page.value.q_row(r0 + r),
                                            v_dst + static_cast<size_t>(r) * kv_dim, kv_dim);
                }
            }
        }
        row_base += rows_in_page;
    }
    table.host_mirror_synced = n;

    // Contiguous [n, kv_dim] views into the persistent mirror storage: no copy,
    // and the next read re-materializes only rows appended after this call.
    *k_out = std::make_shared<Tensor>(table.host_key_mirror->slice(0, 0, n));
    *v_out = std::make_shared<Tensor>(table.host_value_mirror->slice(0, 0, n));
    return true;
}

// =========================================================================
// Phase 4-C: deferred device row writes.
//
// The read-side mirror (read_window_cuda) removed the per-step full-window
// materialization, but the WRITE side still stalled the host once per CUDA
// layer: staging the freshly computed row into a CPU-backed pool went through
// a synchronous D2H, which waits for this layer's GPU pipeline to drain
// (~2-3 ms per layer at long context, 16x per decode step with partial
// offload). Here the row hops to the host asynchronously (pinned staging) and
// the page write is deferred to flush_pending_writes(), which pays ONE stream
// sync per decode step. The row is also absorbed into the device row mirror
// (device-to-device) so the next attention sees it without waiting for the
// page write; until the flush lands, logical_len excludes the row — engines
// that enqueue must therefore flush at the end of every forward pass.
// =========================================================================
bool PagedKVStorage::enqueue_device_row_write(int layer, int seq_id, int64_t pos,
                                              const TensorPtr& k, const TensorPtr& v) {
#ifdef USE_CUDA
    if (!defer_device_writes_ || device_ != DeviceType::CPU)
        return false;
    if (!k || !v || k->device() != DeviceType::CUDA || v->device() != DeviceType::CUDA)
        return false;
    if (k->dtype() != DataType::FP32 || v->dtype() != DataType::FP32)
        return false;

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = seq_tables_.find(seq_id);
    if (it == seq_tables_.end() || layer < 0 || layer >= (int)it->second.size())
        return false;
    SequencePageTable& table = it->second[layer];
    const int kv_dim = (layer < (int)kv_dims_.size()) ? kv_dims_[layer] : kv_dims_[0];
    if (k->numel() != kv_dim || v->numel() != kv_dim)
        return false;  // not a single full row — use the synchronous path
    const size_t row_bytes = (size_t)kv_dim * sizeof(float);

    // Advance the logical length NOW: the synchronous path's write_kv_seq
    // makes the row visible to seq_filled() before attention runs, and this
    // path must behave identically — deferring that visibility to flush time
    // made decode attention skip the current token's own row (corrupted
    // output). The page write itself still lands at flush; until then
    // read_window_cuda serves this row from the device tensors recorded below,
    // never from the unflushed page cell.
    table.pending_pos = pos;
    table.pending_k_dev = k;
    table.pending_v_dev = v;
    table.logical_len = std::max(table.logical_len, pos + 1);

    // Stage the row into pinned memory for the page write at flush time.
    float* k_host = pinned_alloc_unlocked(row_bytes);
    float* v_host = pinned_alloc_unlocked(row_bytes);
    if (!k_host || !v_host) {
        table.pending_pos = -1;
        table.pending_k_dev.reset();
        table.pending_v_dev.reset();
        return false;
    }
    cudaMemcpyAsync(k_host, k->data(), row_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(v_host, v->data(), row_bytes, cudaMemcpyDeviceToHost);

    PendingRowWrite entry;
    entry.layer = layer;
    entry.seq_id = seq_id;
    entry.pos = pos;
    entry.k_host = k_host;
    entry.v_host = v_host;
    pending_row_writes_.push_back(std::move(entry));
    return true;
#else
    (void)layer;
    (void)seq_id;
    (void)pos;
    (void)k;
    (void)v;
    return false;
#endif
}

void PagedKVStorage::flush_pending_writes() {
    std::vector<PendingRowWrite> rows;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        rows.swap(pending_row_writes_);
    }
    if (rows.empty())
        return;
#ifdef USE_CUDA
    // One sync per decode step: every enqueued D2H lands here, so the host
    // never stalls per layer waiting on the GPU pipeline.
    cudaStreamSynchronize(0);
    for (const auto& r : rows) {
        if (!write_kv_seq(r.layer, r.seq_id, r.pos, 1, r.k_host, r.v_host)) {
            pinned_off_ = 0;
            throw std::runtime_error(
                "Paged KV deferred write failed at layer " + std::to_string(r.layer) +
                ": pos=" + std::to_string(r.pos) +
                " (page pool exhausted — reduce --ctx-size or free VRAM)");
        }
    }
    pinned_off_ = 0;  // reuse the pinned pool from the start next step
    // The rows are in their pages now — drop the pending device tensors so
    // read_window_cuda goes back to serving these rows from the pages.
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& r : rows) {
            auto it = seq_tables_.find(r.seq_id);
            if (it == seq_tables_.end() || r.layer < 0 || r.layer >= (int)it->second.size())
                continue;
            auto& t = it->second[r.layer];
            if (t.pending_pos == r.pos) {
                t.pending_pos = -1;
                t.pending_k_dev.reset();
                t.pending_v_dev.reset();
            }
        }
    }
#endif
}

float* PagedKVStorage::pinned_alloc_unlocked(size_t bytes) {
#ifdef USE_CUDA
    if (pinned_blocks_.empty() || pinned_block_cap_ - pinned_off_ < bytes) {
        const size_t block = std::max<size_t>(bytes, 1 << 20);
        void* p = nullptr;
        if (cudaHostAlloc(&p, block, cudaHostAllocDefault) != cudaSuccess) {
            cudaGetLastError();
            return nullptr;
        }
        pinned_blocks_.push_back(p);
        pinned_block_cap_ = block;
        pinned_off_ = 0;
    }
    float* out =
        reinterpret_cast<float*>(static_cast<char*>(pinned_blocks_.back()) + pinned_off_);
    pinned_off_ += bytes;
    return out;
#else
    (void)bytes;
    return nullptr;
#endif
}

void PagedKVStorage::free_pinned_blocks_() {
#ifdef USE_CUDA
    for (void* p : pinned_blocks_)
        cudaFreeHost(p);
    pinned_blocks_.clear();
    pinned_block_cap_ = 0;
    pinned_off_ = 0;
#endif
}

// =========================================================================
// KVMem tiered window (Phase K)
// =========================================================================

bool PagedKVStorage::kvmem_supported() const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& pool : layer_pools_)
        if (pool.policy != KVLayerPolicy::Full)
            return false;
    return !layer_pools_.empty();
}

void PagedKVStorage::attach_working_set(int seq_id, bool active) {
    std::lock_guard<std::mutex> lock(mutex_);
    // Record the arming so init_seq() can re-arm tables re-created after
    // reset()/release() for as long as the manager stays attached.
    if (active) {
        ws_armed_seqs_.insert(seq_id);
    } else {
        ws_armed_seqs_.erase(seq_id);
    }
    // The table is created lazily on the first write, so arming a sequence that
    // has no pages yet must create an empty one rather than silently do nothing.
    auto& table = seq_tables_[seq_id];
    table.resize((size_t)std::max(0, num_layers_));
    for (auto& t : table)
        t.window_active = active;
}

int64_t PagedKVStorage::seq_window_len(int seq_id, int layer) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = seq_tables_.find(seq_id);
    if (it == seq_tables_.end() || layer < 0 || layer >= (int)it->second.size())
        return -1;
    const auto& table = it->second[layer];
    if (!table.window_active)
        return -1;
    return table.logical_len;
}

int64_t PagedKVStorage::seq_window_tokens(int seq_id, int layer) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = seq_tables_.find(seq_id);
    if (it == seq_tables_.end() || layer < 0 || layer >= (int)it->second.size())
        return -1;
    return it->second[layer].logical_len;
}

int64_t PagedKVStorage::seq_archived_tokens(int seq_id, int layer) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = seq_tables_.find(seq_id);
    if (it == seq_tables_.end() || layer < 0 || layer >= (int)it->second.size())
        return -1;
    int64_t total = 0;
    for (const auto& ap : it->second[layer].archived_pages)
        total += ap.tokens;
    return total;
}

int PagedKVStorage::seq_archived_count(int seq_id, int layer) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = seq_tables_.find(seq_id);
    if (it == seq_tables_.end() || layer < 0 || layer >= (int)it->second.size())
        return -1;
    return (int)it->second[layer].archived_pages.size();
}

bool PagedKVStorage::seq_archived_page_ids(int seq_id, int layer,
                                           std::vector<int32_t>* out) const {
    if (!out)
        return false;
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = seq_tables_.find(seq_id);
    if (it == seq_tables_.end() || layer < 0 || layer >= (int)it->second.size())
        return false;
    out->clear();
    for (const auto& ap : it->second[layer].archived_pages)
        out->push_back(ap.page_id);
    return true;
}

int64_t PagedKVStorage::seq_window_archive_page(int seq_id, int layer, int idx) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = seq_tables_.find(seq_id);
    if (it == seq_tables_.end() || layer < 0 || layer >= (int)it->second.size())
        return -1;
    auto& table = it->second[layer];
    if (!table.window_active || idx < 0 || idx >= (int)table.page_ids.size())
        return -1;

    int page_id = table.page_ids[idx];
    auto& page = layer_pages_[layer][page_id];
    int64_t tokens = page.filled;

    // Copy the page's data to the host swap pool before it is overwritten.
    // offload_to_host_unlocked is idempotent for already-evicted pages and releases
    // the device buffers.
    if (!offload_to_host_unlocked(layer, page_id))
        return -1;

    // Deliberately NOT erased from page_ids. write_kv_seq resolves a slot to its
    // page as page_ids[pos / page_size], so page_ids is the *slot -> page* map and
    // must stay hole-free; compacting it here would shift every later slot onto the
    // wrong page and exhaust the pool on the next write. An archived page stays
    // referenced (ref_count > 0, not in the free list) with its data off-device,
    // and is reclaimed by seq_window_restore_page when it is recalled.
    table.archived_pages.push_back({page_id, (int32_t)tokens});
    table.d_dirty = true;
    return tokens;
}

int64_t PagedKVStorage::seq_window_drop_page(int seq_id, int layer, int idx) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = seq_tables_.find(seq_id);
    if (it == seq_tables_.end() || layer < 0 || layer >= (int)it->second.size())
        return -1;
    auto& table = it->second[layer];
    if (!table.window_active || idx < 0 || idx >= (int)table.page_ids.size())
        return -1;

    int page_id = table.page_ids[idx];
    auto& page = layer_pages_[layer][page_id];
    int64_t tokens = page.filled;

    // Stale data: discard entirely (frees any host copy too, see free_page).
    if (page.ref_count > 0) {
        page.ref_count--;
        if (page.ref_count == 0)
            free_page(layer, page_id);
    }

    // page_ids stays hole-free for the same reason as in seq_window_archive_page:
    // it is the slot -> page map that write_kv_seq indexes by pos / page_size.
    // Zero the page id so the slot has no device backing until it is rewritten.
    table.page_ids[idx] = -1;
    table.d_dirty = true;
    return tokens;
}

int64_t PagedKVStorage::seq_window_restore_page(int seq_id, int layer, int archive_idx,
                                                int insert_idx) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = seq_tables_.find(seq_id);
    if (it == seq_tables_.end() || layer < 0 || layer >= (int)it->second.size())
        return -1;
    auto& table = it->second[layer];
    if (!table.window_active || archive_idx < 0 ||
        archive_idx >= (int)table.archived_pages.size())
        return -1;
    if (insert_idx < 0 || insert_idx > (int)table.page_ids.size())
        return -1;

    auto ap = table.archived_pages[archive_idx];
    // Restore the device copy from the host swap pool.
    if (!bring_back_unlocked(layer, ap.page_id))
        return -1;

    table.archived_pages.erase(table.archived_pages.begin() + archive_idx);

    // page_ids is the slot -> page map indexed by pos / page_size, so it must stay
    // hole-free and the same length; inserting here would shift every later slot
    // onto the wrong page. Restore in place when the slot still exists and is empty
    // (the archive/restore pairing the manager uses), otherwise fall back to
    // appending so a mismatch can never silently corrupt the mapping.
    int placed = -1;
    if (insert_idx >= 0 && insert_idx < (int)table.page_ids.size() &&
        table.page_ids[insert_idx] < 0) {
        table.page_ids[insert_idx] = ap.page_id;
        placed = insert_idx;
    } else {
        int free_slot = -1;
        for (int i = 0; i < (int)table.page_ids.size(); ++i) {
            if (table.page_ids[i] < 0) {
                free_slot = i;
                break;
            }
        }
        if (free_slot < 0)
            table.page_ids.push_back(ap.page_id);
        else
            table.page_ids[free_slot] = ap.page_id;
        placed = free_slot >= 0 ? free_slot : (int)table.page_ids.size() - 1;
    }
    table.d_dirty = true;
    // The page keeps its own slot, so the window length is unchanged; only the
    // data came back. (An archived page was always a full window page.)
    (void)placed;
    table.logical_len = std::max<int64_t>(table.logical_len,
                                          static_cast<int64_t>(placed + 1) * page_size_);
    return ap.tokens;
}

}  // namespace forge
