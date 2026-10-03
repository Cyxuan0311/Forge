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
        for (int p = 0; p < (int)table.page_ids.size(); ++p) {
            int page_id = table.page_ids[p];
            const auto& page = layer_pages_[layer][page_id];
            int rows_in_page =
                (p == (int)table.page_ids.size() - 1) ? table.filled_in_last : page_size_;
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
        for (int p = 0; p < (int)table.page_ids.size(); ++p) {
            int page_id = table.page_ids[p];
            const auto& page = layer_pages_[layer][page_id];
            int rows_in_page =
                (p == (int)table.page_ids.size() - 1) ? table.filled_in_last : page_size_;
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
    for (int p = 0; p < (int)table.page_ids.size(); ++p) {
        int page_id = table.page_ids[p];
        const auto& page = layer_pages_[layer][page_id];
        int rows_in_page =
            (p == (int)table.page_ids.size() - 1) ? table.filled_in_last : page_size_;
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
    for (int p = 0; p < (int)table.page_ids.size(); ++p) {
        int page_id = table.page_ids[p];
        const auto& page = layer_pages_[layer][page_id];
        int rows_in_page =
            (p == (int)table.page_ids.size() - 1) ? table.filled_in_last : page_size_;
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

    // Release the persistent device mirrors, then drop the host tables.
#ifdef USE_CUDA
    free_page_table_mirrors(it->second);
#endif
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

    // Free the persistent device mirrors before dropping the host tables.
#ifdef USE_CUDA
    for (auto& kv : seq_tables_)
        free_page_table_mirrors(kv.second);
#endif
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
