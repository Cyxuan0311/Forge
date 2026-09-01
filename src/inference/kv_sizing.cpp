#include "forge/kv_sizing.h"

#include <algorithm>
#include <cmath>

#ifdef USE_CUDA
#    include "forge/backend.h"
#endif

namespace forge {
namespace {

// Held back from the reported free memory for non-KV allocations (activations,
// logits, cuBLAS/cuDNN workspace). Without this the auto-sized KV pool can
// starve the rest of the forward pass.
constexpr size_t kReservedBytes = 256ull * 1024ull * 1024ull;  // 256 MB

// Sanity ceilings. They only matter on absurdly large devices; the real limit
// is almost always the byte budget.
constexpr int kMaxNumSeqsCap = 512;

}  // namespace

size_t kv_token_bytes(KVCacheDType type_k, KVCacheDType type_v, int kv_dim) {
    if (kv_dim <= 0)
        return 0;
    return KVCache::block_nbytes(type_k, kv_dim) + KVCache::block_nbytes(type_v, kv_dim);
}

int kv_layer_effective_len(const KVSizingParams& p, int layer) {
    if (p.max_seq_len <= 0)
        return 0;
    if (layer >= 0 && layer < static_cast<int>(p.policies.size()) &&
        p.policies[layer] == KVLayerPolicy::SlidingWindow && p.swa_window > 0) {
        return std::min(p.max_seq_len, p.swa_window);
    }
    return p.max_seq_len;
}

size_t per_seq_kv_bytes(const KVSizingParams& p) {
    if (p.num_layers <= 0 || p.max_seq_len <= 0 || p.kv_dims.empty())
        return 0;

    size_t total = 0;
    for (int l = 0; l < p.num_layers; ++l) {
        const int kv_dim = (l < static_cast<int>(p.kv_dims.size())) ? p.kv_dims[l] : p.kv_dims[0];
        total += static_cast<size_t>(kv_layer_effective_len(p, l)) *
                 kv_token_bytes(p.type_k, p.type_v, kv_dim);
    }
    return total;
}

size_t kv_page_bytes(const KVSizingParams& p) {
    if (p.num_layers <= 0 || p.kv_dims.empty() || p.page_size <= 0)
        return 0;

    size_t total = 0;
    for (int l = 0; l < p.num_layers; ++l) {
        const int kv_dim = (l < static_cast<int>(p.kv_dims.size())) ? p.kv_dims[l] : p.kv_dims[0];
        total += static_cast<size_t>(p.page_size) * kv_token_bytes(p.type_k, p.type_v, kv_dim);
    }
    return total;
}

size_t query_free_device_bytes(DeviceType device) {
#ifdef USE_CUDA
    if (device != DeviceType::CUDA)
        return 0;

    auto backend = Backend::create_cuda();
    if (!backend)
        return 0;

    const size_t free_bytes = backend->device_memory_free();
    if (free_bytes <= kReservedBytes)
        return 0;
    return free_bytes - kReservedBytes;
#else
    (void)device;
    return 0;
#endif
}

KVSizingResult auto_size_kv(const KVSizingParams& p, size_t budget_bytes, double safety) {
    KVSizingResult r;
    r.per_seq_bytes = per_seq_kv_bytes(p);
    r.page_bytes = kv_page_bytes(p);

    if (!(safety > 0.0))
        safety = 0.9;
    if (safety > 1.0)
        safety = 1.0;
    r.budget_bytes = static_cast<size_t>(static_cast<double>(budget_bytes) * safety);

    if (r.per_seq_bytes == 0 || r.budget_bytes == 0 || p.max_seq_len <= 0 || p.page_size <= 0) {
        // Nothing sensible can be derived; the caller keeps its own default.
        r.max_num_seqs = 0;
        r.auto_sized = false;
        return r;
    }

    const size_t seqs = r.budget_bytes / r.per_seq_bytes;
    r.max_num_seqs = static_cast<int>(std::min<size_t>(std::max<size_t>(seqs, 1), kMaxNumSeqsCap));

    // Pages needed by Full layers to hold every admitted sequence at full length.
    const double pages_needed = static_cast<double>(r.max_num_seqs) *
                                static_cast<double>(p.max_seq_len) /
                                static_cast<double>(p.page_size);
    size_t pages = static_cast<size_t>(std::ceil(pages_needed));

    // Also honour the raw byte budget (matters when a page spans many layers).
    if (r.page_bytes > 0)
        pages = std::min(pages, r.budget_bytes / r.page_bytes);

    r.max_pages_per_layer = static_cast<int>(std::max<size_t>(pages, 1));
    r.auto_sized = true;
    return r;
}

KVSizingResult auto_size_kv_for_device(const KVSizingParams& p, double safety) {
    return auto_size_kv(p, query_free_device_bytes(p.device), safety);
}

}  // namespace forge
