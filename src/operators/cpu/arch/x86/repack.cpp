#include <mutex>
#include <unordered_map>

#include "repack.h"
#include "gemm_microkernel.h"  // forge::cpu::repack_q4_0_weights

#ifdef USE_AVX2

namespace {

struct RepackEntry {
    uint8_t* data = nullptr;
    size_t size = 0;
    int64_t K = 0;
    int64_t N = 0;
};

static std::unordered_map<const void*, RepackEntry>& repack_cache() {
    static std::unordered_map<const void*, RepackEntry> cache;
    return cache;
}

static std::mutex& repack_mutex() {
    static std::mutex mtx;
    return mtx;
}

}  // namespace

namespace forge {
namespace ops {

const uint8_t* get_repacked_q4_0(const void* orig_data, int64_t K, int64_t N) {
    if (N < 4) return nullptr;  // too small for RM=4 tile

    std::lock_guard<std::mutex> lock(repack_mutex());
    auto& cache = repack_cache();
    auto it = cache.find(orig_data);
    if (it != cache.end()) {
        // Validate dimensions match
        if (it->second.K == K && it->second.N == N) {
            return it->second.data;
        }
        // Dimensions changed (shouldn't happen for same tensor, but handle it)
        delete[] it->second.data;
        cache.erase(it);
    }

    // Repack — forge::cpu::repack_q4_0_weights is in gemm_microkernel.h
    auto result = forge::cpu::repack_q4_0_weights(
        static_cast<const uint8_t*>(orig_data), K, N);
    if (!result.first) return nullptr;

    RepackEntry entry;
    entry.data = result.first;
    entry.size = result.second;
    entry.K = K;
    entry.N = N;
    cache[orig_data] = entry;
    return entry.data;
}

}  // namespace ops
}  // namespace forge

#endif  // USE_AVX2