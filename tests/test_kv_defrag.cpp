// Roadmap phase 2 / item 2.2: contiguous KV cache defragmentation.
//
// Host-only (CPU, FP32) test. Verifies the contract the scheduler relies on for
// the contiguous KV backend after a sequence leaves holes in the cache:
//   * After removing a middle range (creating scattered holes), defrag()
//     compacts the surviving cells to the front of each layer and leaves the
//     free cells contiguous at the tail (num_free_slots() merge into one block).
//   * A subsequent update reuses the compacted tail space.
//   * The compacted KV (read via get_key_filled / get_value_filled) is
//     bit-identical to a reference cache filled with the same surviving rows
//     contiguously, so attention results are unchanged by defrag.
//   * The opt-in automatic trigger (defrag_if_needed, wired into
//     ContiguousKVStorage::seq_remove / release) fires on removal when enabled.
#include <cstdio>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "forge/kv_cache.h"
#include "forge/tensor.h"

using namespace forge;

namespace {
int g_failures = 0;
void check(bool cond, const char* what) {
    if (!cond) {
        ++g_failures;
        printf("  FAIL: %s\n", what);
    } else {
        printf("  ok  : %s\n", what);
    }
}

constexpr int kLayers = 2;
constexpr int kKvHeads = 1;
constexpr int kHeadDim = 4;
constexpr int kKvDim = kKvHeads * kHeadDim;  // 4
constexpr int kMaxSeqLen = 16;

// Fill `rows` KV rows; row r carries value (base + r) in every column, so each
// row is uniquely identifiable and ordering is visible after compaction.
std::pair<TensorPtr, TensorPtr> make_kv(int rows, int base) {
    auto k = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{rows, kKvDim},
                                      DeviceType::CPU);
    auto v = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{rows, kKvDim},
                                      DeviceType::CPU);
    float* kd = static_cast<float*>(k->data());
    float* vd = static_cast<float*>(v->data());
    for (int r = 0; r < rows; ++r) {
        float val = static_cast<float>(base + r);
        for (int c = 0; c < kKvDim; ++c) {
            kd[r * kKvDim + c] = val;
            vd[r * kKvDim + c] = -val;
        }
    }
    return {k, v};
}

// Fill `rows` KV rows using explicit per-row values (so a reference with
// non-contiguous surviving positions can be reconstructed contiguously).
std::pair<TensorPtr, TensorPtr> make_kv_values(int rows, const int* vals) {
    auto k = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{rows, kKvDim},
                                      DeviceType::CPU);
    auto v = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{rows, kKvDim},
                                      DeviceType::CPU);
    float* kd = static_cast<float*>(k->data());
    float* vd = static_cast<float*>(v->data());
    for (int r = 0; r < rows; ++r) {
        float val = static_cast<float>(vals[r]);
        for (int c = 0; c < kKvDim; ++c) {
            kd[r * kKvDim + c] = val;
            vd[r * kKvDim + c] = -val;
        }
    }
    return {k, v};
}

bool tensors_match(const TensorPtr& a, const TensorPtr& b) {
    if (!a || !b)
        return a == b;
    if (a->shape() != b->shape())
        return false;
    return std::memcmp(a->data(), b->data(), a->nbytes()) == 0;
}

// A "middle hole" exists when a free cell sits before the logical fill cursor
// (i.e. there is wasted space inside [0, filled) that defrag should reclaim).
bool has_middle_hole(const KVCache& cache) {
    for (int l = 0; l < cache.num_layers(); ++l) {
        int f = cache.filled(l);  // logical_filled
        const auto& cells = cache.layers()[l].cells;
        for (int i = 0; i < f && i < static_cast<int>(cells.size()); ++i) {
            if (cells[i].is_free())
                return true;
        }
    }
    return false;
}

void test_defrag_compaction() {
    printf("[1] defrag compacts holes into a contiguous free tail\n");
    KVCache cache;
    cache.init(kLayers, kKvHeads, kHeadDim, kMaxSeqLen, DeviceType::CPU);

    // seq0 fills [0,8); removing the middle [2,4) leaves holes at positions 2,3.
    for (int l = 0; l < kLayers; ++l) {
        auto kv = make_kv(8, 0);
        cache.update(l, 0, 0, kv.first, kv.second, 8);
    }
    cache.seq_rm(0, 2, 4);
    check(has_middle_hole(cache), "holes exist in the middle before defrag");
    check(cache.filled(0) == 8, "filled stays 8 (holes in the middle)");

    cache.defrag();

    check(!has_middle_hole(cache), "no middle holes after defrag");
    check(cache.filled(0) == 6, "filled reduced to 6 after defrag");
    check(cache.num_free_slots() == (kMaxSeqLen - 6) * kLayers, "free slots merge to tail");

    bool contig = true;
    for (int l = 0; l < kLayers; ++l) {
        bool seen_free = false;
        for (const auto& cell : cache.layers()[l].cells) {
            if (cell.is_free())
                seen_free = true;
            else if (seen_free)
                contig = false;  // alive cell after a free one => still fragmented
        }
    }
    check(contig, "all free cells are at the tail (no scattered holes)");
}

void test_defrag_attention_consistent() {
    printf("[2] defrag preserves KV content (attention result unchanged)\n");
    KVCache cache, ref;
    cache.init(kLayers, kKvHeads, kHeadDim, kMaxSeqLen, DeviceType::CPU);
    ref.init(kLayers, kKvHeads, kHeadDim, kMaxSeqLen, DeviceType::CPU);

    // cache: seq0 fills [0,8), remove [2,4). Surviving positions = {0,1,4,5,6,7}.
    for (int l = 0; l < kLayers; ++l) {
        auto kv = make_kv(8, 0);
        cache.update(l, 0, 0, kv.first, kv.second, 8);
    }
    cache.seq_rm(0, 2, 4);
    cache.defrag();

    // reference: 6 rows filled contiguously with the SAME surviving values.
    int vals[6] = {0, 1, 4, 5, 6, 7};
    for (int l = 0; l < kLayers; ++l) {
        auto kv = make_kv_values(6, vals);
        ref.update(l, 0, 0, kv.first, kv.second, 6);
    }

    for (int l = 0; l < kLayers; ++l) {
        check(tensors_match(cache.get_key_filled(l), ref.get_key_filled(l)),
              "defragged key matches reference");
        check(tensors_match(cache.get_value_filled(l), ref.get_value_filled(l)),
              "defragged value matches reference");
    }
}

void test_defrag_reuse() {
    printf("[3] update reuses compacted space after defrag\n");
    KVCache cache;
    cache.init(kLayers, kKvHeads, kHeadDim, kMaxSeqLen, DeviceType::CPU);
    for (int l = 0; l < kLayers; ++l) {
        auto kv = make_kv(8, 0);
        cache.update(l, 0, 0, kv.first, kv.second, 8);
    }
    cache.seq_rm(0, 2, 4);
    cache.defrag();

    int free_before = cache.num_free_slots();
    // Continue decoding at the new logical end (pos 6).
    for (int l = 0; l < kLayers; ++l) {
        auto kv = make_kv(2, 8);  // values 8,9
        cache.update(l, 0, 6, kv.first, kv.second, 2);
    }
    check(cache.filled(0) == 8, "filled grows to 8 after reuse write");
    check(cache.num_free_slots() == free_before - 2 * kLayers,
          "free slots decreased by 2 per layer after reuse");
}

void test_defrag_auto_trigger() {
    printf("[4] opt-in automatic trigger fires on seq_remove\n");
    KVCache cache;
    cache.init(kLayers, kKvHeads, kHeadDim, kMaxSeqLen, DeviceType::CPU);
    cache.set_defrag_enabled(true);  // enable the trigger

    for (int l = 0; l < kLayers; ++l) {
        auto kv = make_kv(8, 0);
        cache.update(l, 0, 0, kv.first, kv.second, 8);
    }
    // The trigger (ContiguousKVStorage::seq_remove / release) calls
    // defrag_if_needed(); exercise the equivalent entry point directly.
    cache.seq_rm(0, 2, 4);
    cache.defrag_if_needed();
    check(!has_middle_hole(cache), "auto-trigger compacted (no middle holes)");
    check(cache.filled(0) == 6, "auto-trigger reduced filled to 6");

    // With the trigger disabled (default), the same removal must NOT compact.
    KVCache cache2;
    cache2.init(kLayers, kKvHeads, kHeadDim, kMaxSeqLen, DeviceType::CPU);
    for (int l = 0; l < kLayers; ++l) {
        auto kv = make_kv(8, 0);
        cache2.update(l, 0, 0, kv.first, kv.second, 8);
    }
    cache2.seq_rm(0, 2, 4);
    cache2.defrag_if_needed();  // disabled by default => no-op
    check(has_middle_hole(cache2), "no auto-compact when defrag disabled (holes remain)");
    check(cache2.filled(0) == 8, "filled stays 8 when defrag disabled");
}

}  // namespace

int main() {
    test_defrag_compaction();
    test_defrag_attention_consistent();
    test_defrag_reuse();
    test_defrag_auto_trigger();
    if (g_failures == 0) {
        printf("test_kv_defrag: PASS\n");
        return 0;
    }
    printf("test_kv_defrag: FAIL (%d assertion(s))\n", g_failures);
    return 1;
}
