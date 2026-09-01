// Roadmap phase 1 / item 1.1: KV page dynamic offload (GPU <-> pinned host) +
// scheduler swap-out instead of rejection.
//
// Host-only test (no GPU required). It pins down the contract the scheduler
// relies on when it preempts a request under KV pressure:
//   * offload_seq() moves a whole sequence's pages into the host swap pool and
//     releases their primary-memory storage WITHOUT freeing the page slots
//     (free-page accounting must not change);
//   * bring_back_seq() restores the bytes bit-exactly, so attention results
//     are identical to a run that never swapped;
//   * evicted pages keep their slot, making repeated offloads idempotent;
//   * releasing a sequence (or reset()) drops the host copies, so the swap
//     pool never leaks.
#include <cstdio>
#include <cstring>
#include <vector>

#include "forge/kv_block_swapper.h"
#include "forge/kv_storage.h"

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
constexpr int kKvDim = 4;
constexpr int kPageSize = 8;
constexpr int kMaxSeqLen = 16;
constexpr int kMaxSeqs = 2;

// Deterministic, finite pattern: rows are distinct and stable across fills, so
// a restored tensor can be compared against a freshly filled expectation.
// row_offset shifts the pattern so a continuation write (pos > 0) matches the
// same absolute row as the fully filled expectation.
void fill_kv(float* k, float* v, int rows, int kv_dim, int row_offset = 0) {
    for (int r = 0; r < rows; ++r) {
        int abs_r = row_offset + r;
        for (int c = 0; c < kv_dim; ++c) {
            k[r * kv_dim + c] = 0.001f * (abs_r * kv_dim + c) + 0.5f;
            v[r * kv_dim + c] = -0.25f - 0.1f * abs_r - 0.01f * c;
        }
    }
}

bool tensor_matches(const TensorPtr& t, const float* expect, int rows, int kv_dim) {
    if (!t)
        return false;
    if (t->shape() != std::vector<int64_t>{rows, kv_dim})
        return false;
    return std::memcmp(t->data(), expect, sizeof(float) * static_cast<size_t>(rows) * kv_dim) == 0;
}

bool tensors_match(const TensorPtr& a, const TensorPtr& b) {
    if (!a || !b)
        return a == b;
    if (a->shape() != b->shape())
        return false;
    return std::memcmp(a->data(), b->data(), a->nbytes()) == 0;
}

void test_swapper_roundtrip() {
    printf("[1] KVBlockSwapper byte round-trip\n");

    KVBlockSwapper sw;
    const uint8_t k[7] = {1, 2, 3, 4, 5, 6, 7};
    const uint8_t v[5] = {9, 8, 7, 6, 5};

    check(sw.offload(0, 3, k, sizeof(k), v, sizeof(v)), "offload stores the bytes");
    check(sw.has(0, 3), "has() sees the page");
    size_t kb = 0, vb = 0;
    check(sw.size(0, 3, &kb, &vb), "size() reports sizes");
    check(kb == sizeof(k) && vb == sizeof(v), "sizes are preserved");
    check(sw.nbytes() == sizeof(k) + sizeof(v), "nbytes() tracks host usage");

    // Re-offload with the same sizes refreshes the copy.
    check(sw.offload(0, 3, k, sizeof(k), v, sizeof(v)), "re-offload with same size is accepted");
    // A size mismatch on an existing slot is rejected, never corrupted.
    check(!sw.offload(0, 3, k, 3, v, sizeof(v)), "size mismatch is rejected");

    uint8_t k2[7] = {0}, v2[5] = {0};
    check(sw.bring_back(0, 3, k2, sizeof(k2), v2, sizeof(v2)), "bring_back restores");
    check(std::memcmp(k, k2, sizeof(k)) == 0 && std::memcmp(v, v2, sizeof(v)) == 0,
          "restored bytes are bit-exact");
    check(!sw.bring_back(0, 4, k2, sizeof(k2), v2, sizeof(v2)), "unknown page refuses restore");

    sw.drop(0, 3);
    check(!sw.has(0, 3), "drop removes the host copy");
    check(sw.nbytes() == 0, "host bytes drop to zero after drop");

    sw.reset();
    check(sw.num_pages() == 0 && sw.nbytes() == 0, "reset clears the pool");
}

// Small helper: initialize a freshly created PagedKVStorage with a tight page pool.
bool init_storage(PagedKVStorage& storage) {
    KVCacheTypeConfig cfg;  // FP32/FP32 by default
    return storage.init(kLayers, {kKvDim, kKvDim}, kMaxSeqLen, DeviceType::CPU, cfg, kPageSize,
                        kMaxSeqs);
}

void test_init_state(PagedKVStorage& storage) {
    printf("[2] initial state of the page pool\n");

    // 2 layers * (max_seqs * max_seq_len / page_size) = 2 * 4 = 8 pages.
    check(storage.total_page_capacity() == 8, "total_page_capacity() == layers * pages");
    check(storage.num_free_pages() == 8, "all pages start free");
    check(storage.num_device_pages_in_use() == 0, "no device pages in use");
    check(storage.num_offloaded_pages() == 0, "no offloaded pages");
    check(storage.num_brought_back_pages() == 0, "no brought-back pages");
    check(storage.host_pool_bytes() == 0, "host swap pool is empty");
    check(storage.seq_num_pages(0) == 0, "unknown sequence has no pages");
    check(!storage.offload_seq(99), "offloading an unknown sequence fails");
    check(!storage.bring_back_seq(99), "bringing back an unknown sequence fails");
}

void test_offload_bring_back(PagedKVStorage& storage) {
    printf("[3] write -> offload_seq -> device pressure released\n");

    std::vector<float> k(10 * kKvDim), v(10 * kKvDim);
    fill_kv(k.data(), v.data(), 10, kKvDim);

    check(storage.write_kv_seq(0, 0, 0, 10, k.data(), v.data()), "write_kv_seq accepts seq0");
    check(storage.seq_num_pages(0) == 2, "seq0 holds 2 pages (10 tokens @ page_size 8)");
    check(storage.num_device_pages_in_use() == 2, "2 device pages in use");
    check(storage.num_free_pages() == 6, "6 pages still free");
    check(storage.seq_filled(0, 0) == 10, "seq0 filled == 10");

    TensorPtr baseline_k = storage.read_key_seq(0, 0);
    TensorPtr baseline_v = storage.read_value_seq(0, 0);
    check(tensor_matches(baseline_k, k.data(), 10, kKvDim), "baseline K matches the write");
    check(tensor_matches(baseline_v, v.data(), 10, kKvDim), "baseline V matches the write");

    // ---- Offload ----
    check(storage.offload_seq(0), "offload_seq(0) succeeds");
    check(storage.num_offloaded_pages() == 2, "2 pages moved to the host pool");
    check(storage.num_device_pages_in_use() == 0, "device pages in use drops to 0");
    check(storage.num_free_pages() == 6, "free-page count is unchanged (slots stay reserved)");
    check(storage.host_pool_bytes() > 0, "host swap pool now holds bytes");
    check(storage.seq_num_pages(0) == 2, "seq0 page table is untouched by eviction");
    check(!storage.offload_seq(0), "repeated offload is a no-op (idempotent)");
    check(storage.num_offloaded_pages() == 2, "no double-count on repeated offload");
}

void test_restore_is_bit_exact(PagedKVStorage& storage) {
    printf("[4] bring_back_seq -> attention-identical restore\n");

    check(storage.bring_back_seq(0), "bring_back_seq(0) succeeds");
    check(storage.num_brought_back_pages() == 2, "2 pages restored");
    check(storage.num_device_pages_in_use() == 2, "device pages in use back to 2");
    check(storage.host_pool_bytes() > 0, "host copy is retained after restore");

    // Bit-exact: the restored data must be identical to what attention saw
    // before the eviction, for both K and V. The fill is deterministic, so the
    // freshly filled expectation equals the pre-eviction baseline (verified in
    // [3] to match the write itself).
    std::vector<float> expect_k(10 * kKvDim), expect_v(10 * kKvDim);
    fill_kv(expect_k.data(), expect_v.data(), 10, kKvDim);
    check(tensor_matches(storage.read_key_seq(0, 0), expect_k.data(), 10, kKvDim),
          "restored K is bit-exact (== pre-eviction value)");
    check(tensor_matches(storage.read_value_seq(0, 0), expect_v.data(), 10, kKvDim),
          "restored V is bit-exact (== pre-eviction value)");
}

void test_extend_after_restore(PagedKVStorage& storage) {
    printf("[5] decode continues after restore + re-eviction\n");

    // The scheduler's suspended sequence resumes: extend 10 -> 16 tokens.
    // The continuation starts at absolute row 10, so the pattern must line up
    // with the fully filled expectation below.
    std::vector<float> k(6 * kKvDim), v(6 * kKvDim);
    fill_kv(k.data(), v.data(), 6, kKvDim, /*row_offset=*/10);
    check(storage.write_kv_seq(0, 0, 10, 6, k.data(), v.data()), "extension write succeeds");
    check(storage.seq_filled(0, 0) == 16, "seq0 filled == 16");

    // The whole 16-token sequence lives in 2 full pages.
    std::vector<float> expect_k(16 * kKvDim), expect_v(16 * kKvDim);
    fill_kv(expect_k.data(), expect_v.data(), 16, kKvDim);

    check(storage.offload_seq(0), "full sequence offloads");
    check(storage.num_offloaded_pages() == 4, "offload count accumulates to 4");
    check(storage.num_device_pages_in_use() == 0, "all of seq0 is off device");

    check(storage.bring_back_seq(0), "full sequence restores");
    check(storage.num_brought_back_pages() == 4, "bring-back count accumulates to 4");
    check(tensor_matches(storage.read_key_seq(0, 0), expect_k.data(), 16, kKvDim),
          "restored 16-token K is bit-exact");
    check(tensor_matches(storage.read_value_seq(0, 0), expect_v.data(), 16, kKvDim),
          "restored 16-token V is bit-exact");
}

void test_multi_layer_isolation(PagedKVStorage& storage) {
    printf("[6] multi-layer offload + isolation from other sequences\n");

    // seq0 gets layer-1 data; seq1 gets a page on both layers.
    std::vector<float> k0(10 * kKvDim), v0(10 * kKvDim);
    fill_kv(k0.data(), v0.data(), 10, kKvDim);
    std::vector<float> k1(8 * kKvDim), v1(8 * kKvDim);
    fill_kv(k1.data(), v1.data(), 8, kKvDim);

    check(storage.write_kv_seq(1, 0, 0, 10, k0.data(), v0.data()), "seq0 layer1 write");
    check(storage.write_kv_seq(0, 1, 0, 8, k1.data(), v1.data()), "seq1 layer0 write");
    check(storage.write_kv_seq(1, 1, 0, 8, k1.data(), v1.data()), "seq1 layer1 write");

    TensorPtr seq1_k_baseline = storage.read_key_seq(0, 1);
    TensorPtr seq1_v_baseline = storage.read_value_seq(0, 1);

    check(storage.seq_num_pages(0) == 4, "seq0 spans 4 pages (2 layers * 2)");
    check(storage.num_device_pages_in_use() == 6, "6 device pages in use (4 seq0 + 2 seq1)");

    // Evicting seq0 must not disturb seq1's on-device pages.
    check(storage.offload_seq(0), "seq0 offloads across both layers");
    check(storage.num_offloaded_pages() == 8, "4 more pages offloaded (4 -> 8)");
    check(storage.num_device_pages_in_use() == 2, "only seq1 remains on device");
    check(tensors_match(storage.read_key_seq(0, 1), seq1_k_baseline),
          "seq1 K untouched while seq0 is evicted");
    check(tensors_match(storage.read_value_seq(0, 1), seq1_v_baseline),
          "seq1 V untouched while seq0 is evicted");

    check(storage.bring_back_seq(0), "seq0 restores across both layers");
    check(storage.num_brought_back_pages() == 8, "4 more pages restored (4 -> 8)");
    check(storage.num_device_pages_in_use() == 6, "all device pages back in use");
    check(tensor_matches(storage.read_key_seq(1, 0), k0.data(), 10, kKvDim),
          "seq0 layer1 K restored bit-exact");
    check(tensor_matches(storage.read_value_seq(1, 0), v0.data(), 10, kKvDim),
          "seq0 layer1 V restored bit-exact");
    check(tensors_match(storage.read_key_seq(0, 1), seq1_k_baseline),
          "seq1 K still intact after seq0 restore");
}

void test_release_and_reset_drop_host(PagedKVStorage& storage) {
    printf("[7] release / reset drop host copies\n");

    const size_t pool_before = storage.host_pool_bytes();
    check(pool_before > 0, "host pool holds data before cleanup");

    // Releasing seq0 frees its evicted pages; free_page() drops the host copy.
    storage.release(0);
    check(!storage.has_seq(0), "seq0 is gone");
    check(storage.num_device_pages_in_use() == 2, "only seq1 pages remain in use");
    const size_t after_release = storage.host_pool_bytes();
    check(after_release < pool_before, "release(seq0) dropped its host copies");
    check(after_release == 0, "seq1 was never evicted, so the pool is now empty");
    check(storage.num_free_pages() == 6, "seq0's 4 pages are back on the free list");

    // Reset returns everything to the free list and zeroes the counters.
    storage.reset();
    check(storage.num_free_pages() == 8, "reset restores the full free list");
    check(storage.num_offloaded_pages() == 0, "reset zeroes the offload counter");
    check(storage.num_brought_back_pages() == 0, "reset zeroes the bring-back counter");
    check(storage.host_pool_bytes() == 0, "reset empties the host swap pool");
}

}  // namespace

int main() {
    test_swapper_roundtrip();

    PagedKVStorage storage;
    check(init_storage(storage), "PagedKVStorage::init succeeds on CPU");
    test_init_state(storage);
    test_offload_bring_back(storage);
    test_restore_is_bit_exact(storage);
    test_extend_after_restore(storage);
    test_multi_layer_isolation(storage);
    test_release_and_reset_drop_host(storage);

    printf("kv_offload: failures=%d\n", g_failures);
    printf(g_failures == 0 ? "RESULT: PASS\n" : "RESULT: FAIL\n");
    return g_failures == 0 ? 0 : 1;
}
