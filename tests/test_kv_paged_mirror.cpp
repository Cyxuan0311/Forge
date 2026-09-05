// Phase 2-D: persistent per-(seq, layer) device page-table mirror.
//
// Before 2-D, upload_seq_page_table() copied the ENTIRE sequence page table
// into one shared device staging buffer on every call. Consequences:
//   * per-token H2D traffic grew linearly with context length;
//   * concurrent sequences clobbered each other through the shared buffer.
//
// The fix keeps one persistent device mirror per (seq, layer) inside
// SequencePageTable. The hot decode path copies only the appended pages;
// capacity growth reallocates; prefix-changing operations (seq_remove /
// seq_share) mark the table dirty and force a full re-upload.
//
// This test drives the real CUDA path of PagedKVStorage and compares the
// device mirror contents against an exact host-side simulation of the page
// allocator (free lists pop from the back, same rule as alloc_page()), across:
//   1. append-only growth crossing page boundaries,
//   2. mirror capacity growth (realloc) beyond the initial allocation,
//   3. two live sequences with distinct, stable mirrors (no clobbering),
//   4. seq_remove() (prefix-change) followed by reuse of the freed pages.
//
// Skips gracefully when no CUDA device is available.

#include <array>
#include <cstdio>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

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

bool check_cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        printf("  FAIL: %s (cudaError %d: %s)\n", what, (int)e, cudaGetErrorString(e));
        ++g_failures;
        return false;
    }
    return true;
}

constexpr int kLayers = 2;
constexpr int kDim = 4;
constexpr int kPage = 4;      // tokens per page (small so tests grow quickly)
constexpr int kMaxSeqLen = 128;
constexpr int kMaxSeqs = 2;
// pages per layer pool: ceil(max_seqs * max_seq_len / page_size)
constexpr int kPoolPages = (kMaxSeqs * kMaxSeqLen + kPage - 1) / kPage;  // 64

bool cuda_available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

// ---- exact host-side mirror of PagedKVStorage's page allocator ----
// NOTE: every layer has its own independent page pool; allocs pop from the
// back of that pool's free list (same rule as alloc_page()).
struct AllocSim {
    std::array<std::vector<int>, kLayers> free_pool;         // per layer
    std::array<std::array<std::vector<int>, kMaxSeqs>, kLayers> pages;  // [layer][seq]

    void init() {
        for (auto& pool : free_pool) {
            pool.clear();
            for (int i = 0; i < kPoolPages; ++i) pool.push_back(i);  // 0..63 per layer
        }
        for (auto& by_seq : pages)
            for (auto& p : by_seq) p.clear();
    }

    // Replicate ensure_seq_capacity(): pop pages from the layer's free list
    // (back) until the sequence can hold pos+len slots.
    void ensure(int layer, int seq, int pos, int len) {
        auto& pool = free_pool[layer];
        auto& list = pages[layer][seq];
        while ((int)list.size() * kPage < pos + len) {
            int id = pool.back();
            pool.pop_back();
            list.push_back(id);
        }
    }

    // Replicate seq_remove(): drop page positions [first, last] of the layer's
    // list, pushing the freed ids back onto the layer pool in the order they
    // were encountered.
    void remove(int layer, int seq, int p0, int p1) {
        auto& pool = free_pool[layer];
        auto& list = pages[layer][seq];
        std::vector<int> remaining;
        int first = (int)(p0 / kPage);
        int last = (int)((p1 - 1) / kPage);
        for (int p = 0; p < (int)list.size(); ++p) {
            if (p >= first && p <= last)
                pool.push_back(list[p]);
            else
                remaining.push_back(list[p]);
        }
        list = std::move(remaining);
    }
};

// Build a device FP32 K/V buffer holding `rows` deterministic rows and write
// them into the storage (the CUDA scatter path consumes device pointers).
void write_rows(PagedKVStorage& s, AllocSim& sim, int layer, int seq, int pos, int len) {
    sim.ensure(layer, seq, pos, len);

    std::vector<float> kv((size_t)len * kDim);
    for (int r = 0; r < len; ++r) {
        int abs_r = pos + r;
        for (int c = 0; c < kDim; ++c) {
            kv[(size_t)r * kDim + c] = 0.001f * (abs_r * kDim + c) + 0.5f;
        }
    }
    float *d_k = nullptr, *d_v = nullptr;
    const size_t bytes = kv.size() * sizeof(float);
    if (!check_cuda(cudaMalloc(&d_k, bytes), "cudaMalloc d_k") ||
        !check_cuda(cudaMalloc(&d_v, bytes), "cudaMalloc d_v"))
        return;
    check_cuda(cudaMemcpy(d_k, kv.data(), bytes, cudaMemcpyHostToDevice), "H2D k");
    check_cuda(cudaMemcpy(d_v, kv.data(), bytes, cudaMemcpyHostToDevice), "H2D v");

    bool ok = s.write_kv_seq(layer, seq, pos, len, d_k, d_v);
    check(ok, "write_kv_seq accepted");

    cudaFree(d_k);
    cudaFree(d_v);
}

// Read the device mirror back and compare against the simulation.
void verify_mirror(PagedKVStorage& s, const AllocSim& sim, int layer, int seq) {
    const int32_t* d = s.upload_seq_page_table(layer, seq);
    const std::vector<int>& expect = sim.pages[layer][seq];
    char msg[128];

    if (expect.empty()) {
        std::snprintf(msg, sizeof(msg), "upload(L%d,S%d) n=0 -> nullptr", layer, seq);
        check(d == nullptr, msg);
        return;
    }
    std::snprintf(msg, sizeof(msg), "upload(L%d,S%d) returns ptr", layer, seq);
    check(d != nullptr, msg);

    std::vector<int32_t> got(expect.size(), -1);
    const size_t bytes = expect.size() * sizeof(int32_t);
    check_cuda(cudaMemcpy(got.data(), d, bytes, cudaMemcpyDeviceToHost),
               "D2H mirror copy");
    std::snprintf(msg, sizeof(msg), "mirror L%d S%d matches host simulation", layer, seq);
    bool same = (got.size() == expect.size());
    if (same) {
        for (size_t i = 0; i < expect.size(); ++i)
            if (got[i] != expect[i]) { same = false; break; }
    }
    if (!same) {
        printf("    got    :");
        for (size_t i = 0; i < got.size(); ++i) printf(" %d", got[i]);
        printf("\n    expect :");
        for (size_t i = 0; i < expect.size(); ++i) printf(" %d", expect[i]);
        printf("\n");
    }
    check(same, msg);
}

}  // namespace

int main() {
    if (!cuda_available()) {
        printf("SKIP: no CUDA device available\n");
        return 0;
    }

    printf("=== 2-D: per-(seq,layer) device page-table mirror ===\n");

    PagedKVStorage storage;
    KVCacheTypeConfig cfg;  // FP32/FP32
    check(storage.init(kLayers, {kDim, kDim}, kMaxSeqLen, DeviceType::CUDA, cfg, kPage, kMaxSeqs),
          "init CUDA paged storage");

    AllocSim sim;
    sim.init();

    // [1] append-only growth across page boundaries.
    printf("[1] append growth (single page at a time)\n");
    for (int l = 0; l < kLayers; ++l) {
        write_rows(storage, sim, l, 0, 0, 3);    // page 0
        verify_mirror(storage, sim, l, 0);
        write_rows(storage, sim, l, 0, 3, 13);   // reaches token 16 -> pages 2..4
        verify_mirror(storage, sim, l, 0);
    }

    // [2] cross the initial mirror capacity (d_cap starts at 16 entries) so the
    // grow-only realloc path runs; the mirror must stay consistent.
    printf("[2] capacity growth beyond initial allocation\n");
    {
        const int32_t* ptr_before = storage.upload_seq_page_table(0, 0);
        write_rows(storage, sim, 0, 0, 16, 100);  // reaches token 116 -> 29 pages > 16
        verify_mirror(storage, sim, 0, 0);
        const int32_t* ptr_after = storage.upload_seq_page_table(0, 0);
        check(ptr_after != ptr_before, "mirror reallocated (device ptr changed)");
    }
    write_rows(storage, sim, 1, 0, 16, 100);
    verify_mirror(storage, sim, 1, 0);

    // [3] two live sequences -> distinct mirrors that never clobber.
    printf("[3] two live sequences use independent mirrors\n");
    write_rows(storage, sim, 0, 1, 0, 5);   // seq1: 2 pages
    write_rows(storage, sim, 1, 1, 0, 5);
    verify_mirror(storage, sim, 0, 1);
    verify_mirror(storage, sim, 1, 1);
    {
        const int32_t* a = storage.upload_seq_page_table(0, 0);
        const int32_t* b = storage.upload_seq_page_table(0, 1);
        check(a != b, "seq0/seq1 mirrors are distinct pointers");
        const int32_t* stable = storage.upload_seq_page_table(0, 1);
        check(b == stable, "mirror pointer stable while capacity unchanged");
    }

    // [4] seq_remove in the middle of the table (prefix-change) followed by
    // reuse of the freed pages -> the dirty full re-upload must hold.
    printf("[4] seq_remove marks dirty; freed pages are reused correctly\n");
    // seq_remove() spans ALL layers of a sequence at once.
    for (int l = 0; l < kLayers; ++l)
        sim.remove(l, 0, 32, 48);  // removes page positions [8, 12)
    check(storage.seq_remove(0, 32, 48), "seq_remove accepted");
    for (int l = 0; l < kLayers; ++l)
        verify_mirror(storage, sim, l, 0);
    // Overwrite the whole sequence again (longer than before): allocation
    // reuses the pages freed by seq_remove, exercising the dirty path with
    // stale ids absent from the pre-fix implementation.
    for (int l = 0; l < kLayers; ++l) {
        write_rows(storage, sim, l, 0, 0, 120);  // 30 pages; reuses freed slots
        verify_mirror(storage, sim, l, 0);
    }
    // seq1 must be untouched by seq0's churn.
    verify_mirror(storage, sim, 0, 1);
    verify_mirror(storage, sim, 1, 1);

    // Seq release frees its mirrors; seq1's mirror must still be valid.
    storage.release(1);
    check(storage.upload_seq_page_table(0, 1) == nullptr, "released seq mirror gone");
    verify_mirror(storage, sim, 0, 0);
    storage.release(0);
    check(storage.upload_seq_page_table(0, 0) == nullptr, "released seq0 mirror gone");

    storage.reset();
    check(storage.upload_seq_page_table(0, 0) == nullptr, "reset clears mirrors");

    if (g_failures == 0)
        printf("ALL PASS\n");
    else
        printf("%d FAILURE(S)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
