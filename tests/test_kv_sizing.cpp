// Roadmap phase 1 / item 1.2: auto-capacity sizing for the paged KV cache.
//
// Host-only test (no GPU required). It pins down the arithmetic that replaces
// the hard-coded `max_num_seqs = 32`, including the acceptance criterion from
// the roadmap:
//     max_num_seqs * per_seq_kv_bytes <= budget * safety
#include <cmath>
#include <cstdio>
#include <vector>

#include "forge/kv_sizing.h"

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

// Reference params: 32 layers, 8 KV heads * 128 head_dim, 4k context.
KVSizingParams make_params(KVCacheDType dt) {
    KVSizingParams p;
    p.num_layers = 32;
    p.kv_dims.assign(32, 8 * 128);  // kv_dim = 1024
    p.max_seq_len = 4096;
    p.page_size = 16;
    p.device = DeviceType::CPU;  // keep query_free_device_bytes() out of the path
    p.type_k = dt;
    p.type_v = dt;
    return p;
}

void test_token_bytes() {
    printf("[1] kv_token_bytes\n");

    // FP32: K and V each hold kv_dim floats.
    const size_t fp32_bytes = kv_token_bytes(KVCacheDType::FP32, KVCacheDType::FP32, 1024);
    check(fp32_bytes == 2u * 1024u * sizeof(float), "fp32 token = 2 * kv_dim * 4 bytes");

    // Quantized types match KVCache::block_nbytes per side.
    const size_t q8_bytes = kv_token_bytes(KVCacheDType::Q8_0, KVCacheDType::Q8_0, 1024);
    check(q8_bytes == 2u * KVCache::block_nbytes(KVCacheDType::Q8_0, 1024),
          "q8_0 token = 2 * block_nbytes()");
    check(q8_bytes < fp32_bytes, "q8_0 token is smaller than fp32");

    check(kv_token_bytes(KVCacheDType::FP32, KVCacheDType::FP32, 0) == 0,
          "kv_dim 0 yields 0 bytes");
}

void test_per_seq_bytes() {
    printf("[2] per_seq_kv_bytes\n");

    KVSizingParams p = make_params(KVCacheDType::FP32);
    const size_t expect = static_cast<size_t>(p.num_layers) * p.max_seq_len *
                          kv_token_bytes(p.type_k, p.type_v, p.kv_dims[0]);
    check(per_seq_kv_bytes(p) == expect, "fp32 per-seq = layers * max_seq_len * token bytes");

    // Quantized KV must shrink the per-sequence footprint (4x for q8_0).
    KVSizingParams q = make_params(KVCacheDType::Q8_0);
    check(per_seq_kv_bytes(q) < per_seq_kv_bytes(p), "q8_0 per-seq smaller than fp32");

    check(per_seq_kv_bytes(KVSizingParams{}) == 0, "empty params yield 0 bytes");
}

void test_swa_effective_len() {
    printf("[3] SWA effective length\n");

    KVSizingParams p = make_params(KVCacheDType::FP32);
    p.swa_window = 512;
    p.policies.assign(p.num_layers, KVLayerPolicy::SlidingWindow);

    check(kv_layer_effective_len(p, 0) == 512, "SWA layer bounded by window");
    check(per_seq_kv_bytes(p) == static_cast<size_t>(p.num_layers) * 512 *
                                     kv_token_bytes(p.type_k, p.type_v, p.kv_dims[0]),
          "SWA per-seq uses the window, not max_seq_len");

    // A full layer in the same cache still grows to max_seq_len.
    p.policies[0] = KVLayerPolicy::Full;
    check(kv_layer_effective_len(p, 0) == p.max_seq_len, "Full layer keeps max_seq_len");

    // Window larger than the context is clamped to the context.
    p.policies.assign(p.num_layers, KVLayerPolicy::SlidingWindow);
    p.swa_window = 100000;
    check(kv_layer_effective_len(p, 0) == p.max_seq_len, "oversized window clamped");
}

void test_auto_size_budget() {
    printf("[4] auto_size_kv budget contract\n");

    KVSizingParams p = make_params(KVCacheDType::FP32);
    const size_t budget = 8ull * 1024 * 1024 * 1024;  // 8 GiB
    const double safety = 0.9;

    KVSizingResult r = auto_size_kv(p, budget, safety);
    check(r.auto_sized, "auto sizing succeeds with a positive budget");
    check(r.max_num_seqs >= 1, "at least one sequence is admitted");

    const size_t claimed = static_cast<size_t>(r.max_num_seqs) * r.per_seq_bytes;
    const size_t allowed = static_cast<size_t>(static_cast<double>(budget) * safety);
    check(claimed <= allowed, "max_num_seqs * per_seq_bytes <= budget * safety");
    printf("       budget=%.2f GiB safety=%.2f -> max_seqs=%d (per_seq=%.1f MiB)\n",
           static_cast<double>(budget) / (1024.0 * 1024 * 1024), safety, r.max_num_seqs,
           static_cast<double>(r.per_seq_bytes) / (1024.0 * 1024));

    // One more sequence would blow the budget (proves it is maximal).
    check(claimed + r.per_seq_bytes > allowed, "result is maximal for the budget");

    // Monotonicity: a larger budget never admits fewer sequences.
    KVSizingResult r2 = auto_size_kv(p, budget * 2, safety);
    check(r2.max_num_seqs >= r.max_num_seqs, "bigger budget admits >= sequences");

    // Quantized KV admits more sequences than FP32 for the same budget.
    KVSizingResult r3 = auto_size_kv(make_params(KVCacheDType::Q8_0), budget, safety);
    check(r3.max_num_seqs > r.max_num_seqs, "q8_0 admits more sequences than fp32");

    // Pages must cover every admitted sequence at full length.
    check(static_cast<size_t>(r.max_pages_per_layer) * p.page_size >=
              static_cast<size_t>(r.max_num_seqs) * p.max_seq_len,
          "page pool covers all admitted sequences");
}

void test_auto_size_edges() {
    printf("[5] auto_size_kv edge cases\n");

    KVSizingParams p = make_params(KVCacheDType::FP32);

    // No budget (CPU device / unknown memory) -> caller keeps its default.
    KVSizingResult r = auto_size_kv(p, 0, 0.9);
    check(!r.auto_sized && r.max_num_seqs == 0, "zero budget is not auto-sized");

    // Tiny budget still admits one sequence.
    r = auto_size_kv(p, 1024, 0.9);
    check(r.auto_sized && r.max_num_seqs == 1, "tiny budget floors at 1 sequence");
    check(r.max_pages_per_layer >= 1, "at least one page per layer");

    // Degenerate params are rejected rather than dividing by zero.
    check(!auto_size_kv(KVSizingParams{}, 1 << 20, 0.9).auto_sized,
          "empty params are not auto-sized");

    // Safety is clamped into (0, 1].
    KVSizingResult r_odd = auto_size_kv(p, 8ull << 30, 5.0);
    check(r_odd.auto_sized, "safety > 1 is clamped, not rejected");
}

void test_device_query_is_safe() {
    printf("[6] query_free_device_bytes on a CPU device\n");
    // Must be 0 (meaning "cannot auto-size") rather than crashing or returning
    // host RAM, so the caller falls back to its default.
    check(query_free_device_bytes(DeviceType::CPU) == 0, "CPU device reports 0 free bytes");
}

}  // namespace

int main() {
    test_token_bytes();
    test_per_seq_bytes();
    test_swa_effective_len();
    test_auto_size_budget();
    test_auto_size_edges();
    test_device_query_is_safe();

    printf("kv_sizing: failures=%d\n", g_failures);
    printf(g_failures == 0 ? "RESULT: PASS\n" : "RESULT: FAIL\n");
    return g_failures == 0 ? 0 : 1;
}
