// KVBlockSwapper round-trip on *device* memory.
//
// Why this test exists
// --------------------
// KVBlockSwapper::offload()/bring_back() used std::memcpy unconditionally. The
// only callers feed it paged GPU pages, whose bytes are device pointers
// (PagedKVStorage::storage_raw_ptr returns s.d_data for quantized storage and
// tensor->data() for FP32). std::memcpy cannot read device memory: it faults, or
// on some mappings silently returns whatever happens to be there. The existing
// tests/test_kv_offload.cpp declares itself "Host-only test (no GPU required)",
// so this path had no coverage at all -- which is how the defect survived.
//
// This test pins the contract that matters: an offloaded page must come back
// *bit-identical*, on both the host and the device side. It also pins the
// host-side behaviour so the new CopyKind plumbing cannot regress it.
//
// The data is deliberately not all-zero: a memcpy bug that copies from the wrong
// address, or skips a tail, is invisible on zeros but obvious on a pattern.
#include <cstdio>
#include <cstring>
#include <vector>

#ifdef USE_CUDA
#    include <cuda_runtime.h>
#endif

#include "forge/kv_block_swapper.h"

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

// A pattern where every byte position matters and no two 16-byte chunks match.
std::vector<uint8_t> make_pattern(size_t n, uint32_t seed) {
    std::vector<uint8_t> v(n);
    uint32_t s = seed * 2654435761u + 1u;
    for (size_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        v[i] = static_cast<uint8_t>((s >> 16) ^ (i * 31u) ^ (i >> 3));
    }
    return v;
}

bool has_cuda() {
#ifdef USE_CUDA
    int dev = -1;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0)
        return false;
    cudaDeviceProp prop{};
    return cudaGetDeviceProperties(&prop, dev) == cudaSuccess;
#else
    return false;
#endif
}

// ---- Host side: unchanged contract, still must work ----
void test_host_roundtrip() {
    printf("[swapper] host round-trip is bit-exact\n");
    KVBlockSwapper sw;
    const auto k = make_pattern(4096, 1);
    const auto v = make_pattern(2048, 2);

    check(sw.offload(0, 7, k.data(), k.size(), v.data(), v.size(),
                     KVBlockSwapper::CopyKind::Host),
          "offload stores the page");
    check(sw.has(0, 7), "page is present after offload");
    check(sw.nbytes() == k.size() + v.size(), "host bytes accounted");

    std::vector<uint8_t> k2(k.size(), 0xAB), v2(v.size(), 0xAB);
    check(sw.bring_back(0, 7, k2.data(), k2.size(), v2.data(), v2.size(),
                        KVBlockSwapper::CopyKind::Host),
          "bring_back restores the page");
    check(k2 == k, "keys are bit-identical");
    check(v2 == v, "values are bit-identical");

    // A second offload of the same page must refresh, not fail.
    const auto k3 = make_pattern(4096, 99);
    check(sw.offload(0, 7, k3.data(), k3.size(), v.data(), v.size(),
                     KVBlockSwapper::CopyKind::Host),
          "re-offload of an existing page is accepted");
    std::vector<uint8_t> k4(k.size(), 0);
    sw.bring_back(0, 7, k4.data(), k4.size(), v2.data(), v2.size(),
                  KVBlockSwapper::CopyKind::Host);
    check(k4 == k3, "re-offload overwrote the stored copy");

    // Size mismatch must be rejected, not silently reinterpreted.
    check(!sw.offload(0, 7, k3.data(), 16, v.data(), v.size(),
                      KVBlockSwapper::CopyKind::Host),
          "size mismatch on an existing page is rejected");
    check(!sw.bring_back(0, 999, k4.data(), k4.size(), v2.data(), v2.size(),
                         KVBlockSwapper::CopyKind::Host),
          "bring_back of an unknown page fails");

    sw.drop(0, 7);
    check(!sw.has(0, 7), "drop releases the host copy");
    check(sw.nbytes() == 0, "host bytes released");
}

// ---- Device side: the regression this whole change is about ----
void test_device_roundtrip() {
    printf("[swapper] device round-trip is bit-exact (the regression guard)\n");
#ifdef USE_CUDA
    if (!has_cuda()) {
        printf("  skip: no CUDA device\n");
        return;
    }
    KVBlockSwapper sw;
    const auto k_h = make_pattern(64 * 1024, 3);
    const auto v_h = make_pattern(32 * 1024, 4);

    uint8_t* k_d = nullptr;
    uint8_t* v_d = nullptr;
    if (cudaMalloc(&k_d, k_h.size()) != cudaSuccess ||
        cudaMalloc(&v_d, v_h.size()) != cudaSuccess) {
        printf("  skip: cudaMalloc failed\n");
        return;
    }
    // Poison the buffers first so a no-op read cannot pass by luck.
    cudaMemset(k_d, 0xCD, k_h.size());
    cudaMemset(v_d, 0xCD, v_h.size());
    check(cudaMemcpy(k_d, k_h.data(), k_h.size(), cudaMemcpyHostToDevice) == cudaSuccess,
          "seed device keys");
    check(cudaMemcpy(v_d, v_h.data(), v_h.size(), cudaMemcpyHostToDevice) == cudaSuccess,
          "seed device values");

    // This is the call that used to std::memcpy from a device pointer.
    check(sw.offload(0, 3, k_d, k_h.size(), v_d, v_h.size(),
                     KVBlockSwapper::CopyKind::Device),
          "offload accepts device memory");
    check(sw.has(0, 3), "device page is present after offload");

    // Scribble over the device buffers: a correct restore must put the data back.
    cudaMemset(k_d, 0xEF, k_h.size());
    cudaMemset(v_d, 0xEF, v_h.size());

    // Fetch into host staging, then let the caller restore into device memory
    // exactly as PagedKVStorage::bring_back_unlocked does.
    std::vector<uint8_t> k_s(k_h.size(), 0), v_s(v_h.size(), 0);
    check(sw.bring_back(0, 3, k_s.data(), k_s.size(), v_s.data(), v_s.size(),
                        KVBlockSwapper::CopyKind::Host),
          "device page fetched to host staging");
    check(k_s == k_h, "host copy of device keys is bit-identical");
    check(v_s == v_h, "host copy of device values is bit-identical");

    // Host staging -> device again: the round trip must be lossless.
    cudaMemset(k_d, 0x11, k_h.size());
    cudaMemset(v_d, 0x11, v_h.size());
    check(cudaMemcpy(k_d, k_s.data(), k_s.size(), cudaMemcpyHostToDevice) == cudaSuccess,
          "restore keys to device");
    check(cudaMemcpy(v_d, v_s.data(), v_s.size(), cudaMemcpyHostToDevice) == cudaSuccess,
          "restore values to device");

    std::vector<uint8_t> back_k(k_h.size(), 0), back_v(v_h.size(), 0);
    cudaMemcpy(back_k.data(), k_d, k_h.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(back_v.data(), v_d, v_h.size(), cudaMemcpyDeviceToHost);
    check(back_k == k_h, "device keys survive the full round-trip");
    check(back_v == v_h, "device values survive the full round-trip");

    // bring_back straight into device memory (the device-kind path).
    cudaMemset(k_d, 0x22, k_h.size());
    check(sw.bring_back(0, 3, k_d, k_h.size(), v_d, v_h.size(),
                        KVBlockSwapper::CopyKind::Device),
          "bring_back accepts a device destination");
    cudaMemcpy(back_k.data(), k_d, k_h.size(), cudaMemcpyDeviceToHost);
    check(back_k == k_h, "device-kind bring_back wrote the right bytes");

    sw.reset();
    check(sw.nbytes() == 0, "reset releases host bytes");
    cudaFree(k_d);
    cudaFree(v_d);
#else
    printf("  skip: built without CUDA\n");
#endif
}

}  // namespace

int main() {
    test_host_roundtrip();
    test_device_roundtrip();

    if (g_failures == 0) {
        printf("all KVBlockSwapper round-trip tests passed\n");
        return 0;
    }
    printf("%d KVBlockSwapper test(s) FAILED\n", g_failures);
    return 1;
}