// Roadmap phase 2 / item 2.5: prefill whole-segment quantized write.
//
// The CUDA path (KVCache::update_quantized_cuda) already quantizes a whole
// prefill segment in a single launch_quantize_*_matrix call (one kernel launch
// over `num_rows` rows) instead of one launch per token. This test verifies the
// acceptance criterion: the whole-segment write produces bit-identical K/V to
// writing the same tokens one token at a time (per-token, seq_len==1).
//
// Per-row quantization is independent of launch granularity, so the two must
// match exactly. We exercise the real device path (DeviceType::CUDA) and
// compare the dequantized results copied back to the host.
#include "forge/kv_cache.h"
#include "forge/tensor.h"

#ifdef USE_CUDA
#    include <cuda_runtime.h>
#endif

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

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

constexpr int kLayers = 1;
constexpr int kKvHeads = 1;
constexpr int kHeadDim = 8;
constexpr int kKvDim = kKvHeads * kHeadDim;
constexpr int kMaxSeqLen = 32;
constexpr int kSeg = 12;  // prefill segment length

const char* dtype_name(KVCacheDType t) {
    switch (t) {
    case KVCacheDType::Q8_0:
        return "Q8_0";
    case KVCacheDType::Q4_0:
        return "Q4_0";
    case KVCacheDType::Q4_K:
        return "Q4_K";
    case KVCacheDType::F16:
        return "F16";
    case KVCacheDType::FP8_E4M3:
        return "FP8_E4M3";
    case KVCacheDType::FP8_E5M2:
        return "FP8_E5M2";
    default:
        return "FP32";
    }
}

// Deterministic, magnitude-varied fill so per-row quant scales are exercised.
void fill_kv(std::vector<float>& K, std::vector<float>& V) {
    for (int r = 0; r < kMaxSeqLen; ++r) {
        for (int c = 0; c < kKvDim; ++c) {
            float k = std::sin(static_cast<float>(r) * 0.7f + static_cast<float>(c) * 0.13f);
            float v = std::cos(static_cast<float>(r) * 0.5f - static_cast<float>(c) * 0.21f);
            K[r * kKvDim + c] = k;
            V[r * kKvDim + c] = v;
        }
    }
}

TensorPtr to_cpu(const TensorPtr& t) {
    auto c = std::make_shared<Tensor>(DataType::FP32, t->shape(), DeviceType::CPU);
    if (t->device() == DeviceType::CUDA) {
#ifdef USE_CUDA
        cudaDeviceSynchronize();
#endif
    }
    c->copy_from(*t);
    return c;
}

bool tensors_match(const TensorPtr& a, const TensorPtr& b) {
    if (!a || !b)
        return a == b;
    if (a->shape() != b->shape())
        return false;
    return std::memcmp(a->data(), b->data(), a->nbytes()) == 0;
}

TensorPtr row_tensor(const std::vector<float>& data, int row) {
    auto t =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, kKvDim}, DeviceType::CPU);
    std::memcpy(t->data(), data.data() + static_cast<size_t>(row) * kKvDim, kKvDim * sizeof(float));
    return t;
}

void test_segment_vs_per_token(KVCacheDType type) {
    printf("[seg] whole-segment vs per-token: %s\n", dtype_name(type));
    KVCacheTypeConfig cfg;
    cfg.type_k = type;
    cfg.type_v = type;

    KVCache whole;
    KVCache per_token;
    whole.init_quantized(kLayers, kKvHeads, kHeadDim, kMaxSeqLen, DeviceType::CUDA, cfg);
    per_token.init_quantized(kLayers, kKvHeads, kHeadDim, kMaxSeqLen, DeviceType::CUDA, cfg);

    std::vector<float> K(kMaxSeqLen * kKvDim), V(kMaxSeqLen * kKvDim);
    fill_kv(K, V);

    auto k_full = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{kSeg, kKvDim},
                                           DeviceType::CPU);
    auto v_full = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{kSeg, kKvDim},
                                           DeviceType::CPU);
    std::memcpy(k_full->data(), K.data(), kSeg * kKvDim * sizeof(float));
    std::memcpy(v_full->data(), V.data(), kSeg * kKvDim * sizeof(float));

    // Whole-segment write: one update with seq_len == kSeg.
    whole.update(0, 0, 0, k_full, v_full, kSeg);

    // Per-token write: kSeg updates with seq_len == 1.
    for (int t = 0; t < kSeg; ++t) {
        per_token.update(0, 0, t, row_tensor(K, t), row_tensor(V, t), 1);
    }

#ifdef USE_CUDA
    cudaDeviceSynchronize();
#endif

    check(whole.filled(0) == kSeg, "whole-segment filled == kSeg");
    check(per_token.filled(0) == kSeg, "per-token filled == kSeg");

    // get_key_filled() returns the (already populated) dequantized FP32 tensor;
    // the engine triggers dequantize_layer() before attention, so do it here.
    whole.dequantize_layer(0);
    per_token.dequantize_layer(0);

    auto wk = to_cpu(whole.get_key_filled(0));
    auto wv = to_cpu(whole.get_value_filled(0));
    auto pk = to_cpu(per_token.get_key_filled(0));
    auto pv = to_cpu(per_token.get_value_filled(0));

    check(tensors_match(wk, pk), "key whole-segment == per-token (bit identical)");
    check(tensors_match(wv, pv), "value whole-segment == per-token (bit identical)");
}

}  // namespace

int main() {
#ifndef USE_CUDA
    printf("test_kv_quant_segment: SKIP (built without USE_CUDA)\n");
    return 0;
#else
    if (kKvDim <= 0)
        return 1;
    test_segment_vs_per_token(KVCacheDType::Q8_0);
    test_segment_vs_per_token(KVCacheDType::Q4_0);
    test_segment_vs_per_token(KVCacheDType::F16);
    test_segment_vs_per_token(KVCacheDType::FP8_E4M3);
    test_segment_vs_per_token(KVCacheDType::FP8_E5M2);

    if (g_failures == 0) {
        printf("test_kv_quant_segment: PASS\n");
        return 0;
    }
    printf("test_kv_quant_segment: FAIL (%d assertion(s))\n", g_failures);
    return 1;
#endif
}
