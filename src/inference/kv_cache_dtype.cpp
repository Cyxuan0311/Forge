#include "forge/kv_cache_dtype.h"

#include <array>
#include <cstdio>
#include <utility>

namespace forge {
namespace {

struct DTypeEntry {
    KVCacheDType dt;
    const char* name;
    int block;
};

// Order matters: it is the order of kv_cache_dtype_allowed_values(), which
// doubles as the help text, so it reads from widest to narrowest.
constexpr std::array<DTypeEntry, 7> kEntries = {{
    {KVCacheDType::FP32, "fp32", 1},
    {KVCacheDType::F16, "f16", 1},
    {KVCacheDType::Q8_0, "q8_0", 32},
    {KVCacheDType::Q4_0, "q4_0", 32},
    {KVCacheDType::Q4_K, "q4_k", 256},
    {KVCacheDType::FP8_E4M3, "fp8_e4m3", 1},
    {KVCacheDType::FP8_E5M2, "fp8_e5m2", 1},
}};

const DTypeEntry* find(KVCacheDType dt) {
    for (const auto& e : kEntries) {
        if (e.dt == dt) return &e;
    }
    return nullptr;
}

bool is_quantized(KVCacheDType dt) {
    return dt != KVCacheDType::FP32 && dt != KVCacheDType::F16;
}

// Architectures where V is a view of K rather than its own tensor. In the KV
// cache implementations Forge already carries the MLA/DeepSeek-style latent
// attention paths (mla_executor.cpp, deepseek_layer_executor.cpp) plus Gemma4;
// for those, storing K and V at different widths is meaningless, and llama.cpp
// rejects the combination outright (llama-context.cpp:3754-3757).
bool v_shares_k_tensor(const std::string& arch) {
    return arch == "minicpm" || arch == "deepseek2" || arch == "deepseek3" ||
           arch == "deepseek4" || arch == "kimi" || arch == "gemma4" || arch == "gemma3";
}

}  // namespace

int kv_cache_dtype_block_size(KVCacheDType dt) { return find(dt) ? find(dt)->block : 1; }

size_t kv_cache_dtype_block_nbytes(KVCacheDType dt, int64_t n) {
    const int b = kv_cache_dtype_block_size(dt);
    if (b <= 1) return static_cast<size_t>(n) * (dt == KVCacheDType::FP32 ? 4 : 2);
    const int64_t blocks = (n + b - 1) / b;
    // Per-block payload: q8_0 = 4-byte scale + 32 bytes; q4_0 = 4 + 16;
    // q4_k = 144 for 256 elements. Matches KVCache::block_nbytes.
    size_t per_block = 0;
    switch (dt) {
    case KVCacheDType::Q8_0: per_block = 34; break;
    case KVCacheDType::Q4_0: per_block = 18; break;
    case KVCacheDType::Q4_K: per_block = 144; break;
    default: per_block = 0; break;
    }
    return static_cast<size_t>(blocks) * per_block;
}

const char* kv_cache_dtype_name(KVCacheDType dt) {
    const DTypeEntry* e = find(dt);
    return e ? e->name : "unknown";
}

std::string kv_cache_dtype_allowed_values() {
    std::string s;
    for (size_t i = 0; i < kEntries.size(); ++i) {
        if (i) s += ", ";
        s += kEntries[i].name;
    }
    return s;
}

bool parse_kv_cache_dtype(const std::string& name, KVCacheDType* out, std::string* err) {
    for (const auto& e : kEntries) {
        if (name == e.name) {
            if (out) *out = e.dt;
            return true;
        }
    }
    if (err) {
        *err = "unsupported KV cache type '" + name + "'; supported: " +
               kv_cache_dtype_allowed_values();
    }
    return false;
}

KVCacheTypeValidation validate_kv_cache_types(const KVCacheTypeConfig& cfg, int head_dim_k,
                                             int head_dim_v, const std::string& arch) {
    KVCacheTypeValidation r;

    if (cfg.type_k != cfg.type_v && v_shares_k_tensor(arch)) {
        r.ok = false;
        r.error = std::string(arch) +
                 " stores V inside the K tensor, so K and V cache types must match (got " +
                 kv_cache_dtype_name(cfg.type_k) + " / " + kv_cache_dtype_name(cfg.type_v) + ")";
        return r;
    }

    // Block-size divisibility. block_nbytes pads a short row up to a whole block,
    // so a head dim that is not a multiple of the block size yields a row longer
    // than the fused kernels assume. Checked on the head dim because the kernels
    // index per head (cuda_fused_attn.cu: head_byte_offset = kv_h * blocks_per_head).
    const auto check = [&](KVCacheDType dt, int head_dim, const char* which) {
        const int b = kv_cache_dtype_block_size(dt);
        if (b <= 1) return;
        if (head_dim > 0 && head_dim % b != 0) {
            r.ok = false;
            r.error = std::string("KV cache type ") + kv_cache_dtype_name(dt) + " for " + which +
                      " needs a head dim that is a multiple of " + std::to_string(b) +
                      " (this model has " + std::to_string(head_dim) + ")";
        }
    };
    check(cfg.type_k, head_dim_k, "K");
    if (r.ok) check(cfg.type_v, head_dim_v, "V");

    if (!r.ok) return r;

    // Policy note, not an error. At equal bit width, quantising K costs more
    // quality than quantising V: K feeds softmax(Q.K/sqrt(d)) where the error is
    // amplified exponentially and cannot be absorbed, while V enters as a convex
    // combination sum_i p_i V_i, a linear error-smoothing path. Measured KLD vs
    // f16/f16 on Qwen3.5-9B: q4_0/f16 = 0.005426 but f16/q4_0 = 0.003747.
    if (is_quantized(cfg.type_k) && !is_quantized(cfg.type_v)) {
        r.warning = std::string("K is quantised (") + kv_cache_dtype_name(cfg.type_k) +
                    ") while V is not; K is the more sensitive of the two. "
                    "Consider a higher-precision K instead.";
    }
    return r;
}

}  // namespace forge
