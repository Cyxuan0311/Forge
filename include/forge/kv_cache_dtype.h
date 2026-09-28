/**
 * Shared KV-cache dtype naming, parsing and validation.
 *
 * Both forge-cli and forge-server expose a KV cache dtype flag, and they are the
 * two places a typo can silently cost you memory or quality: an unrecognised
 * name used to fall through to f16 without a word (forge-server did exactly
 * that, and --kv-cache-dtype Q8_0 in the wrong case was accepted). So parsing
 * lives here, in forge_infer, and both binaries use it.
 *
 * The allowed set is what Forge actually implements, not what a sibling project
 * accepts. llama.cpp's -ctk lists nine values; three of them (q4_1, q5_0/q5_1,
 * iq4_nl) have no dequantiser here and bf16 has no KVCacheDType at all, so
 * passing them is a hard error naming the value rather than a silent downgrade.
 * Use kv_cache_dtype_allowed_values() for the help text so the two can never
 * drift apart again.
 */

#pragma once

#include <string>

#include "forge/kv_cache.h"

namespace forge {

// Elements per quantisation block. 1 = no block structure (FP32/F16/FP8).
// A dtype is only usable for a given head_dim when the head dim is an exact
// multiple of this: KVCache::block_nbytes PADS a short row up to a whole block
// (see kv_cache.cpp), so a non-divisible head dim produces a row that is longer
// than the fused attention kernels assume and reads misaligned.
int kv_cache_dtype_block_size(KVCacheDType dt);

// The 2D form: block size of the whole KV row (num_kv_heads * head_dim).
size_t kv_cache_dtype_block_nbytes(KVCacheDType dt, int64_t n);

// "f16", "q8_0", ... — the canonical spelling accepted by the parser.
const char* kv_cache_dtype_name(KVCacheDType dt);

// Comma-separated list of every supported dtype, for help text and errors.
std::string kv_cache_dtype_allowed_values();

// Strict parse. Returns false and fills `err` (mentioning the offending value
// and the allowed set) on anything unrecognised. Never falls back to a default.
bool parse_kv_cache_dtype(const std::string& name, KVCacheDType* out, std::string* err);

struct KVCacheTypeValidation {
    bool ok = true;
    std::string error;    // set when !ok
    std::string warning;  // set for non-fatal policy notes
};

// Validate a K/V type pair for a model. `head_dim_k` / `head_dim_v` are the
// per-head dimensions of the K and V caches; for MLA these are the same tensor
// and the pair must match. `arch` is used only to identify MLA-style models.
KVCacheTypeValidation validate_kv_cache_types(const KVCacheTypeConfig& cfg, int head_dim_k,
                                             int head_dim_v, const std::string& arch);

}  // namespace forge
