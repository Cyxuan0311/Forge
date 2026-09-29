/**
 * Arguments shared by forge-cli and forge-server.
 *
 * The two binaries used to have completely independent parsers with no shared
 * header at all, which let them drift: --kv-cache-dtype defaulted to f16 in one
 * and q8_0 in the other, forge-server's -ngl crashed on "auto", its
 * --kv-cache-dtype accepted a typo and silently became f16, and its thread
 * auto-tune still had the pre-fix gate that the CLI had already corrected. The
 * roughly 110 lines that turn these flags into ContextParams + engine setters
 * were duplicated too.
 *
 * So: one struct, one parse chain, one apply function, all in forge_infer, and
 * each binary parses its own extra flags on top. parse_common_arg() returns
 * NotMine for anything it does not own, so the caller's own chain is unchanged
 * in shape.
 */

#pragma once

#include <string>
#include <vector>

#include "forge/kv_cache.h"
#include "forge/kv_cache_dtype.h"
#include "forge/kv_working_set.h"

namespace forge {

class InferenceContext;
class Model;
class TransformerEngine;

// Which defaults a shared flag should start from. forge-server sizes its
// generation reserve for a request's default max_tokens; forge-cli sizes it for
// -n. Same struct, same field, two sensible defaults.
enum class CommonArgsProfile {
    Cli,
    Server,
};

struct CommonArgs {
    // ---- model / placement ----
    std::string model_path;
    int n_gpu_layers = -1;                  // -1 auto, -2 all
    std::vector<int> gpu_layers_per_dev;    // -ngl N:M[:K] multi-GPU split
    int threads = -1;                       // -1 = auto-tune
    int ctx_size = 0;                       // 0 = auto

    // ---- KV cache ----
    // Unset = -1 sentinel so an explicit 0 or an explicit value is
    // distinguishable from "flag absent"; otherwise --cache-type-k 0 would
    // silently mean "not given", which is the same trap --spec-dspark-conf 0
    // used to fall into.
    int kv_cache_type_k = -1;
    int kv_cache_type_v = -1;
    bool offload_kqv = true;  // KV follows the layers onto the GPU
    bool paged_kv = false;
    bool cuda_graph = false;

    // ---- load / memory ----
    std::string load_mode;          // "" default, else mmap | mlock | mmap_mlock
    bool offload_embedding = true;
    bool prefetch = true;

    // ---- KVMem tiered KV working set (arXiv:2609.04852) ----
    bool kvmem = false;
    // Hadamard-rotate activations before a quantized KV cache. Off by default: it
    // costs three transforms per layer and only pays off when K or V is quantized.
    bool rope_k_quant = false;
    int kvmem_budget = 0;
    int kvmem_sink = 256;
    int kvmem_gen_reserve = 0;
    std::string kvmem_retrieval = "recent";  // recent | embed

    // Generation budget used for the kvmem reserve fallback, set from -n by the
    // CLI and from a request's default max_tokens by the server.
    int generation_budget = 256;

    forge::KVemConfig kvem_config() const {
        forge::KVemConfig c;
        c.enabled = kvmem;
        c.budget_tokens = kvmem_budget;
        c.sink_tokens = kvmem_sink;
        c.gen_reserve = kvmem_gen_reserve > 0 ? kvmem_gen_reserve
                                             : (generation_budget > 0 ? generation_budget : 256);
        c.retrieval = kvmem_retrieval;
        return c;
    }

    // Resolved K/V pair. --cache-type-k/-v win; a bare --kv-cache-dtype sets
    // both; neither means f16.
    forge::KVCacheTypeConfig kv_cache_types() const {
        forge::KVCacheTypeConfig cfg;
        const auto pick = [](int v) {
            return v >= 0 ? static_cast<forge::KVCacheDType>(v) : forge::KVCacheDType::F16;
        };
        if (kv_cache_type_k >= 0 && kv_cache_type_v >= 0) {
            cfg.type_k = pick(kv_cache_type_k);
            cfg.type_v = pick(kv_cache_type_v);
        } else if (kv_cache_type_k >= 0) {
            cfg.type_k = cfg.type_v = pick(kv_cache_type_k);
        } else if (kv_cache_type_v >= 0) {
            cfg.type_k = cfg.type_v = pick(kv_cache_type_v);
        } else {
            cfg.type_k = cfg.type_v = forge::KVCacheDType::F16;
        }
        return cfg;
    }
};

enum class ParseResult {
    NotMine,  // the caller should keep looking
    Ok,
    Error,  // *err holds a message; the caller should report it and exit
};

// Parse one shared flag. `i` is the index of `arg` in argv and is advanced past
// the value when the flag takes one. Unknown args are NotMine, so callers do
// `if (auto r = parse_common_arg(...); r != ParseResult::NotMine) { ... }` and
// fall through to their own chain otherwise.
ParseResult parse_common_arg(const std::string& arg, int argc, char** argv, int& i,
                             CommonArgs& out, std::string* err);

// True if `arg` is a shared flag that takes a value, used to validate arity
// before calling parse_common_arg so the error text matches each binary's.
bool common_arg_takes_value(const std::string& arg);

// Flags the shared parser owns, for "did you mean" style diagnostics.
std::vector<std::string> common_arg_names();

// Cross-flag checks that can only run once parsing is done (kvmem budget vs
// sink, unknown kvmem retrieval mode, and so on). Returns false and fills err.
bool validate_common_args(const CommonArgs& a, std::string* err);

// Apply the shared flags to a loaded model / context / engine. `model` may be
// null when the caller loads the model itself (forge-server does). Returns false
// and fills err on a validation failure (e.g. a head dim the chosen K/V types
// cannot support) — the caller must fail loudly rather than silently downgrading,
// which is the failure mode this shared layer exists to remove.
bool apply_common_args(const CommonArgs& a, forge::Model* model, forge::InferenceContext& ctx,
                       forge::TransformerEngine* engine, std::string* err);

}  // namespace forge
