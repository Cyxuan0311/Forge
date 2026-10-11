#include "forge/cli_common_args.h"

#include <algorithm>
#include <cstdlib>

#include "forge/engines/transformer_engine.h"
#include "forge/kv_l3_backend.h"
#include "forge/kv_memory.h"
#include "forge/kv_policy.h"
#include "forge/kv_storage.h"
#include "forge/kv_working_set.h"
#include "forge/logger.h"
#include "forge/prefix_cache.h"
#include "forge/threads.h"
#include "forge/model.h"

namespace forge {
namespace {

// -ngl accepts auto / all / a single N / N:M[:K] for a multi-GPU split. Shared
// because forge-server used to call std::stoi on this, so "-ngl auto" threw
// std::invalid_argument out of main() and terminated.
ParseResult parse_gpu_layers(const std::string& value, int& n_gpu_layers,
                             std::vector<int>& per_dev, std::string* err) {
    if (value == "auto" || value == "-1") {
        n_gpu_layers = -1;
        per_dev.clear();
        return ParseResult::Ok;
    }
    if (value == "all" || value == "-2") {
        n_gpu_layers = -2;
        per_dev.clear();
        return ParseResult::Ok;
    }
    if (value.find(':') != std::string::npos) {
        per_dev.clear();
        size_t pos = 0;
        while (pos <= value.size()) {
            const size_t next = value.find(':', pos);
            const std::string part = value.substr(pos, next == std::string::npos
                                                         ? std::string::npos
                                                         : next - pos);
            if (part.empty()) {
                *err = "Error: --n-gpu-layers '" + value + "' has an empty slot";
                return ParseResult::Error;
            }
            try {
                const int v = std::stoi(part);
                if (v < 0) {
                    *err = "Error: --n-gpu-layers '" + value + "' must be non-negative per device";
                    return ParseResult::Error;
                }
                per_dev.push_back(v);
            } catch (const std::exception&) {
                *err = "Error: --n-gpu-layers '" + value + "' is not a number list";
                return ParseResult::Error;
            }
            if (next == std::string::npos) break;
            pos = next + 1;
        }
        n_gpu_layers = -1;
        return ParseResult::Ok;
    }
    try {
        n_gpu_layers = std::stoi(value);
    } catch (const std::exception&) {
        *err = "Error: --n-gpu-layers '" + value +
               "' is not a number (use auto, all, N, or N:M[:K])";
        return ParseResult::Error;
    }
    per_dev.clear();
    return ParseResult::Ok;
}

ParseResult parse_bool_flag(const std::string& arg, bool& out) {
    if (arg == "-kvo" || arg == "--kv-offload") {
        out = true;
    } else if (arg == "-nkvo" || arg == "--no-kv-offload") {
        out = false;
    } else if (arg == "--paged-kv") {
        out = true;
    } else if (arg == "--cuda-graph") {
        out = true;
    } else if (arg == "--kvmem") {
        out = true;
    } else if (arg == "-rkq" || arg == "--rope-k-quant") {
        out = true;
    } else if (arg == "-nrkq" || arg == "--no-rope-k-quant") {
        out = false;
    } else if (arg == "--offload-embedding") {
        out = true;
    } else if (arg == "--no-offload-embedding") {
        out = false;
    } else if (arg == "--prefetch") {
        out = true;
    } else if (arg == "--no-prefetch") {
        out = false;
    } else {
        return ParseResult::NotMine;
    }
    return ParseResult::Ok;
}

}  // namespace

bool common_arg_takes_value(const std::string& arg) {
    static const char* kValued[] = {
        "-m",       "--model",          "-ngl",           "--n-gpu-layers",
        "-c",       "--ctx-size",       "-t",             "--threads",
        "--kv-cache-dtype",            "-ctk",           "--cache-type-k",
        "-ctv",     "--cache-type-v",   "--kvmem-budget", "--kvmem-sink",
        "--kvmem-gen-reserve",         "--kvmem-retrieval", "--load-mode",
        "--kv-write-policy",           "--kv-host-pool-bytes",
        "--kv-async-backup",           "--kv-prefetch",
        "--kv-l3-backend",             "--kv-l3-dir",
        "--kv-l3-bytes",               "--kv-l3-segment-bytes",
    };
    for (const char* v : kValued) {
        if (arg == v) return true;
    }
    return false;
}

std::vector<std::string> common_arg_names() {
    return {"-m/--model",         "-ngl/--n-gpu-layers",   "-c/--ctx-size",
            "-t/--threads",       "--kv-cache-dtype",       "-ctk/--cache-type-k",
            "-ctv/--cache-type-v", "--kvo/--no-kv-offload",  "--paged-kv",
            "--cuda-graph",       "--kvmem",                "--kvmem-budget",
            "-rkq/--rope-k-quant",  "-nrkq/--no-rope-k-quant",
            "--kvmem-sink",       "--kvmem-gen-reserve",    "--kvmem-retrieval",
            "--load-mode",        "--offload-embedding",    "--no-prefetch",
            "--kv-write-policy",  "--kv-host-pool-bytes",   "--kv-async-backup",
            "--kv-prefetch",      "--kv-l3-backend",        "--kv-l3-dir",
            "--kv-l3-bytes",      "--kv-l3-segment-bytes"};
}

ParseResult parse_common_arg(const std::string& arg, int argc, char** argv, int& i,
                             CommonArgs& out, std::string* err) {
    const auto need_value = [&]() -> const char* {
        if (i + 1 >= argc) {
            if (err) *err = "Error: " + arg + " requires an argument";
            return nullptr;
        }
        return argv[++i];
    };

    if (arg == "-m" || arg == "--model") {
        const char* v = need_value();
        return v ? (out.model_path = v, ParseResult::Ok) : ParseResult::Error;
    }
    if (arg == "-ngl" || arg == "--n-gpu-layers") {
        const char* v = need_value();
        return v ? parse_gpu_layers(v, out.n_gpu_layers, out.gpu_layers_per_dev, err)
                 : ParseResult::Error;
    }
    if (arg == "-c" || arg == "--ctx-size" || arg == "--n-ctx") {
        const char* v = need_value();
        if (!v) return ParseResult::Error;
        try {
            out.ctx_size = std::stoi(v);
        } catch (const std::exception&) {
            *err = std::string("Error: --ctx-size '") + v + "' is not a number";
            return ParseResult::Error;
        }
        if (out.ctx_size < 0) {
            *err = "Error: --ctx-size must be >= 0 (0 = use the model's native length)";
            return ParseResult::Error;
        }
        return ParseResult::Ok;
    }
    if (arg == "-t" || arg == "--threads") {
        const char* v = need_value();
        if (!v) return ParseResult::Error;
        try {
            out.threads = std::stoi(v);
        } catch (const std::exception&) {
            *err = std::string("Error: --threads '") + v + "' is not a number";
            return ParseResult::Error;
        }
        return ParseResult::Ok;
    }
    if (arg == "--kv-cache-dtype" || arg == "-ctk" || arg == "--cache-type-k" ||
        arg == "-ctv" || arg == "--cache-type-v") {
        const char* v = need_value();
        if (!v) return ParseResult::Error;
        KVCacheDType dt;
        std::string e;
        if (!parse_kv_cache_dtype(v, &dt, &e)) {
            *err = "Error: " + e;
            return ParseResult::Error;
        }
        // --kv-cache-dtype sets both (the historical single-dtype behaviour);
        // -ctk/-ctv set one each and take precedence when both are given.
        if (arg == "--kv-cache-dtype") {
            out.kv_cache_type_k = static_cast<int>(dt);
            out.kv_cache_type_v = static_cast<int>(dt);
        } else if (arg == "-ctk" || arg == "--cache-type-k") {
            out.kv_cache_type_k = static_cast<int>(dt);
        } else {
            out.kv_cache_type_v = static_cast<int>(dt);
        }
        return ParseResult::Ok;
    }
    if (arg == "--kvmem-budget" || arg == "--kvmem-sink" || arg == "--kvmem-gen-reserve") {
        const char* v = need_value();
        if (!v) return ParseResult::Error;
        int n = 0;
        try {
            n = std::stoi(v);
        } catch (const std::exception&) {
            *err = std::string("Error: ") + arg + " '" + v + "' is not a number";
            return ParseResult::Error;
        }
        if (n < 0) {
            *err = std::string("Error: ") + arg + " must be >= 0";
            return ParseResult::Error;
        }
        if (arg == "--kvmem-budget") {
            out.kvmem_budget = n;
        } else if (arg == "--kvmem-sink") {
            out.kvmem_sink = n;
        } else {
            out.kvmem_gen_reserve = n;
        }
        return ParseResult::Ok;
    }
    if (arg == "--kvmem-retrieval") {
        const char* v = need_value();
        if (!v) return ParseResult::Error;
        if (std::string(v) != "recent" && std::string(v) != "embed") {
            *err = std::string("Error: --kvmem-retrieval must be 'recent' or 'embed', got '") + v +
                   "'";
            return ParseResult::Error;
        }
        out.kvmem_retrieval = v;
        return ParseResult::Ok;
    }
    if (arg == "--load-mode") {
        const char* v = need_value();
        if (!v) return ParseResult::Error;
        const std::string m = v;
        if (m != "mmap" && m != "mlock" && m != "mmap_mlock") {
            *err = "Error: --load-mode must be 'mmap', 'mlock' or 'mmap_mlock', got '" + m + "'";
            return ParseResult::Error;
        }
        out.load_mode = m;
        return ParseResult::Ok;
    }

    // ---- multi-level prefix residency (paged KV only; flags over env) ----
    if (arg == "--kv-write-policy") {
        const char* v = need_value();
        if (!v) return ParseResult::Error;
        // Fail loudly on a typo instead of silently downgrading to the default,
        // matching the strict-parse intent of kv_write_policy_from_string.
        if (!kv_write_policy_from_string(v, nullptr)) {
            *err = std::string("Error: --kv-write-policy must be 'write_back', "
                               "'write_through' or 'off', got '") +
                   v + "'";
            return ParseResult::Error;
        }
        out.kv_write_policy = v;
        return ParseResult::Ok;
    }
    if (arg == "--kv-host-pool-bytes" || arg == "--kv-l3-bytes" ||
        arg == "--kv-l3-segment-bytes") {
        const char* v = need_value();
        if (!v) return ParseResult::Error;
        long long n = 0;
        try {
            n = std::stoll(v);
        } catch (const std::exception&) {
            *err = std::string("Error: ") + arg + " '" + v + "' is not a number";
            return ParseResult::Error;
        }
        if (n < 0) {
            *err = std::string("Error: ") + arg + " must be >= 0";
            return ParseResult::Error;
        }
        if (arg == "--kv-host-pool-bytes")
            out.kv_host_pool_bytes = n;
        else if (arg == "--kv-l3-bytes")
            out.kv_l3_bytes = n;
        else
            out.kv_l3_segment_bytes = n;
        return ParseResult::Ok;
    }
    if (arg == "--kv-async-backup" || arg == "--kv-prefetch") {
        const char* v = need_value();
        if (!v) return ParseResult::Error;
        const std::string s = v;
        int n;
        if (s == "1" || s == "on" || s == "true" || s == "yes")
            n = 1;
        else if (s == "0" || s == "off" || s == "false" || s == "no")
            n = 0;
        else {
            *err = "Error: " + arg + " must be 0 or 1, got '" + s + "'";
            return ParseResult::Error;
        }
        if (arg == "--kv-async-backup")
            out.kv_async_backup = n;
        else
            out.kv_prefetch = n;
        return ParseResult::Ok;
    }
    if (arg == "--kv-l3-backend") {
        const char* v = need_value();
        if (!v) return ParseResult::Error;
        const std::string s = v;
        if (s != "file" && s != "off" && s != "none" && s != "0") {
            *err = "Error: --kv-l3-backend must be 'file' or 'off', got '" + s + "'";
            return ParseResult::Error;
        }
        out.kv_l3_backend = s;
        return ParseResult::Ok;
    }
    if (arg == "--kv-l3-dir") {
        const char* v = need_value();
        return v ? (out.kv_l3_dir = v, ParseResult::Ok) : ParseResult::Error;
    }

    bool* b = nullptr;
    if (arg == "-kvo" || arg == "--kv-offload" || arg == "-nkvo" || arg == "--no-kv-offload")
        b = &out.offload_kqv;
    else if (arg == "--paged-kv")
        b = &out.paged_kv;
    else if (arg == "--cuda-graph")
        b = &out.cuda_graph;
    else if (arg == "--kvmem") {
        // --kvmem implies --paged-kv: the tiered working set caps the page pool
        // and lets the logical context exceed it, so it is meaningless (and was
        // silently reduced to a no-op) on contiguous storage. This has to happen
        // at parse time, not in apply_common_args, because the storage object is
        // already constructed by the time args are applied -- KVMemory::is_paged()
        // reads the live storage, not params. forge-cli set this in its own
        // parser; forge-server only comes through here, hence the fix.
        out.paged_kv = true;
        return parse_bool_flag(arg, out.kvmem) == ParseResult::Ok ? ParseResult::Ok
                                                                 : ParseResult::Error;
    } else if (arg == "-rkq" || arg == "--rope-k-quant" || arg == "-nrkq" ||
               arg == "--no-rope-k-quant")
        b = &out.rope_k_quant;
    else if (arg == "--offload-embedding" || arg == "--no-offload-embedding")
        b = &out.offload_embedding;
    else if (arg == "--prefetch" || arg == "--no-prefetch")
        b = &out.prefetch;
    if (b) return parse_bool_flag(arg, *b);
    return ParseResult::NotMine;
}

bool validate_common_args(const CommonArgs& a, std::string* err) {
    if (a.kvmem) {
        // --kvmem implies paged storage: the pool is capped to the working set
        // and the logical context may exceed it.
        if (a.kvmem_sink < 0 || a.kvmem_budget < 0 || a.kvmem_gen_reserve < 0) {
            *err = "Error: --kvmem-budget/--kvmem-sink/--kvmem-gen-reserve must be >= 0";
            return false;
        }
        if (a.kvmem_budget > 0 && a.kvmem_sink > a.kvmem_budget) {
            *err = "Error: --kvmem-sink must not exceed --kvmem-budget";
            return false;
        }
    }
    return true;
}

bool apply_common_args(const CommonArgs& a, Model* model, InferenceContext& ctx,
                       TransformerEngine* engine, std::string* err) {
    if (!forge::validate_common_args(a, err)) return false;

    if (model) {
        if (!a.load_mode.empty()) model->set_load_mode(a.load_mode);
        model->set_offload_embedding(a.offload_embedding);
        model->set_prefetch(a.prefetch);
    }

    auto& params = ctx.params_mut();

    params.offload_kqv = a.offload_kqv;
    params.offload_embedding = a.offload_embedding;
    // --kvmem implies --paged-kv, exactly as the help text states: the tiered
    // working set caps the page pool and lets the logical context exceed it.
    // forge-cli set this in its own parser, but forge-server only comes through
    // this shared parser -- without it --kvmem silently degraded to a no-op with
    // "tiered KV disabled". Decided here, at the single consumption point, so
    // neither entry point can be missed.
    const bool paged = a.paged_kv || a.kvmem;
    params.kv_storage_mode = paged ? KVStorageMode::Paged : KVStorageMode::Contiguous;

    if (!engine) return true;

    // ---- KV cache types ----
    const KVCacheTypeConfig kv = a.kv_cache_types();
    const ModelConfig& mcfg = ctx.model().config();
    // Per-layer head dims: a model with interleaved SWA can carry a different
    // head dim for its sliding-window layers, and the block-size check has to
    // hold for both, exactly as llama.cpp iterates n_embd_head_k(il).
    const int hd_k = mcfg.head_dim_swa > 0 ? mcfg.head_dim_swa : mcfg.head_dim;
    const int hd_v = mcfg.head_dim;
    const KVCacheTypeValidation v = validate_kv_cache_types(kv, hd_k, hd_v, mcfg.arch_type);
    if (!v.ok) {
        *err = "Error: " + v.error;
        return false;
    }
    if (!v.warning.empty()) LOG_WARN("KV cache: " + v.warning);
    engine->set_rope_k_quant(a.rope_k_quant);
    engine->set_kv_cache_type_config(kv);

    if (a.ctx_size > 0) engine->set_kv_cache_max_seq_len(a.ctx_size);

    if (!a.gpu_layers_per_dev.empty()) {
        engine->set_gpu_layers(a.n_gpu_layers, a.gpu_layers_per_dev);
    } else {
        // n_gpu_layers == 0 (CPU-only) must reach the engine too: skipping it
        // left gpu_layers_ at the -1 (auto) default, so the engine targeted CUDA
        // for execution while the weights were loaded on CPU.
        engine->set_gpu_layers(a.n_gpu_layers);
    }

    // ---- threads ----
    // The auto-tune gate has to consider hybrid offload: the old form keyed on
    // "no GPU at all", which left CPU-resident layers at the default 4 threads
    // whenever a partial -ngl run put some layers on the GPU, so the CPU became
    // the decode bottleneck while the GPU idled. forge-cli already fixed this;
    // forge-server still had the old predicate.
    if (a.threads > 0) {
        params.n_threads = a.threads;
        params.n_threads_batch = a.threads;
    } else if (a.n_gpu_layers < mcfg.num_layers) {
        // Hybrid offload: some layers still run on CPU, so tune CPU threads even
        // though the model is on the GPU. The old form of this gate keyed on
        // "no GPU at all", which left CPU-resident layers at the default
        // n_threads=4 and made the CPU the decode bottleneck while the GPU idled.
        const int n = detect_best_cpu_threads(mcfg.hidden_dim, mcfg.intermediate_dim);
        if (n > 0) {
            params.n_threads = n;
            params.n_threads_batch = n;
        }
    }

    engine->set_use_graph(a.cuda_graph);
    engine->set_cuda_graph_enabled(a.cuda_graph);
    return true;
}

void apply_prefix_cache_args(const CommonArgs& a, PrefixCache& cache, KVStorage& storage) {
    // Multi-level residency only exists on paged storage; on contiguous storage
    // (the default) every --kv-* flag here would be meaningless.
    if (!storage.is_paged()) return;
    // Flag over env over default, per knob.
    cache.set_policy(kv_write_policy_resolved(a.kv_write_policy));
    cache.set_async_backup(a.kv_async_backup >= 0 ? a.kv_async_backup != 0
                                                  : kv_async_backup_from_env());
    cache.set_prefetch(a.kv_prefetch >= 0 ? a.kv_prefetch != 0 : kv_prefetch_from_env());
    // FORGE_KV_RESTORE_WAIT_MS has no flag; the env var stays authoritative.
    cache.set_restore_wait_ms(kv_restore_wait_ms_from_env());
    storage.set_prefix_pool_max_bytes(a.kv_host_pool_bytes >= 0
                                          ? static_cast<size_t>(a.kv_host_pool_bytes)
                                          : kv_host_pool_bytes_from_env());
    storage.set_l3_backend(make_l3_backend_resolved(a.kv_l3_backend, a.kv_l3_dir, a.kv_l3_bytes,
                                                    a.kv_l3_segment_bytes));
}

}  // namespace forge
