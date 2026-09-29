#include "forge/cli_common_args.h"

#include <algorithm>
#include <cstdlib>

#include "forge/engines/transformer_engine.h"
#include "forge/kv_working_set.h"
#include "forge/logger.h"
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
            "--load-mode",        "--offload-embedding",    "--no-prefetch"};
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

    bool* b = nullptr;
    if (arg == "-kvo" || arg == "--kv-offload" || arg == "-nkvo" || arg == "--no-kv-offload")
        b = &out.offload_kqv;
    else if (arg == "--paged-kv")
        b = &out.paged_kv;
    else if (arg == "--cuda-graph")
        b = &out.cuda_graph;
    else if (arg == "--kvmem")
        b = &out.kvmem;
    else if (arg == "-rkq" || arg == "--rope-k-quant" || arg == "-nrkq" ||
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
    params.kv_storage_mode = a.paged_kv ? KVStorageMode::Paged : KVStorageMode::Contiguous;

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
    } else if (a.n_gpu_layers != 0) {
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

}  // namespace forge
