#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "format.h"
#include "perf/gguf_scanner.h"

namespace forge::inspect {

enum class OutputMode { Summary, Tree, Chart, Metadata, OpPipeline, Peaks, Layers };

// Architecture info derived from a GGUF snapshot's metadata.
struct ArchInfo {
    std::string arch;       // e.g. "llama"
    std::string family;     // e.g. "gqa"
    std::string model_name;
    std::string file_type;  // e.g. "Q4_K_M"
    bool supported = false;
    int64_t hidden_size = 0;
    int64_t n_layers = 0;
    int64_t n_heads = 0;
    int64_t n_kv_heads = 0;

    // Added for the memory / KV estimates. Derived where the file does not state
    // them outright (head_dim is usually only implied), and left 0 when absent so
    // a caller can tell "not stated" from "computed as 0".
    int64_t head_dim = 0;          // attention.key_length, else hidden / n_heads
    int64_t context_length = 0;    // trained max, 0 if not stated
    int64_t vocab_size = 0;        // tokenizer.ggml.tokens.length
    int64_t n_expert = 0;          // MoE expert count, 0 = dense
    int64_t n_expert_used = 0;     // experts used per token
    // Hybrid architectures (lfm2, qwen35) store a PER-LAYER attention.head_count_kv
    // array in which 0 marks a layer with no attention at all. n_attn_layers is
    // how many entries are > 0; n_kv_heads is the max, which is what Forge's
    // config parser derives and what its uniform KV allocation actually uses.
    int64_t n_attn_layers = 0;
    bool per_layer_kv = false;     // true when the array was present
    double rope_theta = 0.0;       // rope.freq_base
    int64_t sliding_window = 0;    // attention.sliding_window, 0 = full attention
    // True when any metadata key the KV estimate depends on was missing, so the
    // estimate can be reported as incomplete instead of silently wrong.
    bool kv_estimable = false;
};

ArchInfo sniff_arch(const GgufSnapshot& snap);

// ---------------------------------------------------------------------------
// KV cache / VRAM estimation
// ---------------------------------------------------------------------------

// Bytes per element for a KV cache dtype, matching KVCache::block_nbytes() in
// src/inference/kv_cache.cpp. Block formats carry a scale, so the figure is not
// the nominal bit width: q8_0 is 34 B per 32 elements and q4_0 is 18 B per 32.
double kv_bytes_per_element(const std::string& dtype);

// Exact replica of KVCache::block_nbytes() for a row of n elements, including the
// per-row round-up to a whole block. Returns -1 for a dtype the engine cannot store.
int64_t kv_row_bytes(const std::string& dtype, int64_t n);

// Parse a KV dtype name for the estimate. Returns false if unknown.
bool parse_kv_dtype(const std::string& s, std::string* out);

// What a given context length and KV dtype costs, plus whether the model fits a
// VRAM budget. All figures are estimates and the assumptions are stated in the
// rendered output rather than hidden here.
struct MemoryEstimate {
    bool ok = false;            // false when the file lacks the metadata to estimate
    std::string reason;         // why not ok
    int64_t ctx = 0;            // context length assumed
    std::string kv_dtype;       // assumed cache dtype
    double bytes_per_elem = 2.0;

    int64_t kv_bytes_per_token = 0;
    int64_t kv_bytes_total = 0;

    int64_t weight_bytes = 0;   // tensor data as stored
    int64_t compute_reserve = 0;  // activations + workspace, a fixed overhead

    // Fit against a user-supplied budget. -1 means "no budget given".
    int64_t vram_budget = -1;
    bool fits = false;
    int64_t total_bytes = 0;
    int64_t headroom_bytes = 0;
    // How many of the model's layers would have to stay on the CPU for the rest
    // to fit; -1 when not applicable (no budget, or it already fits).
    int64_t layers_to_offload = -1;
    int64_t layers_resident = -1;
};

// `overhead` is the assumed non-weight, non-KV footprint (activations, CUDA
// context, workspace) as a fraction of weight bytes, plus a floor. Kept as a
// parameter so the caller -- and the test -- can be explicit about it.
MemoryEstimate estimate_memory(const GgufSnapshot& snap, const ArchInfo& ai, int64_t ctx,
                               const std::string& kv_dtype, int64_t vram_budget,
                               double overhead_frac = 0.15, int64_t overhead_floor_mb = 192);

std::string render_memory(const GgufSnapshot& snap, const ArchInfo& ai,
                          const MemoryEstimate& me);

std::string render_header_line(const std::string& path, const GgufSnapshot& snap);
std::string render_summary(const GgufSnapshot& snap, const ArchInfo& ai,
                           const TensorStats& stats);
std::string render_tree(const GgufSnapshot& snap, const TensorStats& stats, int depth);
std::string render_chart(const TensorStats& stats);
std::string render_metadata(const GgufSnapshot& snap, const std::string& arch);
std::string render_op_pipeline(const ArchInfo& ai);

// New sections added for the size/peaks views.
std::string render_peak_tensors(const GgufSnapshot& snap, int top);
std::string render_layer_table(const TensorStats& stats, int top);

// Side-by-side comparison of two models (human + box table styles).
struct DiffMetric {
    std::string metric;
    std::string a;
    std::string b;
};

// ---------------------------------------------------------------------------
// Machine-readable formats. The CLI builds a single LoadedModel and passes it
// here; these functions never emit ANSI color.
// ---------------------------------------------------------------------------
struct LoadedModel {
    std::string path;
    GgufSnapshot snap;
    TensorStats stats;
    ArchInfo arch;
    double parse_ms = 0.0;
    double stats_ms = 0.0;
    int threads = 0;
};

std::vector<DiffMetric> compute_diff(const LoadedModel& a, const LoadedModel& b);
std::string render_diff_text(const LoadedModel& a, const LoadedModel& b);
std::string render_diff_table(const LoadedModel& a, const LoadedModel& b);
std::string render_summary_table(const LoadedModel& m);

struct RenderOptions {
    bool summary = true;
    bool tree = false;
    bool chart = false;
    bool metadata = false;
    bool pipeline = false;
    int tree_depth = 3;
    int tree_limit = 0;  // 0 = all
    int peaks = 0;       // top N largest tensors (0 = off)
    int top_layers = 0;  // top N layer groups (0 = off)

    // Memory estimate, so the machine formats can carry it too (-M / --ctx /
    // --vram / --kv-dtype). ctx 0 = use the model's trained length;
    // vram_mb 0 = no budget to check against.
    bool memory = false;
    int64_t ctx = 0;
    int64_t vram_mb = 0;
    std::string kv_dtype;
};

std::string render_json(const LoadedModel& m, const RenderOptions& o);
std::string render_yaml(const LoadedModel& m, const RenderOptions& o);
std::string render_markdown(const LoadedModel& m, const RenderOptions& o);
std::string render_csv(const LoadedModel& m, const RenderOptions& o);
std::string render_ini(const LoadedModel& m, const RenderOptions& o);

}  // namespace forge::inspect