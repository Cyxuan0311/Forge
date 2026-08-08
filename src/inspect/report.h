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
    std::string family;     // e.g. "llama"
    std::string model_name;
    std::string file_type;  // e.g. "Q4_K_M"
    bool supported = false;
    int64_t hidden_size = 0;
    int64_t n_layers = 0;
    int64_t n_heads = 0;
    int64_t n_kv_heads = 0;
};

ArchInfo sniff_arch(const GgufSnapshot& snap);

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
};

std::string render_json(const LoadedModel& m, const RenderOptions& o);
std::string render_yaml(const LoadedModel& m, const RenderOptions& o);
std::string render_markdown(const LoadedModel& m, const RenderOptions& o);
std::string render_csv(const LoadedModel& m, const RenderOptions& o);
std::string render_ini(const LoadedModel& m, const RenderOptions& o);

}  // namespace forge::inspect