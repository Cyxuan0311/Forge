#include "report.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <functional>
#include <map>

#include "arch_ops.h"
#include "color.h"
#include "meta_groups.h"

namespace forge::inspect {

namespace {

// Dimmed label helper: wraps `s` in a dim style when color is enabled.
std::string label(const char* s) {
    return col::paint(s, col::Style::Dim);
}

std::string human_bytes(int64_t bytes) {
    static const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    double v = static_cast<double>(bytes);
    int u = 0;
    while (v >= 1024.0 && u < 4) {
        v /= 1024.0;
        ++u;
    }
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.1f %s", v, units[u]);
    return buf;
}

// (key, value) pairs for the model summary block, shared by text and table out.
std::vector<std::pair<std::string, std::string>> summary_rows(const LoadedModel& m) {
    std::vector<std::pair<std::string, std::string>> out;
    const ArchInfo& ai = m.arch;
    const TensorStats& st = m.stats;
    const GgufSnapshot& s = m.snap;
    out.emplace_back("file", m.path);
    out.emplace_back("size", human_bytes(static_cast<int64_t>(s.file_size)));
    out.emplace_back("gguf_version", "v" + std::to_string(s.version));
    out.emplace_back("tensors", std::to_string(s.tensor_count));
    out.emplace_back("kv_pairs", std::to_string(s.metadata_count));
    out.emplace_back("arch", ai.arch.empty() ? "(unknown)" : ai.arch);
    out.emplace_back("family", ai.family.empty() ? "-" : ai.family);
    out.emplace_back("model", ai.model_name.empty() ? "(unnamed)" : ai.model_name);
    out.emplace_back("file_type", ai.file_type.empty() ? "(unspecified)" : ai.file_type);
    out.emplace_back("hidden_size", ai.hidden_size ? std::to_string(ai.hidden_size) : "-");
    out.emplace_back("layers", ai.n_layers ? std::to_string(ai.n_layers) : "-");
    out.emplace_back("heads", ai.n_heads ? std::to_string(ai.n_heads) : "-");
    if (ai.n_kv_heads)
        out.emplace_back("kv_heads", std::to_string(ai.n_kv_heads));
    out.emplace_back("supported", ai.supported ? "yes" : "no");
    out.emplace_back("quant", human_bytes(st.total_bytes));
    out.emplace_back("elements", std::to_string(st.total_elements));
    if (!st.largest_tensor_name.empty())
        out.emplace_back("largest", st.largest_tensor_name + " (" +
                                         human_bytes(st.peak_tensor_bytes) + ")");
    return out;
}

// llama.cpp gguf_file_type enum -> name. Matches the numbers stored in
// "general.file_type" (an int, not a string).
const char* file_type_name(int64_t ft) {
    switch (ft) {
        case 1: return "ALL_F32";
        case 2: return "MOSTLY_F16";
        case 3: return "MOSTLY_Q4_0";
        case 4: return "MOSTLY_Q4_1";
        case 5: return "MOSTLY_Q4_2";
        case 6: return "MOSTLY_Q4_3";
        case 7: return "MOSTLY_Q8_0";
        case 8: return "MOSTLY_Q5_0";
        case 9: return "MOSTLY_Q5_1";
        case 10: return "MOSTLY_Q2_K";
        case 11: return "MOSTLY_Q3_K_S";
        case 12: return "MOSTLY_Q3_K_M";
        case 13: return "MOSTLY_Q3_K_L";
        case 14: return "MOSTLY_Q4_K_S";
        case 15: return "MOSTLY_Q4_K_M";
        case 16: return "MOSTLY_Q5_K_S";
        case 17: return "MOSTLY_Q5_K_M";
        case 18: return "MOSTLY_Q6_K";
        case 19: return "MOSTLY_Q8_K";
        case 20: return "MOSTLY_IQ2_XXS";
        case 21: return "MOSTLY_IQ2_XS";
        case 22: return "MOSTLY_IQ3_XXS";
        case 23: return "MOSTLY_IQ1_S";
        case 24: return "MOSTLY_IQ4_NL";
        case 25: return "MOSTLY_IQ3_S";
        case 26: return "MOSTLY_IQ2_S";
        case 27: return "MOSTLY_IQ4_XS";
        case 28: return "MOSTLY_IQ1_M";
        case 29: return "MOSTLY_BF16";
        case 30: return "MOSTLY_Q4_0_4_4";
        case 31: return "MOSTLY_Q4_0_4_8";
        case 32: return "MOSTLY_Q4_0_8_8";
        case 33: return "MOSTLY_TQ1_0";
        case 34: return "MOSTLY_TQ2_0";
        default: return "(unknown)";
    }
}

// A node in the rendered tensor tree. Build uses the map (insertion by path);
// after conversion, children is a sorted vector.
struct TNode {
    std::string name;
    std::string value;  // e.g. "[Q4_K]"
    std::map<std::string, TNode> m_children;
    std::vector<TNode> children;
};

std::vector<TNode> build_nodes(const GgufSnapshot& snap, const TensorStats& stats,
                               int depth) {
    auto split = [](const std::string& s) {
        std::vector<std::string> parts;
        std::string cur;
        for (char c : s) {
            if (c == '.') {
                parts.push_back(cur);
                cur.clear();
            } else {
                cur.push_back(c);
            }
        }
        if (!cur.empty())
            parts.push_back(cur);
        return parts;
    };

    std::map<std::string, TNode> roots;
    for (const auto& t : snap.tensors) {
        std::vector<std::string> parts = split(t.name);
        if (parts.empty())
            continue;
        std::map<std::string, TNode>* level = &roots;
        std::string path;
        size_t i = 0;
        for (; i + 1 < parts.size() && i < static_cast<size_t>(depth); ++i) {
            path += (i ? "." : "") + parts[i];
            level = &((*level)[path].m_children);
        }
        // Leaf: the remaining segments joined, preserving the tensor name.
        std::string leaf_name = parts[i];
        for (size_t j = i + 1; j < parts.size(); ++j)
            leaf_name += "." + parts[j];
        (*level)[leaf_name].name = leaf_name;
        (*level)[leaf_name].value = "[" + dtype_name(t.orig_dtype) + "]";
    }

    // Convert map -> sorted vector, collapsing single-child chains.
    using Converter = std::function<std::vector<TNode>(const std::map<std::string, TNode>&)>;
    Converter convert = [&convert](const std::map<std::string, TNode>& m) {
        std::vector<TNode> out;
        out.reserve(m.size());
        for (const auto& [name, n] : m) {
            TNode copy;
            copy.name = n.name.empty() ? name : n.name;
            copy.value = n.value;
            copy.m_children = n.m_children;
            if (copy.m_children.size() == 1) {
                auto& only = copy.m_children.begin()->second;
                copy.name += "." + only.name;
                copy.value = only.value;
                copy.m_children = only.m_children;
            }
            copy.children = convert(copy.m_children);
            out.push_back(std::move(copy));
        }
        return out;
    };

    return convert(roots);
}

// JSON escaping (no ANSI involved here).
std::string json_esc(const std::string& s) {
    std::string o;
    for (char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\t': o += "\\t"; break;
            case '\r': o += "\\r"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char b[8];
                    std::snprintf(b, sizeof(b), "\\u%04x", c);
                    o += b;
                } else {
                    o += c;
                }
        }
    }
    return o;
}

std::string json_str(std::string s) { return "\"" + json_esc(s) + "\""; }

// Number columns left-aligned where they are labels, else right-aligned.
bool cells_are_numeric(const std::vector<std::string>& cells) {
    for (const std::string& c : cells)
        if (c.empty() || !std::isdigit(static_cast<unsigned char>(c[0])))
            return false;
    return !cells.empty();
}

}  // namespace

ArchInfo sniff_arch(const GgufSnapshot& snap) {
    ArchInfo ai;
    auto s = [&](const std::string& k) -> std::string {
        auto it = snap.meta_str.find(k);
        return it == snap.meta_str.end() ? std::string() : it->second;
    };
    auto i = [&](const std::string& k) -> int64_t {
        auto it = snap.meta_int.find(k);
        return it == snap.meta_int.end() ? 0 : it->second;
    };
    ai.arch = s("general.architecture");
    ai.model_name = s("general.name");
    if (auto it = snap.meta_int.find("general.file_type"); it != snap.meta_int.end())
        ai.file_type = file_type_name(it->second);
    const std::string ap = ai.arch.empty() ? "arch" : ai.arch;
    ai.hidden_size = i(ap + ".embedding_length");
    ai.n_layers = i(ap + ".block_count");
    ai.n_heads = i(ap + ".attention.head_count");
    ai.n_kv_heads = i(ap + ".attention.head_count_kv");
    ai.supported = arch_supported(ai.arch);
    if (ai.arch == "llama" || ai.arch == "mistral" || ai.arch == "yi" || ai.arch == "qwen" ||
        ai.arch == "qwen2" || ai.arch == "qwen3vl" || ai.arch == "phi3" || ai.arch == "falcon")
        ai.family = "gqa";
    else if (ai.arch == "gemma" || ai.arch == "gemma2")
        ai.family = "gemma";
    else if (ai.arch == "deepseek" || ai.arch == "deepseek_v2" || ai.arch == "deepseek_v3")
        ai.family = "mla";
    else if (ai.arch == "phimoe" || ai.arch == "gemma4")
        ai.family = "moe";
    else if (ai.arch == "qwen35")
        ai.family = "hybrid";
    else
        ai.family = ai.arch;
    return ai;
}

std::string render_header_line(const std::string& path, const GgufSnapshot& snap) {
    std::string out;
    out += col::paint(path, col::Style::BoldCyan);
    out += "  (" + col::paint(human_bytes(static_cast<int64_t>(snap.file_size)),
                              col::Style::Yellow);
    out += ", GGUF v" + col::paint(std::to_string(snap.version), col::Style::Cyan) + ")\n";
    return out;
}

std::string render_summary(const GgufSnapshot& snap, const ArchInfo& ai,
                           const TensorStats& stats) {
    std::string out;
    out += "  " + label("arch") + "        " +
           col::paint(ai.arch.empty() ? "(unknown)" : ai.arch, col::Style::BoldMagenta) +
           "\n";
    out += "  " + label("model") + "       " +
           col::paint(ai.model_name.empty() ? "(unnamed)" : ai.model_name,
                      col::Style::Bold) +
           "\n";
    out += "  " + label("file_type") + "   " +
           col::paint(ai.file_type.empty() ? "(unspecified)" : ai.file_type,
                      col::Style::Yellow) +
           "\n";
    out += "  " + label("tensors") + "     " +
           col::paint(std::to_string(snap.tensor_count), col::Style::Cyan) + "  (" +
           col::paint(human_bytes(stats.total_bytes), col::Style::Yellow) + ")\n";
    out += "  " + label("hidden_size") + " " +
           col::paint(std::to_string(ai.hidden_size), col::Style::Cyan) + "\n";
    out += "  " + label("layers") + "      " +
           col::paint(std::to_string(ai.n_layers), col::Style::Cyan) + "\n";
    out += "  " + label("heads") + "       " +
           col::paint(std::to_string(ai.n_heads), col::Style::Cyan) + " / kv " +
           col::paint(std::to_string(ai.n_kv_heads), col::Style::Cyan) + "\n";
    out += "  " + label("support") + "     " +
           (ai.supported ? col::paint("yes", col::Style::BoldGreen)
                         : col::paint("no", col::Style::BoldRed)) +
           "\n";
    if (!stats.largest_tensor_name.empty()) {
        out += "  " + label("largest") + "     " +
               col::paint(stats.largest_tensor_name, col::Style::Bold) + "  (" +
               col::paint(human_bytes(stats.peak_tensor_bytes), col::Style::Yellow) + ")\n";
    }
    return out;
}

std::string render_tree(const GgufSnapshot& snap, const TensorStats& stats, int depth) {
    std::string out;
    if (depth <= 0)
        return out;
    auto tree = build_nodes(snap, stats, depth);
    std::function<void(const std::vector<TNode>&, const std::string&)> render_nodes =
        [&](const std::vector<TNode>& nodes, const std::string& prefix) {
            for (size_t i = 0; i < nodes.size(); ++i) {
                const auto& n = nodes[i];
                const bool last = (i + 1 == nodes.size());
                out += col::paint(prefix + (last ? "└─ " : "├─ "), col::Style::Dim);
                if (n.children.empty() && !n.value.empty()) {
                    out += n.name;
                    out += "  " + col::paint(n.value, col::Style::Yellow);
                } else {
                    out += col::paint(n.name, col::Style::BoldBlue);
                }
                out += "\n";
                render_nodes(n.children, prefix + (last ? "   " : "│  "));
            }
        };
    render_nodes(tree, "");
    return out;
}

std::string render_chart(const TensorStats& stats) {
    std::string out;
    if (stats.by_dtype.empty())
        return out;
    const int64_t max_bytes = stats.by_dtype.front().bytes > 0 ? stats.by_dtype.front().bytes : 1;
    const int bar_max = 40;
    for (const auto& d : stats.by_dtype) {
        int len = static_cast<int>((d.bytes * bar_max) / max_bytes);
        if (len < 1)
            len = 1;
        std::string bar;
        bar.reserve(static_cast<size_t>(len) * 3);
        for (int i = 0; i < len; ++i)
            bar += "█";
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%5.1f%%",
                      100.0 * static_cast<double>(d.bytes) /
                          static_cast<double>(stats.total_bytes));
        out += std::string(8 - (int)d.name.size(), ' ');
        out += col::paint(d.name, col::Style::BoldMagenta);
        out += "  " + col::paint(bar, col::Style::BoldGreen) + "  ";
        out += col::paint(std::string(buf), col::Style::BoldYellow) + "  (";
        out += col::paint(std::to_string(d.count), col::Style::Dim) + " tensors)\n";
    }
    return out;
}

std::string render_metadata(const GgufSnapshot& snap, const std::string& arch) {
    std::string out;
    auto groups = group_metadata(snap, arch);
    for (const auto& g : groups) {
        out += col::paint("[" + g.name + "]", col::Style::BoldCyan) + "\n";
        for (const auto& [k, v] : g.entries)
            out += "  " + col::paint(k, col::Style::Dim) + " = " + v + "\n";
    }
    return out;
}

std::string render_op_pipeline(const ArchInfo& ai) {
    std::string out;
    const ArchOpPipeline* p = pipeline_for(ai.arch);
    if (!p) {
        out += "  No operator pipeline registered for arch \"" + ai.arch +
               "\".\n";
        return out;
    }
    auto emit = [&](const std::string& title, const std::vector<OpStep>& steps) {
        out += "  " + col::paint(title, col::Style::Bold) + "\n";
        for (size_t i = 0; i < steps.size(); ++i) {
            bool last = (i + 1 == steps.size());
            std::string glyph = last ? "└─ " : "├─ ";
            out += "    " + col::paint(glyph, col::Style::Dim) +
                   col::paint(steps[i].op, col::Style::BoldMagenta);
            if (!steps[i].label.empty())
                out += "  " + col::paint("(" + steps[i].label + ")", col::Style::Dim);
            out += "\n";
        }
    };
    if (!p->pre_steps.empty())
        emit("before layers", p->pre_steps);
    emit("per layer (x " + std::to_string(ai.n_layers) + ")", p->layer_steps);
    if (!p->post_steps.empty())
        emit("after layers", p->post_steps);
    return out;
}

// ---------------------------------------------------------------------------
// New sections: largest tensors + layer byte table.
// ---------------------------------------------------------------------------

std::string render_peak_tensors(const GgufSnapshot& snap, int top) {
    std::string out;
    std::vector<size_t> order(snap.tensors.size());
    for (size_t i = 0; i < order.size(); ++i)
        order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return snap.tensors[a].data_size > snap.tensors[b].data_size;
    });
    const size_t n = std::min(static_cast<size_t>(top), order.size());
    for (size_t i = 0; i < n; ++i) {
        const auto& t = snap.tensors[order[i]];
        int64_t numel = 1;
        for (int64_t d : t.shape)
            numel *= d;
        char cnt[24];
        std::snprintf(cnt, sizeof(cnt), "%lld elem", static_cast<long long>(numel));
        out += "  " + col::paint(t.name, col::Style::Bold) + "  " +
               col::paint(dtype_name(t.orig_dtype), col::Style::Yellow) + "  " +
               col::paint(human_bytes(t.data_size), col::Style::Cyan) + "  (" +
               col::paint(cnt, col::Style::Dim) + ")\n";
    }
    return out;
}

std::string render_layer_table(const TensorStats& stats, int top) {
    std::string out;
    size_t n = std::min(static_cast<size_t>(top), stats.layer_bytes.size());
    for (size_t i = 0; i < n; ++i) {
        const auto& [name, bytes] = stats.layer_bytes[i];
        const double pct = stats.total_bytes ? 100.0 * static_cast<double>(bytes) /
                                                  static_cast<double>(stats.total_bytes)
                                             : 0.0;
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%4.1f%%  ", pct);
        out += "  " + col::paint(name, col::Style::Dim) + "  " +
               col::paint(human_bytes(bytes), col::Style::Cyan) + "  " +
               col::paint(buf, col::Style::BoldYellow) + "\n";
    }
    return out;
}

// ---------------------------------------------------------------------------
// Machine-readable formats. All plain: no ANSI, no terminal-drawing.
// ---------------------------------------------------------------------------

std::vector<DiffMetric> compute_diff(const LoadedModel& a, const LoadedModel& b) {
    std::vector<DiffMetric> out;
    auto add = [&](const char* m, std::function<std::string(const LoadedModel&)> f) {
        out.push_back({m, f(a), f(b)});
    };
    add("size", [](const LoadedModel& x) { return human_bytes(static_cast<int64_t>(x.snap.file_size)); });
    add("tensors", [](const LoadedModel& x) { return std::to_string(x.snap.tensor_count); });
    add("kv_pairs", [](const LoadedModel& x) { return std::to_string(x.snap.metadata_count); });
    add("quant", [](const LoadedModel& x) { return human_bytes(x.stats.total_bytes); });
    add("elements", [](const LoadedModel& x) { return std::to_string(x.stats.total_elements); });
    add("layers", [](const LoadedModel& x) { return std::to_string(x.arch.n_layers); });
    add("file_type", [](const LoadedModel& x) { return x.arch.file_type; });
    return out;
}

std::string render_diff_text(const LoadedModel& a, const LoadedModel& b) {
    std::string out;
    for (const auto& d : compute_diff(a, b)) {
        out += "  " + label(d.metric.c_str()) + "  ";
        const int pad = static_cast<int>(18 - d.metric.size());
        if (pad > 0)
            out += std::string(static_cast<size_t>(pad), ' ');
        out += col::paint(d.a, col::Style::Cyan) + "  →  " +
               col::paint(d.b, col::Style::Yellow) + "\n";
    }
    return out;
}

std::string render_diff_table(const LoadedModel& a, const LoadedModel& b) {
    const std::string name_a = a.path.substr(a.path.find_last_of("/\\") + 1);
    const std::string name_b = b.path.substr(b.path.find_last_of("/\\") + 1);
    std::vector<std::vector<std::string>> cells;
    for (const auto& d : compute_diff(a, b))
        cells.push_back({d.metric, d.a, d.b});
    return box_table({"metric", name_a, name_b}, cells);
}

std::string render_summary_table(const LoadedModel& m) {
    std::vector<std::vector<std::string>> cells;
    for (const auto& r : summary_rows(m)) {
        if (r.first == "file")
            continue;
        cells.push_back({r.first, r.second});
    }
    return box_table({"key", "value"}, cells);
}

namespace {

// Escaped forms for YAML (plain scalar unless it needs quoting).
std::string yaml_scalar(const std::string& s) {
    if (s.empty())
        return "''";
    const bool need = s.find_first_of(":#'\" \t\n") != std::string::npos ||
                      s == "null" || s == "true" || s == "false";
    if (!need)
        return s;
    std::string o = "'";
    for (char c : s) {
        if (c == '\'')
            o += "''";
        else
            o += c;
    }
    return o + "'";
}

std::string md_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '|')
            o += "\\|";
        else
            o += c;
    }
    return o;
}

}  // namespace

std::string render_json(const LoadedModel& m, const RenderOptions& o) {
    const GgufSnapshot& s = m.snap;
    const ArchInfo& ai = m.arch;
    const TensorStats& st = m.stats;
    std::string out;
    const std::string IND = "  ";
    out += "{\n";
    // file + timing
    out += IND + "\"file\": {\n";
    out += IND + IND + json_str("path") + ": " + json_str(m.path) + ",\n";
    out += IND + IND + json_str("size_bytes") + ": " + std::to_string(s.file_size) + ",\n";
    out += IND + IND + json_str("gguf_version") + ": " + std::to_string(s.version) + ",\n";
    out += IND + IND + json_str("tensor_count") + ": " + std::to_string(s.tensor_count) + ",\n";
    out += IND + IND + json_str("kv_count") + ": " + std::to_string(s.metadata_count) + "\n";
    out += IND + "},\n";
    out += IND + "\"parse\": {\n";
    char tbuf[32];
    std::snprintf(tbuf, sizeof(tbuf), "%.3f", m.parse_ms);
    out += IND + IND + "\"parse_ms\": " + tbuf + ",\n";
    std::snprintf(tbuf, sizeof(tbuf), "%.3f", m.stats_ms);
    out += IND + IND + "\"stats_ms\": " + tbuf + ",\n";
    out += IND + IND + "\"threads\": " + std::to_string(m.threads) + "\n";
    out += IND + "},\n";
    // arch
    out += IND + "\"arch\": {\n";
    out += IND + IND + "\"name\": " + json_str(ai.arch) + ",\n";
    out += IND + IND + "\"family\": " + json_str(ai.family) + ",\n";
    out += IND + IND + "\"model\": " + json_str(ai.model_name) + ",\n";
    out += IND + IND + "\"file_type\": " + json_str(ai.file_type) + ",\n";
    out += IND + IND + "\"supported\": " + std::string(ai.supported ? "true" : "false") + ",\n";
    out += IND + IND + "\"hidden\": " + std::to_string(ai.hidden_size) + ",\n";
    out += IND + IND + "\"layers\": " + std::to_string(ai.n_layers) + ",\n";
    out += IND + IND + "\"heads\": " + std::to_string(ai.n_heads) + ",\n";
    out += IND + IND + "\"kv_heads\": " + std::to_string(ai.n_kv_heads) + "\n";
    out += IND + "},\n";
    // summary
    out += IND + "\"summary\": {\n";
    out += IND + IND + "\"total_bytes\": " + std::to_string(st.total_bytes) + ",\n";
    out += IND + IND + "\"total_elements\": " + std::to_string(st.total_elements) + ",\n";
    out += IND + IND + "\"largest_bytes\": " + std::to_string(st.peak_tensor_bytes) + "\n";
    out += IND + "},\n";
    // dtypes
    out += IND + "\"dtypes\": [";
    for (size_t i = 0; i < st.by_dtype.size(); ++i) {
        const auto& d = st.by_dtype[i];
        out += (i ? ",\n" : "\n");
        out += IND + IND + "{\n";
        out += IND + IND + IND + "\"dtype\": " + json_str(d.name) + ",\n";
        out += IND + IND + IND + "\"bytes\": " + std::to_string(d.bytes) + ",\n";
        out += IND + IND + IND + "\"count\": " + std::to_string(d.count) + ",\n";
        out += IND + IND + IND + "\"elements\": " + std::to_string(d.elements) + "\n";
        out += IND + IND + "}";
    }
    out += st.by_dtype.empty() ? "],\n" : "\n" + IND + "],\n";
// layers
    const size_t nlay = std::max(o.top_layers > 0 ? static_cast<size_t>(o.top_layers)
                                                   : st.layer_bytes.size(),
                                 st.layer_bytes.size());
    out += IND + "\"layers\": [";
    for (size_t i = 0; i < nlay; ++i) {
        out += (i ? ",\n" : "\n");
        out += IND + IND + "{\n";
        out += IND + IND + IND + json_str("name") + ": " + json_str(st.layer_bytes[i].first) + ",\n";
        out += IND + IND + IND + json_str("bytes") + ": " + std::to_string(st.layer_bytes[i].second) + "\n";
        out += IND + IND + "}";
    }
    out += st.layer_bytes.empty() || nlay == 0 ? "],\n" : "\n" + IND + "],\n";
    // metadata (last section, no trailing comma)
    out += IND + "\"metadata\": {\n";
    const auto groups = group_metadata(s, ai.arch);
    for (size_t g = 0; g < groups.size(); ++g) {
        const auto& grp = groups[g];
        out += IND + IND + json_str(grp.name) + ": {\n";
        for (size_t e = 0; e < grp.entries.size(); ++e) {
            const auto& [k, v] = grp.entries[e];
            out += IND + IND + IND + json_str(k) + ": " + json_str(v);
            out += (e + 1 < grp.entries.size() ? ",\n" : "\n");
        }
        out += IND + IND + "}";
        out += (g + 1 < groups.size() ? ",\n" : "\n");
    }
    out += IND + "}\n";
    out += "}\n";
    return out;
}

std::string render_yaml(const LoadedModel& m, const RenderOptions& o) {
    const GgufSnapshot& s = m.snap;
    const ArchInfo& ai = m.arch;
    const TensorStats& st = m.stats;
    std::string out;
    out += "file:\n";
    out += "  path: " + yaml_scalar(m.path) + "\n";
    out += "  size_bytes: " + std::to_string(s.file_size) + "\n";
    out += "  gguf_version: " + std::to_string(s.version) + "\n";
    out += "  tensor_count: " + std::to_string(s.tensor_count) + "\n";
    out += "  kv_count: " + std::to_string(s.metadata_count) + "\n";
    out += "parse:\n";
    char tbuf[32];
    std::snprintf(tbuf, sizeof(tbuf), "%.3f", m.parse_ms);
    out += "  parse_ms: " + std::string(tbuf) + "\n";
    std::snprintf(tbuf, sizeof(tbuf), "%.3f", m.stats_ms);
    out += "  stats_ms: " + std::string(tbuf) + "\n";
    out += "  threads: " + std::to_string(m.threads) + "\n";
    out += "arch:\n";
    out += "  name: " + yaml_scalar(ai.arch) + "\n";
    out += "  family: " + yaml_scalar(ai.family) + "\n";
    out += "  model: " + yaml_scalar(ai.model_name) + "\n";
    out += "  file_type: " + yaml_scalar(ai.file_type) + "\n";
    out += "  supported: " + std::string(ai.supported ? "true" : "false") + "\n";
    out += "  hidden: " + std::to_string(ai.hidden_size) + "\n";
    out += "  layers: " + std::to_string(ai.n_layers) + "\n";
    out += "  heads: " + std::to_string(ai.n_heads) + "\n";
    out += "  kv_heads: " + std::to_string(ai.n_kv_heads) + "\n";
    out += "summary:\n";
    out += "  total_bytes: " + std::to_string(st.total_bytes) + "\n";
    out += "  total_elements: " + std::to_string(st.total_elements) + "\n";
    out += "  largest_bytes: " + std::to_string(st.peak_tensor_bytes) + "\n";
    out += "dtypes:\n";
    for (const auto& d : st.by_dtype) {
        out += "  - dtype: " + yaml_scalar(d.name) + "\n";
        out += "    bytes: " + std::to_string(d.bytes) + "\n";
        out += "    count: " + std::to_string(d.count) + "\n";
        out += "    elements: " + std::to_string(d.elements) + "\n";
    }
    const size_t nlay = std::max(o.top_layers > 0 ? static_cast<size_t>(o.top_layers)
                                                  : st.layer_bytes.size(),
                                 st.layer_bytes.size());
    if (nlay) {
        out += "layers:\n";
        for (size_t i = 0; i < nlay; ++i) {
            out += "  - name: " + yaml_scalar(st.layer_bytes[i].first) + "\n";
            out += "    bytes: " + std::to_string(st.layer_bytes[i].second) + "\n";
        }
    }
    out += "metadata:\n";
    for (const auto& g : group_metadata(s, ai.arch)) {
        out += "  " + yaml_scalar(g.name) + ":\n";
        for (const auto& [k, v] : g.entries)
            out += "    " + yaml_scalar(k) + ": " + yaml_scalar(v) + "\n";
    }
    return out;
}

std::string render_markdown(const LoadedModel& m, const RenderOptions& o) {
    const GgufSnapshot& s = m.snap;
    const ArchInfo& ai = m.arch;
    const TensorStats& st = m.stats;
    std::string out;
    out += "# forge-inspect: " + md_escape(m.path) + "\n\n";
    char tbuf[64];
    std::snprintf(tbuf, sizeof(tbuf), "*parsed in %.1f ms + %.1f ms stats*\n\n", m.parse_ms,
                  m.stats_ms);
    out += tbuf;
    out += "## General\n\n";
    out += "| key | value |\n|---|---|\n";
    out += "| size | " + human_bytes(static_cast<int64_t>(s.file_size)) + " |\n";
    out += "| tensor_count | " + std::to_string(s.tensor_count) + " |\n";
    out += "| kv_count | " + std::to_string(s.metadata_count) + " |\n";
    out += "| arch | " + md_escape(ai.arch) + " |\n";
    out += "| model | " + md_escape(ai.model_name) + " |\n";
    out += "| file_type | " + md_escape(ai.file_type) + " |\n\n";
    out += "## Dtype sizes\n\n";
    out += "| dtype | bytes | count | elements |\n|---|---|---|---|\n";
    for (const auto& d : st.by_dtype)
        out += "| " + md_escape(d.name) + " | " + std::to_string(d.bytes) + " | " +
                std::to_string(d.count) + " | " + std::to_string(d.elements) + " |\n";
    out += "\n## Largest tensors\n\n";
    out += "| name | dtype | elements | bytes |\n|---|---|---|---|\n";
    std::vector<size_t> order(s.tensors.size());
    for (size_t i = 0; i < order.size(); ++i)
        order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return s.tensors[a].data_size > s.tensors[b].data_size;
    });
    const size_t npeaks = std::min(o.peaks > 0 ? static_cast<size_t>(o.peaks) : order.size(),
                                   order.size());
    for (size_t i = 0; i < npeaks; ++i) {
        const auto& t = s.tensors[order[i]];
        int64_t numel = 1;
        for (int64_t d : t.shape)
            numel *= d;
        out += "| " + md_escape(t.name) + " | " + md_escape(dtype_name(t.orig_dtype)) + " | " +
               std::to_string(numel) + " | " + std::to_string(t.data_size) + " |\n";
    }
    if (o.tree) {
        out += "\n## Tensor tree\n\n```text\n";
        const auto nodes = build_nodes(s, st, o.tree_depth);
        std::function<void(const std::vector<TNode>&, const std::string&)> emit =
            [&](const std::vector<TNode>& ns, const std::string& prefix) {
                for (size_t i = 0; i < ns.size(); ++i) {
                    const auto& n = ns[i];
                    const bool last = (i + 1 == ns.size());
                    out += prefix + (last ? "└─ " : "├─ ") + n.name;
                    if (!n.value.empty())
                        out += "  " + n.value;
                    out += "\n";
                    emit(n.children, prefix + (last ? "   " : "│  "));
                }
            };
        emit(nodes, "");
        out += "```\n";
    }
    return out;
}

std::string render_csv(const LoadedModel& m, const RenderOptions& o) {
    const GgufSnapshot& s = m.snap;
    const ArchInfo& ai = m.arch;
    const TensorStats& st = m.stats;
    auto join = [](const std::vector<std::string>& cells) {
        std::string out;
        for (size_t i = 0; i < cells.size(); ++i) {
            if (i)
                out += ",";
            bool need = cells[i].find_first_of(",\"\r\n") != std::string::npos;
            if (need) {
                out += '"';
                for (char c : cells[i])
                    out += (c == '"') ? "\"\"" : std::string(1, c);
                out += '"';
            } else {
                out += cells[i];
            }
        }
        return out;
    };
    std::string out;
    out += "section,general\n";
    out += "# file\n";
    out += join({"size", std::to_string(s.file_size)}) + "\n";
    out += join({"tensor_count", std::to_string(s.tensor_count)}) + "\n";
    out += join({"kv_count", std::to_string(s.metadata_count)}) + "\n";
    out += join({"arch", ai.arch}) + "\n";
    out += join({"model", ai.model_name}) + "\n";
    out += join({"file_type", ai.file_type}) + "\n";
    out += "# dtypes\n";
    out += join({"dtype", "bytes", "count", "elements"}) + "\n";
    for (const auto& d : st.by_dtype)
        out += join({d.name, std::to_string(d.bytes), std::to_string(d.count),
                     std::to_string(d.elements)}) +
               "\n";
    const size_t nlay = std::max(o.top_layers > 0 ? static_cast<size_t>(o.top_layers)
                                                 : st.layer_bytes.size(),
                                 st.layer_bytes.size());
    if (nlay) {
        out += "# layers\n";
        out += join({"layer", "bytes"}) + "\n";
        for (size_t i = 0; i < nlay; ++i)
            out += join({st.layer_bytes[i].first, std::to_string(st.layer_bytes[i].second)}) +
                   "\n";
    }
    out += "# peaks\n";
    out += join({"name", "dtype", "elements", "bytes"}) + "\n";
    std::vector<size_t> order(s.tensors.size());
    for (size_t i = 0; i < order.size(); ++i)
        order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return s.tensors[a].data_size > s.tensors[b].data_size;
    });
    const size_t npeaks = std::min(o.peaks > 0 ? static_cast<size_t>(o.peaks) : order.size(),
                                   order.size());
    for (size_t i = 0; i < npeaks; ++i) {
        const auto& t = s.tensors[order[i]];
        int64_t numel = 1;
        for (int64_t d : t.shape)
            numel *= d;
        out += join({t.name, dtype_name(t.orig_dtype), std::to_string(numel),
                     std::to_string(t.data_size)}) +
               "\n";
    }
    return out;
}

std::string render_ini(const LoadedModel& m, const RenderOptions& o) {
    const GgufSnapshot& s = m.snap;
    const ArchInfo& ai = m.arch;
    const TensorStats& st = m.stats;
    std::string out;
    out += "; forge-inspect report\n";
    out += "\n[file]\n";
    out += "path = " + m.path + "\n";
    out += "size = " + std::to_string(s.file_size) + "\n";
    out += "gguf_version = " + std::to_string(s.version) + "\n";
    out += "tensors = " + std::to_string(s.tensor_count) + "\n";
    out += "kv_pairs = " + std::to_string(s.metadata_count) + "\n\n";
    out += "[arch]\n";
    out += "name = " + (ai.arch.empty() ? "unknown" : ai.arch) + "\n";
    out += "family = " + (ai.family.empty() ? "unknown" : ai.family) + "\n";
    out += "model = " + (ai.model_name.empty() ? "unnamed" : ai.model_name) + "\n";
    out += "file_type = " + (ai.file_type.empty() ? "unspecified" : ai.file_type) + "\n";
    out += "supported = " + std::string(ai.supported ? "yes" : "no") + "\n";
    out += "hidden = " + std::to_string(ai.hidden_size) + "\n";
    out += "layers = " + std::to_string(ai.n_layers) + "\n";
    out += "heads = " + std::to_string(ai.n_heads) + "\n";
    out += "kv_heads = " + std::to_string(ai.n_kv_heads) + "\n\n";
    out += "[summary]\n";
    out += "total_bytes = " + std::to_string(st.total_bytes) + "\n";
    out += "total_elements = " + std::to_string(st.total_elements) + "\n";
    out += "largest_bytes = " + std::to_string(st.peak_tensor_bytes) + "\n";
    out += "largest_name = " + (st.largest_tensor_name.empty() ? "none" : st.largest_tensor_name) +
           "\n\n";
    out += "[dtypes]\n";
    for (size_t i = 0; i < st.by_dtype.size(); ++i) {
        const auto& d = st.by_dtype[i];
        out += "dtype_" + std::to_string(i + 1) + "_name = " + d.name + "\n";
        out += "dtype_" + std::to_string(i + 1) + "_bytes = " + std::to_string(d.bytes) + "\n";
        out += "dtype_" + std::to_string(i + 1) + "_count = " + std::to_string(d.count) + "\n";
    }
    return out;
}

}  // namespace forge::inspect