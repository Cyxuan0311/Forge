#include "report.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <unordered_set>
#include <cstdio>
#include <functional>
#include <map>

#include "arch_ops.h"
#include "color.h"
#include "format.h"
#include "meta_groups.h"

namespace forge::inspect {

namespace {

// Dimmed label helper: wraps `s` in a dim style when color is enabled.
std::string label(const char* s) {
    return col::paint(s, col::Style::Dim);
}

// Fixed-point with 3 decimals, for small ratios like "0.563 B/elem".
std::string fmt_fixed(double v, int decimals = 3) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.*f", decimals, v);
    return buf;
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
    // The KV estimate needs layers, kv heads and head dim. Derive head_dim when
    // the file only implies it, and record whether everything needed was present
    // so the caller can refuse to print a confident-looking wrong number.
    ai.head_dim = i(ap + ".attention.key_length");
    if (ai.head_dim == 0 && i(ap + ".attention.value_length") > 0)
        ai.head_dim = i(ap + ".attention.value_length");
    if (ai.head_dim == 0 && ai.n_heads > 0 && ai.hidden_size > 0)
        ai.head_dim = ai.hidden_size / ai.n_heads;
    ai.context_length = i(ap + ".context_length");
    if (ai.context_length == 0) ai.context_length = i(ap + ".max_context_length");
    // There is no GGUF key literally named "...tokens.length": the count only exists
    // as the element count of the string array, which the scanner records. Falling back
    // to the arch's own vocab_size keeps the value right for models that state it.
    ai.vocab_size = i("tokenizer.ggml.vocab_size");
    if (ai.vocab_size == 0) {
        if (auto it = snap.meta_str_array_len.find("tokenizer.ggml.tokens");
            it != snap.meta_str_array_len.end())
            ai.vocab_size = static_cast<int64_t>(it->second);
    }
    if (ai.vocab_size == 0) ai.vocab_size = i(ap + ".vocab_size");
    // Hybrid archs store a per-layer array; 0 means "no attention in this layer".
    // Forge (parse_lfm2_config / the common parser) takes max(array) as
    // num_kv_heads and then allocates KV uniformly for every layer, so the estimate
    // mirrors that rather than the tighter "only the layers that need it" number.
    if (auto it = snap.meta_int_array.find(ap + ".attention.head_count_kv");
        it != snap.meta_int_array.end() && !it->second.empty()) {
        ai.per_layer_kv = true;
        for (int32_t h : it->second) {
            if (h > 0) {
                ++ai.n_attn_layers;
                ai.n_kv_heads = std::max<int64_t>(ai.n_kv_heads, h);
            }
        }
    }
    if (ai.n_attn_layers == 0)
        ai.n_attn_layers = ai.n_layers;  // uniform model
    // Neither a scalar nor an array: MHA, where kv heads equal query heads. Resolve
    // it once here so the summary, the estimate and kv_estimable cannot disagree.
    if (ai.n_kv_heads == 0)
        ai.n_kv_heads = ai.n_heads;
    ai.n_expert = i(ap + ".expert_count");
    ai.n_expert_used = i(ap + ".expert_used_count");
    ai.sliding_window = i(ap + ".attention.sliding_window");
    {
        auto it = snap.meta_float.find(ap + ".rope.freq_base");
        if (it != snap.meta_float.end())
            ai.rope_theta = it->second;
    }
    ai.kv_estimable = ai.n_layers > 0 && ai.head_dim > 0 && ai.n_kv_heads > 0;
    ai.supported = arch_supported(ai.arch);
    // From the shared table rather than a second hand-written list, which is how
    // the family string and the operator pipeline drifted apart before.
    const char* fam = family_name(family_for_arch(ai.arch));
    ai.family = (fam && *fam) ? fam : ai.arch;
    return ai;
}


// ---------------------------------------------------------------------------
// KV / VRAM estimation
// ---------------------------------------------------------------------------

// Exact replica of KVCache::block_nbytes() for a row of `n` elements. Block
// formats round UP to a whole block per row, so a flat bytes/element ratio is
// not enough: for n=512, q8_0 is 17*34 = 578 B, not 512*1.0625 = 544 B, and
// using the ratio understates a q8_0 cache by 6%.
int64_t kv_row_bytes(const std::string& dtype, int64_t n) {
    if (n <= 0) return 0;
    if (dtype == "f16") return n * 2;
    if (dtype == "fp32") return n * 4;
    if (dtype == "q8_0") return ((n + 31) / 32) * 34;
    if (dtype == "q4_0") return ((n + 31) / 32) * 18;
    if (dtype == "q4_k") return ((n + 255) / 256) * 144;
    if (dtype == "fp8_e4m3" || dtype == "fp8_e5m2") return n;  // 1 B/elem, no blocks
    return -1;  // unknown
}

double kv_bytes_per_element(const std::string& dtype) {
    // The engine has no BF16/Q5_K/Q6_K KV dtype, so neither do we: offering them
    // would print a confident number the engine cannot actually produce.
    switch (kv_row_bytes(dtype, 4096)) {
    case -1: return -1.0;
    default: return static_cast<double>(kv_row_bytes(dtype, 4096)) / 4096.0;
    }
}

bool parse_kv_dtype(const std::string& in, std::string* out) {
    static const char* kNames[] = {"f16",  "fp32",       "q8_0",  "q4_0",
                                   "q4_k", "fp8_e4m3",   "fp8_e5m2"};
    std::string v = in;
    for (auto& c : v)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    for (const char* n : kNames) {
        if (v == n) {
            *out = n;
            return true;
        }
    }
    // Accept the fp16/fp32 aliases too.
    if (v == "fp16") { *out = "f16"; return true; }
    if (v == "f32") { *out = "fp32"; return true; }
    return false;
}

MemoryEstimate estimate_memory(const GgufSnapshot& snap, const ArchInfo& ai, int64_t ctx,
                               const std::string& kv_dtype, int64_t vram_budget,
                               double overhead_frac, int64_t overhead_floor_mb) {
    MemoryEstimate me;
    me.ctx = ctx > 0 ? ctx : (ai.context_length > 0 ? ai.context_length : 2048);
    me.vram_budget = vram_budget;
    me.kv_dtype = kv_dtype.empty() ? "f16" : kv_dtype;

    if (!ai.kv_estimable) {
        me.reason = "file does not state enough metadata to size the KV cache "
                    "(need block_count and a head dim)";
        return me;
    }

    // GQA: kv heads default to query heads when not stated (MHA).
    const int64_t kv_heads = ai.n_kv_heads;

    // Both K and V, every layer, using the engine's per-row block rounding so the
    // total matches the KV Cache size Forge itself reports.
    const int64_t row_elems = kv_heads * ai.head_dim;
    const int64_t row = kv_row_bytes(me.kv_dtype, row_elems);
    if (row < 0) {
        me.reason = "unknown KV dtype '" + me.kv_dtype + "'";
        return me;
    }
    me.kv_bytes_per_token = 2 * ai.n_layers * row;
    // The true figure can exceed the flat ratio because each row rounds up to a
    // whole block; report what is actually allocated, not the nominal rate.
    me.bytes_per_elem = static_cast<double>(row) / static_cast<double>(row_elems);
    me.kv_bytes_total = me.kv_bytes_per_token * me.ctx;

    // Weight bytes as stored, from the tensors themselves rather than the file
    // size, so the split is honest (the file also carries metadata and padding).
    for (const auto& t : snap.tensors)
        me.weight_bytes += t.data_size;

    // No term for the recurrent/short-conv state that hybrid models carry: it is a
    // fixed size (l_cache channels per layer), independent of the context length, and
    // in the kilobytes against weights of gigabytes. Modelling it would add a
    // guess without moving the total.

    me.compute_reserve = std::max<int64_t>(
        static_cast<int64_t>(static_cast<double>(me.weight_bytes) * overhead_frac),
        overhead_floor_mb * 1024 * 1024);

    me.total_bytes = me.weight_bytes + me.kv_bytes_total + me.compute_reserve;

    if (vram_budget > 0) {
        me.headroom_bytes = vram_budget - me.total_bytes;
        me.fits = me.headroom_bytes >= 0;
        if (!me.fits && ai.n_layers > 0 && me.weight_bytes > 0) {
            // Assume weights are spread evenly across layers, which holds well
            // except for MoE (where experts dominate) -- noted in the output.
            const int64_t per_layer = me.weight_bytes / ai.n_layers;
            const int64_t budget_for_weights =
                vram_budget - me.kv_bytes_total - me.compute_reserve;
            me.layers_resident = per_layer > 0 ? std::max<int64_t>(0, budget_for_weights / per_layer)
                                               : 0;
            if (me.layers_resident > ai.n_layers)
                me.layers_resident = ai.n_layers;
            me.layers_to_offload = ai.n_layers - me.layers_resident;
        }
    }
    me.ok = true;
    return me;
}

std::string render_memory(const GgufSnapshot& snap, const ArchInfo& ai, const MemoryEstimate& me) {
    std::string out;
    if (!me.ok) {
        out += "  " + label("memory") + "      " + col::paint("cannot estimate: " + me.reason,
                                                                   col::Style::Yellow) +
               "\n";
        return out;
    }
    out += "  " + label("kv dtype") + "    " + col::paint(me.kv_dtype, col::Style::Cyan) +
           "  (" + fmt_fixed(me.bytes_per_elem) + " B/elem)\n";
    out += "  " + label("context") + "     " +
           col::paint(std::to_string(me.ctx) + " tokens", col::Style::Cyan) + "  (" +
           std::to_string(ai.n_layers) + " layers x " + std::to_string(ai.n_kv_heads) +
           " kv heads x " + std::to_string(ai.head_dim) + " head dim)\n";
    if (ai.per_layer_kv && ai.n_attn_layers < ai.n_layers) {
        out += "  " + label("") + "             " +
               col::paint("only " + std::to_string(ai.n_attn_layers) + " of " +
                              std::to_string(ai.n_layers) +
                              " layers have attention; Forge allocates KV for all of them",
                          col::Style::Dim) +
               "\n";
    }
    out += "  " + label("kv / token") + "  " +
           col::paint(human_bytes(me.kv_bytes_per_token), col::Style::Yellow) + "\n";
    out += "  " + label("kv total") + "    " +
           col::paint(human_bytes(me.kv_bytes_total), col::Style::Yellow) + "\n";
    out += "  " + label("weights") + "    " +
           col::paint(human_bytes(me.weight_bytes), col::Style::Yellow) + "\n";
    out += "  " + label("compute") + "     " +
           col::paint("~" + human_bytes(me.compute_reserve) + " (assumed overhead)",
                      col::Style::Dim) +
           "\n";
    out += "  " + label("total") + "       " +
           col::paint(human_bytes(me.total_bytes), col::Style::Bold) + "\n";
    if (me.vram_budget > 0) {
        if (me.fits) {
            out += "  " + label("vram") + "        " +
                   col::paint("fits " + human_bytes(me.vram_budget) + " with " +
                                  human_bytes(me.headroom_bytes) + " to spare",
                              col::Style::Green) +
                   "\n";
        } else {
            out += "  " + label("vram") + "        " +
                   col::paint("OVER by " + human_bytes(-me.headroom_bytes) + " against " +
                                  human_bytes(me.vram_budget),
                              col::Style::Red) +
                   "\n";
            if (me.layers_to_offload >= 0) {
                out += "  " + label("offload") + "     " +
                       col::paint(std::to_string(me.layers_to_offload) + " of " +
                                      std::to_string(ai.n_layers) + " layers to CPU (est.)",
                                  col::Style::Yellow) +
                       "\n";
                if (ai.n_expert > 0)
                    out += "  " + label("") + "             " +
                           col::paint("note: MoE weights are unevenly sized, so the per-layer "
                                      "split is a rough estimate",
                                      col::Style::Dim) +
                           "\n";
            }
            // Shrinking the context is always available, and for long-context models
            // it is by far the biggest lever on the cache.
            out += "  " + label("") + "             " +
                   col::paint("shrinking the context would free up to " +
                                  human_bytes(me.kv_bytes_total),
                              col::Style::Dim) +
                   "\n";
        }
    }
    (void)snap;
    return out;
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
    // Size the bar to the terminal instead of a fixed 40, which ran off the edge of
    // an 80-column terminal. The trailing "(N tensors)" differs per line, so allow
    // for the widest one: 8 for the name, 2+2 around the bar, 6 for the percentage,
    // 2 more, and the suffix.
    size_t widest = 0;
    for (const auto& d : stats.by_dtype)
        widest = std::max(widest, std::string(" (" + std::to_string(d.count) + " tensors)").size());
    const int fixed = 8 + 2 + 2 + 6 + 2 + static_cast<int>(widest);
    const int bar_max = std::max(10, terminal_width(80) - fixed);
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
// Validation
// ---------------------------------------------------------------------------

namespace {

// Layer index for the two naming conventions in the wild: llama.cpp's
// "blk.N.attn_q.weight" and the HF-style "model.layers.N.self_attn.q_proj".
// Returns -1 when the name is not per-layer.
int layer_index_of(const std::string& name) {
    static const char* kMarkers[] = {"blk.", "layers.", "layer.", "block_", "blocks."};
    for (const char* m : kMarkers) {
        const size_t p = name.find(m);
        if (p == std::string::npos) continue;
        const size_t s = p + std::strlen(m);
        if (s >= name.size() || !std::isdigit(static_cast<unsigned char>(name[s]))) continue;
        int64_t v = 0;
        size_t i = s;
        while (i < name.size() && std::isdigit(static_cast<unsigned char>(name[i])))
            v = v * 10 + (name[i++] - '0');
        return static_cast<int>(v);
    }
    return -1;
}

void add(std::vector<Finding>& out, Severity sev, const char* code, std::string msg) {
    out.push_back({sev, code, std::move(msg)});
}

}  // namespace

std::vector<Finding> validate_model(const GgufSnapshot& snap, const ArchInfo& ai) {
    std::vector<Finding> out;
    if (ai.arch.empty()) {
        add(out, Severity::Error, "arch_missing",
            "general.architecture is absent, so the file cannot be loaded by any engine");
        // Without an arch almost nothing else can be cross-checked.
        return out;
    }
    if (!ai.supported) {
        add(out, Severity::Warning, "arch_unknown",
            "architecture '" + ai.arch + "' is not one Forge supports; loading will fail");
    }
    if (ai.n_layers <= 0) {
        add(out, Severity::Error, "block_count_missing",
            ai.arch + ".block_count is missing or zero, so the layer count is unknown");
    }

    // Which layer indices do the tensors actually mention?
    std::vector<int> seen;
    {
        std::unordered_set<int> uniq;
        for (const auto& t : snap.tensors) {
            const int li = layer_index_of(t.name);
            if (li >= 0) uniq.insert(li);
        }
        seen.assign(uniq.begin(), uniq.end());
        std::sort(seen.begin(), seen.end());
    }

    if (ai.n_layers > 0 && !seen.empty()) {
        const int max_layer = seen.back();
        if (max_layer + 1 != ai.n_layers) {
            add(out, Severity::Error, "block_count_mismatch",
                ai.arch + ".block_count is " + std::to_string(ai.n_layers) +
                    " but the tensors cover layers 0.." + std::to_string(max_layer) +
                    " (" + std::to_string(seen.size()) + " distinct); the file is truncated or "
                    "the count is wrong");
        }
        // Layers declared but entirely absent from the tensor list.
        if (static_cast<int>(seen.size()) < ai.n_layers) {
            std::string missing;
            int n_missing = 0;
            for (int l = 0; l < ai.n_layers && n_missing < 4; ++l) {
                if (!std::binary_search(seen.begin(), seen.end(), l)) {
                    if (!missing.empty()) missing += ", ";
                    missing += std::to_string(l);
                    ++n_missing;
                }
            }
            if (n_missing > 0) {
                add(out, Severity::Error, "layers_absent",
                    "layer(s) " + missing + " are in block_count but have no tensors at all" +
                        (ai.n_layers - static_cast<int>(seen.size()) - n_missing > 0
                             ? " (and others)"
                             : ""));
            }
        }
    } else if (ai.n_layers > 0 && seen.empty()) {
        add(out, Severity::Info, "layers_unnamed",
            "no per-layer tensors found (no blk.N / layers.N names), so the layer count "
            "could not be cross-checked");
    }

    // Duplicate tensor names would silently drop weights on load.
    {
        std::unordered_set<std::string> seen_names;
        std::string dup;
        int n_dup = 0;
        for (const auto& t : snap.tensors) {
            if (!seen_names.insert(t.name).second) {
                if (!dup.empty()) dup += ", ";
                if (n_dup < 3) dup += t.name;
                ++n_dup;
            }
        }
        if (n_dup > 0)
            add(out, Severity::Error, "tensor_duplicate",
                std::to_string(n_dup) + " duplicate tensor name(s): " + dup +
                    (n_dup > 3 ? ", ..." : "") + "; loaders keep only the last one");
    }

    // Geometry.
    if (ai.n_heads > 0 && ai.hidden_size > 0) {
        if (ai.hidden_size % ai.n_heads != 0)
            add(out, Severity::Error, "hidden_not_divisible_by_heads",
                "hidden_size " + std::to_string(ai.hidden_size) + " is not divisible by " +
                    std::to_string(ai.n_heads) + " heads, so head_dim " +
                    std::to_string(ai.head_dim) + " is a truncation and the model is malformed");
    }
    if (ai.n_kv_heads > 0 && ai.n_heads > 0 && ai.n_kv_heads > ai.n_heads)
        add(out, Severity::Error, "kv_heads_exceed_heads",
            ai.arch + ".attention.head_count_kv (" + std::to_string(ai.n_kv_heads) +
                ") exceeds head_count (" + std::to_string(ai.n_heads) + ")");
    if (ai.n_kv_heads > 0 && ai.n_heads > 0 && ai.n_kv_heads < ai.n_heads &&
        ai.n_heads % ai.n_kv_heads != 0)
        add(out, Severity::Warning, "heads_not_divisible_by_kv",
            std::to_string(ai.n_heads) + " query heads is not a multiple of " +
                std::to_string(ai.n_kv_heads) +
                " KV heads; GQA implementations normally divide evenly");
    if (ai.n_expert_used > 0 && ai.n_expert > 0 && ai.n_expert_used > ai.n_expert)
        add(out, Severity::Error, "expert_used_exceeds_total",
            ai.arch + ".expert_used_count (" + std::to_string(ai.n_expert_used) +
                ") exceeds expert_count (" + std::to_string(ai.n_expert) + ")");

    // The per-layer KV array must line up with the layer count, or every
    // layer-indexed lookup into it is suspect.
    if (auto it = snap.meta_int_array.find(ai.arch + ".attention.head_count_kv");
        it != snap.meta_int_array.end() && ai.n_layers > 0 &&
        static_cast<int64_t>(it->second.size()) != ai.n_layers) {
        add(out, Severity::Error, "kv_array_length",
            "attention.head_count_kv has " + std::to_string(it->second.size()) +
                " entries but block_count is " + std::to_string(ai.n_layers) +
                "; per-layer KV heads cannot be looked up");
    }

    // Vocabulary vs the embedding matrix.
    if (ai.vocab_size > 0) {
        for (const auto& t : snap.tensors) {
            if (t.name.find("token_embd") == std::string::npos) continue;
            // GGUF stores shape in ggml order, so the vocabulary may be either the
            // first or the last dim depending on who wrote the file. Any dim matching
            // the tokenizer count is fine; none matching is a real mismatch.
            if (t.shape.size() >= 2 &&
                std::find(t.shape.begin(), t.shape.end(), ai.vocab_size) == t.shape.end()) {
                std::string dims;
                for (size_t i = 0; i < t.shape.size(); ++i) {
                    if (i) dims += "x";
                    dims += std::to_string(t.shape[i]);
                }
                add(out, Severity::Error, "vocab_mismatch",
                    "token_embd.weight is " + dims + " but the tokenizer lists " +
                        std::to_string(ai.vocab_size) +
                        " tokens; sampling will read out of bounds");
            }
            break;
        }
    } else {
        add(out, Severity::Warning, "vocab_missing",
            "tokenizer.ggml.tokens is absent, so the vocabulary size is unknown");
    }

    // Data extents. The scanner already rejects a bad index; this catches a file
    // whose tensor table is consistent with itself but overruns the file.
    {
        uint64_t sum = 0;
        uint64_t end = 0;
        for (const auto& t : snap.tensors) {
            sum += static_cast<uint64_t>(t.data_size);
            const uint64_t stop = static_cast<uint64_t>(t.file_offset) +
                                  static_cast<uint64_t>(t.data_size);
            end = std::max(end, stop);
        }
        if (end > snap.file_size) {
            add(out, Severity::Error, "tensor_data_past_eof",
                "tensor data extends to byte " + std::to_string(end) +
                    " but the file is only " + std::to_string(snap.file_size) +
                    " bytes; the file is truncated");
        }
        // Metadata and alignment overhead: a big gap is normal, a huge one is not.
        if (snap.file_size > 0 && end > 0 && sum > 0) {
            const double covered = static_cast<double>(end) / static_cast<double>(snap.file_size);
            if (covered <= 1.0 && covered < 0.80 && snap.tensor_count > 0) {
                add(out, Severity::Info, "tensor_coverage",
                    "tensor data covers only " +
                        fmt_fixed(covered * 100.0, 1) + "% of the file; the rest is metadata, "
                        "padding or data not described by the tensor table");
            }
        }
    }

    if (!ai.kv_estimable && ai.n_layers > 0)
        add(out, Severity::Info, "kv_not_estimable",
            "no head dim could be derived, so the KV cache size cannot be estimated");

    std::stable_sort(out.begin(), out.end(), [](const Finding& a, const Finding& b) {
        return static_cast<int>(a.severity) < static_cast<int>(b.severity);
    });
    return out;
}

std::string render_validation(const std::vector<Finding>& findings) {
    std::string out;
    if (findings.empty()) {
        out += "  " + col::paint("no problems found", col::Style::Green) + "\n";
        return out;
    }
    int n_err = 0, n_warn = 0;
    for (const auto& f : findings) {
        if (f.severity == Severity::Error) ++n_err;
        else if (f.severity == Severity::Warning) ++n_warn;
    }
    if (n_err)
        out += "  " + col::paint(std::to_string(n_err) + " error" + (n_err > 1 ? "s" : ""),
                                 col::Style::Red) +
               "\n";
    if (n_warn)
        out += "  " + col::paint(std::to_string(n_warn) + " warning" + (n_warn > 1 ? "s" : ""),
                                 col::Style::Yellow) +
               "\n";
    for (const auto& f : findings) {
        const char* tag = "info";
        col::Style st = col::Style::Dim;
        if (f.severity == Severity::Error) {
            tag = "error";
            st = col::Style::Red;
        } else if (f.severity == Severity::Warning) {
            tag = "warn ";
            st = col::Style::Yellow;
        }
        out += "  " + col::paint(std::string(tag) + " ", st) + col::paint(f.code, col::Style::Bold) +
               "  " + f.message + "\n";
    }
    return out;
}

// New sections: largest tensors + layer byte table.
// ---------------------------------------------------------------------------

std::string render_peak_tensors(const GgufSnapshot& snap, int top, const std::string& filter,
                                int* matched) {
    std::string out;
    std::string needle = filter;
    for (auto& c : needle)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    std::vector<size_t> order;
    order.reserve(snap.tensors.size());
    for (size_t i = 0; i < snap.tensors.size(); ++i) {
        if (needle.empty()) {
            order.push_back(i);
            continue;
        }
        std::string name = snap.tensors[i].name;
        for (auto& c : name)
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (name.find(needle) != std::string::npos) order.push_back(i);
    }
    if (matched)
        *matched = static_cast<int>(order.size());
    // Sort by size descending, but break ties by name so repeated runs agree.
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        if (snap.tensors[a].data_size != snap.tensors[b].data_size)
            return snap.tensors[a].data_size > snap.tensors[b].data_size;
        return snap.tensors[a].name < snap.tensors[b].name;
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
    if (n < order.size())
        out += "  " + col::paint("... and " + std::to_string(order.size() - n) + " more",
                                 col::Style::Dim) +
               "\n";
    if (order.empty() && !needle.empty())
        out += "  " + col::paint("no tensor matches '" + filter + "'", col::Style::Yellow) + "\n";
    return out;
}

std::string render_tensor_detail(const GgufSnapshot& snap, const std::string& name, bool* found) {
    if (found)
        *found = false;
    for (const auto& t : snap.tensors) {
        if (t.name != name) continue;
        if (found)
            *found = true;
        int64_t numel = 1;
        for (int64_t d : t.shape)
            numel *= d;
        std::string dims;
        for (size_t i = 0; i < t.shape.size(); ++i) {
            if (i) dims += " x ";
            dims += std::to_string(t.shape[i]);
        }
        std::string out;
        auto row = [&](const char* k, const std::string& v) {
            out += "  " + label(k) + "     " + v + "\n";
        };
        row("name", col::paint(t.name, col::Style::Bold));
        row("dtype", col::paint(dtype_name(t.orig_dtype), col::Style::Yellow));
        row("shape", dims.empty() ? std::string("scalar") : dims);
        row("elements", std::to_string(numel));
        row("bytes", col::paint(human_bytes(t.data_size), col::Style::Cyan) + " (" +
                        std::to_string(t.data_size) + ")");
        row("offset", std::to_string(t.file_offset));
        // The packed size the scanner derived versus what a loader would compute from
        // the dtype: a disagreement means the index and the dtype disagree.
        const int64_t expect = packed_bytes(t.orig_dtype, numel);
        if (expect != t.data_size) {
            out += "  " + label("") + "       " +
                   col::paint("packed size " + std::to_string(expect) + " != stored " +
                                  std::to_string(t.data_size) + " B",
                              col::Style::Yellow) +
                   "\n";
        }
        return out;
    }
    return std::string();
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

namespace {

std::string shape_of(const std::vector<int64_t>& dims) {
    if (dims.empty()) return "-";
    std::string out;
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i) out += "x";
        out += std::to_string(dims[i]);
    }
    return out;
}

}  // namespace

TensorDiff compute_tensor_diff(const GgufSnapshot& a, const GgufSnapshot& b) {
    TensorDiff td;
    std::unordered_map<std::string, const GgufLoadedTensor*> ia, ib;
    ia.reserve(a.tensors.size() * 2);
    ib.reserve(b.tensors.size() * 2);
    for (const auto& t : a.tensors)
        ia.emplace(t.name, &t);
    for (const auto& t : b.tensors)
        ib.emplace(t.name, &t);

    for (const auto& [name, t] : ia) {
        auto it = ib.find(name);
        if (it == ib.end()) {
            td.only_a.push_back(name);
            continue;
        }
        ++td.common;
        TensorDelta d;
        d.name = name;
        d.dtype_a = dtype_name(t->orig_dtype);
        d.dtype_b = dtype_name(it->second->orig_dtype);
        d.shape_a = shape_of(t->shape);
        d.shape_b = shape_of(it->second->shape);
        d.bytes_a = t->data_size;
        d.bytes_b = it->second->data_size;
        if (d.dtype_a != d.dtype_b) d.changed = "dtype";
        if (d.shape_a != d.shape_b) {
            d.changed = d.changed.empty() ? "shape" : d.changed + "+shape";
        }
        if (d.bytes_a != d.bytes_b) {
            d.changed = d.changed.empty() ? "size" : d.changed + "+size";
        }
        if (d.changed.empty())
            ++td.common_identical;
        else
            td.changed.push_back(std::move(d));
    }
    for (const auto& [name, t] : ib) {
        (void)t;
        if (ia.find(name) == ia.end()) td.only_b.push_back(name);
    }
    auto by_name = [](const std::string& x, const std::string& y) { return x < y; };
    std::sort(td.only_a.begin(), td.only_a.end(), by_name);
    std::sort(td.only_b.begin(), td.only_b.end(), by_name);
    std::sort(td.changed.begin(), td.changed.end(),
              [](const TensorDelta& x, const TensorDelta& y) { return x.name < y.name; });
    return td;
}

namespace {

// One line per category, with the count elided when zero so a clean diff is quiet.
std::string td_line(const char* name, size_t n, col::Style st) {
    return "  " + label(name) + "   " +
           col::paint(std::to_string(n), n ? st : col::Style::Dim) + "\n";
}

void td_names(std::string& out, const char* label, const std::vector<std::string>& names, int top,
              col::Style st) {
    if (names.empty()) return;
    out += "  " + std::string(label) + "  " +
           col::paint(std::to_string(names.size()), st) + "\n";
    const size_t n = top > 0 ? std::min<size_t>(static_cast<size_t>(top), names.size())
                             : names.size();
    for (size_t i = 0; i < n; ++i)
        out += "      " + names[i] + "\n";
    if (n < names.size())
        out += "      ... and " + std::to_string(names.size() - n) + " more\n";
}

}  // namespace

std::string render_tensor_diff(const TensorDiff& td, int top, bool table) {
    (void)table;
    std::string out;
    out += "  " + label("common") + "    " + std::to_string(td.common) + "  (" +
           std::to_string(td.common_identical) + " identical, " +
           std::to_string(static_cast<int64_t>(td.changed.size())) + " changed)\n";
    out += td_line("only in a", td.only_a.size(), col::Style::Yellow);
    out += td_line("only in b", td.only_b.size(), col::Style::Yellow);
    out += td_line("changed", td.changed.size(), col::Style::Cyan);
    if (td.only_a.empty() && td.only_b.empty() && td.changed.empty()) {
        out += "  " + col::paint("the two tensor indexes are identical", col::Style::Green) + "\n";
        return out;
    }
    td_names(out, "only in a:", td.only_a, top, col::Style::Yellow);
    td_names(out, "only in b:", td.only_b, top, col::Style::Yellow);
    if (!td.changed.empty()) {
        out += "  changed:\n";
        const size_t n = top > 0 ? std::min<size_t>(static_cast<size_t>(top), td.changed.size())
                                 : td.changed.size();
        for (size_t i = 0; i < n; ++i) {
            const auto& d = td.changed[i];
            out += "      " + d.name + "  " + col::paint(d.changed, col::Style::Cyan);
            if (d.changed.find("dtype") != std::string::npos)
                out += "  " + d.dtype_a + " -> " + d.dtype_b;
            if (d.changed.find("shape") != std::string::npos)
                out += "  " + d.shape_a + " -> " + d.shape_b;
            if (d.changed.find("size") != std::string::npos)
                out += "  " + human_bytes(d.bytes_a) + " -> " + human_bytes(d.bytes_b);
            out += "\n";
        }
        if (n < td.changed.size())
            out += "      ... and " + std::to_string(td.changed.size() - n) + " more\n";
    }
    return out;
}

std::string render_tensor_diff_json(const TensorDiff& td, int top) {
    const std::string IND = "  ";
    const size_t limit = top > 0 ? static_cast<size_t>(top) : SIZE_MAX;
    auto arr = [&](const char* key, const std::vector<std::string>& names) {
        std::string o = std::string(IND) + IND + json_str(key) + ": [";
        for (size_t i = 0; i < names.size() && i < limit; ++i)
            o += (i ? ", " : "") + json_str(names[i]);
        o += "]";
        return o;
    };
    std::string out = "{\n";
    out += IND + IND + json_str("common") + ": " + std::to_string(td.common) + ",\n";
    out += IND + IND + json_str("identical") + ": " + std::to_string(td.common_identical) + ",\n";
    out += IND + IND + json_str("changed_count") + ": " +
           std::to_string(static_cast<int64_t>(td.changed.size())) + ",\n";
    out += arr("only_in_a", td.only_a) + ",\n";
    out += arr("only_in_b", td.only_b) + ",\n";
    out += IND + IND + json_str("changed") + ": [";
    for (size_t i = 0; i < td.changed.size() && i < limit; ++i) {
        const auto& d = td.changed[i];
        out += (i ? ",\n" : "\n");
        out += IND + IND + IND + "{ " + json_str("name") + ": " + json_str(d.name) + ", " +
               json_str("changed") + ": " + json_str(d.changed) + ", " + json_str("dtype_a") +
               ": " + json_str(d.dtype_a) + ", " + json_str("dtype_b") + ": " + json_str(d.dtype_b) +
               ", " + json_str("shape_a") + ": " + json_str(d.shape_a) + ", " + json_str("shape_b") +
               ": " + json_str(d.shape_b) + ", " + json_str("bytes_a") + ": " +
               std::to_string(d.bytes_a) + ", " + json_str("bytes_b") + ": " +
               std::to_string(d.bytes_b) + " }";
    }
    out += (td.changed.empty() ? "" : "\n" + IND + IND);
    out += "]\n}\n";
    return out;
}

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

// Re-indent a rendered "{...}" object by `pad` spaces so it nests cleanly. Used to
// embed each single-model report as one element of the models array.
std::string indent_block(const std::string& block, const std::string& pad) {
    std::string out;
    out.reserve(block.size() + pad.size() * 8);
    for (size_t i = 0; i < block.size(); ++i) {
        const char c = block[i];
        if (c == '\n' && i + 1 < block.size() && block[i + 1] != '\n')
            out += pad;
        out += c;
    }
    if (!out.empty() && out.back() == '\n')
        out.pop_back();
    return out;
}

std::string render_multi_json(const std::vector<LoadedModel>& models, const RenderOptions& o) {
    const std::string IND = "  ";
    std::string out = "{\n";
    out += IND + "\"models\": [\n";
    for (size_t i = 0; i < models.size(); ++i) {
        if (i) out += ",\n";
        out += IND + IND + indent_block(render_json(models[i], o), IND + IND);
    }
    out += "\n" + IND + "],\n";

    out += IND + "\"diff\": [\n";
    for (size_t i = 1; i < models.size(); ++i) {
        if (i > 1) out += ",\n";
        out += IND + IND + "{ \"against\": " + json_str(models[i].path) + ",\n";
        out += IND + IND + IND + "\"summary\": {";
        const auto dm = compute_diff(models[0], models[i]);
        for (size_t k = 0; k < dm.size(); ++k) {
            if (k) out += ", ";
            out += json_str(dm[k].metric) + ": { \"a\": " + json_str(dm[k].a) + ", \"b\": " +
                   json_str(dm[k].b) + " }";
        }
        out += " },\n";
        out += IND + IND + IND + "\"tensors\": ";
        out += indent_block(render_tensor_diff_json(compute_tensor_diff(models[0].snap, models[i].snap),
                                                    0),
                            IND + IND + IND);
        out += "\n" + IND + IND + "}";
    }
    out += "\n" + IND + "]\n}\n";
    return out;
}

std::string render_multi_yaml(const std::vector<LoadedModel>& models, const RenderOptions& o) {
    std::string out = "models:\n";
    for (size_t i = 0; i < models.size(); ++i) {
        out += "  - ";
        std::string one = render_yaml(models[i], o);
        // render_yaml emits a top-level mapping; indent it under the list item.
        for (size_t k = 0; k < one.size(); ++k) {
            const char c = one[k];
            out += c;
            if (c == '\n' && k + 1 < one.size())
                out += "    ";
        }
        if (!one.empty() && one.back() != '\n')
            out += "\n";
    }
    if (models.size() > 1) {
        out += "diff:\n";
        for (size_t i = 1; i < models.size(); ++i) {
            out += "  - against: " + models[i].path + "\n";
            out += "    summary:\n";
            for (const auto& d : compute_diff(models[0], models[i]))
                out += "      " + d.metric + ": {a: " + d.a + ", b: " + d.b + "}\n";
            out += "    tensors:\n";
            const auto td = compute_tensor_diff(models[0].snap, models[i].snap);
            out += "      common: " + std::to_string(td.common) + "\n";
            out += "      identical: " + std::to_string(td.common_identical) + "\n";
            out += "      changed: " + std::to_string(static_cast<int64_t>(td.changed.size())) + "\n";
            out += "      only_in_a:\n";
            for (const auto& n : td.only_a)
                out += "        - " + n + "\n";
            out += "      only_in_b:\n";
            for (const auto& n : td.only_b)
                out += "        - " + n + "\n";
        }
    }
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
    const size_t nlay = std::min(o.top_layers > 0 ? static_cast<size_t>(o.top_layers)
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
    // memory estimate (opt-in, before the trailing metadata section)
    if (o.memory) {
        const auto me = estimate_memory(s, ai, o.ctx, o.kv_dtype,
                                        o.vram_mb > 0 ? o.vram_mb * 1024 * 1024 : -1);
        out += IND + "\"memory\": {\n";
        if (!me.ok) {
            out += IND + IND + json_str("error") + ": " + json_str(me.reason) + "\n";
        } else {
            out += IND + IND + json_str("kv_dtype") + ": " + json_str(me.kv_dtype) + ",\n";
            out += IND + IND + json_str("bytes_per_element") + ": " +
                   fmt_fixed(me.bytes_per_elem, 4) + ",\n";
            out += IND + IND + json_str("context") + ": " + std::to_string(me.ctx) + ",\n";
            out += IND + IND + json_str("n_layers") + ": " + std::to_string(ai.n_layers) + ",\n";
            out += IND + IND + json_str("n_attn_layers") + ": " +
                   std::to_string(ai.n_attn_layers) + ",\n";
            out += IND + IND + json_str("n_kv_heads") + ": " + std::to_string(ai.n_kv_heads) + ",\n";
            out += IND + IND + json_str("head_dim") + ": " + std::to_string(ai.head_dim) + ",\n";
            out += IND + IND + json_str("kv_bytes_per_token") + ": " +
                   std::to_string(me.kv_bytes_per_token) + ",\n";
            out += IND + IND + json_str("kv_bytes_total") + ": " +
                   std::to_string(me.kv_bytes_total) + ",\n";
            out += IND + IND + json_str("weight_bytes") + ": " +
                   std::to_string(me.weight_bytes) + ",\n";
            out += IND + IND + json_str("compute_reserve_bytes") + ": " +
                   std::to_string(me.compute_reserve) + ",\n";
            out += IND + IND + json_str("total_bytes") + ": " + std::to_string(me.total_bytes);
            if (me.vram_budget > 0) {
                out += ",\n" + IND + IND + json_str("vram_budget_bytes") + ": " +
                       std::to_string(me.vram_budget) + ",\n";
                out += IND + IND + json_str("fits") + ": " + (me.fits ? "true" : "false");
                if (!me.fits && me.layers_to_offload >= 0) {
                    out += ",\n" + IND + IND + json_str("layers_to_offload") + ": " +
                           std::to_string(me.layers_to_offload);
                }
            }
            out += "\n";
        }
        out += IND + "},\n";
    }
    // validation findings (opt-in)
    if (o.validate) {
        const auto fs = validate_model(s, ai);
        out += IND + "\"validation\": {\n";
        int n_err = 0;
        for (const auto& f : fs)
            if (f.severity == Severity::Error) ++n_err;
        out += IND + IND + json_str("errors") + ": " + std::to_string(n_err) + ",\n";
        out += IND + IND + json_str("ok") + ": " + std::string(n_err == 0 ? "true" : "false") +
               ",\n";
        out += IND + IND + "\"findings\": [";
        for (size_t i = 0; i < fs.size(); ++i) {
            const char* sev = fs[i].severity == Severity::Error
                                  ? "error"
                                  : (fs[i].severity == Severity::Warning ? "warning" : "info");
            out += (i ? ",\n" : "\n");
            out += IND + IND + IND + "{ " + json_str("severity") + ": " + json_str(sev) + ", " +
                   json_str("code") + ": " + json_str(fs[i].code) + ", " + json_str("message") +
                   ": " + json_str(fs[i].message) + " }";
        }
        out += (fs.empty() ? "" : "\n" + IND + IND);
        out += "]\n";
        out += IND + "},\n";
    }
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
    const size_t nlay = std::min(o.top_layers > 0 ? static_cast<size_t>(o.top_layers)
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
    const size_t nlay = std::min(o.top_layers > 0 ? static_cast<size_t>(o.top_layers)
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

    // The -M / -V sections were silently absent from ini output before, so a script
    // reading ini never saw them no matter what was asked for.
    if (o.memory) {
        const auto me = estimate_memory(s, ai, o.ctx, o.kv_dtype,
                                        o.vram_mb > 0 ? o.vram_mb * 1024 * 1024 : -1);
        out += "\n[memory]\n";
        out += "ok = " + std::string(me.ok ? "yes" : "no") + "\n";
        if (!me.ok) {
            out += "reason = " + me.reason + "\n";
        } else {
            out += "kv_dtype = " + me.kv_dtype + "\n";
            out += "bytes_per_element = " + fmt_fixed(me.bytes_per_elem) + "\n";
            out += "context = " + std::to_string(me.ctx) + "\n";
            out += "attn_layers = " + std::to_string(ai.n_attn_layers) + "\n";
            out += "kv_bytes_per_token = " + std::to_string(me.kv_bytes_per_token) + "\n";
            out += "kv_bytes_total = " + std::to_string(me.kv_bytes_total) + "\n";
            out += "weight_bytes = " + std::to_string(me.weight_bytes) + "\n";
            out += "compute_reserve_bytes = " + std::to_string(me.compute_reserve) + "\n";
            out += "total_bytes = " + std::to_string(me.total_bytes) + "\n";
            if (me.vram_budget > 0) {
                out += "vram_budget_bytes = " + std::to_string(me.vram_budget) + "\n";
                out += "fits = " + std::string(me.fits ? "yes" : "no") + "\n";
                out += "layers_to_offload = " + std::to_string(me.layers_to_offload) + "\n";
            }
        }
    }
    if (o.validate) {
        const auto fs = validate_model(s, ai);
        int n_err = 0;
        for (const auto& f : fs)
            if (f.severity == Severity::Error) ++n_err;
        out += "\n[validation]\n";
        out += "ok = " + std::string(n_err == 0 ? "yes" : "no") + "\n";
        out += "errors = " + std::to_string(n_err) + "\n";
        out += "findings = " + std::to_string(static_cast<int64_t>(fs.size())) + "\n";
        for (size_t i = 0; i < fs.size(); ++i) {
            const char* sev = fs[i].severity == Severity::Error
                                  ? "error"
                                  : (fs[i].severity == Severity::Warning ? "warning" : "info");
            out += "finding_" + std::to_string(i + 1) + " = " + sev + " " + fs[i].code + ": " +
                   fs[i].message + "\n";
        }
    }
    return out;
}

}  // namespace forge::inspect