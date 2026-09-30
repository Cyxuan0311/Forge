#include <chrono>
#include <cstdio>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#undef NOMINMAX
#endif

#include "arch_ops.h"
#include "color.h"
#include "format.h"
#include "perf/gguf_scanner.h"
#include "report.h"

namespace {

using forge::inspect::col::Mode;

struct Options {
    std::string path;
    std::vector<std::string> others;  // additional models for diff output
    forge::inspect::Format format = forge::inspect::Format::Text;
    bool summary = false;
    bool tree = false;
    bool chart = false;
    bool metadata = false;
    bool pipeline = false;
    bool all = false;
    int tree_depth = 2;
    int tree_limit = 0;
    int peaks = 0;         // --top N
    int top_layers = 0;    // --layers N
    int64_t ctx = 0;       // --ctx N   (0 = use the model's trained length)
    int64_t vram = 0;      // --vram MB  (0 = no budget to check against)
    std::string kv_dtype;  // --kv-dtype NAME (empty = f16)
    bool memory = false;
    bool validate = false;
    bool tensor_diff = false;   // -d, --tensor-diff
    std::string filter;         // --filter: substring of the tensor name
    std::string tensor;         // --tensor: exact name
    int threads = 0;       // 0 = auto
    std::string out_file;
    forge::inspect::col::Mode color = forge::inspect::col::Mode::Auto;
    bool help = false;
};

void print_help(const char* prog) {
    std::printf(
        "forge-inspect - inspect GGUF model files\n"
        "\n"
        "Usage: %s [options] <file.gguf> [<file2.gguf> ...]\n"
        "\n"
        "Modes (default: summary; -a shows everything):\n"
        "  -s, --summary       show model summary\n"
        "  -t, --tree [DEPTH]  tensor directory tree (default depth 2)\n"
        "  -c, --chart         dtype size bar chart\n"
        "  -m, --metadata      dump metadata KV pairs\n"
        "  -p, --pipeline      show architecture operator pipeline\n"
        "  -a, --all           show all of the above\n"
        "\n"
        "Sizing views:\n"
        "      --top N         list the N largest tensors\n"
        "      --layers N      per-layer byte table for the top N layers\n"
        "\n"
        "Memory estimates:\n"
        "  -V, --validate      cross-check the metadata against the tensors\n"
        "      --filter SUBSTR only tensors whose name contains SUBSTR\n"
        "      --tensor NAME  full detail for one tensor, by exact name\n"
        "  -d, --tensor-diff   per-tensor diff against the second model (header level:\n"
        "                      names, dtypes and shapes, not the weight values)\n"
        "  -M, --memory        KV cache + VRAM estimate for the model\n"
        "      --ctx N         context length to assume (default: the model's own)\n"
        "      --kv-dtype D    KV cache dtype for the estimate (default f16);\n"
        "                      f16|fp32|q8_0|q4_0|q4_k|fp8_e4m3|fp8_e5m2\n"
        "      --vram MB       VRAM budget to check against, and report how many\n"
        "                      layers would have to stay on the CPU if it does not fit\n"
        "\n"
        "Output format:\n"
        "  -f, --format NAME   text|table|json|yaml|markdown|csv|ini (default text)\n"
        "\n"
        "Options:\n"
        "  -j, --threads N     parallel parsing threads (default: hardware)\n"
        "  -o, --out FILE      write output to FILE instead of stdout\n"
        "      --color=MODE    color output: auto|always|never (default auto)\n"
        "      --no-color      same as --color=never\n"
        "  -h, --help          show this help\n"
        "\n"
        "Multiple files are compared side-by-side (text/table only):\n"
        "    %s a.gguf b.gguf --format table\n",
        prog, prog);
}

// Strict integer parse. strtol() stops at the first non-digit, so "--top 5x" was
// silently accepted as 5 and "--layers 3oops" as 3. A typo in a count is a
// number the user believes they set and did not, which is worse than an error.
// Returns false on trailing garbage, empty input, or out-of-int range.
bool parse_int_arg(const char* s, int* out) {
    if (s == nullptr || *s == '\0')
        return false;
    char* end = nullptr;
    errno = 0;
    const long v = std::strtol(s, &end, 10);
    if (errno == ERANGE || end == s || *end != '\0')
        return false;
    if (v < std::numeric_limits<int>::min() || v > std::numeric_limits<int>::max())
        return false;
    *out = static_cast<int>(v);
    return true;
}

bool is_positive_int(const std::string& s) {
    if (s.empty() || s[0] == '-')
        return false;
    for (char c : s)
        if (c < '0' || c > '9')
            return false;
    return true;
}

bool parse_args(int argc, char** argv, Options& opt) {
    std::vector<std::string> args(argv + 1, argv + argc);
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        const auto take_value = [&](const char* flag) -> bool {
            if (i + 1 >= args.size() || args[i + 1].empty()) {
                std::fprintf(stderr, "forge-inspect: missing value for %s\n", flag);
                return false;
            }
            return true;
        };
        if (a == "-h" || a == "--help") {
            opt.help = true;
            return true;
        }
        if (a == "-s" || a == "--summary") {
            opt.summary = true;
        } else if (a == "-a" || a == "--all") {
            opt.all = true;
        } else if (a == "-c" || a == "--chart") {
            opt.chart = true;
        } else if (a == "-m" || a == "--metadata") {
            opt.metadata = true;
        } else if (a == "-p" || a == "--pipeline") {
            opt.pipeline = true;
        } else if (a == "-t" || a == "--tree") {
            opt.tree = true;
            if (i + 1 < args.size() && is_positive_int(args[i + 1])) {
                if (!parse_int_arg(args[i + 1].c_str(), &opt.tree_depth)) {
                    std::fprintf(stderr, "forge-inspect: --tree expects an integer, got '%s'\n",
                                 args[i + 1].c_str());
                    return false;
                }
                ++i;
            }
        } else if (a.rfind("--tree=", 0) == 0) {
            opt.tree = true;
            if (!parse_int_arg(a.c_str() + 7, &opt.tree_depth)) {
                std::fprintf(stderr, "forge-inspect: --tree expects an integer\n");
                return false;
            }
        } else if (a == "--top" || a == "--layers" || a == "--tree-limit" || a == "--ctx" ||
                   a == "--vram") {
            if (!take_value(a.c_str()))
                return false;
            int v = 0;
            if (!parse_int_arg(args[++i].c_str(), &v) || v <= 0) {
                std::fprintf(stderr, "forge-inspect: %s expects a positive integer, got '%s'\n",
                             a.c_str(), args[i].c_str());
                return false;
            }
            if (a == "--top")
                opt.peaks = v;
            else if (a == "--layers")
                opt.top_layers = v;
            else if (a == "--ctx")
                opt.ctx = v;
            else if (a == "--vram")
                opt.vram = v;
            else
                opt.tree_limit = v;
        } else if (a.rfind("--top=", 0) == 0) {
            int v = 0;
            if (!parse_int_arg(a.c_str() + 6, &v) || v <= 0) {
                std::fprintf(stderr, "forge-inspect: --top expects a positive integer\n");
                return false;
            }
            opt.peaks = v;
        } else if (a.rfind("--layers=", 0) == 0) {
            int v = 0;
            if (!parse_int_arg(a.c_str() + 9, &v) || v <= 0) {
                std::fprintf(stderr, "forge-inspect: --layers expects a positive integer\n");
                return false;
            }
            opt.top_layers = v;
        } else if (a.rfind("--ctx=", 0) == 0) {
            int v = 0;
            if (!parse_int_arg(a.c_str() + 6, &v) || v <= 0) {
                std::fprintf(stderr, "forge-inspect: --ctx expects a positive integer\n");
                return false;
            }
            opt.ctx = v;
        } else if (a.rfind("--vram=", 0) == 0) {
            int v = 0;
            if (!parse_int_arg(a.c_str() + 7, &v) || v <= 0) {
                std::fprintf(stderr, "forge-inspect: --vram expects a positive integer\n");
                return false;
            }
            opt.vram = v;
        } else if (a == "--filter") {
            if (!take_value(a.c_str()))
                return false;
            opt.filter = args[++i];
        } else if (a.rfind("--filter=", 0) == 0) {
            opt.filter = a.substr(9);
            if (opt.filter.empty()) {
                std::fprintf(stderr, "forge-inspect: --filter needs a non-empty substring\n");
                return false;
            }
        } else if (a == "--tensor") {
            if (!take_value(a.c_str()))
                return false;
            opt.tensor = args[++i];
        } else if (a.rfind("--tensor=", 0) == 0) {
            opt.tensor = a.substr(9);
            if (opt.tensor.empty()) {
                std::fprintf(stderr, "forge-inspect: --tensor needs a tensor name\n");
                return false;
            }
        } else if (a == "--tensor-diff" || a == "-d") {
            // Needs a second file to be meaningful; the check happens once the
            // models are loaded, where we know whether there is one.
            opt.tensor_diff = true;
        } else if (a == "--validate" || a == "-V") {
            opt.validate = true;
        } else if (a == "-M" || a == "--memory") {
            opt.memory = true;
        } else if (a == "--kv-dtype" || a.rfind("--kv-dtype=", 0) == 0) {
            std::string v;
            if (a == "--kv-dtype") {
                if (!take_value(a.c_str()))
                    return false;
                v = args[++i];
            } else {
                v = a.substr(11);
            }
            std::string canon;
            if (!forge::inspect::parse_kv_dtype(v, &canon)) {
                std::fprintf(stderr,
                             "forge-inspect: unknown --kv-dtype '%s' (want "
                             "f16|fp32|q8_0|q4_0|q4_k|fp8_e4m3|fp8_e5m2)\n",
                             v.c_str());
                return false;
            }
            opt.kv_dtype = canon;
            opt.memory = true;
        } else if (a == "-f" || a == "--format") {
            if (!take_value(a.c_str()))
                return false;
            if (!forge::inspect::format_from_name(args[++i], opt.format)) {
                std::fprintf(stderr, "forge-inspect: unknown format '%s'\n",
                             args[i].c_str());
                return false;
            }
        } else if (a.rfind("--format=", 0) == 0) {
            if (!forge::inspect::format_from_name(a.substr(9), opt.format)) {
                std::fprintf(stderr, "forge-inspect: unknown format '%s'\n",
                             a.substr(9).c_str());
                return false;
            }
        } else if (a == "-j" || a == "--threads") {
            if (!take_value(a.c_str()))
                return false;
            if (!parse_int_arg(args[++i].c_str(), &opt.threads) || opt.threads <= 0) {
                std::fprintf(stderr, "forge-inspect: --threads expects a positive integer, "
                                     "got '%s'\n",
                             args[i].c_str());
                return false;
            }
        } else if (a == "-o" || a == "--out") {
            if (!take_value(a.c_str()))
                return false;
            opt.out_file = args[++i];
        } else if (a == "--no-color") {
            opt.color = Mode::Never;
        } else if (a.rfind("--color=", 0) == 0) {
            const std::string v = a.substr(8);
            if (v == "auto")
                opt.color = Mode::Auto;
            else if (v == "always")
                opt.color = Mode::Always;
            else if (v == "never")
                opt.color = Mode::Never;
            else {
                // Silently exiting 2 with no message is indistinguishable from a
                // crash or a bad file.
                std::fprintf(stderr,
                             "forge-inspect: unknown --color mode '%s' (want auto|always|never)\n",
                             v.c_str());
                return false;
            }
        } else if (!a.empty() && a[0] == '-') {
            std::fprintf(stderr, "forge-inspect: unknown option '%s'\n", a.c_str());
            return false;
        } else if (opt.path.empty()) {
            opt.path = a;
        } else {
            opt.others.push_back(a);
        }
    }
    if (opt.path.empty() && !opt.help)
        return false;
    if (opt.threads < 1)
        opt.threads = 0;
    return true;
}

int write_output(const std::string& content, const std::string& out_file) {
    if (out_file.empty()) {
        std::fwrite(content.data(), 1, content.size(), stdout);
        return 0;
    }
    FILE* f = std::fopen(out_file.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "forge-inspect: cannot write %s\n", out_file.c_str());
        return 1;
    }
    std::fwrite(content.data(), 1, content.size(), f);
    std::fclose(f);
    return 0;
}

bool is_plain(forge::inspect::Format f) {
    return f == forge::inspect::Format::Json || f == forge::inspect::Format::Yaml ||
           f == forge::inspect::Format::Markdown || f == forge::inspect::Format::Csv ||
           f == forge::inspect::Format::Ini;
}

}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    // The 48000 consoles need UTF-8 code pages for box-drawing/metadata output.
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    Options opt;
    if (!parse_args(argc, argv, opt)) {
        print_help(argv[0]);
        return 2;
    }
    if (opt.help) {
        print_help(argv[0]);
        return 0;
    }

    // Color policy: machine formats are always plain; files default to plain.
    if (is_plain(opt.format) || opt.format == forge::inspect::Format::Markdown) {
        forge::inspect::col::set_mode(Mode::Never);
    } else {
        forge::inspect::col::set_mode(opt.color);
        if (!opt.out_file.empty() && opt.color == Mode::Auto)
            forge::inspect::col::set_mode(Mode::Never);
    }

    // json and yaml can hold several models, so they are loaded and rendered
    // together rather than rejected. The rest really are one-record formats.
    const bool multi_ok = opt.format == forge::inspect::Format::Json ||
                          opt.format == forge::inspect::Format::Yaml;
    const bool multi_ok_with_others = multi_ok && !opt.others.empty();
    if (is_plain(opt.format) && !opt.others.empty() && !multi_ok) {
        std::fprintf(stderr,
                     "forge-inspect: %s output supports a single model; use "
                     "format text/table to compare.\n",
                     forge::inspect::format_name(opt.format));
        return 2;
    }

    int threads = opt.threads > 0
                      ? opt.threads
                      : std::max(1, static_cast<int>(std::thread::hardware_concurrency()));

    using forge::inspect::ArchInfo;
    using forge::inspect::GgufSnapshot;
    using forge::inspect::LoadedModel;
    using forge::inspect::TensorStats;

    auto t0 = std::chrono::steady_clock::now();
    GgufSnapshot snap = forge::inspect::load_gguf(opt.path, threads);
    auto t1 = std::chrono::steady_clock::now();
    if (!snap.ok) {
        std::fprintf(stderr, "forge-inspect: %s\n", snap.error.c_str());
        return 1;
    }
    TensorStats stats = forge::inspect::compute_stats(snap, threads);
    auto t2 = std::chrono::steady_clock::now();
    ArchInfo arch = forge::inspect::sniff_arch(snap);

    const double parse_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double stats_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    LoadedModel primary{opt.path, std::move(snap), std::move(stats), std::move(arch),
                        parse_ms, stats_ms, threads};

    // ------ machine-readable single-model formats ------
    // json/yaml with several paths render everything together further down.
    if (is_plain(opt.format) && !multi_ok_with_others) {
        forge::inspect::RenderOptions ro;
        ro.top_layers = opt.top_layers;
        ro.memory = opt.memory;
        ro.validate = opt.validate;
        ro.ctx = opt.ctx;
        ro.vram_mb = opt.vram;
        ro.kv_dtype = opt.kv_dtype;
        ro.peaks = opt.peaks;
        std::string out;
        switch (opt.format) {
            case forge::inspect::Format::Json:
                out = forge::inspect::render_json(primary, ro);
                break;
            case forge::inspect::Format::Yaml:
                out = forge::inspect::render_yaml(primary, ro);
                break;
            case forge::inspect::Format::Markdown:
                out = forge::inspect::render_markdown(primary, ro);
                break;
            case forge::inspect::Format::Csv:
                out = forge::inspect::render_csv(primary, ro);
                break;
            case forge::inspect::Format::Ini:
                out = forge::inspect::render_ini(primary, ro);
                break;
            default:
                break;
        }
        return write_output(out, opt.out_file);
    }

    if (opt.tensor_diff && opt.others.empty()) {
        std::fprintf(stderr,
                     "forge-inspect: --tensor-diff needs a second model to compare against\n");
        return 1;
    }

    // ------ human formats: text or table, optional diff ------
    // The extra models are parsed concurrently. They are independent files, and
    // load_gguf already saturates `threads` on its own, so running them one at a
    // time made the wall clock the sum of every parse.
    std::vector<LoadedModel> models;
    models.reserve(opt.others.size() + 1);
    models.push_back(std::move(primary));
    {
        const size_t n = opt.others.size();
        std::vector<GgufSnapshot> snaps(n);
        std::vector<TensorStats> stats(n);
        std::vector<std::string> errors(n);
        std::vector<double> parse_ms(n, 0.0), stats_ms(n, 0.0);
        std::atomic<size_t> next{0};
        const size_t workers = std::min<size_t>(n, static_cast<size_t>(std::max(1, threads)));
        auto worker = [&]() {
            for (;;) {
                const size_t i = next.fetch_add(1);
                if (i >= n) return;
                const auto t0 = std::chrono::steady_clock::now();
                snaps[i] = forge::inspect::load_gguf(opt.others[i], threads);
                const auto t1 = std::chrono::steady_clock::now();
                if (!snaps[i].ok) {
                    errors[i] = snaps[i].error;
                    continue;
                }
                stats[i] = forge::inspect::compute_stats(snaps[i], threads);
                const auto t2 = std::chrono::steady_clock::now();
                parse_ms[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
                stats_ms[i] = std::chrono::duration<double, std::milli>(t2 - t1).count();
            }
        };
        if (workers <= 1) {
            worker();
        } else {
            std::vector<std::thread> pool;
            pool.reserve(workers);
            for (size_t k = 0; k < workers; ++k)
                pool.emplace_back(worker);
            for (auto& th : pool)
                th.join();
        }
        for (size_t i = 0; i < n; ++i) {
            if (!errors[i].empty()) {
                std::fprintf(stderr, "forge-inspect: %s: %s\n", opt.others[i].c_str(),
                             errors[i].c_str());
                return 1;
            }
            LoadedModel lm;
            lm.path = opt.others[i];
            lm.arch = forge::inspect::sniff_arch(snaps[i]);
            lm.snap = std::move(snaps[i]);
            lm.stats = std::move(stats[i]);
            lm.parse_ms = parse_ms[i];
            lm.stats_ms = stats_ms[i];
            lm.threads = threads;
            models.push_back(std::move(lm));
        }
    }
    if (opt.format == forge::inspect::Format::Json || opt.format == forge::inspect::Format::Yaml) {
        forge::inspect::RenderOptions mo;
        mo.memory = opt.memory;
        mo.ctx = opt.ctx;
        mo.vram_mb = opt.vram;
        mo.kv_dtype = opt.kv_dtype;
        mo.validate = opt.validate;
        mo.peaks = opt.peaks;
        mo.top_layers = opt.top_layers;
        const std::string mout =
            opt.format == forge::inspect::Format::Json
                ? forge::inspect::render_multi_json(models, mo)
                : forge::inspect::render_multi_yaml(models, mo);
        // --validate still gates the exit code, same as the human path.
        if (opt.validate) {
            for (const auto& lm : models) {
                for (const auto& f : forge::inspect::validate_model(lm.snap, lm.arch)) {
                    if (f.severity == forge::inspect::Severity::Error) {
                        write_output(mout, opt.out_file);
                        return 2;
                    }
                }
            }
        }
        return write_output(mout, opt.out_file);
    }

    std::vector<const LoadedModel*> others;
    for (size_t i = 1; i < models.size(); ++i)
        others.push_back(&models[i]);

    LoadedModel& m = models.front();

    using forge::inspect::col::paint;
    using forge::inspect::col::Style;

    bool want_summary = opt.summary || opt.all ||
                        (!opt.tree && !opt.chart && !opt.metadata && !opt.pipeline);

    std::string out;
    out += forge::inspect::render_header_line(m.path, m.snap);

    const bool table = opt.format == forge::inspect::Format::Table;

    char tbuf[64];
    std::snprintf(tbuf, sizeof(tbuf), "%.1f", m.parse_ms);
    std::string parse_s = tbuf;
    std::snprintf(tbuf, sizeof(tbuf), "%.1f", m.stats_ms);
    std::string stats_s = tbuf;
    out += "  " + paint("parsed in " + parse_s + " ms (" + std::to_string(threads) +
                            " threads) + " + stats_s + " ms stats",
                        Style::Dim) +
           "\n";

    if (want_summary) {
        out += table ? forge::inspect::render_summary_table(m)
                     : forge::inspect::render_summary(m.snap, m.arch, m.stats);
    }
    if (!opt.tensor.empty()) {
        bool found = false;
        const std::string detail = forge::inspect::render_tensor_detail(m.snap, opt.tensor, &found);
        if (!found) {
            std::fprintf(stderr, "forge-inspect: no tensor named '%s'\n", opt.tensor.c_str());
            return 1;
        }
        write_output(detail, opt.out_file);
        return 0;
    }
    if (opt.validate) {
        out += "\n  " + paint("validation", Style::BoldCyan) + "\n";
        const auto findings = forge::inspect::validate_model(m.snap, m.arch);
        out += forge::inspect::render_validation(findings);
        // Exit non-zero on errors so this can gate a CI step. --validate is a
        // question about the file, not a request to render, so a bad model is a
        // failure even though the report itself succeeded.
        for (const auto& f : findings) {
            if (f.severity == forge::inspect::Severity::Error) {
                write_output(out, opt.out_file);
                return 2;
            }
        }
    }
    if (opt.memory) {
        // --vram is in MiB; the rest of the tool is byte-based.
        const int64_t budget = opt.vram > 0 ? opt.vram * 1024 * 1024 : -1;
        const auto me = forge::inspect::estimate_memory(m.snap, m.arch, opt.ctx, opt.kv_dtype,
                                                        budget);
        out += "\n  " + paint("memory estimate", Style::BoldCyan) + "\n";
        out += forge::inspect::render_memory(m.snap, m.arch, me);
    }
    if (opt.top_layers) {
        out += "\n  " + paint("layer sizes (top " + std::to_string(opt.top_layers) + ")",
                              Style::BoldCyan) + "\n";
        out += forge::inspect::render_layer_table(m.stats, opt.top_layers);
    }
    if (opt.peaks || !opt.filter.empty()) {
        const int limit = opt.peaks > 0 ? opt.peaks : 20;
        out += "\n  " + paint(opt.filter.empty()
                                   ? "largest tensors (top " + std::to_string(limit) + ")"
                                   : "tensors matching '" + opt.filter + "' (top " +
                                         std::to_string(limit) + ")",
                              Style::BoldCyan) + "\n";
        int matched = 0;
        out += forge::inspect::render_peak_tensors(m.snap, limit, opt.filter, &matched);
        if (!opt.filter.empty())
            out += "  " + paint(std::to_string(matched) + " of " +
                                    std::to_string(m.snap.tensors.size()) + " tensors match",
                                Style::Dim) +
                   "\n";
    }
    if (opt.tree || opt.all) {
        out += "\n  " + paint("tensor tree (depth " + std::to_string(opt.tree_depth) + ")",
                              Style::BoldCyan) + "\n";
        out += forge::inspect::render_tree(m.snap, m.stats, opt.tree_depth);
    }
    if (opt.chart || opt.all) {
        out += "\n  " + paint("dtype sizes", Style::BoldCyan) + "\n";
        out += forge::inspect::render_chart(m.stats);
    }
    if (opt.pipeline || opt.all) {
        out += "\n  " + paint("operator pipeline (" + m.arch.arch + ")", Style::BoldCyan) +
               "\n";
        out += forge::inspect::render_op_pipeline(m.arch);
    }
    if (opt.metadata || opt.all) {
        out += "\n  " + paint("metadata", Style::BoldCyan) + "\n";
        out += forge::inspect::render_metadata(m.snap, m.arch.arch);
    }
    for (const LoadedModel* o : others) {
        out += "\n  " + paint("diff vs " + o->path, Style::BoldYellow) + "\n";
        out += table ? forge::inspect::render_diff_table(m, *o)
                     : forge::inspect::render_diff_text(m, *o);
        if (opt.tensor_diff) {
            out += "  " + paint("tensors", Style::BoldYellow) + "\n";
            out += forge::inspect::render_tensor_diff(
                forge::inspect::compute_tensor_diff(m.snap, o->snap),
                opt.top_layers > 0 ? opt.top_layers : 20, table);
        }
    }

    return write_output(out, opt.out_file);
}