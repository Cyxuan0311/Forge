// Shared argument layer: KV dtype parsing, the common flag parser, and the
// speculative flag parser.
//
// These exist because forge-cli and forge-server had two independent parsers
// that drifted (different --kv-cache-dtype defaults, a server -ngl that
// terminated on "auto", a typo'd dtype that silently became f16, and a server
// with no speculative decoding at all). The fix was one parse chain in
// forge_infer; this test is what keeps the two binaries honest, because both
// link the same functions and a regression in one shows up here.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "forge/cli_common_args.h"
#include "forge/cli_dashboard_args.h"
#include "forge/cli_spec_args.h"
#include "forge/kv_cache_dtype.h"

using namespace forge;

static int g_fail = 0;
static int g_checks = 0;

#define CHECK(cond, msg)                                                     \
    do {                                                                     \
        ++g_checks;                                                          \
        if (!(cond)) {                                                       \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, msg);        \
            ++g_fail;                                                        \
        } else {                                                             \
            std::printf("ok   %s\n", msg);                                   \
        }                                                                    \
    } while (0)

// Build an argv from a vector and hand it to a parser at index 1.
static std::vector<std::string> g_argv_storage;

static void set_args(std::vector<const char*> a) {
    g_argv_storage.clear();
    for (const char* s : a) g_argv_storage.push_back(s);
}

// ---------------------------------------------------------------------------
// KV cache dtype names / parsing
// ---------------------------------------------------------------------------
static void test_kv_dtype_parse() {
    std::printf("\n--- KV cache dtype ---\n");

    KVCacheDType dt;
    std::string err;

    // Every advertised name must parse, and the list must be exactly them.
    const char* all[] = {"fp32", "f16", "q8_0", "q4_0", "q4_k", "fp8_e4m3", "fp8_e5m2"};
    for (const char* n : all) {
        err.clear();
        CHECK(parse_kv_cache_dtype(n, &dt, &err), n);
    }
    std::string listed = kv_cache_dtype_allowed_values();
    CHECK(listed == "fp32, f16, q8_0, q4_0, q4_k, fp8_e4m3, fp8_e5m2",
          "allowed values match the implemented set");

    // llama.cpp accepts bf16/q4_1/q5_0/q5_1/iq4_nl. Forge has no dequantiser
    // for them, so they must be a hard error naming the value -- never a silent
    // downgrade to f16, which is exactly the bug the server had.
    for (const char* n : {"bf16", "q4_1", "q5_0", "q5_1", "iq4_nl", "Q8_0", "q8", "fp16"}) {
        dt = KVCacheDType::F16;
        err.clear();
        CHECK(!parse_kv_cache_dtype(n, &dt, &err), n);
        CHECK(err.find(n) != std::string::npos, "  error names the bad value");
    }

    // Block sizes drive the divisibility check.
    CHECK(kv_cache_dtype_block_size(KVCacheDType::Q8_0) == 32, "q8_0 block = 32");
    CHECK(kv_cache_dtype_block_size(KVCacheDType::Q4_0) == 32, "q4_0 block = 32");
    CHECK(kv_cache_dtype_block_size(KVCacheDType::Q4_K) == 256, "q4_k block = 256");
    CHECK(kv_cache_dtype_block_size(KVCacheDType::F16) == 1, "f16 has no block structure");
}

// ---------------------------------------------------------------------------
// The K/V pair resolution and its validation
// ---------------------------------------------------------------------------
static void test_kv_type_pair() {
    std::printf("\n--- K/V type pair ---\n");
    const int HD = 128;  // MiniCPM5-2B / most current models

    // No flags -> f16/f16.
    {
        forge::CommonArgs a;
        CHECK(a.kv_cache_types().type_k == KVCacheDType::F16 &&
                  a.kv_cache_types().type_v == KVCacheDType::F16,
              "default is f16/f16");
    }
    // --kv-cache-dtype sets both (the historical behaviour).
    {
        forge::CommonArgs a;
        a.kv_cache_type_k = (int)KVCacheDType::Q8_0;
        a.kv_cache_type_v = (int)KVCacheDType::Q8_0;
        CHECK(a.kv_cache_types().type_k == KVCacheDType::Q8_0 &&
                  a.kv_cache_types().type_v == KVCacheDType::Q8_0,
              "--kv-cache-dtype sets both");
    }
    // -ctk alone -> K quantised, V follows (back-compat with "one dtype").
    {
        forge::CommonArgs a;
        a.kv_cache_type_k = (int)KVCacheDType::Q8_0;
        CHECK(a.kv_cache_types().type_k == KVCacheDType::Q8_0 &&
                  a.kv_cache_types().type_v == KVCacheDType::Q8_0,
              "-ctk alone applies to both");
    }
    // -ctk and -ctv together -> the asymmetric pair, which is the whole point.
    {
        forge::CommonArgs a;
        a.kv_cache_type_k = (int)KVCacheDType::Q8_0;
        a.kv_cache_type_v = (int)KVCacheDType::Q4_0;
        CHECK(a.kv_cache_types().type_k == KVCacheDType::Q8_0 &&
                  a.kv_cache_types().type_v == KVCacheDType::Q4_0,
              "-ctk + -ctv give an asymmetric pair");
        // And it must cost less than either symmetric option.
        const size_t both = kv_cache_dtype_block_nbytes(KVCacheDType::Q8_0, 256) +
                            kv_cache_dtype_block_nbytes(KVCacheDType::Q4_0, 256);
        const size_t k_q = kv_cache_dtype_block_nbytes(KVCacheDType::Q8_0, 256) * 2;
        const size_t q_q = kv_cache_dtype_block_nbytes(KVCacheDType::Q4_0, 256) * 2;
        CHECK(both < k_q, "q8_0/q4_0 smaller than q8_0/q8_0");
        // It is deliberately LARGER than q4_0/q4_0: that extra K precision is
        // exactly what is being bought, and the quality data says it is the
        // right side to spend it on.
        CHECK(both > q_q, "q8_0/q4_0 is larger than q4_0/q4_0 (bought K precision)");
    }

    // Validation: head dim must be a multiple of the block size. block_nbytes
    // PADS a short row, so a non-divisible head dim would silently mis-stride
    // the fused kernels -- this check is the only thing preventing that.
    {
        KVCacheTypeConfig c{KVCacheDType::Q8_0, KVCacheDType::Q8_0};
        CHECK(validate_kv_cache_types(c, HD, HD, "llama").ok, "q8_0 with head_dim 128 is fine");
        CHECK(!validate_kv_cache_types(c, 100, 100, "llama").ok, "q8_0 with head_dim 100 is rejected");
        CHECK(validate_kv_cache_types(c, 96, 96, "llama").ok,
              "  head_dim 96 is a multiple of 32, so q8_0 is fine there");
        // q4_k's block is 256, so a 128 head dim cannot use it.
        KVCacheTypeConfig k{KVCacheDType::Q4_K, KVCacheDType::Q4_K};
        CHECK(!validate_kv_cache_types(k, HD, HD, "llama").ok, "q4_k rejected at head_dim 128");
    }
    // MLA-style archs store V inside K, so the pair must match.
    {
        KVCacheTypeConfig c{KVCacheDType::Q8_0, KVCacheDType::Q4_0};
        CHECK(!validate_kv_cache_types(c, HD, HD, "deepseek3").ok, "MLA rejects K != V");
        CHECK(validate_kv_cache_types(c, HD, HD, "llama").ok, "llama allows K != V");
    }
    // Policy warning when K is quantised and V is not: K is the sensitive side.
    {
        KVCacheTypeConfig c{KVCacheDType::Q4_0, KVCacheDType::F16};
        const auto v = validate_kv_cache_types(c, HD, HD, "llama");
        CHECK(v.ok && !v.warning.empty(), "quantised K / f16 V warns");
    }
}

// ---------------------------------------------------------------------------
// The shared flag parser
// ---------------------------------------------------------------------------
// Walk a whole command line the way both binaries do. Returns the first Error
// seen, otherwise NotMine -- it must NOT stop at the first Ok, or a two-flag
// command line like "-ctk q8_0 -ctv q4_0" would only ever parse its first flag
// (which is how this helper hid a real coverage gap when first written).
static forge::ParseResult parse_common(std::vector<const char*> argv, forge::CommonArgs& out,
                                       std::string* err) {
    set_args(argv);
    std::vector<char*> cargv;
    for (auto& s : g_argv_storage) cargv.push_back(const_cast<char*>(s.c_str()));
    // int index, not size_t: the parser takes int& and advances it when a flag
    // consumes a value, so a size_t& here is a strict-aliasing violation and
    // would also corrupt the loop index.
    forge::ParseResult result = forge::ParseResult::NotMine;
    for (int i = 1; i < (int)cargv.size(); ++i) {
        const forge::ParseResult r =
            forge::parse_common_arg(cargv[i], (int)cargv.size(), cargv.data(), i, out, err);
        if (r == forge::ParseResult::Error) return r;
    }
    return result;
}

static void test_common_parser() {
    std::printf("\n--- shared flag parser ---\n");
    forge::CommonArgs a;
    std::string err;

    // -ngl accepted every form the CLI documented. The server used a bare
    // std::stoi, so "-ngl auto" threw out of main() and terminated.
    struct { const char* in; int want; } ngl[] = {
        {"auto", -1}, {"-1", -1}, {"all", -2}, {"-2", -2}, {"42", 42}, {"0", 0},
    };
    for (const auto& c : ngl) {
        a = forge::CommonArgs();
        parse_common({"forge", "-ngl", c.in}, a, &err);
        char msg[96];
        std::snprintf(msg, sizeof(msg), "-ngl %s -> %d", c.in, c.want);
        CHECK(a.n_gpu_layers == c.want, msg);
    }
    // multi-GPU split
    {
        a = forge::CommonArgs();
        parse_common({"forge", "-ngl", "16:16"}, a, &err);
        CHECK(a.gpu_layers_per_dev.size() == 2 && a.gpu_layers_per_dev[0] == 16 &&
                  a.gpu_layers_per_dev[1] == 16,
              "-ngl 16:16 splits across two devices");
        a = forge::CommonArgs();
        parse_common({"forge", "-ngl", "8:4:4"}, a, &err);
        CHECK(a.gpu_layers_per_dev.size() == 3, "-ngl 8:4:4 splits three ways");
    }
    // a bad -ngl is an error, not a crash
    {
        a = forge::CommonArgs();
        err.clear();
        CHECK(parse_common({"forge", "-ngl", "abc"}, a, &err) == forge::ParseResult::Error,
              "-ngl abc is an error");
        CHECK(!err.empty(), "  with a message");
        a = forge::CommonArgs();
        err.clear();
        CHECK(parse_common({"forge", "-ngl", "16:"}, a, &err) == forge::ParseResult::Error,
              "-ngl 16: (empty slot) is an error");
    }

    // -ctk / -ctv
    {
        a = forge::CommonArgs();
        parse_common({"forge", "-ctk", "q8_0", "-ctv", "q4_0"}, a, &err);
        CHECK(a.kv_cache_types().type_k == KVCacheDType::Q8_0 &&
                  a.kv_cache_types().type_v == KVCacheDType::Q4_0,
              "-ctk/-ctv parse");
        a = forge::CommonArgs();
        err.clear();
        CHECK(parse_common({"forge", "-ctk", "bf16"}, a, &err) == forge::ParseResult::Error,
              "-ctk bf16 is a hard error (llama.cpp accepts it, Forge cannot)");
    }

    // load-mode validation
    {
        for (const char* m : {"mmap", "mlock", "mmap_mlock"}) {
            a = forge::CommonArgs();
            parse_common({"forge", "--load-mode", m}, a, &err);
            CHECK(a.load_mode == m, "--load-mode accepts the valid mode");
        }
        a = forge::CommonArgs();
        err.clear();
        CHECK(parse_common({"forge", "--load-mode", "mmapp"}, a, &err) == forge::ParseResult::Error,
              "--load-mode typo is an error (server used to accept it silently)");
    }

    // kvmem validation lives in the shared layer now
    {
        a = forge::CommonArgs();
        a.kvmem = true;
        a.kvmem_sink = 512;
        a.kvmem_budget = 128;
        err.clear();
        CHECK(!forge::validate_common_args(a, &err), "sink > budget rejected");
        a.kvmem_sink = 64;
        err.clear();
        CHECK(forge::validate_common_args(a, &err), "sink <= budget accepted");
    }
    // generation budget drives the reserve fallback (256 for the CLI, 512 for
    // the server -- the two kvem_config() copies had drifted on this).
    {
        forge::CommonArgs c;
        c.generation_budget = 256;
        CHECK(c.kvem_config().gen_reserve == 256, "CLI reserve fallback = -n default");
        forge::CommonArgs sv;
        sv.generation_budget = 512;
        CHECK(sv.kvem_config().gen_reserve == 512, "server reserve fallback = request default");
    }

    // An unknown flag must be NotMine so the caller's own chain still sees it.
    {
        a = forge::CommonArgs();
        CHECK(parse_common({"forge", "--spec"}, a, &err) == forge::ParseResult::NotMine,
              "unknown flag is NotMine (caller's chain handles it)");
    }
}

// ---------------------------------------------------------------------------
// The speculative flag parser (shared with the server)
// ---------------------------------------------------------------------------
static void test_spec_parser() {
    std::printf("\n--- speculative flag parser ---\n");
    forge::SpeculativeConfig spec;
    std::string err;
    int i;

    auto run = [&](std::vector<const char*> argv) {
        set_args(argv);
        std::vector<char*> cargv;
        for (auto& s : g_argv_storage) cargv.push_back(const_cast<char*>(s.c_str()));
        for (int k = 1; k < (int)cargv.size(); ++k) {
            i = k;
            const forge::SpecParseResult r =
                forge::parse_spec_arg(cargv[i], (int)cargv.size(), cargv.data(), i, spec, &err);
            if (r == forge::SpecParseResult::Ok) return r;
            if (r == forge::SpecParseResult::Error) return r;
        }
        return forge::SpecParseResult::NotMine;
    };

    spec = forge::SpeculativeConfig();
    CHECK(run({"forge", "--spec"}) == forge::SpecParseResult::Ok && spec.enabled &&
              spec.use_ngram,
          "--spec enables n-gram drafting");
    spec = forge::SpeculativeConfig();
    CHECK(run({"forge", "--spec-draft", "7"}) == forge::SpecParseResult::Ok && spec.n_draft == 7,
          "--spec-draft sets n_draft");
    spec = forge::SpeculativeConfig();
    CHECK(run({"forge", "--spec-draft-arch", "dspark"}) == forge::SpecParseResult::Ok &&
              spec.draft_arch == "dspark",
          "--spec-draft-arch dspark");
    spec = forge::SpeculativeConfig();
    err.clear();
    CHECK(run({"forge", "--spec-draft-arch", "bogus"}) == forge::SpecParseResult::Error &&
              !err.empty(),
          "--spec-draft-arch bogus is an error");
    spec = forge::SpeculativeConfig();
    CHECK(run({"forge", "--spec-draft-target-layers", "2,5,8"}) ==
                  forge::SpecParseResult::Ok &&
              spec.draft_target_layers.size() == 3 &&
              spec.draft_target_layers[2] == 8,
          "--spec-draft-target-layers parses a CSV");
    spec = forge::SpeculativeConfig();
    err.clear();
    CHECK(run({"forge", "--spec-draft-target-layers", "2,x,8"}) == forge::SpecParseResult::Error,
          "--spec-draft-target-layers rejects a non-numeric entry");
    spec = forge::SpeculativeConfig();
    CHECK(run({"forge", "--no-adaptive-draft"}) == forge::SpecParseResult::Ok &&
              !spec.adaptive_draft,
          "--no-adaptive-draft clears adaptive sizing");
    spec = forge::SpeculativeConfig();
    CHECK(run({"forge", "--spec-min-acc", "0.4"}) == forge::SpecParseResult::Ok &&
              spec.min_accept_rate > 0.39f,
          "--spec-min-acc sets the floor");
    spec = forge::SpeculativeConfig();
    CHECK(run({"forge", "--spec-stats"}) == forge::SpecParseResult::Ok && spec.print_stats,
          "--spec-stats sets print_stats");
    spec = forge::SpeculativeConfig();
    CHECK(run({"forge", "--temp", "0.5"}) == forge::SpecParseResult::NotMine,
          "non-spec flag is NotMine");

    // Missing value is an error, not a read past the end of argv.
    spec = forge::SpeculativeConfig();
    err.clear();
    CHECK(run({"forge", "--spec-draft"}) == forge::SpecParseResult::Error && !err.empty(),
          "--spec-draft with no value is an error");
}

// ---------------------------------------------------------------------------
// Web dashboard parser
//
// Shared by forge-cli and forge-server for the same reason the rest of this
// file exists: two independent flag parsers drift. --dashboard-port in
// particular had a real failure mode worth pinning -- atoi() returns 0 for a
// typo, and 0 here means "let the OS pick a port", so "--dashboard-port abc"
// would have silently worked.
// ---------------------------------------------------------------------------
static forge::DashboardParseResult run_dash(const std::vector<const char*>& a,
                                            forge::DashboardArgs& d, std::string& err) {
    set_args(a);
    // Mirror of parse_common()'s helper: the parser takes (int argc, char** argv,
    // int& i) and advances i past a flag's value, so argv has to be a real
    // char** and the loop index has to be an int.
    std::vector<char*> cargv;
    for (auto& str : g_argv_storage) cargv.push_back(const_cast<char*>(str.c_str()));
    // Keep scanning after a hit, exactly as the callers in forge-cli and
    // forge-server do. Bailing out on the first match would make combinations
    // like "--dashboard --no-open" untestable and would not reflect real argv
    // handling.
    forge::DashboardParseResult result = forge::DashboardParseResult::NotMine;
    for (int i = 1; i < static_cast<int>(cargv.size()); ++i) {
        const forge::DashboardParseResult r = forge::parse_dashboard_arg(
            cargv[i], static_cast<int>(cargv.size()), cargv.data(), i, d, &err);
        if (r == forge::DashboardParseResult::Error) return r;
        if (r == forge::DashboardParseResult::Ok) result = r;
    }
    return result;
}

static void test_dashboard_parser() {
    std::printf("\n--- dashboard parser ---\n");
    forge::DashboardArgs d;
    std::string err;

    // Defaults: off, loopback, 9090, browser allowed.
    d = forge::DashboardArgs();
    CHECK(!d.enabled, "dashboard is off by default");
    CHECK(d.host == "127.0.0.1", "dashboard binds loopback by default");
    CHECK(d.port == 9090, "dashboard port defaults to 9090");
    CHECK(d.open_browser, "the browser is launched by default");

    d = forge::DashboardArgs();
    CHECK(run_dash({"forge", "--dashboard"}, d, err) == forge::DashboardParseResult::Ok &&
              d.enabled,
          "--dashboard enables the dashboard");
    d = forge::DashboardArgs();
    CHECK(run_dash({"forge", "--no-open"}, d, err) == forge::DashboardParseResult::Ok &&
              !d.open_browser,
          "--no-open suppresses the browser");
    d = forge::DashboardArgs();
    CHECK(run_dash({"forge", "--dashboard", "--no-open"}, d, err) ==
              forge::DashboardParseResult::Ok &&
              d.enabled && !d.open_browser,
          "--dashboard and --no-open combine");
    d = forge::DashboardArgs();
    CHECK(!d.capture_operators, "the operator panel is opt-in by default");
    d = forge::DashboardArgs();
    CHECK(run_dash({"forge", "--dashboard-ops"}, d, err) == forge::DashboardParseResult::Ok &&
              d.capture_operators && d.enabled,
          "--dashboard-ops enables the panel and implies --dashboard");
    d = forge::DashboardArgs();
    CHECK(run_dash({"forge", "--dashboard-host", "0.0.0.0"}, d, err) ==
              forge::DashboardParseResult::Ok &&
              d.host == "0.0.0.0",
          "--dashboard-host sets the bind address");
    d = forge::DashboardArgs();
    CHECK(run_dash({"forge", "--dashboard-port", "9123"}, d, err) ==
              forge::DashboardParseResult::Ok &&
              d.port == 9123 && d.enabled,
          "--dashboard-port sets the port and implies --dashboard");
    d = forge::DashboardArgs();
    CHECK(run_dash({"forge", "--dashboard-port", "0"}, d, err) == forge::DashboardParseResult::Ok &&
              d.port == 0,
          "--dashboard-port 0 means \"OS picks\"");

    // A typo must be an error, not a silent 0 (= "OS picks a port").
    d = forge::DashboardArgs();
    err.clear();
    CHECK(run_dash({"forge", "--dashboard-port", "abc"}, d, err) ==
              forge::DashboardParseResult::Error &&
              !err.empty(),
          "--dashboard-port abc is an error, not port 0");
    d = forge::DashboardArgs();
    err.clear();
    CHECK(run_dash({"forge", "--dashboard-port", "80x"}, d, err) ==
              forge::DashboardParseResult::Error,
          "--dashboard-port with trailing garbage is an error");
    d = forge::DashboardArgs();
    err.clear();
    CHECK(run_dash({"forge", "--dashboard-port", "70000"}, d, err) ==
              forge::DashboardParseResult::Error,
          "out-of-range port is an error");

    // Missing value must not read past the end of argv.
    d = forge::DashboardArgs();
    err.clear();
    CHECK(run_dash({"forge", "--dashboard-port"}, d, err) == forge::DashboardParseResult::Error &&
              !err.empty(),
          "--dashboard-port with no value is an error");

    d = forge::DashboardArgs();
    err.clear();
    CHECK(run_dash({"forge", "--dashboard-host"}, d, err) ==
              forge::DashboardParseResult::Error,
          "--dashboard-host with no value is an error");

    d = forge::DashboardArgs();
    err.clear();
    CHECK(run_dash({"forge", "--temp", "0.5"}, d, err) ==
              forge::DashboardParseResult::NotMine,
          "a non-dashboard flag is NotMine");
    d = forge::DashboardArgs();
    CHECK(run_dash({"forge", "--dashboard"}, d, err) == forge::DashboardParseResult::Ok &&
              is_dashboard_arg("--dashboard"),
          "is_dashboard_arg recognises the flag");
    CHECK(!is_dashboard_arg("--dashboardx"), "is_dashboard_arg is not a prefix match");
}

int main() {
    std::printf("=== forge-args-test ===\n");
    test_kv_dtype_parse();
    test_kv_type_pair();
    test_common_parser();
    test_spec_parser();
    test_dashboard_parser();
    std::printf("\n=== %d/%d checks passed ===\n", g_checks - g_fail, g_checks);
    return g_fail == 0 ? 0 : 1;
}
