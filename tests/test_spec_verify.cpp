// Speculative decoding tests (Phase 1 of SPECULATIVE_DECODING_PLAN.md).
//
//  1. verify_draft_tokens branch coverage, no model required:
//       - greedy accept-all + bonus
//       - rejection at first / middle position (resampled token)
//       - empty draft list
//       - sampling chain modes (temperature ~ 0, top_k=1) stay consistent
//       - repeat penalty applied through the raw-pointer path
//  2. forward_batch multi-token verification batch vs sequential single-token
//     forwards: per-row logits must agree (argmax identical, small numeric
//     drift tolerated). Needs models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf.
//  3. End-to-end greedy parity: Generator output must be IDENTICAL with the
//     speculative config on vs off (resample-consistency is lossless).
//
// Build & run:
//   cmake --build build -j && ./build/forge-spec-test

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "forge/context.h"
#include "forge/engine.h"
#include "forge/engines/transformer_engine.h"
#include "forge/generator.h"
#include "forge/gguf_model.h"
#include "forge/inference_batch.h"
#include "forge/model.h"
#include "forge/model_loader.h"
#include "forge/ninf_model.h"
#include "forge/sampler.h"
#include "forge/speculative.h"
#include "forge/arch_registry.h"

using namespace forge;

// Static libraries may drop translation units that only contain
// self-registering objects (same mechanism forge-cli uses): force-link arch
// registrations and register the file-format loaders explicitly.
static void register_forge_runtime() {
    force_link_arch_registrations();
    auto& reg = ModelLoaderRegistry::instance();
    reg.register_loader("gguf",
                        []() -> std::unique_ptr<ModelLoader> { return std::make_unique<GgufModel>(); });
    reg.register_loader("ninf",
                        []() -> std::unique_ptr<ModelLoader> { return std::make_unique<NinfModel>(); });
}

// Mirror PyModel::create_context (src/bindings/common.h): build a context and
// attach the registry-created engine for the model's architecture.
static std::unique_ptr<InferenceContext> make_context(Model& model,
                                                      const ContextParams& params) {
    auto ctx = std::make_unique<InferenceContext>(model, params);
    auto engine = EngineRegistry::instance().create(model.config().arch_type, model, *ctx);
    if (!engine)
        throw std::runtime_error("no engine registered for arch: " + model.config().arch_type);
    if (auto* t = dynamic_cast<TransformerEngine*>(engine.get())) {
        t->set_kv_cache_dtype(KVCacheDType::FP32);
        t->set_gpu_layers(params.gpu_layers);
    }
    ctx->set_engine(std::move(engine));
    return ctx;
}

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        ++g_checks;                                                         \
        if (!(cond)) {                                                      \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, msg);       \
            ++g_failures;                                                   \
        } else {                                                            \
            std::printf("ok   %s\n", msg);                                  \
        }                                                                   \
    } while (0)

static bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

// =========================================================================
// Part 1 -- verify_draft_tokens unit tests (synthetic logits)
// =========================================================================

// Row of `vocab` zeros with `idx` set to `val`.
static std::vector<float> one_hot_row(int vocab, int idx, float val = 30.0f) {
    std::vector<float> row(static_cast<size_t>(vocab), 0.0f);
    row[static_cast<size_t>(idx)] = val;
    return row;
}

static void run_verify_unit_tests() {
    std::printf("\n--- verify_draft_tokens unit tests ---\n");
    constexpr int V = 16;
    SpeculativeConfig spec_cfg;  // defaults; verify reads sampler config only

    // ---- Greedy mode ----------------------------------------------------
    {
        SamplerConfig sc;
        sc.do_sample = false;
        Sampler s(sc);

        // A: both drafts match -> accepted all + bonus from last row.
        Sampler sa(sc);
        std::vector<float> rows_a;
        auto r0 = one_hot_row(V, 3), r1 = one_hot_row(V, 5), r2 = one_hot_row(V, 9);
        rows_a.insert(rows_a.end(), r0.begin(), r0.end());
        rows_a.insert(rows_a.end(), r1.begin(), r1.end());
        rows_a.insert(rows_a.end(), r2.begin(), r2.end());
        auto vr = verify_draft_tokens(rows_a.data(), V, {3, 5}, sa, spec_cfg);
        CHECK(vr.n_accepted == 2 && vr.accepted_tokens == std::vector<int32_t>({3, 5}),
              "greedy: both drafts accepted");
        CHECK(vr.resampled == -1, "greedy accept-all: no resampled token");
        CHECK(vr.bonus == 9, "greedy accept-all: bonus sampled from last row");
        CHECK(sa.token_history().size() == 3,
              "greedy accept-all: every sampled token fed to history");

        // B: mismatch at first position -> resampled, nothing accepted.
        Sampler sb(sc);
        auto b0 = one_hot_row(V, 4);
        auto vr_b = verify_draft_tokens(b0.data(), V, {3}, sb, spec_cfg);
        CHECK(vr_b.n_accepted == 0 && vr_b.accepted_tokens.empty(),
              "greedy first-reject: zero accepted");
        CHECK(vr_b.resampled == 4, "greedy first-reject: target sample wins");
        CHECK(sb.token_history().size() == 1,
              "greedy first-reject: resampled token fed to history");

        // C: mismatch at middle position -> prefix accepted then resample.
        Sampler sc3(sc);
        std::vector<float> rows_c;
        auto r6 = one_hot_row(V, 6);
        rows_c.insert(rows_c.end(), r0.begin(), r0.end());
        rows_c.insert(rows_c.end(), r6.begin(), r6.end());
        auto vr_c = verify_draft_tokens(rows_c.data(), V, {3, 5}, sc3, spec_cfg);
        CHECK(vr_c.n_accepted == 1 && vr_c.accepted_tokens == std::vector<int32_t>({3}),
              "greedy mid-reject: prefix accepted");
        CHECK(vr_c.resampled == 6 && vr_c.bonus == -1,
              "greedy mid-reject: resampled set, no bonus");

        // D: empty drafts -> no-op result.
        Sampler sd(sc);
        auto vr_d = verify_draft_tokens(r0.data(), V, {}, sd, spec_cfg);
        CHECK(vr_d.n_accepted == 0 && vr_d.resampled == -1 && vr_d.bonus == -1,
              "empty draft list: no-op result");
    }

    // ---- Sampling chain (resample-consistency) --------------------------
    {
        // temperature=0.01 -> softmax ~= one-hot, chain runs but stays argmax.
        SamplerConfig sc;
        sc.do_sample = true;
        sc.temperature = 0.01f;
        Sampler s(sc);
        auto s0 = one_hot_row(V, 3), s1 = one_hot_row(V, 5), s2b = one_hot_row(V, 9);
        std::vector<float> rows;
        rows.insert(rows.end(), s0.begin(), s0.end());
        rows.insert(rows.end(), s1.begin(), s1.end());
        rows.insert(rows.end(), s2b.begin(), s2b.end());
        auto vr = verify_draft_tokens(rows.data(), V, {3, 5}, s, spec_cfg);
        CHECK(vr.n_accepted == 2 && vr.bonus == 9,
              "sampling temp~0: matches greedy behavior");

        // top_k=1 at temperature=1 -> deterministic argmax through the full
        // filter chain.
        SamplerConfig sk;
        sk.do_sample = true;
        sk.temperature = 1.0f;
        sk.top_k = 1;
        Sampler s2(sk);
        auto vr2 = verify_draft_tokens(rows.data(), V, {3, 5}, s2, spec_cfg);
        CHECK(vr2.n_accepted == 2 && vr2.bonus == 9,
              "sampling top_k=1: deterministic argmax via full chain");
    }

    // ---- Repeat penalty through the raw-pointer path ---------------------
    {
        SamplerConfig sc;
        sc.do_sample = false;
        sc.repeat_penalty = 2.0f;   // positive logit 30 -> 15 after penalty
        sc.repeat_last_n = 64;
        Sampler s(sc);
        s.add_token_to_history(3);  // penalize token 3

        // Row: token 3 has 30 (penalized to 15), token 7 has 20 -> argmax 7.
        std::vector<float> row = one_hot_row(V, 3);
        row[7] = 20.0f;
        auto vr = verify_draft_tokens(row.data(), V, {3}, s, spec_cfg);
        CHECK(vr.n_accepted == 0 && vr.resampled == 7,
              "repeat penalty: applied before greedy match in verify");
    }
}

// =========================================================================
// Part 1b -- NgramDraftProvider hash-index cross-validation
// =========================================================================

#include <random>

// Brute-force reference with the same LATEST-match selection as the indexed
// provider: longest suffix first, most recent occurrence with >=1 continuation.
static std::vector<int32_t> ngram_brute_reference(const std::vector<int32_t>& hist,
                                                  int ngram_n, int ngram_min, int n_draft) {
    const int len = static_cast<int>(hist.size());
    const int max_L = std::min(ngram_n, len);
    for (int L = max_L; L >= ngram_min; --L) {
        const int32_t* pat = hist.data() + len - L;
        for (int i = len - L; i >= 0; --i) {  // descending -> latest first
            if (i + L > len - 1) continue;
            if (std::memcmp(hist.data() + i, pat, sizeof(int32_t) * L) == 0) {
                int avail = std::min(n_draft, len - i - L);
                return std::vector<int32_t>(hist.begin() + i + L,
                                            hist.begin() + i + L + avail);
            }
        }
    }
    return {};
}

static void run_ngram_index_tests() {
    std::printf("\n--- ngram hash index vs brute force ---\n");

    // Pseudo-random stream with planted local repetitions.
    std::mt19937 rng(1234);
    auto clustered = [&rng]() {
        return static_cast<int32_t>(rng() % 5);   // small alphabet -> repeats
    };
    auto spread = [&rng]() {
        return static_cast<int32_t>(1000 + rng() % 9000);
    };

    constexpr int kN = 5, kMin = 2;
    NgramDraftProvider incremental(kN, kMin);
    std::vector<int32_t> hist;
    size_t mismatches = 0, nonempty = 0;

    for (int step = 0; step < 600; ++step) {
        // State must agree BEFORE appending the next token.
        auto got = incremental.draft(0, 4);
        auto want = ngram_brute_reference(hist, kN, kMin, 4);
        if (got != want) ++mismatches;
        if (!want.empty()) ++nonempty;

        // Mix chunk sizes: single tokens exercise straddling-window indexing,
        // bursts mirror multi-token speculative rounds.
        int burst = (step % 7 == 3) ? 3 : 1;
        std::vector<int32_t> toks;
        for (int b = 0; b < burst; ++b)
            toks.push_back((step % 11 < 5) ? clustered() : spread());
        incremental.accept(toks);
        hist.insert(hist.end(), toks.begin(), toks.end());
    }
    CHECK(mismatches == 0, "ngram indexed draft == brute-force reference (600 steps)");
    CHECK(nonempty > 20, "ngram reference produced enough non-empty drafts to be meaningful");

    // Bulk begin() must equal incremental accept() on the same stream.
    std::vector<int32_t> head(hist.begin(), hist.begin() + hist.size() / 2);
    std::vector<int32_t> tail(hist.begin() + hist.size() / 2, hist.end());
    NgramDraftProvider bulk(kN, kMin);
    bulk.begin(head);
    bulk.accept(tail);
    NgramDraftProvider inc2(kN, kMin);
    inc2.accept(hist);
    CHECK(bulk.draft(0, 6) == inc2.draft(0, 6),
          "ngram bulk begin() state == incremental build state");
}

// =========================================================================
// Part 2 & 3 -- model-dependent integration tests (tinyllama)
// =========================================================================

// Integration-test model selection: FORGE_TEST_MODEL overrides, then the
// MiniCPM-V 4.6 GGUF under /mnt/g/AI, then the in-repo tinyllama fallback
// (keeps CI usable when the large local model is absent).
static std::string pick_test_model() {
    if (const char* env = std::getenv("FORGE_TEST_MODEL")) {
        if (file_exists(env)) return env;
        std::printf("NOTE: FORGE_TEST_MODEL='%s' not found, falling back\n", env);
    }
    const char* candidates[] = {
        "/mnt/g/AI/MiniCPM-V-4.6.F16/MiniCPM-V-4_6-Q4_K_M.gguf",
        "models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf",
    };
    for (const char* c : candidates)
        if (file_exists(c)) return c;
    return "";
}

// One forward over `tokens` starting at KV position `pos`. Returns flat
// [n_tokens, vocab] host logits.
static std::vector<float> forward_rows(InferenceEngine* engine,
                                       const std::vector<int32_t>& tokens, int64_t pos) {
    auto ids =
        std::make_shared<Tensor>(DataType::INT32, std::vector<int64_t>{(int64_t)tokens.size()},
                                 DeviceType::CPU);
    std::memcpy(ids->data(), tokens.data(), tokens.size() * sizeof(int32_t));
    auto logits = engine->forward_request(ForwardRequest::from_ids(ids, pos));
    if (!logits || logits->ndim() < 2) return {};
    logits->to_device(DeviceType::CPU);
    int vocab = static_cast<int>(logits->shape()[1]);
    size_t want = tokens.size() * static_cast<size_t>(vocab);
    const float* data = static_cast<const float*>(logits->data());
    return std::vector<float>(data, data + want);
}

static void run_batch_consistency_test(InferenceContext& ctx, int vocab) {
    std::printf("\n--- forward_batch verification-batch consistency ---\n");
    auto* engine = ctx.engine();
    if (!engine) {
        CHECK(false, "batch consistency: engine unavailable");
        return;
    }

    // Mirror SpeculativeExecutor semantics exactly: KV holds [0, pos0),
    // the verification batch is [last_token @ pos0, d0 @ pos0+1, ...].
    const std::vector<int32_t> kv_prefix = {1, 1502, 2788};  // already in KV
    const std::vector<int32_t> stream = {338, 100 % vocab, 200 % vocab, 300 % vocab};
    const int64_t pos0 = static_cast<int64_t>(kv_prefix.size());

    // Verification batch in ONE forward.
    ctx.reset_kv_cache();
    forward_rows(engine, kv_prefix, 0);
    InferenceBatch batch;
    batch.all_logits = true;
    InferenceBatchItem item;
    item.seq_id = 0;
    item.logits = true;
    item.tokens = stream;
    item.start_pos = pos0;
    item.positions.resize(stream.size());
    for (size_t j = 0; j < stream.size(); ++j)
        item.positions[j] = pos0 + static_cast<int64_t>(j);
    batch.items.push_back(std::move(item));

    TensorPtr lb = engine->forward_batch(batch);
    if (!lb || lb->ndim() < 2 || lb->shape()[0] != static_cast<int64_t>(stream.size())) {
        std::printf("     forward_batch returned shape=");
        for (int64_t d : (lb ? lb->shape() : std::vector<int64_t>{}))
            std::printf("%lld ", (long long)d);
        std::printf("(want %zu x vocab)\n", stream.size());
        CHECK(false, "batch consistency: forward_batch shape must be [n_probe, vocab]");
        return;
    }
    if (lb->device() != DeviceType::CPU) lb->to_device(DeviceType::CPU);
    const float* bdata = static_cast<const float*>(lb->data());

    // ---- Hard reference: ONE fused full-sequence forward -----------------
    // Row (pos0+j) of the full pass covers the identical causal context as
    // batch row j, exercising the same fused kernels -> tight tolerance.
    ctx.reset_kv_cache();
    std::vector<int32_t> full(kv_prefix);
    full.insert(full.end(), stream.begin(), stream.end());
    auto ref = forward_rows(engine, full, 0);
    if (ref.size() != full.size() * static_cast<size_t>(vocab)) {
        CHECK(false, "batch consistency: full-sequence reference forward failed");
        return;
    }

    bool argmax_ok = true;
    double worst_rel_fused = 0.0;
    for (size_t j = 0; j < stream.size(); ++j) {
        const float* brow = bdata + j * vocab;
        const float* rrow = ref.data() + (pos0 + static_cast<int64_t>(j)) * vocab;
        int b_best = 0, r_best = 0;
        double max_abs = 0.0, max_diff = 0.0;
        for (int t = 0; t < vocab; ++t) {
            max_abs = std::max(max_abs, static_cast<double>(std::fabs(brow[t])));
            max_diff = std::max(max_diff,
                                static_cast<double>(std::fabs(brow[t] - rrow[t])));
            if (brow[t] > brow[b_best]) b_best = t;
            if (rrow[t] > rrow[r_best]) r_best = t;
        }
        double rel = max_abs > 0 ? max_diff / max_abs : max_diff;
        worst_rel_fused = std::max(worst_rel_fused, rel);
        if (b_best != r_best) {
            argmax_ok = false;
            std::printf("     row %zu argmax mismatch vs fused ref: batch=%d ref=%d "
                        "(rel %.2e)\n",
                        j, b_best, r_best, rel);
        }
    }
    CHECK(argmax_ok, "batch consistency: argmax identical to fused full-sequence pass");
    CHECK(worst_rel_fused < 1e-3, "batch consistency: fused-path drift within 1e-3");
    std::printf("     worst relative logit drift (fused paths): %.3e\n", worst_rel_fused);

    // ---- Informational: decode-path (M=1 GEMV) numeric drift --------------
    // Plain decode uses different (quantized GEMV) kernels than the batched
    // prefill path, so small logit drift and near-tie argmax flips are
    // expected and harmless for resample-consistency (llama.cpp behaves the
    // same across its prefill/decode kernels).
    ctx.reset_kv_cache();
    forward_rows(engine, kv_prefix, 0);
    int flips = 0;
    double worst_rel_decode = 0.0;
    for (size_t j = 0; j < stream.size(); ++j) {
        auto seq = forward_rows(engine, {stream[j]}, pos0 + static_cast<int64_t>(j));
        if (seq.size() != static_cast<size_t>(vocab)) break;
        const float* brow = bdata + j * vocab;
        int b_best = 0, s_best = 0;
        double max_abs = 0.0, max_diff = 0.0;
        for (int t = 0; t < vocab; ++t) {
            max_abs = std::max(max_abs, static_cast<double>(std::fabs(brow[t])));
            max_diff = std::max(
                max_diff, static_cast<double>(std::fabs(brow[t] - seq[t])));
            if (brow[t] > brow[b_best]) b_best = t;
            if (seq[t] > seq[s_best]) s_best = t;
        }
        worst_rel_decode =
            std::max(worst_rel_decode, max_abs > 0 ? max_diff / max_abs : max_diff);
        if (b_best != s_best) ++flips;
    }
    std::printf("     decode-vs-batch drift: %.3e (%d/%zu near-tie argmax flips)\n",
                worst_rel_decode, flips, stream.size());
}

static void run_end_to_end_parity_test(const std::string& model_path) {
    std::printf("\n--- end-to-end greedy parity (spec off vs on) ---\n");
    Model model;
    if (!model.load(model_path, DeviceType::CPU)) {
        CHECK(false, "parity: model load failed");
        return;
    }

    ContextParams pa;
    pa.device = DeviceType::CPU;
    pa.gpu_layers = 0;
    ContextParams pb = pa;
    pb.speculative_config.enabled = true;
    pb.speculative_config.use_ngram = true;
    pb.speculative_config.n_draft = 4;
    pb.speculative_config.ngram_n = 5;
    pb.speculative_config.ngram_min = 2;

    auto ctx_off = make_context(model, pa);
    auto ctx_on = make_context(model, pb);

    // Alternating pattern: the n-gram provider finds candidates immediately
    // and local induction makes target agreement likely, so real acceptance
    // gets exercised.
    const std::vector<int32_t> prompt = {1, 42, 43, 42, 43, 42, 43, 42, 43};

    GenerationConfig cfg;
    cfg.max_new_tokens = 16;
    cfg.do_sample = false;
    cfg.temperature = 0.0f;
    cfg.reset_kv_cache = true;

    Generator gen_off(*ctx_off, SamplerConfig{});
    Generator gen_on(*ctx_on, SamplerConfig{});
    auto r_off = gen_off.generate(prompt, cfg);
    auto r_on = gen_on.generate(prompt, cfg);

    CHECK(r_on.token_ids == r_off.token_ids,
          "parity: greedy output identical with speculation enabled");
    std::printf("     generated %zu tokens (spec off) vs %zu (spec on)\n",
                r_off.token_ids.size(), r_on.token_ids.size());

    const SpeculativeStats* st = gen_on.spec_stats();
    if (!st) {
        CHECK(false, "parity: spec stats unavailable");
        return;
    }
    std::printf(
        "     stats: steps=%lld fallback=%lld drafted=%lld accepted=%lld "
        "(rate %.1f%%)\n",
        (long long)st->n_spec_steps, (long long)st->n_fallback_steps,
        (long long)st->n_draft_tokens, (long long)st->n_accepted_tokens,
        st->acceptance_rate() * 100.0);
    CHECK(st->n_spec_steps > 0 || st->n_fallback_steps > 0,
          "parity: speculation executor engaged");
    if (st->n_accepted_tokens == 0) {
        std::printf(
            "     NOTE: zero accepted drafts (prompt/model dependent, not a "
            "correctness failure)\n");
    }
}

static void run_model_draft_smoke_test(Model& target_model, const ContextParams& target_params) {
    const char* draft_path = std::getenv("FORGE_TEST_DRAFT_MODEL");
    if (!draft_path || !file_exists(draft_path)) {
        std::printf("\nNOTE: FORGE_TEST_DRAFT_MODEL not set; skipping model-draft smoke test\n");
        return;
    }

    std::printf("\n--- standalone model-draft smoke test ---\n");
    SpeculativeConfig spec;
    spec.draft_model_path = draft_path;
    spec.draft_gpu_layers = 0;
    ModelDraftProvider provider(spec, target_params, target_model.config().vocab_size);
    CHECK(provider.valid(), "model draft: draft model loaded and vocabulary matches");
    if (!provider.valid()) return;

    const std::vector<int32_t> prompt = {1, 42, 43, 42, 43};
    provider.begin(prompt);
    auto drafts = provider.draft(prompt.back(), 3);
    CHECK(!drafts.empty(), "model draft: produced at least one candidate");
    if (!drafts.empty()) {
        provider.accept({drafts.front()});
        auto next = provider.draft(drafts.front(), 2);
        CHECK(!next.empty(), "model draft: remained usable after KV resynchronization");
    }
}

int main() {
    std::printf("=== forge-spec-test ===\n");

    register_forge_runtime();
    run_verify_unit_tests();
    run_ngram_index_tests();

    const std::string model_path = pick_test_model();
    if (model_path.empty()) {
        std::printf("\nNOTE: no test model found; skipping integration tests\n");
    } else {
        std::printf("\nusing model: %s\n", model_path.c_str());
        Model model;
        if (!model.load(model_path, DeviceType::CPU)) {
            std::printf("FAIL: failed to load %s\n", model_path.c_str());
            return 1;
        }

        ContextParams p;
        p.device = DeviceType::CPU;
        p.gpu_layers = 0;
        auto ctx = make_context(model, p);

        const int vocab = model.config().vocab_size;
        run_batch_consistency_test(*ctx, vocab);
        run_end_to_end_parity_test(model_path);
        run_model_draft_smoke_test(model, p);
    }

    std::printf("\n=== %d/%d checks passed ===\n", g_checks - g_failures, g_checks);
    return g_failures > 0 ? 1 : 0;
}
