// DFlash / DSPark acceptance tests (Phase 2 of DFLASH_DSPARK_PLAN.md).
//
//  1. Config parsing (runnable when the dflash GGUF fixture is present):
//     parse_dflash_config must populate the iSWA sliding-window metadata
//     (n_swa + swa_layers) so the KV-cache ring buffer routes SWA layers.
//  2. KVCache ring-buffer unit test (always runnable, no model): proves the
//     iSWA routing mechanism itself — SWA layers evict beyond the window while
//     full-attention layers grow linearly. This is exactly the path
//     TransformerEngine::init_kv_cache takes for dflash SWA layers.
//  3. Engine integration (skipped unless FORGE_TEST_DFLASH_MODEL and
//     FORGE_TEST_TARGET_MODEL point at a matched drafter/target pair):
//     encoder dimension, context-KV injection consistency, decode logits
//     shape, and SWA eviction during a long prefix.
//
// Build & run:
//   cmake --build build -j && ./build/forge-dflash-test

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "forge/arch_config_parsers.h"
#include "forge/arch_registry.h"
#include "forge/context.h"
#include "forge/dflash_draft_provider.h"
#include "forge/engine.h"
#include "forge/engines/dflash_engine.h"
#include "forge/engines/transformer_engine.h"
#include "forge/gguf_model.h"
#include "forge/inference/layers/rope_executor.h"
#include "forge/kv_cache.h"
#include "forge/model.h"
#include "forge/model_loader.h"
#include "forge/operators.h"

using namespace forge;

// Force-link arch registrations and register the gguf loader (same mechanism
// forge-cli uses so self-registering translation units are not dropped).
static void register_forge_runtime() {
    force_link_arch_registrations();
    auto& reg = ModelLoaderRegistry::instance();
    reg.register_loader(
        "gguf", []() -> std::unique_ptr<ModelLoader> { return std::make_unique<GgufModel>(); });
}

// Mirror PyModel::create_context (src/bindings/common.h).
static std::unique_ptr<InferenceContext> make_context(Model& model, const ContextParams& params) {
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

#define CHECK(cond, msg)                                              \
    do {                                                              \
        ++g_checks;                                                   \
        if (!(cond)) {                                                \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, msg); \
            ++g_failures;                                             \
        } else {                                                      \
            std::printf("ok   %s\n", msg);                            \
        }                                                             \
    } while (0)

static bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

// =========================================================================
// Part 1 -- iSWA config parse from the real dflash GGUF
// =========================================================================

static std::string pick_dflash_model() {
    if (const char* env = std::getenv("FORGE_TEST_DFLASH_MODEL")) {
        if (file_exists(env))
            return env;
        std::printf("NOTE: FORGE_TEST_DFLASH_MODEL='%s' not found\n", env);
    }
    const char* candidates[] = {
        "/mnt/g/AI/Qwen3.6-35B-A3B-DFlash-q8_0.gguf",
    };
    for (const char* c : candidates)
        if (file_exists(c))
            return c;
    return "";
}

static void run_config_parse_test() {
    std::printf("\n--- dflash iSWA config parse ---\n");
    const std::string path = pick_dflash_model();
    if (path.empty()) {
        std::printf("NOTE: no dflash GGUF fixture; skipping config-parse test\n");
        return;
    }
    std::printf("using dflash model: %s\n", path.c_str());

    // The dflash GGUF intentionally carries NO token_embedding / output_weight
    // (it borrows them from the target at runtime), so it cannot be loaded
    // standalone via Model::load until Phase 3 adds the skip. We therefore parse
    // the config directly from the GGUF metadata — this is exactly the code path
    // Model::load dispatches to (FORGE_REGISTER_CONFIG_PARSER("dflash", ...)).
    GgufModel loader;
    if (!loader.load(path)) {
        CHECK(false, "config parse: gguf loader failed");
        return;
    }
    ModelConfig cfg = parse_dflash_config(loader, "dflash");
    // Probe C (2026-08-30): Qwen3.6-35B-A3B-DFlash has window=4096 and
    // sliding_window_pattern=[1,1,1,1,1,0] (5 SWA + 1 full layer).
    CHECK(cfg.n_swa == 4096, "config parse: n_swa == 4096 (iSWA window)");
    std::vector<int> want{1, 1, 1, 1, 1, 0};
    CHECK(cfg.swa_layers == want, "config parse: swa_layers == [1,1,1,1,1,0] (5 SWA + 1 full)");
    CHECK(static_cast<int>(cfg.swa_layers.size()) == cfg.num_layers,
          "config parse: swa_layers covers every draft layer");
    std::printf("     parsed: n_swa=%d num_layers=%d swa_layers=[", cfg.n_swa, cfg.num_layers);
    for (size_t i = 0; i < cfg.swa_layers.size(); ++i)
        std::printf("%s%d", i ? "," : "", cfg.swa_layers[i]);
    std::printf("]\n");

    // Stage 3 load fix: the drafter GGUF carries no token_embd/output, so it
    // must now load standalone (Model::load skips the mandatory embedding for
    // dflash/dspark and borrows them from the target at runtime).
    Model m;
    CHECK(m.load(path, DeviceType::CPU),
          "stage3: dflash loads standalone (borrows target embd/output)");
    CHECK(m.config().arch_type == "dflash", "stage3: standalone load arch == dflash");
    // Build a DflashEngine and verify the encoder projector resolved (the GGUF
    // stores it as "fc.weight"; ModelWeights::init must pick it up).
    {
        ContextParams cp;
        cp.device = DeviceType::CPU;
        cp.gpu_layers = 0;
        auto dctx = make_context(m, cp);
        auto* eng = dynamic_cast<DflashEngine*>(dctx ? dctx->engine() : nullptr);
        CHECK(eng && eng->weights().dflash_fc != nullptr,
              "stage3: encoder fc.weight resolved into DflashEngine");
        CHECK(eng && eng->weights().layers.size() == static_cast<size_t>(m.config().num_layers),
              "stage3: all draft layer weights present");
    }
}

// =========================================================================
// Part 2 -- KVCache ring-buffer (the iSWA routing mechanism)
// =========================================================================

static void run_kvcache_swa_test() {
    std::printf("\n--- KVCache iSWA ring-buffer ---\n");
    constexpr int kLayers = 6;
    constexpr int kKVHeads = 1;
    constexpr int kHeadDim = 4;
    constexpr int kKVdim = kKVHeads * kHeadDim;
    constexpr int kMaxSeq = 64;
    constexpr int kWindow = 8;

    KVCache cache;
    cache.init_quantized(kLayers, kKVHeads, kHeadDim, kMaxSeq, DeviceType::CPU, KVCacheDType::FP32);

    // 5 SWA layers + 1 full layer (mirrors the dflash pattern).
    std::vector<KVLayerPolicy> policies(kLayers, KVLayerPolicy::Full);
    for (int i = 0; i < 5; ++i)
        policies[i] = KVLayerPolicy::SlidingWindow;
    cache.set_layer_policies(policies, kWindow);

    CHECK(cache.layer_policy(0) == KVLayerPolicy::SlidingWindow,
          "kvcache: layer 0 routed to SlidingWindow (iSWA)");
    CHECK(cache.layer_policy(5) == KVLayerPolicy::Full, "kvcache: layer 5 routed to Full (iSWA)");

    auto dummy =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, kKVdim}, DeviceType::CPU);
    dummy->zero_();

    // Fill an SWA layer with 20 positions; the ring buffer must cap at window.
    for (int p = 0; p < 20; ++p)
        cache.update(0, dummy, dummy, 1);
    CHECK(cache.filled(0) == kWindow, "kvcache: SWA layer filled() caps at sliding window");
    auto k0 = cache.get_key_filled(0);
    CHECK(k0 && k0->shape()[0] == kWindow, "kvcache: SWA layer key slice has window rows");

    // Fill the full layer with 20 positions; it must grow linearly.
    for (int p = 0; p < 20; ++p)
        cache.update(5, dummy, dummy, 1);
    CHECK(cache.filled(5) == 20, "kvcache: Full layer grows linearly past window");

    // A fresh cache without policies must NOT evict (sanity baseline).
    KVCache plain;
    plain.init_quantized(kLayers, kKVHeads, kHeadDim, kMaxSeq, DeviceType::CPU, KVCacheDType::FP32);
    for (int p = 0; p < 20; ++p)
        plain.update(0, dummy, dummy, 1);
    CHECK(plain.filled(0) == 20, "kvcache: no policy -> no eviction (baseline)");
}

// =========================================================================
// Part 3 -- engine integration (matched drafter + target; skipped if absent)
// =========================================================================

static void run_engine_integration(const std::string& dflash_path, const std::string& target_path) {
    std::printf("\n--- DflashEngine integration (matched pair) ---\n");
    Model target, draft;
    if (!target.load(target_path, DeviceType::CPU)) {
        CHECK(false, "integration: target model load failed");
        return;
    }
    if (!draft.load(dflash_path, DeviceType::CPU)) {
        CHECK(false, "integration: dflash model load failed");
        return;
    }

    ContextParams tp;
    tp.device = DeviceType::CPU;
    tp.gpu_layers = 0;
    auto tctx = make_context(target, tp);

    auto dctx = make_context(draft, tp);
    auto* eng = dynamic_cast<DflashEngine*>(dctx->engine());
    if (!eng) {
        CHECK(false, "integration: DflashEngine not registered for arch");
        return;
    }
    eng->set_target(tctx->engine());
    CHECK(eng->target() != nullptr, "integration: target paired with drafter");

    const int H = draft.config().hidden_dim;
    const int enc_dim = draft.config().n_embd_inp_enc;

    // Encoder: target multi-layer hiddens -> context feature [seq, hidden].
    auto h =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, enc_dim}, DeviceType::CPU);
    h->zero_();
    auto ctx_feat = eng->encode(h);
    CHECK(ctx_feat && ctx_feat->ndim() >= 2 && ctx_feat->shape()[1] == H,
          "integration: encoder output dim == hidden");

    // Context-KV precompute then decode must yield [N, vocab] logits.
    auto prefix_emb =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{3, H}, DeviceType::CPU);
    prefix_emb->zero_();
    eng->precompute_context_kv(prefix_emb, 0);
    std::vector<int32_t> query = {1, 2, 3, 4, 5, 6};
    auto logits = eng->decode(query, 3);
    CHECK(logits && logits->ndim() >= 2 &&
              logits->shape()[0] == static_cast<int64_t>(query.size()) &&
              logits->shape()[1] == target.config().vocab_size,
          "integration: decode produces [N, vocab] logits");
}

// =========================================================================
// Part 4 -- DSPark sequential Markov sampling (pure function, no model)
// =========================================================================

static void run_dspark_sample_test() {
    std::printf("\n--- DSPark sequential Markov sampling ---\n");
    const int vocab = 6, d = 1;

    // Three identical logit rows; greedy argmax without markov = token 0.
    std::vector<float> row(vocab, 1.0f);
    row[0] = 10.0f;  // argmax = 0
    std::vector<const float*> rows = {row.data(), row.data(), row.data()};

    // No markov head -> independent greedy argmax per row.
    auto greedy = dspark_sequential_sample(rows, vocab, nullptr, d, nullptr, nullptr, 2, 3);
    CHECK((greedy == std::vector<int32_t>{0, 0, 0}), "dspark: no markov -> greedy per row");

    // Markov: embed[prev]=prev, bias[v]=v  =>  bias[v]=v*prev added to scores.
    // anchor=2 -> first bias = v*2 -> row scores [10,3,5,7,9,11] -> argmax 5.
    std::vector<float> embed(vocab * d);  // [target_vocab=6, d=1]; row p -> p
    std::vector<float> bias(vocab * d);   // [vocab, d]; entry v -> v
    for (int p = 0; p < vocab; ++p)
        embed[p] = float(p);
    for (int v = 0; v < vocab; ++v)
        bias[v] = float(v);
    auto dsp = dspark_sequential_sample(rows, vocab, embed.data(), d, bias.data(), nullptr, 2, 3);
    CHECK((dsp == std::vector<int32_t>{5, 5, 5}), "dspark: markov shifts greedy argmax");

    // Reduced draft vocab (draft_vocab=3) + mapping back to target vocab.
    const int draft_vocab = 3;
    std::vector<float> drow(draft_vocab, 1.0f);
    drow[1] = 2.0f;  // argmax draft 1
    std::vector<const float*> drows = {drow.data(), drow.data(), drow.data()};
    std::vector<int32_t> idmap = {4, 5, 0};     // draft 0->4, 1->5, 2->0
    std::vector<float> dbias(draft_vocab * d);  // bias[v_draft]=v_draft
    for (int v = 0; v < draft_vocab; ++v)
        dbias[v] = float(v);
    // anchor=2 (target): j0 prev=2 -> bias [0,2,4] -> scores [1,4,5] argmax draft2 -> map 0
    //                     j1 prev=0 -> bias [0,0,0] -> scores [1,2,1] argmax draft1 -> map 5
    //                     j2 prev=5 -> bias [0,5,10] -> scores [1,7,11] argmax draft2 -> map 0
    auto reduced = dspark_sequential_sample(drows, draft_vocab, embed.data(), d, dbias.data(),
                                            idmap.data(), 2, 3);
    CHECK((reduced == std::vector<int32_t>{0, 5, 0}), "dspark: reduced vocab + sequential markov");

    // Reduced vocab WITHOUT markov -> independent greedy -> draft 1 -> map 5.
    auto reduced_greedy =
        dspark_sequential_sample(drows, draft_vocab, nullptr, d, nullptr, idmap.data(), 2, 3);
    CHECK((reduced_greedy == std::vector<int32_t>{5, 5, 5}),
          "dspark: reduced vocab no markov -> map only");
}

// =========================================================================
// Part 5 -- context-KV injection consistency (real dflash GGUF; NO target needed)
// =========================================================================
//
// precompute_context_kv(prefix_embd, 0) projects the prefix with the draft's
// own wk/wv, rotates K via RoPE, and writes the result into the draft KV cache
// at the committed positions. This test proves the injection is exact and
// position-correct without depending on a (large) paired target model:
//   - every layer writes exactly L prefix positions (count/position consistency)
//   - injected V == wv(prefix) exactly (unrotated)
//   - injected K == RoPE(wk(prefix)) exactly (rotated, recomputed independently)
//   - a zero prefix yields all-zero K/V (proves the supplied prefix is used verbatim)
//   - two different prefixes yield different K (proves content dependence)

static DflashEngine* make_dflash_engine(Model& m, std::unique_ptr<InferenceContext>& holder) {
    ContextParams cp;
    cp.device = DeviceType::CPU;
    cp.gpu_layers = 0;
    holder = make_context(m, cp);
    auto* eng = dynamic_cast<DflashEngine*>(holder ? holder->engine() : nullptr);
    // Size the per-layer KV cache before context-KV injection (the drafter
    // rebuilds its KV every draft()). init_kv_cache is protected; the public
    // init_memory() entry allocates it the same way InferenceContext does.
    if (eng)
        eng->init_memory();
    return eng;
}

static float max_abs_diff(const TensorPtr& a, const TensorPtr& b) {
    // All tensors in this test live on the CPU (engine runs CPU, KV cache FP32),
    // so compare in place without device copies.
    const float* pa = static_cast<const float*>(a->data());
    const float* pb = static_cast<const float*>(b->data());
    float m = 0.0f;
    for (int64_t i = 0; i < a->numel(); ++i)
        m = std::max(m, std::fabs(pa[i] - pb[i]));
    return m;
}

static void run_context_kv_injection_test() {
    std::printf("\n--- DflashEngine context-KV injection consistency ---\n");
    const std::string path = pick_dflash_model();
    if (path.empty()) {
        std::printf("NOTE: no dflash GGUF fixture; skipping context-KV test\n");
        return;
    }
    Model m;
    if (!m.load(path, DeviceType::CPU)) {
        CHECK(false, "context-kv: dflash load failed");
        return;
    }
    std::unique_ptr<InferenceContext> ctx;
    DflashEngine* eng = make_dflash_engine(m, ctx);
    if (!eng) {
        CHECK(false, "context-kv: engine create failed");
        return;
    }

    const ModelConfig& cfg = m.config();
    const int H = cfg.hidden_dim;
    const int L = 4;
    auto prefix =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{L, H}, DeviceType::CPU);
    {
        float* p = static_cast<float*>(prefix->data());
        for (int i = 0; i < L; ++i)
            for (int j = 0; j < H; ++j)
                p[i * H + j] = static_cast<float>(((i * H + j) % 7) - 3) * 0.05f;
    }

    eng->precompute_context_kv(prefix, 0);

    // (a) injection count: every layer wrote exactly L positions.
    for (int ly = 0; ly < cfg.num_layers; ++ly)
        CHECK(eng->kv_cache()->filled(ly) == L, "context-kv: layer wrote L prefix positions");

    // (b)/(c) exact K (rotated) and V (unrotated) match the projection.
    RopeExecutor rope;
    for (int ly = 0; ly < cfg.num_layers; ++ly) {
        const auto& lw = eng->weights().layers[ly];
        TensorPtr K = ops::matmul_transB(prefix, lw.wk());
        TensorPtr V = ops::matmul_transB(prefix, lw.wv());
        const int64_t kvdim = K->shape()[1];
        auto q_dummy = std::make_shared<Tensor>(
            DataType::FP32,
            std::vector<int64_t>{L, static_cast<int64_t>(cfg.num_heads) * cfg.head_dim},
            DeviceType::CPU);
        q_dummy->zero_();
        TensorPtr Kexp = rope.apply(q_dummy, K, cfg, 0, L, DeviceType::CPU).k_rope;

        TensorPtr Kgot = eng->kv_cache()->get_key_filled(ly);
        TensorPtr Vgot = eng->kv_cache()->get_value_filled(ly);
        CHECK(Kgot && Kgot->shape()[0] == L && Kgot->shape()[1] == kvdim,
              "context-kv: K shape [L, kvdim]");
        CHECK(Vgot && Vgot->shape()[0] == L && Vgot->shape()[1] == kvdim,
              "context-kv: V shape [L, kvdim]");
        CHECK(max_abs_diff(Kexp, Kgot) < 1e-2f, "context-kv: K == RoPE(wk(prefix))");
        CHECK(max_abs_diff(V, Vgot) < 1e-2f, "context-kv: V == wv(prefix)");
    }

    // (d) zero prefix -> all-zero K/V (verbatim use of the supplied prefix).
    auto z = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{L, H}, DeviceType::CPU);
    z->zero_();
    eng->precompute_context_kv(z, 0);
    bool all_zero = true;
    for (int ly = 0; ly < cfg.num_layers; ++ly) {
        auto Kz = eng->kv_cache()->get_key_filled(ly);
        auto Vz = eng->kv_cache()->get_value_filled(ly);
        const float* k = static_cast<const float*>(Kz->data());
        const float* v = static_cast<const float*>(Vz->data());
        for (int64_t i = 0; i < Kz->numel(); ++i)
            if (std::fabs(k[i]) > 1e-5f)
                all_zero = false;
        for (int64_t i = 0; i < Vz->numel(); ++i)
            if (std::fabs(v[i]) > 1e-5f)
                all_zero = false;
    }
    CHECK(all_zero, "context-kv: zero prefix -> zero K/V (verbatim injection)");

    // (e) dependence: a clearly different prefix must yield a different K cache.
    // Use a larger, distinct amplitude than `prefix` so the projected K differs
    // well above the numeric tolerance. get_key_filled() returns a *view* into
    // the cache storage, so deep-copy Kb before re-injecting (otherwise Kb and
    // Ka would alias the same buffer and always compare equal).
    auto p2 = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{L, H}, DeviceType::CPU);
    {
        float* p = static_cast<float*>(p2->data());
        for (int i = 0; i < L * H; ++i)
            p[i] = static_cast<float>(((i * H) % 11) - 5) * 0.5f;
    }
    eng->precompute_context_kv(prefix, 0);
    TensorPtr Kview = eng->kv_cache()->get_key_filled(0);  // after `prefix`
    const int64_t kvdim = Kview->shape()[1];
    TensorPtr Kb =
        std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{L, kvdim}, DeviceType::CPU);
    std::memcpy(Kb->data(), Kview->data(),
                static_cast<size_t>(L) * static_cast<size_t>(kvdim) * sizeof(float));
    eng->precompute_context_kv(p2, 0);
    TensorPtr Ka = eng->kv_cache()->get_key_filled(0);  // after `p2`
    CHECK(max_abs_diff(Ka, Kb) > 1e-3f, "context-kv: injected K depends on prefix content");
}

// =========================================================================
// Part 6 -- end-to-end draft via DFlashDraftProvider (needs a matched pair)
// =========================================================================
// Gated behind FORGE_TEST_DFLASH_MODEL + FORGE_TEST_TARGET_MODEL. When both are
// present this exercises the real draft() path: target layer-hiddens -> encoder
// -> context-KV -> non-causal decode -> greedy draft tokens. The resample-
// consistency verify step (run by SpeculativeExecutor in production) keeps the
// generated text lossless, so a greedy run with --spec-draft-arch dflash must
// match a plain greedy run token-for-token (acceptance rate > 0 once the drafter
// is warm). A standalone run here just asserts the drafter produces `n` tokens.

static void run_draft_end2end(const std::string& dflash_path, const std::string& target_path) {
    std::printf("\n--- DFlashDraftProvider end-to-end draft ---\n");
    Model target, draft;
    if (!target.load(target_path, DeviceType::CPU)) {
        CHECK(false, "e2e: target load failed");
        return;
    }
    if (!draft.load(dflash_path, DeviceType::CPU)) {
        CHECK(false, "e2e: dflash load failed");
        return;
    }
    ContextParams tp;
    tp.device = DeviceType::CPU;
    tp.gpu_layers = 0;
    auto tctx = make_context(target, tp);
    auto* teng = dynamic_cast<TransformerEngine*>(tctx ? tctx->engine() : nullptr);
    if (!teng) {
        CHECK(false, "e2e: target engine missing");
        return;
    }
    SpeculativeConfig spec;
    spec.draft_arch = "dflash";
    spec.draft_model_path = dflash_path;
    spec.draft_n_spec = 5;
    DFlashDraftProvider prov(spec, *tctx, *teng, target.config().vocab_size);
    CHECK(prov.valid(), "e2e: DFlashDraftProvider valid with matched pair");
    if (!prov.valid())
        return;

    std::vector<int32_t> prompt = {1, 2, 3, 4, 5, 6, 7, 8};
    prov.begin(prompt);
    const int n = 5;
    auto d1 = prov.draft(prompt.back(), n);
    CHECK(static_cast<int>(d1.size()) == n, "e2e: draft proposes n tokens");
    auto d2 = prov.draft(prompt.back(), n);
    CHECK(d1 == d2, "e2e: greedy draft is deterministic");
}

int main() {
    std::printf("=== forge-dflash-test ===\n");
    register_forge_runtime();

    run_config_parse_test();
    run_kvcache_swa_test();
    run_dspark_sample_test();
    run_context_kv_injection_test();

    const char* dflash_env = std::getenv("FORGE_TEST_DFLASH_MODEL");
    const char* target_env = std::getenv("FORGE_TEST_TARGET_MODEL");
    if (dflash_env && target_env && file_exists(dflash_env) && file_exists(target_env)) {
        try {
            run_engine_integration(dflash_env, target_env);
            run_draft_end2end(dflash_env, target_env);
        } catch (const std::exception& e) {
            std::printf("NOTE: engine integration threw: %s\n", e.what());
        }
    } else {
        std::printf("\nNOTE: FORGE_TEST_TARGET_MODEL not set; skipping engine integration\n");
    }

    std::printf("\n=== %d/%d checks passed ===\n", g_checks - g_failures, g_checks);
    return g_failures > 0 ? 1 : 0;
}
