/**
 * forge-server - model loading + chat generation
 *
 * The load pipeline mirrors forge-cli's text-only path (forge_cli.cpp):
 * tokenizer -> model -> engine -> KV cache -> CUDA warmup.
 */

#include "server.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#include "forge/arch_registry.h"
#include "forge/engines/transformer_engine.h"
#include "forge/gguf_model.h"
#include "forge/model_loader.h"
#include "forge/ninf_model.h"
#include "forge/perf_profiler.h"
#include "forge/threads.h"

namespace forge {

// ============================================================================
// Engine & loader registration (same as forge-cli)
// ============================================================================

static void ensure_engines_registered() {
    // Force arch_registrations.cpp to be linked (static auto-registrations).
    (void)_arch_registrations_linked;
    (void)EngineRegistry::instance().registered_archs();
}

static void ensure_loaders_registered() {
    static bool registered = false;
    if (registered)
        return;
    registered = true;

    auto& reg = ModelLoaderRegistry::instance();
    reg.register_loader(
        "ninf", []() -> std::unique_ptr<ModelLoader> { return std::make_unique<NinfModel>(); });
    reg.register_loader(
        "gguf", []() -> std::unique_ptr<ModelLoader> { return std::make_unique<GgufModel>(); });
}

// ============================================================================
// UTF-8-safe streaming helpers
// ============================================================================

// Length of the complete-UTF-8 prefix of `s` (multi-byte sequences that spill
// across decode_token() calls are held back until their tail arrives).
static size_t utf8_complete_len(const std::string& s) {
    size_t i = s.size();
    // Walk back over continuation bytes (10xxxxxx), at most 3.
    size_t cont = 0;
    while (cont < 3 && i > 0 && (static_cast<unsigned char>(s[i - 1]) & 0xC0) == 0x80) {
        --i;
        ++cont;
    }
    if (i > 0) {
        unsigned char lead = static_cast<unsigned char>(s[i - 1]);
        if (lead & 0x80) {  // incomplete sequence starts at s[i-1]
            size_t need = (lead & 0xE0) == 0xC0   ? 2
                          : (lead & 0xF0) == 0xE0 ? 3
                          : (lead & 0xF8) == 0xF0 ? 4
                                                  : 1;
            if (s.size() - (i - 1) < need)
                return i - 1;
        }
    }
    return s.size();
}

// ============================================================================
// ServerState
// ============================================================================

bool ServerState::load(const ServerOptions& opt, std::string& error) {
    opt_ = opt;
    ensure_engines_registered();
    ensure_loaders_registered();

    // --perf-detail: enable the operator-level PerfProfiler (built with
    // FORGE_PROFILING=ON; in a normal build the macros are no-ops and only
    // the per-request summary line will carry data).
    if (opt.perf_detail) {
        PerfProfiler::instance().enable();
        // CUDA event timing is only meaningful when GPU offload is active.
        // With -ngl 0 (pure CPU inference) CUDA events record nothing and
        // every operator would report ~0 ms - use wall-clock timers instead.
        const bool gpu_active = opt.n_gpu_layers != 0;
#ifdef USE_CUDA
        PerfProfiler::instance().set_use_cuda_events(gpu_active);
#endif
    }

    // ---- Tokenizer ----
    auto t0 = std::chrono::high_resolution_clock::now();
    tokenizer_ = std::make_unique<Tokenizer>();
    if (!tokenizer_->load_from_gguf(opt.model_path)) {
        error = "Failed to load tokenizer from: " + opt.model_path;
        return false;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Tokenizer loaded: vocab_size=" << tokenizer_->vocab_size() << ", bos="
              << tokenizer_->bos_token_id() << ", eos=" << tokenizer_->eos_token_id() << " ["
              << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms]\n";

    // ---- Model ----
    auto t2 = std::chrono::high_resolution_clock::now();
    model_ = std::make_unique<Model>();
    if (!opt.load_mode.empty()) {
        model_->set_load_mode(opt.load_mode);
    }
    model_->set_offload_embedding(true);
    model_->set_prefetch(true);
    if (!model_->load(opt.model_path, opt.n_gpu_layers)) {
        error = "Failed to load model: " + opt.model_path;
        return false;
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    const auto& cfg = model_->config();
    std::cout << "Model loaded: arch=" << cfg.arch_type << ", layers=" << cfg.num_layers
              << ", hidden=" << cfg.hidden_dim << ", heads=" << cfg.num_heads << " ["
              << std::chrono::duration<double, std::milli>(t3 - t2).count() << " ms]\n";

    // ---- Inference context ----
    auto t4 = std::chrono::high_resolution_clock::now();
    ctx_ = std::make_unique<InferenceContext>(*model_);
    if (opt.threads > 0) {
        ctx_->params_mut().n_threads = opt.threads;
        ctx_->params_mut().n_threads_batch = opt.threads;
        std::cout << "CPU threads set to: " << opt.threads << "\n";
    } else if (model_->device() != DeviceType::CUDA) {
        int t = detect_best_cpu_threads(cfg.hidden_dim, cfg.intermediate_dim);
        ctx_->params_mut().n_threads = t;
        ctx_->params_mut().n_threads_batch = t;
        std::cout << "CPU threads auto-detected: " << t << "\n";
    }

    auto engine = EngineRegistry::instance().create(cfg.arch_type, *model_, *ctx_);
    if (!engine) {
        error = "No matching engine for architecture: " + cfg.arch_type;
        return false;
    }

    auto* tfm_eng = dynamic_cast<TransformerEngine*>(engine.get());
    if (tfm_eng) {
        KVCacheDType kv_dtype = KVCacheDType::FP32;
        if (opt.kv_cache_dtype == "f16")
            kv_dtype = KVCacheDType::F16;
        else if (opt.kv_cache_dtype == "q8_0")
            kv_dtype = KVCacheDType::Q8_0;
        else if (opt.kv_cache_dtype == "q4_0")
            kv_dtype = KVCacheDType::Q4_0;
        tfm_eng->set_kv_cache_dtype(kv_dtype);
        tfm_eng->set_gpu_layers(opt.n_gpu_layers);
        if (opt.ctx_size > 0) {
            tfm_eng->set_kv_cache_max_seq_len(opt.ctx_size);
        }
        if (opt.cuda_graph && model_->device() == DeviceType::CUDA) {
            tfm_eng->set_use_graph(true);
            tfm_eng->set_cuda_graph_enabled(true);
        }
    }
    ctx_->set_engine(std::move(engine));

    ctx_->params_mut().offload_kqv = true;
    ctx_->params_mut().offload_embedding = model_->offload_embedding();
    if (!ctx_->init_kv_cache()) {
        error = "Failed to initialize KV cache";
        return false;
    }

    if (model_->device() == DeviceType::CUDA) {
        try {
            ctx_->warmup();
            std::cout << "CUDA warmup done\n";
        } catch (const std::runtime_error& e) {
            std::cout << "Warmup skipped (" << e.what() << ")\n";
        }
    }

    std::cout << "Startup total: "
              << std::chrono::duration<double, std::milli>(
                     std::chrono::high_resolution_clock::now() - t0)
                     .count()
              << " ms\n";

    // ---- Chat template (jinja + fallback) ----
    chat_template_ =
        std::make_unique<ChatTemplateEngine>(ChatTemplateEngine::from_tokenizer(*tokenizer_));

    // Model id = file name without directory / extension.
    auto slash = opt.model_path.find_last_of("/\\");
    std::string name = (slash == std::string::npos) ? opt.model_path : opt.model_path.substr(slash + 1);
    auto dot = name.find_last_of('.');
    model_id_ = (dot == std::string::npos) ? name : name.substr(0, dot);
    return true;
}

ChatCompletionResult ServerState::chat_completion(
    const ChatCompletionRequest& req,
    const std::function<bool(const std::string& delta)>& on_token) {
    // Single model / KV cache instance: serialize inference across HTTP workers.
    std::lock_guard<std::mutex> lock(inference_mutex_);

    // ---- Context-capacity-aware history truncation ----
    // The KV cache is capped for memory (e.g. 4096 for large-ctx models, see
    // TransformerEngine::init_kv_cache). A prompt longer than the cap would
    // overflow the cache mid-prefill and produce garbage logits (KVCache
    // refuses the write and attention reads stale cells). Mirror llama.cpp
    // server semantics: keep the system prompt and the most recent turns,
    // dropping oldest non-system messages until the rendered prompt plus the
    // requested generation budget fits.
    int kv_capacity = 0;
    if (auto* eng = ctx_->engine()) {
        if (auto* kvc = eng->kv_cache()) {
            kv_capacity = kvc->max_seq_len();
        }
    }
    if (kv_capacity <= 0) {
        kv_capacity = model_->config().max_seq_len;
    }
    const int gen_budget = req.max_tokens > 0 ? req.max_tokens : 512;

    ChatTemplateInput input;
    input.messages = req.messages;
    input.add_generation_prompt = true;
    std::vector<int32_t> prompt_ids = chat_template_->apply(input);

    if (kv_capacity > 0 &&
        static_cast<int>(prompt_ids.size()) + gen_budget > kv_capacity &&
        req.messages.size() > 2) {
        // Keep message[0] if it is the system prompt; always keep the final
        // message. Progressively drop the oldest non-system, non-final
        // messages, re-rendering after each drop, until the prompt fits.
        const bool has_system = req.messages[0].role == "system";
        std::vector<ChatTemplateMessage> kept;
        if (has_system) {
            kept.push_back(req.messages[0]);
        }
        // Candidate droppable messages: everything after the (optional)
        // system prompt except the final message.
        const size_t first_drop_idx = has_system ? 1 : 0;
        const size_t last_keep_idx = req.messages.size() - 1;  // final message
        size_t drop_upto = first_drop_idx;  // drop messages [first_drop_idx, drop_upto)

        bool fits = false;
        while (drop_upto <= last_keep_idx) {
            ChatTemplateInput trimmed;
            trimmed.messages = kept;
            trimmed.messages.insert(trimmed.messages.end(),
                                    req.messages.begin() + drop_upto,
                                    req.messages.end());
            trimmed.add_generation_prompt = true;
            auto ids = chat_template_->apply(trimmed);
            if (static_cast<int>(ids.size()) + gen_budget <= kv_capacity) {
                prompt_ids = std::move(ids);
                fits = true;
                break;
            }
            ++drop_upto;
        }

        if (fits) {
            std::cout << "[server] prompt truncated to fit KV capacity "
                      << kv_capacity << " (dropped " << (drop_upto - first_drop_idx)
                      << " oldest history messages)\n";
        } else {
            // Even the minimal message set does not fit (huge system prompt or
            // single enormous message). Fail loudly instead of corrupting KV.
            throw std::runtime_error(
                "prompt exceeds model context capacity (" +
                std::to_string(prompt_ids.size()) + " tokens > " +
                std::to_string(kv_capacity - gen_budget) +
                " available after reserving " + std::to_string(gen_budget) +
                " generation tokens)");
        }
    }

    // OpenAI default: temperature=0 means greedy; other defaults are the
    // anti-degeneration choices in ChatCompletionRequest (repeat penalty etc.).
    // --perf-detail: open a fresh profiling window for this request.
    if (opt_.perf_detail) {
        PerfProfiler::instance().reset();
    }
    GenerationConfig gen_cfg;
    gen_cfg.max_new_tokens = req.max_tokens > 0 ? req.max_tokens : 512;
    gen_cfg.temperature = req.temperature;
    gen_cfg.top_k = req.top_k;
    gen_cfg.top_p = req.top_p;
    gen_cfg.repeat_penalty = req.repeat_penalty;
    gen_cfg.repeat_last_n = req.repeat_last_n;
    gen_cfg.do_sample = req.temperature > 0.0f;
    gen_cfg.seed = req.seed;
    gen_cfg.eos_token_id = tokenizer_->eos_token_id();

    std::atomic<bool> abort_flag{false};
    gen_cfg.abort_flag = &abort_flag;

    SamplerConfig sampler_cfg;
    sampler_cfg.temperature = req.temperature;
    sampler_cfg.top_k = req.top_k;
    sampler_cfg.top_p = req.top_p;
    sampler_cfg.repeat_penalty = req.repeat_penalty;
    sampler_cfg.repeat_last_n = req.repeat_last_n;
    sampler_cfg.do_sample = gen_cfg.do_sample;
    sampler_cfg.seed = req.seed;
    sampler_cfg.logit_softcapping = model_->config().f_final_logit_softcapping;

    // ---- Streaming + stop-string handling ----
    // Decoded bytes accumulate in `pending`; only complete UTF-8 sequences are
    // emitted. With stop strings configured, the last (max_stop_len - 1) bytes
    // are held back so a stop string is never partially streamed; when one
    // matches, generation aborts and nothing beyond the match is emitted.
    size_t max_stop = 0;
    for (const auto& s : req.stop) {
        max_stop = std::max(max_stop, s.size());
    }
    const size_t hold = max_stop > 0 ? max_stop - 1 : 0;
    const size_t tail_keep = max_stop * 2 + 128;

    std::string pending;
    std::string tail;  // recent decoded text for stop matching
    bool stop_hit = false;

    Generator::TokenCallback cb;
    cb = [&](int32_t tok, int /*step*/) {
        if (tok == gen_cfg.eos_token_id) {
            // EOS is invoked by generate() before stopping; decode_token() does
            // not skip special tokens, so emit nothing for it (the non-stream
            // path already filters via decode(skip_special=true)).
            return;
        }
        pending += tokenizer_->decode_token(tok);
        size_t ok = utf8_complete_len(pending);
        if (ok == 0)
            return;
        tail += pending.substr(0, ok);
        if (tail.size() > tail_keep) {
            tail.erase(0, tail.size() - tail_keep);
        }
        for (const auto& s : req.stop) {
            if (!s.empty() && tail.find(s) != std::string::npos) {
                stop_hit = true;
                break;
            }
        }
        size_t emit = ok;
        if (max_stop > 0) {
            emit = ok > hold ? ok - hold : 0;
        }
        if (stop_hit) {
            pending.clear();
            abort_flag.store(true, std::memory_order_relaxed);
            return;
        }
        if (emit > 0) {
            std::string delta = pending.substr(0, emit);
            if (on_token && !on_token(delta)) {
                // Client disconnected: stop burning compute on this request.
                abort_flag.store(true, std::memory_order_relaxed);
            }
        }
        // Keep [emit, ok) in `pending` (stop-string hold window); the byte
        // range beyond `ok` stays anyway (incomplete UTF-8 sequence).
        pending.erase(0, emit);
    };

    Generator gen(*ctx_, sampler_cfg);
    GenerationResult result = gen.generate(prompt_ids, gen_cfg, cb);

    if (on_token && !stop_hit && !pending.empty()) {
        on_token(pending);  // flush any trailing partial sequence
    }

    ChatCompletionResult out;
    out.text = tokenizer_->decode(result.token_ids, true, false);

    // Truncate at the earliest stop string (also a safety net for the
    // non-streaming path, where stop detection only drives the abort flag).
    if (max_stop > 0) {
        size_t cut = std::string::npos;
        for (const auto& s : req.stop) {
            size_t p = out.text.find(s);
            if (p != std::string::npos && p < cut) {
                cut = p;
            }
        }
        if (cut != std::string::npos) {
            out.text.resize(cut);
            stop_hit = true;
        }
    }

    out.prompt_tokens = result.num_prompt_tokens;
    out.completion_tokens = result.num_generated_tokens;

    if (stop_hit) {
        out.finish_reason = "stop";
    } else if (result.finish_reason == "length") {
        out.finish_reason = "length";
    } else if (result.finish_reason == "aborted") {
        // Client disconnected mid-stream: reason is irrelevant (response is
        // dropped), report "stop" for consistency.
        out.finish_reason = "stop";
    } else {
        out.finish_reason = "stop";  // eos / stop token
    }

    // ---- --perf / --perf-detail report (stdout, after every completion) ----
    if (opt_.perf) {
        const int gen_tok = result.num_generated_tokens;
        const double prefill_ms = result.prefill_ms;
        const double decode_ms = result.decode_ms;
        std::cout << "[perf] prompt=" << out.prompt_tokens
                  << " tok | gen=" << gen_tok << " tok"
                  << " | prefill=" << std::fixed << std::setprecision(1) << prefill_ms
                  << " ms (" << std::setprecision(2) << result.prefill_ms_per_1k()
                  << " ms/1k-tok)"
                  << " | decode=" << std::setprecision(1) << decode_ms << " ms"
                  << " | " << std::setprecision(2) << result.tokens_per_sec()
                  << " tok/s"
                  << " | finish=" << out.finish_reason << std::endl;
        if (result.spec_stats) {
            std::cout << "[perf] speculative: accepted=" << result.spec_stats->n_accepted_tokens
                      << " drafted=" << result.spec_stats->n_draft_tokens
                      << " (rate=" << std::setprecision(1)
                      << 100.0 * result.spec_stats->acceptance_rate() << "%)\n";
        }
        if (opt_.perf_detail) {
            // Per-request window: PerfProfiler is reset at the start of every
            // chat_completion (below), so this summary covers exactly the
            // current request. flush_deferred merges CUDA-event timings that
            // were recorded asynchronously during the request.
#ifdef USE_CUDA
            PerfProfiler::instance().flush_deferred();
#endif
            auto rows = PerfProfiler::instance().summary();
            std::sort(rows.begin(), rows.end(),
                      [](const auto& a, const auto& b) {
                          return a.second.total_ms > b.second.total_ms;
                      });
            const size_t top_n = 12;
            std::cout << "[perf] top operators (this request):\n";
            if (rows.empty()) {
                std::cout << "  (no operator data - rebuild with "
                             "FORGE_PROFILING=ON for --perf-detail)\n";
            }
            for (size_t i = 0; i < rows.size() && i < top_n; ++i) {
                const auto& [name, rec] = rows[i];
                const double avg = rec.count > 0 ? rec.total_ms / rec.count : 0.0;
                std::cout << "  " << std::left << std::setw(38) << name << std::right
                          << " n=" << std::setw(6) << rec.count
                          << "  total=" << std::fixed << std::setprecision(2)
                          << std::setw(9) << rec.total_ms << " ms"
                          << "  avg=" << std::setw(7) << avg << " ms\n";
            }
            std::cout.flush();
        }
    }
    return out;
}

// ============================================================================
// Admission control
// ============================================================================

bool ServerState::try_acquire_slot() {
    int cur = active_requests_.load(std::memory_order_relaxed);
    const int cap = opt_.max_inflight > 0 ? opt_.max_inflight : 1;
    while (cur < cap) {
        if (active_requests_.compare_exchange_weak(cur, cur + 1, std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

void ServerState::release_slot() {
    active_requests_.fetch_sub(1, std::memory_order_relaxed);
}

}  // namespace forge
