/**
 * forge-server - model loading + chat generation
 *
 * The load pipeline mirrors forge-cli's text-only path (forge_cli.cpp):
 * tokenizer -> model -> engine -> KV cache -> CUDA warmup.
 */

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>

#include "forge/agent/tool_dialect.h"
#include "forge/agent/tool_registry.h"
#include "forge/arch_registry.h"
#include "forge/engines/transformer_engine.h"
#include "forge/gguf_model.h"
#include "forge/kv_memory.h"
#include "forge/logger.h"
#include "forge/model_loader.h"
#include "forge/ninf_model.h"
#include "forge/perf_profiler.h"
#include "forge/prefix_cache.h"
#include "forge/threads.h"
#include "forge/vision_encoder.h"
#include "forge/vision_registry.h"
#include "forge/operator_embedding.h"
#include "forge/tensor.h"
#include "forge/weight_store.h"
#include "server.h"

// Vendored single-header image decoder (public domain, https://github.com/nothings/stb).
// The CLI only handles PPM; the server must decode the PNG/JPEG (etc.) images the
// web dashboard sends as data URLs.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace forge {

namespace {
// Decode a PNG/JPEG/BMP/TGA/GIF (stb_image) into interleaved RGB float
// [w*h*3], values 0..255 -- the exact layout the vision encoder's preprocess
// expects (matching the CLI's PPM path).
bool decode_image_rgb(const std::vector<uint8_t>& bytes, int& w, int& h,
                      std::vector<float>& rgb) {
    if (bytes.empty()) return false;
    int ch = 0;
    stbi_uc* px = stbi_load_from_memory(bytes.data(), static_cast<int>(bytes.size()),
                                        &w, &h, &ch, 3);
    if (!px) return false;
    const size_t n = static_cast<size_t>(w) * static_cast<size_t>(h) * 3u;
    rgb.resize(n);
    for (size_t i = 0; i < n; ++i) rgb[i] = static_cast<float>(px[i]);
    stbi_image_free(px);
    return true;
}
}  // namespace

ServerState::~ServerState() {
    delete vision_;
}

// Tool-call catalogue injection, dialect parsing and wrapper stripping moved
// to the forge_agent module (src/agent/, include/forge/agent/); the request
// pipeline is built per chat_completion() call from agent_cfg_.

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
    // Tool-call dialect selection is a process-wide setting: resolve the env
    // override once (FORGE_TOOL_DIALECT, default "auto").
    agent_cfg_ = agent_pipeline_config_from_env();
    // --enable-code-exec gates the run_python tool. The registry reads
    // FORGE_ALLOW_CODE_EXEC itself, so the flag has to be pushed in before the
    // tool catalogue is first built (first request, or the startup probe).
    forge::set_code_exec_enabled(opt.enable_code_exec);
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
        const bool gpu_active = opt.common.n_gpu_layers != 0;
#ifdef USE_CUDA
        PerfProfiler::instance().set_use_cuda_events(gpu_active);
#endif
    }

    // ---- Tokenizer ----
    auto t0 = std::chrono::high_resolution_clock::now();
    tokenizer_ = std::make_unique<Tokenizer>();
    if (!tokenizer_->load_from_gguf(opt.common.model_path)) {
        error = "Failed to load tokenizer from: " + opt.common.model_path;
        return false;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Tokenizer loaded: vocab_size=" << tokenizer_->vocab_size()
              << ", bos=" << tokenizer_->bos_token_id() << ", eos=" << tokenizer_->eos_token_id()
              << " [" << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms]\n";

    // ---- Model ----
    auto t2 = std::chrono::high_resolution_clock::now();
    model_ = std::make_unique<Model>();
    if (!opt.common.load_mode.empty()) {
        model_->set_load_mode(opt.common.load_mode);
    }
    model_->set_offload_embedding(true);
    model_->set_prefetch(true);
    if (!model_->load(opt.common.model_path, opt.common.n_gpu_layers)) {
        error = "Failed to load model: " + opt.common.model_path;
        return false;
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    const auto& cfg = model_->config();
    std::cout << "Model loaded: arch=" << cfg.arch_type << ", layers=" << cfg.num_layers
              << ", hidden=" << cfg.hidden_dim << ", heads=" << cfg.num_heads << " ["
              << std::chrono::duration<double, std::milli>(t3 - t2).count() << " ms]\n";

    // ---- Vision encoder (multimodal / VLM) ----
    // Mirrors forge-cli: load the mmproj weights and create the registered
    // encoder. Absent or failed => text-only; requests carrying images are then
    // rejected at the API boundary with a clear error rather than silently
    // dropping the image. The final step -- splicing the encoded image tokens
    // into the transformer -- is engine work that lives outside this file.
    if (!opt.mmproj_path.empty()) {
        (void)forge::_vision_registrations_linked;
        (void)VisionEncoderRegistry::instance().registered_encoders();
        if (!model_->load_vision_weights(opt.mmproj_path, DeviceType::CPU)) {
            std::cerr << "Warning: failed to load vision encoder (mmproj): " << opt.mmproj_path
                      << "\n";
        } else {
            auto vloader = ModelLoaderRegistry::instance().create_loader(opt.mmproj_path);
            std::string vis_arch;
            VisionConfig vis_cfg;
            if (vloader && vloader->load(opt.mmproj_path)) {
                vis_arch = detect_vision_arch(*vloader);
                vis_cfg = VisionConfigParserRegistry::instance().parse(*vloader, vis_arch);
                vloader->close();
            }
            // The projector output dim is the LLM hidden dim, never the raw
            // mmproj metadata: forge-cli and the Python bindings override it
            // after parsing (see forge_cli.cpp / multimodal.cpp), but this path
            // did not, so Qwen3-VL requests failed the projection check with
            // "vision projection_dim (0) != model hidden_dim (2560)".
            vis_cfg.projection_dim = model_->config().hidden_dim;
            vision_ = VisionEncoderRegistry::instance().create(vis_arch).release();
            if (!vision_) {
                std::cerr << "Warning: no vision encoder registered for arch '" << vis_arch
                          << "'\n";
            } else if (!vision_->init(model_->weights(), vis_cfg)) {
                std::cerr << "Warning: vision encoder initialization failed\n";
                delete vision_;
                vision_ = nullptr;
            } else {
                std::cout << "Vision encoder loaded: arch=" << vis_arch << "\n";
            }
        }
    }

    // ---- Inference context ----
    auto t4 = std::chrono::high_resolution_clock::now();
    ctx_ = std::make_unique<InferenceContext>(*model_);
    if (opt.common.threads > 0) {
        ctx_->params_mut().n_threads = opt.common.threads;
        ctx_->params_mut().n_threads_batch = opt.common.threads;
        std::cout << "CPU threads set to: " << opt.common.threads << "\n";
    } else if (model_->device() != DeviceType::CUDA) {
        int t = detect_best_cpu_threads(cfg.hidden_dim, cfg.intermediate_dim);
        ctx_->params_mut().n_threads = t;
        ctx_->params_mut().n_threads_batch = t;
        std::cout << "CPU threads auto-detected: " << t << "\n";
    }

    if (opt.common.paged_kv) {
        ctx_->params_mut().kv_storage_mode = KVStorageMode::Paged;
        std::cout << "Paged KV cache storage enabled\n";
    }
    auto engine = EngineRegistry::instance().create(cfg.arch_type, *model_, *ctx_);
    if (!engine) {
        error = "No matching engine for architecture: " + cfg.arch_type;
        return false;
    }

    auto* tfm_eng = dynamic_cast<TransformerEngine*>(engine.get());
    // One shared application of the common flags, same as forge-cli. This also
    // fixes two things the local copy got wrong: it had no else on its dtype
    // ladder, so a typo'd --kv-cache-dtype silently became f16, and its thread
    // auto-tune keyed on "no GPU at all" rather than the hybrid-offload case, so
    // a partial -ngl run left the CPU-resident layers at n_threads=4 while the
    // GPU idled. It also picks up -ctk/-ctv, which this binary never had.
    {
        std::string err;
        if (!forge::apply_common_args(opt.common, nullptr, *ctx_, tfm_eng, &err)) {
            error = err;
            return false;
        }
    }
    if (tfm_eng && opt.common.cuda_graph && model_->device() == DeviceType::CUDA) {
        tfm_eng->set_use_graph(true);
        tfm_eng->set_cuda_graph_enabled(true);
    }

    // ---- KVMem tiered KV: cap the GPU page pool BEFORE init_kv_cache ----
    // Same ordering constraint as forge-cli: the manager is pure policy, but
    // the pool cap is read by PagedKVStorage::init() while sizing the page
    // pools, so it must be set before ctx_->init_kv_cache(). The manager is
    // attached right after, once the pool geometry is known.
    bool kvmem_active = false;
    if (opt.common.kvmem) {
        auto* kv_mem = tfm_eng ? tfm_eng->kv_memory() : nullptr;
        if (!tfm_eng || !kv_mem) {
            std::cerr << "Warning: --kvmem needs a transformer engine with KV memory; "
                         "tiered KV disabled\n";
        } else if (!kv_mem->is_paged()) {
            std::cerr << "Warning: --kvmem needs paged KV storage; tiered KV disabled\n";
        } else if (cfg.n_swa > 0) {
            // PagedKVStorage::kvmem_supported() rejects mixed policies: pages can
            // only be archived uniformly when every layer is Full-attention.
            std::cerr << "Warning: --kvmem needs all-Full attention layers; model has " << cfg.n_swa
                      << " sliding-window layer(s); tiered KV disabled\n";
        } else {
            // 0 = keep the device-sized pool; the budget then falls back to it.
            if (opt.common.kvmem_budget > 0)
                kv_mem->storage().set_pool_token_cap(opt.common.kvmem_budget);
            kvmem_active = true;
        }
    }

    ctx_->set_engine(std::move(engine));

    // Speculative decoding. The server never set this, so Generator's gate on
    // params().speculative_config.enabled was always false and the spec_stats
    // reporting further down was unreachable. Mirrors what cli_generation.cpp
    // does for the CLI.
    if (opt.spec.enabled) {
        ctx_->params_mut().speculative_config = opt.spec;
        std::cout << "Speculative decoding: arch="
                  << (opt.spec.draft_arch.empty() ? "n-gram" : opt.spec.draft_arch)
                  << ", n_draft=" << opt.spec.n_draft << ", draft_n=" << opt.spec.draft_n_spec
                  << "\n";
    }

    // offload_kqv and offload_embedding now come from the flags (they were
    // hard-coded here with no way to change them).
    ctx_->params_mut().offload_kqv = opt.common.offload_kqv;
    ctx_->params_mut().offload_embedding = model_->offload_embedding();
    if (!ctx_->init_kv_cache()) {
        error = "Failed to initialize KV cache";
        return false;
    }

    // Multi-level prefix residency knobs (paged mode only): write policy, host
    // blob pool budget and the L3 disk backend — --kv-* flags over the
    // FORGE_KV_* env vars. The server owns a separate PrefixCache instance from
    // the RequestScheduler path, so both wiring points need this.
    if (tfm_eng) {
        prefix_cache_enabled_ = tfm_eng->prefix_cache_supported();
        if (auto* kv_mem = tfm_eng->kv_memory()) {
            if (kv_mem->storage().is_paged())
                apply_prefix_cache_args(opt.common, prefix_cache_, kv_mem->storage());
        }
    }

    // Report the resolved KV geometry. With -ctk/-ctv the K and V types can
    // differ, and a line naming only one of them makes an asymmetric config
    // indistinguishable from a symmetric one -- which is how a silent dtype
    // downgrade stayed invisible in this binary for so long.
    if (tfm_eng) {
        auto* kv_mem = tfm_eng->kv_memory();
        const bool paged = kv_mem && kv_mem->storage().is_paged();
        KVCacheTypeConfig kv = opt.common.kv_cache_types();
        size_t bytes = 0;
        int max_seq = 0;
        if (paged) {
            const auto& st = static_cast<const PagedKVStorage&>(kv_mem->storage());
            bytes = st.nbytes();
            max_seq = st.max_seq_len();
        } else {
            const KVCache* cache = tfm_eng->kv_cache();
            bytes = cache->nbytes();
            max_seq = cache->max_seq_len();
        }
        const char* kn = kv_cache_dtype_name(kv.type_k);
        const char* vn = kv_cache_dtype_name(kv.type_v);
        std::cout << "KV Cache" << (paged ? " (paged)" : "") << ": dtype="
                  << (kv.type_k == kv.type_v ? kn : std::string(kn) + "/" + vn + " (K/V)")
                  << ", size=" << (bytes / (1024 * 1024)) << " MB, max_seq=" << max_seq << "\n";
    }

    // ---- Attach the KVMem working-set manager ----
    // After init_kv_cache: kvmem_supported() needs the page pools to exist and
    // the manager needs the real page_size / pool_tokens. From here the
    // generator reads window + rope metadata off it, and the request path
    // (below) hands history management over to the tier instead of doing its
    // own context shift.
    if (kvmem_active && tfm_eng) {
        auto* kv_mem = tfm_eng->kv_memory();
        if (!kv_mem->storage().kvmem_supported()) {
            std::cerr << "Warning: --kvmem unsupported by this KV storage "
                         "(mixed layer policies); tiered KV disabled\n";
        } else {
            const int page_size = kv_mem->storage().page_size();
            const int pool_tokens = kv_mem->max_pool_tokens();
            auto ws =
                std::make_unique<KVWorkingSetManager>(opt.kvem_config(), page_size, pool_tokens);
            std::cout << "KVMem tiered KV enabled: budget=" << ws->budget()
                      << ", sink=" << ws->sink_tokens() << ", gen_reserve=" << ws->gen_reserve()
                      << ", retrieval=" << opt.common.kvmem_retrieval << "\n"
                      << "  logical ctx=" << kv_mem->max_seq_len()
                      << " tokens, gpu pool=" << pool_tokens << " tokens, page=" << page_size
                      << "\n";
            ws->attach_storage(&kv_mem->storage(), kv_mem->storage().num_layers(), 0);
            kv_mem->set_working_set(std::move(ws));
        }
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
    // Native tool templates (MiniCPM/Qwen/GLM jinja chat templates with a
    // tools branch) render the catalogue and replay tool calls themselves;
    // the forge_text catalogue injection would teach a competing format.
    agent_cfg_.native_template_tools = chat_template_->supports_tools();

    // Model id = file name without directory / extension.
    auto slash = opt.common.model_path.find_last_of("/\\");
    std::string name = (slash == std::string::npos) ? opt.common.model_path
                                                    : opt.common.model_path.substr(slash + 1);
    auto dot = name.find_last_of('.');
    model_id_ = (dot == std::string::npos) ? name : name.substr(0, dot);
    return true;
}

ChatCompletionResult ServerState::chat_completion(
    const ChatCompletionRequest& req, const std::function<bool(const std::string& delta)>& on_token,
    const std::function<bool(const std::string& reasoning_delta)>& on_reasoning,
    MetricsRegistry::RequestScope* scope) {
    // Single model / KV cache instance: serialize inference across HTTP workers.
    std::lock_guard<std::mutex> lock(inference_mutex_);

    // Everything below runs under inference_mutex_, so this is the one safe
    // place to read KV occupancy for the dashboard. See engine.h's note.
    //
    // The message count is only a placeholder until the prompt is tokenized;
    // it is replaced by the real count just below (see AFTER_PROMPT_IDS).
    // `req.messages.size()` used to be the final value, which reported 1 for a
    // single-turn chat no matter how long the rendered prompt was.
    if (scope && scope->recording()) {
        scope->on_lock_acquired();
        scope->set_prompt_tokens(static_cast<int>(req.messages.size()));
    }

    // Per-request agent pipeline: tool catalogue injection (dialect-rendered),
    // streaming think-separation, and the final reasoning/tool-call split.
    // Cheap to construct and stateless across requests.
    AgentPipeline pipeline(agent_cfg_);

    // ---- Context-capacity-aware history truncation ----
    // The KV cache is capped for memory (e.g. 4096 for large-ctx models, see
    // TransformerEngine::init_kv_cache). A prompt longer than the cap would
    // overflow the cache mid-prefill and produce garbage logits (KVCache
    // refuses the write and attention reads stale cells). Mirror llama.cpp
    // server semantics: keep the system prompt and the most recent turns,
    // dropping oldest non-system messages until the rendered prompt plus the
    // requested generation budget fits.
    int kv_capacity = 0;
    // KVMem: with the tier attached, the working set — not the server — owns
    // history management for this request. The server steps back from the three
    // places it would otherwise fight the tier: the capacity bound below, the
    // structural context shift, and the paged radix-prefix turn hooks.
    bool tiered = false;
    if (auto* eng = ctx_->engine()) {
        if (auto* mem = eng->kv_memory())
            tiered = mem->tiered();
        // Paged mode: the pool is the real token capacity (it is fully
        // allocated up front) while the contiguous cache is never allocated —
        // reading kv_cache() here returned an uninitialized 0 and fell through
        // to the model's native max_seq_len (e.g. 262144), disabling the
        // truncation below entirely. Combine the pool bound with max_seq_len
        // like vLLM's min(max_model_len, num_gpu_blocks * block_size) check.
        // Tiered (KVMem): the pool is deliberately only the *working set*, so
        // binding capacity to it would reject every prompt longer than the
        // window — exactly the case the tier exists to serve. The logical bound
        // is max_seq_len; the working set drops/recalls history inside it.
        if (auto* mem = eng->kv_memory(); mem && mem->is_paged()) {
            kv_capacity =
                tiered ? mem->max_seq_len() : std::min(mem->max_seq_len(), mem->max_pool_tokens());
            LOG_WARN(
                "[capacity-check] paged storage: max_seq=" + std::to_string(mem->max_seq_len()) +
                " pool_tokens=" + std::to_string(mem->max_pool_tokens()) + " tiered=" +
                std::to_string(tiered ? 1 : 0) + " -> capacity=" + std::to_string(kv_capacity));
        } else if (auto* kvc = eng->kv_cache()) {
            kv_capacity = kvc->max_seq_len();
            LOG_WARN("[capacity-check] contiguous cache: capacity=" + std::to_string(kv_capacity));
        } else {
            LOG_WARN("[capacity-check] no kv_memory and no kv_cache");
        }
    }
    if (kv_capacity <= 0) {
        kv_capacity = model_->config().max_seq_len;
    }
    const int gen_budget = req.max_tokens > 0 ? req.max_tokens : 512;

    // ---- Tool injection (OpenAI function calling) ----
    // When the client offers tools (and does not forbid calling them), the
    // model must learn what it can invoke: either the jinja template renders
    // the catalogue natively (native_template_tools: tools_json feeds the
    // template's `tools` variable), or the configured dialect splices a
    // text protocol block into the system message.
    bool tools_active = false;
    std::string tools_json;
    if (agent_cfg_.native_template_tools && !req.tools.empty() && req.tool_choice != "none") {
        nlohmann::json arr = nlohmann::json::array();
        for (const auto& t : req.tools) {
            nlohmann::json params = nlohmann::json::parse(t.parameters_json, nullptr, false);
            if (params.is_discarded())
                params = nlohmann::json::object();
            arr.push_back({{"name", t.name},
                           {"description", t.description},
                           {"parameters", std::move(params)}});
        }
        tools_json = arr.dump();
        tools_active = true;
    }
    // ---- Conversation assembly ----
    // `messages` is the complete history; `delta_messages` appends to the session's
    // stored history. Either way the template sees the full conversation, so
    // prefix reuse and the rendered prompt are identical -- only what the client
    // had to send differs.
    //
    // Reached before the session is acquired below, so this uses a lookup that
    // does not create: a delta against a session that was never created (or was
    // evicted) would otherwise silently render a truncated conversation and answer
    // from a fraction of the real history.
    std::vector<ChatTemplateMessage> history;
    if (req.reset_session) {
        // Release the working set and forget the history; the request below then
        // rebuilds from whatever it carries.
        if (ctx_) {
            if (auto* eng = ctx_->engine()) {
                if (auto* mem = eng->kv_memory())
                    mem->release(acquire_session(req.session_id).seq_id);
            }
        }
        sessions_[req.session_id].messages.clear();
        sessions_[req.session_id].last_reply.clear();
        sessions_[req.session_id].last_reply_tool_calls.clear();
        sessions_[req.session_id].kv_tokens.clear();
    }
    if (!req.delta_messages.empty()) {
        auto it = sessions_.find(req.session_id);
        if (it == sessions_.end()) {
            throw ContextLengthExceededError(
                "Unknown or expired session_id '" + req.session_id +
                "': this server holds no stored history for it. Send the full "
                "'messages' array for the first turn of a conversation.");
        }
        history = it->second.messages;
        // The assistant turn the client omitted (it does not know our reply) is
        // re-inserted from the tokens we actually generated, so the conversation
        // stays alternating. A client that does send its own assistant message
        // overrides nothing -- we only add one when the last stored turn is not
        // already an assistant turn.
        if (!history.empty() && history.back().role != "assistant") {
            ChatTemplateMessage reply;
            reply.role = "assistant";
            reply.content = it->second.last_reply;
            // Carry the parsed calls too. `last_reply` is the raw generated text:
            // for a tool turn it is the call syntax itself, which the jinja
            // template would render as literal assistant prose. Replaying it
            // without `tool_calls` leaves the following tool result unpaired, and
            // templates that render <tool_call> via tool_call_id then show the
            // model a result with no call -- which it answers by calling the
            // tool again, forever.
            reply.tool_calls = it->second.last_reply_tool_calls;
            // Strip any leftover call syntax so it cannot be double-rendered as
            // text when the template does not use the tool_calls variable.
            if (!reply.tool_calls.empty() && !reply.content.empty())
                reply.content = strip_all_tool_syntax(reply.content);
            if (!reply.content.empty() || !reply.tool_calls.empty())
                history.push_back(std::move(reply));
        }
        history.insert(history.end(), req.delta_messages.begin(), req.delta_messages.end());
        it->second.messages = history;
    } else {
        history = req.messages;
    }

    std::vector<ChatTemplateMessage> tooled_messages =
        tools_json.empty()
            ? pipeline.inject_tools(history, req.tools, req.tool_choice, &tools_active)
            : history;

    ChatTemplateInput input;
    input.messages = tooled_messages;
    input.add_generation_prompt = true;
    input.tools_json = tools_json;
    std::vector<int32_t> prompt_ids = chat_template_->apply(input);

    // Replace the message-count placeholder with the real tokenised size, now
    // that the template has been rendered. The dashboard reads this from the
    // live snapshot, so during a decoding request it is what makes "prompt
    // tokens" meaningful instead of a constant 1.
    if (scope && scope->recording())
        scope->set_prompt_tokens(static_cast<int>(prompt_ids.size()));

    // ---- DEBUG (set FORGE_DEBUG_TEMPLATE=1 to enable) ----
    // Dumps the rendered chat-template prompt so we can see whether the VLM
    // template is bloating text-only turns with <|image_pad|> placeholders or
    // otherwise mis-rendering (which would starve generation / produce no answer).
    if (std::getenv("FORGE_DEBUG_TEMPLATE")) {
        std::string dbg_render = chat_template_->render(input);
        size_t img_pad = 0, p = 0;
        while ((p = dbg_render.find("image_pad", p)) != std::string::npos) {
            ++img_pad;
            p += 9;
        }
        size_t vision = 0;
        p = 0;
        while ((p = dbg_render.find("vision", p)) != std::string::npos) {
            ++vision;
            p += 6;
        }
        std::cerr << "[debug-template] messages=" << input.messages.size()
                  << " jinja=" << (chat_template_->uses_jinja() ? 1 : 0)
                  << " rendered_len=" << dbg_render.size() << " prompt_tokens=" << prompt_ids.size()
                  << " 'image_pad'=" << img_pad << " 'vision'=" << vision << "\n";
        for (size_t mi = 0; mi < input.messages.size(); ++mi) {
            std::cerr << "[debug-template]   msg[" << mi << "] role=" << input.messages[mi].role
                      << " content_len=" << input.messages[mi].content.size() << "\n";
        }
        std::string head = dbg_render.size() > 1600 ? dbg_render.substr(0, 1600) : dbg_render;
        std::cerr << "[debug-template] RENDER_HEAD:\n"
                  << head << "\n"
                  << "[debug-template] RENDER_TAIL:\n"
                  << (dbg_render.size() > 400 ? dbg_render.substr(dbg_render.size() - 400)
                                              : dbg_render)
                  << "\n";
    }

    // ---- Multimodal (vision) embedding injection ----
    // When a vision encoder is loaded (--mmproj) and the request carries images,
    // decode + encode each image and splice the resulting vision embeddings into
    // the token-embedding matrix at the template's <|image_pad|> placeholder
    // positions. The prefill then runs from those embeddings (Generator's
    // from_embedding path) so the model actually conditions on the image.
    // Text-only requests are completely unaffected.
    bool multimodal = false;
    TensorPtr mm_embeddings;
    {
        std::vector<std::vector<uint8_t>> all_images;
        for (const auto& m : tooled_messages)
            for (const auto& im : m.images) all_images.push_back(im);
        if (vision_ && !all_images.empty()) {
            // build_mm expands each image placeholder into the encoder's full
            // feature run, so prompt_ids (and thus the prefill token count)
            // grows to match the returned embedding matrix. The tiered working
            // set plans window slots from the UNexpanded ids, so the two paths
            // cannot be combined yet — reject explicitly instead of desyncing.
            if (tiered)
                throw std::runtime_error(
                    "multimodal (image) requests are not supported with --kvmem yet");
            // Build [prompt_len, H] embeddings = text token embeddings with each
            // image's vision embeddings spliced over its placeholder tokens.
            auto build_mm = [this](std::vector<int32_t>& pids,
                                   const std::vector<std::vector<uint8_t>>& imgs)
                                  -> TensorPtr {
                static const char* kPadCandidates[] = {
                    "<|image_pad|>", "<|IMAGE_PLACEHOLDER|>", "<image>",
                    "<|vision_pad|>", "<|img_pad|>", "<|IMAGE_TOKEN|>"};
                int pad_id = -1;
                for (const char* c : kPadCandidates) {
                    const int id = tokenizer_->token_to_id(c);
                    if (id >= 0) {
                        pad_id = id;
                        break;
                    }
                }
                if (pad_id < 0)
                    throw std::runtime_error(
                        "multimodal: vision encoder loaded but the tokenizer has no image "
                        "placeholder token (e.g. <|image_pad|>); cannot inject image "
                        "embeddings. The model's chat template must emit image placeholders.");
                const int H = model_->config().hidden_dim;
                const int proj = vision_->config().projection_dim;
                if (proj != H)
                    throw std::runtime_error(
                        "multimodal: vision projection_dim (" + std::to_string(proj) +
                        ") != model hidden_dim (" + std::to_string(H) + ")");

                std::vector<std::vector<float>> img_embs;
                std::vector<size_t> vis_counts;
                size_t total_vis = 0;
                for (const auto& img : imgs) {
                    int w = 0, h = 0;
                    std::vector<float> rgb;
                    if (!decode_image_rgb(img, w, h, rgb))
                        throw std::runtime_error(
                            "multimodal: failed to decode image (" +
                            std::to_string(img.size()) +
                            " bytes); the server decodes PNG/JPEG/BMP/TGA/GIF.");
                    std::vector<float> emb = vision_->encode(rgb.data(), w, h, 3);
                    const size_t nfeat = emb.size() / static_cast<size_t>(proj);
                    if (emb.size() != nfeat * static_cast<size_t>(proj))
                        throw std::runtime_error(
                            "multimodal: vision encoder produced an unexpected output size.");
                    img_embs.push_back(std::move(emb));
                    vis_counts.push_back(nfeat);
                    total_vis += nfeat;
                }

                size_t pad_count = 0;
                for (int t : pids)
                    if (t == pad_id) ++pad_count;
                if (pad_count != imgs.size())
                    throw std::runtime_error(
                        "multimodal: prompt has " + std::to_string(pad_count) +
                        " image placeholder tokens for " + std::to_string(imgs.size()) +
                        " image(s); expected exactly one <|image_pad|> per image "
                        "(the chat template must emit the vision placeholder).");

                // Expand placeholders: the template emits ONE <|image_pad|> per
                // image (Qwen-VL convention), while the encoder produces nfeat
                // feature rows per image. Expand each placeholder into nfeat
                // pad tokens (llama.cpp-style) so the splice below fills every
                // row and prompt_ids stays aligned with the returned embedding
                // matrix the prefill consumes. Must happen BEFORE the token
                // embedding lookup, whose row count follows pids.size().
                {
                    std::vector<int32_t> expanded;
                    expanded.reserve(pids.size() + total_vis - pad_count);
                    size_t img_idx = 0;
                    for (int t : pids) {
                        if (t == pad_id) {
                            for (size_t f = 0; f < vis_counts[img_idx]; ++f)
                                expanded.push_back(pad_id);
                            ++img_idx;
                        } else {
                            expanded.push_back(t);
                        }
                    }
                    pids = std::move(expanded);
                }

                auto token_emb = model_->weights().get("token_embedding");
                if (!token_emb)
                    throw std::runtime_error("multimodal: token_embedding weight not found");
                const int nt = static_cast<int>(pids.size());
                auto ids_tensor = std::make_shared<Tensor>(
                    DataType::INT32, std::vector<int64_t>{nt}, DeviceType::CPU);
                std::memcpy(ids_tensor->data(), pids.data(),
                            static_cast<size_t>(nt) * sizeof(int32_t));
                TensorPtr text_emb = ops::embedding(token_emb, ids_tensor);
                text_emb->to_device(DeviceType::CPU);

                float* out = static_cast<float*>(text_emb->data());
                size_t img_idx = 0, feat_idx = 0;
                for (int t : pids) {
                    if (t == pad_id) {
                        const float* src =
                            img_embs[img_idx].data() + feat_idx * static_cast<size_t>(H);
                        std::memcpy(out, src, static_cast<size_t>(H) * sizeof(float));
                        if (++feat_idx >= vis_counts[img_idx]) {
                            feat_idx = 0;
                            ++img_idx;
                        }
                    }
                    out += H;
                }
                return text_emb;
            };
            mm_embeddings = build_mm(prompt_ids, all_images);
            multimodal = true;
            std::cout << "[server] multimodal: injected " << all_images.size()
                      << " image(s) through the vision encoder\n";
        }
    }

    // Tiered (KVMem): skip the structural context shift. The working set
    // already decides, per turn, which prefix stays resident (its sink) and
    // which stale history is archived/recalled, and it does so in *logical*
    // space with the rope remap to match. Shifting the message list here would
    // rewrite the prompt under the manager's feet — its common-prefix match
    // would diverge every turn and the resident window would be recomputed
    // from scratch, which is the opposite of what the tier is for.
    if (!multimodal && !tiered && kv_capacity > 0 &&
        static_cast<int>(prompt_ids.size()) + gen_budget > kv_capacity &&
        tooled_messages.size() > 2) {
        // ---- Context shift (llama.cpp `n_keep` semantics) ----
        // Keep the leading prefix fixed and discard the oldest history until the
        // rendered prompt plus the generation budget fits the KV cache. This is a
        // pure structural shift (no semantic summary): the surviving prefix is
        // re-attended on the next turn via the existing session prefix-reuse
        // (vLLM-style prefix caching). Mirrors llama.cpp's token-level context
        // shift + n_keep and vLLM's eviction of out-of-window tokens.
        const bool has_system = !tooled_messages.empty() && tooled_messages[0].role == "system";

        // Protected prefix boundary: the first `n_keep` tokens are never evicted
        // (llama.cpp semantics). n_keep<=0 protects only the system prompt.
        int protected_idx = has_system ? 1 : 0;
        if (opt_.n_keep > 0) {
            int cum = 0;
            for (size_t i = 0; i < tooled_messages.size(); ++i) {
                ChatTemplateInput one;
                one.messages = {tooled_messages[i]};
                one.add_generation_prompt = false;
                cum += static_cast<int>(chat_template_->apply(one).size());
                if (cum >= opt_.n_keep) {
                    protected_idx = static_cast<int>(i) + 1;
                    break;
                }
            }
        }
        const size_t first_drop_idx = static_cast<size_t>(protected_idx);
        const size_t last_keep_idx = tooled_messages.size() - 1;  // always keep final message

        std::vector<ChatTemplateMessage> kept(tooled_messages.begin(),
                                              tooled_messages.begin() + first_drop_idx);

        auto render = [&](const std::vector<ChatTemplateMessage>& msgs) {
            ChatTemplateInput inp;
            inp.messages = msgs;
            inp.add_generation_prompt = true;
            inp.tools_json = tools_json;  // keep the native catalogue on re-render
            return chat_template_->apply(inp);
        };
        // Token-granular trim: keep the most-recent `keep` tokens of a boundary
        // message's content (llama.cpp discards the *oldest* tokens, so we keep
        // its tail) instead of dropping the whole message.
        auto trim_tail = [&](const ChatTemplateMessage& m, int keep) -> std::string {
            std::vector<int32_t> toks = tokenizer_->encode(m.content, false, false, false);
            if (static_cast<int>(toks.size()) <= keep)
                return m.content;
            std::vector<int32_t> tail(toks.end() - keep, toks.end());
            return tokenizer_->decode(tail, false, false);
        };

        size_t drop_upto = first_drop_idx;
        bool fits = false;
        bool trimmed = false;
        while (drop_upto <= last_keep_idx) {
            std::vector<ChatTemplateMessage> candidate = kept;
            candidate.insert(candidate.end(), tooled_messages.begin() + drop_upto,
                             tooled_messages.end());
            auto ids = render(candidate);
            if (static_cast<int>(ids.size()) + gen_budget <= kv_capacity) {
                prompt_ids = std::move(ids);
                fits = true;
                break;
            }
            // Whole-message drop still overflows. Before discarding message
            // `drop_upto` entirely, try trimming its tail to free just enough
            // (preserves more of the boundary turn than a wholesale drop).
            if (drop_upto < last_keep_idx) {
                const auto& m = tooled_messages[drop_upto];
                int overflow = static_cast<int>(ids.size()) + gen_budget - kv_capacity;
                int m_tok =
                    static_cast<int>(tokenizer_->encode(m.content, false, false, false).size());
                int keep = m_tok - overflow;
                if (keep > 0) {
                    ChatTemplateMessage tm = m;
                    tm.content = trim_tail(m, keep);
                    std::vector<ChatTemplateMessage> cand2 = kept;
                    cand2.push_back(std::move(tm));
                    cand2.insert(cand2.end(), tooled_messages.begin() + drop_upto + 1,
                                 tooled_messages.end());
                    auto ids2 = render(cand2);
                    if (static_cast<int>(ids2.size()) + gen_budget <= kv_capacity) {
                        prompt_ids = std::move(ids2);
                        fits = true;
                        trimmed = true;
                        break;
                    }
                }
            }
            ++drop_upto;
        }

        if (fits) {
            std::cout << "[server] context shifted to fit KV capacity " << kv_capacity << " (kept "
                      << first_drop_idx << " prefix message(s)"
                      << (trimmed ? ", token-trimmed 1 boundary message" : "") << ", dropped "
                      << (drop_upto - first_drop_idx) << " oldest history message(s))\n";
        } else {
            // Even the minimal message set does not fit. Fail loudly instead of
            // corrupting the KV cache.
            throw ContextLengthExceededError(
                "prompt exceeds model context capacity (" + std::to_string(prompt_ids.size()) +
                " tokens > " + std::to_string(kv_capacity - gen_budget) +
                " available after reserving " + std::to_string(gen_budget) + " generation tokens)");
        }
    }

    // vLLM-style hard admission check: a prompt that still exceeds capacity
    // after optional history truncation is rejected up front. Without this,
    // single oversized messages (message count <= 2 never enters the context
    // shift above — e.g. opencode pasting one large file) run the whole
    // prefill and die mid-write on page-pool exhaustion (a 500 after burning
    // the full prefill compute) instead of an immediate 400.
    // Tiered (KVMem): the working set owns history truncation and generation
    // capping, so the server must NOT reject on the raw prompt length here —
    // doing so would block every long prompt the tier exists to serve. The
    // generator caps decode at plan.max_generate, so the pool can never overflow.
    if (!tiered && kv_capacity > 0 &&
        static_cast<int>(prompt_ids.size()) + gen_budget > kv_capacity) {
        LOG_WARN("[capacity-check] REJECTING: prompt=" + std::to_string(prompt_ids.size()) +
                 " budget=" + std::to_string(gen_budget) +
                 " capacity=" + std::to_string(kv_capacity));
        throw ContextLengthExceededError(
            "This model's maximum context length is " + std::to_string(kv_capacity) +
            " tokens, but the request is " +
            std::to_string(static_cast<int>(prompt_ids.size()) + gen_budget) + " tokens (prompt " +
            std::to_string(prompt_ids.size()) + " + generation " + std::to_string(gen_budget) +
            ")");
    }

    // OpenAI default: temperature=0 means greedy; other defaults are the
    // anti-degeneration choices in ChatCompletionRequest (repeat penalty etc.).
    // --perf-detail: open a fresh profiling window for this request. The
    // dashboard's operator panel reads the same window, so it needs the reset
    // too -- otherwise the "per-request" breakdown would accumulate across
    // every request since start-up.
    // The dashboard's operator panel needs an explicit opt-in (--dashboard-ops):
    // arming PerfProfiler around every request would slow decode for a chart the
    // user may not even have open. --perf-detail has always paid that cost
    // because it is a diagnostic flag.
    const bool want_operator_panel =
        opt_.perf_detail || (opt_.dashboard.capture_operators && scope && scope->recording());
    if (want_operator_panel) {
        PerfProfiler::instance().reset();
        // Recording is opt-in and not free (a mutex-guarded accumulator update
        // per scope), so only arm it when something will actually read it.
        PerfProfiler::instance().enable();
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
    // Chat turn terminator: MiniCPM/Qwen-style templates end an assistant turn
    // with <|im_end|> while the vocab eos is a different id (here eos=1).
    // Stop at the first turn terminator (llama.cpp/vLLM derive this from the
    // chat template), so trailing control tokens never leak into the reply.
    if (int32_t turn_end = tokenizer_->token_to_id("<|im_end|>");
        turn_end >= 0 && turn_end != tokenizer_->unk_token_id() &&
        turn_end != gen_cfg.eos_token_id) {
        gen_cfg.stop_token_ids.push_back(turn_end);
    }

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
        // Dashboard TTFT / inter-token latency. Called before the early returns
        // below so EOS and stop tokens still count as emitted steps: the time
        // to produce them was spent, and the client's first byte did not wait
        // for them.
        if (scope && scope->recording())
            scope->on_token();
        if (tok == gen_cfg.eos_token_id) {
            // EOS is invoked by generate() before stopping; decode_token() does
            // not skip special tokens, so emit nothing for it (the non-stream
            // path already filters via decode(skip_special=true)).
            return;
        }
        // Stop tokens reach this callback before generate() acts on them
        // (generator.cpp pushes the token and fires the callback first), so
        // their pieces would otherwise be streamed verbatim.
        for (int32_t sid : gen_cfg.stop_token_ids) {
            if (tok == sid)
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
            // Agent pipeline: split the delta into visible content and
            // <think> reasoning before it reaches the client callbacks.
            std::string reasoning_delta;
            std::string visible =
                pipeline.process_delta(delta, on_reasoning ? &reasoning_delta : nullptr);
            if (on_reasoning && !reasoning_delta.empty() && !on_reasoning(reasoning_delta)) {
                // Client disconnected: stop burning compute on this request.
                abort_flag.store(true, std::memory_order_relaxed);
            }
            if (!visible.empty() && on_token && !on_token(visible)) {
                // Client disconnected: stop burning compute on this request.
                abort_flag.store(true, std::memory_order_relaxed);
            }
        }
        // Keep [emit, ok) in `pending` (stop-string hold window); the byte
        // range beyond `ok` stays anyway (incomplete UTF-8 sequence).
        pending.erase(0, emit);
    };

    // ---- Session prefix reuse ----
    // Two storage backends, two reuse strategies:
    //  * Contiguous: a single-entry mirror of the last prompt+output; on a
    //    matching prefix we forward the tail and let generate() rollback().
    //  * Paged: the radix PrefixCache (same machinery as the Python
    //    RequestScheduler). Each request's working set is seq_id 0; cached
    //    prefixes live in dedicated cache seq_ids whose KV pages are
    //    reference-counted. On a hit, try_lookup() shares the cached prefix
    //    pages into seq 0 via one seq_share (ref_count++) and we skip prefilling
    //    those tokens. We must NOT call ctx_.reset_kv_cache() — that resets the
    //    whole KV cache, wiping the radix cache's cache seq_ids — so instead we
    //    release the previous request's working set (freeing only its unique
    //    suffix pages; cached prefix pages survive via their cache seq_ids).
    //    This mirrors vLLM's free(request) + per-block ref_count eviction.
    int prefix_len = 0;
    const bool paged_mode = (ctx_->params().kv_storage_mode == KVStorageMode::Paged);
    // Per-session state: which seq_id this conversation's working set lives in,
    // and the token mirror the contiguous backend compares against. A request
    // without session_id lands on the default session, which is seq 0 -- exactly
    // what every request used before sessions existed, so old clients behave
    // identically.
    SessionState& sess = acquire_session(req.session_id);
    const int sess_seq = sess.seq_id;
    // The contiguous mirror is per session, so a second conversation cannot
    // shorten the prefix the first one is about to match against.
    std::vector<int32_t>& cached_kv_tokens = sess.kv_tokens;
    if (paged_mode && tiered) {
        // ---- KVMem turn boundary ----
        // The working set is a single-conversation policy object bound to seq 0,
        // so try_acquire_slot() caps admission at one request while it is
        // attached: only the default session can make progress here, and a second
        // conversation would interleave prompts through one on_prompt() state.
        // Per-session KVMem needs one manager per session -- follow-up work.
        //
        // seq 0's resident window IS the conversation state, tracked by the
        // working-set manager across turns. Releasing it, or sharing its pages
        // into radix cache seq_ids, would invalidate the window layout that
        // on_prompt() plans against — so nothing is released or looked up here.
        // The manager computes the common prefix in logical space and the
        // generator forwards only the non-resident tail. The free-page
        // admission check below is skipped for the same reason: it measures the
        // whole prompt against the *pool*, while the tier only ever needs the
        // working set's worth of pages.
        gen_cfg.prefix_len = 0;
        gen_cfg.reset_kv_cache = false;
    } else if (paged_mode) {
        if (auto* km = ctx_->engine()->kv_memory()) {
            // Drop the previous request's radix-node reference and working set
            // so seq 0 starts empty (register_prefix() may have left a trailing
            // unaligned page). The RequestScheduler calls release_prefix() when
            // a request finishes; the server path must do it here — otherwise
            // the matched node keeps ref_count >= 1 forever and its pages can
            // never be evicted.
            prefix_cache_.release_prefix(sess_seq, km->storage());
            km->storage().release(sess_seq);
            prefix_len = (multimodal || !prefix_cache_enabled_)
                             ? 0
                             : prefix_cache_.try_lookup(prompt_ids, sess_seq, km->storage());
            // vLLM-style on-demand eviction before prefill. Cached prefixes
            // hold pages after register_prefix(); a new sequence needs
            // ceil((prompt_tail + gen_budget) / page_size) fresh pages in every
            // Full-attention pool. When free pages run short, evict unused
            // cached prefixes (LRU) first; refuse only when even a fully
            // evicted pool cannot hold the sequence — before any prefill work,
            // so the client gets a clean 400 instead of a mid-prefill 500.
            const int page = km->storage().page_size();
            const long tail = (long)prompt_ids.size() - prefix_len;
            const int need_pages = (int)((tail + gen_budget + page - 1) / page);
            const int free_pages = km->free_pages_full();
            if (free_pages >= 0 && free_pages < need_pages) {
                prefix_cache_.evict_lru(0, km->storage());
                const int free_after = km->free_pages_full();
                if (free_after < need_pages) {
                    // The rejection is driven by *free* pages, not the pool's
                    // total capacity — the cached prefix and prior working sets
                    // already occupy most of the pool. Report both so the user
                    // understands it is a fragmentation/occupancy problem, not a
                    // raw-capacity one.
                    const long free_tokens = (long)free_after * page;
                    throw ContextLengthExceededError(
                        "This model's paged KV pool holds " +
                        std::to_string(km->max_pool_tokens()) + " tokens total, but only " +
                        std::to_string(free_tokens) +
                        " are free after evicting cached prefixes; the request "
                        "needs " +
                        std::to_string(tail + gen_budget) + " new tokens (prompt " +
                        std::to_string(prompt_ids.size()) + " - cached prefix " +
                        std::to_string(prefix_len) + " + max_tokens " + std::to_string(gen_budget) +
                        "). Reduce the input length, lower max_tokens, or enlarge "
                        "the KV pool (free VRAM for the 741MB Q3K scratch via a "
                        "tensor-core-eligible quant, or raise --max-num-seqs).");
                }
            }
        }
        gen_cfg.prefix_len = prefix_len;
        // Keep reset_kv_cache false: seq 0 was cleared above; a true reset would
        // also nuke the radix cache. On a miss seq 0 is already empty.
        gen_cfg.reset_kv_cache = false;
    } else {
        // Contiguous single-entry mirror (behavior unchanged).
        if (!cached_kv_tokens.empty()) {
            const size_t lim = std::min(cached_kv_tokens.size(), prompt_ids.size());
            size_t m = 0;
            while (m < lim && cached_kv_tokens[m] == prompt_ids[m]) {
                ++m;
            }
            prefix_len = static_cast<int>(m);
        }
        gen_cfg.prefix_len = prefix_len;
        gen_cfg.reset_kv_cache = (prefix_len == 0);
        if (prefix_len == 0) {
            // generate() resets the KV cache on this path; keep the mirror honest
            // so a later request cannot reuse rows that no longer exist.
            cached_kv_tokens.clear();
        }
    }
    if (opt_.log_prompt_prefix) {
        // Diagnostic: emit FNV-1a hashes of the leading prompt at several depths.
        // Diff consecutive turns — the deepest depth whose hash is identical
        // across turns marks the stable prefix; the next depth is where the
        // agent's prompt starts to diverge (the cache-break point).
        const int depths[] = {64, 256, 1024, 4096, 8192, 16384};
        std::string hs;
        for (int d : depths) {
            const size_t n = std::min((size_t)d, prompt_ids.size());
            uint32_t h = 2166136261u;
            for (size_t k = 0; k < n; ++k) {
                h ^= static_cast<uint32_t>(prompt_ids[k]);
                h *= 16777619u;
            }
            hs += " " + std::to_string(n) + ":" + std::to_string(h);
        }
        LOG_WARN("[prompt-prefix] len=" + std::to_string(prompt_ids.size()) +
                 " hit=" + std::to_string(prefix_len) + " hashes" + hs);
    }
    if (opt_.perf && prefix_len > 0) {
        std::cout << "[server] prefix reuse: " << prefix_len << "/" << prompt_ids.size()
                  << " tokens from cache (" << (prompt_ids.size() - prefix_len) << " to prefill)\n";
    }
    if (scope && scope->recording())
        scope->set_prefix_reused(prefix_len);

    GenerationResult result;
    try {
        Generator gen(*ctx_, sampler_cfg);
        result = gen.generate(prompt_ids, gen_cfg, cb,
                              multimodal ? mm_embeddings : nullptr);
    } catch (...) {
        // A failed request can leave the KV cache partially overwritten, so the
        // recorded prefix no longer describes what is resident. Drop the mirror
        // and let the next request start from a clean cache. In paged mode also
        // release seq 0's working set (its prefix pages are shared with the radix
        // cache, which survives) so the next request does not append onto stale KV.
        // Tiered: seq 0 holds the tier's window, which must survive the failed
        // request for the next turn's plan to match against.
        if (paged_mode && !tiered) {
            if (auto* km = ctx_->engine()->kv_memory()) {
                km->storage().release(sess_seq);
            }
        }
        cached_kv_tokens.clear();
        throw;
    }

    // Remember the conversation and our reply so the next delta request can be
    // assembled without the client resending them. For a full-history request the
    // client is authoritative, so only the reply is recorded.
    sess.messages = history;
    sess.last_reply = result.text;

    // Record what the KV cache now holds, per backend.
    if (paged_mode) {
        // Register this request's full prompt KV into the radix cache so future
        // requests can reuse it, then evict LRU leaf nodes beyond capacity.
        // register_prefix() transfers seq 0's prefix pages into per-node cache
        // seq_ids (ref-counted, no copy) and releases seq 0's own reference; the
        // cached copy survives. Short prompts are handled internally (no-op).
        // Tiered: skipped — the pages must stay in seq 0's window, which is the
        // tier's resident state for the next turn.
        if (!tiered && prefix_cache_enabled_) {
            if (auto* km = ctx_->engine()->kv_memory()) {
                // Register the prompt AND the reply, not just the prompt. Decode
                // writes result.token_ids into seq 0 starting at prompt_ids.size(),
                // so that whole span is resident; the next turn's prompt starts with
                // exactly that sequence. Registering only the prompt left the reply's
                // KV unreferenced, and the release(0) at the top of the next request
                // threw it away -- so every turn re-prefilled the entire previous
                // assistant message. The contiguous branch below already did this.
                std::vector<int32_t> resident(prompt_ids);
                resident.insert(resident.end(), result.token_ids.begin(), result.token_ids.end());
                prefix_cache_.register_prefix(resident, sess_seq, km->storage());
                prefix_cache_.evict_lru(PrefixCache::DEFAULT_MAX_ENTRIES, km->storage());
            }
        }
    } else {
        // Contiguous single-entry mirror.
        cached_kv_tokens = prompt_ids;
        cached_kv_tokens.insert(cached_kv_tokens.end(), result.token_ids.begin(),
                                result.token_ids.end());
        if (static_cast<int>(cached_kv_tokens.size()) + gen_budget > kv_capacity) {
            cached_kv_tokens.clear();
        }
    }

    if (!stop_hit && !pending.empty()) {
        // Flush any trailing partial sequence through the same agent-pipeline
        // split as the regular streaming deltas.
        std::string reasoning_delta;
        std::string visible =
            pipeline.process_delta(pending, on_reasoning ? &reasoning_delta : nullptr);
        if (on_reasoning && !reasoning_delta.empty())
            (void)on_reasoning(reasoning_delta);
        if (on_token && !visible.empty())
            (void)on_token(visible);  // client-disconnect return is moot here
    }

    ChatCompletionResult out;
    // skip_special=false: MiniCPM5-style vocabularies encode the tool-call
    // syntax itself as CONTROL-type special tokens (function-open/param-open/
    // close tags). Decoding with skip_special=true strips those tags and
    // leaves mangled attribute/CDATA fragments that no tool-call dialect can
    // parse, which then leak into the visible text. Keep the literals, then
    // cut the turn at the first turn-terminator or EOS literal.
    out.text = tokenizer_->decode(result.token_ids, false, false);
    {
        const char* const kTurnEnds[] = {
            "<|"
            "im_end"
            "|>",
            "<|"
            "im_start"
            "|>",
            "</"
            "s"
            ">",
            "<|"
            "end"
            "|>",
            "<|"
            "eot_id"
            "|>"};
        size_t cut = std::string::npos;
        for (const char* s : kTurnEnds) {
            const size_t p = out.text.find(s);
            if (p != std::string::npos && p < cut)
                cut = p;
        }
        if (cut != std::string::npos)
            out.text.resize(cut);
    }

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

    // ---- Agent post-processing: reasoning separation + tool-call extraction ----
    // Split <think>...</think> into out.reasoning and, when tools were
    // offered, parse the dialect's call blocks into OpenAI-shaped tool_calls
    // (stripped from the assistant text).
    {
        std::vector<std::string> tool_names;
        if (tools_active) {
            tool_names.reserve(req.tools.size());
            for (const auto& t : req.tools)
                tool_names.push_back(t.name);
        }
        auto fin = pipeline.finalize(out.text, tools_active ? &tool_names : nullptr);
        out.text = std::move(fin.text);
        out.reasoning = std::move(fin.reasoning);
        if (tools_active)
            out.tool_calls = std::move(fin.tool_calls);
    }

    // Record the parsed calls for the next delta turn's assistant replay (see
    // SessionState::last_reply_tool_calls). Must happen after the finalize
    // above, which is where the calls are extracted.
    sess.last_reply_tool_calls.clear();
    for (const auto& tc : out.tool_calls)
        sess.last_reply_tool_calls.push_back(ChatTemplateToolCall{tc.id, tc.name, tc.arguments});

    out.prompt_tokens = result.num_prompt_tokens;
    out.completion_tokens = result.num_generated_tokens;
    // Timings for the client's per-turn perf line (the same numbers --perf
    // prints); zero when the generator did not time the request.
    out.prefill_ms = result.prefill_ms;
    out.decode_ms = result.decode_ms;

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

    if (!out.tool_calls.empty()) {
        out.finish_reason = "tool_calls";
    }

    // ---- PerfProfiler window: close it EXACTLY ONCE per request ----
    //
    // PerfProfiler::summary() is not idempotent -- it merges every thread-local
    // accumulator into the global records_ on each call -- so calling it twice
    // for one request doubles every number (the same trap forge-cli's bench path
    // documents). Both consumers therefore share ONE capture here: the dashboard
    // stores the copy, --perf-detail prints from it.
    std::vector<std::pair<std::string, PerfRecord>> perf_rows;
    if (opt_.perf_detail || (scope && scope->recording())) {
        // flush_deferred merges CUDA-event timings that were recorded
        // asynchronously during the request.
#ifdef USE_CUDA
        PerfProfiler::instance().flush_deferred();
#endif
        perf_rows = PerfProfiler::instance().summary();
        std::sort(perf_rows.begin(), perf_rows.end(),
                  [](const std::pair<std::string, PerfRecord>& a,
                     const std::pair<std::string, PerfRecord>& b) {
                      return a.second.total_ms > b.second.total_ms;
                  });
    }
    if (scope && scope->recording()) {
        std::vector<OperatorRecord> ops;
        ops.reserve(perf_rows.size());
        for (const auto& [name, r] : perf_rows) {
            OperatorRecord o;
            o.name = name;
            o.total_ms = r.total_ms;
            o.count = r.count;
            o.min_ms = r.min_ms;
            o.max_ms = r.max_ms;
            o.avg_ms = r.count > 0 ? r.total_ms / static_cast<double>(r.count) : 0.0;
            ops.push_back(std::move(o));
        }
        scope->set_operators(std::move(ops));
        scope->finish(
            result, result.finish_reason == "aborted" ? RequestStatus::Aborted : RequestStatus::Ok);
        publish_engine_gauges();
    }

    // ---- --perf / --perf-detail report (stdout, after every completion) ----
    if (opt_.perf) {
        const int gen_tok = result.num_generated_tokens;
        const double prefill_ms = result.prefill_ms;
        const double decode_ms = result.decode_ms;
        std::cout << "[perf] prompt=" << out.prompt_tokens << " tok | gen=" << gen_tok << " tok"
                  << " | prefill=" << std::fixed << std::setprecision(1) << prefill_ms << " ms ("
                  << std::setprecision(2) << result.prefill_ms_per_1k() << " ms/1k-tok)"
                  << " | decode=" << std::setprecision(1) << decode_ms << " ms"
                  << " | " << std::setprecision(2) << result.tokens_per_sec() << " tok/s"
                  << " | finish=" << out.finish_reason << std::endl;
        if (result.spec_stats) {
            std::cout << "[perf] speculative: accepted=" << result.spec_stats->n_accepted_tokens
                      << " drafted=" << result.spec_stats->n_draft_tokens
                      << " (rate=" << std::setprecision(1)
                      << 100.0 * result.spec_stats->acceptance_rate() << "%)\n";
        }
        if (opt_.perf_detail) {
            // Reads the single capture taken above; must NOT call summary() again.
            const size_t top_n = 12;
            std::cout << "[perf] top operators (this request):\n";
            if (perf_rows.empty()) {
                std::cout << "  (no operator data - rebuild with "
                             "FORGE_PROFILING=ON for --perf-detail)\n";
            }
            for (size_t i = 0; i < perf_rows.size() && i < top_n; ++i) {
                const auto& [rname, rec] = perf_rows[i];
                const double avg =
                    rec.count > 0 ? rec.total_ms / static_cast<double>(rec.count) : 0.0;
                std::cout << "  " << std::left << std::setw(38) << rname << std::right
                          << " n=" << std::setw(6) << rec.count << "  total=" << std::fixed
                          << std::setprecision(2) << std::setw(9) << rec.total_ms << " ms"
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

// ============================================================================
// Session state
// ============================================================================
namespace {
// Session working-set seq ids start well above PrefixCache::CACHE_SEQ_ID_BASE
// (1e6) and its successors, so the two allocators can never hand out the same id.
constexpr int kSessionSeqBase = 1;
}  // namespace

ServerState::SessionState& ServerState::acquire_session(const std::string& session_id) {
    const int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now().time_since_epoch())
                            .count();

    auto it = sessions_.find(session_id);
    if (it != sessions_.end()) {
        it->second.last_active_ms = now;
        ++it->second.use_count;
        return it->second;
    }

    // Full: retire the least-recently-used session. Without this, a client that
    // sends a fresh id per request (or an attacker) would pin every page in the
    // pool, since each session's working set holds pages until it is released.
    if (static_cast<int>(sessions_.size()) >= kMaxSessions) {
        std::string victim;
        int64_t oldest = now;
        bool have = false;
        for (const auto& kv : sessions_) {
            if (!have || kv.second.last_active_ms < oldest) {
                oldest = kv.second.last_active_ms;
                victim = kv.first;
                have = true;
            }
        }
        if (have) {
            drop_session(victim, /*release_kv=*/true);
        }
    }

    SessionState st;
    // seq 0 is reserved for the default session so that a request without a
    // session_id keeps hitting exactly the pages it did before sessions existed.
    st.seq_id = session_id.empty() ? 0 : kSessionSeqBase + static_cast<int>(next_session_seq_++);
    st.last_active_ms = now;
    st.use_count = 1;
    auto res = sessions_.emplace(session_id, std::move(st));
    return res.first->second;
}

void ServerState::drop_session(const std::string& session_id, bool release_kv) {
    auto it = sessions_.find(session_id);
    if (it == sessions_.end())
        return;
    if (release_kv) {
        if (auto* eng = ctx_ ? ctx_->engine() : nullptr) {
            if (auto* mem = eng->kv_memory())
                mem->release(it->second.seq_id);
        }
    }
    sessions_.erase(it);
}

bool ServerState::tiered_kv_active() const {
    const InferenceContext* ctx = ctx_.get();
    if (!ctx)
        return false;
    const InferenceEngine* eng = ctx->engine();
    if (!eng)
        return false;
    const KVMemory* mem = eng->kv_memory();
    return mem != nullptr && mem->working_set() != nullptr;
}

bool ServerState::try_acquire_slot() {
    int cur = active_requests_.load(std::memory_order_relaxed);
    int cap = opt_.max_inflight > 0 ? opt_.max_inflight : 1;
    // KVMem's working set is a single-conversation policy object: logical_hist_,
    // the segment table and archived_chunks_ are per-instance, not per-seq, and
    // the class documents itself as "not thread-safe: one conversation at a time".
    // The whole engine is already serialised by inference_mutex_, but queueing two
    // requests would still interleave their prompts through one on_prompt() state
    // and silently corrupt the window layout. Admit exactly one.
    if (tiered_kv_active())
        cap = 1;
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

// ============================================================================
// Dashboard gauges
// ============================================================================

std::string ServerState::arch() const {
    return model_ ? model_->config().arch_type : std::string();
}
//
// publish_engine_gauges() is the ONLY function that reads KV occupancy, and it
// must be called from the thread holding inference_mutex_.
//
// The reason is specific to the contiguous backend: KVCache::active_bytes()
// (kv_cache.cpp) walks every layer's logical_filled counter and
// KVCache::num_free_slots() iterates the per-layer cell tables looking at
// is_free() -- neither takes a lock, so a concurrent reader from the dashboard
// thread is a data race, not merely a stale value. PagedKVStorage's equivalents
// DO lock, which is exactly the kind of asymmetry that makes "just poll it from
// the HTTP handler" look correct on a paged build and corrupt on a contiguous
// one.
//
// So the inference thread publishes an atomic snapshot at the end of every
// request, and the dashboard thread only ever reads that snapshot. Between
// requests the numbers do not change anyway -- nothing else mutates the KV
// cache -- so a per-request cadence is exactly as fresh as the data allows.

EngineGauges ServerState::engine_gauges() const {
    // Reads the last published snapshot, never prefix_cache_ directly: those
    // counters are only written under inference_mutex_, and taking that lock
    // from a const accessor would make the dashboard's 1 Hz poll contend with
    // real inference.
    return MetricsRegistry::instance().engine_gauges();
}

void ServerState::fill_engine_gauges(EngineGauges& g) {
    // Caller must hold inference_mutex_.
    g.sessions = session_count();
    g.tiered_kv = tiered_kv_active();
    g.prefix_cache_hits = prefix_cache_.hits();
    g.prefix_cache_misses = prefix_cache_.misses();
    g.prefix_cache_host_hits = prefix_cache_.host_hits();
    if (auto* eng = ctx_ ? ctx_->engine() : nullptr) {
        if (auto* mem = eng->kv_memory()) {
            g.paged_kv = mem->is_paged();
            auto& st = mem->storage();
            g.kv_total_bytes = static_cast<int64_t>(st.nbytes());
            g.kv_active_bytes = static_cast<int64_t>(st.active_bytes());
            g.kv_free_pages = st.is_paged() ? st.num_free_pages() : -1;
            g.kv_pool_tokens = mem->is_paged() ? mem->max_pool_tokens() : 0;
            g.kv_page_size = st.page_size();
        }
    }
}

void ServerState::publish_engine_gauges() {
    if (!MetricsRegistry::instance().enabled())
        return;
    EngineGauges g;
    // active_requests is deliberately left at 0 here: MetricsRegistry overwrites
    // it from the live RequestScope count, so setting it would be misleading
    // (this runs before the scope that is about to be created).
    fill_engine_gauges(g);
    MetricsRegistry::instance().publish_engine_gauges(g);
}

}  // namespace forge
