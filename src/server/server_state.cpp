/**
 * forge-server - model loading + chat generation
 *
 * The load pipeline mirrors forge-cli's text-only path (forge_cli.cpp):
 * tokenizer -> model -> engine -> KV cache -> CUDA warmup.
 */

#include "server.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "forge/logger.h"

#include <nlohmann/json.hpp>

#include "forge/arch_registry.h"
#include "forge/engines/transformer_engine.h"
#include "forge/gguf_model.h"
#include "forge/kv_memory.h"
#include "forge/model_loader.h"
#include "forge/ninf_model.h"
#include "forge/perf_profiler.h"
#include "forge/prefix_cache.h"
#include "forge/threads.h"

namespace forge {

namespace {

// Unique id for a parsed tool call (call-<hex-time>-<counter>).
std::string make_tool_call_id() {
    static std::atomic<uint64_t> counter{0};
    auto now = std::chrono::system_clock::now().time_since_epoch().count();
    std::ostringstream oss;
    oss << "call-" << std::hex << now << std::dec << "-" << ++counter;
    return oss.str();
}

// Find a tool-call open marker. The catalogue instructs the model to emit
// "###TOOL###", but quantized models routinely drop a '#' and emit
// "###TOOL##" (or add extras). We therefore match "###TOOL" followed by one
// or more '#'. Returns the position of the "###TOOL" prefix, or npos;
// `inner_start` is set just past the trailing '#'s (where the JSON begins).
size_t find_tool_open(const std::string& t, size_t from, size_t& inner_start) {
    static const std::string prefix = "###TOOL";
    size_t p = t.find(prefix, from);
    if (p == std::string::npos)
        return std::string::npos;
    size_t i = p + prefix.size();
    while (i < t.size() && t[i] == '#')
        ++i;
    inner_start = i;
    return p;
}

// Normalize a tool name for fuzzy matching against the registered catalogue:
// lowercase and strip everything but alphanumerics, so "get_weather",
// "getWeather" and "getweather" all collapse to "getweather" and match.
std::string normalize_tool_name(const std::string& s) {
    std::string o;
    o.reserve(s.size());
    for (char c : s) {
        if (std::isalnum(static_cast<unsigned char>(c)))
            o += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return o;
}

// Render the tool catalogue into a system-prompt block the model can follow.
std::string build_tools_prompt(const std::vector<forge::ToolSpec>& tools,
                               const std::string& tool_choice) {
    std::string p = "You have access to tools. To call one, respond with exactly one block of the form:\n"
                    "###TOOL###{name: <tool_name>, arguments: <json_object>}###END###\n"
                    "Only call a tool when it is genuinely needed to answer the user.";
    if (!tool_choice.empty() && tool_choice != "auto")
        p += " You MUST call the requested tool.";
    p += "\n\n# Tools\n";
    for (const auto& t : tools) {
        p += "- name: " + t.name + "\n";
        if (!t.description.empty())
            p += "  description: " + t.description + "\n";
        p += "  parameters: " + (t.parameters_json.empty() ? "{}" : t.parameters_json) + "\n";
    }
    return p;
}

// Prepend/append the tool catalogue to the system message.
std::vector<forge::ChatTemplateMessage> inject_tools_into_messages(
    const std::vector<forge::ChatTemplateMessage>& msgs,
    const std::vector<forge::ToolSpec>& tools, const std::string& tool_choice) {
    if (tools.empty())
        return msgs;
    std::vector<forge::ChatTemplateMessage> out = msgs;
    std::string block = build_tools_prompt(tools, tool_choice);
    if (!out.empty() && out[0].role == "system") {
        out[0].content += "\n\n" + block;
    } else {
        forge::ChatTemplateMessage sys;
        sys.role = "system";
        sys.content = block;
        out.insert(out.begin(), std::move(sys));
    }
    return out;
}

// Read a scalar field from a possibly-unquoted-key object body, so that a
// model emitting {name: fn, arguments: {...}} still yields a usable call.
std::string extract_tool_field(const std::string& body, const std::string& key) {
    size_t k = body.find(key);
    if (k == std::string::npos)
        return "";
    size_t c = body.find(':', k + key.size());
    if (c == std::string::npos)
        return "";
    size_t s = c + 1;
    while (s < body.size() && (body[s] == ' ' || body[s] == '\t'))
        ++s;
    if (s < body.size() && body[s] == '"') {
        size_t end = body.find('"', s + 1);
        if (end == std::string::npos)
            return "";
        return body.substr(s + 1, end - s - 1);
    }
    size_t end = body.find_first_of(",}", s);
    if (end == std::string::npos)
        end = body.size();
    std::string v = body.substr(s, end - s);
    size_t a = v.find_first_not_of(" \t\r\n");
    if (a == std::string::npos)
        return "";
    size_t b = v.find_last_not_of(" \t\r\n");
    return v.substr(a, b - a + 1);
}

// Read a balanced JSON object field from a possibly-unquoted-key object body.
std::string extract_tool_object(const std::string& body, const std::string& key) {
    size_t k = body.find(key);
    if (k == std::string::npos)
        return "";
    size_t s = body.find('{', k + key.size());
    if (s == std::string::npos)
        return "";
    int depth = 0;
    for (size_t i = s; i < body.size(); ++i) {
        if (body[i] == '{')
            ++depth;
        else if (body[i] == '}') {
            --depth;
            if (depth == 0)
                return body.substr(s, i - s + 1);
        }
    }
    return "";
}

// Lenient JSON repair: some quantized models emit invalid string escapes
// (e.g. \' or \. inside a JSON string) that strict parsers reject with
// "Invalid escape character". We cannot re-train the model, so repair the
// text: any backslash that does NOT begin a valid JSON escape is turned into
// a literal backslash (\\), which is valid JSON and preserves the character
// content. Only string literals are touched; structure outside quotes is left
// untouched so balanced-brace recovery still works.
std::string repair_json_invalid_escapes(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    bool in_str = false;
    for (size_t i = 0; i < s.size(); ++i) {
        char c = s[i];
        if (in_str) {
            if (c == '\\') {
                bool valid = false;
                if (i + 1 < s.size()) {
                    char n = s[i + 1];
                    valid = (n == '"' || n == '\\' || n == '/' || n == 'b' ||
                             n == 'f' || n == 'n' || n == 'r' || n == 't');
                    if (n == 'u' && i + 5 < s.size() &&
                        std::isxdigit(static_cast<unsigned char>(s[i + 2])) &&
                        std::isxdigit(static_cast<unsigned char>(s[i + 3])) &&
                        std::isxdigit(static_cast<unsigned char>(s[i + 4])) &&
                        std::isxdigit(static_cast<unsigned char>(s[i + 5]))) {
                        valid = true;
                    }
                }
                if (!valid) {
                    // Escape the offending backslash so the following character
                    // stays literal; the result is valid JSON and survives parse.
                    out += '\\';
                    out += '\\';
                    if (i + 1 < s.size()) {
                        out += s[i + 1];
                        ++i;  // consume the escaped character
                    }
                    continue;
                }
            } else if (c == '"') {
                in_str = false;
            }
            out += c;
        } else {
            if (c == '"')
                in_str = true;
            out += c;
        }
    }
    return out;
}

// Extract tool calls (###TOOL[#]...###END### wrapper) from generated text.
// `allowed`, when non-null and non-empty, is the set of tool names registered
// by the client; a name the model emitted (often mangled, e.g. "getweather"
// for "get_weather") is normalized and fuzzy-matched back to a registered
// name so the upstream caller receives a valid tool id. Calls that cannot be
// resolved to any registered tool are dropped.
std::vector<forge::ToolCall> parse_tool_calls(
    const std::string& text, const std::vector<std::string>* allowed = nullptr) {
    std::vector<forge::ToolCall> calls;
    static const std::string close = "###END###";
    size_t pos = 0;
    while (true) {
        size_t s = 0, inner_start = 0;
        s = find_tool_open(text, pos, inner_start);
        if (s == std::string::npos)
            break;
        size_t e = text.find(close, inner_start);
        if (e == std::string::npos)
            break;
        std::string inner = text.substr(inner_start, e - inner_start);
        // Repair invalid escapes (e.g. \') that strict parsers reject, so a
        // model-emitted tool call is not silently dropped (and so the client
        // does not choke re-parsing the arguments JSON).
        inner = repair_json_invalid_escapes(inner);
        nlohmann::json j;
        bool parsed = false;
        try {
            j = nlohmann::json::parse(inner);
            parsed = true;
        } catch (...) {
            size_t b = inner.find('{');
            size_t en = inner.rfind('}');
            if (b != std::string::npos && en != std::string::npos && en > b) {
                try {
                    j = nlohmann::json::parse(inner.substr(b, en - b + 1));
                    parsed = true;
                } catch (...) {
                }
            }
        }
        forge::ToolCall tc;
        tc.type = "function";
        if (parsed && j.is_object()) {
            if (j.contains("name") && j["name"].is_string())
                tc.name = j["name"].get<std::string>();
            if (j.contains("arguments")) {
                const auto& a = j["arguments"];
                tc.arguments = a.is_string() ? a.get<std::string>() : a.dump();
            }
        } else {
            // Tolerant path: models often emit unquoted keys, e.g.
            // {name: fn, arguments: {...}}. Recover name/arguments manually.
            tc.name = extract_tool_field(inner, "name");
            std::string args = extract_tool_object(inner, "arguments");
            tc.arguments = args.empty() ? "{}" : args;
        }
        if (tc.name.empty())
            break;
        // Resolve the (possibly mangled) name against the registered catalogue.
        if (allowed && !allowed->empty()) {
            bool resolved = false;
            const std::string norm = normalize_tool_name(tc.name);
            if (!norm.empty()) {
                for (const auto& a : *allowed) {
                    if (a == tc.name) {
                        tc.name = a;
                        resolved = true;
                        break;
                    }
                }
                if (!resolved) {
                    for (const auto& a : *allowed) {
                        if (normalize_tool_name(a) == norm) {
                            tc.name = a;
                            resolved = true;
                            break;
                        }
                    }
                }
            }
            if (!resolved)
                break;  // not a known tool: drop this block
        }
        tc.id = make_tool_call_id();
        calls.push_back(std::move(tc));
        pos = e + close.size();
    }
    return calls;
}

// Remove tool-call wrapper blocks from the assistant text.
std::string strip_tool_call_blocks(const std::string& text) {
    std::string out = text;
    static const std::string close = "###END###";
    size_t pos = 0;
    while (true) {
        size_t s = 0, inner_start = 0;
        s = find_tool_open(out, pos, inner_start);
        if (s == std::string::npos)
            break;
        size_t e = out.find(close, inner_start);
        if (e == std::string::npos) {
            out.erase(s);
            break;
        }
        out.erase(s, e + close.size() - s);
        pos = s;
    }
    size_t a = out.find_first_not_of(" \t\r\n");
    if (a == std::string::npos)
        return "";
    size_t b = out.find_last_not_of(" \t\r\n");
    return out.substr(a, b - a + 1);
}

}  // namespace

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

    if (opt.paged_kv) {
        ctx_->params_mut().kv_storage_mode = KVStorageMode::Paged;
        std::cout << "Paged KV cache storage enabled\n";
    }
    auto engine = EngineRegistry::instance().create(cfg.arch_type, *model_, *ctx_);
    if (!engine) {
        error = "No matching engine for architecture: " + cfg.arch_type;
        return false;
    }

    auto* tfm_eng = dynamic_cast<TransformerEngine*>(engine.get());
    if (tfm_eng) {
        KVCacheDType kv_dtype = KVCacheDType::F16;
        if (opt.kv_cache_dtype == "fp32")
            kv_dtype = KVCacheDType::FP32;
        else if (opt.kv_cache_dtype == "f16")
            kv_dtype = KVCacheDType::F16;
        else if (opt.kv_cache_dtype == "q8_0")
            kv_dtype = KVCacheDType::Q8_0;
        else if (opt.kv_cache_dtype == "q4_0")
            kv_dtype = KVCacheDType::Q4_0;
        else if (opt.kv_cache_dtype == "fp8_e4m3")
            kv_dtype = KVCacheDType::FP8_E4M3;
        else if (opt.kv_cache_dtype == "fp8_e5m2")
            kv_dtype = KVCacheDType::FP8_E5M2;
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
        // Paged mode: the pool is the real token capacity (it is fully
        // allocated up front) while the contiguous cache is never allocated —
        // reading kv_cache() here returned an uninitialized 0 and fell through
        // to the model's native max_seq_len (e.g. 262144), disabling the
        // truncation below entirely. Combine the pool bound with max_seq_len
        // like vLLM's min(max_model_len, num_gpu_blocks * block_size) check.
        if (auto* mem = eng->kv_memory(); mem && mem->is_paged()) {
            kv_capacity = std::min(mem->max_seq_len(), mem->max_pool_tokens());
            LOG_WARN("[capacity-check] paged storage: max_seq=" +
                     std::to_string(mem->max_seq_len()) + " pool_tokens=" +
                     std::to_string(mem->max_pool_tokens()) + " -> capacity=" +
                     std::to_string(kv_capacity));
        } else if (auto* kvc = eng->kv_cache()) {
            kv_capacity = kvc->max_seq_len();
            LOG_WARN("[capacity-check] contiguous cache: capacity=" +
                     std::to_string(kv_capacity));
        } else {
            LOG_WARN("[capacity-check] no kv_memory and no kv_cache");
        }
    }
    if (kv_capacity <= 0) {
        kv_capacity = model_->config().max_seq_len;
    }
    const int gen_budget = req.max_tokens > 0 ? req.max_tokens : 512;

    // ---- Tool injection (OpenAI function calling) ----
    // When the client offers tools (and does not forbid calling them), splice a
    // tool catalogue into the prompt so the model knows what it can invoke.
    const bool tools_active = !req.tools.empty() && req.tool_choice != "none";
    std::vector<ChatTemplateMessage> tooled_messages =
        tools_active ? inject_tools_into_messages(req.messages, req.tools, req.tool_choice)
                     : req.messages;

    ChatTemplateInput input;
    input.messages = tooled_messages;
    input.add_generation_prompt = true;
    std::vector<int32_t> prompt_ids = chat_template_->apply(input);

    // ---- DEBUG (set FORGE_DEBUG_TEMPLATE=1 to enable) ----
    // Dumps the rendered chat-template prompt so we can see whether the VLM
    // template is bloating text-only turns with <|image_pad|> placeholders or
    // otherwise mis-rendering (which would starve generation / produce no answer).
    if (std::getenv("FORGE_DEBUG_TEMPLATE")) {
        std::string dbg_render = chat_template_->render(input);
        size_t img_pad = 0, p = 0;
        while ((p = dbg_render.find("image_pad", p)) != std::string::npos) { ++img_pad; p += 9; }
        size_t vision = 0; p = 0;
        while ((p = dbg_render.find("vision", p)) != std::string::npos) { ++vision; p += 6; }
        std::cerr << "[debug-template] messages=" << input.messages.size()
                  << " jinja=" << (chat_template_->uses_jinja() ? 1 : 0)
                  << " rendered_len=" << dbg_render.size()
                  << " prompt_tokens=" << prompt_ids.size()
                  << " 'image_pad'=" << img_pad << " 'vision'=" << vision << "\n";
        for (size_t mi = 0; mi < input.messages.size(); ++mi) {
            std::cerr << "[debug-template]   msg[" << mi << "] role=" << input.messages[mi].role
                      << " content_len=" << input.messages[mi].content.size() << "\n";
        }
        std::string head = dbg_render.size() > 1600 ? dbg_render.substr(0, 1600) : dbg_render;
        std::cerr << "[debug-template] RENDER_HEAD:\n" << head << "\n"
                  << "[debug-template] RENDER_TAIL:\n"
                  << (dbg_render.size() > 400 ? dbg_render.substr(dbg_render.size() - 400) : dbg_render)
                  << "\n";
    }

    if (kv_capacity > 0 &&
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
                int overflow =
                    static_cast<int>(ids.size()) + gen_budget - kv_capacity;
                int m_tok = static_cast<int>(
                    tokenizer_->encode(m.content, false, false, false).size());
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
            std::cout << "[server] context shifted to fit KV capacity " << kv_capacity
                      << " (kept " << first_drop_idx << " prefix message(s)"
                      << (trimmed ? ", token-trimmed 1 boundary message" : "")
                      << ", dropped " << (drop_upto - first_drop_idx)
                      << " oldest history message(s))\n";
        } else {
            // Even the minimal message set does not fit. Fail loudly instead of
            // corrupting the KV cache.
            throw ContextLengthExceededError(
                "prompt exceeds model context capacity (" +
                std::to_string(prompt_ids.size()) + " tokens > " +
                std::to_string(kv_capacity - gen_budget) +
                " available after reserving " + std::to_string(gen_budget) +
                " generation tokens)");
        }
    }

    // vLLM-style hard admission check: a prompt that still exceeds capacity
    // after optional history truncation is rejected up front. Without this,
    // single oversized messages (message count <= 2 never enters the context
    // shift above — e.g. opencode pasting one large file) run the whole
    // prefill and die mid-write on page-pool exhaustion (a 500 after burning
    // the full prefill compute) instead of an immediate 400.
    if (kv_capacity > 0 &&
        static_cast<int>(prompt_ids.size()) + gen_budget > kv_capacity) {
        LOG_WARN("[capacity-check] REJECTING: prompt=" + std::to_string(prompt_ids.size()) +
                 " budget=" + std::to_string(gen_budget) + " capacity=" +
                 std::to_string(kv_capacity));
        throw ContextLengthExceededError(
            "This model's maximum context length is " + std::to_string(kv_capacity) +
            " tokens, but the request is " +
            std::to_string(static_cast<int>(prompt_ids.size()) + gen_budget) +
            " tokens (prompt " + std::to_string(prompt_ids.size()) + " + generation " +
            std::to_string(gen_budget) + ")");
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
    if (paged_mode) {
        if (auto* km = ctx_->engine()->kv_memory()) {
            // Drop the previous request's radix-node reference and working set
            // so seq 0 starts empty (register_prefix() may have left a trailing
            // unaligned page). The RequestScheduler calls release_prefix() when
            // a request finishes; the server path must do it here — otherwise
            // the matched node keeps ref_count >= 1 forever and its pages can
            // never be evicted.
            prefix_cache_.release_prefix(0, km->storage());
            km->storage().release(0);
            prefix_len = prefix_cache_.try_lookup(prompt_ids, 0, km->storage());
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
                        std::to_string(km->max_pool_tokens()) +
                        " tokens total, but only " + std::to_string(free_tokens) +
                        " are free after evicting cached prefixes; the request "
                        "needs " +
                        std::to_string(tail + gen_budget) +
                        " new tokens (prompt " + std::to_string(prompt_ids.size()) +
                        " - cached prefix " + std::to_string(prefix_len) +
                        " + max_tokens " + std::to_string(gen_budget) +
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
        if (!cached_kv_tokens_.empty()) {
            const size_t lim = std::min(cached_kv_tokens_.size(), prompt_ids.size());
            size_t m = 0;
            while (m < lim && cached_kv_tokens_[m] == prompt_ids[m]) {
                ++m;
            }
            prefix_len = static_cast<int>(m);
        }
        gen_cfg.prefix_len = prefix_len;
        gen_cfg.reset_kv_cache = (prefix_len == 0);
        if (prefix_len == 0) {
            // generate() resets the KV cache on this path; keep the mirror honest
            // so a later request cannot reuse rows that no longer exist.
            cached_kv_tokens_.clear();
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

    GenerationResult result;
    try {
        Generator gen(*ctx_, sampler_cfg);
        result = gen.generate(prompt_ids, gen_cfg, cb);
    } catch (...) {
        // A failed request can leave the KV cache partially overwritten, so the
        // recorded prefix no longer describes what is resident. Drop the mirror
        // and let the next request start from a clean cache. In paged mode also
        // release seq 0's working set (its prefix pages are shared with the radix
        // cache, which survives) so the next request does not append onto stale KV.
        if (paged_mode) {
            if (auto* km = ctx_->engine()->kv_memory()) {
                km->storage().release(0);
            }
        }
        cached_kv_tokens_.clear();
        throw;
    }

    // Record what the KV cache now holds, per backend.
    if (paged_mode) {
        // Register this request's full prompt KV into the radix cache so future
        // requests can reuse it, then evict LRU leaf nodes beyond capacity.
        // register_prefix() transfers seq 0's prefix pages into per-node cache
        // seq_ids (ref-counted, no copy) and releases seq 0's own reference; the
        // cached copy survives. Short prompts are handled internally (no-op).
        if (auto* km = ctx_->engine()->kv_memory()) {
            prefix_cache_.register_prefix(prompt_ids, 0, km->storage());
            prefix_cache_.evict_lru(PrefixCache::DEFAULT_MAX_ENTRIES, km->storage());
        }
    } else {
        // Contiguous single-entry mirror.
        cached_kv_tokens_ = prompt_ids;
        cached_kv_tokens_.insert(cached_kv_tokens_.end(), result.token_ids.begin(),
                                 result.token_ids.end());
        if (static_cast<int>(cached_kv_tokens_.size()) + gen_budget > kv_capacity) {
            cached_kv_tokens_.clear();
        }
    }

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

    // ---- Tool-call extraction (OpenAI function calling) ----
    // Parse any ###TOOL###...###END### blocks the model emitted into
    // OpenAI-shaped tool_calls and strip them from the assistant text.
    if (tools_active) {
        std::vector<std::string> tool_names;
        tool_names.reserve(req.tools.size());
        for (const auto& t : req.tools)
            tool_names.push_back(t.name);
        auto calls = parse_tool_calls(out.text, &tool_names);
        if (!calls.empty()) {
            out.text = strip_tool_call_blocks(out.text);
            out.tool_calls = std::move(calls);
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

    if (!out.tool_calls.empty()) {
        out.finish_reason = "tool_calls";
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
