#include "lexer.h"
#include "parser.h"
#include "runtime.h"
#include "caps.h"

#include "forge/chat_template.h"
#include "forge/tokenizer.h"

#include <nlohmann/json.hpp>

#include <stdexcept>

namespace forge {

// ---- Opaque JinjaProgram wrapper ----
struct ChatTemplateEngine::JinjaProgram {
    jinja::program prog;
    jinja::caps caps;
};

ChatTemplateEngine::ChatTemplateEngine() = default;
ChatTemplateEngine::~ChatTemplateEngine() = default;
ChatTemplateEngine::ChatTemplateEngine(ChatTemplateEngine&&) noexcept = default;
ChatTemplateEngine& ChatTemplateEngine::operator=(ChatTemplateEngine&&) noexcept = default;

// ========================================================================
// from_tokenizer
// ========================================================================

ChatTemplateEngine ChatTemplateEngine::from_tokenizer(const Tokenizer& tok) {
    ChatTemplateEngine engine;
    engine.tok_ = &tok;
    engine.tmpl_source_ = tok.chat_template();

    if (!engine.tmpl_source_.empty()) {
        try {
            // Compile the jinja template
            jinja::lexer lex;
            auto lex_res = lex.tokenize(engine.tmpl_source_);
            auto prog = jinja::parse_from_tokens(lex_res);

            auto jp = std::make_unique<JinjaProgram>();
            jp->prog = std::move(prog);
            // Trial-execute the template with probe messages/tools to learn
            // which OpenAI features it actually renders (tools branch, tool
            // call replay, object vs string arguments, system role).
            jp->caps = jinja::caps_get(jp->prog);
            engine.jinja_prog_ = std::move(jp);

            engine.jinja_supports_system_ = engine.jinja_prog_->caps.supports_system_role;
            engine.jinja_supports_tools_ = engine.jinja_prog_->caps.supports_tools;
        } catch (const std::exception& e) {
            // Jinja compilation failed — fall back to hardcoded templates
            engine.jinja_prog_.reset();
        }
    }

    // Set up fallback path if jinja is not available
    if (!engine.jinja_prog_) {
        // Detect fallback template type from tokenizer special tokens
        if (tok.token_to_id("<|im_start|>") >= 0) {
            engine.fallback_type_ = FallbackType::ChatML;
        } else if (tok.token_to_id("  ") >= 0) {
            engine.fallback_type_ = FallbackType::DeepSeek;
        } else if (tok.token_to_id("[INST]") >= 0) {
            engine.fallback_type_ = FallbackType::Llama;
        } else {
            engine.fallback_type_ = FallbackType::Plain;
        }
    }

    return engine;
}

// ========================================================================
// Capability queries
// ========================================================================

bool ChatTemplateEngine::uses_jinja() const {
    return jinja_prog_ != nullptr;
}

bool ChatTemplateEngine::supports_system_role() const {
    if (jinja_prog_) {
        return jinja_supports_system_;
    }
    // Fallback: ChatML supports system, others don't
    return fallback_type_ == FallbackType::ChatML;
}

bool ChatTemplateEngine::supports_tools() const {
    return jinja_prog_ && jinja_supports_tools_;
}

// ========================================================================
// render
// ========================================================================

std::string ChatTemplateEngine::render(const ChatTemplateInput& input) const {
    if (jinja_prog_) {
        return render_jinja(input);
    }
    return render_fallback(input);
}

// ========================================================================
// apply
// ========================================================================

std::vector<int32_t> ChatTemplateEngine::apply(const ChatTemplateInput& input) const {
    if (!tok_) {
        return {};
    }
    std::string rendered = render(input);
    return tok_->encode(rendered, /*add_bos=*/false, /*add_eos=*/false,
                        /*add_dummy_prefix=*/true, /*parse_special=*/true);
}

// ========================================================================
// Jinja rendering
// ========================================================================

std::string ChatTemplateEngine::render_jinja(const ChatTemplateInput& input) const {
    using json = nlohmann::ordered_json;

    // ---- Build the template context as JSON, then inject via global_from_json.
    // OpenAI-shaped messages/tools: templates that render a native tool
    // catalogue read `tools` and replay `tool_calls` from assistant turns.
    json msgs = json::array();
    if (!input.system_prompt.empty()) {
        msgs.push_back(json{{"role", "system"}, {"content", input.system_prompt}});
    }
    bool any_tool_calls = false;
    int call_seq = 0;
    for (const auto& msg : input.messages) {
        json jm = json{{"role", msg.role}, {"content", msg.content}};
        if (!msg.reasoning_content.empty())
            jm["reasoning_content"] = msg.reasoning_content;
        if (!msg.tool_calls.empty()) {
            any_tool_calls = true;
            json tcs = json::array();
            for (const auto& tc : msg.tool_calls) {
                json args = json::parse(tc.arguments_json, nullptr, false);
                if (args.is_discarded())
                    args = json::object();
                // Dict-iterating templates (e.g. MiniCPM's XML dialect) need
                // object arguments; string-argument ones get the JSON text.
                if (!jinja_prog_->caps.supports_object_arguments && args.is_object())
                    args = args.dump();
                std::string id = tc.id.empty() ? ("call" + std::to_string(++call_seq)) : tc.id;
                tcs.push_back(json{{"id", id},
                                   {"type", "function"},
                                   {"function", {{"name", tc.name}, {"arguments", args}}}});
            }
            jm["tool_calls"] = std::move(tcs);
        }
        msgs.push_back(std::move(jm));
    }

    json ctx_json = json{{"messages", std::move(msgs)},
                         {"bos_token", tok_->decode_token(tok_->bos_token_id())},
                         {"eos_token", tok_->decode_token(tok_->eos_token_id())},
                         {"add_generation_prompt", input.add_generation_prompt}};

    if (!input.tools_json.empty()) {
        json tools = json::parse(input.tools_json, nullptr, false);
        if (!tools.is_discarded() && tools.is_array() && !tools.empty()) {
            json wrapped = json::array();
            for (auto& t : tools) {
                if (t.is_object() && t.contains("function"))
                    wrapped.push_back(std::move(t));
                else
                    wrapped.push_back(json{{"type", "function"}, {"function", std::move(t)}});
            }
            ctx_json["tools"] = std::move(wrapped);
        }
    }
    // Templates that interleave replayed calls into assistant content at
    // <tool_sep> markers also render a fallback copy unless told the calls
    // were already handled (undefined == false). We always replay via the
    // template's own tool_calls path, so mark it done.
    if (any_tool_calls)
        ctx_json["has_tool_sep"] = true;

    jinja::context ctx(tmpl_source_);
    jinja::global_from_json(ctx, ctx_json, false);

    // Execute
    jinja::runtime runtime(ctx);
    const jinja::value results = runtime.execute(jinja_prog_->prog);
    auto parts = jinja::runtime::gather_string_parts(results);

    return parts->as_string().str();
}

// ========================================================================
// Fallback rendering (hardcoded templates)
// ========================================================================

std::string ChatTemplateEngine::render_fallback(const ChatTemplateInput& input) const {
    std::string result;

    std::string bos, eos;
    if (tok_) {
        bos = tok_->decode_token(tok_->bos_token_id());
        eos = tok_->decode_token(tok_->eos_token_id());
    }

    switch (fallback_type_) {
        case FallbackType::ChatML: {
            // <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n...
            if (!input.system_prompt.empty()) {
                result += "<|im_start|>system\n" + input.system_prompt + "<|im_end|>\n";
            }
            for (const auto& msg : input.messages) {
                result += "<|im_start|>" + msg.role + "\n" + msg.content + "<|im_end|>\n";
            }
            if (input.add_generation_prompt) {
                result += "<|im_start|>assistant\n";
            }
            break;
        }
        case FallbackType::DeepSeek: {
            // 用户: ...\n助手: ...
            for (const auto& msg : input.messages) {
                if (msg.role == "user") {
                    result += "用户: " + msg.content + "\n";
                } else if (msg.role == "assistant") {
                    result += "助手: " + msg.content + "\n";
                } else if (msg.role == "system") {
                    result += msg.content + "\n";
                }
            }
            if (input.add_generation_prompt) {
                result += "助手: ";
            }
            break;
        }
        case FallbackType::Llama: {
            // [INST] ... [/INST] ...
            if (!input.system_prompt.empty()) {
                result += "<<SYS>>\n" + input.system_prompt + "\n<</SYS>>\n\n";
            }
            for (const auto& msg : input.messages) {
                if (msg.role == "user") {
                    result += "[INST] " + msg.content + " [/INST]";
                } else if (msg.role == "assistant") {
                    result += " " + msg.content + " ";
                }
            }
            break;
        }
        case FallbackType::Plain:
        default: {
            for (const auto& msg : input.messages) {
                result += msg.role + ": " + msg.content + "\n";
            }
            if (input.add_generation_prompt) {
                result += "assistant: ";
            }
            break;
        }
    }

    return result;
}

}  // namespace forge