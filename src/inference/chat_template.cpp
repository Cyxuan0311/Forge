#include "lexer.h"
#include "parser.h"
#include "runtime.h"
#include "caps.h"

#include "forge/chat_template.h"
#include "forge/tokenizer.h"

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

            // Detect capabilities
            jinja::caps caps;

            auto jp = std::make_unique<JinjaProgram>();
            jp->prog = std::move(prog);
            jp->caps = caps;
            engine.jinja_prog_ = std::move(jp);

            // Detect system role support
            try {
                jinja::context ctx(engine.tmpl_source_);
                auto messages_arr = jinja::mk_val<jinja::value_array>();
                auto msg = jinja::mk_val<jinja::value_object>();
                msg->insert("role", jinja::mk_val<jinja::value_string>("system"));
                msg->insert("content", jinja::mk_val<jinja::value_string>("test"));
                messages_arr->push_back(msg);
                ctx.set_val("messages", messages_arr);
                ctx.set_val("bos_token", jinja::mk_val<jinja::value_string>(""));
                ctx.set_val("eos_token", jinja::mk_val<jinja::value_string>(""));
                ctx.set_val("add_generation_prompt", jinja::mk_val<jinja::value_bool>(false));

                jinja::runtime runtime(ctx);
                runtime.execute(engine.jinja_prog_->prog);
                engine.jinja_supports_system_ = true;
            } catch (...) {
                // Template doesn't handle system role — that's fine
            }
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
    jinja::context ctx(tmpl_source_);

    // Build messages array
    auto messages_arr = jinja::mk_val<jinja::value_array>();

    // Add system prompt if provided (prepend as system message)
    if (!input.system_prompt.empty()) {
        auto msg = jinja::mk_val<jinja::value_object>();
        msg->insert("role", jinja::mk_val<jinja::value_string>("system"));
        msg->insert("content", jinja::mk_val<jinja::value_string>(input.system_prompt));
        messages_arr->push_back(msg);
    }

    for (const auto& msg : input.messages) {
        auto msg_obj = jinja::mk_val<jinja::value_object>();
        msg_obj->insert("role", jinja::mk_val<jinja::value_string>(msg.role));
        msg_obj->insert("content", jinja::mk_val<jinja::value_string>(msg.content));
        messages_arr->push_back(msg_obj);
    }

    ctx.set_val("messages", messages_arr);

    // Set bos/eos tokens
    std::string bos_str = tok_->decode_token(tok_->bos_token_id());
    std::string eos_str = tok_->decode_token(tok_->eos_token_id());
    ctx.set_val("bos_token", jinja::mk_val<jinja::value_string>(bos_str));
    ctx.set_val("eos_token", jinja::mk_val<jinja::value_string>(eos_str));

    // Add generation prompt flag
    ctx.set_val("add_generation_prompt",
                jinja::mk_val<jinja::value_bool>(input.add_generation_prompt));

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