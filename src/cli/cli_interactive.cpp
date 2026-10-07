/**
 * Forge CLI - Interactive chat mode
 */

#include <chrono>
#include <fstream>
#include <iostream>

#include "cli_common.h"
#include "forge/chat_template.h"
#include "forge/context.h"
#include "forge/engine.h"
#include "forge/engines/transformer_engine.h"
#include "forge/kv_cache.h"
#include "forge/model.h"
#include "forge/prefix_cache.h"
#include "forge/tokenizer.h"
#include "forge/types.h"
#include "forge/vision_encoder.h"

using namespace forge;

void interactive_chat(InferenceContext& ctx, Model& model, Tokenizer& tokenizer,
                      VisionEncoder* vision, const CliArgs& args) {
    const auto& cfg = model.config();

    // Build ChatTemplateEngine — jinja path is the default;
    // --no-jinja forces fallback to hardcoded templates.
    auto build_engine = [&]() -> ChatTemplateEngine {
        if (args.no_jinja) {
            std::cout << "  [Note: Jinja disabled by --no-jinja, using fallback templates]\n";
            return ChatTemplateEngine();
        }
        auto eng = ChatTemplateEngine::from_tokenizer(tokenizer);
        if (!eng.uses_jinja()) {
            std::cout << "  [Note: Using fallback chat template (no GGUF chat_template found)]\n";
        }
        return eng;
    };
    ChatTemplateEngine tmpl_engine = build_engine();

    std::vector<ChatMessage> conversation;
    std::string current_system = args.system_prompt;
    std::string current_image_path = args.image_path;
    std::vector<float> current_image_embeddings;
    int current_num_img_tokens = 0;

    if (!current_system.empty()) {
        conversation.push_back({"system", current_system});
    }

    print_logo();

    std::cout << "\n";
    std::cout << "  " << cli_color::green("Model:") << " " << cli_color::bold(cfg.arch_type) << " ("
              << cli_color::cyan(std::to_string(cfg.num_layers)) << " layers)\n";
    std::cout << "  " << cli_color::green("Device:") << " "
              << cli_color::bold(args.common.n_gpu_layers == 0 ? "CPU" : "CUDA") << "\n";
    std::cout << "  " << cli_color::green("GPU layers:") << " "
              << cli_color::cyan(std::to_string(args.common.n_gpu_layers)) << "\n";
    std::cout << "  " << cli_color::green("Commands:") << " "
              << cli_color::dim("/quit, /clear, /system, /image, /save, /help") << "\n";
    std::cout << "\n";

    // Reuse the InferenceContext (and its engine + KV cache) created and
    // initialized by the CLI main flow. A fresh context here would re-probe
    // device memory after the weights + the main context already occupied VRAM
    // and get its KV cache clamped to a tiny max_seq (observed: 15872 -> 256,
    // KVCache::update cache overflow at token 256). The main context was sized
    // with the full memory budget and is kept between turns; reset_kv_cache()
    // below only clears contents, never the resident KV buffers.

    // Persisted across turns; holds cached prefixes in dedicated cache seq_ids.
    PrefixCache prefix_cache;

    while (!g_interrupted) {
        // Orange input prompt: open the SGR span before reading so the
        // terminal's own echo of the typed line renders orange as well, then
        // close it the moment getline returns (see input_echo_color()).
        const bool orange_input = cli_color::input_echo_color();
        if (orange_input)
            std::cout << cli_color::orange_open() << "> ";
        else
            std::cout << "> ";
        std::cout.flush();
        std::string user_input;
        if (!std::getline(std::cin, user_input)) {
            if (orange_input)
                std::cout << cli_color::sgr_reset();
            std::cout << "\n";
            break;
        }
        if (orange_input)
            std::cout << cli_color::sgr_reset();

        user_input = trim(user_input);
        if (user_input.empty())
            continue;

        if (user_input == "/quit" || user_input == "/exit") {
            std::cout << "Goodbye!\n";
            break;
        } else if (user_input == "/clear") {
            conversation.clear();
            if (!current_system.empty()) {
                conversation.push_back({"system", current_system});
            }
            current_image_embeddings.clear();
            current_num_img_tokens = 0;
            std::cout << "  " << cli_color::green("[Conversation cleared]") << "\n\n";
            continue;
        } else if (user_input == "/help") {
            std::cout << "  /quit, /exit      Exit\n";
            std::cout << "  /clear            Clear conversation history\n";
            std::cout << "  /system TEXT      Set system prompt\n";
            std::cout << "  /image PATH       Load image (multimodal)\n";
            std::cout << "  /save PATH        Save conversation to file\n";
            std::cout << "  /help             Show help\n\n";
            continue;
        } else if (user_input.rfind("/system ", 0) == 0) {
            current_system = user_input.substr(8);
            if (!conversation.empty() && conversation[0].role == "system") {
                conversation[0].content = current_system;
            } else if (!current_system.empty()) {
                conversation.insert(conversation.begin(), {"system", current_system});
            }
            std::cout << "  " << cli_color::green("[System prompt set: ") << current_system
                      << cli_color::green("]") << "\n\n";
            continue;
        } else if (user_input.rfind("/image ", 0) == 0) {
            if (!vision) {
                std::cout << "  "
                          << cli_color::red("[Error: Vision encoder not loaded, use --mmproj]")
                          << "\n\n";
                continue;
            }
            std::string img_path = user_input.substr(7);
            int n_tokens = 0;
            auto t0 = std::chrono::high_resolution_clock::now();
            auto emb = encode_image(*vision, img_path, n_tokens);
            auto t1 = std::chrono::high_resolution_clock::now();
            if (emb.empty()) {
                continue;
            }
            current_image_embeddings = std::move(emb);
            current_num_img_tokens = n_tokens;
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::cout << "  " << cli_color::cyan("[Image loaded: ") << n_tokens << " tokens, " << ms
                      << "ms" << cli_color::cyan("]") << "\n\n";
            continue;
        } else if (user_input.rfind("/save ", 0) == 0) {
            std::string save_path = user_input.substr(6);
            std::ofstream out(save_path);
            if (!out.is_open()) {
                std::cout << "  " << cli_color::red("[Error: Cannot create file: ") << save_path
                          << cli_color::red("]") << "\n\n";
                continue;
            }
            for (const auto& msg : conversation) {
                out << "[" << msg.role << "]\n" << msg.content << "\n\n";
            }
            out.close();
            std::cout << "  " << cli_color::green("[Conversation saved to: ") << save_path
                      << cli_color::green("]") << "\n\n";
            continue;
        }

        conversation.push_back({"user", user_input});

        // Build ChatTemplateInput: extract system message, pass rest as messages
        ChatTemplateInput tmpl_input;
        tmpl_input.add_generation_prompt = true;
        for (const auto& msg : conversation) {
            if (msg.role == "system") {
                tmpl_input.system_prompt = msg.content;
            } else {
                tmpl_input.messages.push_back({msg.role, msg.content});
            }
        }

        std::vector<int32_t> prompt_ids;
        if (tmpl_engine.uses_jinja() || !args.no_jinja) {
            prompt_ids = tmpl_engine.apply(tmpl_input);
        } else {
            // --no-jinja: use legacy fallback directly
            auto tmpl_type = detect_template_type(tokenizer);
            prompt_ids = apply_chat_template(tokenizer, conversation, tmpl_type, true);
        }

        // Contiguous mode keeps the old per-turn behavior (fresh KV each turn);
        // paged mode leaves seq 0 to the radix PrefixCache (lookup + register).
        if (ctx.params().kv_storage_mode != KVStorageMode::Paged) {
            ctx.reset_kv_cache();
        }

        std::cout << "\n";
        std::cout.flush();

        GenerationStats stats;
        // The reply text is needed to build the next turn's prompt. Storing an
        // empty assistant turn (as this used to) meant the model could not see its
        // own previous answers at all, and the KV of every reply became
        // unreachable -- so each turn re-prefilled the whole previous message.
        std::string reply_text;

        if (args.no_stream) {
            stats = generate_batch(ctx, tokenizer, prompt_ids, args.n_predict, args.temperature,
                                   args.top_k, args.top_p, args.repeat_penalty, args.repeat_last_n,
                                   !args.no_sample, args.seed, tokenizer.eos_token_id(), args.spec,
                                   &prefix_cache, &reply_text, args.dashboard.capture_operators);
        } else {
            stats =
                generate_streaming(ctx, tokenizer, prompt_ids, args.n_predict, args.temperature,
                                   args.top_k, args.top_p, args.repeat_penalty, args.repeat_last_n,
                                   !args.no_sample, args.seed, tokenizer.eos_token_id(), args.spec,
                                   &prefix_cache, &reply_text, args.dashboard.capture_operators);
        }

        std::cout << "\n";

        forge_print_stats_line(stats);

        // Dashboard: session count and KV occupancy for the turn that just
        // finished. Safe to call every turn -- it early-outs on one relaxed
        // atomic when the registry is disabled.
        publish_cli_engine_gauges(ctx, &prefix_cache, 1);

        conversation.push_back({"assistant", reply_text});

        std::cout << "\n";

        current_image_embeddings.clear();
        current_num_img_tokens = 0;
    }
}
