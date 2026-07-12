/**
 * NanoInfer CLI - Interactive chat mode
 */

#include "cli_common.h"

#include <iostream>
#include <fstream>
#include <chrono>

#include "nanoinfer/model.h"
#include "nanoinfer/tokenizer.h"
#include "nanoinfer/context.h"
#include "nanoinfer/engine.h"
#include "nanoinfer/types.h"
#include "nanoinfer/vision_encoder.h"
#include "nanoinfer/kv_cache.h"
#include "nanoinfer/engines/transformer_engine.h"

using namespace nanoinfer;

void interactive_chat(
    Model& model,
    Tokenizer& tokenizer,
    VisionEncoder* vision,
    const CliArgs& args)
{
    const auto& cfg = model.config();
    auto tmpl_type = detect_template_type(tokenizer);

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
    std::cout << "  Model: " << cfg.arch_type << " (" << cfg.num_layers << " layers)\n";
    std::cout << "  Device: " << (args.n_gpu_layers == 0 ? "CPU" : "CUDA") << "\n";
    std::cout << "  GPU layers: " << args.n_gpu_layers << "\n";
    std::cout << "  Commands: /quit, /clear, /system, /image, /save, /help\n";
    std::cout << "\n";

    std::unique_ptr<InferenceContext> ctx;

    while (!g_interrupted) {
        std::cout << "> ";
        std::string user_input;
        if (!std::getline(std::cin, user_input)) {
            std::cout << "\n";
            break;
        }

        user_input = trim(user_input);
        if (user_input.empty()) continue;

        if (user_input == "/quit" || user_input == "/exit") {
            std::cout << "Goodbye!\n";
            break;
        }
        else if (user_input == "/clear") {
            conversation.clear();
            if (!current_system.empty()) {
                conversation.push_back({"system", current_system});
            }
            current_image_embeddings.clear();
            current_num_img_tokens = 0;
            std::cout << "  [Conversation cleared]\n\n";
            continue;
        }
        else if (user_input == "/help") {
            std::cout << "  /quit, /exit      Exit\n";
            std::cout << "  /clear            Clear conversation history\n";
            std::cout << "  /system TEXT      Set system prompt\n";
            std::cout << "  /image PATH       Load image (multimodal)\n";
            std::cout << "  /save PATH        Save conversation to file\n";
            std::cout << "  /help             Show help\n\n";
            continue;
        }
        else if (user_input.rfind("/system ", 0) == 0) {
            current_system = user_input.substr(8);
            if (!conversation.empty() && conversation[0].role == "system") {
                conversation[0].content = current_system;
            } else if (!current_system.empty()) {
                conversation.insert(conversation.begin(), {"system", current_system});
            }
            std::cout << "  [System prompt set: " << current_system << "]\n\n";
            continue;
        }
        else if (user_input.rfind("/image ", 0) == 0) {
            if (!vision) {
                std::cout << "  [Error: Vision encoder not loaded, use --mmproj]\n\n";
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
            std::cout << "  [Image loaded: " << n_tokens << " tokens, " << ms << "ms]\n\n";
            continue;
        }
        else if (user_input.rfind("/save ", 0) == 0) {
            std::string save_path = user_input.substr(6);
            std::ofstream out(save_path);
            if (!out.is_open()) {
                std::cout << "  [Error: Cannot create file: " << save_path << "]\n\n";
                continue;
            }
            for (const auto& msg : conversation) {
                out << "[" << msg.role << "]\n" << msg.content << "\n\n";
            }
            out.close();
            std::cout << "  [Conversation saved to: " << save_path << "]\n\n";
            continue;
        }

        conversation.push_back({"user", user_input});

        auto prompt_ids = apply_chat_template(tokenizer, conversation, tmpl_type, true);

        ctx = std::make_unique<InferenceContext>(model);

        auto engine = EngineRegistry::instance().create(cfg.arch_type, model, *ctx);
        if (!engine) {
            std::cerr << "Error: No matching engine for architecture: " << cfg.arch_type << "\n";
            conversation.pop_back();
            continue;
        }

        auto* tfm_eng = dynamic_cast<TransformerEngine*>(engine.get());
        if (tfm_eng) {
            KVCacheDType kv_dtype = KVCacheDType::FP32;
            if (args.kv_cache_dtype == "q4_0") kv_dtype = KVCacheDType::Q4_0;
            tfm_eng->set_kv_cache_dtype(kv_dtype);
            tfm_eng->set_gpu_layers(args.n_gpu_layers);
        }

        ctx->set_engine(std::move(engine));

        std::cout << "\n";
        std::cout.flush();

        GenerationStats stats;

        if (args.no_stream) {
            stats = generate_batch(*ctx, tokenizer, prompt_ids,
                args.n_predict, args.temperature, args.top_k, args.top_p,
                args.repeat_penalty, !args.no_sample, args.seed,
                tokenizer.eos_token_id());
        } else {
            stats = generate_streaming(*ctx, tokenizer, prompt_ids,
                args.n_predict, args.temperature, args.top_k, args.top_p,
                args.repeat_penalty, !args.no_sample, args.seed,
                tokenizer.eos_token_id());
        }

        std::cout << "\n";

        if (stats.num_generated_tokens > 0) {
            double prompt_tok_s = stats.num_prompt_tokens / (stats.prompt_eval_ms / 1000.0);
            double gen_tok_s = stats.num_generated_tokens / (stats.elapsed_ms / 1000.0);
            printf("  [%d prompt tokens, %.1f ms, %.1f tok/s | %d generated, %.1f ms, %.1f tok/s]\n",
                   stats.num_prompt_tokens, stats.prompt_eval_ms, prompt_tok_s,
                   stats.num_generated_tokens, stats.elapsed_ms, gen_tok_s);
        }

        conversation.push_back({"assistant", ""});

        std::cout << "\n";

        current_image_embeddings.clear();
        current_num_img_tokens = 0;
    }
}
