/**
 * NanoInfer CLI - Chat template detection and application
 */

#include "cli_common.h"

#include "nanoinfer/tokenizer.h"

using namespace nanoinfer;

ChatTemplateType detect_template_type(const Tokenizer& tokenizer) {
    if (tokenizer.token_to_id("<|im_start|>") >= 0) {
        return ChatTemplateType::ChatML;
    }
    if (tokenizer.token_to_id("<｜User｜>") >= 0) {
        return ChatTemplateType::DeepSeek;
    }
    if (tokenizer.token_to_id("[INST]") >= 0) {
        return ChatTemplateType::Llama;
    }
    return ChatTemplateType::Plain;
}

std::vector<int32_t> apply_chat_template(
    const Tokenizer& tokenizer,
    const std::vector<ChatMessage>& messages,
    ChatTemplateType tmpl_type,
    bool add_generation_prompt)
{
    std::vector<int32_t> ids;

    switch (tmpl_type) {
    case ChatTemplateType::ChatML: {
        int32_t im_start_id = tokenizer.token_to_id("<|im_start|>");
        int32_t im_end_id = tokenizer.token_to_id("<|im_end|>");

        for (const auto& msg : messages) {
            ids.push_back(im_start_id);
            auto role_ids = tokenizer.encode(msg.role + "\n", false, false);
            ids.insert(ids.end(), role_ids.begin(), role_ids.end());
            auto content_ids = tokenizer.encode(msg.content, false, false);
            ids.insert(ids.end(), content_ids.begin(), content_ids.end());
            ids.push_back(im_end_id);
            auto nl_ids = tokenizer.encode("\n", false, false);
            ids.insert(ids.end(), nl_ids.begin(), nl_ids.end());
        }

        if (add_generation_prompt) {
            ids.push_back(im_start_id);
            auto asst_ids = tokenizer.encode("assistant\n", false, false);
            ids.insert(ids.end(), asst_ids.begin(), asst_ids.end());
        }
        break;
    }

    case ChatTemplateType::DeepSeek: {
        int32_t bos_id = tokenizer.bos_token_id();
        int32_t eos_id = tokenizer.eos_token_id();
        int32_t user_id = tokenizer.token_to_id("<｜User｜>");
        int32_t asst_id = tokenizer.token_to_id("<｜Assistant｜>");

        if (bos_id >= 0) ids.push_back(bos_id);

        for (const auto& msg : messages) {
            if (msg.role == "system") {
                auto sys_ids = tokenizer.encode(msg.content, false, false);
                ids.insert(ids.end(), sys_ids.begin(), sys_ids.end());
            } else if (msg.role == "user") {
                ids.push_back(user_id);
                auto content_ids = tokenizer.encode(msg.content, false, false);
                ids.insert(ids.end(), content_ids.begin(), content_ids.end());
            } else if (msg.role == "assistant") {
                ids.push_back(asst_id);
                auto content_ids = tokenizer.encode(msg.content, false, false);
                ids.insert(ids.end(), content_ids.begin(), content_ids.end());
                ids.push_back(eos_id);
            }
        }

        if (add_generation_prompt) {
            ids.push_back(asst_id);
        }
        break;
    }

    case ChatTemplateType::Llama: {
        int32_t bos_id = tokenizer.bos_token_id();
        int32_t eos_id = tokenizer.eos_token_id();

        bool first = true;
        for (const auto& msg : messages) {
            if (msg.role == "system" && first) {
                ids.push_back(bos_id >= 0 ? bos_id : tokenizer.token_to_id("<s>"));
                auto inst_ids = tokenizer.encode("[INST] <<SYS>>\n" + msg.content + "\n<</SYS>>\n\n", false, false);
                ids.insert(ids.end(), inst_ids.begin(), inst_ids.end());
                first = false;
            } else if (msg.role == "user") {
                if (first) {
                    ids.push_back(bos_id >= 0 ? bos_id : tokenizer.token_to_id("<s>"));
                    auto inst_ids = tokenizer.encode("[INST] " + msg.content + " [/INST]", false, false);
                    ids.insert(ids.end(), inst_ids.begin(), inst_ids.end());
                    first = false;
                } else {
                    auto inst_ids = tokenizer.encode("<s>[INST] " + msg.content + " [/INST]", false, false);
                    ids.insert(ids.end(), inst_ids.begin(), inst_ids.end());
                }
            } else if (msg.role == "assistant") {
                auto content_ids = tokenizer.encode(" " + msg.content + " ", false, false);
                ids.insert(ids.end(), content_ids.begin(), content_ids.end());
                ids.push_back(eos_id >= 0 ? eos_id : tokenizer.token_to_id("</s>"));
            }
        }

        if (add_generation_prompt && !ids.empty()) {
            // Last message is user, already has [/INST], ready for generation
        }
        break;
    }

    case ChatTemplateType::Plain: {
        for (const auto& msg : messages) {
            if (msg.role == "system") {
                auto sys_ids = tokenizer.encode(msg.content + "\n", true, false);
                ids.insert(ids.end(), sys_ids.begin(), sys_ids.end());
            } else if (msg.role == "user") {
                auto user_ids = tokenizer.encode(msg.content + "\n", false, false);
                ids.insert(ids.end(), user_ids.begin(), user_ids.end());
            } else if (msg.role == "assistant") {
                auto asst_ids = tokenizer.encode(msg.content + "\n", false, false);
                ids.insert(ids.end(), asst_ids.begin(), asst_ids.end());
            }
        }
        break;
    }
    }

    return ids;
}
