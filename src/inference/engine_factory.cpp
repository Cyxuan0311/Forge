#include "forge/engine.h"
#include "forge/inference_batch.h"
#include "forge/logger.h"
#include "forge/model.h"

#include <cstring>
#include <mutex>

namespace forge {

// Static table of engine capabilities for compatibility checking.
// Keyed by engine registration name. GenericEngine handles all architectures
// through strategy sub-functions, so its entries are comprehensive.
static const std::unordered_map<std::string, EngineCapability>& engine_capabilities() {
    static const std::unordered_map<std::string, EngineCapability> caps = {
        // GenericEngine (registered under many architecture names)
        {"llama", EngineCapability{
            .supported_norm = NormType::RMSNorm,
            .supported_activation = ActivationType::SiLU_GELU,
            .supports_qk_norm = true,
            .supports_embedding_scale = true,
            .supports_post_attention_norm = true,
            .supports_post_ffn_norm = true,
            .supports_mrope = true,
            .supports_neox_rope = true,
        }},
        {"mistral", EngineCapability{
            .supported_norm = NormType::RMSNorm,
            .supported_activation = ActivationType::SiLU_GELU,
            .supports_neox_rope = true,
        }},
        {"qwen", EngineCapability{
            .supported_norm = NormType::RMSNorm,
            .supported_activation = ActivationType::SiLU_GELU,
        }},
        {"qwen2", EngineCapability{
            .supported_norm = NormType::RMSNorm,
            .supported_activation = ActivationType::SiLU_GELU,
        }},
        {"qwen3vl", EngineCapability{
            .supported_norm = NormType::RMSNorm,
            .supported_activation = ActivationType::SiLU_GELU,
            .supports_qk_norm = true,
            .supports_mrope = true,
        }},
        {"yi", EngineCapability{
            .supported_norm = NormType::RMSNorm,
            .supported_activation = ActivationType::SiLU_GELU,
            .supports_neox_rope = true,
        }},
        {"deepseek", EngineCapability{
            .supported_norm = NormType::RMSNorm,
            .supported_activation = ActivationType::SiLU_GELU,
        }},
        // Gemma family (GeGLU + embedding scale)
        {"gemma", EngineCapability{
            .supported_norm = NormType::RMSNorm,
            .supported_activation = ActivationType::GeGLU,
            .supports_embedding_scale = true,
            .supports_neox_rope = true,
        }},
        {"gemma2", EngineCapability{
            .supported_norm = NormType::RMSNorm,
            .supported_activation = ActivationType::GeGLU,
            .supports_embedding_scale = true,
            .supports_post_attention_norm = true,
            .supports_post_ffn_norm = true,
            .supports_neox_rope = true,
        }},
        {"gemma4", EngineCapability{
            .supported_norm = NormType::RMSNorm,
            .supported_activation = ActivationType::GeGLU,
            .supports_qk_norm = true,
            .supports_embedding_scale = true,
            .supports_neox_rope = true,
        }},
        // Falcon (LayerNorm + bias + parallel residual)
        {"falcon", EngineCapability{
            .supported_norm = NormType::LayerNorm,
            .supports_norm_bias = true,
            .supported_activation = ActivationType::GELU,
            .supports_qkv_bias = true,
            .supports_parallel_residual = true,
        }},
        // Specialized engines
        {"deepseek_v2", EngineCapability{
            .supported_norm = NormType::RMSNorm,
            .supported_activation = ActivationType::SiLU_GELU,
            .supports_neox_rope = true,
        }},
        {"deepseek_v3", EngineCapability{
            .supported_norm = NormType::RMSNorm,
            .supported_activation = ActivationType::SiLU_GELU,
            .supports_neox_rope = true,
        }},
        {"qwen35", EngineCapability{
            .supported_norm = NormType::RMSNorm,
            .supported_activation = ActivationType::SiLU_GELU,
            .supports_mrope = true,
        }},
    };
    return caps;
}

// Helper: check fallback compatibility and throw if incompatible
static void check_fallback_compatibility(const std::string& engine_name,
                                          const std::string& arch,
                                          const ArchCapability& cap) {
    auto& caps = engine_capabilities();
    auto cap_it = caps.find(engine_name);
    if (cap_it == caps.end())
        return;  // No capability info registered, skip check
    auto reasons = cap_it->second.check_compatibility(cap);
    if (!reasons.empty()) {
        throw std::runtime_error(
            "Engine '" + engine_name + "' cannot handle architecture '" + arch +
            "': " + reasons +
            "Please register a dedicated engine for this architecture.");
    }
}

// ============================================================================
// Startup compatibility validation
// ============================================================================
// Runs once (via std::call_once) on first EngineRegistry::create() call.
// Verifies that all registered architectures have compatible engine backends.

static void ensure_arch_compatibility() {
    auto& cap_reg = ArchCapabilityRegistry::instance();
    auto& eng_reg = EngineRegistry::instance();
    auto& caps = engine_capabilities();

    std::string errors;

    for (const auto& [arch, cap] : cap_reg.all()) {
        // Check 1: if the architecture has a dedicated engine, verify that the
        // engine's declared capability covers the architecture's requirements.
        if (eng_reg.has(arch)) {
            auto cap_it = caps.find(arch);
            if (cap_it != caps.end()) {
                auto reasons = cap_it->second.check_compatibility(cap);
                if (!reasons.empty()) {
                    errors += "  - Architecture '" + arch +
                              "' has a dedicated engine but EngineCapability mismatch: " +
                              reasons + "\n";
                }
            }
        }

        // Check 2: if the architecture has NO dedicated engine, verify the
        // fallback engine can handle it.
        if (!eng_reg.has(arch)) {
            // Determine fallback engine (same logic as in create())
            std::string fallback;
            if (cap.use_ssm)       fallback = "qwen35";
            else if (cap.use_mla)  fallback = "deepseek_v2";
            else if (cap.use_gqa)  fallback = "llama";

            if (!fallback.empty()) {
                auto cap_it = caps.find(fallback);
                if (cap_it != caps.end()) {
                    auto reasons = cap_it->second.check_compatibility(cap);
                    if (!reasons.empty()) {
                        errors += "  - Architecture '" + arch +
                                  "' will fallback to engine '" + fallback +
                                  "' but: " + reasons + "\n";
                    }
                } else {
                    errors += "  - Architecture '" + arch +
                              "' will fallback to engine '" + fallback +
                              "' but no EngineCapability declared for that engine\n";
                }
            }
        }
    }

    if (!errors.empty()) {
        LOG_ERROR("COMPATIBILITY ERRORS AT STARTUP:");
        throw std::runtime_error(
            "COMPATIBILITY ERROR at startup:\n" + errors +
            "Please register a dedicated engine or update EngineCapability for the affected architectures.");
    }

    LOG_INFO("Architecture compatibility check passed");
}

EngineRegistry& EngineRegistry::instance() {
    static EngineRegistry registry;
    return registry;
}

void EngineRegistry::register_engine(const std::string& arch, EngineCreator creator) {
    creators_[arch] = std::move(creator);
}

std::unique_ptr<InferenceEngine> EngineRegistry::create(const std::string& arch, Model& model,
                                                        InferenceContext& ctx) {
    // Lazy one-time compatibility check before first engine creation
    static std::once_flag compat_flag;
    std::call_once(compat_flag, ensure_arch_compatibility);

    // Try exact match first
    auto it = creators_.find(arch);
    if (it != creators_.end())
        return it->second(model, ctx);

    // Fallback: use architecture capability to auto-select engine
    auto& cap_registry = ArchCapabilityRegistry::instance();
    if (cap_registry.has(arch)) {
        auto cap = cap_registry.get(arch);
        // SSM → Qwen35Engine
        if (cap.use_ssm) {
            auto ssm_it = creators_.find("qwen35");
            if (ssm_it != creators_.end()) {
                check_fallback_compatibility("qwen35", arch, cap);
                return ssm_it->second(model, ctx);
            }
        }
        // MLA → DeepSeekEngine
        if (cap.use_mla) {
            auto mla_it = creators_.find("deepseek_v2");
            if (mla_it != creators_.end()) {
                check_fallback_compatibility("deepseek_v2", arch, cap);
                return mla_it->second(model, ctx);
            }
        }
        // GQA (default) → GenericEngine (registered as "llama")
        if (cap.use_gqa) {
            auto gqa_it = creators_.find("llama");
            if (gqa_it != creators_.end()) {
                check_fallback_compatibility("llama", arch, cap);
                return gqa_it->second(model, ctx);
            }
        }
    }

    // No matching engine found — return nullptr (caller will raise error)
    return nullptr;
}

std::vector<std::string> EngineRegistry::registered_archs() const {
    std::vector<std::string> result;
    result.reserve(creators_.size());
    for (const auto& [name, _] : creators_) {
        result.push_back(name);
    }
    return result;
}

bool EngineRegistry::has(const std::string& arch) const {
    return creators_.find(arch) != creators_.end();
}

EngineAutoRegister::EngineAutoRegister(const std::string& arch, EngineCreator creator) {
    EngineRegistry::instance().register_engine(arch, std::move(creator));
}

// Default forward_batch implementation: sequential fallback to forward()
// Returns [n_seq, vocab_size] with each sequence's last-token logits on CPU.
TensorPtr InferenceEngine::forward_batch(const InferenceBatch& batch) {
    if (batch.empty())
        return nullptr;

    // Call forward() for each sequence individually
    std::vector<TensorPtr> all_logits;
    for (const auto& item : batch.items) {
        int seq_len = static_cast<int>(item.tokens.size());
        auto input_ids =
            std::make_shared<Tensor>(DataType::INT32, std::vector<int64_t>{seq_len}, DeviceType::CPU);
        std::memcpy(input_ids->data(), item.tokens.data(), seq_len * sizeof(int32_t));
        all_logits.push_back(forward(input_ids, item.start_pos, item.seq_id));
    }

    int n_seq = static_cast<int>(all_logits.size());
    int vocab_size = static_cast<int>(all_logits[0]->shape().back());

    // Stack last-token logits from each sequence into [n_seq, vocab_size] on CPU
    auto result = std::make_shared<Tensor>(DataType::FP32,
                                            std::vector<int64_t>{n_seq, vocab_size},
                                            DeviceType::CPU);

    for (int i = 0; i < n_seq; i++) {
        int seq_len_i = static_cast<int>(all_logits[i]->shape()[0]);
        // Bring to CPU if needed
        TensorPtr logits_cpu = all_logits[i];
        if (all_logits[i]->device() == DeviceType::CUDA) {
            logits_cpu = std::make_shared<Tensor>(DataType::FP32, all_logits[i]->shape(),
                                                   DeviceType::CPU);
            logits_cpu->copy_from(*all_logits[i]);
        }
        // Copy last row into result row i
        const float* src = static_cast<const float*>(logits_cpu->data()) +
                           (seq_len_i - 1) * vocab_size;
        float* dst = static_cast<float*>(result->data()) + i * vocab_size;
        std::memcpy(dst, src, vocab_size * sizeof(float));
    }

    return result;
}

// Split a batch into micro-batches, each with at most n_ubatch tokens.
// Per-sequence chunking: long sequences are split into chunks; short sequences
// may be packed together as long as the total token count stays <= n_ubatch.
std::vector<InferenceBatch> split_batch(const InferenceBatch& batch, int n_ubatch) {
    if (batch.empty() || n_ubatch <= 0)
        return {batch};

    // Fast path: entire batch fits in one micro-batch
    if (batch.n_tokens() <= n_ubatch)
        return {batch};

    std::vector<InferenceBatch> micros;
    InferenceBatch current;
    int current_tokens = 0;

    for (const auto& item : batch.items) {
        int item_len = static_cast<int>(item.tokens.size());
        int offset = 0;

        while (offset < item_len) {
            int remaining = item_len - offset;
            int available = n_ubatch - current_tokens;

            if (available <= 0) {
                // Flush current micro-batch
                micros.push_back(std::move(current));
                current = InferenceBatch();
                current_tokens = 0;
                available = n_ubatch;
            }

            int chunk_len = std::min(remaining, available);

            InferenceBatchItem chunk;
            chunk.seq_id = item.seq_id;
            chunk.logits = item.logits && (offset + chunk_len == item_len);
            chunk.start_pos = item.start_pos + offset;
            chunk.tokens.assign(item.tokens.begin() + offset,
                                item.tokens.begin() + offset + chunk_len);
            if (!item.positions.empty()) {
                chunk.positions.assign(item.positions.begin() + offset,
                                       item.positions.begin() + offset + chunk_len);
            }

            current.items.push_back(std::move(chunk));
            current_tokens += chunk_len;
            offset += chunk_len;
        }
    }

    if (!current.empty())
        micros.push_back(std::move(current));

    return micros;
}

}  // namespace forge
