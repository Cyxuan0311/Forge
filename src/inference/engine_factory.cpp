#include "forge/engine.h"
#include "forge/inference/execution_plan.h"
#include "forge/inference_batch.h"
#include "forge/logger.h"
#include "forge/model.h"

#include <cstring>
#include <mutex>

namespace forge {

// ============================================================================
// Startup plan validation
// ============================================================================
// Runs once (via std::call_once) before the first engine is created.
// Every registered architecture must produce a valid ExecutionPlan whose engine
// kind resolves to a registered engine creator.
//
// This replaces the previous hand-maintained EngineCapability string table:
// that table was a second source of truth next to the ArchCapability
// registrations and could silently drift from them. The ExecutionPlan is now
// the only place where "which engine can run this architecture" is decided.

static void ensure_plans_valid() {
    auto& cap_reg = ArchCapabilityRegistry::instance();
    auto& eng_reg = EngineRegistry::instance();

    std::string errors;

    for (const auto& [arch, cap] : cap_reg.all()) {
        (void)cap;
        try {
            auto plan = build_execution_plan_from_capability(arch);
            const char* engine_name = engine_name_for(plan.engine_kind);
            if (!eng_reg.has(arch) && !eng_reg.has(engine_name)) {
                errors += "  - Architecture '" + arch + "' resolves to engine '" +
                          std::string(engine_name) + "' but that engine is not registered\n";
                continue;
            }
            LOG_DEBUG("Execution plan: " + plan.plan_id());
        } catch (const std::exception& e) {
            errors += "  - Architecture '" + arch + "': " + e.what() + "\n";
        }
    }

    if (!errors.empty()) {
        LOG_ERROR("EXECUTION PLAN ERRORS AT STARTUP:");
        throw std::runtime_error(
            "COMPATIBILITY ERROR at startup:\n" + errors +
            "Please fix the architecture registration or register a dedicated engine.");
    }

    LOG_INFO("Architecture execution plan check passed");
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
    static std::once_flag plan_flag;
    std::call_once(plan_flag, ensure_plans_valid);

    // An explicit registration for this architecture always wins.
    auto it = creators_.find(arch);
    if (it != creators_.end())
        return it->second(model, ctx);

    // Otherwise the ExecutionPlan decides which engine can execute this model.
    // The plan is built from the real ModelConfig, so a model whose metadata
    // contradicts its architecture declaration fails here rather than at
    // forward time. Selection no longer falls back on a single bool.
    if (!ArchCapabilityRegistry::instance().has(arch))
        return nullptr;

    auto plan = build_execution_plan(arch, model.config());
    auto engine_it = creators_.find(engine_name_for(plan.engine_kind));
    if (engine_it == creators_.end())
        return nullptr;

    LOG_INFO("Architecture '" + arch + "' executed by engine '" +
             std::string(engine_name_for(plan.engine_kind)) + "', plan: " + plan.plan_id());
    return engine_it->second(model, ctx);
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

// Default forward_batch implementation: sequential fallback to forward_request()
// Returns [n_seq, vocab_size] with each sequence's last-token logits on CPU.
TensorPtr InferenceEngine::forward_batch(const InferenceBatch& batch) {
    if (batch.empty())
        return nullptr;

    // Call forward_request() for each sequence individually
    std::vector<TensorPtr> all_logits;
    for (const auto& item : batch.items) {
        int seq_len = static_cast<int>(item.tokens.size());
        auto input_ids =
            std::make_shared<Tensor>(DataType::INT32, std::vector<int64_t>{seq_len}, DeviceType::CPU);
        std::memcpy(input_ids->data(), item.tokens.data(), seq_len * sizeof(int32_t));
        all_logits.push_back(forward_request(ForwardRequest::from_ids(input_ids, item.start_pos, item.seq_id)));
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
