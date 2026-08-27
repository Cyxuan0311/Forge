#pragma once

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "inference/forward_request.h"
#include "inference_batch.h"
#include "model.h"
#include "tensor.h"

namespace forge {

class Model;
class InferenceContext;
class KVCache;
class KVMemory;

class InferenceEngine {
public:
    virtual ~InferenceEngine() = default;

    // Request-based forward. This is the primary entry point: all runtime
    // semantics (start_pos, seq_id, token count, prefill/decode) travel together
    // in one object instead of as separate positional arguments.
    virtual TensorPtr forward_request(const ForwardRequest& req) = 0;

    // Multi-sequence batch forward.
    // Default implementation: fall back to sequential forward() calls.
    // Subclasses can override for true batched computation.
    virtual TensorPtr forward_batch(const InferenceBatch& batch);

    virtual TensorPtr forward_from_hidden(const TensorPtr& hidden, int64_t start_pos) {
        (void)hidden;
        (void)start_pos;
        throw std::runtime_error("forward_from_hidden not implemented for this engine");
    }
    // Post-final-norm hidden state from the most recent forward_request
    // (DeepSeek-MTP style draft input). Returns nullptr when the engine does
    // not expose it or the last forward went through a path that cannot.
    virtual TensorPtr take_last_hidden() { return nullptr; }

    virtual std::string name() const = 0;
    virtual void reset() {}
    virtual void set_gpu_layers(int layers) { (void)layers; }
    virtual int gpu_layers() const { return -1; }

    // Access the engine's KV cache (for attention_executor / memory_stats).
    // The cache itself is owned by InferenceContext; this is a forwarding accessor.
    virtual KVCache* kv_cache() { return nullptr; }
    virtual const KVCache* kv_cache() const { return nullptr; }

    // Access the engine's KV memory (unified interface for scheduler).
    // Returns nullptr if not available.
    virtual KVMemory* kv_memory() { return nullptr; }
    virtual const KVMemory* kv_memory() const { return nullptr; }

    // Allocate the engine's runtime memory (KV cache and, later, recurrent
    // state) if it has not been allocated yet. InferenceContext calls this so
    // that callers can query cache stats before the first forward.
    virtual void init_memory() {}
};

using EngineCreator = std::function<std::unique_ptr<InferenceEngine>(Model&, InferenceContext&)>;

class EngineRegistry {
public:
    static EngineRegistry& instance();

    void register_engine(const std::string& arch, EngineCreator creator);
    std::unique_ptr<InferenceEngine> create(const std::string& arch, Model& model,
                                            InferenceContext& ctx);
    std::vector<std::string> registered_archs() const;
    bool has(const std::string& arch) const;

private:
    EngineRegistry() = default;
    std::unordered_map<std::string, EngineCreator> creators_;
};

struct EngineAutoRegister {
    EngineAutoRegister(const std::string& arch, EngineCreator creator);
};

#define FORGE_REGISTER_ENGINE_IMPL2(line, arch, creator) \
    static ::forge::EngineAutoRegister _engine_reg_##line(arch, creator)

#define FORGE_REGISTER_ENGINE_IMPL(line, arch, creator) \
    FORGE_REGISTER_ENGINE_IMPL2(line, arch, creator)

#define FORGE_REGISTER_ENGINE(arch, creator) FORGE_REGISTER_ENGINE_IMPL(__LINE__, arch, creator)

}  // namespace forge
