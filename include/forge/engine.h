#pragma once

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "inference_batch.h"
#include "model.h"
#include "tensor.h"

namespace forge {

class Model;
class InferenceContext;
class KVCache;

// Describes what an engine can handle, used for compatibility checking
// when falling back from an unregistered architecture to a known engine.
struct EngineCapability {
    NormType supported_norm = NormType::RMSNorm;
    bool supports_norm_bias = false;
    ActivationType supported_activation = ActivationType::SiLU_GELU;
    bool supports_qkv_bias = false;
    bool supports_parallel_residual = false;
    bool supports_qk_norm = false;
    bool supports_embedding_scale = false;
    bool supports_post_attention_norm = false;
    bool supports_post_ffn_norm = false;
    bool supports_mrope = false;
    bool supports_neox_rope = false;

    // Returns empty string if compatible, otherwise describes the incompatibility
    std::string check_compatibility(const ArchCapability& cap) const {
        std::string reasons;
        if (cap.norm_type == NormType::LayerNorm && supported_norm != NormType::LayerNorm)
            reasons += "requires LayerNorm but engine only supports RMSNorm; ";
        if (cap.has_norm_bias && !supports_norm_bias)
            reasons += "requires norm bias but engine doesn't support it; ";
        if (cap.ffn_activation != supported_activation &&
            cap.ffn_activation != ActivationType::SiLU_GELU)
            reasons += "requires different activation type; ";
        if (cap.has_qkv_bias && !supports_qkv_bias)
            reasons += "requires QKV bias but engine doesn't support it; ";
        if (cap.use_parallel_residual && !supports_parallel_residual)
            reasons += "requires parallel residual but engine doesn't support it; ";
        if (cap.use_qk_norm && !supports_qk_norm)
            reasons += "requires QK-norm but engine doesn't support it; ";
        if (cap.embedding_scale && !supports_embedding_scale)
            reasons += "requires embedding scaling but engine doesn't support it; ";
        if (cap.has_post_attention_norm && !supports_post_attention_norm)
            reasons += "requires post-attention norm but engine doesn't support it; ";
        if (cap.has_post_ffn_norm && !supports_post_ffn_norm)
            reasons += "requires post-FFN norm but engine doesn't support it; ";
        if (cap.use_mrope && !supports_mrope)
            reasons += "requires MRoPE but engine doesn't support it; ";
        if (cap.use_neox_rope && !supports_neox_rope)
            reasons += "requires NeoX RoPE but engine doesn't support it; ";
        return reasons;
    }
};

class InferenceEngine {
public:
    virtual ~InferenceEngine() = default;

    // Single-sequence forward (backward compatible, seq_id defaults to 0)
    virtual TensorPtr forward(const TensorPtr& input_ids, int64_t start_pos, int seq_id = 0) = 0;

    // Multi-sequence batch forward.
    // Default implementation: fall back to sequential forward() calls.
    // Subclasses can override for true batched computation.
    virtual TensorPtr forward_batch(const InferenceBatch& batch);

    virtual TensorPtr forward_from_hidden(const TensorPtr& hidden, int64_t start_pos) {
        (void)hidden;
        (void)start_pos;
        throw std::runtime_error("forward_from_hidden not implemented for this engine");
    }
    virtual std::string name() const = 0;
    virtual void reset() {}
    virtual void set_gpu_layers(int layers) { (void)layers; }
    virtual int gpu_layers() const { return -1; }

    // Access the engine's KV cache (returns nullptr if not available)
    virtual KVCache* kv_cache() { return nullptr; }
    virtual const KVCache* kv_cache() const { return nullptr; }
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
