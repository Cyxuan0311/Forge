#include "forge/engines/deepseek_engine.h"

#include <stdexcept>

namespace forge {

DeepSeekEngine::DeepSeekEngine(Model& model, InferenceContext& ctx)
    : TransformerEngine(model, ctx), layer_executor_(kv_cache_) {
    if (!init_weights()) {
        throw std::runtime_error("DeepSeekEngine: failed to initialize weights");
    }
}

bool DeepSeekEngine::init_weights() {
    return weights_.init(model_.weights(), model_.config());
}

TensorPtr DeepSeekEngine::forward_layer(const TensorPtr& hidden,
                                       const LayerExecutionContext& lctx) {
    return layer_executor_.execute(hidden, lctx);
}

}  // namespace forge
