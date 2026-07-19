#pragma once

// LlamaEngine is now a thin wrapper around GenericEngine.
// All GQA-based architectures (llama, mistral, qwen, qwen2, qwen3vl, yi, deepseek)
// are handled by GenericEngine's strategy sub-functions driven by ModelConfig.

#include "forge/engines/generic_engine.h"

namespace forge {

using LlamaEngine = GenericEngine;

}  // namespace forge
