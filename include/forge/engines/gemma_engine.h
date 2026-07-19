#pragma once

// GemmaEngine is now a thin wrapper around GenericEngine.
// Gemma/Gemma2 architectures are handled by GenericEngine's strategy
// sub-functions driven by ModelConfig.

#include "forge/engines/generic_engine.h"

namespace forge {

using GemmaEngine = GenericEngine;

}  // namespace forge
