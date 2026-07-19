#pragma once

// FalconEngine is now a thin wrapper around GenericEngine.
// Falcon architecture (parallel residual, LayerNorm) is handled by
// GenericEngine's strategy sub-functions driven by ModelConfig.

#include "forge/engines/generic_engine.h"

namespace forge {

using FalconEngine = GenericEngine;

}  // namespace forge
