#pragma once

// Named weight init functions for each architecture category.
// These are referenced by arch_registrations.cpp for FORGE_REGISTER_ARCH.

#include "forge/model.h"

namespace forge {

void init_gqa_layer_weights(LayerWeightInitContext& ctx);
void init_mla_layer_weights(LayerWeightInitContext& ctx);
void init_qwen35_layer_weights(LayerWeightInitContext& ctx);
void init_falcon_layer_weights(LayerWeightInitContext& ctx);
void init_gemma4_layer_weights(LayerWeightInitContext& ctx);

}  // namespace forge
