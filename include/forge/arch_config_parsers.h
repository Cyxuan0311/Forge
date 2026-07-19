#pragma once

// Named config parser functions for each architecture.
// These are referenced by arch_registrations.cpp for FORGE_REGISTER_ARCH.

#include "forge/model.h"

namespace forge {

ModelConfig parse_llama_config(ModelLoader& loader, const std::string& arch);
ModelConfig parse_mistral_config(ModelLoader& loader, const std::string& arch);
ModelConfig parse_qwen_config(ModelLoader& loader, const std::string& arch);
ModelConfig parse_qwen2_config(ModelLoader& loader, const std::string& arch);
ModelConfig parse_qwen3vl_config(ModelLoader& loader, const std::string& arch);
ModelConfig parse_yi_config(ModelLoader& loader, const std::string& arch);
ModelConfig parse_deepseek_config(ModelLoader& loader, const std::string& arch);
ModelConfig parse_deepseek_v2_config(ModelLoader& loader, const std::string& arch);
ModelConfig parse_deepseek_v3_config(ModelLoader& loader, const std::string& arch);
ModelConfig parse_qwen35_config(ModelLoader& loader, const std::string& arch);
ModelConfig parse_phi3_config(ModelLoader& loader, const std::string& arch);
ModelConfig parse_gemma_config(ModelLoader& loader, const std::string& arch);
ModelConfig parse_gemma2_config(ModelLoader& loader, const std::string& arch);
ModelConfig parse_falcon_config(ModelLoader& loader, const std::string& arch);
ModelConfig parse_gemma4_config(ModelLoader& loader, const std::string& arch);

}  // namespace forge
