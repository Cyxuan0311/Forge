#pragma once

#include <string>
#include <utility>
#include <vector>

namespace forge::inspect {

// One step of an architecture's operator pipeline: (OpType, human label).
struct OpStep {
    std::string op;
    std::string label;
};

// Describes the operator pipeline Forge uses to execute a given architecture.
// Derived from the graph builders (src/inference/graph/*) and engines.
struct ArchOpPipeline {
    std::vector<OpStep> pre_steps;     // before the layer stack (embedding etc.)
    std::vector<OpStep> layer_steps;   // per-transformer-layer steps
    std::vector<OpStep> post_steps;    // after the layer stack (output norm + proj)
};

// Returns the pipeline description for an arch, or nullptr if unknown.
const ArchOpPipeline* pipeline_for(const std::string& arch);

// True if Forge can load/execute this architecture (config parser + capability
// registered, or a dedicated engine).
bool arch_supported(const std::string& arch);

// All registered architectures (config parsers + capabilities), sorted.
std::vector<std::string> supported_archs();

}  // namespace forge::inspect
