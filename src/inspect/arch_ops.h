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

// Architecture families. Architectures that execute identically share one.
enum class ArchFamily {
    Gqa,      // standard GQA/MHA decoder: llama, mistral, qwen2, phi3, ...
    Gemma,    // gemma / gemma2 (norm placement differs)
    Mla,      // deepseek multi-head latent attention
    Moe,      // mixture of experts
    Hybrid,   // qwen35: interleaved attention and linear-attention layers
    Unknown,
};

// The ONE place arch -> family is decided.
//
// This used to be a hand-written if/else chain in two files: sniff_arch() built
// the human-facing family string from one list, and pipeline_for() picked the
// operator pipeline from a second, near-identical list. They had already drifted
// (qwen3vl is in the GQA group; nothing forced them to agree), and every new
// architecture meant editing both. Both now go through here, so a new arch is one
// row.
ArchFamily family_for_arch(const std::string& arch);

// Stable lower-case name for a family ("gqa", "gemma", "mla", "moe", "hybrid").
// Unknown maps to the arch name itself when one is available.
const char* family_name(ArchFamily f);

// Returns the pipeline description for an arch, or nullptr if unknown.
const ArchOpPipeline* pipeline_for(const std::string& arch);

// True if Forge can load/execute this architecture (config parser + capability
// registered, or a dedicated engine).
bool arch_supported(const std::string& arch);

// All registered architectures (config parsers + capabilities), sorted.
std::vector<std::string> supported_archs();

}  // namespace forge::inspect
