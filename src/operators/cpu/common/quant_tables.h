#pragma once
// I-quant dequantization lookup tables. Definitions live in quant_tables.cpp.
// Both scalar dequant and SIMD kernels reference these; CUDA kernels extern them.

#include <cstdint>

namespace forge {
namespace ops {

extern const uint64_t iq2s_grid[1024];
extern const uint64_t iq2xxs_grid[256];
extern const int8_t kvalues_iq4nl[16];
extern const uint64_t iq2xs_grid[512];
extern const uint32_t iq3s_grid[512];

extern const uint8_t kmask_iq2xs[8];
extern const uint8_t ksigns_iq2xs[128];

}  // namespace ops
}  // namespace forge
