#pragma once

#include "tensor.h"

namespace forge {
namespace ops {

TensorPtr rope(const TensorPtr& q, const TensorPtr& k, int64_t pos, float theta);

// In-place interleaved RoPE on a single vector of `dim` floats.
void rope_interleaved(float* vec, int dim, int64_t pos, float theta);

}  // namespace ops
}  // namespace forge
