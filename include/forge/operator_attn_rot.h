#pragma once

#include "tensor.h"

namespace forge {
namespace ops {

// In-place Hadamard (Walsh-Hadamard) rotation along the last dimension of `x`,
// applied independently to every `order`-wide sub-chunk. `order` must divide the
// last dimension exactly and be a power of two in 32..512; anything else is
// refused (returns false, data untouched) rather than approximated.
//
// Self-inverse: calling this twice returns the original tensor, because the
// Hadamard matrix is symmetric. So the same op both rotates an activation and
// un-rotates the attention output.
bool hadamard_rotate(const TensorPtr& x, int order);

}  // namespace ops
}  // namespace forge
