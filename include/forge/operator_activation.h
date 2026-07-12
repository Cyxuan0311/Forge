#pragma once

#include "tensor.h"

namespace forge {
namespace ops {

TensorPtr silu(const TensorPtr& x);
TensorPtr gelu(const TensorPtr& x);
TensorPtr gelu_tanh(const TensorPtr& x);

} // namespace ops
} // namespace forge
