#pragma once

#include "tensor.h"

namespace forge {
namespace ops {

TensorPtr silu(const TensorPtr& x);
TensorPtr gelu(const TensorPtr& x);
TensorPtr gelu_tanh(const TensorPtr& x);
TensorPtr sigmoid(const TensorPtr& x);

}  // namespace ops
}  // namespace forge
