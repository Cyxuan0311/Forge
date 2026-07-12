#pragma once

#include "tensor.h"

namespace forge {
namespace ops {

TensorPtr add(const TensorPtr& a, const TensorPtr& b);
TensorPtr multiply(const TensorPtr& a, const TensorPtr& b);
TensorPtr silu_multiply(const TensorPtr& gate, const TensorPtr& up);
TensorPtr softmax(const TensorPtr& x, float temperature = 1.0f);

}  // namespace ops
}  // namespace forge
