#pragma once

#include "tensor.h"

namespace forge {
namespace ops {

TensorPtr rms_norm(const TensorPtr& x, const TensorPtr& weight, float eps);
TensorPtr layer_norm(const TensorPtr& x, const TensorPtr& weight, const TensorPtr& bias, float eps);

}  // namespace ops
}  // namespace forge
