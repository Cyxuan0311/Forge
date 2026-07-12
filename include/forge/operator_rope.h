#pragma once

#include "tensor.h"

namespace forge {
namespace ops {

TensorPtr rope(const TensorPtr& q, const TensorPtr& k, int64_t pos, float theta);

}  // namespace ops
}  // namespace forge
