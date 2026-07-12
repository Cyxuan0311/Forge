#pragma once

#include "tensor.h"

namespace forge {
namespace ops {

TensorPtr embedding(const TensorPtr& weight, const TensorPtr& indices,
                    const TensorPtr& fp32_cache = nullptr);

} // namespace ops
} // namespace forge
