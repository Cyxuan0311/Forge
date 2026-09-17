#pragma once

#include "tensor.h"

namespace forge {
namespace ops {

TensorPtr add(const TensorPtr& a, const TensorPtr& b);
TensorPtr multiply(const TensorPtr& a, const TensorPtr& b);
TensorPtr subtract(const TensorPtr& a, const TensorPtr& b);
TensorPtr scale(const TensorPtr& x, float s);
TensorPtr silu_multiply(const TensorPtr& gate, const TensorPtr& up);
TensorPtr gelu_multiply(const TensorPtr& gate, const TensorPtr& up);
TensorPtr softmax(const TensorPtr& x, float temperature = 1.0f);

// Generic matrix-vector product: mat [M,K] @ vec [K] -> [M]. SIMD dot product.
TensorPtr matvec(const TensorPtr& mat, const TensorPtr& vec);

// L2-normalize each row of `x` over its last dimension to unit norm
// (row /= sqrt(sum(row^2) + eps)). SIMD accelerated.
TensorPtr l2_normalize(const TensorPtr& x, float eps);

}  // namespace ops
}  // namespace forge
