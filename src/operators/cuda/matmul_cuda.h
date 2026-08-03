#pragma once

#include "forge/tensor.h"

#ifdef USE_CUDA

namespace forge {
namespace ops {

// CUDA dispatch entry points for the matmul family.
// Phase 4: these were moved out of src/operators/cpu/matmul.cpp so that CPU and CUDA
// dispatch no longer live in the same translation unit.

void cuda_apply_bias(const TensorPtr& out, const TensorPtr& bias, int M, int N);

// Dequantize a device-resident quantized matrix into a new FP32 CUDA tensor.
// `rows` x `cols` describes the logical row-major layout of the quantized tensor.
// Throws when the dtype has no CUDA dequant kernel (no host staging fallback).
TensorPtr cuda_dequantize_matrix(const TensorPtr& b, int rows, int cols);

void cuda_matmul(const TensorPtr& a, const TensorPtr& b_fp32, const TensorPtr& out, int M, int K,
                 int N);

void cuda_matmul_transB(const TensorPtr& a, const TensorPtr& b, const TensorPtr& out, int M, int K,
                        int N);

void cuda_matmul_transB_dual(const TensorPtr& a, const TensorPtr& b1, const TensorPtr& b2,
                             const TensorPtr& out, int M, int K, int N1, int N2);

// Returns the tensor holding the result: either `out` (written in place) or a new tensor
// produced by the separate matmul + silu_multiply path.
TensorPtr cuda_ffn_up_fused(const TensorPtr& input, const TensorPtr& w1, const TensorPtr& w3,
                            const TensorPtr& out, int M, int K, int intermediate_dim);

}  // namespace ops
}  // namespace forge

#endif  // USE_CUDA
