// =========================================================================
// matmul_dispatch.h — CUDA dispatch table lookups (internal)
//
// Single source of truth for "does this dtype have a GEMV / batched GEMV /
// MMQ / matrix-dequant kernel?" queries. Shared by the unified kernel
// selector (kernel_selector.cpp) and the matmul launchers (matmul_cuda.cpp).
// The dispatch arrays themselves are declared in forge/cuda_kernels.h and
// defined in cuda_gemv_instances.cu / cuda_mmq.cu.
// =========================================================================

#pragma once

#include "forge/cuda_kernels.h"

namespace forge {
namespace ops {
namespace internal {

cuda::GemvFn lookup_gemv(DataType dt);
cuda::GemvBatchFn lookup_gemv_batch(DataType dt);
cuda::MmqFn lookup_mmq(DataType dt);

using DequantMatrixFn = void (*)(const void*, float*, int, int, cudaStream_t);

// FP32-domain whole-matrix dequant kernels, indexed by dtype. nullptr means
// the dtype has no CUDA matrix-dequant kernel.
DequantMatrixFn lookup_dequant_matrix(DataType dt);

}  // namespace internal
}  // namespace ops
}  // namespace forge
