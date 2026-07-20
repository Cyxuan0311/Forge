// Explicit template instantiations for typed GEMV kernels + dispatch tables.
// Each template instantiation generates a separate set of kernel code,
// allowing nvcc to compile them in parallel if split into separate files
// (see tools/generate_gemv_instances.py).

#include "cuda_gemv_tmpl.cuh"
#include "cuda_gemv.h"  // for launch_gemv_q4_0_transB (special dispatch)

namespace forge {
namespace cuda {

// ============================================================================
// Explicit template instantiation macros
// ============================================================================

#define INST_GEMV(DT)                                                       \
    template void launch_gemv_typed_transB<DT>(                             \
        const float*, const void*, float*, int, int, cudaStream_t);         \
    template void launch_gemv_typed_transB_batch<DT>(                       \
        const float*, const void*, float*, int, int, int, cudaStream_t);

INST_GEMV(DataType::Q4_0)
INST_GEMV(DataType::Q4_1)
INST_GEMV(DataType::Q4_K)
INST_GEMV(DataType::Q8_0)
INST_GEMV(DataType::Q3_K)
INST_GEMV(DataType::Q6_K)

#undef INST_GEMV

// ============================================================================
// Function pointer dispatch tables
// Indexed by DataType enum value. nullptr for unsupported types.
// ============================================================================

const GemvFn gemv_dispatch[18] = {
    /* FP32=0  */ nullptr,
    /* FP16=1  */ nullptr,
    /* Q4_0=2  */ launch_gemv_q4_0_transB,                          // special smem/splitK
    /* Q4_1=3  */ launch_gemv_typed_transB<DataType::Q4_1>,
    /* Q4_K=4  */ launch_gemv_typed_transB<DataType::Q4_K>,
    /* INT8=5  */ nullptr,
    /* INT32=6 */ nullptr,
    /* Q8_0=7  */ launch_gemv_typed_transB<DataType::Q8_0>,
    /* Q5_0=8  */ nullptr,
    /* Q5_1=9  */ nullptr,
    /* Q2_K=10 */ nullptr,
    /* Q3_K=11 */ launch_gemv_typed_transB<DataType::Q3_K>,
    /* Q5_K=12 */ nullptr,
    /* Q6_K=13 */ launch_gemv_typed_transB<DataType::Q6_K>,
    /* IQ2_S=14*/ nullptr,
    /* BF16=15 */ nullptr,
    /* IQ2_XXS=16 */ nullptr,
    /* IQ4_NL=17  */ nullptr,
};

const GemvBatchFn gemv_batch_dispatch[18] = {
    /* FP32=0  */ nullptr,
    /* FP16=1  */ nullptr,
    /* Q4_0=2  */ launch_gemv_typed_transB_batch<DataType::Q4_0>,
    /* Q4_1=3  */ launch_gemv_typed_transB_batch<DataType::Q4_1>,
    /* Q4_K=4  */ launch_gemv_typed_transB_batch<DataType::Q4_K>,
    /* INT8=5  */ nullptr,
    /* INT32=6 */ nullptr,
    /* Q8_0=7  */ launch_gemv_typed_transB_batch<DataType::Q8_0>,
    /* Q5_0=8  */ nullptr,
    /* Q5_1=9  */ nullptr,
    /* Q2_K=10 */ nullptr,
    /* Q3_K=11 */ launch_gemv_typed_transB_batch<DataType::Q3_K>,
    /* Q5_K=12 */ nullptr,
    /* Q6_K=13 */ launch_gemv_typed_transB_batch<DataType::Q6_K>,
    /* IQ2_S=14*/ nullptr,
    /* BF16=15 */ nullptr,
    /* IQ2_XXS=16 */ nullptr,
    /* IQ4_NL=17  */ nullptr,
};

}  // namespace cuda
}  // namespace forge
