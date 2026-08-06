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
INST_GEMV(DataType::Q2_K)
INST_GEMV(DataType::Q3_K)
INST_GEMV(DataType::Q5_K)
INST_GEMV(DataType::Q6_K)

#undef INST_GEMV

// ============================================================================
// Function pointer dispatch tables
// Indexed by DataType enum value. nullptr for unsupported types.
// ============================================================================

const GemvFn gemv_dispatch[20] = {
    /* FP32=0  */ nullptr,
    /* FP16=1  */ nullptr,
    /* Q4_0=2  */ launch_gemv_q4_0_q8_1,                          // Q8_1+dp4a
    /* Q4_1=3  */ launch_gemv_q4_1_q8_1,                            // Q8_1+dp4a (Phase 5)
    /* Q4_K=4  */ launch_gemv_q4_k_q8_1,                            // special Q8_1+dp4a
    /* INT8=5  */ nullptr,
    /* INT32=6 */ nullptr,
    /* Q8_0=7  */ launch_gemv_q8_0_q8_1,                            // Q8_1+dp4a (Phase 5)
    /* Q5_0=8  */ launch_gemv_q5_0_q8_1,                             // Q8_1+dp4a (Phase 5)
    /* Q5_1=9  */ launch_gemv_q5_1_q8_1,                             // Q8_1+dp4a (Phase 5)
    /* Q2_K=10 */ launch_gemv_q2_k_q8_1,                             // special Q8_1+dp4a
    /* Q3_K=11 */ launch_gemv_q3_k_smem,                            // special smem+dp4a
    /* Q5_K=12 */ launch_gemv_q5_k_q8_1,                             // Q8_1+dp4a (Phase 5)
    /* Q6_K=13 */ launch_gemv_q6_k_q8_1,                             // Q8_1+dp4a (Phase 5)
    /* IQ2_S=14  */ launch_gemv_iq2_s_q8_1,                          // Q8_1+dp4a (Phase 5)
    /* BF16=15  */ nullptr,
    /* IQ2_XXS=16 */ launch_gemv_iq2_xxs_q8_1,                      // Q8_1+dp4a (Phase 5)
    /* IQ4_NL=17  */ launch_gemv_iq4_nl_q8_1,                      // Q8_1+dp4a (Phase 5)
    /* IQ2_XS=18 */ nullptr,                       // M=1: dequant fallback is faster (low GEMV occupancy at K=4096)
    /* IQ3_S=19  */ nullptr,                        // M=1: dequant fallback is faster (low GEMV occupancy at K=4096)
};

const GemvBatchFn gemv_batch_dispatch[20] = {
    /* FP32=0  */ nullptr,
    /* FP16=1  */ nullptr,
    /* Q4_0=2  */ launch_gemv_q4_0_q8_1_batch,                          // Q8_1+dp4a (Phase 5)
    /* Q4_1=3  */ launch_gemv_q4_1_q8_1_batch,                        // Q8_1+dp4a (Phase 5)
    /* Q4_K=4  */ launch_gemv_q4_k_q8_1_batch,                          // Q8_1+dp4a (Phase 5)
    /* INT8=5  */ nullptr,
    /* INT32=6 */ nullptr,
    /* Q8_0=7  */ launch_gemv_q8_0_q8_1_batch,                        // Q8_1+dp4a (Phase 5)
    /* Q5_0=8  */ launch_gemv_q5_0_q8_1_batch,                         // Q8_1+dp4a (Phase 5)
    /* Q5_1=9  */ launch_gemv_q5_1_q8_1_batch,                         // Q8_1+dp4a (Phase 5)
    /* Q2_K=10 */ launch_gemv_q2_k_q8_1_batch,                           // Q8_1+dp4a (Phase 5)
    /* Q3_K=11 */ launch_gemv_q3_k_q8_1_batch,                           // Q8_1+dp4a (Phase 5)
    /* Q5_K=12 */ launch_gemv_q5_k_q8_1_batch,                           // Q8_1+dp4a (Phase 5)
    /* Q6_K=13 */ launch_gemv_q6_k_q8_1_batch,                           // Q8_1+dp4a (Phase 5)
    /* IQ2_S=14  */ launch_gemv_iq2_s_q8_1_batch,                        // Q8_1+dp4a (Phase 5)
    /* BF16=15  */ nullptr,
    /* IQ2_XXS=16 */ launch_gemv_iq2_xxs_q8_1_batch,                    // Q8_1+dp4a (Phase 5)
    /* IQ4_NL=17  */ launch_gemv_iq4_nl_q8_1_batch,                    // Q8_1+dp4a (Phase 5)
    /* IQ2_XS=18 */ launch_gemv_iq2_xs_q8_1_batch,                   // Q8_1+dp4a
    /* IQ3_S=19  */ launch_gemv_iq3_s_q8_1_batch,                    // Q8_1+dp4a
};

}  // namespace cuda
}  // namespace forge
