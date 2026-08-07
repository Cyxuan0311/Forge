#pragma once
// PPC64 VSX I-quant kernels stub.
// I-quant (IQ2_S, IQ2_XS, IQ3_S, IQ4_NL) fused dot-product and GEMV kernels
// are not yet ported to PPC64. This stub provides the necessary declarations
// so the code compiles.
// To be implemented: vec_dot_iq2_s_q8_0, etc.

#ifdef USE_VSX
#include <altivec.h>
#endif
