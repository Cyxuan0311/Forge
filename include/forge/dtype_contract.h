#pragma once

// =========================================================================
// Phase 6-A-1: dtype contract hardening.
//
// Contract: a DataType reaching an operator means "a kernel can legally consume
// this data". Until an operator gains a native low-precision kernel it must
// FAIL LOUDLY on FP16/BF16 instead of reinterpreting the bytes as float.
//
// Why: silently reading FP16 bytes through a `const float*` is the worst failure
// mode in the 6-A-0 audit (finding #10 and the FP16 dispatch slot #8) -- no
// crash, no error, just wrong numbers that surface much later as a parity diff.
//
// Extension protocol for 6-A-2..6-A-8: when an operator grows a real
// __half / __nv_bfloat16 kernel, whitelist it in `op_supports_lowp()` below.
// Call sites keep calling `require_dtype_supported()` unchanged, so coverage
// grows one line at a time instead of by editing every operator.
// =========================================================================

#include <stdexcept>
#include <string>

#include "forge/op_enum.h"
#include "forge/types.h"

namespace forge {

// Dtypes an operator may consume without a dtype-specific kernel. FP32 and the
// quantized formats are real execution dtypes (kernels are written for them);
// FP16/BF16 are not -- they would currently be reinterpreted as float.
inline bool is_native_execution_dtype(DataType dt) {
    return dt != DataType::FP16 && dt != DataType::BF16;
}

// Per-operator escape hatch -- the single extension point for Phase 6-A..6-A-8.
// 6-A-1: empty (no CUDA operator had a native low-precision kernel).
// 6-A-2: RMS_NORM whitelisted -- real __half kernel (FP16 storage/arithmetic
//        with an FP32 reduction).
// 6-A-3: SILU whitelisted -- real __half kernel (FP16 storage, FP32 sigmoid).
// 6-A-4: GELU + GELU_TANH whitelisted (unary activation family, real __half
//        kernels, FP16 storage + FP32 erf/tanh). The gated variants
//        silu_multiply / gelu_multiply reuse the SILU / GELU gate -- no new enum
//        needed. All: BF16 still has no kernel; it passes this gate (op is
//        "lowp-aware") but is rejected downstream by the CUDA dtype routing --
//        BF16-native lands with 6-B.
// 6-A-5: MUL_MAT_TRANSB whitelisted (native FP16 matmul, real __half GEMM with
//        FP32 accumulate; output stored FP16). The non-transB MUL_MAT stays
//        audited -- no native FP16 row-major-B path yet, so it keeps rejecting
//        FP16 loudly. BF16 still rejected downstream by the CUDA routing (6-B).
// 6-A-7: FLASH_ATTN_GQA whitelisted (native FP16 attention: FP16 Q/K/V
//        storage, FP32 QK^T/softmax/PV, FP16 output). SOFT_MAX stays audited --
//        there is no standalone softmax kernel (attention softmax is fused
//        online). BF16 still rejected downstream (6-B).
inline bool op_supports_lowp(OpType op) {
    switch (op) {
        case OpType::RMS_NORM:
        case OpType::SILU:
        case OpType::GELU:
        case OpType::GELU_TANH:
        case OpType::MUL_MAT_TRANSB:
        case OpType::FLASH_ATTN_GQA:
            return true;
        default:
            return false;
    }
}

inline bool op_supports_dtype(OpType op, DataType dt) {
    if (is_native_execution_dtype(dt)) {
        return true;
    }
    return op_supports_lowp(op);
}

[[noreturn]] inline void throw_dtype_unsupported(const char* where, OpType op, DataType dt) {
    throw std::runtime_error(
        std::string(where) + ": dtype '" + dtype_name(dt) + "' has no native kernel for op '" +
        op_type_name(op) +
        "'. Phase 6-A-1 rejects this explicitly instead of reinterpreting the buffer as float "
        "(which silently produced wrong results). Convert to FP32 at the operator boundary, or "
        "implement the native kernel -- see docs/optimization_plan_zh.md Phase 6-A.");
}

// Explicit failure at the operator boundary: throw / return Unsupported, never
// "fallback -> float".
inline void require_dtype_supported(const char* where, OpType op, DataType dt) {
    if (!op_supports_dtype(op, dt)) {
        throw_dtype_unsupported(where, op, dt);
    }
}

}  // namespace forge
