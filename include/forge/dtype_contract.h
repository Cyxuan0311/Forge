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

// Dtypes an operator may consume without a dtype-specific kernel.
//
// Phase 6-C-1: WHITELIST form (was a blacklist `!FP16 && !BF16`). The blacklist
// shape meant every future dtype addition defaulted to "native execution" --
// i.e. the 6-B-0 silent-misread failure mode would replay for each new dtype
// (FP8 would have slipped straight through every gate). Whitelist semantics:
// unknown dtypes default to NOT native and are rejected loudly until an
// explicit landing here retires them op-by-op (op_supports_lowp).
inline bool is_native_execution_dtype(DataType dt) {
    switch (dt) {
        case DataType::FP32:
        case DataType::INT8:
        case DataType::INT32:
        case DataType::Q4_0:
        case DataType::Q4_1:
        case DataType::Q4_K:
        case DataType::Q8_0:
        case DataType::Q5_0:
        case DataType::Q5_1:
        case DataType::Q2_K:
        case DataType::Q3_K:
        case DataType::Q5_K:
        case DataType::Q6_K:
        case DataType::IQ2_S:
        case DataType::IQ2_XXS:
        case DataType::IQ2_XS:
        case DataType::IQ3_S:
        case DataType::IQ4_NL:
        case DataType::IQ4_XS:
            return true;
        case DataType::FP16:
        case DataType::BF16:
        case DataType::FP8_E4M3:
            return false;  // lowp: need a real dtype-specific kernel (op_supports_lowp)
    }
    return false;  // unknown future dtypes: reject loudly (whitelist semantics)
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
// 6-A-8.0: ADD + MUL whitelisted (native FP16 elementwise: FP16 storage, FP32
//        per-element compute, FP16 output) -- the residual-add and gate*up
//        nodes required by the 6-A-8 end-to-end FP16 loop. BF16 still rejected
//        downstream by the CUDA dtype routing (6-B).
// Phase 2 round 5 (FLASH_ATTN_GQA prefill amendment): the production prefill
//        entry routes hd=64 causal no-mask through the wmma TensorCore kernel
//        (Br=64/Bc=16/4 warps). One documented amendment vs the 6-A-7
//        vectorized baseline: PV's P operand is STORED f16 (new rounding,
//        <= 2^-11 relative per element); QK^T products stay exact in f32
//        (f16*f16 fits f32), online softmax and PV accumulation remain f32.
//        Every other scope (other head dims, masks, non-causal) keeps the
//        vectorized kernel and the original contract.
inline bool op_supports_lowp(OpType op) {
    switch (op) {
        case OpType::RMS_NORM:
        case OpType::SILU:
        case OpType::GELU:
        case OpType::GELU_TANH:
        case OpType::MUL_MAT_TRANSB:
        case OpType::FLASH_ATTN_GQA:
        case OpType::ADD:
        case OpType::MUL:
            return true;
        default:
            return false;
    }
}

// 6-C-1: FP8 retirement ledger -- the FP8 analogue of bf16_retired_ops, but at
// the CONTRACT layer instead of the routing layer. Rationale: for BF16 the gate
// (op_supports_lowp) deliberately passes and per-site require_no_bf16 calls
// reject -- that two-layer split exists because BF16 retirement happened
// op-by-op while whitelisted ops kept passing. FP8 starts fully closed: no
// native FP8 kernel exists anywhere (6-C-0 audit), so the contract itself
// rejects FP8 for every op and require_dtype_supported throws at every gated
// site (eager entries, dst-injection CUDA fallbacks, matmul entry). Retirement
// is per-op as 6-C-2..6-C-6 land real kernels (see op_supports_fp8 below).
// Every FP8 byte reaching an FP32/FP16 kernel is a bug.
// Retirement protocol (mirrors bf16_retired_ops / 6-B): an op leaves this
// ledger TOGETHER with its native FP8 kernel + routing change landing in the
// same phase -- kernel, routing, ledger are one atomic retirement.
//
// Retired:
//   - RMS_NORM (6-C-2): launch_rms_norm_fp8 / launch_rms_norm_unweighted_fp8
//     (FP8 storage, FP32 reduction, FP8 output; FP8 x + FP8 weight only --
//     mixed dtypes still fail loudly). Eager ops::rms_norm + dst injection
//     both route FP8.
//   - SILU / GELU / GELU_TANH (6-C-3): launch_silu_fp8 / launch_gelu_fp8 /
//     launch_gelu_tanh_fp8 (FP8 storage, FP32 transcendental compute, FP8
//     output). The gated variants silu_multiply / gelu_multiply reuse the SILU /
//     GELU gate (no new enum). Eager ops::* + SILU dst injection route FP8;
//     GELU/GELU_TANH/gated take the legacy W2 eager path.
inline bool op_supports_fp8(OpType op) {
    switch (op) {
        case OpType::RMS_NORM:    // 6-C-2
        case OpType::SILU:        // 6-C-3 (+ gated silu_multiply reuses the gate)
        case OpType::GELU:        // 6-C-3 (+ gated gelu_multiply reuses the gate)
        case OpType::GELU_TANH:   // 6-C-3
        case OpType::MUL_MAT_TRANSB:  // 6-C-4.1d (W8A8 FP8 GEMM, FP32 out)
            return true;
        default:
            return false;
    }
}

inline bool op_supports_dtype(OpType op, DataType dt) {
    if (dt == DataType::FP8_E4M3) {
        return op_supports_fp8(op);
    }
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

// Phase 6-B-1: explicit BF16 gate. `op_supports_lowp()` deliberately does NOT
// distinguish FP16 from BF16 (an op is either "lowp-aware" or not), so every
// whitelisted op passes BF16 through the gate above. But no BF16 kernel exists
// yet (6-B-2..6-B-6 land them one at a time), so each routing site must reject
// BF16 loudly BEFORE any float* cast -- never let BF16 bytes reach an FP32 or
// FP16 kernel. Semantics: no-op for every non-BF16 dtype (zero behavior change
// for FP16/FP32/quantized); throws for BF16. Retired op-by-op as native
// __nv_bfloat16 kernels land (each retirement whitelists one op here).
[[noreturn]] inline void throw_bf16_unsupported(const char* where, OpType op) {
    throw std::runtime_error(
        std::string(where) + ": dtype 'bf16' has no native kernel for op '" +
        op_type_name(op) +
        "'. Phase 6-B-1 rejects BF16 explicitly instead of reinterpreting the buffer as "
        "float (which silently produced wrong results). Convert to FP32 at the operator "
        "boundary, or wait for the native BF16 kernel -- see docs/optimization_plan_zh.md "
        "Phase 6-B.");
}

inline void require_no_bf16(const char* where, OpType op, DataType dt) {
    if (dt == DataType::BF16) {
        throw_bf16_unsupported(where, op);
    }
}

// 6-C-1: explicit FP8 routing gate -- mirrors require_no_bf16. Covers the sites
// that do NOT call require_dtype_supported before a float* cast (dst-injection
// CPU fallbacks, CPU attention, AttentionExecutor FP32 paths): the contract
// ledger (op_supports_fp8) only protects sites that go through
// require_dtype_supported, so those bare routing sites need this gate.
// Semantics: no-op for every non-FP8 dtype (zero behavior change for
// FP32/FP16/BF16/quantized); throws for FP8_E4M3. Retired op-by-op alongside
// the op_supports_fp8 ledger as native FP8 kernels land (6-C-2+).
[[noreturn]] inline void throw_fp8_unsupported(const char* where, OpType op) {
    throw std::runtime_error(
        std::string(where) + ": dtype 'fp8_e4m3' has no native kernel for op '" +
        op_type_name(op) +
        "'. Phase 6-C-1 rejects FP8 explicitly instead of reinterpreting the buffer as "
        "float/half (which silently produced wrong results). Convert to FP32 at the "
        "operator boundary, or wait for the native FP8 kernel -- see "
        "docs/optimization_plan_zh.md Phase 6-C.");
}

inline void require_no_fp8(const char* where, OpType op, DataType dt) {
    if (dt == DataType::FP8_E4M3) {
        throw_fp8_unsupported(where, op);
    }
}

}  // namespace forge
