// =========================================================================
// test_dtype_contract.cpp — Phase 6-A-1: dtype contract + ComputeDTypePolicy
//
// Pure host logic (both headers are header-only): no GPU, no kernels launched.
//
// Covers:
//   1. ComputeDTypePolicy defaults to the FP32-everywhere configuration, and
//      exposes the four dimensions Phase 6 needs (weight / activation /
//      accumulator / authoritative KV).
//   2. FP32 and the quantized formats are legal execution dtypes. FP16/BF16 are
//      NOT — they would be reinterpreted as float (6-A-0 #10). Exception:
//      RMS_NORM (6-A-2), SILU (6-A-3), GELU / GELU_TANH (6-A-4) and MUL_MAT_TRANSB
//      (6-A-5, native FP16 matmul) are whitelisted (real __half kernels), so the
//      contract accepts FP16/BF16 for them and the CUDA routing picks the native
//      kernel.
//   3. require_dtype_supported() FAILS LOUDLY for FP16/BF16 on every operator
//      and is a no-op for FP32/Q4_K — no silent "fallback to float".
//   3b. (6-B-1) require_no_bf16() FAILS LOUDLY for BF16 on every operator
//      (audited + whitelisted + ROPE/MUL_MAT) and is a no-op for FP32/FP16/
//      quantized — the routing-layer half of the two-layer BF16 contract.
//   3c. (6-C-1) FP8_E4M3 is rejected for EVERY operator at the contract layer
//      itself (op_supports_fp8 ledger is empty), and is_native_execution_dtype
//      is now a whitelist so unknown future dtypes default to loud rejection.
//      require_no_fp8 covers the bare routing sites the ledger cannot reach.
//   4. The extension point is a single function (op_supports_lowp): 6-A-2 landed
//      a real __half RMSNorm (RMS_NORM whitelisted) and 6-A-3 landed a real
//      __half SiLU (SILU whitelisted), so this test now records both the
//      still-rejected ops and the accepted ones.
// =========================================================================

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "forge/compute_dtype_policy.h"
#include "forge/dtype_contract.h"
#include "forge/types.h"

namespace {

int failures = 0;

#define CHECK(cond)                                                     \
    do {                                                                \
        if (!(cond)) {                                                  \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                 \
        }                                                               \
    } while (0)

using forge::DataType;
using forge::OpType;

// Returns true when the contract rejects (dtype, op) with an explanatory error.
bool rejected(OpType op, DataType dt, std::string* msg = nullptr) {
    try {
        forge::require_dtype_supported("test_dtype_contract", op, dt);
        return false;
    } catch (const std::runtime_error& e) {
        if (msg) {
            *msg = e.what();
        }
        return true;
    }
}

// Returns true when the 6-B-1 BF16 gate rejects (op, dtype).
bool bf16_rejected(OpType op, DataType dt, std::string* msg = nullptr) {
    try {
        forge::require_no_bf16("test_dtype_contract", op, dt);
        return false;
    } catch (const std::runtime_error& e) {
        if (msg) {
            *msg = e.what();
        }
        return true;
    }
}

// Ops still without a native low-precision kernel (6-A-0 audit). RMS_NORM (6-A-2),
// SILU (6-A-3), GELU / GELU_TANH (6-A-4), MUL_MAT_TRANSB (6-A-5),
// FLASH_ATTN_GQA (6-A-7) and ADD / MUL (6-A-8.0) left this set once they gained
// a real __half kernel (see supported_lowp_ops).
const std::vector<OpType>& audited_ops() {
    static const std::vector<OpType> ops = {
        OpType::MUL_MAT, OpType::ROPE,
        OpType::SOFT_MAX, OpType::EMBEDDING};
    return ops;
}

// Ops that have gained a real native low-precision kernel (6-A-2+). The contract
// accepts FP16/BF16 for these; exact dtype routing happens downstream (CUDA).
const std::vector<OpType>& supported_lowp_ops() {
    static const std::vector<OpType> ops = {OpType::RMS_NORM, OpType::SILU, OpType::GELU,
                                            OpType::GELU_TANH, OpType::MUL_MAT_TRANSB,
                                            OpType::FLASH_ATTN_GQA, OpType::ADD, OpType::MUL};
    return ops;
}

bool is_supported(OpType op) {
    for (OpType s : supported_lowp_ops())
        if (s == op) return true;
    return false;
}

const std::vector<DataType>& quantized_dtypes() {
    static const std::vector<DataType> dts = {DataType::Q4_0, DataType::Q4_K, DataType::Q5_K,
                                             DataType::Q6_K, DataType::Q8_0};
    return dts;
}

void test_policy_defaults() {
    forge::ComputeDTypePolicy p;
    CHECK(p.weight == DataType::FP32);
    CHECK(p.activation == DataType::FP32);
    CHECK(p.accumulator == DataType::FP32);
    CHECK(p.norm_compute == DataType::FP32);   // 6-A-2: RMSNorm reduces in FP32
    CHECK(p.kv == DataType::FP32);
    CHECK(forge::is_fp32_policy(p));
    CHECK(forge::is_fp32_policy(forge::fp32_policy()));

    // A non-default policy is representable (this is what Phase 6-A-7 will
    // actually run: FP16 weights/activations, FP32 accumulation + FP32 norm).
    forge::ComputeDTypePolicy native_fp16;
    native_fp16.weight = DataType::FP16;
    native_fp16.activation = DataType::FP16;
    native_fp16.kv = DataType::FP32;
    CHECK(!forge::is_fp32_policy(native_fp16));
    CHECK(native_fp16.accumulator == DataType::FP32);   // accumulation stays FP32 by default
    CHECK(native_fp16.norm_compute == DataType::FP32);  // norm reduction stays FP32 even when activation=FP16

    // 6-B-5: the BF16 native-execution policy -- what the 6-B-2..6-B-5 kernels
    // actually implement (BF16 storage, FP32 accumulate/reduction, FP32 KV).
    // 6-B-6.7 consumes the kv slot: BF16 authoritative KV for decode.
    forge::ComputeDTypePolicy native_bf16 = forge::bf16_policy();
    CHECK(forge::is_bf16_policy(native_bf16));
    CHECK(!forge::is_fp32_policy(native_bf16));
    CHECK(native_bf16.weight == DataType::BF16);
    CHECK(native_bf16.activation == DataType::BF16);
    CHECK(native_bf16.accumulator == DataType::FP32);
    CHECK(native_bf16.norm_compute == DataType::FP32);
    CHECK(native_bf16.kv == DataType::BF16);  // 6-B-6.7: BF16 KV decode landed
}

void test_execution_dtypes() {
    CHECK(forge::is_native_execution_dtype(DataType::FP32));
    for (DataType dt : quantized_dtypes()) {
        CHECK(forge::is_native_execution_dtype(dt));
    }
    // The whole point of 6-A-1: FP16/BF16 are NOT yet execution dtypes.
    CHECK(!forge::is_native_execution_dtype(DataType::FP16));
    CHECK(!forge::is_native_execution_dtype(DataType::BF16));
}

void test_explicit_reject_no_silent_fallback() {
    for (OpType op : audited_ops()) {
        // These ops still have no native low-precision kernel: FP16/BF16 must be
        // rejected loudly, never silently reinterpreted as float.
        CHECK(is_supported(op) == false);
        std::string msg;
        CHECK(rejected(op, DataType::FP16, &msg));
        CHECK(rejected(op, DataType::BF16));
        CHECK(msg.find("has no native kernel") != std::string::npos);
        CHECK(msg.find(forge::op_type_name(op)) != std::string::npos);

        // FP32 and the quantized formats: unchanged, must not throw.
        CHECK(forge::op_supports_dtype(op, DataType::FP32));
        for (DataType dt : quantized_dtypes()) {
            CHECK(forge::op_supports_dtype(op, dt));
        }
    }
}

void test_lowp_accepted() {
    // For ops that have a real native kernel the contract must ACCEPT FP16/BF16;
    // the concrete dtype routing (FP16 vs BF16) happens downstream in the CUDA
    // layer, not at the contract gate.
    for (OpType op : supported_lowp_ops()) {
        CHECK(forge::op_supports_lowp(op));
        std::string msg;
        CHECK(!rejected(op, DataType::FP16, &msg));
        CHECK(!rejected(op, DataType::BF16));
    }
}

// 6-B-1: the BF16 gate (require_no_bf16) is the second layer of the contract.
// The gate (op_supports_lowp) deliberately passes BF16 for whitelisted ops, so
// every routing site must reject BF16 loudly before any float* cast. This test
// locks that two-layer split: BF16 throws via the BF16 gate for every audited
// AND every whitelisted op (+ ROPE/MUL_MAT), while FP32/quantized/FP16 pass the
// BF16 gate untouched (zero behavior change off the BF16 path).
// Ops whose routing sites no longer call require_no_bf16 because a real
// __nv_bfloat16 kernel owns the dtype (6-B-2+). Retirement is a call-site
// property, NOT a gate-function property: require_no_bf16 itself keeps throwing
// for every op (zero semantic change), and each retirement must land together
// with its routing change (kernel + eager branch + dst branch + graph gate).
// The executable proof of a retirement is the op's own native test (e.g.
// test_rms_norm_bf16 calls ops::rms_norm with BF16 and expects success -- it
// fails if any routing site still rejects); the bookkeeping below locks the
// list itself.
const std::vector<OpType>& bf16_retired_ops() {
    static const std::vector<OpType> ops = {
        OpType::RMS_NORM,  // 6-B-2: native BF16 RMSNorm (BF16 storage, FP32 reduction)
        OpType::SILU,      // 6-B-3: native BF16 SiLU (+ gated silu_multiply reuses the gate)
        OpType::GELU,      // 6-B-3: native BF16 GELU (+ gated gelu_multiply reuses the gate)
        OpType::GELU_TANH,  // 6-B-3: native BF16 GELU-tanh
        OpType::MUL_MAT_TRANSB,  // 6-B-4: native BF16 matmul (CUDA_R_16BF, FP32 accumulate)
        OpType::ADD,  // 6-B-5: native BF16 elementwise add (graph residual node)
        OpType::MUL,  // 6-B-5: native BF16 elementwise multiply (graph scale node)
        // 6-B-6: native BF16-Q decode (BF16 Q + FP32 KV, FP32 math, BF16 out).
        // Decode scope only -- same partial-scope precedent as MUL_MAT_TRANSB
        // (dual still rejects): BF16 prefill, BF16 K/V, and non-GQA BF16 all
        // keep loud rejection at their own routing sites.
        OpType::FLASH_ATTN_GQA,
    };
    return ops;
}

bool is_bf16_retired(OpType op) {
    for (OpType s : bf16_retired_ops())
        if (s == op) return true;
    return false;
}

void test_bf16_gate() {
    const std::vector<OpType> all_ops = {OpType::RMS_NORM, OpType::SILU, OpType::GELU,
                                         OpType::GELU_TANH, OpType::MUL_MAT_TRANSB,
                                         OpType::FLASH_ATTN_GQA, OpType::ADD, OpType::MUL,
                                         OpType::MUL_MAT, OpType::ROPE, OpType::SOFT_MAX,
                                         OpType::EMBEDDING};
    for (OpType op : all_ops) {
        // The gate function itself is unchanged by retirements: BF16 must throw
        // loudly for every op, with the op name and a BF16 marker in the message.
        std::string msg;
        CHECK(bf16_rejected(op, DataType::BF16, &msg));
        CHECK(msg.find(forge::op_type_name(op)) != std::string::npos);
        CHECK(msg.find("bf16") != std::string::npos);

        // Non-BF16 dtypes are untouched by this gate (no-op, never throws).
        CHECK(!bf16_rejected(op, DataType::FP32));
        CHECK(!bf16_rejected(op, DataType::FP16));
        for (DataType dt : quantized_dtypes()) {
            CHECK(!bf16_rejected(op, dt));
        }
    }

    // Consistency invariant for the two-layer split: whitelisted ops accept BF16
    // at the contract gate (lowp-aware) AND the gate function rejects it --
    // routing sites decide per op (reject or native dispatch). Retirements are
    // recorded in bf16_retired_ops (checked below), never by weakening this gate.
    for (OpType op : supported_lowp_ops()) {
        CHECK(forge::op_supports_dtype(op, DataType::BF16));  // gate passes ...
        CHECK(bf16_rejected(op, DataType::BF16));             // ... BF16 gate still throws
    }
}

void test_bf16_retirement() {
    // Every retired op must be whitelisted (a non-whitelisted op has no lowp
    // routing to retire in the first place). The routing half of each
    // retirement is proven by that op's native test, not here.
    for (OpType op : bf16_retired_ops()) {
        CHECK(forge::op_supports_lowp(op));
        CHECK(is_supported(op));
    }
    // 6-B-2 retired RMS_NORM; 6-B-3 retires SILU + GELU + GELU_TANH (gated
    // variants reuse the SILU / GELU gate, no new enum); 6-B-4 retires
    // MUL_MAT_TRANSB (native BF16 cuBLAS GEMM); 6-B-5 retires ADD + MUL
    // (native BF16 elementwise, eager + dst); 6-B-6 retires FLASH_ATTN_GQA
    // (native BF16-Q decode scope -- prefill / BF16-KV / MHA still reject).
    // Future 6-B steps append here op-by-op.
    CHECK(bf16_retired_ops().size() == 8);
    CHECK(is_bf16_retired(OpType::RMS_NORM));
    CHECK(is_bf16_retired(OpType::SILU));
    CHECK(is_bf16_retired(OpType::GELU));
    CHECK(is_bf16_retired(OpType::GELU_TANH));
    CHECK(is_bf16_retired(OpType::MUL_MAT_TRANSB));
    CHECK(is_bf16_retired(OpType::ADD));
    CHECK(is_bf16_retired(OpType::MUL));
    CHECK(is_bf16_retired(OpType::FLASH_ATTN_GQA));
}

void test_extension_point() {
    // The executable record of how much of Phase 6-A is real: RMS_NORM (6-A-2),
    // SILU (6-A-3), GELU / GELU_TANH (6-A-4), MUL_MAT_TRANSB (6-A-5, native FP16
    // matmul), FLASH_ATTN_GQA (6-A-7) and ADD / MUL (6-A-8.0, elementwise) are
    // whitelisted so far; everything else is still false.
    CHECK(forge::op_supports_lowp(OpType::RMS_NORM));
    CHECK(forge::op_supports_lowp(OpType::SILU));
    CHECK(forge::op_supports_lowp(OpType::GELU));
    CHECK(forge::op_supports_lowp(OpType::GELU_TANH));
    CHECK(forge::op_supports_lowp(OpType::MUL_MAT_TRANSB));
    CHECK(forge::op_supports_lowp(OpType::FLASH_ATTN_GQA));
    CHECK(forge::op_supports_lowp(OpType::ADD));
    CHECK(forge::op_supports_lowp(OpType::MUL));
    for (OpType op : audited_ops()) {
        CHECK(!forge::op_supports_lowp(op));
    }
    // Consistency invariant: once an op is whitelisted, the contract must accept
    // FP16/BF16 for it, otherwise the whitelist is a lie.
    for (OpType op : supported_lowp_ops()) {
        CHECK(forge::op_supports_dtype(op, DataType::FP16));
        CHECK(forge::op_supports_dtype(op, DataType::BF16));
    }
}

void test_fp8_contract() {
    // ---- 6-C-1: is_native_execution_dtype is now a WHITELIST. ----
    // Known native dtypes unchanged; FP8 joins FP16/BF16 in the lowp set; any
    // future dtype defaults to NOT native (compile-time: the switch lists every
    // member, so adding one breaks this function loudly until updated).
    CHECK(forge::is_native_execution_dtype(DataType::FP32));
    CHECK(forge::is_native_execution_dtype(DataType::INT8));
    CHECK(forge::is_native_execution_dtype(DataType::INT32));
    for (DataType dt : quantized_dtypes()) {
        CHECK(forge::is_native_execution_dtype(dt));
    }
    CHECK(!forge::is_native_execution_dtype(DataType::FP16));
    CHECK(!forge::is_native_execution_dtype(DataType::BF16));
    CHECK(!forge::is_native_execution_dtype(DataType::FP8_E4M3));

    // 6-C-4.1d/4.3 retired MUL_MAT_TRANSB (native FP8: CUTLASS GEMM for M>1, hand
    // written GEMV for M==1), so it is no longer in the "still rejected" set.
    const std::vector<OpType> all_ops = {
        OpType::FLASH_ATTN_GQA, OpType::ADD, OpType::MUL,
        OpType::MUL_MAT, OpType::ROPE, OpType::SOFT_MAX,
        OpType::EMBEDDING};
    // 6-C-2 retired RMS_NORM; 6-C-3 retires SILU + GELU + GELU_TANH (the gated
    // variants silu_multiply / gelu_multiply reuse the SILU / GELU gate, so no
    // separate enum); 6-C-4.1d/4.3 retires MUL_MAT_TRANSB. All must NOT be
    // rejected anymore.
    const std::vector<OpType> fp8_retired_ops = {OpType::RMS_NORM, OpType::SILU,
                                                 OpType::GELU, OpType::GELU_TANH,
                                                 OpType::MUL_MAT_TRANSB};

    // ---- FP8 retirement ledger: 6-C-1 opened it empty ("reject everything");
    // 6-C-2 retires RMS_NORM; 6-C-3 retires SILU / GELU / GELU_TANH (gated
    // variants reuse their gates). Every other op stays rejected, and the
    // retired set must not grow implicitly -- add to fp8_retired_ops explicitly
    // when a new native FP8 kernel lands.
    for (OpType op : fp8_retired_ops) {
        CHECK(forge::op_supports_fp8(op));
        CHECK(forge::op_supports_dtype(op, DataType::FP8_E4M3));
    }
    for (OpType op : all_ops) {
        CHECK(!forge::op_supports_fp8(op));
        CHECK(!forge::op_supports_dtype(op, DataType::FP8_E4M3));
        std::string msg;
        CHECK(rejected(op, DataType::FP8_E4M3, &msg));
        CHECK(msg.find("fp8_e4m3") != std::string::npos);
        CHECK(msg.find("has no native kernel") != std::string::npos);
        CHECK(msg.find(forge::op_type_name(op)) != std::string::npos);
    }

    // ---- require_no_fp8 routing gate: throws for FP8, no-op for every other
    // dtype (zero behavior change off the FP8 path).
    for (OpType op : all_ops) {
        std::string msg;
        try {
            forge::require_no_fp8("test_dtype_contract", op, DataType::FP8_E4M3);
            CHECK(false);
        } catch (const std::runtime_error& e) {
            msg = e.what();
            CHECK(msg.find("fp8_e4m3") != std::string::npos);
            CHECK(msg.find(forge::op_type_name(op)) != std::string::npos);
            CHECK(msg.find("Phase 6-C-1") != std::string::npos);
        }
    }

    // Non-FP8 dtypes pass the FP8 routing gate untouched.
    for (OpType op : all_ops) {
        try {
            forge::require_no_fp8("test_dtype_contract", op, DataType::FP32);
            forge::require_no_fp8("test_dtype_contract", op, DataType::FP16);
            forge::require_no_fp8("test_dtype_contract", op, DataType::BF16);
            for (DataType dt : quantized_dtypes()) {
                forge::require_no_fp8("test_dtype_contract", op, dt);
            }
        } catch (const std::runtime_error&) {
            CHECK(false);
        }
    }
}

}  // namespace

int main() {
    test_policy_defaults();
    test_execution_dtypes();
    test_explicit_reject_no_silent_fallback();
    test_lowp_accepted();
    test_extension_point();
    test_bf16_gate();
    test_bf16_retirement();
    test_fp8_contract();

    if (failures == 0) {
        std::printf("PASS dtype-contract (Phase 6-A-1 + 6-A-2 + 6-A-3 + 6-A-4 + 6-A-5 + 6-B-1 "
                    "+ 6-C-1)\n");
        return 0;
    }
    std::printf("FAILURES: %d\n", failures);
    return 1;
}
