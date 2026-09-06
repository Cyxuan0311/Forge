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

// Ops still without a native low-precision kernel (6-A-0 audit). RMS_NORM (6-A-2),
// SILU (6-A-3), GELU / GELU_TANH (6-A-4), MUL_MAT_TRANSB (6-A-5) and
// FLASH_ATTN_GQA (6-A-7) left this set once they gained a real __half kernel
// (see supported_lowp_ops).
const std::vector<OpType>& audited_ops() {
    static const std::vector<OpType> ops = {
        OpType::ADD,
        OpType::MUL,       OpType::MUL_MAT, OpType::ROPE,
        OpType::SOFT_MAX, OpType::EMBEDDING};
    return ops;
}

// Ops that have gained a real native low-precision kernel (6-A-2+). The contract
// accepts FP16/BF16 for these; exact dtype routing happens downstream (CUDA).
const std::vector<OpType>& supported_lowp_ops() {
    static const std::vector<OpType> ops = {OpType::RMS_NORM, OpType::SILU, OpType::GELU,
                                            OpType::GELU_TANH, OpType::MUL_MAT_TRANSB,
                                            OpType::FLASH_ATTN_GQA};
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

void test_extension_point() {
    // The executable record of how much of Phase 6-A is real: RMS_NORM (6-A-2),
    // SILU (6-A-3), GELU / GELU_TANH (6-A-4) and MUL_MAT_TRANSB (6-A-5, native FP16
    // matmul) are whitelisted so far; everything else is still false.
    CHECK(forge::op_supports_lowp(OpType::RMS_NORM));
    CHECK(forge::op_supports_lowp(OpType::SILU));
    CHECK(forge::op_supports_lowp(OpType::GELU));
    CHECK(forge::op_supports_lowp(OpType::GELU_TANH));
    CHECK(forge::op_supports_lowp(OpType::MUL_MAT_TRANSB));
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

}  // namespace

int main() {
    test_policy_defaults();
    test_execution_dtypes();
    test_explicit_reject_no_silent_fallback();
    test_lowp_accepted();
    test_extension_point();

    if (failures == 0) {
        std::printf("PASS dtype-contract (Phase 6-A-1 + 6-A-2 + 6-A-3 + 6-A-4 + 6-A-5)\n");
        return 0;
    }
    std::printf("FAILURES: %d\n", failures);
    return 1;
}
