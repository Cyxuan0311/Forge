#pragma once

// =========================================================================
// Phase 6-A-1: ComputeDTypePolicy
//
// Before Phase 6 the DataType enum did double duty: it described both how a
// tensor is *stored* and what the kernels actually *compute* in, and the two
// had silently diverged (6-A-0 audit: `launch_rms_norm_fp16` / `launch_silu_fp16`
// only reinterpreted their buffers as float). The policy makes the split
// explicit and gives every layer one object to ask:
//
//     weight       -- dtype of the weight tensors feeding GEMM / GEMV / MMQ
//     activation   -- dtype of the tensors flowing between operators
//     accumulator  -- dtype GEMM and reduction kernels accumulate in
//     norm_compute -- dtype RMSNorm-family ops reduce/normalize in (6-A-2; the
//                     native FP16 RMSNorm reduces in FP32 regardless of
//                     `activation`, so this stays FP32 while `activation`=FP16)
//     kv           -- AUTHORITATIVE KV cache dtype, not merely its output dtype
//                     (today the FP32 store is authoritative and FP16/FP8 KV is
//                     derived from it -- see 6-A-0 #16; the policy is what
//                     eventually reverses that).
//
// `norm_compute` is modelled (6-A-2) for the reason given above. Scalar /
// residual_compute etc. remain deliberately unmodelled until a real operator
// needs them -- no speculative abstraction before then.
// =========================================================================

#include "forge/types.h"

namespace forge {

struct ComputeDTypePolicy {
    DataType weight = DataType::FP32;
    DataType activation = DataType::FP32;
    DataType accumulator = DataType::FP32;
    DataType norm_compute = DataType::FP32;  // 6-A-2: RMSNorm reduction dtype
    DataType kv = DataType::FP32;
};

// Today's only fully supported configuration; also the safe default for any
// graph that has not opted into a native low-precision path.
inline ComputeDTypePolicy fp32_policy() {
    return ComputeDTypePolicy{};
}

inline bool is_fp32_policy(const ComputeDTypePolicy& p) {
    return p.weight == DataType::FP32 && p.activation == DataType::FP32 &&
           p.accumulator == DataType::FP32 && p.norm_compute == DataType::FP32 &&
           p.kv == DataType::FP32;
}

}  // namespace forge
