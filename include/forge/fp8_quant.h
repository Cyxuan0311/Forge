// =========================================================================
// fp8_quant.h — 6-C-5.1a: W8A8 FP8 E4M3 per-tensor quantize PRODUCER
//
// Deliberately a SEPARATE path from the KV-cache FP8 quantize in cuda_quant.cu:
//
//   KV FP8 quantize      -> per-(row, kv_head) scale, stored in its own device
//                           `scales` buffer. Untouched by this file.
//   W8A8 FP8 quantize    -> ONE per-tensor scale, stored in the output Tensor's
//                           FP8 scale sidecar (data layer). This file.
//
// The two scales do NOT share a data structure.
//
// Everything runs on device (amax reduction -> scale -> encode) so there is no
//   device amax -> D2H -> host /240 -> H2D scale
// round trip. That is what keeps the producer capturable by CUDA Graphs (6-C-5.3)
// and it is why the output Tensor's device scalar is written by a kernel via
// Tensor::fp8_scale_device_slot().
// =========================================================================

#pragma once

#include "forge/tensor.h"  // TensorPtr, cudaStream_t

namespace forge {

// 6-C-5.1b: amax -> scale contract, host-side reference (mirrors the device kernel).
enum class Fp8AmaxStatus {
    Ok,         // amax > 0 && finite  -> scale = amax / 240
    Zero,       // amax == 0           -> scale = 1, all-zero FP8 output
    NonFinite,  // amax is NaN/Inf     -> REJECT (must not poison the graph)
};

Fp8AmaxStatus fp8_amax_status(float amax);

// Applies the 6-C-5.1b contract. Returns NaN for NonFinite (device path) so the
// poisoning is visible rather than silent.
float fp8_w8a8_scale_from_amax(float amax);

// E4M3 max normal (6-C-1 contract: scale = amax / 240).
constexpr float kFp8E4M3Max = 240.0f;

// ---------------------------------------------------------------------------
// 6-C-5.5: FP8 OUTPUT quantize producer
//
//   FP32 D -> device amax (+non-finite flag) -> scale = amax/240 -> FP8 encode
//
// Shares the 5.1a/5.1b reduction verbatim (finite abs-max + explicit non-finite flag).
// It deliberately does NOT re-implement `amax = fmaxf(amax, fabsf(x))`, which silently
// swallows NaN (fmaxf(NaN, x) == x), and it does NOT depend on a host scale: the
// finalize kernel writes *scale_slot = scale directly into the Tensor's device scalar.
// ---------------------------------------------------------------------------

// 6-C-5.5a: convenience form -- allocates the FP8 output Tensor.
TensorPtr quantize_fp8_e4m3_output(const TensorPtr& src, cudaStream_t stream = 0);

// 6-C-5.5d: dst-injection form -- writes into a PRE-ALLOCATED FP8 Tensor, reusing its
// stable device scale slot. This is the capture-safe form: no allocation happens inside
// the captured region, so the whole amax -> scale -> encode producer can live in a CUDA
// graph and the scale slot is updated BY THE GRAPH on every replay.
//
// 6-C-5.6a: `apply_relu` turns the same producer into the FP8 ACTIVATION. ReLU is fused
// into the reduction AND the encode pass, so the activation still uses exactly one
// (5.1b/5.5b-verified) scale contract and materializes NO intermediate FP32 tensor.
// Deliberately NOT a separate FP8 scale implementation.
void quantize_fp8_e4m3_output_into(const TensorPtr& src, const TensorPtr& dst,
                                   cudaStream_t stream = 0, bool apply_relu = false);

// 6-C-5.5a: quantize an FP32 tensor to FP8_E4M3 with a per-tensor W8A8 scale.
//
// Returns a new FP8_E4M3 Tensor with the same shape/device as `src`, whose FP8 scale
// sidecar is a DEVICE-COMPUTED, Tensor-owned, graph-visible scalar:
//   1. create FP8 output Tensor
//   2. allocate/output its stable device scale slot
//   3. device amax reduction
//   4. scale = amax / 240   (amax == 0 -> 1; NaN/Inf -> NaN, see fp8_amax_status)
//   5. q = encode(x / scale), reading the scale from device memory
//
// No host<->device round trip: safe to call inside CUDA Graph capture.
TensorPtr quantize_fp8_e4m3_w8a8(const TensorPtr& src, cudaStream_t stream = 0);

}  // namespace forge
