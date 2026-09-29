/**
 * AttnRot: Hadamard activation rotation for a quantized KV cache.
 *
 * When K or V is stored quantized, every element carries a quantization error
 * proportional to the largest magnitude in its block. Attention weights are
 * what an outlier destroys: one large activation sets a large block scale and
 * crushes the precision of its 31 neighbours, and the K error lands inside a
 * softmax exponent where it cannot be absorbed. This is why 4-bit KV caches
 * degrade far worse than the nominal bit width suggests.
 *
 * The fix, from upstream llama.cpp PR #21038, is to rotate before quantizing.
 * With R orthonormal:
 *
 *     (Rq) . (Rk) = q . k          the logit is unchanged
 *     sum_i p_i (R v_i) = R (sum_i p_i v_i)
 *
 * so caching Rk and Rv, and rotating q the same way, leaves every logit
 * identical; the attention output comes back as R applied to the true output,
 * and R^-1 recovers it. A Hadamard matrix spreads any single large component
 * over all coordinates instead of leaving it to dominate one block, which is
 * exactly the property the quantizer needs. The Hadamard matrix is also
 * symmetric, so R^-1 == R: one transform both ways.
 *
 * Orthonormality requires the 1/sqrt(n) scaling, which is what makes this
 * exact rather than merely a rescaling of the whole attention.
 *
 * K and V are rotated by different sizes, following upstream: K is rotated
 * across the whole head dim (its rotation has to match Q's exactly to preserve
 * the dot product), while V is rotated in 64-wide sub-blocks, which measured
 * better and lets head dims that are not a power of two still work.
 *
 * R costs three transforms per layer (q, k on the way in, v's inverse on the
 * output) and is only worth it when something is actually quantized -- upstream
 * measured 2-3% on prefill and 7-12% on decode.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "forge/tensor.h"

namespace forge {

// The Sylvester Hadamard matrix of order n (n a power of two), normalised to
// 1/sqrt(n) so it is orthonormal. Row-major, n*n. Builds in O(n log n).
std::vector<float> hadamard_matrix(int n);

// Largest power-of-two divisor of `n` that is at least `min_order`, capped at n.
// The K rotation must divide the head dim exactly, or the dot product is not
// preserved. Returns 0 when no such divisor exists.
int rotation_order_for_head_dim(int head_dim, int min_order = 64);

class AttnRotExecutor {
public:
    // `enabled` turns the whole thing off; the caller decides based on whether
    // either KV type is quantized. When off, apply/undo are exact no-ops that
    // return the input unchanged.
    void configure(bool enabled, int head_dim, DeviceType device);

    bool enabled() const { return enabled_; }
    int order() const { return order_; }
    int v_block() const { return v_block_; }

    // Rotate q and k so their dot product is preserved. Both are
    // [rows, n_heads, head_dim] with the head dim last. No-op when disabled.
    void apply_qk(const TensorPtr& q, const TensorPtr& k, TensorPtr* q_out,
                  TensorPtr* k_out) const;

    // Rotate v before it enters the cache (in v_block-wide sub-blocks).
    TensorPtr apply_v(const TensorPtr& v) const;

    // Recover the true attention output from the rotated one. Same transform,
    // because the Hadamard matrix is its own inverse.
    TensorPtr undo_v(const TensorPtr& out) const;

    // Serialise which rotation a cache holds, so a cache written with rotation
    // on is never read with it off (or at a different order) -- that would
    // silently corrupt every token. Returns false when nothing is cached yet.
    int64_t signature() const;

private:
    // Apply the n-point transform to each `order`-wide sub-chunk of the last
    // dimension, in place, on the device.
    void transform_last_dim_(const TensorPtr& x, int order, bool inverse) const;

    bool enabled_ = false;
    int order_ = 0;    // K/Q rotation size == the head dim
    int v_block_ = 64;  // V rotation sub-block
    DeviceType device_ = DeviceType::CPU;
    mutable TensorPtr rot_k_;  // [order, order] on device
    mutable TensorPtr rot_v_;  // [v_block, v_block] on device
};

}  // namespace forge
