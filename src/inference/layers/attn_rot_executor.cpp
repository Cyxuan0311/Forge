#include "forge/inference/layers/attn_rot_executor.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "forge/cuda_kernels.h"
#include "forge/operators.h"

namespace forge {

std::vector<float> hadamard_matrix(int n) {
    std::vector<float> h(static_cast<size_t>(n) * n, 0.0f);
    if (n <= 0)
        return h;
    h[0] = 1.0f;
    int m = 1;
    while (m < n) {
        // Sylvester construction: H_2m = [[H_m, H_m], [H_m, -H_m]].
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < m; ++j) {
                const float v = h[static_cast<size_t>(i) * n + j];
                h[static_cast<size_t>(i + m) * n + j] = v;
                h[static_cast<size_t>(i) * n + j + m] = v;
                h[static_cast<size_t>(i + m) * n + j + m] = -v;
            }
        }
        m <<= 1;
    }
    // Normalise to 1/sqrt(n) so the matrix is orthonormal. H*H = n*I unnormalised,
    // so the scale has to be 1/sqrt(n) on both sides.
    const float inv = 1.0f / std::sqrt(static_cast<float>(n));
    for (auto& v : h)
        v *= inv;
    return h;
}

int rotation_order_for_head_dim(int head_dim, int min_order) {
    if (head_dim <= 0 || min_order <= 0)
        return 0;
    // The rotation must divide the head dim exactly, otherwise the dot product is
    // not preserved. So the order is a power of two dividing head_dim, at least
    // min_order, and the LARGEST such value: a smaller rotation leaves more of
    // the outlier structure intact, which is the thing being removed. Search
    // upward, starting from the first power of two >= min_order.
    int lo = 1;
    while (lo < min_order)
        lo <<= 1;
    int best = 0;
    for (int o = lo; o <= head_dim; o <<= 1) {
        if (head_dim % o == 0)
            best = o;  // keep going: we want the largest
    }
    return best;
}

void AttnRotExecutor::configure(bool enabled, int head_dim, DeviceType device) {
    device_ = device;
    rot_k_.reset();
    rot_v_.reset();
    enabled_ = false;
    order_ = 0;
    v_block_ = 64;

    if (!enabled || head_dim <= 0)
        return;

    // V's rotation is a 64-wide sub-block, so it only needs the head dim to be a
    // multiple of 64 -- the same condition llama.cpp requires. Head dims that are
    // not a power of two (e.g. 80, 96 for some models) then still work for V.
    if (head_dim % 64 != 0) {
        enabled_ = false;
        return;
    }
    v_block_ = 64;

    // K and Q must share one rotation across the whole head dim.
    order_ = rotation_order_for_head_dim(head_dim, 64);
    if (order_ == 0) {
        enabled_ = false;
        return;
    }
    if (order_ > 512) {
        enabled_ = false;
        return;
    }
    enabled_ = true;
}

int64_t AttnRotExecutor::signature() const {
    if (!enabled_)
        return 0;
    return (static_cast<int64_t>(order_) << 16) | static_cast<int64_t>(v_block_);
}

// Butterfly, one vector of `order` floats. Matches the CUDA kernel exactly.
static void hadamard_rotate_inplace(float* p, int order) {
    for (int stride = 1; stride < order; stride <<= 1) {
        for (int i = 0; i < order; ++i) {
            if (i < (i ^ stride)) {
                const int partner = i ^ stride;
                const float a = p[i];
                const float b = p[partner];
                p[i] = a + b;
                p[partner] = a - b;
            }
        }
    }
    const float inv = 1.0f / std::sqrt(static_cast<float>(order));
    for (int i = 0; i < order; ++i)
        p[i] *= inv;
}

void AttnRotExecutor::transform_last_dim_(const TensorPtr& x, int order, bool /*inverse*/) const {
    if (!x)
        return;
    const int n = static_cast<int>(x->shape().back());
    if (order <= 0 || n % order != 0)
        return;
    float* p = static_cast<float*>(x->data());
    const int64_t rows = x->numel() / n;
    const int chunks = n / order;
    for (int64_t r = 0; r < rows; ++r) {
        float* row = p + r * n;
        for (int c = 0; c < chunks; ++c)
            hadamard_rotate_inplace(row + c * order, order);
    }
}

void AttnRotExecutor::apply_qk(const TensorPtr& q, const TensorPtr& k, TensorPtr* q_out,
                               TensorPtr* k_out) const {
    if (!enabled_)
        return;
    // Rotate in place. Both are [rows, heads, head_dim] with head_dim last, and
    // the same rotation is applied to each, which is what preserves q.k.
    if (q_out && *q_out) {
        transform_last_dim_(*q_out, order_, false);
    } else if (q) {
        transform_last_dim_(q, order_, false);
    }
    if (k_out && *k_out) {
        transform_last_dim_(*k_out, order_, false);
    } else if (k) {
        transform_last_dim_(k, order_, false);
    }
}

TensorPtr AttnRotExecutor::apply_v(const TensorPtr& v) const {
    if (!enabled_ || !v)
        return v;
    transform_last_dim_(v, v_block_, false);
    return v;
}

TensorPtr AttnRotExecutor::undo_v(const TensorPtr& out) const {
    if (!enabled_ || !out)
        return out;
    // Self-inverse: the Hadamard matrix is symmetric, so the same transform
    // recovers the true output.
    transform_last_dim_(out, v_block_, true);
    return out;
}

namespace ops {

bool hadamard_rotate(const TensorPtr& x, int order) {
    if (!x || order <= 0)
        return false;
    const int n = static_cast<int>(x->shape().back());
    if (n % order != 0)
        return false;
#ifdef USE_CUDA
    if (x->device() == DeviceType::CUDA) {
        const int64_t rows = x->numel() / n;
        return cuda::launch_hadamard_rotate(static_cast<const float*>(x->data()),
                                           static_cast<float*>(x->data()),
                                           static_cast<int>(rows), order);
    }
#endif
    float* p = static_cast<float*>(x->data());
    const int64_t rows = x->numel() / n;
    const int chunks = n / order;
    for (int64_t r = 0; r < rows; ++r) {
        float* row = p + r * n;
        for (int c = 0; c < chunks; ++c)
            hadamard_rotate_inplace(row + c * order, order);
    }
    return true;
}

}  // namespace ops
}  // namespace forge
