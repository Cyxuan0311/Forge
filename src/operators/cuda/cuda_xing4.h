#pragma once

#include "forge/tensor.h"

// CUDA kernels for the Xing4.0 mHC (Manifold-Constrained Hyper-Connections)
// residual stream, mirroring the CPU reference in xing4_engine.cpp. The 4-way
// stream is kept row-major [nt, 4*hidden] with channel ih occupying
// [ih*hidden, (ih+1)*hidden) of each row.
namespace forge {
namespace cuda {

// hc_pre tail: from the per-token mixes [nt, 24] (already computed as
// rms_norm(x4) @ hc_fn via ops), compute
//   pre[t,i]  = sigmoid(scale[0]*mixes[t,i] + base[i])                (i < 4)
//   post[t,i] = sigmoid(scale[1]*mixes[t,4+i] + base[4+i]) * 2.0
//   comb[t,src*4+dst] = clamp(scale[2]*mixes[t,8+src+4*dst]
//                             + base[8+src+4*dst], -30, 30)  -> sinkhorn
// and fold the stream mixture x_in[t,d] = sum_i x4[t,ih*H+d] * pre[t,ih].
// One block per token.
void xing4_hc_pre_launch(const TensorPtr& mixes, const TensorPtr& x4,
                         const TensorPtr& scale, const TensorPtr& base,
                         TensorPtr& x_in, TensorPtr& post, TensorPtr& comb,
                         int nt, int hidden, float eps, int iters);

// hc_post: out4[t, dst*H+d] = x[t,d] * post[t,dst]
//          + sum_src res4[t, src*H+d] * comb[t, src*4+dst]
void xing4_hc_post_launch(const TensorPtr& x, const TensorPtr& res4,
                          const TensorPtr& post, const TensorPtr& comb,
                          TensorPtr& out4, int nt, int hidden);

// Average the 4 residual streams: out[t,d] = mean_ih x4[t, ih*H+d].
void xing4_hc_mean_launch(const TensorPtr& x4, TensorPtr& out, int nt, int hidden);

// Expand a single-stream [nt, H] tensor into the 4-way stream [nt, 4*H]
// (channel ih = an identical copy of the input row). Used for the embedding.
void xing4_hc_expand_launch(const TensorPtr& x, TensorPtr& x4, int nt, int hidden);

}  // namespace cuda
}  // namespace forge
