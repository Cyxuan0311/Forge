// CUDA implementation of the Xing4.0 mHC operators, mirroring the CPU loops in
// xing4_engine.cpp (which mirror llama.cpp PR #29012 xing4_0.cpp). The hc_pre
// head (rms_norm over the 4-way stream + fn projection) stays in ops::rms_norm /
// ops::matmul_transB; these kernels cover the per-token tail: gate affine +
// sigmoid + clamp, the 4x4 Sinkhorn normalization, the stream mixture, hc_post
// and the final stream mean.

#include "cuda_xing4.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>

namespace forge {
namespace cuda {

// ---------------------------------------------------------------------------
// hc_pre tail: one block (256 threads) per token.
// ---------------------------------------------------------------------------
__global__ void k_xing4_hc_pre(const float* __restrict__ mixes,
                               const float* __restrict__ x4,
                               const float* __restrict__ scale,
                               const float* __restrict__ base,
                               float* __restrict__ x_in,
                               float* __restrict__ post_out,
                               float* __restrict__ comb_out,
                               int hidden, float eps, int iters) {
    const int t = blockIdx.x;
    __shared__ float pre_sh[4];
    __shared__ float comb_sh[16];

    const float* mrow = mixes + static_cast<int64_t>(t) * 24;

    // threads 0..23: affine + gates. comb goes to shared for the Sinkhorn.
    if (threadIdx.x < 4) {
        pre_sh[threadIdx.x] =
            1.0f / (1.0f + __expf(-(scale[0] * mrow[threadIdx.x] + base[threadIdx.x])));
    } else if (threadIdx.x < 8) {
        const int i = threadIdx.x - 4;
        post_out[static_cast<int64_t>(t) * 4 + i] =
            (1.0f / (1.0f + __expf(-(scale[1] * mrow[threadIdx.x] + base[threadIdx.x])))) * 2.0f;
    } else if (threadIdx.x < 24) {
        const int r = threadIdx.x - 8;  // r = src + 4*dst
        float v = scale[2] * mrow[threadIdx.x] + base[threadIdx.x];
        comb_sh[r] = fminf(fmaxf(v, -30.0f), 30.0f);
    }
    __syncthreads();

    // Sinkhorn on the 4x4 comb (single thread; ~500 flops). comb[src*4+dst].
    // Reference order: softmax over src (no eps) -> norm_dst -> (iters-1) x
    // {norm_src, norm_dst}; eps is only added to the normalization denominators.
    if (threadIdx.x == 0) {
        float c[16];
        for (int i = 0; i < 16; ++i) c[i] = comb_sh[i];

        // softmax over src: per dst column
        for (int dst = 0; dst < 4; ++dst) {
            float mx = -FLT_MAX;
            for (int src = 0; src < 4; ++src) mx = fmaxf(mx, c[src * 4 + dst]);
            float sum = 0.0f;
            for (int src = 0; src < 4; ++src) {
                float e = __expf(c[src * 4 + dst] - mx);
                c[src * 4 + dst] = e;
                sum += e;
            }
            for (int src = 0; src < 4; ++src) c[src * 4 + dst] /= sum;
        }
        // norm_dst: per src row sums to 1
        auto norm_dst = [&]() {
            for (int src = 0; src < 4; ++src) {
                float sum = eps;
                for (int dst = 0; dst < 4; ++dst) sum += c[src * 4 + dst];
                for (int dst = 0; dst < 4; ++dst) c[src * 4 + dst] /= sum;
            }
        };
        // norm_src: per dst column sums to 1
        auto norm_src = [&]() {
            for (int dst = 0; dst < 4; ++dst) {
                float sum = eps;
                for (int src = 0; src < 4; ++src) sum += c[src * 4 + dst];
                for (int src = 0; src < 4; ++src) c[src * 4 + dst] /= sum;
            }
        };
        norm_dst();
        for (int i = 1; i < iters; ++i) {
            norm_src();
            norm_dst();
        }
        for (int i = 0; i < 16; ++i) comb_out[static_cast<int64_t>(t) * 16 + i] = c[i];
    }
    __syncthreads();

    // Stream mixture: x_in[t,d] = sum_ih x4[t, ih*H+d] * pre[t,ih].
    for (int d = threadIdx.x; d < hidden; d += blockDim.x) {
        float acc = 0.0f;
        for (int ih = 0; ih < 4; ++ih)
            acc += x4[static_cast<int64_t>(t) * 4 * hidden + ih * hidden + d] * pre_sh[ih];
        x_in[static_cast<int64_t>(t) * hidden + d] = acc;
    }
}

void xing4_hc_pre_launch(const TensorPtr& mixes, const TensorPtr& x4,
                         const TensorPtr& scale, const TensorPtr& base,
                         TensorPtr& x_in, TensorPtr& post, TensorPtr& comb,
                         int nt, int hidden, float eps, int iters) {
    const int block = 256;
    k_xing4_hc_pre<<<nt, block>>>(
        static_cast<const float*>(mixes->data()), static_cast<const float*>(x4->data()),
        static_cast<const float*>(scale->data()), static_cast<const float*>(base->data()),
        static_cast<float*>(x_in->data()), static_cast<float*>(post->data()),
        static_cast<float*>(comb->data()), hidden, eps, iters);
}

// ---------------------------------------------------------------------------
// hc_post: elementwise over [nt, 4*hidden].
// ---------------------------------------------------------------------------
__global__ void k_xing4_hc_post(const float* __restrict__ x,
                                const float* __restrict__ res4,
                                const float* __restrict__ post,
                                const float* __restrict__ comb,
                                float* __restrict__ out4,
                                int64_t total, int hidden) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int64_t t = idx / (4 * hidden);
    const int p = static_cast<int>(idx - t * 4 * hidden);  // dst*hidden + d
    const int dst = p / hidden;
    const int d = p - dst * hidden;

    float acc = x[static_cast<int64_t>(t) * hidden + d] * post[t * 4 + dst];
    for (int src = 0; src < 4; ++src)
        acc += res4[t * 4 * hidden + src * hidden + d] * comb[t * 16 + src * 4 + dst];
    out4[idx] = acc;
}

void xing4_hc_post_launch(const TensorPtr& x, const TensorPtr& res4,
                          const TensorPtr& post, const TensorPtr& comb,
                          TensorPtr& out4, int nt, int hidden) {
    const int64_t total = static_cast<int64_t>(nt) * 4 * hidden;
    const int block = 256;
    const int grid = static_cast<int>((total + block - 1) / block);
    k_xing4_hc_post<<<grid, block>>>(
        static_cast<const float*>(x->data()), static_cast<const float*>(res4->data()),
        static_cast<const float*>(post->data()), static_cast<const float*>(comb->data()),
        static_cast<float*>(out4->data()), total, hidden);
}

// ---------------------------------------------------------------------------
// hc_mean: average the 4 residual streams.
// ---------------------------------------------------------------------------
__global__ void k_xing4_hc_mean(const float* __restrict__ x4,
                                float* __restrict__ out,
                                int64_t total, int hidden) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int64_t t = idx / hidden;
    const int d = static_cast<int>(idx - t * hidden);
    float acc = 0.0f;
    for (int ih = 0; ih < 4; ++ih)
        acc += x4[t * 4 * hidden + ih * hidden + d];
    out[idx] = acc * 0.25f;
}

void xing4_hc_mean_launch(const TensorPtr& x4, TensorPtr& out, int nt, int hidden) {
    const int64_t total = static_cast<int64_t>(nt) * hidden;
    const int block = 256;
    const int grid = static_cast<int>((total + block - 1) / block);
    k_xing4_hc_mean<<<grid, block>>>(
        static_cast<const float*>(x4->data()), static_cast<float*>(out->data()),
        total, hidden);
}

// ---------------------------------------------------------------------------
// hc_expand: single stream -> 4 identical channels (embedding init).
// ---------------------------------------------------------------------------
__global__ void k_xing4_hc_expand(const float* __restrict__ x,
                                  float* __restrict__ x4,
                                  int64_t total, int hidden) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total4 = total * 4;  // x4 covers 4 streams
    if (idx >= total4) return;
    // idx = t*4*hidden + ch*hidden + d  =>  x4[idx] = x[t*hidden + d]
    x4[idx] = x[(idx / (4 * hidden)) * hidden + (idx % hidden)];
}

void xing4_hc_expand_launch(const TensorPtr& x, TensorPtr& x4, int nt, int hidden) {
    const int64_t total = static_cast<int64_t>(nt) * hidden;
    const int64_t total4 = total * 4;  // launch over the full 4-way stream
    const int block = 256;
    const int grid = static_cast<int>((total4 + block - 1) / block);
    k_xing4_hc_expand<<<grid, block>>>(
        static_cast<const float*>(x->data()), static_cast<float*>(x4->data()),
        total, hidden);
}

}  // namespace cuda
}  // namespace forge
