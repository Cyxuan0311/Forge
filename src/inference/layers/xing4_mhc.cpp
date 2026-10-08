#include "forge/inference/layers/xing4_mhc.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include "forge/inference/tensor_device_utils.h"
#include "forge/logger.h"
#include "forge/operator_matmul.h"
#include "forge/operator_norm.h"
#include "forge/perf_profiler.h"

#include "cuda/cuda_xing4.h"

namespace forge {

void MhcMixer::sinkhorn(float* c) const {
    auto norm_row = [&]() {  // each src row sums to 1
        for (int src = 0; src < kStreams; ++src) {
            float sum = eps_;
            for (int dst = 0; dst < kStreams; ++dst) sum += c[src * kStreams + dst];
            for (int dst = 0; dst < kStreams; ++dst) c[src * kStreams + dst] /= sum;
        }
    };
    auto norm_col = [&]() {  // each dst column sums to 1
        for (int dst = 0; dst < kStreams; ++dst) {
            float sum = eps_;
            for (int src = 0; src < kStreams; ++src) sum += c[src * kStreams + dst];
            for (int src = 0; src < kStreams; ++src) c[src * kStreams + dst] /= sum;
        }
    };
    for (int dst = 0; dst < kStreams; ++dst) {  // softmax over src
        float mx = -std::numeric_limits<float>::infinity();
        for (int src = 0; src < kStreams; ++src) mx = std::max(mx, c[src * kStreams + dst]);
        float sum = 0.0f;
        for (int src = 0; src < kStreams; ++src) {
            const float e = std::exp(c[src * kStreams + dst] - mx);
            c[src * kStreams + dst] = e;
            sum += e;
        }
        for (int src = 0; src < kStreams; ++src) c[src * kStreams + dst] /= sum;
    }
    norm_row();
    for (int i = 1; i < iters_; ++i) {
        norm_col();
        norm_row();
    }
}

TensorPtr MhcMixer::pre(const TensorPtr& x4, const TensorPtr& fn, const TensorPtr& scale,
                        const TensorPtr& base, TensorPtr& post, TensorPtr& comb,
                        DeviceType dev, int layer_idx) {
    PERF_SCOPE("layer/hc_pre");
    const int H = hidden_;
    const int nt = static_cast<int>(x4->shape()[0]);

    if (!fn || !scale || !base) {
        LOG_ERROR("MhcMixer: layer " + std::to_string(layer_idx) + " missing hc tensors");
        return nullptr;
    }

    // mixes = rms_norm(x4) @ fn^T -> [nt, 24]; 前 4 列 pre, 接着 4 列 post, 后 16 列 comb
    TensorPtr flat_norm = ops::rms_norm(x4, nullptr, eps_);
    TensorPtr mixes = ops::matmul_transB(flat_norm, fn);
    if (!mixes)
        return nullptr;

    post = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{nt, kStreams}, dev);
    comb = std::make_shared<Tensor>(DataType::FP32,
                                    std::vector<int64_t>{nt, kStreams * kStreams}, dev);
    auto x_in = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{nt, H}, dev);

    if (dev == DeviceType::CUDA) {
#ifdef USE_CUDA
        cuda::xing4_hc_pre_launch(mixes, x4, scale, base, x_in, post, comb, nt, H, eps_,
                                  iters_);
#endif
    } else {
        const int mix_dim = kStreams * (2 + kStreams);
        const float* m = static_cast<const float*>(ensure_cpu(mixes)->data());
        const float* sc = static_cast<const float*>(ensure_cpu(scale)->data());
        const float* b = static_cast<const float*>(ensure_cpu(base)->data());
        const float* x4d = static_cast<const float*>(ensure_cpu(x4)->data());
        float* pd = static_cast<float*>(post->data());
        float* cd = static_cast<float*>(comb->data());
        float* xd = static_cast<float*>(x_in->data());

        std::vector<float> c(kStreams * kStreams), pre(kStreams);
        for (int t = 0; t < nt; ++t) {
            const float* mt = m + static_cast<int64_t>(t) * mix_dim;
            for (int i = 0; i < kStreams; ++i)
                pre[i] = 1.0f / (1.0f + std::exp(-(sc[0] * mt[i] + b[i])));
            for (int i = 0; i < kStreams; ++i)
                pd[static_cast<int64_t>(t) * kStreams + i] =
                    (1.0f / (1.0f + std::exp(-(sc[1] * mt[kStreams + i] + b[kStreams + i])))) *
                    2.0f;
            for (int r = 0; r < kStreams * kStreams; ++r) {  // r = src + kStreams*dst
                float v = sc[2] * mt[2 * kStreams + r] + b[2 * kStreams + r];
                c[r] = std::clamp(v, -30.0f, 30.0f);
            }
            sinkhorn(c.data());
            for (int r = 0; r < kStreams * kStreams; ++r)
                cd[static_cast<int64_t>(t) * kStreams * kStreams + r] = c[r];
            // x_in[t,d] = sum_ih x4[t, ih*H+d] * pre[ih]
            float* xr = xd + static_cast<int64_t>(t) * H;
            const float* x4r = x4d + static_cast<int64_t>(t) * kStreams * H;
            for (int d = 0; d < H; ++d) {
                float acc = 0.0f;
                for (int ih = 0; ih < kStreams; ++ih)
                    acc += x4r[ih * H + d] * pre[ih];
                xr[d] = acc;
            }
        }
    }
    return x_in;
}

TensorPtr MhcMixer::post(const TensorPtr& x, const TensorPtr& res4, const TensorPtr& post,
                         const TensorPtr& comb, DeviceType dev) {
    PERF_SCOPE("layer/hc_post");
    const int H = hidden_;
    const int nt = static_cast<int>(x->shape()[0]);
    auto out4 = std::make_shared<Tensor>(
        DataType::FP32, std::vector<int64_t>{nt, static_cast<int64_t>(kStreams) * H}, dev);
    if (dev == DeviceType::CUDA) {
#ifdef USE_CUDA
        cuda::xing4_hc_post_launch(x, res4, post, comb, out4, nt, H);
#endif
    } else {
        const float* xd = static_cast<const float*>(x->data());
        const float* rd = static_cast<const float*>(res4->data());
        const float* pd = static_cast<const float*>(post->data());
        const float* cd = static_cast<const float*>(comb->data());
        float* od = static_cast<float*>(out4->data());
        for (int t = 0; t < nt; ++t) {
            for (int dst = 0; dst < kStreams; ++dst) {
                const float pw = pd[static_cast<int64_t>(t) * kStreams + dst];
                const float* xr = xd + static_cast<int64_t>(t) * H;
                float* orow = od + static_cast<int64_t>(t) * kStreams * H + dst * H;
                const float* rrow = rd + static_cast<int64_t>(t) * kStreams * H;
                const float* crow = cd + static_cast<int64_t>(t) * kStreams * kStreams;
                for (int d = 0; d < H; ++d) {
                    float acc = xr[d] * pw;
                    for (int src = 0; src < kStreams; ++src)
                        acc += rrow[src * H + d] * crow[src * kStreams + dst];
                    orow[d] = acc;
                }
            }
        }
    }
    return out4;
}

TensorPtr MhcMixer::mean(const TensorPtr& x4, DeviceType dev) {
    PERF_SCOPE("head/hc_mean");
    const int H = hidden_;
    const int nt = static_cast<int>(x4->shape()[0]);
    auto out = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{nt, H}, dev);
    if (dev == DeviceType::CUDA) {
#ifdef USE_CUDA
        cuda::xing4_hc_mean_launch(x4, out, nt, H);
#endif
    } else {
        const float* xd = static_cast<const float*>(x4->data());
        float* od = static_cast<float*>(out->data());
        for (int t = 0; t < nt; ++t) {
            const float* xr = xd + static_cast<int64_t>(t) * kStreams * H;
            float* orow = od + static_cast<int64_t>(t) * H;
            for (int d = 0; d < H; ++d) {
                float acc = 0.0f;
                for (int ih = 0; ih < kStreams; ++ih) acc += xr[ih * H + d];
                orow[d] = acc / static_cast<float>(kStreams);
            }
        }
    }
    return out;
}

TensorPtr MhcMixer::expand(const TensorPtr& x, DeviceType dev) {
    const int H = hidden_;
    const int nt = static_cast<int>(x->shape()[0]);
    if (x->shape()[1] == static_cast<int64_t>(kStreams) * H)
        return x;  // already a 4-way stream
    auto x4 = std::make_shared<Tensor>(
        DataType::FP32, std::vector<int64_t>{nt, static_cast<int64_t>(kStreams) * H}, dev);
    if (x->device() != dev) {
        // Caller guarantees placement; guard anyway.
        auto xc = ensure_cpu(x);
        const float* src = static_cast<const float*>(xc->data());
        float* dst = static_cast<float*>(x4->data());
        for (int t = 0; t < nt; ++t) {
            const float* row = src + static_cast<int64_t>(t) * H;
            float* orow = dst + static_cast<int64_t>(t) * kStreams * H;
            for (int ih = 0; ih < kStreams; ++ih)
                std::memcpy(orow + ih * H, row, H * sizeof(float));
        }
        return x4;
    }
    if (dev == DeviceType::CUDA) {
#ifdef USE_CUDA
        cuda::xing4_hc_expand_launch(x, x4, nt, H);
#endif
    } else {
        const float* src = static_cast<const float*>(x->data());
        float* dst = static_cast<float*>(x4->data());
        for (int t = 0; t < nt; ++t) {
            const float* row = src + static_cast<int64_t>(t) * H;
            float* orow = dst + static_cast<int64_t>(t) * kStreams * H;
            for (int ih = 0; ih < kStreams; ++ih)
                std::memcpy(orow + ih * H, row, H * sizeof(float));
        }
    }
    return x4;
}

}  // namespace forge
