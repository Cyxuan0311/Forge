#pragma once

// YaRN (Yet another RoPE extensioN) 频率表的公共构建 / 上传。
//
// xing4_engine.cpp 与 mellum_engine.cpp 原本各持有一份 build_yarn_table(),
// 两者的 corr_dim / low / high / denom / ramp 计算逐行相同 (mellum 侧的注释
// 直接写着"照抄 Xing4_0Engine::build_yarn_table")。这里收敛为一份。
//
// 注意: dim 的取值策略保持由调用方决定 —— xing4 在 rope_dimension_count <= 0
// 时视为不启用, mellum 则回退到 head_dim。两侧原有行为都不变。

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "forge/model.h"
#include "forge/tensor.h"
#include "forge/types.h"

namespace forge {

// 逐维 YaRN 频率 scale s_i (theta_i = pos * inv_freq_i * s_i) 与旋转幅值
// mscale, 镜像 ggml rope_yarn() + ggml_rope_yarn_corr_dims()。
//
// dim: 参与旋转的维度数, 由调用方按自身策略从 cfg.rope_dimension_count /
//      cfg.head_dim 推导。
// 返回 false 表示 YaRN 未启用或 dim 非法; 此时 out_s 被清空、mscale 置 1。
inline bool build_yarn_freq_table(const ModelConfig& cfg, int dim, std::vector<float>& out_s,
                                  float& out_mscale) {
    out_s.clear();
    out_mscale = 1.0f;
    if (!cfg.rope_yarn_enabled || dim <= 0)
        return false;

    auto corr_dim = [&](float n_rot) {
        return static_cast<float>(dim) *
               std::log(static_cast<float>(cfg.rope_yarn_orig_ctx) /
                        (n_rot * 2.0f * 3.14159265358979323846f)) /
               (2.0f * std::log(cfg.rope_theta));
    };
    const float low = std::max(0.0f, std::floor(corr_dim(cfg.rope_yarn_beta_fast)));
    const float high =
        std::min(static_cast<float>(dim - 1), std::ceil(corr_dim(cfg.rope_yarn_beta_slow)));
    const float denom = std::max(0.001f, high - low);

    out_mscale = (cfg.rope_yarn_ext_factor != 0.0f)
                     ? cfg.rope_yarn_attn_factor *
                           (1.0f + 0.1f * std::log(1.0f / cfg.rope_yarn_freq_scale))
                     : cfg.rope_yarn_attn_factor;

    out_s.assign(static_cast<size_t>(dim), 1.0f);
    for (int i0 = 0; i0 + 1 < dim; i0 += 2) {
        const float yr = (static_cast<float>(i0 / 2) - low) / denom;
        const float ramp = 1.0f - std::min(1.0f, std::max(0.0f, yr));
        const float ramp_mix = ramp * cfg.rope_yarn_ext_factor;
        const float s = cfg.rope_yarn_freq_scale * (1.0f - ramp_mix) + ramp_mix;
        out_s[static_cast<size_t>(i0)] = s;
        out_s[static_cast<size_t>(i0) + 1] = s;
    }
    return true;
}

// half-split 配对表: 第 d 对 (d, d + half_dim) 对应 ggml 的 dim 对 (2d, 2d+1)。
inline std::vector<float> yarn_pair_table(const std::vector<float>& yarn_s) {
    const size_t half = yarn_s.size() / 2;
    std::vector<float> out(half, 1.0f);
    for (size_t d = 0; d < half; ++d)
        out[d] = yarn_s[2 * d];
    return out;
}

// 把 host 表一次性上传到 CUDA 并缓存到 dev。返回 device 指针; 表为空或 CUDA
// 不可用时返回 nullptr。
inline const float* yarn_table_dev_ptr(const std::vector<float>& host, TensorPtr& dev) {
    if (host.empty())
        return nullptr;
#ifdef USE_CUDA
    if (!dev) {
        const int n = static_cast<int>(host.size());
        auto h = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{n},
                                          DeviceType::CPU);
        std::memcpy(h->data(), host.data(), sizeof(float) * static_cast<size_t>(n));
        dev = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{n}, DeviceType::CUDA);
        dev->copy_from(*h);
    }
    return static_cast<const float*>(dev->data());
#else
    (void)dev;
    return nullptr;
#endif
}

}  // namespace forge
