#pragma once

// per-head RMSNorm 与 per-head norm 权重读取的公共实现。
//
// 此前 gemma4_attention.cpp / gemma_embedding2_engine.cpp / mellum_engine.cpp /
// lfm2_engine.cpp / qwen35_full_attention.cpp 各自持有一份副本, 数学完全一致,
// 仅权重形参是 "const float* 可为空" 还是 "std::vector<float>" 的差别。这里
// 收敛为一份: 权重以 const float* 传入, nullptr 表示退化为不带 learnable
// weight 的普通 RMSNorm (Gemma4 的 V-norm 走的就是这条)。

#include <cmath>
#include <cstring>
#include <vector>

#include "cpu/common/thread_pool.h"
#include "forge/inference/tensor_device_utils.h"

namespace forge {

// 读取 per-head norm 权重到 host vector。空张量或非 FP32 权重退化为全 1,
// 与各 engine 原有副本的行为一致。
inline std::vector<float> load_head_norm_weight(const TensorPtr& w, int head_dim) {
    std::vector<float> out(static_cast<size_t>(head_dim), 1.0f);
    if (!w || w->dtype() != DataType::FP32)
        return out;
    auto cpu = ensure_cpu(w);
    std::memcpy(out.data(), cpu->data(), static_cast<size_t>(head_dim) * sizeof(float));
    return out;
}

// 对 [seq_len, num_heads, head_dim] 缓冲的最后一维做就地 RMSNorm。
// weight == nullptr 时使用不带 learnable weight 的形式。
inline void per_head_rms_norm_cpu(float* data, const float* weight, int seq_len, int num_heads,
                                  int head_dim, float eps) {
    cpu::parallel_for2(seq_len, num_heads, [&](int s, int h) {
        float* head = data + (static_cast<size_t>(s) * num_heads + h) * head_dim;
        float ss = 0.0f;
        for (int d = 0; d < head_dim; ++d) ss += head[d] * head[d];
        const float inv_rms = 1.0f / std::sqrt(ss / head_dim + eps);
        if (weight) {
            for (int d = 0; d < head_dim; ++d) head[d] = head[d] * inv_rms * weight[d];
        } else {
            for (int d = 0; d < head_dim; ++d) head[d] *= inv_rms;
        }
    });
}

// 权重以 vector 形式持有的调用方重载。
inline void per_head_rms_norm_cpu(float* data, const std::vector<float>& weight, int seq_len,
                                  int num_heads, int head_dim, float eps) {
    per_head_rms_norm_cpu(data, weight.empty() ? nullptr : weight.data(), seq_len, num_heads,
                          head_dim, eps);
}

}  // namespace forge
