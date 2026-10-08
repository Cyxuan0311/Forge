#pragma once

// MhcMixer: Xing4.0 的 Manifold-Constrained Hyper-Connections (mHC) 残差流。
//
// 从 Xing4_0Engine 抽出。模型在层间维护 count(=4) 条并行残差流
// x4 [nt, count*hidden]，每层用 hc_{attn,ffn}_{fn,scale,base} 做混合：
//
//   expand(x)            : embedding [nt,H] -> 复制成 [nt, count*H]
//   pre(x4, fn, ...)     : 混合出本层输入 x_in [nt,H]，并产出
//                          post [nt,count] 与 Sinkhorn 归一化后的
//                          comb [nt,count*count]（供 post 使用）
//   post(x, x4, ...)     : out4[dst] = x*post[dst] + sum_src x4[src]*comb[src,dst]
//   mean(x4)             : 收尾，count 条流取平均 -> [nt,H]
//
// 每个 token 独立，CPU/CUDA 双路径；CUDA 侧为融合 kernel（cuda_xing4.cu）。
// 语义对齐 llama.cpp PR #29012 xing4_0.cpp 的 build_hc_{pre,post,sinkhorn}。

#include "forge/tensor.h"

namespace forge {

class MhcMixer {
public:
    // `eps` 同时用于 RMSNorm 与 Sinkhorn 归一化分母。
    MhcMixer(int sinkhorn_iters, float eps, int hidden_dim)
        : iters_(sinkhorn_iters), eps_(eps), hidden_(hidden_dim) {}

    // 4 条残差流（llama.cpp repeat_4d；parser 已保证 hyper_connection.count == 4）
    static constexpr int kStreams = 4;

    TensorPtr expand(const TensorPtr& x, DeviceType dev);

    // 返回 x_in [nt, hidden]；post/comb 为出参，交给同层的 post() 使用。
    // 缺张量时记日志并返回 nullptr。
    TensorPtr pre(const TensorPtr& x4, const TensorPtr& fn, const TensorPtr& scale,
                  const TensorPtr& base, TensorPtr& post, TensorPtr& comb,
                  DeviceType dev, int layer_idx);

    TensorPtr post(const TensorPtr& x, const TensorPtr& res4, const TensorPtr& post,
                   const TensorPtr& comb, DeviceType dev);

    TensorPtr mean(const TensorPtr& x4, DeviceType dev);

    int hidden() const { return hidden_; }

private:
    // Sinkhorn on c[src*kStreams + dst]. Reference order: softmax over src (no
    // eps) -> norm rows -> (iters-1) x {norm cols, norm rows}; eps is only added
    // to the normalization denominators.
    void sinkhorn(float* c) const;

    int iters_ = 20;
    float eps_ = 1e-6f;
    int hidden_ = 0;
};

}  // namespace forge
