#pragma once

// 融合 (fused) MoE GPU 路径的公共小工具。
//
// mellum_moe.cpp 与 gemma4_moe.cpp 各自持有一份逐字相同的副本: expert 轴布局
// 检查、dtype 的 kernel 可用性探测、以及 router 的三张 scratch 张量 + 一次
// launch_moe_router。这里收敛为一份。
//
// 注意: 这里只抽取完全相同的"脚手架"。各架构的激活函数 (silu vs gelu)、
// gate/up 的 buffer 布局 (交错单 buffer vs 两个独立 buffer)、down 的加权时机、
// 以及 fused gate_up 张量的行偏移计算都是真实的架构差异, 保留在各处不动。

#include "forge/cuda_kernels.h"
#include "forge/kernel_selector.h"
#include "forge/tensor.h"
#include "forge/types.h"

#ifdef USE_CUDA
#include <memory>
#include <vector>
#endif

namespace forge {

// slot kernel 的权重字节布局假设 expert 张量是 [n_expert, OUT, IN] (expert 轴
// 最外层, 即 Forge shape axis 0, 与 GGUF ne = [K, N, E] 一致)。expert 轴不在
// axis 0 时不能走融合路径 —— kernel 会按错误的 expert stride 取权重。
inline bool expert_axis_outermost(const TensorPtr& w3d, int n_expert) {
    if (!w3d)
        return false;
    const auto& shp = w3d->shape();
    return shp.size() == 3 && shp[0] == n_expert;
}

// 该 dtype 在给定 seq_len 下是否有可用的 MoE kernel。N/K 传 0 表示只按 domain
// 与 dtype 探测 (与各 engine 原有副本的行为一致)。
inline bool moe_kernel_supported(DataType dt, int seq_len) {
    KernelProblem p;
    p.domain = KernelDomain::MoE;
    p.dtype = dt;
    p.M = seq_len;
    p.N = 0;
    p.K = 0;
    return select_kernel(p) != KernelVariant::UNSUPPORTED;
}

#ifdef USE_CUDA
// 融合 MoE 的 router 产物: top-k 专家下标与其归一化权重, 外加 softmax 暂存。
struct MoeRouterScratch {
    TensorPtr indices;  // INT32 [seq_len * n_expert_used]
    TensorPtr weights;  // FP32  [seq_len * n_expert_used]
    TensorPtr softmax;  // FP32  [seq_len * n_expert]
};

// 在 CUDA 上分配三张 scratch 并完成一次路由 (无 CPU 往返)。
inline MoeRouterScratch run_moe_router_gpu(const float* router_logits, int n_expert,
                                           int n_expert_used, int seq_len) {
    MoeRouterScratch s;
    s.indices = std::make_shared<Tensor>(
        DataType::INT32, std::vector<int64_t>{seq_len * n_expert_used}, DeviceType::CUDA);
    s.weights = std::make_shared<Tensor>(
        DataType::FP32, std::vector<int64_t>{seq_len * n_expert_used}, DeviceType::CUDA);
    s.softmax = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{seq_len * n_expert},
                                         DeviceType::CUDA);
    cuda::launch_moe_router(router_logits, static_cast<int*>(s.indices->data()),
                            static_cast<float*>(s.weights->data()),
                            static_cast<float*>(s.softmax->data()), n_expert, n_expert_used,
                            seq_len);
    return s;
}
#endif

}  // namespace forge
