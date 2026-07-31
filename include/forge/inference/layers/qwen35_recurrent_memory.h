#pragma once

// Qwen35RecurrentMemory: Qwen3.5 linear attention (Gated Delta Net) 的循环状态。
//
// 重构前 conv state / ssm state 及其 CPU/GPU 双份缓冲直接定义在 Qwen35Engine
// header 里, SSM 维度也是 engine 的七个成员变量, 任何想单独测试 delta net 的代码
// 都必须构造整个 engine。这里把状态所有权和维度推导收敛为一个组件。
//
// 阶段 8 已把所有权收归 InferenceContext 的 HybridMemory, engine 只持有引用。
// 状态是逐 token 累积的, 与 KV cache 一样属于"推理期可变状态"。

#include <vector>

#include "forge/model.h"
#include "forge/model_weights.h"

namespace forge {

// SSM 维度。全部从 ModelConfig 读取, 缺失时按权重形状推导, 再退回默认值。
struct Qwen35SsmDims {
    int d_inner = 0;
    int d_state = 0;
    int n_group = 0;
    int dt_rank = 0;
    int d_conv = 0;
    int head_v_dim = 0;
    int conv_channels = 0;

    int head_k_dim() const { return d_state; }
    int num_k_heads() const { return n_group; }
    int num_v_heads() const { return dt_rank; }
    int key_dim() const { return d_state * n_group; }
    int value_dim() const { return head_v_dim * dt_rank; }
    int state_size() const { return head_v_dim * head_v_dim * dt_rank; }
    int conv_state_size() const { return (d_conv - 1) * conv_channels; }
};

class Qwen35RecurrentMemory {
public:
    ~Qwen35RecurrentMemory();

    // 推导维度并为每个 LinearAttention 层分配状态。cfg.use_ssm 为 false 时不做事。
    void init(const ModelConfig& cfg, const ModelWeights& weights);

    // 状态清零。维度和已分配的缓冲保持不变。
    void reset();

    const Qwen35SsmDims& dims() const { return dims_; }
    bool initialized() const { return initialized_; }

    float* conv_state_cpu(int layer_idx) { return cpu_states_[layer_idx].conv_state.data(); }
    float* ssm_state_cpu(int layer_idx) { return cpu_states_[layer_idx].ssm_state.data(); }

    // 未编译 CUDA 时 device_states_ 为空, 返回 nullptr 而不是越界。
    float* conv_state_gpu(int layer_idx) {
        if (layer_idx >= static_cast<int>(device_states_.size())) return nullptr;
        return device_states_[layer_idx].conv_state;
    }
    float* ssm_state_gpu(int layer_idx) {
        if (layer_idx >= static_cast<int>(device_states_.size())) return nullptr;
        return device_states_[layer_idx].ssm_state;
    }

private:
    struct CpuState {
        std::vector<float> conv_state;  // [(d_conv-1) * conv_channels]
        std::vector<float> ssm_state;   // [num_v_heads * head_v_dim * head_v_dim]
    };
    struct DeviceState {
        float* conv_state = nullptr;
        float* ssm_state = nullptr;
    };

    Qwen35SsmDims dims_;
    std::vector<CpuState> cpu_states_;
    std::vector<DeviceState> device_states_;
    bool initialized_ = false;
};

}  // namespace forge
