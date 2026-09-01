#pragma once

// Qwen35RecurrentMemory: Qwen3.5 linear attention (Gated Delta Net) 的循环状态。
//
// 重构前 conv state / ssm state 及其 CPU/GPU 双份缓冲直接定义在 Qwen35Engine
// header 里, SSM 维度也是 engine 的七个成员变量, 任何想单独测试 delta net 的代码
// 都必须构造整个 engine。这里把状态所有权和维度推导收敛为一个组件。
//
// 阶段 8 已把所有权收归 InferenceContext 的 HybridMemory, engine 只持有引用。
// 状态是逐 token 累积的, 与 KV cache 一样属于"推理期可变状态"。
//
// 2.3: 状态改为按 seq_id 隔离（每序列一份 conv/ssm 状态, 惰性分配）。speculative
// decoding 部分拒绝 draft 时, 通过 snapshot(seq_id)/rollback(seq_id) 把状态回滚到
// 接受点, 与 KVCache::rollback 的语义对齐。

#include <unordered_map>
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

    // 推导维度。状态缓冲按 seq_id 惰性分配（首次访问某序列时创建）。
    void init(const ModelConfig& cfg, const ModelWeights& weights);

    // 清空所有序列的状态内容（已分配的缓冲保留, 维度不变）。
    void reset();

    const Qwen35SsmDims& dims() const { return dims_; }
    bool initialized() const { return initialized_; }

    // ---- per-sequence state accessors (首次访问时惰性分配该序列缓冲) ----
    float* conv_state_cpu(int seq_id, int layer_idx);
    float* ssm_state_cpu(int seq_id, int layer_idx);

    // 未编译 CUDA 或该序列无 GPU 缓冲时返回 nullptr 而不是越界。
    float* conv_state_gpu(int seq_id, int layer_idx);
    float* ssm_state_gpu(int seq_id, int layer_idx);

    // ---- speculative rollback (per sequence) ----
    // snapshot: 保存当前状态到快照缓冲（CPU 深拷贝 + GPU D2D 拷贝）。
    // rollback: 从快照恢复; 若该序列从未 snapshot 过则为 no-op。
    // reset_seq: 仅清零该序列的状态内容。
    void snapshot(int seq_id);
    void rollback(int seq_id);
    void reset_seq(int seq_id);

private:
    struct CpuState {
        std::vector<float> conv_state;  // [(d_conv-1) * conv_channels]
        std::vector<float> ssm_state;   // [num_v_heads * head_v_dim * head_v_dim]
    };
    struct DeviceState {
        float* conv_state = nullptr;
        float* ssm_state = nullptr;
    };
    // 每序列: 活跃状态(CPU+GPU) + 快照(CPU+GPU)。快照 GPU 缓冲在首次
    // snapshot() 时惰性分配。snap_valid 标记该序列是否 snapshot 过:
    // 从未 snapshot 的序列 rollback() 必须是 no-op（否则会错误地把尚未
    // 初始化的零快照写回活跃状态）。
    struct PerSeqState {
        std::vector<CpuState> cpu;
        std::vector<DeviceState> dev;
        std::vector<CpuState> cpu_snap;
        std::vector<DeviceState> dev_snap;
        bool snap_valid = false;
    };

    PerSeqState& ensure_seq(int seq_id);

    Qwen35SsmDims dims_;
    int num_layers_ = 0;
    std::vector<int> linear_layer_idx_;  // 只对 LinearAttention 层持有真实缓冲
    std::unordered_map<int, PerSeqState> seqs_;
    bool initialized_ = false;
};

}  // namespace forge
