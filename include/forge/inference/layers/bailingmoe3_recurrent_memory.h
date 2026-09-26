#pragma once

// BailingMoE3 (Ling 3.0) GDN/KDA 循环状态。
//
// 与 Qwen35 的 Gated Delta Net 同属 delta-net 家族, 但权重布局不同:
//   - 三个独立的因果卷积: ssm_conv1d_q / ssm_conv1d_k / ssm_conv1d_v
//     (Qwen35 是单个 attn_qkv 后接一个 conv)
//   - 单独的遗忘门 ssm_f_a 与输出门 ssm_g_a (Qwen35 用 ssm_alpha/ssm_beta)
//   - 每个 (k,v) 头一个状态矩阵 S ∈ R^{v_dim × k_dim}
//
// 当前实现为 CPU-only (GPU 缓冲未分配)。状态按 seq_id 惰性分配, 支持
// snapshot/rollback 以配合 speculative decoding。

#include <unordered_map>
#include <vector>

#include "forge/model.h"
#include "forge/model_weights.h"

namespace forge {

struct Bailingmoe3SsmDims {
    int kd = 0;        // per-head key/query dim  (GGUF kda.head_dim, e.g. 128)
    int vd = 0;        // per-head value dim      (== kd for BailingMoE3 KDA)
    int d_conv = 0;    // conv kernel size        (ssm.conv_kernel)
    int num_k_heads = 1;
    int num_v_heads = 1;

    // Every projection (q/k/v/ssm_f_a/ssm_g_a) spans all heads.
    int d_inner() const { return num_v_heads * vd; }
    // 每个 (k,v) 头一个状态矩阵 S ∈ R^{vd × kd}
    int state_size() const { return num_v_heads * vd * kd; }
    // 三个因果卷积 (q/k/v) 都跑满 d_inner 个通道
    int conv_state_size_q() const { return (d_conv - 1) * d_inner(); }
    int conv_state_size_k() const { return (d_conv - 1) * d_inner(); }
    int conv_state_size_v() const { return (d_conv - 1) * d_inner(); }
};

class Bailingmoe3RecurrentMemory {
public:
    ~Bailingmoe3RecurrentMemory();

    // 推导维度并从权重读取缺省值。状态缓冲按 seq_id 惰性分配。
    void init(const ModelConfig& cfg, const ModelWeights& weights);

    bool initialized() const { return initialized_; }
    void reset();

    const Bailingmoe3SsmDims& dims() const { return dims_; }

    // ---- per-sequence state accessors (首次访问时惰性分配该序列缓冲) ----
    float* conv_state_q_cpu(int seq_id, int layer_idx);
    float* conv_state_k_cpu(int seq_id, int layer_idx);
    float* conv_state_v_cpu(int seq_id, int layer_idx);
    float* ssm_state_cpu(int seq_id, int layer_idx);

    // ---- per-sequence device state accessors (CUDA 模式) ----
    float* conv_state_q_dev(int seq_id, int layer_idx);
    float* conv_state_k_dev(int seq_id, int layer_idx);
    float* conv_state_v_dev(int seq_id, int layer_idx);
    float* ssm_state_dev(int seq_id, int layer_idx);

    // ---- speculative rollback (per sequence) ----
    void snapshot(int seq_id);
    void rollback(int seq_id);
    void reset_seq(int seq_id);

    // 切换 CPU/GPU 状态后端。在 set_gpu_layers 中于首次 forward 之前调用;
    // 切换后 ensure_seq 会按对应后端惰性分配缓冲。
    void set_cuda_mode(bool on) { cuda_mode_ = on; }
    bool cuda_mode() const { return cuda_mode_; }

private:
    struct CpuState {
        std::vector<float> conv_q;  // [(d_conv-1) * d_inner]
        std::vector<float> conv_k;  // [(d_conv-1) * d_inner]
        std::vector<float> conv_v;  // [(d_conv-1) * d_inner]
        std::vector<float> ssm_state;  // [num_v_heads * vd * kd]
    };
    // GPU 后端: 所有缓冲都是设备指针 (cudaMalloc), 不含可分页内存。
    struct DevState {
        float* conv_q = nullptr;
        float* conv_k = nullptr;
        float* conv_v = nullptr;
        float* ssm_state = nullptr;
        float* snap_conv_q = nullptr;
        float* snap_conv_k = nullptr;
        float* snap_conv_v = nullptr;
        float* snap_ssm_state = nullptr;
    };

    struct PerSeq {
        std::vector<CpuState> cur;   // 按层 (CPU 后端)
        std::vector<CpuState> snap;  // 快照 (CPU 后端)
        std::vector<DevState> dev;   // 按层 (CUDA 后端)
        bool snap_valid = false;
    };

    PerSeq& ensure_seq(int seq_id);
    void alloc_state(CpuState& s);
    void alloc_device(DevState& s);
    void free_device(DevState& s);

    Bailingmoe3SsmDims dims_;
    int num_layers_ = 0;
    std::unordered_map<int, PerSeq> seqs_;
    bool initialized_ = false;
    bool cuda_mode_ = false;
};

}  // namespace forge
