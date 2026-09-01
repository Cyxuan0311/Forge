#include "forge/inference/layers/qwen35_recurrent_memory.h"

#include <algorithm>
#include <string>

#include "forge/logger.h"

#ifdef USE_CUDA
#    include <cuda_runtime.h>
#endif

namespace forge {

Qwen35RecurrentMemory::~Qwen35RecurrentMemory() {
#ifdef USE_CUDA
    for (auto& [seq_id, seq] : seqs_) {
        (void)seq_id;
        for (auto& ds : seq.dev) {
            if (ds.conv_state)
                cudaFree(ds.conv_state);
            if (ds.ssm_state)
                cudaFree(ds.ssm_state);
        }
        for (auto& ds : seq.dev_snap) {
            if (ds.conv_state)
                cudaFree(ds.conv_state);
            if (ds.ssm_state)
                cudaFree(ds.ssm_state);
        }
    }
#endif
}

void Qwen35RecurrentMemory::init(const ModelConfig& cfg, const ModelWeights& weights) {
    if (!cfg.use_ssm)
        return;

    dims_.d_state = cfg.ssm_state_size;
    dims_.n_group = cfg.ssm_group_count;
    dims_.dt_rank = cfg.ssm_time_step_rank;
    dims_.d_inner = cfg.ssm_inner_size;
    dims_.d_conv = cfg.ssm_conv_kernel;

    // config 缺失时按第一个 LinearAttention 层的权重形状推导。
    for (int i = 0; i < cfg.num_layers; ++i) {
        const auto& lw = weights.layers[i];
        if (lw.layer_type != LayerType::LinearAttention)
            continue;
        if (lw.ssm_conv1d() && dims_.d_conv == 0) {
            dims_.d_conv = static_cast<int>(lw.ssm_conv1d()->shape()[0]);
        }
        if (lw.ssm_a() && dims_.dt_rank == 0) {
            dims_.dt_rank = static_cast<int>(lw.ssm_a()->numel());
        }
        break;
    }

    if (dims_.d_state == 0)
        dims_.d_state = 128;
    if (dims_.n_group == 0)
        dims_.n_group = 16;
    if (dims_.dt_rank == 0)
        dims_.dt_rank = 16;
    if (dims_.d_inner == 0)
        dims_.d_inner = 2 * cfg.hidden_dim;
    if (dims_.d_conv == 0)
        dims_.d_conv = 4;

    dims_.head_v_dim = dims_.d_inner / dims_.dt_rank;
    dims_.conv_channels = dims_.d_inner + 2 * dims_.n_group * dims_.d_state;

    LOG_INFO("Qwen35RecurrentMemory: Gated Delta Net params:");
    LOG_INFO("  d_inner=" + std::to_string(dims_.d_inner) + ", d_state=" +
             std::to_string(dims_.d_state) + ", n_group=" + std::to_string(dims_.n_group) +
             ", dt_rank=" + std::to_string(dims_.dt_rank));
    LOG_INFO("  head_v_dim=" + std::to_string(dims_.head_v_dim) + ", conv_channels=" +
             std::to_string(dims_.conv_channels) + ", d_conv=" + std::to_string(dims_.d_conv));

    const int state_size = dims_.state_size();
    const int conv_state_size = dims_.conv_state_size();

    num_layers_ = cfg.num_layers;
    linear_layer_idx_.clear();
    for (int i = 0; i < cfg.num_layers; ++i) {
        if (weights.layers[i].layer_type == LayerType::LinearAttention) {
            linear_layer_idx_.push_back(i);
        }
    }

    const size_t total_ssm_bytes = static_cast<size_t>(linear_layer_idx_.size()) *
                                   (state_size + conv_state_size) * sizeof(float);
    LOG_INFO("Qwen35RecurrentMemory: SSM state allocation (per-sequence, lazy):");
    LOG_INFO("  num_linear_layers=" + std::to_string(linear_layer_idx_.size()) +
             ", state_size_per_layer=" + std::to_string(state_size) + " floats (" +
             std::to_string(state_size * sizeof(float) / (1024 * 1024)) + " MB)" +
             ", conv_state_per_layer=" + std::to_string(conv_state_size) + " floats (" +
             std::to_string(conv_state_size * sizeof(float) / 1024) + " KB)");
    LOG_INFO("  Total SSM state per sequence: " + std::to_string(total_ssm_bytes / (1024 * 1024)) +
             " MB");

    initialized_ = true;
    LOG_INFO("Qwen35RecurrentMemory: SSM state allocation ready");
}

Qwen35RecurrentMemory::PerSeqState& Qwen35RecurrentMemory::ensure_seq(int seq_id) {
    auto it = seqs_.find(seq_id);
    if (it != seqs_.end())
        return it->second;

    PerSeqState seq;
    seq.cpu.assign(num_layers_, CpuState{});
    seq.cpu_snap.assign(num_layers_, CpuState{});
#ifdef USE_CUDA
    seq.dev.assign(num_layers_, DeviceState{});
    seq.dev_snap.assign(num_layers_, DeviceState{});
#endif
    const int conv_state_size = dims_.conv_state_size();
    const int state_size = dims_.state_size();
    for (int layer_idx : linear_layer_idx_) {
        seq.cpu[layer_idx].conv_state.assign(conv_state_size, 0.0f);
        seq.cpu[layer_idx].ssm_state.assign(state_size, 0.0f);
        // 快照 CPU 缓冲与活跃缓冲同大小, 内容在 snapshot() 时写入。
        seq.cpu_snap[layer_idx].conv_state.assign(conv_state_size, 0.0f);
        seq.cpu_snap[layer_idx].ssm_state.assign(state_size, 0.0f);
#ifdef USE_CUDA
        cudaMalloc(&seq.dev[layer_idx].conv_state, conv_state_size * sizeof(float));
        cudaMalloc(&seq.dev[layer_idx].ssm_state, state_size * sizeof(float));
        cudaMemset(seq.dev[layer_idx].conv_state, 0, conv_state_size * sizeof(float));
        cudaMemset(seq.dev[layer_idx].ssm_state, 0, state_size * sizeof(float));
#endif
    }
    // GPU 快照缓冲在首次 snapshot() 时惰性分配。

    auto [it2, inserted] = seqs_.emplace(seq_id, std::move(seq));
    (void)inserted;
    return it2->second;
}

float* Qwen35RecurrentMemory::conv_state_cpu(int seq_id, int layer_idx) {
    return ensure_seq(seq_id).cpu[layer_idx].conv_state.data();
}

float* Qwen35RecurrentMemory::ssm_state_cpu(int seq_id, int layer_idx) {
    return ensure_seq(seq_id).cpu[layer_idx].ssm_state.data();
}

float* Qwen35RecurrentMemory::conv_state_gpu(int seq_id, int layer_idx) {
    auto& seq = ensure_seq(seq_id);
    if (layer_idx >= static_cast<int>(seq.dev.size()))
        return nullptr;
    return seq.dev[layer_idx].conv_state;
}

float* Qwen35RecurrentMemory::ssm_state_gpu(int seq_id, int layer_idx) {
    auto& seq = ensure_seq(seq_id);
    if (layer_idx >= static_cast<int>(seq.dev.size()))
        return nullptr;
    return seq.dev[layer_idx].ssm_state;
}

void Qwen35RecurrentMemory::snapshot(int seq_id) {
    auto& seq = ensure_seq(seq_id);
    for (int layer_idx : linear_layer_idx_) {
        seq.cpu_snap[layer_idx].conv_state = seq.cpu[layer_idx].conv_state;
        seq.cpu_snap[layer_idx].ssm_state = seq.cpu[layer_idx].ssm_state;
    }
    seq.snap_valid = true;
#ifdef USE_CUDA
    const int conv_state_size = dims_.conv_state_size();
    const int state_size = dims_.state_size();
    for (int layer_idx : linear_layer_idx_) {
        auto& snap = seq.dev_snap[layer_idx];
        if (!snap.conv_state) {
            cudaMalloc(&snap.conv_state, conv_state_size * sizeof(float));
            cudaMalloc(&snap.ssm_state, state_size * sizeof(float));
        }
        cudaMemcpy(snap.conv_state, seq.dev[layer_idx].conv_state, conv_state_size * sizeof(float),
                   cudaMemcpyDeviceToDevice);
        cudaMemcpy(snap.ssm_state, seq.dev[layer_idx].ssm_state, state_size * sizeof(float),
                   cudaMemcpyDeviceToDevice);
    }
#endif
}

void Qwen35RecurrentMemory::rollback(int seq_id) {
    auto it = seqs_.find(seq_id);
    if (it == seqs_.end())
        return;  // 该序列尚无状态, 无需回滚
    auto& seq = it->second;
    if (!seq.snap_valid)
        return;  // 从未 snapshot 过 -> no-op
    for (int layer_idx : linear_layer_idx_) {
        seq.cpu[layer_idx].conv_state = seq.cpu_snap[layer_idx].conv_state;
        seq.cpu[layer_idx].ssm_state = seq.cpu_snap[layer_idx].ssm_state;
    }
#ifdef USE_CUDA
    const int conv_state_size = dims_.conv_state_size();
    const int state_size = dims_.state_size();
    for (int layer_idx : linear_layer_idx_) {
        auto& snap = seq.dev_snap[layer_idx];
        if (!snap.conv_state)
            continue;  // 从未 snapshot 过
        cudaMemcpy(seq.dev[layer_idx].conv_state, snap.conv_state, conv_state_size * sizeof(float),
                   cudaMemcpyDeviceToDevice);
        cudaMemcpy(seq.dev[layer_idx].ssm_state, snap.ssm_state, state_size * sizeof(float),
                   cudaMemcpyDeviceToDevice);
    }
#endif
}

void Qwen35RecurrentMemory::reset_seq(int seq_id) {
    auto it = seqs_.find(seq_id);
    if (it == seqs_.end())
        return;
    auto& seq = it->second;
    for (int layer_idx : linear_layer_idx_) {
        std::fill(seq.cpu[layer_idx].conv_state.begin(), seq.cpu[layer_idx].conv_state.end(), 0.0f);
        std::fill(seq.cpu[layer_idx].ssm_state.begin(), seq.cpu[layer_idx].ssm_state.end(), 0.0f);
    }
#ifdef USE_CUDA
    const int conv_state_size = dims_.conv_state_size();
    const int state_size = dims_.state_size();
    for (int layer_idx : linear_layer_idx_) {
        auto& ds = seq.dev[layer_idx];
        if (ds.conv_state)
            cudaMemset(ds.conv_state, 0, conv_state_size * sizeof(float));
        if (ds.ssm_state)
            cudaMemset(ds.ssm_state, 0, state_size * sizeof(float));
    }
#endif
    seq.snap_valid = false;  // 状态已清空, 之前的快照失效
}

void Qwen35RecurrentMemory::reset() {
    std::vector<int> seq_ids;
    seq_ids.reserve(seqs_.size());
    for (auto& [seq_id, seq] : seqs_) {
        (void)seq;
        seq_ids.push_back(seq_id);
    }
    for (int seq_id : seq_ids) {
        reset_seq(seq_id);
    }
}

}  // namespace forge
