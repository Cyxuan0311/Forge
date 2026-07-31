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
    for (auto& ds : device_states_) {
        if (ds.conv_state) cudaFree(ds.conv_state);
        if (ds.ssm_state) cudaFree(ds.ssm_state);
    }
#endif
}

void Qwen35RecurrentMemory::init(const ModelConfig& cfg, const ModelWeights& weights) {
    if (!cfg.use_ssm) return;

    dims_.d_state = cfg.ssm_state_size;
    dims_.n_group = cfg.ssm_group_count;
    dims_.dt_rank = cfg.ssm_time_step_rank;
    dims_.d_inner = cfg.ssm_inner_size;
    dims_.d_conv = cfg.ssm_conv_kernel;

    // config 缺失时按第一个 LinearAttention 层的权重形状推导。
    for (int i = 0; i < cfg.num_layers; ++i) {
        const auto& lw = weights.layers[i];
        if (lw.layer_type != LayerType::LinearAttention) continue;
        if (lw.ssm_conv1d() && dims_.d_conv == 0) {
            dims_.d_conv = static_cast<int>(lw.ssm_conv1d()->shape()[0]);
        }
        if (lw.ssm_a() && dims_.dt_rank == 0) {
            dims_.dt_rank = static_cast<int>(lw.ssm_a()->numel());
        }
        break;
    }

    if (dims_.d_state == 0) dims_.d_state = 128;
    if (dims_.n_group == 0) dims_.n_group = 16;
    if (dims_.dt_rank == 0) dims_.dt_rank = 16;
    if (dims_.d_inner == 0) dims_.d_inner = 2 * cfg.hidden_dim;
    if (dims_.d_conv == 0) dims_.d_conv = 4;

    dims_.head_v_dim = dims_.d_inner / dims_.dt_rank;
    dims_.conv_channels = dims_.d_inner + 2 * dims_.n_group * dims_.d_state;

    LOG_INFO("Qwen35RecurrentMemory: Gated Delta Net params:");
    LOG_INFO("  d_inner=" + std::to_string(dims_.d_inner) +
             ", d_state=" + std::to_string(dims_.d_state) +
             ", n_group=" + std::to_string(dims_.n_group) +
             ", dt_rank=" + std::to_string(dims_.dt_rank));
    LOG_INFO("  head_v_dim=" + std::to_string(dims_.head_v_dim) +
             ", conv_channels=" + std::to_string(dims_.conv_channels) +
             ", d_conv=" + std::to_string(dims_.d_conv));

    const int state_size = dims_.state_size();
    const int conv_state_size = dims_.conv_state_size();

    int num_linear_layers = 0;
    for (int i = 0; i < cfg.num_layers; ++i) {
        if (weights.layers[i].layer_type == LayerType::LinearAttention) num_linear_layers++;
    }

    size_t total_ssm_bytes =
        static_cast<size_t>(num_linear_layers) * (state_size + conv_state_size) * sizeof(float);
    LOG_INFO("Qwen35RecurrentMemory: SSM state allocation:");
    LOG_INFO("  num_linear_layers=" + std::to_string(num_linear_layers) +
             ", state_size_per_layer=" + std::to_string(state_size) + " floats (" +
             std::to_string(state_size * sizeof(float) / (1024 * 1024)) + " MB)" +
             ", conv_state_per_layer=" + std::to_string(conv_state_size) + " floats (" +
             std::to_string(conv_state_size * sizeof(float) / 1024) + " KB)");
    LOG_INFO("  Total SSM state: " + std::to_string(total_ssm_bytes / (1024 * 1024)) + " MB");

    // 按 num_layers 分配索引槽, 只有 LinearAttention 层持有真实缓冲,
    // 这样 layer_idx 可以直接当下标用, 与重构前一致。
    cpu_states_.assign(cfg.num_layers, CpuState{});
#ifdef USE_CUDA
    device_states_.assign(cfg.num_layers, DeviceState{});
#endif
    for (int i = 0; i < cfg.num_layers; ++i) {
        if (weights.layers[i].layer_type != LayerType::LinearAttention) continue;
        cpu_states_[i].conv_state.assign(conv_state_size, 0.0f);
        cpu_states_[i].ssm_state.assign(state_size, 0.0f);
#ifdef USE_CUDA
        cudaMalloc(&device_states_[i].conv_state, conv_state_size * sizeof(float));
        cudaMalloc(&device_states_[i].ssm_state, state_size * sizeof(float));
        cudaMemset(device_states_[i].conv_state, 0, conv_state_size * sizeof(float));
        cudaMemset(device_states_[i].ssm_state, 0, state_size * sizeof(float));
#endif
    }

    initialized_ = true;
    LOG_INFO("Qwen35RecurrentMemory: SSM states allocated successfully");
}

void Qwen35RecurrentMemory::reset() {
    for (auto& state : cpu_states_) {
        std::fill(state.conv_state.begin(), state.conv_state.end(), 0.0f);
        std::fill(state.ssm_state.begin(), state.ssm_state.end(), 0.0f);
    }
#ifdef USE_CUDA
    const size_t conv_bytes = static_cast<size_t>(dims_.conv_state_size()) * sizeof(float);
    const size_t ssm_bytes = static_cast<size_t>(dims_.state_size()) * sizeof(float);
    for (auto& ds : device_states_) {
        if (ds.conv_state) cudaMemset(ds.conv_state, 0, conv_bytes);
        if (ds.ssm_state) cudaMemset(ds.ssm_state, 0, ssm_bytes);
    }
#endif
}

}  // namespace forge
