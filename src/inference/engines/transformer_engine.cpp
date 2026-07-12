#include "forge/engines/transformer_engine.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "forge/backend_scheduler.h"
#include "forge/cuda_kernels.h"
#include "forge/logger.h"
#include "forge/operators.h"
#include "forge/perf_profiler.h"

#ifdef USE_CUDA
#    include <cuda_runtime.h>
#endif

#ifdef USE_AVX2
#    include <immintrin.h>
#endif

namespace forge {

TransformerEngine::TransformerEngine(Model& model, InferenceContext& ctx)
    : model_(model), ctx_(ctx), workspace_pool_(model.device()) {}

void TransformerEngine::reset() {
    kv_cache_.reset();
    kv_cache_initialized_ = false;
}

void TransformerEngine::set_gpu_layers(int gpu_layers) {
    gpu_layers_ = gpu_layers;
    const auto& cfg = model_.config();
    int num_layers = cfg.num_layers;

    DeviceType first_dev = layer_device(0);
    auto token_emb = model_.weights().get("token_embedding");
    if (token_emb && token_emb->device() != first_dev) {
        token_emb->to_device(first_dev);
    }

    for (int i = 0; i < num_layers; ++i) {
        weights_.move_layer_weights(i, layer_device(i));
    }

    DeviceType last_dev = layer_device(num_layers - 1);
    weights_.move_output_weights(last_dev);

    LOG_INFO("CPU offload configured: gpu_layers=" + std::to_string(gpu_layers) + "/" +
             std::to_string(num_layers));
}

DeviceType TransformerEngine::layer_device(int layer_idx) const {
    if (gpu_layers_ < 0)
        return model_.device();
    if (layer_idx < gpu_layers_)
        return DeviceType::CUDA;
    return DeviceType::CPU;
}

TensorPtr TransformerEngine::transfer_hidden(const TensorPtr& hidden, DeviceType target) const {
    if (hidden->device() == target)
        return hidden;
    auto transferred = std::make_shared<Tensor>(hidden->dtype(), hidden->shape(), target);
    transferred->copy_from(*hidden);
    return transferred;
}

TensorPtr TransformerEngine::forward(const TensorPtr& input_ids, int64_t start_pos) {
    const auto& cfg = model_.config();
    int seq_len = static_cast<int>(input_ids->numel());

    init_kv_cache(cfg);

    DeviceType first_dev = layer_device(0);
    auto ids_on_dev = transfer_hidden(input_ids, first_dev);

    auto token_emb = model_.weights().get("token_embedding");
    if (!token_emb) {
        fprintf(stderr, "[FATAL] token_embedding is NULL!\n");
        fflush(stderr);
        return nullptr;
    }

    TensorPtr hidden;
    {
        PERF_SCOPE("forward/embedding");
        hidden = ops::embedding(token_emb, ids_on_dev, weights_.token_embedding_fp32);
    }
    if (!hidden) {
        fprintf(stderr, "[FATAL] embedding returned NULL!\n");
        fflush(stderr);
        return nullptr;
    }

    return forward_layers(hidden, seq_len, start_pos);
}

TensorPtr TransformerEngine::forward_from_hidden(const TensorPtr& hidden, int64_t start_pos) {
    const auto& cfg = model_.config();
    int seq_len = static_cast<int>(hidden->shape()[0]);

    init_kv_cache(cfg);

    return forward_layers(hidden, seq_len, start_pos);
}

void TransformerEngine::init_kv_cache(const ModelConfig& cfg) {
    if (kv_cache_initialized_)
        return;

    int kv_max_seq = cfg.max_seq_len;
    const int KV_MAX_SEQ_CAP = 4096;
    if (kv_max_seq > KV_MAX_SEQ_CAP) {
        LOG_INFO("Capping KV cache max_seq_len from " + std::to_string(kv_max_seq) + " to " +
                 std::to_string(KV_MAX_SEQ_CAP) + " to avoid OOM");
        kv_max_seq = KV_MAX_SEQ_CAP;
    }

    DeviceType kv_dev = (gpu_layers_ >= cfg.num_layers) ? DeviceType::CUDA : DeviceType::CPU;
    LOG_INFO("KV cache init: layers=" + std::to_string(cfg.num_layers) + ", kv_heads=" +
             std::to_string(cfg.num_kv_heads) + ", head_dim=" + std::to_string(cfg.head_dim) +
             ", max_seq_len=" + std::to_string(kv_max_seq) +
             ", dev=" + (kv_dev == DeviceType::CUDA ? "CUDA" : "CPU"));
    size_t kv_bytes = (size_t)cfg.num_layers * 2 * (size_t)kv_max_seq * cfg.num_kv_heads *
                      cfg.head_dim * sizeof(float);
    LOG_INFO("KV cache estimated size: " + std::to_string(kv_bytes / (1024 * 1024)) + " MB");
    kv_cache_.init_quantized(cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, kv_max_seq, kv_dev,
                             kv_cache_dtype_);
    kv_cache_initialized_ = true;
    LOG_INFO("KV cache initialized successfully, actual size: " +
             std::to_string(kv_cache_.nbytes() / (1024 * 1024)) + " MB");
}

TensorPtr TransformerEngine::forward_layers(const TensorPtr& hidden, int seq_len,
                                            int64_t start_pos) {
    // Try graph-based execution if a graph builder is available
    if (use_graph_ && graph_builder_) {
        return forward_layers_graph(hidden, seq_len, start_pos);
    }

    // Fallback: try to create a graph builder from registry
    if (use_graph_ && !graph_builder_) {
        const auto& cfg = model_.config();
        auto builder = GraphBuilderRegistry::instance().create(cfg.arch_type);
        if (builder) {
            graph_builder_ = std::move(builder);
            LOG_INFO("Using graph-based execution for arch: " + cfg.arch_type);
            return forward_layers_graph(hidden, seq_len, start_pos);
        } else {
            LOG_WARN("No graph builder for arch: " + cfg.arch_type +
                     ", falling back to imperative mode");
            use_graph_ = false;
        }
    }

    // Imperative execution (original path)
    const auto& cfg = model_.config();
    auto t0 = std::chrono::steady_clock::now();

    auto cur_hidden = hidden;
    for (int layer = 0; layer < cfg.num_layers; ++layer) {
        DeviceType layer_dev = layer_device(layer);
        cur_hidden = transfer_hidden(cur_hidden, layer_dev);
        {
            std::string perf_name = "forward/layer_" + std::to_string(layer);
            PERF_SCOPE(perf_name.c_str());
            cur_hidden = forward_layer(cur_hidden, layer, seq_len, start_pos, layer_dev);
        }
        if (!cur_hidden) {
            fprintf(stderr, "[FATAL] Layer %d returned NULL!\n", layer);
            fflush(stderr);
            return nullptr;
        }
    }

    // Use unified weights for output norm and projection
    auto output_norm = weights_.output_norm;
    {
        PERF_SCOPE("forward/output_norm");
        cur_hidden = ops::rms_norm(cur_hidden, output_norm, cfg.rms_norm_eps);
    }

    auto output_weight = weights_.output_weight;
    if (!output_weight && cfg.tie_embeddings) {
        output_weight = weights_.token_embedding;
    }
    // CPU: output_weight keeps its native quantized format in ModelWeights::init()
    // (Q4_0/Q8_0/Q4_1/Q4_K/Q6_K supported), dispatched to fused GEMV kernels by matmul_transB.
    if (output_weight) {
        cur_hidden = transfer_hidden(cur_hidden, output_weight->device());
    }
    TensorPtr logits;
    {
        PERF_SCOPE("forward/output_proj");
        // Use specialized output_proj kernel for Q4_0 decode (M=1, large N)
        if (output_weight && output_weight->device() == DeviceType::CUDA &&
            output_weight->dtype() == DataType::Q4_0 && seq_len == 1) {
            int K = static_cast<int>(output_weight->shape()[1]);
            int N = static_cast<int>(output_weight->shape()[0]);
            logits = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, N},
                                              DeviceType::CUDA);
            cuda::launch_output_proj_q4_0(static_cast<const float*>(cur_hidden->data()),
                                          output_weight->data(),
                                          static_cast<float*>(logits->data()), K, N);
        } else {
            logits = ops::matmul_transB(cur_hidden, output_weight);
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    LOG_INFO("Forward total: " + std::to_string((int)total_ms) + "ms (seq_len=" +
             std::to_string(seq_len) + ", start_pos=" + std::to_string(start_pos) + ")");

    return logits;
}

TensorPtr TransformerEngine::forward_layers_graph(const TensorPtr& hidden, int seq_len,
                                                  int64_t start_pos) {
    const auto& cfg = model_.config();
    auto t0 = std::chrono::steady_clock::now();

    ComputeGraph graph;

    // Add hidden state as graph input
    int hidden_idx = graph.add_input(hidden);

    // Build layer graphs
    int cur_idx = hidden_idx;
    for (int layer = 0; layer < cfg.num_layers; ++layer) {
        DeviceType layer_dev = layer_device(layer);
        const auto& lw = weights_.layers[layer];
        cur_idx = graph_builder_->build_layer_graph(graph, cur_idx, lw, cfg, layer, seq_len,
                                                    start_pos, layer_dev, kv_cache_);
    }

    // Build output head graph
    int output_idx = graph_builder_->build_output_graph(graph, cur_idx, weights_, cfg);

    // Schedule: assign nodes to optimal devices
    {
        BackendScheduler scheduler;
        SchedulingPlan plan = scheduler.schedule(graph);
        if (plan.valid) {
            graph.apply_schedule(plan);
        }
    }

    // Execute the graph
    auto result = graph.execute();

    auto t1 = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    LOG_INFO("Forward (graph) total: " + std::to_string((int)total_ms) +
             "ms (seq_len=" + std::to_string(seq_len) + ", start_pos=" + std::to_string(start_pos) +
             ", nodes=" + std::to_string(graph.num_nodes()) + ")");

    return result;
}

void TransformerEngine::apply_rope_standard(const float* q_data, const float* k_data, float* q_out,
                                            float* k_out, int seq_len, int num_heads,
                                            int num_kv_heads, int head_dim, int64_t start_pos,
                                            float theta) {
    int half_dim = head_dim / 2;
    int q_stride = num_heads * head_dim;
    int k_stride = num_kv_heads * head_dim;
    for (int s = 0; s < seq_len; ++s) {
        for (int h = 0; h < num_heads; ++h) {
            for (int d = 0; d < half_dim; ++d) {
                float freq = 1.0f / std::pow(theta, 2.0f * d / head_dim);
                float angle = (start_pos + s) * freq;
                float cos_a = std::cos(angle);
                float sin_a = std::sin(angle);

                int q_idx0 = s * q_stride + h * head_dim + d;
                int q_idx1 = q_idx0 + half_dim;

                q_out[q_idx0] = q_data[q_idx0] * cos_a - q_data[q_idx1] * sin_a;
                q_out[q_idx1] = q_data[q_idx0] * sin_a + q_data[q_idx1] * cos_a;

                if (h < num_kv_heads) {
                    int k_idx0 = s * k_stride + h * head_dim + d;
                    int k_idx1 = k_idx0 + half_dim;

                    k_out[k_idx0] = k_data[k_idx0] * cos_a - k_data[k_idx1] * sin_a;
                    k_out[k_idx1] = k_data[k_idx0] * sin_a + k_data[k_idx1] * cos_a;
                }
            }
        }
    }
}

TensorPtr TransformerEngine::expand_kv_heads(const TensorPtr& kv, int seq_len, int num_heads,
                                             int num_kv_heads, int head_dim, DeviceType dev) {
    int kv_groups = num_heads / num_kv_heads;
    auto expanded = std::make_shared<Tensor>(
        DataType::FP32, std::vector<int64_t>{seq_len, num_heads * head_dim}, dev);

    if (dev == DeviceType::CUDA) {
#ifdef USE_CUDA
        cuda::launch_expand_kv(static_cast<const float*>(kv->data()),
                               static_cast<float*>(expanded->data()), seq_len, num_heads,
                               num_kv_heads, head_dim);
#endif
    } else {
        const float* kv_data = static_cast<const float*>(kv->data());
        float* out_data = static_cast<float*>(expanded->data());
#ifdef USE_AVX2
// Vectorized expand: copy each KV head to all query heads in its group.
// For TinyLlama: kv_groups=8, head_dim=64 (8 AVX2 vectors per head).
// Process all seq_len positions, each KV head is replicated kv_groups times.
#    pragma omp parallel for schedule(static)
        for (int s = 0; s < seq_len; ++s) {
            const float* kv_row = kv_data + s * num_kv_heads * head_dim;
            float* out_row = out_data + s * num_heads * head_dim;
            for (int kv_h = 0; kv_h < num_kv_heads; ++kv_h) {
                const float* src = kv_row + kv_h * head_dim;
                // Replicate this KV head to all query heads in its group
                for (int g = 0; g < kv_groups; ++g) {
                    int dst_h = kv_h * kv_groups + g;
                    float* dst = out_row + dst_h * head_dim;
                    // head_dim is typically 64 = 8 * 8 floats, use AVX2
                    for (int d = 0; d + 8 <= head_dim; d += 8) {
                        __m256 v = _mm256_loadu_ps(src + d);
                        _mm256_storeu_ps(dst + d, v);
                    }
                    // Handle remaining elements (unlikely for head_dim=64)
                    for (int d = (head_dim / 8) * 8; d < head_dim; ++d) {
                        dst[d] = src[d];
                    }
                }
            }
        }
#else
        for (int s = 0; s < seq_len; ++s) {
            for (int h = 0; h < num_heads; ++h) {
                int kv_h = h / kv_groups;
                for (int d = 0; d < head_dim; ++d) {
                    out_data[s * num_heads * head_dim + h * head_dim + d] =
                        kv_data[s * num_kv_heads * head_dim + kv_h * head_dim + d];
                }
            }
        }
#endif
    }

    return expanded;
}

}  // namespace forge
