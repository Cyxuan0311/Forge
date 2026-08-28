#include "forge/inference/layers/gemma4_moe.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#include "forge/cuda_kernels.h"
#include "forge/engines/transformer_engine.h"
#include "forge/inference/tensor_device_utils.h"
#include "forge/operators.h"
#include "forge/perf_profiler.h"
#include "forge/quant_traits.h"

namespace forge {

namespace {

// Router 输入: 可选的 gate_inp_s 缩放 (rms_norm 后乘 scale * 1/sqrt(hidden))。
TensorPtr build_router_input(const TensorPtr& attn_residual, const LayerWeights& lw,
                            const ModelConfig& cfg, int seq_len) {
    if (!lw.ffn_gate_inp_s()) return attn_residual;

    bool use_gpu = (attn_residual->device() == DeviceType::CUDA &&
                    lw.ffn_gate_inp_s()->device() == DeviceType::CUDA);
    auto ones = std::make_shared<Tensor>(DataType::FP32,
                                        std::vector<int64_t>{1, cfg.hidden_dim},
                                        use_gpu ? DeviceType::CUDA : DeviceType::CPU);
    float* ones_data = static_cast<float*>(ones->data());
    std::fill_n(ones_data, cfg.hidden_dim, 1.0f);

    auto normed_for_router = ops::rms_norm(attn_residual, ones, cfg.rms_norm_eps);
    float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(cfg.hidden_dim));

#ifdef USE_CUDA
    if (use_gpu) {
        auto scaled = std::make_shared<Tensor>(DataType::FP32, normed_for_router->shape(),
                                              DeviceType::CUDA);
        cuda::launch_moe_router_scale(static_cast<const float*>(normed_for_router->data()),
                                      static_cast<const float*>(lw.ffn_gate_inp_s()->data()),
                                      static_cast<float*>(scaled->data()), cfg.hidden_dim, inv_sqrt,
                                      cfg.rms_norm_eps, seq_len);
        return scaled;
    }
#endif
    normed_for_router = ensure_cpu(normed_for_router);
    float* nr_data = static_cast<float*>(normed_for_router->data());
    int nr_n = static_cast<int>(normed_for_router->numel());
    for (int i = 0; i < nr_n; ++i) nr_data[i] *= inv_sqrt;
    auto s_cpu = ensure_cpu(lw.ffn_gate_inp_s());
    const float* s_data = static_cast<const float*>(s_cpu->data());
    for (int s = 0; s < seq_len; ++s) {
        float* row = nr_data + s * cfg.hidden_dim;
        for (int d = 0; d < cfg.hidden_dim; ++d) {
            row[d] *= s_data[d];
        }
    }
    return normed_for_router;
}

#ifdef USE_CUDA
// 按 dtype 分派 MoE expert GEMV。不支持的 dtype 不做任何事, 与重构前一致。
void moe_gemv(DataType dt, const float* x, const void* w, float* out, const int* d_indices,
              const float* d_weights, int K, int N, int n_expert, int n_expert_used, int seq_len) {
    switch (dt) {
        case DataType::Q4_0:
            cuda::launch_moe_expert_gemv<DataType::Q4_0>(x, w, out, d_indices, d_weights, K, N,
                                                         n_expert, n_expert_used, seq_len);
            break;
        case DataType::Q4_K:
            cuda::launch_moe_expert_gemv<DataType::Q4_K>(x, w, out, d_indices, d_weights, K, N,
                                                         n_expert, n_expert_used, seq_len);
            break;
        case DataType::Q6_K:
            cuda::launch_moe_expert_gemv<DataType::Q6_K>(x, w, out, d_indices, d_weights, K, N,
                                                         n_expert, n_expert_used, seq_len);
            break;
        case DataType::Q8_0:
            cuda::launch_moe_expert_gemv<DataType::Q8_0>(x, w, out, d_indices, d_weights, K, N,
                                                         n_expert, n_expert_used, seq_len);
            break;
        default:
            break;
    }
}
#endif

}  // namespace

TensorPtr Gemma4Moe::apply(const TensorPtr& attn_residual, const LayerExecutionContext& lctx,
                             TransformerEngine* engine) {
    PERF_SCOPE("layer/moe");
    const auto& cfg = lctx.config;
    const auto& lw = lctx.weights;
    const int seq_len = lctx.seq_len();
    const DeviceType dev = lctx.device.type;

    // ---- Shared expert (标准 GeGLU FFN) ----
    TensorPtr shared_out;
    {
        auto cur_mlp = ops::rms_norm(attn_residual, lw.ffn_norm(), cfg.rms_norm_eps);
        auto gate = ops::matmul_transB(cur_mlp, lw.w1());
        auto up = ops::matmul_transB(cur_mlp, lw.w3());
        auto gated = ops::gelu_multiply(gate, up);
        shared_out = ops::matmul_transB(gated, lw.w2());
    }
    if (lw.ffn_post_norm_1()) {
        shared_out = ops::rms_norm(shared_out, lw.ffn_post_norm_1(), cfg.rms_norm_eps);
    }

    // ---- Routed experts ----
    auto cur_moe = ops::rms_norm(attn_residual, lw.ffn_pre_norm_2(), cfg.rms_norm_eps);
    auto router_input = build_router_input(attn_residual, lw, cfg, seq_len);
    auto router_logits = ops::matmul_transB(router_input, lw.ffn_gate_inp());

    const int n_expert = cfg.n_expert;
    const int n_expert_used = cfg.n_expert_used > 0 ? cfg.n_expert_used : 1;

    auto expert_out = std::make_shared<Tensor>(DataType::FP32,
                                              std::vector<int64_t>{seq_len, cfg.hidden_dim}, dev);
    std::fill_n(static_cast<float*>(expert_out->data()), seq_len * cfg.hidden_dim, 0.0f);

    if (router_logits->device() == DeviceType::CUDA && dev == DeviceType::CUDA) {
#ifdef USE_CUDA
        // 全 GPU 路径, 无 CPU 往返。
        auto indices_tensor = std::make_shared<Tensor>(
            DataType::INT32, std::vector<int64_t>{seq_len * n_expert_used}, DeviceType::CUDA);
        auto weights_tensor = std::make_shared<Tensor>(
            DataType::FP32, std::vector<int64_t>{seq_len * n_expert_used}, DeviceType::CUDA);
        auto softmax_buf = std::make_shared<Tensor>(
            DataType::FP32, std::vector<int64_t>{seq_len * n_expert}, DeviceType::CUDA);

        cuda::launch_moe_router(static_cast<const float*>(router_logits->data()),
                                static_cast<int*>(indices_tensor->data()),
                                static_cast<float*>(weights_tensor->data()),
                                static_cast<float*>(softmax_buf->data()), n_expert, n_expert_used,
                                seq_len);

        const int* d_indices = static_cast<const int*>(indices_tensor->data());
        const float* d_weights = static_cast<const float*>(weights_tensor->data());
        auto w_dtype = lw.ffn_gate_up_exps()
                           ? lw.ffn_gate_up_exps()->dtype()
                           : (lw.ffn_gate_exps() ? lw.ffn_gate_exps()->dtype() : DataType::FP32);

        if (lw.ffn_gate_up_exps() && is_quantized_type(w_dtype)) {
            // 融合 gate_up: GEMV -> GeGLU split -> down GEMV
            int K_gate = cfg.hidden_dim;
            int N_gate = static_cast<int>(lw.ffn_gate_up_exps()->shape()[1]);
            int n_ff = N_gate / 2;

            auto gate_up_out = std::make_shared<Tensor>(
                DataType::FP32, std::vector<int64_t>{seq_len, N_gate}, DeviceType::CUDA);
            cudaMemset(static_cast<float*>(gate_up_out->data()), 0,
                       seq_len * N_gate * sizeof(float));

            moe_gemv(w_dtype, static_cast<const float*>(cur_moe->data()),
                     lw.ffn_gate_up_exps()->data(), static_cast<float*>(gate_up_out->data()),
                     d_indices, d_weights, K_gate, N_gate, n_expert, n_expert_used, seq_len);

            auto gated = std::make_shared<Tensor>(DataType::FP32,
                                                 std::vector<int64_t>{seq_len, n_ff},
                                                 DeviceType::CUDA);
            cuda::launch_gelu_tanh_multiply_split(static_cast<const float*>(gate_up_out->data()),
                                                 static_cast<float*>(gated->data()), n_ff, seq_len);

            moe_gemv(lw.ffn_down_exps()->dtype(), static_cast<const float*>(gated->data()),
                     lw.ffn_down_exps()->data(), static_cast<float*>(expert_out->data()), d_indices,
                     d_weights, n_ff, cfg.hidden_dim, n_expert, n_expert_used, seq_len);

        } else if (lw.ffn_gate_exps() && lw.ffn_up_exps()) {
            // 分离的 gate + up
            int K_gu = cfg.hidden_dim;
            int N_gate = static_cast<int>(lw.ffn_gate_exps()->shape()[1]);
            int N_up = static_cast<int>(lw.ffn_up_exps()->shape()[1]);

            auto gate_out = std::make_shared<Tensor>(
                DataType::FP32, std::vector<int64_t>{seq_len, N_gate}, DeviceType::CUDA);
            cudaMemset(static_cast<float*>(gate_out->data()), 0, seq_len * N_gate * sizeof(float));
            moe_gemv(lw.ffn_gate_exps()->dtype(), static_cast<const float*>(cur_moe->data()),
                     lw.ffn_gate_exps()->data(), static_cast<float*>(gate_out->data()), d_indices,
                     d_weights, K_gu, N_gate, n_expert, n_expert_used, seq_len);

            auto up_out = std::make_shared<Tensor>(DataType::FP32,
                                                  std::vector<int64_t>{seq_len, N_up},
                                                  DeviceType::CUDA);
            cudaMemset(static_cast<float*>(up_out->data()), 0, seq_len * N_up * sizeof(float));
            moe_gemv(lw.ffn_up_exps()->dtype(), static_cast<const float*>(cur_moe->data()),
                     lw.ffn_up_exps()->data(), static_cast<float*>(up_out->data()), d_indices,
                     d_weights, K_gu, N_up, n_expert, n_expert_used, seq_len);

            auto gated = std::make_shared<Tensor>(DataType::FP32,
                                                 std::vector<int64_t>{seq_len, N_gate},
                                                 DeviceType::CUDA);
            cuda::launch_gelu_multiply(static_cast<const float*>(gate_out->data()),
                                       static_cast<const float*>(up_out->data()),
                                       static_cast<float*>(gated->data()), seq_len * N_gate);

            moe_gemv(lw.ffn_down_exps()->dtype(), static_cast<const float*>(gated->data()),
                     lw.ffn_down_exps()->data(), static_cast<float*>(expert_out->data()), d_indices,
                     d_weights, N_gate, cfg.hidden_dim, n_expert, n_expert_used, seq_len);

            // Phase P0 hook (CUDA): router decisions live on device; mirror the
            // top-k indices to host and report them. P0: empty sync_experts_resident.
            if (engine) {
                auto idx_h = ensure_cpu(indices_tensor);
                const int* id = static_cast<const int*>(idx_h->data());
                std::vector<int> active(id, id + seq_len * n_expert_used);
                engine->sync_experts_resident(lctx.layer_idx, active);
            }
        }
#endif
    } else {
        // CPU router 路径: 逐 token 做 softmax + top-k, 再逐 expert 计算。
        auto router_logits_cpu = ensure_cpu(router_logits);
        auto cur_moe_cpu = ensure_cpu(cur_moe);
        const float* logits_data = static_cast<const float*>(router_logits_cpu->data());

        for (int s = 0; s < seq_len; ++s) {
            std::vector<float> probs(n_expert);
            float max_logit = -std::numeric_limits<float>::infinity();
            for (int e = 0; e < n_expert; ++e) {
                probs[e] = logits_data[s * n_expert + e];
                if (probs[e] > max_logit) max_logit = probs[e];
            }
            float sum_exp = 0.0f;
            for (int e = 0; e < n_expert; ++e) {
                probs[e] = std::exp(probs[e] - max_logit);
                sum_exp += probs[e];
            }
            for (int e = 0; e < n_expert; ++e) {
                probs[e] /= sum_exp;
            }

            std::vector<int> indices(n_expert);
            std::iota(indices.begin(), indices.end(), 0);
            std::partial_sort(indices.begin(), indices.begin() + n_expert_used, indices.end(),
                              [&](int a, int b) { return probs[a] > probs[b]; });

            float topk_sum = 0.0f;
            for (int k = 0; k < n_expert_used; ++k) {
                topk_sum += probs[indices[k]];
            }

            // Phase P0 hook: report the routed (top-k) experts for this token so
            // later phases can page them onto the device. P0: no-op when engine
            // is null or sync_experts_resident() is the default empty impl.
            if (engine && n_expert_used > 0) {
                std::vector<int> active(indices.begin(),
                                         indices.begin() + n_expert_used);
                engine->sync_experts_resident(lctx.layer_idx, active);
            }

            expert_out = ensure_cpu(expert_out);
            float* expert_out_data = static_cast<float*>(expert_out->data());

            for (int k = 0; k < n_expert_used; ++k) {
                int expert_idx = indices[k];
                float weight = probs[expert_idx] / topk_sum;

                auto token_hidden = std::make_shared<Tensor>(
                    DataType::FP32, std::vector<int64_t>{1, cfg.hidden_dim}, DeviceType::CPU);
                const float* moe_in =
                    static_cast<const float*>(cur_moe_cpu->data()) + s * cfg.hidden_dim;
                std::memcpy(token_hidden->data(), moe_in, cfg.hidden_dim * sizeof(float));

                // 零拷贝 expert 抽取: 沿 expert 维切片再 view 成 2D, 避免反量化
                // 整个 3D 权重 (n_expert_used << n_expert 时浪费 99%+ 算力)。
                auto extract_expert_2d = [&](const TensorPtr& w3d) -> TensorPtr {
                    if (!w3d) return nullptr;
                    auto& shp = w3d->shape();
                    if (shp.size() < 2) return w3d;
                    if (shp.size() == 3 && expert_idx < shp[2]) {
                        auto expert_slice = w3d->slice(2, expert_idx, expert_idx + 1);
                        return std::make_shared<Tensor>(expert_slice.view({shp[0], shp[1]}));
                    }
                    return w3d;
                };

                TensorPtr expert_result;
                if (lw.ffn_gate_up_exps()) {
                    auto gate_up_w = extract_expert_2d(lw.ffn_gate_up_exps());
                    auto down_w = extract_expert_2d(lw.ffn_down_exps());
                    if (gate_up_w && down_w) {
                        auto gate_up = ops::matmul_transB(token_hidden, gate_up_w);
                        int half_out = static_cast<int>(gate_up->shape()[1]) / 2;
                        auto gate_t = std::make_shared<Tensor>(
                            DataType::FP32, std::vector<int64_t>{1, half_out}, DeviceType::CPU);
                        auto up_t = std::make_shared<Tensor>(
                            DataType::FP32, std::vector<int64_t>{1, half_out}, DeviceType::CPU);
                        auto gate_up_cpu = ensure_cpu(gate_up);
                        const float* gu_data = static_cast<const float*>(gate_up_cpu->data());
                        std::memcpy(gate_t->data(), gu_data, half_out * sizeof(float));
                        std::memcpy(up_t->data(), gu_data + half_out, half_out * sizeof(float));
                        auto gated = ops::gelu_multiply(gate_t, up_t);
                        expert_result = ops::matmul_transB(gated, down_w);
                    }
                } else if (lw.ffn_gate_exps() && lw.ffn_up_exps()) {
                    auto gate_w = extract_expert_2d(lw.ffn_gate_exps());
                    auto up_w = extract_expert_2d(lw.ffn_up_exps());
                    auto down_w = extract_expert_2d(lw.ffn_down_exps());
                    if (gate_w && up_w && down_w) {
                        auto gate_t = ops::matmul_transB(token_hidden, gate_w);
                        auto up_t = ops::matmul_transB(token_hidden, up_w);
                        auto gated = ops::gelu_multiply(gate_t, up_t);
                        expert_result = ops::matmul_transB(gated, down_w);
                    }
                }

                if (expert_result) {
                    auto er_cpu = ensure_cpu(expert_result);
                    const float* er_data = static_cast<const float*>(er_cpu->data());
                    float* out_row = expert_out_data + s * cfg.hidden_dim;
                    for (int d = 0; d < cfg.hidden_dim; ++d) {
                        out_row[d] += weight * er_data[d];
                    }
                }
            }
        }
        expert_out = restore_device(expert_out, dev);
    }

    if (lw.ffn_post_norm_2()) {
        expert_out = ops::rms_norm(expert_out, lw.ffn_post_norm_2(), cfg.rms_norm_eps);
    }
    expert_out = restore_device(expert_out, dev);

    return ops::add(shared_out, expert_out);
}

}  // namespace forge
