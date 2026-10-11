#include "forge/inference/layers/gemma4_moe.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#include "forge/cuda_kernels.h"
#include "forge/engines/transformer_engine.h"
#include "forge/inference/layers/moe_fused_common.h"
#include "forge/inference/tensor_device_utils.h"
#include "forge/kernel_selector.h"
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
// 按 dtype 分派 slot 语义的融合 expert GEMV (Q4_0/Q4_K/Q6_K/Q8_0)。
// gate/up 结果按 slot 逐 expert 存放, 激活后由 down 在 token 内按 slot 序加权
// 累加, 完整语义 = Σ_k w_k · down_{e_k}(act(gate_{e_k}(x)) ⊙ up_{e_k}(x))。
void moe_gateup_slots(DataType dt, const float* x, const void* w_gate, const void* w_up,
                      float* o_gate, float* o_up, const int* d_indices, int K, int N, int n_expert,
                      int n_expert_used, int seq_len) {
    KernelProblem problem;
    problem.domain = KernelDomain::MoE;
    problem.dtype = dt;
    problem.M = seq_len;
    problem.N = N;
    problem.K = K;
    if (select_kernel(problem) == KernelVariant::UNSUPPORTED)
        return;
    switch (dt) {
    case DataType::Q4_0:
        cuda::launch_moe_expert_gateup_slots<DataType::Q4_0>(
            x, w_gate, w_up, o_gate, o_up, d_indices, K, N, n_expert, n_expert_used, seq_len, 0);
        break;
    case DataType::Q4_K:
        cuda::launch_moe_expert_gateup_slots<DataType::Q4_K>(
            x, w_gate, w_up, o_gate, o_up, d_indices, K, N, n_expert, n_expert_used, seq_len, 0);
        break;
    case DataType::Q5_0:
        cuda::launch_moe_expert_gateup_slots<DataType::Q5_0>(
            x, w_gate, w_up, o_gate, o_up, d_indices, K, N, n_expert, n_expert_used, seq_len, 0);
        break;
    case DataType::Q6_K:
        cuda::launch_moe_expert_gateup_slots<DataType::Q6_K>(
            x, w_gate, w_up, o_gate, o_up, d_indices, K, N, n_expert, n_expert_used, seq_len, 0);
        break;
    case DataType::Q8_0:
        cuda::launch_moe_expert_gateup_slots<DataType::Q8_0>(
            x, w_gate, w_up, o_gate, o_up, d_indices, K, N, n_expert, n_expert_used, seq_len, 0);
        break;
    default:
        break;
    }
}

void moe_down_slots(DataType dt, const float* x, const void* w, float* out, const int* d_indices,
                    const float* d_weights, int K, int N, int n_expert, int n_expert_used,
                    int seq_len) {
    KernelProblem problem;
    problem.domain = KernelDomain::MoE;
    problem.dtype = dt;
    problem.M = seq_len;
    problem.N = N;
    problem.K = K;
    if (select_kernel(problem) == KernelVariant::UNSUPPORTED)
        return;
    switch (dt) {
    case DataType::Q4_0:
        cuda::launch_moe_expert_down_slots<DataType::Q4_0>(x, w, out, d_indices, d_weights, K, N,
                                                           n_expert, n_expert_used, seq_len, 0);
        break;
    case DataType::Q4_K:
        cuda::launch_moe_expert_down_slots<DataType::Q4_K>(x, w, out, d_indices, d_weights, K, N,
                                                           n_expert, n_expert_used, seq_len, 0);
        break;
    case DataType::Q5_0:
        cuda::launch_moe_expert_down_slots<DataType::Q5_0>(x, w, out, d_indices, d_weights, K, N,
                                                           n_expert, n_expert_used, seq_len, 0);
        break;
    case DataType::Q6_K:
        cuda::launch_moe_expert_down_slots<DataType::Q6_K>(x, w, out, d_indices, d_weights, K, N,
                                                           n_expert, n_expert_used, seq_len, 0);
        break;
    case DataType::Q8_0:
        cuda::launch_moe_expert_down_slots<DataType::Q8_0>(x, w, out, d_indices, d_weights, K, N,
                                                           n_expert, n_expert_used, seq_len, 0);
        break;
    default:
        break;
    }
}

// 量化权重一行的字节数 (= K 方向块数 × 块大小), 用于在融合 gate_up 张量里
// 定位 up 半段的基址。
template <DataType DT>
size_t expert_row_bytes_t(int K) {
    return (size_t)((K + QuantTraits<DT>::block_elements - 1) / QuantTraits<DT>::block_elements) *
           QuantTraits<DT>::block_size;
}

size_t expert_row_bytes(DataType dt, int K) {
    switch (dt) {
    case DataType::Q4_0:
        return expert_row_bytes_t<DataType::Q4_0>(K);
    case DataType::Q8_0:
        return expert_row_bytes_t<DataType::Q8_0>(K);
    case DataType::Q5_0:
        return expert_row_bytes_t<DataType::Q5_0>(K);
    case DataType::Q4_K:
        return expert_row_bytes_t<DataType::Q4_K>(K);
    case DataType::Q6_K:
        return expert_row_bytes_t<DataType::Q6_K>(K);
    default:
        return 0;
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

#ifdef USE_CUDA
    // 融合路径的前提: dtype 有可用 kernel, 且 expert 轴布局符合 slot kernel 的
    // 字节布局假设 (expert 最外层)。任一条不满足都走 CPU 路径 —— 旧实现在这里
    // 只是"什么都不算", 会把全零的 expert_out 当结果返回。
    const bool fused_ok =
        lw.ffn_down_exps() && moe_kernel_supported(lw.ffn_down_exps()->dtype(), seq_len) &&
        ((lw.ffn_gate_up_exps() && is_quantized_type(lw.ffn_gate_up_exps()->dtype()) &&
          moe_kernel_supported(lw.ffn_gate_up_exps()->dtype(), seq_len) &&
          expert_axis_outermost(lw.ffn_gate_up_exps(), n_expert) &&
          expert_axis_outermost(lw.ffn_down_exps(), n_expert)) ||
         (lw.ffn_gate_exps() && lw.ffn_up_exps() &&
          moe_kernel_supported(lw.ffn_gate_exps()->dtype(), seq_len) &&
          moe_kernel_supported(lw.ffn_up_exps()->dtype(), seq_len) &&
          expert_axis_outermost(lw.ffn_gate_exps(), n_expert) &&
          expert_axis_outermost(lw.ffn_up_exps(), n_expert) &&
          expert_axis_outermost(lw.ffn_down_exps(), n_expert)));
    const bool gpu_fused = fused_ok && std::getenv("FORGE_FORCE_MOE_FALLBACK") == nullptr;
#else
    const bool gpu_fused = false;
#endif

    if (gpu_fused && router_logits->device() == DeviceType::CUDA && dev == DeviceType::CUDA) {
#ifdef USE_CUDA
        // 全 GPU 路径, 无 CPU 往返。
        auto router = run_moe_router_gpu(static_cast<const float*>(router_logits->data()),
                                         n_expert, n_expert_used, seq_len);
        auto indices_tensor = router.indices;
        auto weights_tensor = router.weights;

        const int* d_indices = static_cast<const int*>(indices_tensor->data());
        const float* d_weights = static_cast<const float*>(weights_tensor->data());
        // slot = t*n_expert_used + k: gate/up 逐 expert 存放 → 逐 expert 激活 →
        // down 加权累加。语义 = Σ_k w_k·down_{e_k}(gelu(gate_{e_k})⊙up_{e_k})。
        // 路由权重只在 down 步乘一次: 旧路径 gate/up/down 三次都乘了 w, 且激活
        // 作用在跨 expert 的和上, n_expert_used > 1 时结果错误。
        const int n_slots = seq_len * n_expert_used;
        const int K = cfg.hidden_dim;
        const int n_ff = lw.ffn_gate_up_exps()
                             ? static_cast<int>(lw.ffn_gate_up_exps()->shape()[1]) / 2
                             : static_cast<int>(lw.ffn_gate_exps()->shape()[1]);

        auto gate_buf = std::make_shared<Tensor>(
            DataType::FP32, std::vector<int64_t>{n_slots, n_ff}, DeviceType::CUDA);
        auto up_buf = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{n_slots, n_ff},
                                               DeviceType::CUDA);

        if (lw.ffn_gate_up_exps()) {
            // 融合 gate_up 张量: 每个 expert 是 [gate(n_ff); up(n_ff)] 两段行。
            // gateup 融合 kernel 收两个基址, 传 (tensor, tensor + up 半段偏移)
            // 即可分别算两段, 不需要额外的 split kernel。
            const DataType gu_dtype = lw.ffn_gate_up_exps()->dtype();
            const uint8_t* gu_base = static_cast<const uint8_t*>(lw.ffn_gate_up_exps()->data());
            const size_t up_row_offset = expert_row_bytes(gu_dtype, K) * (size_t)n_ff;
            moe_gateup_slots(gu_dtype, static_cast<const float*>(cur_moe->data()), gu_base,
                             gu_base + up_row_offset, static_cast<float*>(gate_buf->data()),
                             static_cast<float*>(up_buf->data()), d_indices, K, n_ff, n_expert,
                             n_expert_used, seq_len);
        } else {
            // 分离的 gate + up
            moe_gateup_slots(lw.ffn_gate_exps()->dtype(),
                             static_cast<const float*>(cur_moe->data()), lw.ffn_gate_exps()->data(),
                             lw.ffn_up_exps()->data(), static_cast<float*>(gate_buf->data()),
                             static_cast<float*>(up_buf->data()), d_indices, K, n_ff, n_expert,
                             n_expert_used, seq_len);
        }

        auto gated = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{n_slots, n_ff},
                                              DeviceType::CUDA);
        // exact-erf gelu, 与 CPU 路径的 ops::gelu_multiply 一致 (旧融合路径用的
        // 是 gelu_tanh 近似, 同样与 CPU 路径不一致)。
        cuda::launch_gelu_multiply(static_cast<const float*>(gate_buf->data()),
                                   static_cast<const float*>(up_buf->data()),
                                   static_cast<float*>(gated->data()), n_slots * n_ff);

        moe_down_slots(lw.ffn_down_exps()->dtype(), static_cast<const float*>(gated->data()),
                       lw.ffn_down_exps()->data(), static_cast<float*>(expert_out->data()),
                       d_indices, d_weights, n_ff, cfg.hidden_dim, n_expert, n_expert_used,
                       seq_len);

        // Router decisions live on device; mirror the top-k indices to host and
        // report them (router statistics / expert paging hook).
        if (engine) {
            auto idx_h = ensure_cpu(indices_tensor);
            const int* id = static_cast<const int*>(idx_h->data());
            std::vector<int> active(id, id + seq_len * n_expert_used);
            engine->sync_experts_resident(lctx.layer_idx, active);
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
                // expert 轴动态定位 (按 size == n_expert), 不硬编码 axis 2 ——
                // 不同的 gemma 变体把 expert 轴放在不同位置。
                auto extract_expert_2d = [&](const TensorPtr& w3d) -> TensorPtr {
                    if (!w3d) return nullptr;
                    auto& shp = w3d->shape();
                    if (shp.size() < 2) return w3d;
                    if (shp.size() == 3) {
                        int axis = -1;
                        for (int a = 0; a < 3; ++a) {
                            if (shp[a] == n_expert) { axis = a; break; }
                        }
                        if (axis < 0 || expert_idx >= shp[axis]) return w3d;
                        auto expert_slice = w3d->slice(axis, expert_idx, expert_idx + 1);
                        std::vector<int64_t> d2;
                        for (int a = 0; a < 3; ++a) {
                            if (a != axis) d2.push_back(expert_slice.shape()[a]);
                        }
                        return std::make_shared<Tensor>(expert_slice.view(d2));
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
