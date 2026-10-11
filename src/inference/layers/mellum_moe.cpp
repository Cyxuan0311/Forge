#include "forge/inference/layers/mellum_moe.h"

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

#ifdef USE_CUDA
// 按 dtype 分派 slot 语义的融合 expert GEMV (Q4_0/Q4_K/Q5_0/Q6_K/Q8_0)。
// gate/up 一次算出两个投影 (共享同一份量化激活), 结果按 slot 逐 expert 存放;
// 激活后由 down 在 token 内按 slot 序加权累加, 完整语义 =
// Σ_k w_k · down_{e_k}(silu(gate_{e_k}(x)) ⊙ up_{e_k}(x))。
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
#endif

// 零拷贝 expert 抽取: 3D 权重沿 expert 轴切片再 view 成 2D。
// Forge shape 是反转后的 GGUF ne (gguf_model.cpp), mellum 的 expert 张量
// shape = [n_expert, OUT, IN], expert 轴 = axis 0; 切片 view 后天然得到
// matmul_transB 所需的 [N, K] 行主序 (K = 最后一维 = 输入宽度)。这里动态
// 定位 size == n_expert 的轴, 不依赖具体排布。
TensorPtr extract_expert_2d(const TensorPtr& w3d, int expert_idx, int n_expert) {
    if (!w3d)
        return nullptr;
    auto& shp = w3d->shape();
    if (shp.size() < 2) return w3d;
    if (shp.size() == 3) {
        int axis = 2;
        for (int a = 0; a < 3; ++a) {
            if (shp[a] == n_expert) { axis = a; break; }
        }
        if (expert_idx >= shp[axis]) return nullptr;
        auto expert_slice = w3d->slice(axis, expert_idx, expert_idx + 1);
        std::vector<int64_t> d2;
        for (int a = 0; a < 3; ++a) {
            if (a != axis) d2.push_back(expert_slice.shape()[a]);
        }
        return std::make_shared<Tensor>(expert_slice.view(d2));
    }
    return w3d;
}

}  // namespace

TensorPtr MellumMoe::apply(const TensorPtr& attn_residual, const LayerExecutionContext& lctx,
                           TransformerEngine* engine) {
    PERF_SCOPE("layer/moe");
    const auto& cfg = lctx.config;
    const auto& lw = lctx.weights;
    const int seq_len = lctx.seq_len();
    const DeviceType dev = lctx.device.type;
    const int n_expert = cfg.n_expert;
    const int n_expert_used = cfg.n_expert_used > 0 ? cfg.n_expert_used : 1;

    // ---- 输入 norm (mellum: ffn_norm 本身, 无 gemma4 的 router 缩放) ----
    auto normed = ops::rms_norm(attn_residual, lw.ffn_norm(), cfg.rms_norm_eps);
    auto router_logits = ops::matmul_transB(normed, lw.ffn_gate_inp());

#ifdef USE_CUDA
    // ---- 全 GPU fused 路径: launch_moe_router (softmax-all + top-k + topk 归一,
    // 与 mellum norm_weights 语义一致) + 融合 expert GEMV, 无 CPU 往返。
    if (router_logits->device() == DeviceType::CUDA && dev == DeviceType::CUDA) {
        auto router = run_moe_router_gpu(static_cast<const float*>(router_logits->data()),
                                         n_expert, n_expert_used, seq_len);
        auto indices_tensor = router.indices;
        auto weights_tensor = router.weights;

        // Q4_K_M 是混合精度策略: gate/up = Q4_K, down = Q5_0 或 Q8_0 (逐层)。
        // 三者都被 kernel selector 放行、且 expert 轴布局符合 kernel 假设才走
        // fused; 否则回落到下方逐 token 路径。FORGE_FORCE_MOE_FALLBACK 用于
        // 无需重编的 GPU/CPU 对照排查。
        // gate/up 宽度必须一致 (融合 buffer 布局假设), 否则不走 fused。
        const bool widths_equal = lw.ffn_gate_exps()->shape()[1] == lw.ffn_up_exps()->shape()[1];
        const bool fused_ok = moe_kernel_supported(lw.ffn_gate_exps()->dtype(), seq_len) &&
                              moe_kernel_supported(lw.ffn_up_exps()->dtype(), seq_len) &&
                              moe_kernel_supported(lw.ffn_down_exps()->dtype(), seq_len) &&
                              widths_equal &&
                              expert_axis_outermost(lw.ffn_gate_exps(), n_expert) &&
                              expert_axis_outermost(lw.ffn_up_exps(), n_expert) &&
                              expert_axis_outermost(lw.ffn_down_exps(), n_expert) &&
                              std::getenv("FORGE_FORCE_MOE_FALLBACK") == nullptr;

        if (fused_ok) {
            const int N_gate = static_cast<int>(lw.ffn_gate_exps()->shape()[1]);
            const int N_up = static_cast<int>(lw.ffn_up_exps()->shape()[1]);
            const int* d_indices = static_cast<const int*>(indices_tensor->data());
            const float* d_weights = static_cast<const float*>(weights_tensor->data());

            // 融合路径 = Σ_k w_k · down_{e_k}(silu(gate_{e_k}(x)) ⊙ up_{e_k}(x))。
            // gate/up 必须逐 slot 存放: 激活作用在跨 expert 的和上是错的
            // (launch_moe_expert_gemv 那类 atomicAdd 累加语义表达不了它)。
            const int n_slots = seq_len * n_expert_used;

            // 布局: [n_slots, N_gate + N_up], 前 N_gate 列 gate、后 N_up 列 up,
            // 与 Mellum 分离的 gate_exps / up_exps 两个张量对应。
            auto gate_up_buf = std::make_shared<Tensor>(
                DataType::FP32, std::vector<int64_t>{n_slots, N_gate + N_up}, DeviceType::CUDA);
            float* gu = static_cast<float*>(gate_up_buf->data());

            moe_gateup_slots(lw.ffn_gate_exps()->dtype(), static_cast<const float*>(normed->data()),
                             lw.ffn_gate_exps()->data(), lw.ffn_up_exps()->data(), gu,
                             gu + (size_t)n_slots * N_gate, d_indices, cfg.hidden_dim, N_gate,
                             n_expert, n_expert_used, seq_len);

            // silu(gate) ⊙ up, 逐 slot (element-wise, 原地安全)。
            auto gated = std::make_shared<Tensor>(
                DataType::FP32, std::vector<int64_t>{n_slots, N_gate}, DeviceType::CUDA);
            cuda::launch_silu_multiply(gu, gu + (size_t)n_slots * N_gate,
                                       static_cast<float*>(gated->data()), n_slots * N_gate);

            auto expert_out = std::make_shared<Tensor>(
                DataType::FP32, std::vector<int64_t>{seq_len, cfg.hidden_dim}, DeviceType::CUDA);
            moe_down_slots(lw.ffn_down_exps()->dtype(), static_cast<const float*>(gated->data()),
                           lw.ffn_down_exps()->data(), static_cast<float*>(expert_out->data()),
                           d_indices, d_weights, N_gate, cfg.hidden_dim, n_expert, n_expert_used,
                           seq_len);

            // P0 hook: router top-k 上报 (expert paging)
            if (engine) {
                auto idx_h = ensure_cpu(indices_tensor);
                const int* id = static_cast<const int*>(idx_h->data());
                std::vector<int> active(id, id + seq_len * n_expert_used);
                engine->sync_experts_resident(lctx.layer_idx, active);
            }
            return expert_out;
        }

        // unsupported dtype: router 结果拷回 CPU, 走下方逐 token 路径
        auto idx_cpu = ensure_cpu(indices_tensor);
        auto w_cpu = ensure_cpu(weights_tensor);
        const int* id = static_cast<const int*>(idx_cpu->data());
        const float* wd = static_cast<const float*>(w_cpu->data());
        std::vector<int> out_indices(id, id + seq_len * n_expert_used);
        std::vector<float> out_weights(wd, wd + seq_len * n_expert_used);
        if (engine) {
            std::vector<int> active(id, id + seq_len * n_expert_used);
            engine->sync_experts_resident(lctx.layer_idx, active);
        }

        // ---- 逐 token × 逐 top-k expert (GPU 权重, CPU 累加; 正确性优先) ----
        auto expert_out = std::make_shared<Tensor>(DataType::FP32,
                                                   std::vector<int64_t>{seq_len, cfg.hidden_dim},
                                                   DeviceType::CPU);
        std::fill_n(static_cast<float*>(expert_out->data()), seq_len * cfg.hidden_dim, 0.0f);
        float* expert_out_data = static_cast<float*>(expert_out->data());

        for (int s = 0; s < seq_len; ++s) {
            const int* idx = out_indices.data() + s * n_expert_used;
            const float* wgt = out_weights.data() + s * n_expert_used;
            auto token_hidden = std::make_shared<Tensor>(normed->slice(0, s, s + 1));
            float* out_row = expert_out_data + s * cfg.hidden_dim;
            for (int k = 0; k < n_expert_used; ++k) {
                const int e = idx[k];
                auto gate_w = extract_expert_2d(lw.ffn_gate_exps(), e, n_expert);
                auto up_w = extract_expert_2d(lw.ffn_up_exps(), e, n_expert);
                auto down_w = extract_expert_2d(lw.ffn_down_exps(), e, n_expert);
                auto gate_t = ops::matmul_transB(token_hidden, gate_w);
                auto up_t = ops::matmul_transB(token_hidden, up_w);
                auto gated = ops::silu_multiply(gate_t, up_t);
                auto expert_result = ops::matmul_transB(gated, down_w);
                auto er_cpu = ensure_cpu(expert_result);
                const float* er_data = static_cast<const float*>(er_cpu->data());
                for (int d = 0; d < cfg.hidden_dim; ++d)
                    out_row[d] += wgt[k] * er_data[d];
            }
        }
        return restore_device(expert_out, dev);
    }
#endif

    // ---- CPU 路径: 逐 token softmax + top-k + 逐 expert 计算 ----
    auto router_logits_cpu = ensure_cpu(router_logits);
    auto normed_cpu = ensure_cpu(normed);
    const float* logits_data = static_cast<const float*>(router_logits_cpu->data());

    auto expert_out = std::make_shared<Tensor>(DataType::FP32,
                                               std::vector<int64_t>{seq_len, cfg.hidden_dim},
                                               DeviceType::CPU);
    std::fill_n(static_cast<float*>(expert_out->data()), seq_len * cfg.hidden_dim, 0.0f);
    float* expert_out_data = static_cast<float*>(expert_out->data());

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
        for (int e = 0; e < n_expert; ++e) probs[e] /= sum_exp;

        std::vector<int> indices(n_expert);
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + n_expert_used, indices.end(),
                          [&](int a, int b) { return probs[a] > probs[b]; });

        float topk_sum = 0.0f;
        for (int k = 0; k < n_expert_used; ++k) topk_sum += probs[indices[k]];

        if (engine && n_expert_used > 0) {
            std::vector<int> active(indices.begin(), indices.begin() + n_expert_used);
            engine->sync_experts_resident(lctx.layer_idx, active);
        }

        const float* token_in =
            static_cast<const float*>(normed_cpu->data()) + s * cfg.hidden_dim;
        float* out_row = expert_out_data + s * cfg.hidden_dim;

        for (int k = 0; k < n_expert_used; ++k) {
            const int expert_idx = indices[k];
            const float weight = probs[expert_idx] / topk_sum;

            auto token_hidden = std::make_shared<Tensor>(
                DataType::FP32, std::vector<int64_t>{1, cfg.hidden_dim}, DeviceType::CPU);
            std::memcpy(token_hidden->data(), token_in, cfg.hidden_dim * sizeof(float));

            auto gate_w = extract_expert_2d(lw.ffn_gate_exps(), expert_idx, n_expert);
            auto up_w = extract_expert_2d(lw.ffn_up_exps(), expert_idx, n_expert);
            auto down_w = extract_expert_2d(lw.ffn_down_exps(), expert_idx, n_expert);

            auto gate_t = ops::matmul_transB(token_hidden, gate_w);
            auto up_t = ops::matmul_transB(token_hidden, up_w);
            auto gated = ops::silu_multiply(gate_t, up_t);
            auto expert_result = ops::matmul_transB(gated, down_w);

            auto er_cpu = ensure_cpu(expert_result);
            const float* er_data = static_cast<const float*>(er_cpu->data());
            for (int d = 0; d < cfg.hidden_dim; ++d)
                out_row[d] += weight * er_data[d];
        }
    }

    return restore_device(expert_out, dev);
}

}  // namespace forge
