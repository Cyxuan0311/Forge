#include "forge/engines/phimoe_engine.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "forge/inference/tensor_device_utils.h"
#include "forge/inference_batch.h"
#include "forge/logger.h"
#include "forge/operators.h"
#include "forge/perf_profiler.h"

#ifdef _OPENMP
#    include <omp.h>
#endif

#ifdef USE_CUDA
#    include <cuda_runtime.h>
#    include "forge/cuda_kernels.h"
#endif

namespace forge {

PhimoeEngine::PhimoeEngine(Model& model, InferenceContext& ctx)
    : TransformerEngine(model, ctx),
      attention_executor_(kv_cache_, kv_memory_ ? &kv_memory_->storage() : nullptr) {
    if (!init_weights()) {
        throw std::runtime_error("PhimoeEngine: failed to initialize weights");
    }
}

bool PhimoeEngine::init_weights() {
    return weights_.init(model_.weights(), model_.config());
}

// ----------------------------------------------------------------------------
// RMSNorm + bias helper
// ----------------------------------------------------------------------------

#ifdef USE_CUDA
static TensorPtr add_bias_cuda(const TensorPtr& normed, const TensorPtr& bias);
#endif

TensorPtr PhimoeEngine::rms_norm_with_bias(const TensorPtr& x, const TensorPtr& weight,
                                           const TensorPtr& bias, float eps) {
    auto normed = ops::rms_norm(x, weight, eps);
    if (!bias) return normed;

    // Add 1D bias broadcast over rows. CPU path is direct; CUDA adds on device.
    int M = static_cast<int>(normed->shape()[0]);
    int N = static_cast<int>(normed->shape()[1]);

    if (normed->device() == DeviceType::CPU) {
        float* data = static_cast<float*>(normed->data());
        const float* bias_data = static_cast<const float*>(bias->data());
        for (int m = 0; m < M; ++m) {
            float* row = data + m * N;
            for (int n = 0; n < N; ++n) {
                row[n] += bias_data[n];
            }
        }
        return normed;
    }

#ifdef USE_CUDA
    if (normed->device() == DeviceType::CUDA) {
        return add_bias_cuda(normed, bias);
    }
#endif

    // Fallback: copy to CPU, add, copy back
    auto cpu_normed = ensure_cpu(normed);
    auto cpu_bias = ensure_cpu(bias);
    float* data = static_cast<float*>(cpu_normed->data());
    const float* bias_data = static_cast<const float*>(cpu_bias->data());
    for (int m = 0; m < M; ++m) {
        float* row = data + m * N;
        for (int n = 0; n < N; ++n) {
            row[n] += bias_data[n];
        }
    }
    return restore_device(cpu_normed, normed->device());
}

// CUDA branch of the same helper: broadcast the 1D bias over rows entirely on
// device (no host copy). Bias lives on the layer device in CUDA mode; a CPU
// bias tensor is uploaded once if it happens to be host-resident.
#ifdef USE_CUDA
static TensorPtr add_bias_cuda(const TensorPtr& normed, const TensorPtr& bias) {
    int M = static_cast<int>(normed->shape()[0]);
    int N = static_cast<int>(normed->shape()[1]);

    TensorPtr bias_cuda;
    const float* bias_data = static_cast<const float*>(bias->data());
    if (bias->device() != DeviceType::CUDA) {
        bias_cuda = std::make_shared<Tensor>(bias->dtype(), bias->shape(), DeviceType::CUDA);
        bias_cuda->copy_from(*bias);
        bias_data = static_cast<const float*>(bias_cuda->data());
    }

    auto out = std::make_shared<Tensor>(DataType::FP32, normed->shape(), DeviceType::CUDA);
    const float* n_data = static_cast<const float*>(normed->data());
    float* o_data = static_cast<float*>(out->data());
    for (int m = 0; m < M; ++m) {
        cuda::launch_add_bias(n_data + m * N, bias_data, o_data + m * N, N);
    }
    return out;
}
#endif

// ----------------------------------------------------------------------------
// MoE FFN (CPU router path, SiLU-gated, top-k, no shared expert)
// ----------------------------------------------------------------------------

TensorPtr PhimoeEngine::moe_ffn_cpu(const TensorPtr& ffn_normed, const LayerWeights& lw,
                                    const ModelConfig& cfg, int seq_len, int layer_idx) {
    const int n_expert = cfg.n_expert;
    const int n_expert_used = cfg.n_expert_used > 0 ? cfg.n_expert_used : 1;
    const int hidden_dim = cfg.hidden_dim;

    // Router logits: [seq_len, n_expert]
    auto gate_inp = lw.ffn_gate_inp();
    auto router_logits = ops::matmul_transB(ffn_normed, gate_inp);
    auto logits_cpu = ensure_cpu(router_logits);
    auto ffn_in_cpu = ensure_cpu(ffn_normed);
    const float* logits_data = static_cast<const float*>(logits_cpu->data());
    const float* ffn_in_data = static_cast<const float*>(ffn_in_cpu->data());

    // Output accumulator
    auto expert_out = std::make_shared<Tensor>(DataType::FP32,
                                                std::vector<int64_t>{seq_len, hidden_dim},
                                                DeviceType::CPU);
    float* expert_out_data = static_cast<float*>(expert_out->data());
    std::fill_n(expert_out_data, (size_t)seq_len * hidden_dim, 0.0f);

    // Zero-copy expert extraction: phimoe stores expert axis as FIRST dim (axis 0).
    // gate_exps=[n_expert, n_ff, hidden], down_exps=[n_expert, hidden, n_ff].
    // After slicing axis 0 and viewing as 2D, the result is [rows, cols] which
    // matches matmul_transB's [N, K] convention.
    auto extract_expert_2d = [&](const TensorPtr& w3d, int expert_idx,
                                 ExpertSlot slot) -> TensorPtr {
        // P2: when per-expert paging is on, ExpertPageCache owns an independent
        // 2D copy of each materialised expert. Prefer it over slicing the
        // monolithic 3D tensor so the compute path consumes the paged weights.
        // A nullptr result just falls through to the 3D slice (e.g. expert not
        // materialised yet), so paging can never change results.
        if (expert_paging_enabled()) {
            auto paged = expert_weight(layer_idx, expert_idx, slot);
            if (paged) return paged;
        }
        if (!w3d) return nullptr;
        auto& shp = w3d->shape();
        if (shp.size() < 2) return w3d;
        if (shp.size() == 3 && expert_idx < shp[0]) {
            auto expert_slice = w3d->slice(0, expert_idx, expert_idx + 1);
            return std::make_shared<Tensor>(expert_slice.view({shp[1], shp[2]}));
        }
        return w3d;
    };

    // Cache CPU copies of expert weights to avoid repeated GPU→CPU transfers.
    // Weight tensors are constant during inference, so (data_ptr, expert_idx) is a
    // stable key. Without this cache, each decode step re-transfers ~7.5MB/layer
    // of expert weights from GPU to CPU (48960 transfers over 255 decode steps).
    static thread_local std::unordered_map<uintptr_t, TensorPtr> expert_w_cache;
    auto get_cpu_expert_weight = [&](const TensorPtr& w3d, int expert_idx,
                                     ExpertSlot slot) -> TensorPtr {
        // P2: paged weights are independent tensors already; only make sure they
        // sit on CPU for this path (no-op when the layer device is CPU).
        if (expert_paging_enabled()) {
            auto paged = expert_weight(layer_idx, expert_idx, slot);
            if (paged) {
                return paged->device() == DeviceType::CPU ? paged : ensure_cpu(paged);
            }
        }
        if (!w3d) return nullptr;
        if (w3d->device() == DeviceType::CPU)
            return extract_expert_2d(w3d, expert_idx, slot);
        auto key = reinterpret_cast<uintptr_t>(w3d->data()) ^ static_cast<uintptr_t>(expert_idx);
        auto it = expert_w_cache.find(key);
        if (it != expert_w_cache.end())
            return it->second;
        auto cpu_copy = ensure_cpu(extract_expert_2d(w3d, expert_idx, slot));
        expert_w_cache[key] = cpu_copy;
        return cpu_copy;
    };

    for (int s = 0; s < seq_len; ++s) {
        // Softmax over experts
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

        // Top-k selection
        std::vector<int> indices(n_expert);
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + n_expert_used, indices.end(),
                          [&](int a, int b) { return probs[a] > probs[b]; });

        float topk_sum = 0.0f;
        for (int k = 0; k < n_expert_used; ++k) {
            topk_sum += probs[indices[k]];
        }

        // Phase P0 hook: report routed (top-k) experts for this token so later
        // phases can page them onto the device. P0: empty sync_experts_resident.
        {
            std::vector<int> active(indices.begin(), indices.begin() + n_expert_used);
            sync_experts_resident(layer_idx, active);
        }

        // Extract token hidden as [1, hidden_dim]
        auto token_hidden = std::make_shared<Tensor>(DataType::FP32,
                                                       std::vector<int64_t>{1, hidden_dim},
                                                       DeviceType::CPU);
        std::memcpy(token_hidden->data(), ffn_in_data + s * hidden_dim,
                    hidden_dim * sizeof(float));

        float* out_row = expert_out_data + s * hidden_dim;

        // Collect top-k expert weights and routing weights up front so the
        // gate/up and down matmuls can be batched into single OpenMP regions
        // (reduces fork/join overhead and shares Q8_K input quantization).
        std::vector<TensorPtr> gate_w_k(n_expert_used), up_w_k(n_expert_used),
            down_w_k(n_expert_used);
        std::vector<float> route_w(n_expert_used);
        bool all_valid = true;
        for (int k = 0; k < n_expert_used; ++k) {
            int expert_idx = indices[k];
            gate_w_k[k] =
                get_cpu_expert_weight(lw.ffn_gate_exps(), expert_idx, ExpertSlot::Gate);
            up_w_k[k] = get_cpu_expert_weight(lw.ffn_up_exps(), expert_idx, ExpertSlot::Up);
            down_w_k[k] =
                get_cpu_expert_weight(lw.ffn_down_exps(), expert_idx, ExpertSlot::Down);
            route_w[k] = probs[expert_idx] / topk_sum;
            if (!gate_w_k[k] || !up_w_k[k] || !down_w_k[k]) {
                all_valid = false;
                break;
            }
        }

        if (!all_valid) {
            // Fallback: original sequential per-expert path (preserves
            // skip-on-null behaviour when an expert weight is unavailable).
            for (int k = 0; k < n_expert_used; ++k) {
                if (!gate_w_k[k] || !up_w_k[k] || !down_w_k[k]) continue;
                int expert_idx = indices[k];
                float weight = probs[expert_idx] / topk_sum;
                auto gate_t = ops::matmul_transB(token_hidden, gate_w_k[k]);
                auto up_t = ops::matmul_transB(token_hidden, up_w_k[k]);
                auto gated = ops::silu_multiply(gate_t, up_t);
                auto expert_result = ops::matmul_transB(gated, down_w_k[k]);
                if (expert_result) {
                    auto er_cpu = ensure_cpu(expert_result);
                    const float* er_data = static_cast<const float*>(er_cpu->data());
                    for (int d = 0; d < hidden_dim; ++d) {
                        out_row[d] += weight * er_data[d];
                    }
                }
            }
            continue;
        }

        // Batched gate+up: shared input [1, hidden] × {gate_k, up_k} ([n_ff, hidden])
        // -> [2*n_expert_used, n_ff]. Quantizes the input to Q8_K once and GEMVs
        // all weight rows in a single OpenMP region.
        std::vector<TensorPtr> gate_up_w;
        gate_up_w.reserve(2 * n_expert_used);
        for (int k = 0; k < n_expert_used; ++k) gate_up_w.push_back(gate_w_k[k]);
        for (int k = 0; k < n_expert_used; ++k) gate_up_w.push_back(up_w_k[k]);
        auto gate_up_out = ops::matmul_transB_shared_input(token_hidden, gate_up_w);

        const int n_ff = static_cast<int>(gate_w_k[0]->shape()[0]);
        const float* gu_data = static_cast<const float*>(gate_up_out->data());

        // gated_stack[k] = silu(gate_k) * up_k, stacked as [n_expert_used, n_ff].
        // SiLU(x) = x / (1 + exp(-x)), matching ops::silu_multiply.
        auto gated_stack = std::make_shared<Tensor>(
            DataType::FP32, std::vector<int64_t>{n_expert_used, n_ff}, DeviceType::CPU);
        float* gated_data = static_cast<float*>(gated_stack->data());
        for (int k = 0; k < n_expert_used; ++k) {
            const float* gate_row = gu_data + (size_t)k * n_ff;
            const float* up_row = gu_data + (size_t)(n_expert_used + k) * n_ff;
            float* out_row_k = gated_data + (size_t)k * n_ff;
            for (int i = 0; i < n_ff; ++i) {
                float g = gate_row[i];
                out_row_k[i] = g / (1.0f + std::exp(-g)) * up_row[i];
            }
        }

        // Batched down: [n_expert_used, n_ff] × {down_k} ([hidden, n_ff])
        // -> [n_expert_used, hidden_dim] in one OpenMP region.
        auto down_out = ops::matmul_transB_batched_pairs(gated_stack, down_w_k);
        const float* do_data = static_cast<const float*>(down_out->data());

        for (int k = 0; k < n_expert_used; ++k) {
            float w = route_w[k];
            const float* er_data = do_data + (size_t)k * hidden_dim;
            for (int d = 0; d < hidden_dim; ++d) {
                out_row[d] += w * er_data[d];
            }
        }
    }

    return expert_out;
}

#ifdef USE_CUDA
// ----------------------------------------------------------------------------
// MoE FFN, fully on GPU (decode/prefill). Reuses the device MoE router from
// cuda_moe.cu (softmax + top-k + renormalize) and drives each active expert
// through the per-dtype CUDA GEMV dispatch via zero-copy 2D weight views.
// No per-layer D2H of activations and no H2D restore.
// ----------------------------------------------------------------------------

TensorPtr PhimoeEngine::moe_ffn_cuda(const TensorPtr& ffn_normed, const LayerWeights& lw,
                                     const ModelConfig& cfg, int seq_len, int layer_idx) {
    const int n_expert = cfg.n_expert;
    const int n_expert_used = cfg.n_expert_used > 0 ? cfg.n_expert_used : 1;
    const int hidden_dim = cfg.hidden_dim;

    // Non-blocking error check: cudaGetLastError is sticky, so a single final
    // cudaDeviceSynchronize at the end of this function catches any launch error
    // without draining the pipeline after every kernel.
    auto check_err = [&](const char* tag) {
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess)
            throw std::runtime_error(std::string("moe_ffn_cuda[") + tag + "]: " +
                                     cudaGetErrorString(e));
    };

    // ---- Router logits on GPU: [seq_len, hidden] x [n_expert, hidden]^T ----
    auto router_logits = ops::matmul_transB(ffn_normed, lw.ffn_gate_inp());
    check_err("router_logits");

    // ---- Reusable scratch (allocated once, grown if the batch grows) ----
    const int n_ids = seq_len * n_expert_used;
    const int n_sm = seq_len * n_expert;
    if (!moe_router_indices_ || moe_router_indices_->numel() < n_ids) {
        moe_router_indices_ = std::make_shared<Tensor>(
            DataType::INT32, std::vector<int64_t>{n_ids}, DeviceType::CUDA);
        moe_router_weights_ = std::make_shared<Tensor>(
            DataType::FP32, std::vector<int64_t>{n_ids}, DeviceType::CUDA);
        moe_router_softmax_ = std::make_shared<Tensor>(
            DataType::FP32, std::vector<int64_t>{n_sm}, DeviceType::CUDA);
        moe_indices_h_ = std::make_shared<Tensor>(
            DataType::INT32, std::vector<int64_t>{n_ids}, DeviceType::CPU);
        moe_weights_h_ = std::make_shared<Tensor>(
            DataType::FP32, std::vector<int64_t>{n_ids}, DeviceType::CPU);
    }
    if (!moe_expert_out_ || moe_expert_out_->numel() < (size_t)seq_len * hidden_dim) {
        moe_expert_out_ = std::make_shared<Tensor>(
            DataType::FP32, std::vector<int64_t>{seq_len, hidden_dim}, DeviceType::CUDA);
    }

    // ---- Softmax + top-k + renormalize on device (moe_router_kernel) ----
    cuda::launch_moe_router(static_cast<const float*>(router_logits->data()),
                            static_cast<int*>(moe_router_indices_->data()),
                            static_cast<float*>(moe_router_weights_->data()),
                            static_cast<float*>(moe_router_softmax_->data()), n_expert,
                            n_expert_used, seq_len);
    check_err("router");

    const int* d_idx = static_cast<const int*>(moe_router_indices_->data());
    const float* d_w = static_cast<const float*>(moe_router_weights_->data());

    // ---- All-device grouped path (IQ2_S): no D2H, no host loop, ~4 launches ----
    auto gate3d = lw.ffn_gate_exps();
    auto up3d = lw.ffn_up_exps();
    auto down3d = lw.ffn_down_exps();
    // The grouped kernel indexes the monolithic 3D tensor directly, so it cannot
    // consume paged per-expert weights. When paging is on we must take the
    // per-expert fallback path (which does) — otherwise paging would only pay
    // transfer cost while the compute still reads the whole 3D tensor.
    const bool grouped_ok =
        gate3d && up3d && down3d && gate3d->device() == DeviceType::CUDA &&
        gate3d->dtype() == DataType::IQ2_S && up3d->dtype() == DataType::IQ2_S &&
        down3d->dtype() == DataType::IQ2_S && !expert_paging_enabled() &&
        std::getenv("FORGE_FORCE_MOE_FALLBACK") == nullptr;

    if (grouped_ok) {
        const int n_ff = static_cast<int>(gate3d->shape()[1]);
        const size_t n_gated = (size_t)n_ids * n_ff;
        if (!moe_gate_out_ || moe_gate_out_->numel() < n_gated) {
            std::vector<int64_t> shp{seq_len * n_expert_used, n_ff};
            moe_gate_out_ = std::make_shared<Tensor>(DataType::FP32, shp, DeviceType::CUDA);
            moe_up_out_ = std::make_shared<Tensor>(DataType::FP32, shp, DeviceType::CUDA);
            moe_gated_out_ = std::make_shared<Tensor>(DataType::FP32, shp, DeviceType::CUDA);
        }

        // gate/up: shared input quantized once, both projections per slot.
        cuda::launch_moe_expert_iq2_s_gateup(
            static_cast<const float*>(ffn_normed->data()), gate3d->data(), up3d->data(),
            static_cast<float*>(moe_gate_out_->data()),
            static_cast<float*>(moe_up_out_->data()), d_idx, hidden_dim, n_ff, n_expert,
            n_expert_used, seq_len);
        check_err("gateup");

        cuda::launch_silu_multiply(static_cast<const float*>(moe_gate_out_->data()),
                                   static_cast<const float*>(moe_up_out_->data()),
                                   static_cast<float*>(moe_gated_out_->data()),
                                   static_cast<int>(n_gated));
        check_err("silu");

        // down: per-token weighted accumulate in slot order (deterministic).
        cuda::launch_moe_expert_iq2_s_down(
            static_cast<const float*>(moe_gated_out_->data()), down3d->data(),
            static_cast<float*>(moe_expert_out_->data()), d_idx, d_w, n_ff, hidden_dim,
            n_expert, n_expert_used, seq_len);
        check_err("down");

        return moe_expert_out_;
    }

    // ---- Fallback: per-expert path (non-IQ2_S dtypes) ----
    // Tiny host mirror drives expert slicing; all heavy compute stays on device.
    moe_indices_h_->copy_from(*moe_router_indices_);
    moe_weights_h_->copy_from(*moe_router_weights_);
    const int* idx_data = static_cast<const int*>(moe_indices_h_->data());
    const float* w_data = static_cast<const float*>(moe_weights_h_->data());

    moe_expert_out_->zero_();
    check_err("zero_");
    float* out_data = static_cast<float*>(moe_expert_out_->data());

    // Zero-copy expert extraction: phimoe stores the expert axis as FIRST dim
    // (axis 0). The 2D view shares GPU storage, so matmul_transB dispatches to
    // the CUDA GEMV/MMQ kernels (on-the-fly dequant) directly.
    auto extract_expert_2d = [&](const TensorPtr& w3d, int expert_idx,
                                 ExpertSlot slot) -> TensorPtr {
        // P2: prefer the paged per-expert weights. Only use them when they are
        // actually resident on the compute device; an evicted (CPU) expert would
        // make the CUDA GEMV see a host tensor, so fall back to the 3D slice
        // (which stays fully on device) in that case.
        if (expert_paging_enabled()) {
            auto paged = expert_weight(layer_idx, expert_idx, slot);
            if (paged && paged->device() == DeviceType::CUDA) return paged;
        }
        if (!w3d) return nullptr;
        auto& shp = w3d->shape();
        if (shp.size() < 2) return w3d;
        if (shp.size() == 3 && expert_idx < shp[0]) {
            auto expert_slice = w3d->slice(0, expert_idx, expert_idx + 1);
            return std::make_shared<Tensor>(expert_slice.view({shp[1], shp[2]}));
        }
        return w3d;
    };

    for (int s = 0; s < seq_len; ++s) {
        auto token_hidden =
            std::make_shared<Tensor>(ffn_normed->slice(0, s, s + 1).view({1, hidden_dim}));

        // Phase P0 hook: report routed (top-k) experts for this token. The host
        // mirror idx_data already holds the router decisions (no extra D2H).
        // P0: empty sync_experts_resident.
        std::vector<int> active(idx_data + s * n_expert_used,
                                idx_data + (s + 1) * n_expert_used);
        sync_experts_resident(layer_idx, active);

        for (int k = 0; k < n_expert_used; ++k) {
            const int expert_idx = idx_data[s * n_expert_used + k];
            const float weight = w_data[s * n_expert_used + k];

            auto gate_w = extract_expert_2d(lw.ffn_gate_exps(), expert_idx, ExpertSlot::Gate);
            auto up_w = extract_expert_2d(lw.ffn_up_exps(), expert_idx, ExpertSlot::Up);
            auto down_w = extract_expert_2d(lw.ffn_down_exps(), expert_idx, ExpertSlot::Down);
            if (!gate_w || !up_w || !down_w) continue;

            auto gate = ops::matmul_transB(token_hidden, gate_w);  // [1, n_ff]
            check_err("gate gemv");
            auto up = ops::matmul_transB(token_hidden, up_w);      // [1, n_ff]
            check_err("up gemv");
            auto gated = ops::silu_multiply(gate, up);
            check_err("silu_multiply");
            auto down = ops::matmul_transB(gated, down_w);         // [1, hidden]
            check_err("down gemv");

            cuda::launch_scale_accumulate(static_cast<const float*>(down->data()),
                                          out_data + s * hidden_dim, weight, hidden_dim);
            check_err("scale_accumulate");

        }
    }

    // No cudaDeviceSynchronize here: kernels stay pipelined across layers; the
    // next op consuming expert_out is queued on the same stream, and launch
    // errors are surfaced by the sticky check_err above. Only a single flush
    // happens at the end of forward_request (logits read-back).
    return moe_expert_out_;
}
#endif  // USE_CUDA

// ----------------------------------------------------------------------------
// forward_layer
// ----------------------------------------------------------------------------

TensorPtr PhimoeEngine::forward_layer(const TensorPtr& hidden, const LayerExecutionContext& lctx) {
    const auto& cfg = lctx.config;
    const int layer_idx = lctx.layer_idx;
    const int seq_len = lctx.seq_len();
    const int64_t start_pos = lctx.start_pos();
    const DeviceType dev = lctx.device.type;
    const auto& lw = lctx.weights;

    bool has_qkv_bias = (lw.bq() && lw.bq()->numel() > 0);

    // ---- 1. Pre-attention norm (RMSNorm + bias) ----
    TensorPtr normed;
    {
        PERF_SCOPE("layer/attn_norm");
        normed = rms_norm_with_bias(hidden, lw.attn_norm(), lw.attn_norm_bias(), cfg.rms_norm_eps);
    }

    // ---- 2. QKV projection ----
    TensorPtr q, k, v;
    {
        PERF_SCOPE("layer/qkv_proj");
        auto qkv = AttentionExecutor::project_qkv(normed, lw, has_qkv_bias, dev, seq_len);
        q = qkv.q;
        k = qkv.k;
        v = qkv.v;
    }

    // ---- 3. RoPE (weights pre-permuted to half-split, so standard RoPE applies) ----
    TensorPtr q_rope, k_rope;
    {
        PERF_SCOPE("layer/rope");
        auto rope_result = rope_executor_.apply(q, k, cfg, start_pos, seq_len, dev);
        q_rope = rope_result.q_rope;
        k_rope = rope_result.k_rope;
    }

    // ---- 4. KV cache update ----
    {
        PERF_SCOPE("layer/kv_cache_update");

        bool paged = kv_memory_ && kv_memory_->is_paged();

        if (paged) {
            // CUDA-backed paged storage takes device pointers directly (no D2H);
            // CPU-backed paged storage needs the data on CPU.
            DeviceType storage_dev = kv_memory_->storage().device();
            const float* k_data;
            const float* v_data;
            TensorPtr k_cpu, v_cpu;
            if (storage_dev == DeviceType::CUDA) {
                k_data = static_cast<const float*>(k_rope->data());
                v_data = static_cast<const float*>(v->data());
            } else {
                k_cpu = k_rope;
                v_cpu = v;
                if (k_cpu->device() == DeviceType::CUDA) {
                    k_cpu = std::make_shared<Tensor>(k_cpu->dtype(), k_cpu->shape(), DeviceType::CPU);
                    k_cpu->copy_from(*k_rope);
                }
                if (v_cpu->device() == DeviceType::CUDA) {
                    v_cpu = std::make_shared<Tensor>(v_cpu->dtype(), v_cpu->shape(), DeviceType::CPU);
                    v_cpu->copy_from(*v);
                }
                k_data = static_cast<const float*>(k_cpu->data());
                v_data = static_cast<const float*>(v_cpu->data());
            }
            kv_memory_->storage().write_kv_seq(layer_idx, lctx.seq_id(), start_pos,
                                               seq_len, k_data, v_data);
        } else {
            kv_cache_.update(layer_idx, lctx.seq_id(), start_pos, k_rope, v, seq_len);

            const auto& kv_cfg = kv_cache_.kv_config();
            bool use_fused_decode =
                (dev == DeviceType::CUDA && seq_len == 1 &&
                 kv_cache_.d_q_key_cache(layer_idx) != nullptr &&
                 ((kv_cfg.type_k == KVCacheDType::Q4_0 && kv_cfg.type_v == KVCacheDType::Q4_0) ||
                  (kv_cfg.type_k == KVCacheDType::F16 && kv_cfg.type_v == KVCacheDType::F16) ||
                  (kv_cfg.type_k == KVCacheDType::Q8_0 && kv_cfg.type_v == KVCacheDType::Q8_0)));
            if (!use_fused_decode && kv_cache_.kv_dtype() != KVCacheDType::FP32) {
                kv_cache_.dequantize_layer(layer_idx);
            }
        }
    }

    // ---- 5. Attention ----
    TensorPtr attn_out;
    {
        PERF_SCOPE("layer/attention");
        attn_out = attention_executor_.attend(q_rope, cfg, layer_idx, seq_len, dev,
                                              nullptr, lctx.seq_id());
    }

    // ---- 6. Attention output projection + bias ----
    {
        PERF_SCOPE("layer/attn_proj");
        attn_out = ops::matmul_transB(attn_out, lw.wo(), lw.bo());
    }

    // ---- 7. Residual ----
    auto attn_residual = ops::add(hidden, attn_out);

    // ---- 8. FFN norm (RMSNorm + bias) ----
    TensorPtr ffn_normed;
    {
        PERF_SCOPE("layer/ffn_norm");
        ffn_normed =
            rms_norm_with_bias(attn_residual, lw.ffn_norm(), lw.ffn_norm_bias(), cfg.rms_norm_eps);
    }

    // ---- 9. MoE FFN ----
    TensorPtr ffn_out;
    {
        PERF_SCOPE("layer/moe_ffn");
#ifdef USE_CUDA
        if (dev == DeviceType::CUDA && ffn_normed->device() == DeviceType::CUDA) {
            ffn_out = moe_ffn_cuda(ffn_normed, lw, cfg, seq_len, lctx.layer_idx);
        } else
#endif
        {
            ffn_out = moe_ffn_cpu(ffn_normed, lw, cfg, seq_len, lctx.layer_idx);
            ffn_out = restore_device(ffn_out, dev);
        }
    }

    // ---- 10. Output residual ----
    return ops::add(attn_residual, ffn_out);
}

// ----------------------------------------------------------------------------
// forward_request (custom head with output_norm_bias + output_bias)
// ----------------------------------------------------------------------------

TensorPtr PhimoeEngine::forward_request(const ForwardRequest& req) {
    const auto& cfg = model_.config();

#ifdef _OPENMP
    omp_set_num_threads(ctx_.params().n_threads);
#endif

    init_kv_cache(cfg);

    DeviceType first_dev = layer_device(0);
    auto ids_on_dev = transfer_hidden(req.input_ids, first_dev);
    auto token_emb = model_.weights().get("token_embedding");

    TensorPtr hidden;
    {
        PERF_SCOPE("forward/embedding");
        hidden = ops::embedding(token_emb, ids_on_dev, weights_.token_embedding_fp32);
    }
    if (!hidden) return nullptr;

    // Layer loop (inline — cannot use base forward_layers because the base head
    // has no norm_bias / output_bias support)
    auto cur_hidden = hidden;
    for (int layer = 0; layer < cfg.num_layers; ++layer) {
        DeviceType layer_dev = layer_device(layer);
        cur_hidden = transfer_hidden(cur_hidden, layer_dev);
        {
            PERF_SCOPE_FMT("forward/layer_%d", layer);
            SET_PERF_CONTEXT(req.seq_id, "layer", layer,
                             layer_dev == DeviceType::CUDA ? "cuda" : "cpu", req.n_tokens);
            cur_hidden = forward_layer(cur_hidden, make_layer_context(layer, req, layer_dev));
        }
        if (!cur_hidden) return nullptr;
    }

    // Custom head: RMSNorm + output_norm_bias, then matmul + output_bias
    {
        PERF_SCOPE("forward/output_norm");
        cur_hidden = rms_norm_with_bias(cur_hidden, weights_.output_norm, weights_.output_norm_bias,
                                        cfg.rms_norm_eps);
    }

    auto output_weight = weights_.output_weight;
    if (!output_weight && cfg.tie_embeddings) {
        output_weight = weights_.token_embedding;
    }
    if (output_weight) {
        cur_hidden = transfer_hidden(cur_hidden, output_weight->device());
    }

    TensorPtr logits;
    {
        PERF_SCOPE("forward/output_proj");
        logits = ops::matmul_transB(cur_hidden, output_weight, weights_.output_bias);
    }

    return logits;
}

// ----------------------------------------------------------------------------
// forward_batch (per-sequence fallback)
// ----------------------------------------------------------------------------

TensorPtr PhimoeEngine::forward_batch(const InferenceBatch& batch) {
    if (batch.empty()) return nullptr;

    const auto& cfg = model_.config();
    init_kv_cache(cfg);

    int n_seq = batch.size();
    int vocab_size = -1;

    struct SeqLogits { int idx; std::vector<float> data; };
    std::vector<SeqLogits> all_logits;

    for (int i = 0; i < n_seq; i++) {
        const auto& item = batch.items[i];
        int seq_len = static_cast<int>(item.tokens.size());
        auto input_ids = std::make_shared<Tensor>(DataType::INT32,
                                                   std::vector<int64_t>{seq_len}, DeviceType::CPU);
        std::memcpy(input_ids->data(), item.tokens.data(), seq_len * sizeof(int32_t));

        auto logits = forward_request(
            ForwardRequest::from_ids(input_ids, item.start_pos, item.seq_id));
        if (!logits) continue;

        TensorPtr logits_cpu = ensure_cpu(logits);
        vocab_size = static_cast<int>(logits_cpu->shape().back());
        int seq_len_out = static_cast<int>(logits_cpu->shape()[0]);
        const float* src = static_cast<const float*>(logits_cpu->data()) +
                           (seq_len_out - 1) * vocab_size;

        SeqLogits sl;
        sl.idx = i;
        sl.data.assign(src, src + vocab_size);
        all_logits.push_back(std::move(sl));
    }

    if (all_logits.empty() || vocab_size <= 0) return nullptr;

    auto result = std::make_shared<Tensor>(DataType::FP32,
                                            std::vector<int64_t>{n_seq, vocab_size},
                                            DeviceType::CPU);
    float* dst = static_cast<float*>(result->data());
    std::memset(dst, 0, (size_t)n_seq * vocab_size * sizeof(float));

    for (auto& sl : all_logits) {
        std::memcpy(dst + sl.idx * vocab_size, sl.data.data(), vocab_size * sizeof(float));
    }

    return result;
}

}  // namespace forge
