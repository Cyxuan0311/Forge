#include "forge/engines/transformer_engine.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "forge/backend.h"
#include "forge/backend_scheduler.h"
#include "forge/context.h"
#include "forge/cuda_kernels.h"
#include "forge/inference_batch.h"
#include "forge/kv_memory.h"
#include "forge/logger.h"
#include "forge/operators.h"
#include "forge/perf_profiler.h"

#include "cpu/simd.h"

#ifdef USE_CUDA
#    include <cuda_runtime.h>
#endif

#ifdef _OPENMP
#    include <omp.h>
#endif

namespace forge {

TransformerEngine::TransformerEngine(Model& model, InferenceContext& ctx)
    : model_(model),
      ctx_(ctx),
      plan_(build_execution_plan(model.config().arch_type, model.config())),
      memory_(ctx.memory()),
      kv_cache_(*ctx.memory().kv()),
      kv_memory_(std::make_unique<KVMemory>(*ctx.memory().kv(), ctx.params().kv_storage_mode)) {
    LOG_INFO("Execution plan: " + plan_.plan_id());
    graph_runtime_.set_builder(plan_.graph_builder);
}

void TransformerEngine::reset() {
    kv_cache_.reset();
    if (kv_memory_) {
        kv_memory_->storage().reset();
    }
    layer_hiddens_.clear();
    set_kv_cache_initialized(false);
    graph_runtime_.invalidate();
}

TensorPtr TransformerEngine::take_layer_hiddens(const std::vector<int>& layers) {
    if (layer_hiddens_.empty()) {
        return nullptr;
    }
    // Gather the requested layers, copying each to CPU so we can concatenate
    // them row-wise on the host. copy_from() is a synchronous d2h copy for
    // CUDA->CPU, so the source tensors are left untouched.
    std::vector<TensorPtr> selected;
    selected.reserve(layers.size());
    for (int li : layers) {
        if (li < 0 || li >= static_cast<int>(layer_hiddens_.size()) || !layer_hiddens_[li]) {
            LOG_WARN("take_layer_hiddens: layer " + std::to_string(li) + " unavailable");
            continue;
        }
        const TensorPtr& t = layer_hiddens_[li];
        auto cpu = std::make_shared<Tensor>(t->dtype(), t->shape(), DeviceType::CPU);
        cpu->copy_from(*t);
        selected.push_back(std::move(cpu));
    }
    if (selected.empty()) {
        return nullptr;
    }
    const int64_t seq = selected[0]->shape()[0];
    int64_t total_hidden = 0;
    for (auto& t : selected) {
        total_hidden += t->shape()[1];
    }
    auto out = std::make_shared<Tensor>(selected[0]->dtype(),
                                        std::vector<int64_t>{seq, total_hidden}, DeviceType::CPU);
    const size_t elem = out->nbytes() / out->numel();  // bytes per element
    uint8_t* dst = static_cast<uint8_t*>(out->data());
    int64_t col = 0;
    for (auto& t : selected) {
        const uint8_t* src = static_cast<const uint8_t*>(t->data());
        const int64_t H = t->shape()[1];
        const size_t row_bytes = static_cast<size_t>(H) * elem;
        for (int64_t r = 0; r < seq; ++r) {
            std::memcpy(dst + (r * total_hidden + col) * elem, src + r * row_bytes, row_bytes);
        }
        col += H;
    }
    return out;
}

void TransformerEngine::set_gpu_layers(int gpu_layers) {
    // Default: all layers on GPU 0
    std::vector<int> per_dev = {gpu_layers};
    set_gpu_layers(gpu_layers, per_dev);
}

void TransformerEngine::set_gpu_layers(int gpu_layers, const std::vector<int>& gpu_layers_per_dev) {
    graph_runtime_.invalidate();
    gpu_layers_ = gpu_layers;
    const auto& cfg = model_.config();
    int num_layers = cfg.num_layers;

    // Build per-layer DeviceTarget vector from per-GPU layer counts.
    // gpu_layers_per_dev[0] = layers on GPU 0, gpu_layers_per_dev[1] = layers on GPU 1, etc.
    // If gpu_layers_per_dev is empty, fall back to single-GPU behavior (all on GPU 0).
    layer_devices_.resize(num_layers);
    if (gpu_layers_per_dev.empty()) {
        for (int i = 0; i < num_layers; ++i) {
            layer_devices_[i] = (gpu_layers_ < 0) ? DeviceTarget::cuda(0)
                               : (i < gpu_layers_)  ? DeviceTarget::cuda(0)
                                                    : DeviceTarget::cpu();
        }
    } else {
        int offset = 0;
        for (size_t dev_id = 0; dev_id < gpu_layers_per_dev.size() && offset < num_layers; ++dev_id) {
            int n = gpu_layers_per_dev[dev_id];
            for (int i = 0; i < n && offset < num_layers; ++i, ++offset) {
                layer_devices_[offset] = n > 0 ? DeviceTarget::cuda(static_cast<int>(dev_id))
                                                : DeviceTarget::cpu();
            }
        }
        // Remaining layers (if any) go to CPU
        for (int i = offset; i < num_layers; ++i) {
            layer_devices_[i] = DeviceTarget::cpu();
        }
        // If gpu_layers < 0 (auto), mark all remaining as GPU 0
        if (gpu_layers_ < 0) {
            for (int i = offset; i < num_layers; ++i) {
                layer_devices_[i] = DeviceTarget::cuda(0);
            }
        }
    }

    // Phase P0: expose per-(layer, expert) placement. MoE layers get an
    // [n_expert] device vector seeded from the layer device; set_expert_placement()
    // can later override individual experts. Default = inherits layer device,
    // so current behavior is unchanged.
    if (!expert_devices_.empty()) expert_devices_.clear();
    expert_devices_.resize(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        auto gate_exps = weights_.layers[i].ffn_gate_exps();
        if (gate_exps && gate_exps->shape().size() == 3) {
            const int n_expert = static_cast<int>(gate_exps->shape()[2]);
            if (n_expert > 0) expert_devices_[i].assign(n_expert, layer_devices_[i]);
        }
    }

    // Validation mode: weights are already placed during loading.
    DeviceType first_dev = layer_device(0);
    auto token_emb = model_.weights().get("token_embedding");
    if (token_emb && token_emb->device() != first_dev) {
        if (!ctx_.params().offload_embedding && first_dev == DeviceType::CUDA &&
            token_emb->device() == DeviceType::CPU) {
            LOG_INFO("set_gpu_layers: token_embedding kept on CPU (offload_embedding=false)");
        } else {
            LOG_WARN("set_gpu_layers: token_embedding is on " +
                     std::to_string(static_cast<int>(token_emb->device())) +
                     " but expected " + std::to_string(static_cast<int>(first_dev)) +
                     ". Moving to target device (legacy loading path).");
            token_emb->to_device(first_dev);
        }
    }

    // Validate + repair layer weights
    for (int i = 0; i < num_layers; ++i) {
        DeviceType expected_type = layer_devices_[i].type;
        auto& lw = weights_.layers[i];
        for (auto& [name, tensor] : lw.weights) {
            if (!tensor || tensor->device() == expected_type) continue;
            // P2: with expert paging enabled, deliberately keep the monolithic
            // 3D expert tensors on the host. Only individually paged experts are
            // uploaded (by ensure_resident), so expert VRAM tracks the routed
            // working set instead of the whole expert pool. Without this the 3D
            // tensor would occupy VRAM in full and paging could never save any.
            const bool is_expert_3d =
                tensor->shape().size() == 3 &&
                (name == "ffn_gate_exps" || name == "ffn_up_exps" ||
                 name == "ffn_down_exps" || name == "ffn_gate_up_exps");
            if (expert_paging_enabled_ && is_expert_3d) continue;

            LOG_WARN("set_gpu_layers: layer " + std::to_string(i) + " weight '" + name +
                     "' is on device " + std::to_string(static_cast<int>(tensor->device())) +
                     " but expected " + std::to_string(static_cast<int>(expected_type)) +
                     ". Moving to target device (legacy loading path).");
            tensor->to_device(expected_type);
        }
    }

    DeviceType last_dev = layer_devices_[num_layers - 1].type;
    auto out_w = weights_.output_weight;
    if (out_w && out_w->device() != last_dev) {
        LOG_WARN("set_gpu_layers: output_weight is on " +
                 std::to_string(static_cast<int>(out_w->device())) +
                 " but expected " + std::to_string(static_cast<int>(last_dev)) +
                 ". Moving to target device (legacy loading path).");
    }

    weights_.move_output_weights(last_dev);

    // Update KV cache per-layer devices (if already initialized)
    if (kv_cache_initialized()) {
        kv_cache_.set_layer_devices(layer_devices_);
    }

    int num_cuda = 0;
    for (auto d : layer_devices_) if (d.is_cuda()) ++num_cuda;
    LOG_INFO("CPU offload configured: gpu_layers=" + std::to_string(gpu_layers) + "/" +
             std::to_string(num_layers) + ", CUDA layers=" + std::to_string(num_cuda));

    // Phase P1: size the ExpertPageCache for router statistics. MoE layers have
    // 3D expert tensors; dense layers contribute 0 experts and are skipped.
    // The authoritative expert count comes from ModelConfig (tensor expert dim
    // varies by architecture: dim 0 for PhiMoE, dim 2 for Gemma4), so infer the
    // per-layer count from cfg.n_expert rather than the tensor shape directly.
    {
        const ModelConfig& mcfg = model_.config();
        const int cfg_n_expert = mcfg.n_expert > 0 ? mcfg.n_expert : 0;
        std::vector<int> experts_per_layer(num_layers, 0);
        for (int i = 0; i < num_layers; ++i) {
            auto ge = weights_.layers[i].ffn_gate_exps();
            auto geu = weights_.layers[i].ffn_gate_up_exps();
            const bool is_moe =
                (ge && ge->shape().size() == 3) || (geu && geu->shape().size() == 3);
            if (is_moe) experts_per_layer[i] = cfg_n_expert;
        }
        expert_page_cache_.resize(num_layers, experts_per_layer);
    }
}

DeviceType TransformerEngine::layer_device(int layer_idx) const {
    if (!layer_devices_.empty() && layer_idx >= 0 && layer_idx < static_cast<int>(layer_devices_.size()))
        return layer_devices_[layer_idx].type;
    if (gpu_layers_ < 0)
        return model_.device();
    if (layer_idx < gpu_layers_)
        return DeviceType::CUDA;
    return DeviceType::CPU;
}

DeviceTarget TransformerEngine::layer_device_target(int layer_idx) const {
    if (!layer_devices_.empty() && layer_idx >= 0 && layer_idx < static_cast<int>(layer_devices_.size()))
        return layer_devices_[layer_idx];
    if (gpu_layers_ < 0)
        return DeviceTarget::cuda(0);
    if (layer_idx < gpu_layers_)
        return DeviceTarget::cuda(0);
    return DeviceTarget::cpu();
}

// ---- Expert-level placement (MoE partial activation) ----

void TransformerEngine::set_expert_placement(int layer, const std::vector<int>& gpu_experts,
                                              int n_expert) {
    if (layer < 0 || layer >= static_cast<int>(weights_.layers.size()))
        return;
    if (expert_devices_.size() < weights_.layers.size())
        expert_devices_.resize(weights_.layers.size());
    auto& vec = expert_devices_[layer];
    vec.assign(n_expert, layer_device_target(
                              layer));  // default: inherit layer device
    for (int e : gpu_experts) {
        if (e >= 0 && e < n_expert) vec[e] = DeviceTarget::cuda(0);
    }
    LOG_INFO("set_expert_placement: layer " + std::to_string(layer) + " -> " +
             std::to_string(gpu_experts.size()) + "/" + std::to_string(n_expert) +
             " experts on GPU");
}

DeviceTarget TransformerEngine::expert_device(int layer, int expert) const {
    if (layer >= 0 && layer < static_cast<int>(expert_devices_.size())) {
        const auto& vec = expert_devices_[layer];
        if (expert >= 0 && expert < static_cast<int>(vec.size()))
            return vec[expert];
    }
    return layer_device_target(layer);  // fallback: whole-layer device
}

bool TransformerEngine::expert_source(int layer, std::vector<TensorPtr>& src3d,
                                      std::vector<ExpertSlot>& slots,
                                      int& expert_dim) const {
    if (layer < 0 || layer >= static_cast<int>(weights_.layers.size())) return false;
    const auto& lw = weights_.layers[layer];

    // Architectures differ: PhiMoE exposes separate ffn_gate/up/down_exps;
    // Gemma4 exposes a combined ffn_gate_up_exps plus ffn_down_exps. Collect
    // whichever 3D expert tensors exist, tagging each with its slot.
    src3d.clear();
    slots.clear();
    auto add = [&](const TensorPtr& t, ExpertSlot s) {
        if (t && t->shape().size() == 3) {
            src3d.push_back(t);
            slots.push_back(s);
        }
    };
    add(lw.ffn_gate_exps(), ExpertSlot::Gate);
    add(lw.ffn_up_exps(), ExpertSlot::Up);
    add(lw.ffn_down_exps(), ExpertSlot::Down);
    add(lw.ffn_gate_up_exps(), ExpertSlot::GateUp);
    if (src3d.empty()) return false;

    // Which axis indexes experts? Match against the configured expert count; the
    // axis differs by architecture (PhiMoE: 0, Gemma4: 2). If it cannot be
    // determined we refuse rather than guess, and paging stays off for this
    // model (the non-paging path keeps working unchanged).
    const int n_expert = model_.config().n_expert;
    if (n_expert <= 0) return false;
    const auto& s = src3d[0]->shape();
    expert_dim = -1;
    for (int d = 0; d < 3; ++d) {
        if (s[d] == n_expert) {
            expert_dim = d;
            break;
        }
    }
    return expert_dim >= 0;
}

void TransformerEngine::sync_experts_resident(int layer,
                                              const std::vector<int>& active_experts) const {
    // P2: record the routed experts and, when paging is enabled, materialise +
    // move each one onto the layer's device. Paging is off by default, in which
    // case this only accumulates router statistics and touches no weight.
    if (active_experts.empty()) return;

    std::vector<TensorPtr> src3d;
    std::vector<ExpertSlot> slots;
    int expert_dim = 0;
    if (!expert_source(layer, src3d, slots, expert_dim)) return;

    const DeviceTarget target = layer_device_target(layer);
    const int64_t step = ++expert_step_;  // mutable, LRU ordering
    expert_page_cache_.record_active(layer, active_experts, step,
                                     expert_paging_enabled_, src3d, slots,
                                     expert_dim, target);
}

TensorPtr TransformerEngine::transfer_hidden(const TensorPtr& hidden, DeviceTarget target) const {
    if (hidden->device() == target.type)
        return hidden;

#ifdef USE_CUDA
    // Cross-GPU peer transfer: use cudaMemcpyPeer when both src and dst are CUDA
    // but on different GPUs
    if (hidden->device() == DeviceType::CUDA && target.is_cuda()) {
        int current_dev = -1;
        cudaGetDevice(&current_dev);
        // Get source device from hidden — since Tensor doesn't carry device_id,
        // we must infer it. For now, use the current CUDA device as source.
        // The caller should have called cudaSetDevice(src_device_id) before.
        int src_dev = current_dev;
        int dst_dev = target.device_id;

        if (src_dev != dst_dev) {
            // Enable peer access (once; no-op if already enabled)
            cudaDeviceEnablePeerAccess(dst_dev, 0);
            cudaDeviceEnablePeerAccess(src_dev, 0);

            cudaSetDevice(src_dev);
            auto transferred = std::make_shared<Tensor>(hidden->dtype(), hidden->shape(),
                                                        DeviceType::CUDA);
            cudaMemcpyPeer(transferred->data(), dst_dev,
                           hidden->data(), src_dev,
                           hidden->nbytes());
            cudaSetDevice(dst_dev);
            return transferred;
        }
    }
#endif

    auto transferred = std::make_shared<Tensor>(hidden->dtype(), hidden->shape(), target.type);
    transferred->copy_from(*hidden);
    return transferred;
}

TensorPtr TransformerEngine::forward_request(const ForwardRequest& req) {
    const auto& cfg = model_.config();

    // Decode path: use fewer threads (memory-bandwidth bound)
#ifdef _OPENMP
    omp_set_num_threads(ctx_.params().n_threads);
#endif

    init_kv_cache(cfg);

    DeviceTarget first_dev = layer_device_target(0);
#ifdef USE_CUDA
    if (first_dev.is_cuda()) cudaSetDevice(first_dev.device_id);
#endif

    auto token_emb = model_.weights().get("token_embedding");
    if (!token_emb) {
        fprintf(stderr, "[FATAL] token_embedding is NULL!\n");
        fflush(stderr);
        return nullptr;
    }

    // When token_embedding is on CPU (offload_embedding=false), run embedding on CPU
    // then transfer result to the first layer's device.
    DeviceType emb_dev = token_emb->device();
    auto ids_for_embed = transfer_hidden(req.input_ids,
                                         (emb_dev != first_dev.type && emb_dev == DeviceType::CPU)
                                             ? DeviceTarget::cpu() : first_dev);

    TensorPtr hidden;
    {
        PERF_SCOPE("forward/embedding");
        hidden = ops::embedding(token_emb, ids_for_embed, weights_.token_embedding_fp32);
    }
    if (!hidden) {
        fprintf(stderr, "[FATAL] embedding returned NULL!\n");
        fflush(stderr);
        return nullptr;
    }

    if (emb_dev != first_dev.type) {
        hidden = transfer_hidden(hidden, first_dev);
    }

    return forward_layers(hidden, req);
}

TensorPtr TransformerEngine::forward_batch(const InferenceBatch& batch) {
    if (batch.empty())
        return nullptr;

    const auto& cfg = model_.config();

    // Prefill/batch path: use more threads (compute-bound)
#ifdef _OPENMP
    omp_set_num_threads(ctx_.params().n_threads_batch);
#endif

    init_kv_cache(cfg);

    int n_ubatch = ctx_.params().n_ubatch;
    if (n_ubatch <= 0)
        n_ubatch = 256;

    // Split into micro-batches if needed
    auto micros = split_batch(batch, n_ubatch);

    // Collect per-sequence logits results
    struct SeqResult {
        int batch_idx;   // index in original batch
        int vocab_size;
        std::vector<float> logits;
    };
    std::vector<SeqResult> results;
    int vocab_size = -1;
    TensorPtr all_logits_result;  // all_logits 模式下的完整 logits

    for (const auto& ubatch : micros) {
        int total_tokens = ubatch.n_tokens();
        if (total_tokens == 0)
            continue;

        // ==== Step 1: Fused embedding for all tokens ====
        DeviceTarget first_dev = layer_device_target(0);
#ifdef USE_CUDA
        if (first_dev.is_cuda()) cudaSetDevice(first_dev.device_id);
#endif

        // Build flat [total_tokens] token ID tensor
        auto flat_ids = std::make_shared<Tensor>(DataType::INT32,
                                                  std::vector<int64_t>{total_tokens},
                                                  DeviceType::CPU);
        int32_t* ids_ptr = static_cast<int32_t*>(flat_ids->data());
        for (const auto& item : ubatch.items) {
            std::memcpy(ids_ptr, item.tokens.data(), item.tokens.size() * sizeof(int32_t));
            ids_ptr += item.tokens.size();
        }

        auto ids_on_dev = transfer_hidden(flat_ids, first_dev);
        auto token_emb = model_.weights().get("token_embedding");
        if (!token_emb) {
            LOG_ERROR("forward_batch: token_embedding is NULL");
            return nullptr;
        }

        // When token_embedding is intentionally on CPU (offload_embedding=false),
        // run embedding on CPU then transfer result to the first layer's device.
        DeviceType emb_dev = token_emb->device();
        TensorPtr ids_for_embed = ids_on_dev;
        if (emb_dev != first_dev.type && emb_dev == DeviceType::CPU) {
            ids_for_embed = transfer_hidden(flat_ids, DeviceTarget::cpu());
        }

        TensorPtr hidden;
        {
            PERF_SCOPE("forward_batch/embedding");
            hidden = ops::embedding(token_emb, ids_for_embed, weights_.token_embedding_fp32);
        }
        if (!hidden) {
            LOG_ERROR("forward_batch: embedding returned NULL");
            return nullptr;
        }

        if (emb_dev != first_dev.type) {
            hidden = transfer_hidden(hidden, first_dev);
        }
        // hidden shape: [total_tokens, hidden_dim]

        // ==== Step 2: Per-layer forward (split by sequence, forward_layer, concat) ====
        for (int layer = 0; layer < cfg.num_layers; ++layer) {
            DeviceTarget layer_dev = layer_device_target(layer);
#ifdef USE_CUDA
            if (layer_dev.is_cuda()) cudaSetDevice(layer_dev.device_id);
#endif
            hidden = transfer_hidden(hidden, layer_dev);

            // Split hidden by sequence, forward_layer per sequence, concat back
            auto offsets = ubatch.token_offsets();
            int hidden_dim = static_cast<int>(hidden->shape().back());
            auto layer_out = std::make_shared<Tensor>(DataType::FP32,
                                                       std::vector<int64_t>{total_tokens, hidden_dim},
                                                       layer_dev.type);
            float* dst = static_cast<float*>(layer_out->data());

            for (int i = 0; i < ubatch.size(); i++) {
                const auto& item = ubatch.items[i];
                int seq_len = static_cast<int>(item.tokens.size());
                int offset = offsets[i];

                // Extract this sequence's slice from flat hidden
                TensorPtr seq_hidden = std::make_shared<Tensor>(hidden->slice(0, offset, offset + seq_len));

                // Forward through this layer for this sequence
                {
                    PERF_SCOPE_FMT("forward_batch/layer_%d", layer);
                    SET_PERF_CONTEXT(item.seq_id, "layer", layer, layer_dev.is_cuda() ? "cuda" : "cpu", seq_len);
                    auto seq_req = ForwardRequest::from_hidden(seq_len, item.start_pos, item.seq_id);
                    seq_hidden =
                        forward_layer(seq_hidden, make_layer_context(layer, seq_req, layer_dev));
                }

                if (!seq_hidden) {
                    LOG_ERROR("forward_batch: layer " + std::to_string(layer) + " returned NULL");
                    return nullptr;
                }

                // Copy result back into flat output
                TensorPtr seq_cpu = seq_hidden;
                if (seq_hidden->device() == DeviceType::CUDA && layer_dev == DeviceType::CUDA) {
#ifdef USE_CUDA
                    // GPU→GPU: use cudaMemcpy instead of CPU memmove
                    size_t bytes = seq_len * hidden_dim * sizeof(float);
                    cudaMemcpyAsync(dst + offset * hidden_dim,
                                     static_cast<const float*>(seq_hidden->data()),
                                     bytes, cudaMemcpyDeviceToDevice);
#else
                    const float* src = static_cast<const float*>(seq_hidden->data());
                    size_t bytes = seq_len * hidden_dim * sizeof(float);
                    std::memcpy(dst + offset * hidden_dim, src, bytes);
#endif
                } else if (seq_hidden->device() != layer_dev) {
                    seq_cpu = transfer_hidden(seq_hidden, layer_dev);
                    const float* src = static_cast<const float*>(seq_cpu->data());
                    size_t bytes = seq_len * hidden_dim * sizeof(float);
                    std::memcpy(dst + offset * hidden_dim, src, bytes);
                } else {
                    const float* src = static_cast<const float*>(seq_hidden->data());
                    size_t bytes = seq_len * hidden_dim * sizeof(float);
                    std::memcpy(dst + offset * hidden_dim, src, bytes);
                }
            }

            hidden = layer_out;
        }

        // ==== Step 3: Fused output norm + projection ====
        {
            PERF_SCOPE("forward_batch/output_norm");
            auto output_norm = weights_.output_norm;
            hidden = ops::rms_norm(hidden, output_norm, cfg.rms_norm_eps);
        }
        last_hidden_ = hidden;  // forward_batch (prefill) hidden exposure

        auto output_weight = weights_.output_weight;
        if (!output_weight && cfg.tie_embeddings) {
            output_weight = weights_.token_embedding;
        }

        TensorPtr logits;
        {
            PERF_SCOPE("forward_batch/output_proj");
            if (output_weight) {
                hidden = transfer_hidden(hidden, output_weight->device());
            }
            logits = ops::matmul_transB(hidden, output_weight);
        }

        // ==== Step 4: Extract logits ====
        if (!logits)
            continue;

        TensorPtr logits_cpu = logits;
        if (logits->device() == DeviceType::CUDA) {
            logits_cpu = std::make_shared<Tensor>(DataType::FP32, logits->shape(), DeviceType::CPU);
            logits_cpu->copy_from(*logits);
        }

        // all_logits 模式：直接返回完整 [total_tokens, vocab] 张量
        if (batch.all_logits) {
            // 保存第一个 micro-batch 的完整 logits 用于返回
            if (!all_logits_result) {
                all_logits_result = logits_cpu;
            } else {
                // 拼接多个 micro-batch 的 logits
                int prev_rows = static_cast<int>(all_logits_result->shape()[0]);
                int cur_rows = static_cast<int>(logits_cpu->shape()[0]);
                int vs = static_cast<int>(logits_cpu->shape()[1]);
                auto merged = std::make_shared<Tensor>(DataType::FP32,
                                                        std::vector<int64_t>{prev_rows + cur_rows, vs},
                                                        DeviceType::CPU);
                std::memcpy(merged->data(), all_logits_result->data(), prev_rows * vs * sizeof(float));
                std::memcpy(static_cast<float*>(merged->data()) + prev_rows * vs,
                            logits_cpu->data(), cur_rows * vs * sizeof(float));
                all_logits_result = merged;
            }
            continue;
        }

        vocab_size = static_cast<int>(logits_cpu->shape().back());
        const float* logits_data = static_cast<const float*>(logits_cpu->data());
        auto offsets = ubatch.token_offsets();

        for (int i = 0; i < ubatch.size(); i++) {
            if (!ubatch.items[i].logits)
                continue;

            int seq_len = static_cast<int>(ubatch.items[i].tokens.size());
            int last_row = offsets[i] + seq_len - 1;

            // Find original batch index
            int batch_idx = -1;
            for (int j = 0; j < batch.size(); j++) {
                if (batch.items[j].seq_id == ubatch.items[i].seq_id) {
                    batch_idx = j;
                    break;
                }
            }

            SeqResult res;
            res.batch_idx = batch_idx;
            res.vocab_size = vocab_size;
            res.logits.assign(logits_data + last_row * vocab_size,
                              logits_data + (last_row + 1) * vocab_size);
            results.push_back(std::move(res));
        }
    }

    if (results.empty() || vocab_size <= 0) {
        if (batch.all_logits && all_logits_result)
            return all_logits_result;
        return nullptr;
    }

    // Assemble [n_seq, vocab_size] result in original batch order
    int n_seq = batch.size();
    auto result = std::make_shared<Tensor>(DataType::FP32,
                                            std::vector<int64_t>{n_seq, vocab_size},
                                            DeviceType::CPU);
    float* dst = static_cast<float*>(result->data());
    std::memset(dst, 0, n_seq * vocab_size * sizeof(float));

    for (auto& res : results) {
        if (res.batch_idx >= 0 && res.batch_idx < n_seq) {
            std::memcpy(dst + res.batch_idx * vocab_size,
                        res.logits.data(), vocab_size * sizeof(float));
        }
    }

    return result;
}

TensorPtr TransformerEngine::forward_from_hidden(const TensorPtr& hidden, int64_t start_pos) {
    const auto& cfg = model_.config();
    int seq_len = static_cast<int>(hidden->shape()[0]);

    init_kv_cache(cfg);

    return forward_layers(hidden, ForwardRequest::from_hidden(seq_len, start_pos, /*seq_id=*/0));
}

void TransformerEngine::init_kv_cache(const ModelConfig& cfg) {
    if (kv_cache_initialized())
        return;

    int kv_max_seq = cfg.max_seq_len;
    const int KV_MAX_SEQ_CAP = 4096;
    if (kv_max_seq > KV_MAX_SEQ_CAP) {
        LOG_INFO("Capping KV cache max_seq_len from " + std::to_string(kv_max_seq) + " to " +
                 std::to_string(KV_MAX_SEQ_CAP) + " to avoid OOM");
        kv_max_seq = KV_MAX_SEQ_CAP;
    }

    // Determine the primary KV device.
    // When offload_kqv is false, KV cache stays on CPU even if layers are on GPU.
    // This enables "weights on GPU, KV on CPU" for long-context scenarios.
    DeviceType kv_dev;
    if (ctx_.params().offload_kqv) {
        kv_dev = (gpu_layers_ >= cfg.num_layers) ? DeviceType::CUDA : DeviceType::CPU;
    } else {
        kv_dev = DeviceType::CPU;
    }

    // If CUDA, check available memory and reduce if needed
    if (kv_dev == DeviceType::CUDA) {
        size_t kv_bytes = (size_t)cfg.num_layers * 2 * (size_t)kv_max_seq * cfg.num_kv_heads *
                          cfg.head_dim * sizeof(float);
        auto backend = BackendManager::instance().get_cuda_backend();
        if (backend) {
            size_t free_mem = backend->device_memory_free();
            // Leave 256MB headroom for other allocations
            const size_t headroom = 256 * 1024 * 1024;
            if (kv_bytes + headroom > free_mem && free_mem > headroom) {
                size_t available = free_mem - headroom;
                int reduced_seq =
                    static_cast<int>(available / ((size_t)cfg.num_layers * 2 * cfg.num_kv_heads *
                                                  cfg.head_dim * sizeof(float)));
                if (reduced_seq < kv_max_seq) {
                    // Round down to nearest 256 for alignment
                    reduced_seq = (reduced_seq / 256) * 256;
                    if (reduced_seq < 256)
                        reduced_seq = 256;
                    LOG_WARN("KV cache needs " + std::to_string(kv_bytes / (1024 * 1024)) +
                             "MB but only " + std::to_string(free_mem / (1024 * 1024)) +
                             "MB free, reducing max_seq_len from " + std::to_string(kv_max_seq) +
                             " to " + std::to_string(reduced_seq));
                    kv_max_seq = reduced_seq;
                }
            }
        }
    }

    // Reserve one extra KV layer per MTP nextn block: the draft module reuses
    // engine attention at layer index cfg.num_layers.
    const int kv_num_layers = cfg.num_layers + std::max(0, cfg.n_nextn_layers);

    LOG_INFO("KV cache init: layers=" + std::to_string(kv_num_layers) + ", kv_heads=" +
             std::to_string(cfg.num_kv_heads) + ", head_dim=" + std::to_string(cfg.head_dim) +
             ", max_seq_len=" + std::to_string(kv_max_seq) +
             ", dev=" + (kv_dev == DeviceType::CUDA ? "CUDA" : "CPU"));
    size_t kv_bytes = (size_t)kv_num_layers * 2 * (size_t)kv_max_seq * cfg.num_kv_heads *
                      cfg.head_dim * sizeof(float);
    LOG_INFO("KV cache estimated size: " + std::to_string(kv_bytes / (1024 * 1024)) + " MB");

    // Allocate KV cache on the primary device
    // For paged mode, KVCache is still initialized (for transitional compatibility)
    // but PagedKVStorage is the primary storage backend.
    kv_cache_.init_quantized(kv_num_layers, cfg.num_kv_heads, cfg.head_dim, kv_max_seq, kv_dev,
                             kv_cache_dtype_);

    // Phase 6: set per-layer memory policies from arch config.
    // SlidingWindow layers use ring buffer eviction; Full layers grow linearly.
    if (cfg.n_swa > 0) {
        std::vector<KVLayerPolicy> policies(kv_num_layers, KVLayerPolicy::Full);
        bool has_swa_layers = false;
        for (int i = 0; i < cfg.num_layers; ++i) {
            if (i < (int)cfg.swa_layers.size() && cfg.swa_layers[i] == 1) {
                policies[i] = KVLayerPolicy::SlidingWindow;
                has_swa_layers = true;
            }
        }
        // If swa_layers is empty but n_swa > 0, treat as uniform SWA (all layers)
        if (!has_swa_layers) {
            policies.assign(cfg.num_layers, KVLayerPolicy::SlidingWindow);
        }
        kv_cache_.set_layer_policies(policies, cfg.n_swa);
    }

    // Place each layer's KV cache on the corresponding device.
    // When offload_kqv is false, skip per-layer device placement so KV stays on CPU
    // even if the layer weights are on GPU.
    if (!layer_devices_.empty() && ctx_.params().offload_kqv) {
        kv_cache_.set_layer_devices(layer_devices_);
    }

    // Initialize paged storage if paged mode is enabled
    if (kv_memory_ && kv_memory_->is_paged()) {
        std::vector<int> kv_dims(cfg.num_layers, cfg.num_kv_heads * cfg.head_dim);
        int page_size = 16;  // Phase 3: fixed page size, benchmark-tuned later
        int max_num_seqs = 32;
        KVCacheTypeConfig kv_config;
        kv_config.type_k = kv_cache_dtype_;
        kv_config.type_v = kv_cache_dtype_;
        // Phase 6: set layer policies before init so SWA pools are sized correctly
        if (cfg.n_swa > 0) {
            std::vector<KVLayerPolicy> policies(kv_num_layers, KVLayerPolicy::Full);
            for (int i = 0; i < cfg.num_layers; ++i) {
                if (i < (int)cfg.swa_layers.size() && cfg.swa_layers[i] == 1)
                    policies[i] = KVLayerPolicy::SlidingWindow;
            }
            kv_memory_->set_layer_policies(policies, cfg.n_swa);
        }
        // Paged storage follows the KV cache device: CUDA engine → CUDA pages,
        // CPU engine → CPU pages (Phase 3 behavior).
        if (!kv_memory_->init_storage(cfg.num_layers, kv_dims, kv_max_seq,
                                      kv_dev, kv_config, page_size, max_num_seqs)) {
            LOG_ERROR("Failed to initialize paged KV storage");
        }
    }

    set_kv_cache_initialized(true);
    LOG_INFO("KV cache initialized successfully, actual size: " +
             std::to_string(kv_cache_.nbytes() / (1024 * 1024)) + " MB");
}

LayerExecutionContext TransformerEngine::make_layer_context(int layer_idx,
                                                            const ForwardRequest& req,
                                                            DeviceTarget dev) const {
    return LayerExecutionContext{model_.config(), weights_.layers[layer_idx], req, layer_idx, dev};
}

TensorPtr TransformerEngine::forward_layers(const TensorPtr& hidden, const ForwardRequest& req) {
    const int seq_len = req.n_tokens;
    const int64_t start_pos = req.start_pos;

    // DFlash/DSPark: when capturing per-layer hiddens, drop the graph path (it
    // packs the whole forward into one kernel) and reset the cache so only this
    // forward's layers are retained.
    if (captures_layer_hiddens()) {
        layer_hiddens_.clear();
    }
    const bool use_graph = use_graph_ && !captures_layer_hiddens();

    // Graph 执行。builder 由 ExecutionPlan 在构造时决定, 执行期不再按架构名查表。
    // 不支持 graph 的架构直接退回 imperative, 不使用 placeholder builder。
    if (use_graph) {
        if (!graph_runtime_.has_builder()) {
            LOG_WARN("Graph execution not supported by plan " + plan_.plan_id() +
                     ", falling back to imperative mode");
        } else {
            auto key = GraphKey::from_request(req, plan_.plan_id(), gpu_layers_, hidden->device());
            return graph_runtime_.run(hidden, req, key, model_.config(), weights_, kv_cache_,
                                      layer_devices_, model_.device());
        }
    }

    // Imperative execution (original path)
    const auto& cfg = model_.config();
    auto t0 = std::chrono::steady_clock::now();

    auto cur_hidden = hidden;
    for (int layer = 0; layer < cfg.num_layers; ++layer) {
        DeviceTarget layer_dev = layer_device_target(layer);
#ifdef USE_CUDA
        if (layer_dev.is_cuda()) cudaSetDevice(layer_dev.device_id);
#endif
        cur_hidden = transfer_hidden(cur_hidden, layer_dev);
        {
            PERF_SCOPE_FMT("forward/layer_%d", layer);
            SET_PERF_CONTEXT(req.seq_id, "layer", layer, layer_dev.is_cuda() ? "cuda" : "cpu", req.n_tokens);
            cur_hidden = forward_layer(cur_hidden, make_layer_context(layer, req, layer_dev));
        }
        if (captures_layer_hiddens()) {
            layer_hiddens_.push_back(cur_hidden);
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
    // Post-final-norm hidden (LM-head input feature), stashed for the MTP
    // draft module. One [M, H] tensor per forward is cheap.
    last_hidden_ = cur_hidden;

    auto output_weight = weights_.output_weight;
    if (!output_weight && cfg.tie_embeddings) {
        output_weight = weights_.token_embedding;
    }
    // CPU: output_weight keeps its native quantized format in ModelWeights::init()
    // (Q4_0/Q8_0/Q4_1/Q4_K/Q6_K supported), dispatched to fused GEMV kernels by matmul_transB.
    if (output_weight) {
        // output_weight->device() returns DeviceType; implicit conversion to DeviceTarget
        cur_hidden = transfer_hidden(cur_hidden, DeviceTarget(output_weight->device()));
    }
    TensorPtr logits;
    {
        PERF_SCOPE("forward/output_proj");
        // Use specialized output_proj kernel for decode (M=1, large N)
        if (output_weight && output_weight->device() == DeviceType::CUDA && seq_len == 1) {
            int K = static_cast<int>(output_weight->shape()[1]);
            int N = static_cast<int>(output_weight->shape()[0]);
            logits = std::make_shared<Tensor>(DataType::FP32, std::vector<int64_t>{1, N},
                                              DeviceType::CUDA);
#ifdef USE_CUDA
            auto dtype = output_weight->dtype();
            if (dtype == DataType::Q4_0) {
                cuda::launch_output_proj_q4_0(static_cast<const float*>(cur_hidden->data()),
                                              output_weight->data(),
                                              static_cast<float*>(logits->data()), K, N);
            } else if (dtype == DataType::Q4_K) {
                cuda::launch_output_proj_q4_k(static_cast<const float*>(cur_hidden->data()),
                                              output_weight->data(),
                                              static_cast<float*>(logits->data()), K, N);
            } else if (dtype == DataType::Q5_K) {
                cuda::launch_output_proj_q5_k(static_cast<const float*>(cur_hidden->data()),
                                              output_weight->data(),
                                              static_cast<float*>(logits->data()), K, N);
            } else if (dtype == DataType::Q6_K) {
                cuda::launch_output_proj_q6_k(static_cast<const float*>(cur_hidden->data()),
                                              output_weight->data(),
                                              static_cast<float*>(logits->data()), K, N);
            } else {
                logits = ops::matmul_transB(cur_hidden, output_weight);
            }
#endif
        } else {
            logits = ops::matmul_transB(cur_hidden, output_weight);
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
#if defined(FORGE_PROFILING) && FORGE_PROFILING == 1
    LOG_INFO("Forward total: " + std::to_string((int)total_ms) + "ms (seq_len=" +
             std::to_string(seq_len) + ", start_pos=" + std::to_string(start_pos) + ")");
#endif

    return logits;
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
        forge::cpu::expand_kv_heads_f32(kv_data, out_data, seq_len, num_heads, num_kv_heads, head_dim);
    }

    return expanded;
}

}  // namespace forge
