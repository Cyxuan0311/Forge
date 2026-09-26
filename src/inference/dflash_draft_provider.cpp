#include "forge/dflash_draft_provider.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>

#include "forge/arch_registry.h"
#include "forge/context.h"
#include "forge/engine.h"
#include "forge/engines/dflash_engine.h"
#include "forge/engines/transformer_engine.h"
#include "forge/gguf_model.h"
#include "forge/inference/forward_request.h"
#include "forge/logger.h"
#include "forge/model.h"
#include "forge/model_loader.h"
#include "forge/ninf_model.h"
#include "forge/sampler.h"

namespace forge {

namespace {

void register_runtime() {
    force_link_arch_registrations();
    auto& registry = ModelLoaderRegistry::instance();
    registry.register_loader(
        "gguf", []() -> std::unique_ptr<ModelLoader> { return std::make_unique<GgufModel>(); });
    registry.register_loader(
        "ninf", []() -> std::unique_ptr<ModelLoader> { return std::make_unique<NinfModel>(); });
}

std::unique_ptr<InferenceContext> make_context(Model& model, const ContextParams& target,
                                               int gpu_layers) {
    ContextParams params = target;
    params.gpu_layers = gpu_layers;
    params.speculative_config = SpeculativeConfig{};  // drafter must not recurse

    auto ctx = std::make_unique<InferenceContext>(model, params);
    auto engine = EngineRegistry::instance().create(model.config().arch_type, model, *ctx);
    if (!engine)
        return nullptr;
    if (auto* transformer = dynamic_cast<TransformerEngine*>(engine.get())) {
        transformer->set_gpu_layers(gpu_layers);
        transformer->set_kv_cache_dtype(params.kv_cache_dtype);
        // drafter 只需要 [已确认前缀 + query block] 的 KV, 按模型原生 262144
        // 全量分配毫无意义: 27B 目标场景实测 3GB 匿名内存挤占权重页缓存,
        // 整机 swap 颠簸。上限 8192 (约 328MB @fp32) 覆盖绝大多数 CLI 场景;
        // 超长上下文场景需另行放宽。
        transformer->set_kv_cache_max_seq_len(
            std::min(8192, model.config().max_seq_len));
    }
    ctx->set_engine(std::move(engine));
    return ctx;
}

float top_probability(const float* logits, int vocab_size) {
    float max_logit = logits[0];
    for (int i = 1; i < vocab_size; ++i)
        max_logit = std::max(max_logit, logits[i]);
    double sum = 0.0;
    for (int i = 0; i < vocab_size; ++i)
        sum += std::exp(static_cast<double>(logits[i] - max_logit));
    return sum > 0.0 ? static_cast<float>(1.0 / sum) : 0.0f;
}

}  // namespace

struct DFlashDraftProvider::Impl {
    Model model;
    std::unique_ptr<InferenceContext> ctx;
    DflashEngine* engine = nullptr;     // raw pointer into ctx->engine()
    InferenceEngine* target = nullptr;  // borrowed target engine
    Sampler sampler;
    ContextParams target_params;
    std::string model_path;
    int gpu_layers = 0;
    int vocab_size = 0;
    float p_min = 0.0f;
    int n_spec = 5;
    int mask_token_id_ = 0;
    bool is_dspark = false;
    bool valid = false;

    int64_t committed_len = 0;
    std::vector<int32_t> committed_ids;
    // drafter KV 中已写入上下文 K/V 的位置数 (持久化, 逐轮增量追加)
    int64_t kv_filled = 0;

    Impl(const SpeculativeConfig& spec, InferenceContext& tctx, InferenceEngine& t,
         int target_vocab)
        : sampler([] {
              SamplerConfig config;
              config.do_sample = false;  // greedy proposals (verify is lossless)
              config.temperature = 0.0f;
              config.top_k = 0;
              config.top_p = 1.0f;
              config.repeat_penalty = 1.0f;
              return config;
          }()),
          target_params(tctx.params()),
          model_path(spec.draft_model_path),
          gpu_layers(spec.draft_gpu_layers >= 0
                         ? spec.draft_gpu_layers
                         : (dynamic_cast<const TransformerEngine*>(&t)
                                ? dynamic_cast<const TransformerEngine*>(&t)->gpu_layers()
                                : tctx.params().gpu_layers)),
          vocab_size(target_vocab),
          p_min(spec.p_min),
          n_spec(std::max(1, spec.draft_n_spec)),
          mask_token_id_(spec.draft_mask_token_id >= 0 ? spec.draft_mask_token_id : -1),
          is_dspark(spec.draft_arch == "dspark"),
          target(&t) {
        register_runtime();

        if (model_path.empty() || !model.load(model_path, gpu_layers)) {
            LOG_WARN("DFlashDraftProvider: failed to load drafter model '" + model_path + "'");
            return;
        }

        // MASK token id resolution chain (mirrors vLLM get_parallel_drafting_token_id):
        //   1. explicit --spec-draft-mask-id (kept from the member initializer);
        //   2. GGUF "dflash.parallel_drafting_token_id" (parsed into ModelConfig);
        //   3. GGUF "tokenizer.ggml.mask_token_id" (common in Qwen3 DFlash2 checkpoints).
        // Falling back to 0 produces a wrong parallel block (id 0 is an ordinary
        // token), so we bail out loudly instead of silently degrading acceptance.
        if (mask_token_id_ < 0) {
            if (model.config().parallel_drafting_token_id >= 0) {
                mask_token_id_ = model.config().parallel_drafting_token_id;
            } else {
                mask_token_id_ =
                    static_cast<int>(model.metadata_int("tokenizer.ggml.mask_token_id", -1));
            }
        }
        if (mask_token_id_ < 0) {
            LOG_WARN(
                "DFlashDraftProvider: no MASK token id in the drafter GGUF "
                "(dflash.parallel_drafting_token_id / tokenizer.ggml.mask_token_id). "
                "Pass --spec-draft-mask-id <id> explicitly.");
            return;
        }

        // Target must share token embeddings (embed_via_target static_casts).
        auto* t_transformer = dynamic_cast<TransformerEngine*>(target);
        if (!t_transformer) {
            LOG_WARN("DFlashDraftProvider: target engine is not a TransformerEngine");
            return;
        }
        // DFlash/DSPark drafter 借用 target 的 token embedding 与 lm_head
        // (见 DflashEngine::embed_via_target / set_target), 从不使用自身词表,
        // 因此 GGUF 头里的 vocab_size 元数据不可作为配对依据 (转换器常写入
        // 默认值 32000, 与 248320 的 Qwen 词表不符)。真正需要一致的是 hidden
        // 维度: target embedding 直接流入 drafter 各层。
        const int64_t target_hidden =
            t_transformer->weights().token_embedding
                ? static_cast<int64_t>(t_transformer->weights().token_embedding->shape()[1])
                : 0;
        if (target_hidden > 0 && model.config().hidden_dim != static_cast<int>(target_hidden)) {
            LOG_WARN("DFlashDraftProvider: drafter/target hidden size mismatch (" +
                     std::to_string(model.config().hidden_dim) + " vs " +
                     std::to_string(target_hidden) + ")");
            return;
        }
        if (target_hidden > 0 && model.config().target_hidden_size > 0 &&
            model.config().target_hidden_size != static_cast<int>(target_hidden)) {
            LOG_WARN("DFlashDraftProvider: drafter metadata target_hidden_size (" +
                     std::to_string(model.config().target_hidden_size) +
                     ") != actual target hidden (" + std::to_string(target_hidden) + ")");
            return;
        }

        ctx = make_context(model, target_params, gpu_layers);
        if (!ctx || !ctx->engine()) {
            LOG_WARN("DFlashDraftProvider: failed to create drafter inference context");
            return;
        }
        engine = dynamic_cast<DflashEngine*>(ctx->engine());
        if (!engine) {
            LOG_WARN("DFlashDraftProvider: engine is not a DflashEngine");
            return;
        }
        engine->set_target(target);
        if (is_dspark) {
            engine->set_dspark(true);
            if (!engine->has_markov_head()) {
                // DSPark requires the sequential Markov head; a plain DFlash GGUF
                // does not carry markov_embed/markov_bias. Skip gracefully so the
                // run falls back to plain (non-speculative) decoding.
                LOG_WARN(
                    "DFlashDraftProvider: --spec-draft-arch dspark requested but the "
                    "drafter has no markov head; skipping DSPark");
                return;
            }
        }

        valid = true;
        LOG_INFO(std::string("DFlashDraftProvider: ready (") + (is_dspark ? "dspark" : "dflash") +
                 ") n_spec=" + std::to_string(n_spec) + " (incremental context KV)");
    }

    void reset() {
        if (ctx)
            ctx->reset_kv_cache();
        committed_len = 0;
        committed_ids.clear();
        kv_filled = 0;
    }
};

DFlashDraftProvider::DFlashDraftProvider(const SpeculativeConfig& spec, InferenceContext& ctx,
                                         InferenceEngine& target, int target_vocab)
    : impl_(nullptr),
      is_dspark_(spec.draft_arch == "dspark") {
    impl_ = std::make_unique<Impl>(spec, ctx, target, target_vocab);
}

DFlashDraftProvider::~DFlashDraftProvider() = default;

bool DFlashDraftProvider::valid() const {
    return impl_ && impl_->valid;
}

void DFlashDraftProvider::begin(const std::vector<int32_t>& prompt) {
    if (!valid())
        return;
    impl_->reset();
    impl_->committed_ids.assign(prompt.begin(), prompt.end());
    impl_->committed_len = static_cast<int64_t>(prompt.size());
}

std::vector<int32_t> DFlashDraftProvider::draft(int32_t last_token, int n_draft) {
    std::vector<int32_t> result;
    if (!valid() || !impl_->engine || n_draft <= 0)
        return result;

    const int n = std::min(n_draft, impl_->n_spec);

    // anchor 吸收: prefill 产出的首个 token (t_0) 不经过 accept()/notify_confirmed
    // (begin() 只登记了 prompt), committed_ids 必须在此把它并入, 否则 drafter
    // 上下文从第二轮起整体错一位。executor 的 plain 步骤会逐 token notify, 此后
    // back()==last_token 恒成立, 本分支自然退化为何等判断, 幂等。
    if (impl_->committed_ids.empty() || impl_->committed_ids.back() != last_token) {
        impl_->committed_ids.push_back(last_token);
        impl_->committed_len += 1;
    }

    // query block 宽度固定 (上游 n_max 语义): 块是非因果 forward, 宽度收缩会让
    // 上一轮更宽 block 留下的 MASK 残行逃过本轮覆盖, 注意力读到垃圾 —— 因此宽度
    // 只由配置上限与训练块大小决定, 不随 adaptive n_draft 变化。返回值仍截断到
    // 本轮请求的 n, executor 的 adaptive/budget 语义不变。
    // DFlash2 的 conv/selector 还要求块宽不超过训练时的 dflash.block_size。
    int n_block = impl_->n_spec;
    if (impl_->model.config().dflash_block_size > 1)
        n_block = std::min(impl_->model.config().dflash_block_size - 1, impl_->n_spec);

    // 1. 增量注入上下文 K/V: delta = committed_ids[kv_filled, committed_len-1)。
    //    末位的 anchor 不经 target 重放镜像, 而是作为 query block 首行在它的真实
    //    位置 (committed_len-1) 写入 drafter KV。旧实现把 anchor 也算进 delta,
    //    导致 block 整体后移一位 (anchor 重复 + MASK 行位置 +1)。上一轮 query
    //    block 的残行 (旧 MASK 行) 已由 decode() 末尾的 KV 回滚清除, 本 delta
    //    从 anchor 位置起重写, 每轮 O(delta), 不再整前缀重建。
    const int64_t ctx_end = impl_->committed_len - 1;
    if (impl_->kv_filled < ctx_end) {
        std::vector<int32_t> delta(impl_->committed_ids.begin() + impl_->kv_filled,
                                   impl_->committed_ids.begin() + ctx_end);
        impl_->engine->append_context_kv(delta, impl_->kv_filled);
        impl_->kv_filled = ctx_end;
    }

    // 2. query block: [anchor=last_token, MASK*n_block] (non-causal forward),
    //    anchor 置于它的真实位置 committed_len-1 (上游 dp.pos0)。
    std::vector<int32_t> query(static_cast<size_t>(n_block) + 1);
    query[0] = last_token;
    for (int i = 1; i <= n_block; ++i)
        query[static_cast<size_t>(i)] = impl_->mask_token_id_;

    TensorPtr logits = impl_->engine->decode(query, impl_->committed_len - 1);
    if (!logits || logits->ndim() < 2) {
        LOG_WARN("DFlashDraftProvider: decode failed; skipping draft");
        return result;
    }
    if (logits->device() != DeviceType::CPU)
        logits->to_device(DeviceType::CPU);

    // 3. sample n_block tokens. DFlash takes greedy argmax of each row (parallel,
    //    position-independent drafts); DSPark walks the rows left-to-right and
    //    injects the previous sampled token through the markov head, so the block
    //    acquires an intra-dependence (vLLM dspark/_sample_sequential). Both feed
    //    the same [anchor, MASK*N] decode output; resample-consistency in the
    //    verify step keeps generation lossless regardless of the sampling policy.
    //    DFlash2 skips the logits entirely: the decode() call packs a top-k
    //    transition lattice (backend sampling disabled upstream for DFlash2) and
    //    the CPU greedy walk reads it.
    const int vocab = static_cast<int>(logits->shape()[1]);
    const float* data = static_cast<const float*>(logits->data());

    if (impl_->engine->is_dflash2()) {
        TensorPtr lattice = impl_->engine->dflash2_lattice();
        if (!lattice || lattice->ndim() < 2)
            return result;  // block too short / lattice unavailable → no draft
        result = dflash2_lattice_walk(static_cast<const float*>(lattice->data()),
                                      lattice->shape()[1],
                                      impl_->model.config().dflash_selector_top_k,
                                      static_cast<int>(lattice->shape()[0]), impl_->p_min);
        if (static_cast<int>(result.size()) > n)
            result.resize(n);
        return result;
    }

    if (impl_->is_dspark && impl_->engine->has_markov_head()) {
        std::vector<const float*> rows;
        rows.reserve(n_block);
        // Row 0 is the anchor/bonus position: its logits predict last_token
        // itself and are never part of the draft (vLLM samples only the MASK
        // rows at offsets >= 1, see _prepare_dflash_inputs_kernel). Skip it.
        for (int j = 0; j < n_block; ++j)
            rows.push_back(data + static_cast<int64_t>(j + 1) * vocab);
        TensorPtr me = impl_->engine->markov_embed();
        TensorPtr mb = impl_->engine->markov_bias();
        TensorPtr idmap = impl_->engine->draft_id_map();
        const float* mep = me ? static_cast<const float*>(me->data()) : nullptr;
        const float* mbp = mb ? static_cast<const float*>(mb->data()) : nullptr;
        const int32_t* idp = idmap ? static_cast<const int32_t*>(idmap->data()) : nullptr;
        const int md = me ? static_cast<int>(me->shape()[1]) : 0;
        result = dspark_sequential_sample(rows, vocab, mep, md, mbp, idp, last_token, n_block);
        // p_min confidence gate still applies: drop the tail if the first markov
        // position is below threshold (rare for DSPark, kept for parity).
        if (impl_->p_min > 0.0f && !result.empty()) {
            const float* row0 = data + vocab;  // first MASK row, not the anchor
            if (top_probability(row0, vocab) < impl_->p_min)
                result.clear();
        }
        if (static_cast<int>(result.size()) > n)
            result.resize(n);
        return result;
    }

    for (int j = 0; j < n_block; ++j) {
        // Skip the anchor/bonus row (index 0): only the MASK rows (1..n_block) are
        // sampled, matching vLLM's _prepare_dflash_inputs_kernel.
        const float* row = data + static_cast<int64_t>(j + 1) * vocab;
        if (impl_->p_min > 0.0f && top_probability(row, vocab) < impl_->p_min)
            break;
        int best = 0;
        for (int t = 1; t < vocab; ++t)
            if (row[t] > row[best])
                best = t;
        result.push_back(best);
    }
    if (static_cast<int>(result.size()) > n)
        result.resize(n);
    return result;
}

void DFlashDraftProvider::accept(const std::vector<int32_t>& tokens) {
    if (!valid() || tokens.empty())
        return;
    // drafter KV 现在持久化并在 draft() 里增量追加, 这里只推进已确认前缀的
    // 记账。SpeculativeExecutor 传入完整确认序列 (accepted + resampled/bonus),
    // 与 target 的确认边界一致; 部分接受时 executor 的 KV rollback 只作用于
    // target 侧缓存, drafter 缓存中的旧 query 残行会在下轮被覆盖。
    impl_->committed_ids.insert(impl_->committed_ids.end(), tokens.begin(), tokens.end());
    impl_->committed_len += static_cast<int64_t>(tokens.size());
}

void DFlashDraftProvider::reset() {
    if (impl_)
        impl_->reset();
}

}  // namespace forge
