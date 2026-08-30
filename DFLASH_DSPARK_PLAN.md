# Forge DFlash / DSPark 投机解码接入计划

> 参考实现:
> - llama.cpp `/mnt/f/My__StudyStack/From_Github/llama.cpp/src/models/dflash.cpp`（GGUF 架构、encoder/decoder 图）
> - vLLM `/mnt/f/My__StudyStack/From_Github/vllm/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`（propose / context-KV 预计算 / 并行采样）
> - vLLM `.../dspark/speculator.py`（顺序 Markov head、`sample_from_anchor`）
>
> 本计划对齐 `SPECULATIVE_DECODING_PLAN.md` 的 `IDraftProvider` 框架，**只新增草稿源，不改验证算法**。

---

## 一、目标

在现有 `IDraftProvider` + `SpeculativeExecutor` + `verify_draft_tokens` 框架上，新增 **DFlash** 与 **DSPark** 两种草稿源。两者都是**独立的轻量草稿 Transformer**，但共享 target 模型的 `token_embedding` 与 `output_weight`（lm_head），并把 target 的**多层中间隐状态**作为编码器输入。

验证侧（`verify_draft_tokens` 的 greedy argmax + resample-consistency）与统计（`SpeculativeStats` / `--spec-stats`）**完全复用**，无需改动。

---

## 二、算法要点

### 2.1 DFlash（一次性并行草稿）

草稿模型是一个标准 transformer：`n_layer` 层，每层 attn + ffn（权重见 llama.cpp `dflash.cpp:35-61`）。GGUF 元数据 `target_layers` 给出要从 target 抽取的特征层，encoder 维度 `n_embd_inp_enc = |target_layers| * n_embd`。

`propose`（一次投机轮）流程：

1. **encoder**：取 target 在承诺前缀上的 `target_layers` 隐状态 → 按层序拼接为 `[seq, n_embd_inp_enc]` → `fc` 投影到 `n_embd` → `output_norm_enc`（RMSNorm）→ context feature（可缓存）。
2. **context-KV 预计算**：把 target 的 token embedding 经 `wk/wv` 投影 + RoPE，**注入** draft 自己的 KV cache（位置 = 承诺前缀）。这样 draft 的 attention 能看到真实的 target 前缀 K/V（llama.cpp `dflash.cpp:132-196` 的 `ubatch.embd` 分支）。
3. **query-block forward**：一次前向处理 `anchor + N×MASK`（`MASK` = `parallel_drafting_token_id` 噪声 token），attention 为非因果、看 `[committed, MASK...]`。
4. **采样**：DFlash 默认 `sample_from_anchor=false` —— anchor 是 bonus token（由 target 给出），只有 N 个 MASK 位置各自预测**自己位置**的 token → 产出 N 个草稿 token（vLLM `_prepare_dflash_inputs_kernel` 的 `is_sample = query_off >= 1`）。

### 2.2 DSPark（复用 DFlash 主干）

- `sample_from_anchor=true`：每请求只发 `N` 个 query（anchor + N-1 noise），**anchor 也预测第一个草稿 token**，所有位置采样且 `sample_pos = query_pos + 1`（标准 next-token）。
- **顺序 Markov head**（DFlash 无）：backbone 输出后从左到右逐位置采样，每步把"上一采样 token"经 `markov_embed` + `markov_bias` 注入 logits，注入块内依赖（vLLM `dspark/speculator.py:_sample_sequential`）。
- 可选 **reduced draft vocab**：draft 模型用缩小词表，`draft_id_to_target_id` 映射回 target vocab（DFlash encoder 输出 logits 后 scatter）。

### 2.3 与已有 MTP 的对比

| | MTP (已有) | DFlash | DSPark |
|---|---|---|---|
| 草稿来源 | target 的 nextn 头 | 独立草稿 Transformer | 同 DFlash |
| 输入特征 | 上一层 hidden | target **多层** hidden 拼接 | 同 DFlash |
| 生成 | 自回归 | 一次并行 N token | 并行主干 + 顺序 Markov |
| 权重共享 | 复用 target | 共享 `tok_embd`/`lm_head` | 同 DFlash |

`MtpDraftProvider` 已经示范"草稿源直接复用 target 引擎资源"（`mtp_step` / `take_last_hidden`），DFlash 走同一条路但更重：需独立架构引擎 + target 多层特征 + 独立 KV cache。

---

## 三、Forge 现状与已确认事实

| 能力 | 状态 | 证据 |
|------|------|------|
| `IDraftProvider` 接口（draft/begin/accept/reset/name） | ✅ 已有 | `include/forge/speculative.h:87-110` |
| `verify_draft_tokens`（greedy + resample-consistency） | ✅ 可复用 | `include/forge/speculative.h:234` |
| `SpeculativeExecutor` 按 `cfg_` 顺序 push provider | ✅ 接入点明确 | `src/inference/speculative_executor.cpp:26-54` |
| `ModelDraftProvider` 内部独立 `InferenceContext` 加载 draft 模型 | ✅ 模板 | `src/inference/model_draft_provider.cpp:35-51` |
| `ModelWeights` 暴露 `token_embedding`/`output_weight`/`output_norm` | ✅ 共享权重可行 | `include/forge/model.h:239-243` |
| `TransformerEngine::weights()` 返回 `ModelWeights&` | ✅ 可访问 target 权重 | `include/forge/engines/transformer_engine.h:93` |
| `dflash` / `dspark` 架构支持 | ❌ 零命中 | 全仓搜索 `dflash\|dspark\|target_layers` = 0 |
| `take_last_hidden` 仅最后一层的 hidden | ❌ 需多层导出 | `transformer_engine.h:77-81` |
| `ForwardRequest` 仅 `input_ids` / `from_hidden` | ❌ 需 embedding 入口 | `include/forge/inference/forward_request.h:19-61` |
| attention 是否支持非因果/自定义 mask | ❓ 待探针（阶段 2） | 见第六节 A |

---

## 四、总体设计

```
Generator
  └─ SpeculativeExecutor (src/inference/speculative_executor.cpp)
        └─ providers_ (向量, 按 cfg_ 顺序)
              ├─ MtpDraftProvider        (已有)
              ├─ ModelDraftProvider       (已有)
              ├─ Ngram*/NgramMod          (已有)
              └─ DFlashDraftProvider ──── (新增, is_dspark_ 控制 DSpark 变体)
                    └─ 自持 DflashEngine (独立 InferenceContext, 独立 KV cache)
                          ├─ 引用 target engine: 取 take_layer_hiddens(target_layers_) + 共享 weights().token_embedding / output_weight
                          ├─ encoder: 多层 hidden → fc → output_norm_enc → context feature
                          ├─ decoder: [anchor + N×MASK] 并行 forward
                          └─ 可选 markov_embed / markov_bias (DSPark)
```

数据流（单轮 `draft()`）：
```
target_engine.forward(承诺前缀)
   → take_layer_hiddens(target_layers_)            // [seq, enc_dim]
   → DflashEngine.encoder(ctx_feature)             // 缓存
   → DflashEngine.precompute_context_kv(prefix_embd) // 注入 draft KV cache
   → DflashEngine.forward([anchor, MASK×N])         // 并行 → logits
   → 采样 N token → 返回 IDraftProvider::draft 结果
verify_draft_tokens(logits_all, draft_tokens, ...)  // 复用, 不改
```

---

## 五、分阶段实施

### 阶段 0 — 架构与配置骨架（可编译, 不改行为）

| 文件 | 动作 |
|------|------|
| `include/forge/speculative.h` | `SpeculativeConfig` 增字段：`std::string draft_arch;`（`""`/`"dflash"`/`"dspark"`）、`std::vector<int> draft_target_layers;`、`int draft_mask_token_id = -1;`、`int draft_n_spec = 5;` |
| `src/model/arch_config_parser.cpp` | 新增 `dflash` ConfigParser（复用 llama 解析，额外读 `target_layers` 数组、enc 维度 `n_embd_inp_enc`），`FORGE_REGISTER_CONFIG_PARSER("dflash", ...)` |
| `src/inference/arch_registrations.cpp` | `FORGE_REGISTER_ENGINE("dflash", ...)`（引擎类在阶段 2 落地，先占位返回 nullptr 或最小 stub） |
| `src/cli/cli_args.cpp` + `src/cli/forge_cli.cpp` | 新增 `--spec-draft-arch dflash\|dspark`、`--spec-draft-target-layers "2,5,8"`、`--spec-draft-mask-id N`、`--spec-draft-n N` |
| `src/bindings/core_types.cpp` | 绑定 `draft_arch` 等新字段 |

验收：`cmake --build build -j` 通过；`--spec-draft-arch dflash` 能解析但不触发实际 draft（provider 未注册时打日志跳过）。

### 阶段 1 — target 多层特征导出（前置件）

| 文件 | 动作 |
|------|------|
| `include/forge/engine.h` | `InferenceEngine` 增虚接口：`virtual TensorPtr take_layer_hiddens(const std::vector<int>& layers) { return nullptr; }` 与 `virtual bool captures_layer_hiddens() const { return false; }` |
| `include/forge/engines/transformer_engine.h` | 增成员 `std::vector<TensorPtr> layer_hiddens_;`；`captures_layer_hiddens() override`；`take_layer_hiddens(layers)` 拼接选定层为 `[seq, Σhidden]` |
| `src/inference/engines/transformer_engine.cpp` | `forward_layers` 内每层后（当 `captures_layer_hiddens()`）缓存隐藏态；清理 `reset()`；`take_layer_hiddens` 按 `target_layers` 顺序 concat 并返回 |
| `include/forge/inference/forward_request.h` | （可选，阶段 2 用）增 `TensorPtr input_embeddings;` + `static ForwardRequest from_embedding(...)` |

验收：加单测 —— 同一前缀下 `take_layer_hiddens({0,2})` 与逐层拦截的 hidden 数值一致；不开启 capture 时无开销。

### 阶段 2 — DFlash 引擎 + embd batch + 共享权重

| 文件 | 动作 |
|------|------|
| `include/forge/inference/forward_request.h` | 加 `TensorPtr input_embeddings;` + `bool from_embedding=false;` + `from_embedding(embd, n, start_pos, seq_id)` |
| `include/forge/engines/dflash_engine.h` | 新建 `class DflashEngine : public TransformerEngine`（或 `InferenceEngine`）。持有 `InferenceEngine* target_`；权重 `fc`、`output_norm_enc`、`output_norm`、每层 `attn_norm/wq/wk/wv/wo/ffn_*` |
| `src/inference/engines/dflash_engine.cpp` | 实现：① `encoder(ctx_hiddens)`；② `precompute_context_kv(prefix_embd)`（投影 K/V 注入自身 KV cache，复用 target RoPE/位置）；③ `forward_request` 双模式（embd batch 注入 / token batch 并行 mask attention）；④ `output` 取自 `target_->weights().output_weight` |
| `src/model/gguf_model.cpp` | dflash 架构权重映射（`fc` / `output_norm_enc` / `output_norm` / 各层）与 `target_layers` 元数据读取 |
| `src/inference/graph/...` 或 `src/operators/...`（attention） | **探针 A**：确认/改造 attention 以支持 cache-aware 非因果 mask（DFlash decoder 的 `[committed, MASK...]` 窗口） |

接口签名（DFlashEngine 关键方法）：
```cpp
class DflashEngine {
public:
    DflashEngine(Model& model, InferenceContext& ctx, InferenceEngine* target);
    // encoder: target 多层 hidden -> context feature [seq, n_embd]
    TensorPtr encode(const TensorPtr& target_layer_hiddens);
    // context-KV 预计算: 把 target token embedding 投影注入自身 KV cache
    void precompute_context_kv(const TensorPtr& prefix_embd, int64_t start_pos);
    // decoder: [anchor, MASK×N] 并行 forward -> [N+1, vocab] logits
    TensorPtr decode(const std::vector<int32_t>& query_ids, int64_t start_pos);
    bool is_dspark() const;
};
```

验收：单测 —— encoder 输出维度 `n_embd`、context-KV 注入后 draft 首层 K/V 与 target 投影值一致；小模型端到端能产出 N 个候选 token（greedy 校验）。

### 阶段 3 — DFlashDraftProvider（草稿逻辑）

| 文件 | 动作 |
|------|------|
| `include/forge/speculative.h` | 声明 `class DFlashDraftProvider : public IDraftProvider` |
| `src/inference/dflash_draft_provider.cpp` | 新建，实现 `IDraftProvider`：内部 `make_context(dflash_model, target_params, gpu_layers)`（仿 `ModelDraftProvider::make_context`），持有 `DflashEngine` |

接口签名：
```cpp
class DFlashDraftProvider : public IDraftProvider {
public:
    DFlashDraftProvider(InferenceContext& ctx, InferenceEngine& target,
                        float p_min, int n_spec, bool is_dspark);
    bool valid() const;
    void begin(const std::vector<int32_t>& prompt) override;
    std::vector<int32_t> draft(int32_t last_token, int n_draft) override;
    void accept(const std::vector<int32_t>& tokens) override;
    void reset() override;
    const char* name() const override { return is_dspark_ ? "dspark" : "dflash"; }
private:
    std::unique_ptr<InferenceContext> draft_ctx_;  // 持有 DflashEngine + 独立 KV cache
    std::vector<int> target_layers_;
    int mask_token_id_;
    int n_spec_;
    bool is_dspark_;
};
```

`draft()` 流程：
```
1. h = target_.take_layer_hiddens(target_layers_);          // 承诺前缀多层 hidden
2. ctx = eng.encode(h);                                      // 缓存 context feature
3. eng.precompute_context_kv(prefix_embd, committed_len_);   // 注入 draft KV
4. logits = eng.decode({bonus, MASK×N}, committed_len_);     // 并行
5. 采样 N token（DFlash: MASK 位各预测自己位置; DSPark: sample_from_anchor）
6. return tokens
```

`accept()`：同 `ModelDraftProvider::accept`——回滚 draft KV 到承诺前缀，重放确认 token（保持下轮前缀一致）。

### 阶段 4 — DSPark 变体

| 文件 | 动作 |
|------|------|
| `src/inference/engines/dflash_engine.cpp` | 加 `markov_embed` / `markov_bias` 权重（来自 dflash GGUF 的 DSpark 变体）；`decode` 支持 `sample_from_anchor`（N query 预测 N token）；新增 `sample_sequential(logits, prev_token)` |
| `src/inference/dflash_draft_provider.cpp` | `is_dspark_` 时走 `sample_from_anchor` + 顺序 Markov 采样；reduced-vocab scatter（若 `draft_id_to_target_id` 存在） |

对齐 vLLM `dspark/speculator.py:_sample_sequential`：从左到右每个位置 `logits_i = base_logits[:,i] + markov_bias(markov_embed(prev))`，再 Gumbel/argmax 采样，`prev = sampled`。

### 阶段 5 — 接入 / CLI / 验证

| 文件 | 动作 |
|------|------|
| `src/inference/speculative_executor.cpp` | provider 注册顺序加分支（紧接 MTP 之后）：`if (cfg_.draft_arch=="dflash"||"dspark")` push `DFlashDraftProvider(ctx_, *ctx_.engine(), cfg_.p_min, cfg_.draft_n_spec, dsark)`，`valid()` 失败打日志跳过 |
| `src/cli/forge_cli.cpp` | 把 `--spec-draft-arch` 等填入 `SpeculativeConfig` |
| `tests/test_dflash.cpp` | 单测：encoder 维度、context-KV 注入一致性、端到端接受率 > 0；greedy 下开启 DFlash 与关闭 spec 文本逐 token 一致 |

---

## 六、探针结论（已验证，2026-08-30）

- **A. attention 非因果 mask —— ✅ 可行，无需改 kernel**
  - `src/operators/cuda/cuda_flash_attn.h:9-13`：`launch_flash_attention(..., const float* mask=nullptr, bool causal=true, ...)` 与 `launch_flash_attention_gqa` 均接受自定义 mask 且 `causal` 可关。
  - `src/inference/attention_mask.cpp:14`：`AttentionMaskBuilder::build_kq_mask(..., bool causal, ...)` 支持任意 mask 构造（line 66-69 的因果屏蔽逻辑可反转为非因果）。
  - DFlash decoder 只需以 `causal=false` + 自构建 `[N+1, kv_len]` mask（prefix 全 0 + block 内非因果）调用 attention；底层能力齐备。
  - 注：`build_kq_mask` 当前基于 `InferenceBatch` 序列偏移假设；DFlash engine 需自建 mask（直接构造全 0 或带 block 边界的 mask），**不改 kernel**。

- **B. embd batch K/V 注入 —— ✅ 可行**
  - `forward_layers(hidden, req)`（`src/inference/engines/transformer_engine.cpp:776`）入口即 hidden state；`forward_request` 在 `ops::embedding` 后注入（line 351-380）。加 `ForwardRequest::from_embedding(embd, n, start_pos, seq_id)` 即可跳过 embedding lookup，直接 `forward_layers(input_embeddings, req)`。
  - `KVCache::update(layer, seq_id, pos, K, V, seq_len)`（`include/forge/kv_cache.h:200-201`）提供按位置写 K/V 接口。DFlash 的 context-KV 预计算 = 遍历 target prefix 位置，每层用自身 `wk/wv` 投影 prefix embedding → RoPE → `kv_cache_->update(layer, 0, pos, K, V, 1)`。
  - 改造点均局部、低侵入：① `forward_request.h` 加 `input_embeddings` + `from_embedding`；② `DflashEngine` 实现 `precompute_context_kv`（复用自身 wk/wv + RoPE + kv_cache::update）。

- **C. 权重 —— ✅ 已下载并校验通过**
  - 来源：ModelScope `AI-ModelScope/Qwen3.6-35B-A3B-DFlash-GGUF-Test`（国内镜像 wget）。
  - 文件：`/mnt/g/AI/Qwen3.6-35B-A3B-DFlash-q8_0.gguf`（421,060,320 B，SHA256 `7847c17a…0e64`，与官方一致）。

  **实测规格**（`forge-inspect -m`，2026-08-30）：

  | 项 | 值 | 含义 |
  |----|----|------|
  | `general.architecture` | `dflash` | 草稿模型架构 |
  | `dflash.target_layers` | `[2, 7, 12, 17, 23, 28, 33, 38]` | **8 个 target 层** → encoder 输入宽 `8 × 2048 = 16384` |
  | `dflash.block_count` | 6 | 草稿仅 6 层（很轻） |
  | `dflash.embedding_length` | 2048 | 草稿 hidden |
  | `attention.head_count` / `_kv` | 32 / 8 | GQA |
  | `attention.key_length` / `value_length` | 128 / 128 | head_dim |
  | `attention.sliding_window` | 4096 | **iSWA** |
  | `attention.sliding_window_pattern` | `[1,1,1,1,1,0]` | 5 层 SWA + 1 层 full |
  | `dflash.context_length` | 262144 | |
  | `dflash.rope.freq_base` | 1e7 | |

  **两个关键结论（直接影响阶段 2/3 实现）**：

  1. **草稿模型不含 `token_embd` / `output` 张量**（69 tensors 只有 `fc` + 6 层 + norms）。单独 `forge-cli -m <dflash.gguf> --info` 会报 `Failed to load token embedding` —— 这是 DFlash 的设计（共享 target 的 embedding 与 lm_head，llama.cpp 走 `ctx_other`），不是解析 bug。因此：
     - `DflashEngine` 必须**配对 target 加载**，不能沿用 `ModelDraftProvider` 的独立加载路径；
     - 阶段 3 的加载流程需跳过 embedding/output 的必需性检查，并从 target 借用这两个张量。
  2. **iSWA（混合滑动窗口）**：草稿层分 SWA / full 两类，阶段 2 引擎需按 `sliding_window_pattern` 路由 K/V（对齐 llama.cpp `dflash.cpp` 的 `use_iswa` 分支）。
  - 端到端验证还需配对 **target 模型**（Qwen3.6-35B-A3B，35B MoE；当前环境 RAM 7.4G / GPU 6G 跑不动）。可改用手头小 target（如 `/mnt/g/AI/Qwen3.8-27B`）或更大环境验证。

---

## 七、验收与回归

- 每阶段：`cmake --build build -j && ctest --test-dir build` 通过（已注册测试不退化）。
- 阶段 3/5 端到端：`./build/forge-cli -m <target> --spec-draft-arch dflash --spec-draft-model <dflash.gguf>` 能产出草稿并被验证；`--spec-stats` 打印接受率。
- greedy 模式下，开启 DFlash 与关闭 spec 的生成文本**逐 token 一致**（resample-consistency 保证分布无损）。
- 不设置 `--spec-draft-arch` 时，行为与现状完全不变（所有改动对默认路径零影响）。
