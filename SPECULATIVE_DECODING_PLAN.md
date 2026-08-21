# Forge 投机解码增强计划(Speculative Decoding Plan)

> 参考实现:llama.cpp(`/mnt/f/My__StudyStack/From_Github/llama.cpp`)
> 关键参考文件:`common/speculative.{h,cpp}`、`common/sampling.cpp`(common_sampler_sample_and_accept_n)、
> `examples/speculative-simple/speculative-simple.cpp`、`examples/speculative/speculative.cpp`(stochastic 拒绝采样)、
> `docs/speculative.md`

---

## 一、现状分析

Forge 已具备:

| 组件 | 位置 | 状态 |
|------|------|------|
| n-gram 自投机最小实现 | `include/forge/speculative.h` + `src/inference/speculative.cpp` | O(n²) 暴力匹配 |
| 投机主循环集成 | `src/inference/generator.cpp:141-238` | draft→verify→rollback 已通,goto 跳转 |
| KV 回滚 API | `include/forge/kv_cache.h`(`rollback`/`seq_rm`/`seq_cp`/`seq_keep`) | llama.cpp 风格,已就绪 |
| 多 token 前向 | `InferenceEngine::forward_batch` + `InferenceBatch` | 各引擎行为需对照验证 |

主要缺口:

1. **正确性**:`verify_draft_tokens` 仅 greedy argmax 匹配;`do_sample=true`(temperature/top_p/top_k 生效)时输出分布不保真。
   llama.cpp 解法:**重采样一致性验证**——target 在每个位置走完整采样链,采出的 token 与 draft 相同才接受,
   数学上保证输出仍严格服从 target 分布(`common_sampler_sample_and_accept_n`,sampling.cpp:635)。
2. **性能**:n-gram 每轮全量扫描历史(O(history×n));llama.cpp 用 hash map 索引(ngram-map)。
3. **能力**:仅有 n-gram draft;无独立小模型 draft(DRAFT_SIMPLE)、无 MTP/EAGLE3。
4. **接入**:CLI / Python bindings 均未暴露参数;`SpeculativeConfig` 只能代码内设置。
5. **质量**:无测试、无接受率统计。

---

## 二、架构设计(先立骨架,再填肉)

```
Generator (编排,瘦身)
  └── SpeculativeExecutor (新控制器, src/inference/speculative_executor.*)
        ├── IDraftProvider (扩展现有 DraftModel 接口)
        │     ├── NgramDraftProvider   —— hash map 索引 (Phase 2)
        │     ├── ModelDraftProvider   —— 独立小模型 (Phase 3)
        │     └── MtpDraftProvider     —— DeepSeek nextn (Phase 5)
        ├── Verifier —— greedy 快路径 + 重采样一致性 (Phase 1)
        ├── 能力探测 can_speculate(engine) —— SSM/MLA 回滚约束
        └── SpeculativeStats —— 接受率统计 (Phase 4)
```

关键架构决策:

- **`IDraftProvider` 抽象基类**(对齐 llama.cpp `common_speculative_impl`):
  `begin()/draft()/accept()/reset()`。新增 draft 来源不改 Generator。
- **能力探测**(对齐 llama.cpp `common_context_can_seq_rm`):
  Qwen3.5 hybrid SSM 的 recurrent 状态无法按位置回滚 → `can_speculate()==false` 时自动降级普通解码并打日志。
- **配置归属**:`SpeculativeConfig` 留在 `ContextParams`(ctx 生命周期);
  `GenerationConfig` 不动;per-request 开关走 `enabled` 位。
- **验证策略**:采用重采样一致性(temp>0 时每个位置走完整采样链,采出==draft 则接受),
  不做拒绝采样(p_target/p_draft residual 方案),复杂度高且收益有限。

---

## 三、分阶段实施

### Phase 0 — 架构抽取(不改行为)

| 文件 | 动作 |
|------|------|
| `include/forge/speculative.h` | 重构:`IDraftProvider` 接口 + `Verifier` 声明 + `SpeculativeStats`;`NgramDraftModel`→`NgramDraftProvider`;`SpeculativeConfig` 增字段 |
| `src/inference/speculative_executor.h/.cpp` | 新建:从 `generator.cpp:141-238` 抽出 draft→verify→rollback 流程,消除 goto |
| `generator.cpp` | 主循环改为调 `executor.step()`;normal_decode 逻辑不变 |

### Phase 1 — 验证算法修正(正确性核心)

1. `Sampler` 增加零拷贝入口 `sample_from_ptr(const float*, int vocab_size)` 及暴露 penalty→top_k→top_p 链的
   `sample_consistent()`(当前每位置 memcpy 整个 vocab 行,speculative.cpp:77-80)。
2. `Verifier::verify()` 双模式:
   - `do_sample=false`:argmax 匹配快路径(现状保留);
   - `do_sample=true`:重采样一致性——每位置走完整采样链,采样==draft 则接受,否则以 target 新采样为准;
     repeat_penalty 按 accepted 顺序喂历史;全接受时末位采 bonus token。
3. 校验回滚位置 `pos + n_accepted + 1` 语义(generator.cpp:189),补注释。
4. **前置验证**:LLaMA/Qwen 引擎 `forward_batch` 多 token logits 逐位正确
   (对照测试:batch 前向 vs 逐 token 前向 logits 一致性)。

### Phase 2 — n-gram 提速

- `NgramDraftProvider`:后缀 key → `unordered_map<key, vector<pos>>` 索引替代暴力扫描(对齐 llama.cpp ngram-map);
- `accept()` 增量插索引;prompt 批量建索引;同 key 多候选取最新出现(可配随机化);
- 单测:建索引/匹配/增量更新正确性 vs 旧暴力实现交叉验证。

### Phase 3 — ModelDraftProvider(独立小模型)

1. `SpeculativeConfig` 增:`draft_model_path`、`draft_gpu_layers`、`draft_p_min`;
2. `ModelDraftProvider` 内部持有独立 `Model + InferenceContext`(自带 KVCache);
   加载时校验 vocab_size/n_vocab 与 target 一致;
3. draft 循环(参考 speculative-simple.cpp):seed=id_last → top-k=10 高置信自回归 ≤n_draft 步,
   p_min 早停,低于 n_min 丢弃;
4. **双侧 KV 同步**:target 回滚后,accepted tokens 回填 draft ctx(`seq_rm` + 重喂),保证下轮 draft 前缀一致;
5. 显存预算写入 docs。

### Phase 4 — API / CLI / Python + 统计 + 测试

- CLI(`cli_args.cpp`):`--spec`、`--spec-draft N`、`--spec-model PATH`、
  `--spec-ngram-n/--spec-ngram-min`、`--spec-p-min`、`--spec-stats`;
- Python(`bindings/core_types.cpp`):绑定 `SpeculativeConfig`;`forge.Model(..., spec=...)` 便捷入口;
- `GenerationResult` 增 optional `SpeculativeStats`(steps/draft 数/接受数/接受率/tokens-per-step);CLI 结束打印;
- 测试 `tests/test_speculative.cpp` + pytest 冒烟:
  verifier 各分支、n-gram 索引、端到端接受率>0 且 greedy 下文本与关闭 spec 时严格一致。

### Phase 5 — DeepSeek MTP(nextn)(探索性)

1. 调研 GGUF 中 `nextn` 权重组织(llama.cpp `llama_get_embeddings_nextn` 对应物);
2. Engine 增加中间层/nextn 隐状态导出接口(`forward_batch` 可选输出);
3. `MtpDraftProvider`:复用 target 权重 + nextn 头做单步高置信 draft,可链式多步;
4. 仅 DeepSeek 引擎启用;能力探测失败即禁用。

---

## 四、风险与对策

| 风险 | 对策 |
|------|------|
| Qwen3.5 SSM 状态不可按位置回滚 | Phase 0 能力探测,自动禁用并日志提示 |
| CUDA Graph 变长 batch 反复重捕获 | 验证 batch 长度固定为 `1+n_draft`(pad) |
| forward_batch 各引擎行为差异(Gemma4 per-seq fallback) | Phase 1 对照测试先行;不支持则该引擎禁用 spec |
| MLA 压缩 latent KV 的回滚语义 | 位置级删除同样适用;Phase 3 实测验证 |

## 五、实施顺序与验收

Phase 0 → 1(含对照测试)→ 2 → 3 → 4 → 5。

每阶段验收:
```bash
cmake --build build -j && ctest --test-dir build          # 或对应测试目标
./build/forge-cli -m <model> --stream                      # 手工冒烟
python -m report.runner                                    # 加速比基准(阶段收尾)
```

greedy 模式下,开启 spec 与关闭 spec 的生成文本必须逐 token 一致(Phase 4 自动化校验)。
