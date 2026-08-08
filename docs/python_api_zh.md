# Python API

[English](python_api.md)

> **版本**: forge 0.5.0
> **最后更新**: Phase 6 完成后 (2026-08)
>
> 本文档与 `src/bindings/` 中的实际 pybind11 绑定保持一致。
> 如发现不一致，以本文档为准。

## 快速示例

```python
import forge

# 1. 加载 GGUF 模型
model = forge.Model()
model.load_gguf("model.gguf", device="cuda")

# 2. 加载 tokenizer
tokenizer = forge.Tokenizer()
tokenizer.load_from_gguf("model.gguf")

# 3. 编码 → 生成 → 解码
tokens = tokenizer.encode("你好，世界！")
result = model.generate(tokens, max_new_tokens=128)
print(tokenizer.decode(result["token_ids"]))
```

## 核心类

### Model

```python
model = forge.Model()

# GGUF 格式加载
model.load_gguf("model.gguf", device="cuda")              # device: "cuda" | "cpu"
model.load_gguf("model.gguf", device="cuda", quant_policy=quant)  # 支持量化策略

# 自动检测格式加载
model.load_auto("model.gguf", device="cuda")

# NINF 格式加载（用于测试/自定义模型）
model.load(path, vocab_size=100, hidden_dim=32, intermediate_dim=64,
           num_layers=1, num_heads=2, num_kv_heads=1, head_dim=16,
           rope_theta=10000.0, rms_norm_eps=1e-6, max_seq_len=4096,
           arch_type="llama", norm_type="rmsnorm", activation="silu_gelu",
           tie_embeddings=False, device="cpu",
           n_swa=0, swa_layers=[])

# 视觉编码器权重（多模态）
model.load_vision_weights("mmproj.gguf", device="cpu")

# 属性
model.config          # ModelConfig — 模型配置
model.device          # 设备类型字符串
```

| 方法 | 返回 | 说明 |
|------|------|------|
| `load_gguf(path, device, quant_policy)` | — | 从 GGUF 文件加载模型 |
| `load_auto(path, device)` | — | 自动检测格式并加载 |
| `load(path, vocab_size, ...)` | — | 从 NINF 文件加载（参数见上表） |
| `load_vision_weights(path, device)` | — | 加载多模态视觉编码器权重 |
| `create_context(kv_cache_dtype, gpu_layers)` | `InferenceContext` | 创建推理上下文 |
| `generate(prompt_ids, ...)` | `dict` | 同步生成 |
| `generate_stream(prompt_ids, callback, ...)` | — | 流式生成（回调模式） |
| `registered_archs()` | `list[str]` | 已注册的模型架构列表 |
| `detect_format(path)` | `str` | 检测模型文件格式 |

**`generate()` 完整签名：**

```python
result = model.generate(
    prompt_ids,          # list[int] / numpy array — 输入 token 序列
    max_new_tokens=256,  # 最大生成 token 数
    temperature=1.0,     # 采样温度
    top_k=0,             # Top-K 采样（0 = 禁用）
    top_p=1.0,           # Top-P（核）采样
    repeat_penalty=1.0,  # 重复惩罚
    do_sample=True,      # 是否采样（False = 贪婪）
    seed=0,              # 随机种子
    eos_token_id=-1,     # 停止 token ID（-1 = 忽略）
    kv_cache_dtype="fp32", # KV cache 数据类型: "fp32"|"fp16"|"q8_0"|"q4_0"|"q4_k"
    gpu_layers=-1,       # GPU 层数（-1 = 全部）
    stop_token_ids=[],   # 额外的停止 token ID 列表
)
```

**返回值 `dict` 字段：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `token_ids` | `list[int]` | 全部 token（prompt + 生成） |
| `num_prompt_tokens` | `int` | prompt 的 token 数量 |
| `num_generated_tokens` | `int` | 生成的 token 数量 |
| `finished` | `bool` | 是否正常完成 |
| `finish_reason` | `str` | 完成原因（如 "eos"/"length"/"stop"） |

**注意**: 每次调用 `generate()` 都会**重新创建** `InferenceContext`，无法跨调用复用 KV cache。
如需 KV 复用，请使用 `create_context()` + `ctx.forward()` 手动驱动推理循环。

**`generate_stream()` 完整签名：**

```python
model.generate_stream(
    prompt_ids,           # 输入 token 序列
    callback,             # callable(token_id: int, step: int) — 每生成一个 token 调用
    max_new_tokens=256,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    repeat_penalty=1.0,
    do_sample=True,
    seed=0,
    eos_token_id=-1,
    kv_cache_dtype="fp32",
    gpu_layers=-1,
    stop_token_ids=[],
)
```

### InferenceContext

```python
ctx = model.create_context(kv_cache_dtype="fp32", gpu_layers=-1)
```

| 方法 | 说明 |
|------|------|
| `forward(input_ids, start_pos=0)` | 单次前向传播 |
| `forward_sample(input_ids, start_pos, temperature, top_k, top_p, repeat_penalty, token_history)` | 前向传播 + 采样一步 |
| `forward_with_embeddings(embeddings, start_pos=0)` | 以 embedding 作为输入的前向传播（多模态） |
| `get_embeddings(input_ids)` | 获取 token 的 embedding 向量 |
| `reset_kv()` | 清空 KV cache |
| `reset()` | 完全重置 context 状态 |
| `warmup()` | 预热 CUDA kernel / graph |
| `set_gpu_layers(n)` | 设置 GPU 卸载层数 |
| `gpu_layers()` | 获取当前 GPU 卸载层数 |
| `set_use_graph(use_graph)` | 启用/禁用 CUDA graph |
| `use_graph()` | CUDA graph 是否启用 |
| `set_cuda_graph_enabled(v)` | 启用/禁用 CUDA graph 路径 |
| `cuda_graph_enabled()` | CUDA graph 路径是否启用 |
| `memory_stats()` | 获取当前内存统计信息 |

| 属性 | 类型 | 说明 |
|------|------|------|
| `n_batch` | `int` | 批处理大小 |
| `n_ubatch` | `int` | 微批处理大小 |
| `n_threads` | `int` | CPU 线程数 |
| `n_threads_batch` | `int` | 批处理时的 CPU 线程数 |
| `device` | `str` | 设备类型 |

### Tokenizer

```python
tokenizer = forge.Tokenizer()
tokenizer.load_from_gguf("model.gguf")
```

| 方法 | 说明 |
|------|------|
| `encode(text, add_bos=True, add_eos=False, add_dummy_prefix=True)` | 将文本编码为 token ID 列表 |
| `decode(ids, skip_special=True, strip_leading_space=True)` | 将 token ID 列表解码为文本 |
| `decode_token(id)` | 将单个 token ID 解码为文本片段 |
| `token_to_id(token)` | token 字符串 → ID |
| `id_to_token(id)` | ID → token 字符串 |
| `token_score(id)` | 获取 token 的分数 |
| `token_type(id)` | 获取 token 的类型 |

| 属性 | 类型 | 说明 |
|------|------|------|
| `vocab_size` | `int` | 词表大小 |
| `bos_token_id` | `int` | BOS token ID |
| `eos_token_id` | `int` | EOS token ID |
| `pad_token_id` | `int` | PAD token ID |
| `unk_token_id` | `int` | UNK token ID |
| `model_type` | `TokenizerModelType` | 分词器模型类型 (SPM / BPE) |
| `chat_template` | `str` | GGUF 元数据中的 chat template 字符串 |
| `is_loaded` | `bool` | 是否已加载 |

### MultimodalModel

```python
mm = forge.MultimodalModel()

# 方式 1: 仅加载文本模型
mm.load("model.gguf", device="cuda")

# 方式 2: 加载文本模型 + 视觉编码器
mm.load_with_mmproj("model.gguf", "mmproj.gguf", device="cuda")

# 属性
mm.config           # ModelConfig
mm.vision_config    # VisionConfig

# 图像编码
embeddings = mm.encode_image(image)   # image: numpy (H,W) / (H,W,3) / (H,W,4) uint8
# 返回: numpy float32 (num_tokens, projection_dim)

# 生成（与 Model 相同签名）
result = mm.generate(prompt_ids, max_new_tokens=128)
mm.generate_stream(prompt_ids, callback, max_new_tokens=128)

# 创建 context（多模态可用 forward_with_embeddings 注入图像）
ctx = mm.create_context(kv_cache_dtype="fp32", gpu_layers=-1)
```

### RequestScheduler

使用分页 KV 缓存进行多请求批量调度。

```python
scheduler = forge.RequestScheduler(model, block_size=16, max_num_seqs=4)
cfg = forge.SamplerConfig(do_sample=False)

# 提交请求
scheduler.submit(input_tokens, max_new_tokens=128, eos_token_id=-1, sampler_config=cfg)

# 调度循环
while scheduler.has_pending():
    scheduler.step()
    finished = scheduler.get_finished()  # 返回 list[GenerateRequest]
    for req in finished:
        print(f"Request {req.request_id}: {req.output_tokens}")

# 属性
scheduler.n_batch        # 批处理大小
scheduler.n_ubatch       # 微批处理大小
scheduler.n_threads      # CPU 线程数
scheduler.n_threads_batch # 批处理时的 CPU 线程数
scheduler.prefix_cache_hits   # 前缀缓存命中次数
scheduler.prefix_cache_misses # 前缀缓存未命中次数

# 方法
scheduler.num_active()   # 活跃请求数
scheduler.num_waiting()  # 等待中的请求数
scheduler.has_pending()  # 是否有待处理的请求
scheduler.abort(request_id)  # 中止请求
scheduler.reset()        # 重置调度器
scheduler.memory_stats() # 内存统计（Phase 6+）
```

**`GenerateRequest` 只读属性：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `request_id` | `int` | 请求 ID |
| `status` | `RequestStatus` | 当前状态 |
| `output_tokens` | `list[int]` | 全部输出 token |
| `num_generated` | `int` | 已生成的 token 数量 |
| `finish_reason` | `str` | 完成原因 |
| `prefix_len` | `int` | 命中的前缀长度 |
| `from_cache` | `bool` | 是否来自缓存 |

**`RequestStatus` 枚举**：`Waiting`, `Prefilling`, `Decoding`, `Finished`, `Failed`

**`SamplerConfig` 构造参数**：

```python
SamplerConfig(
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    repeat_penalty=1.0,
    repeat_last_n=64,
    do_sample=True,
    seed=0,
)
# 额外属性（读/写）
sampler_config.logit_softcapping  # float
```

### CachedPrompt

前缀缓存命中时返回的对象。

| 字段 | 类型 | 说明 |
|------|------|------|
| `tokens` | `list[int]` | 缓存的前缀 token |
| `seq_id` | `int` | 序列 ID |
| `valid` | `bool` | 是否有效 |

---

## 枚举

| 枚举 | 值 |
|------|-----|
| `DataType` | `FP32`, `FP16`, `Q4_0`, `Q4_1`, `Q4_K`, `Q5_0`, `Q5_1`, `Q2_K`, `Q3_K`, `Q5_K`, `Q6_K`, `Q8_0`, `INT8`, `INT32`, `IQ2_S`, `IQ2_XXS`, `IQ4_NL`, `IQ2_XS`, `IQ3_S`, `BF16` |
| `DeviceType` | `CPU`, `CUDA` |
| `KVCacheDType` | `FP32`, `F16`, `Q8_0`, `Q4_0`, `Q4_K` |
| `KVLayerPolicy` | `None_`, `Full`, `SlidingWindow`, `Recurrent` |
| `LogLevel` | `NONE`, `LOG_ERROR`, `WARN`, `INFO`, `DEBUG`, `TRACE` |
| `NormType` | `RMSNorm`, `LayerNorm` |
| `ActivationType` | `SiLU_GELU`, `GELU`, `ReLU` |
| `RopeType` | `None`, `Standard`, `LinearScaling`, `NTK_Scaled`, `NeoX`, `MRoPE`, `Proportional` |
| `FFNType` | `SiLUGated`, `GeGLU`, `SimpleGELU`, `MoE` |
| `TokenizerModelType` | `SPM`, `BPE` |
| `RequestStatus` | `Waiting`, `Prefilling`, `Decoding`, `Finished`, `Failed` |

---

## 配置类型

### ModelConfig

```python
cfg = model.config
# 字段: vocab_size, hidden_dim, intermediate_dim, num_layers, num_heads,
# num_kv_heads, head_dim, rope_theta, rms_norm_eps, max_seq_len, arch_type,
# tie_embeddings, use_gqa, use_neox_rope, norm_type, ffn_activation,
# use_ssm, ssm_group_count, ssm_time_step_rank, ssm_inner_size,
# ssm_state_size, ssm_conv_kernel, full_attention_interval,
# rope_dimension_count, use_mrope, f_attn_logit_softcapping,
# f_final_logit_softcapping, use_parallel_residual, n_embd_per_layer,
# n_ff_exp, n_expert, n_expert_used, n_swa, n_layer_kv_from_start,
# use_qk_norm, head_dim_swa, num_heads_swa, num_kv_heads_swa,
# suppress_tokens, rope_type, ffn_type, rope_q_scale,
# has_post_attention_norm, has_post_ffn_norm
```

### QuantPolicy

```python
quant = forge.QuantPolicy()
quant.default_type    # DataType  — 默认量化类型
quant.attn_wv_type    # DataType  — 注意力 W/V 量化类型
quant.ffn_down_type   # DataType  — FFN down 层量化类型
quant.output_type     # DataType  — 输出层量化类型
quant.enabled()       # bool — 是否启用量化

# 预定义策略
QuantPolicy.q4_k_m()  # Q4_K_M 推荐策略
```

### SpeculativeConfig

```python
spec = forge.SpeculativeConfig()
spec.n_draft       # int — draft token 数量
spec.p_min         # float — 最低接受概率
spec.use_ngram     # bool — 使用 n-gram draft
spec.ngram_n       # int — n-gram N
spec.ngram_min     # int — n-gram 最小长度
spec.enabled       # bool — 是否启用推测解码
```

---

## 工具函数

```python
# 线程控制
forge.set_num_threads(8)

# dtype 查询
forge.dtype_size(dt)            # 元素大小（字节）
forge.dtype_name(dt)            # 可读的 dtype 名称
forge.dtype_block_size(dt)      # 量化块大小
forge.dtype_block_elements(dt)  # 每块的元素数
forge.is_quantized_type(dt)     # 是否是量化类型
forge.compute_quantized_bytes(numel, dt)  # 计算量化后的字节数

# 内存计数器
forge.get_memory_counters()     # 返回 dict: cpu_malloc, cpu_free, cuda_malloc, ...
forge.reset_memory_counters()   # 归零所有计数器
```

---

## 其他工具类

### Tensor

支持 CPU 和 CUDA 存储的多维数组。

```python
t = forge.Tensor(forge.DataType.FP32, [1024, 4096], forge.DeviceType.CUDA)
t.shape              # list[int]
t.dtype              # DataType
t.device             # DeviceType
t.numel()            # 总元素数
t.nbytes()           # 总字节数
t.strides()          # 步长
t.zero_()            # 原地置零
t.to_device(dev)     # 跨设备传输
t.copy_from(other)   # 从另一个 Tensor 复制
t.view([...])        # 创建视图
t.slice(dim, start, end)  # 切片
t.numpy()            # 转为 numpy 数组（仅 CPU）

# 从 buffer 创建
Tensor.from_buffer(ptr, dtype, shape, device, own)
```

### BackendManager

```python
backend = forge.BackendManager.instance()
backend.available_backends()   # list[str]
backend.has_cuda()             # bool
```

### Logger

```python
logger = forge.Logger.instance()
logger.set_level(forge.LogLevel.DEBUG)
```

### PerfProfiler

```python
profiler = forge.PerfProfiler.instance()
profiler.enable()
# ... 运行推理 ...
profiler.disable()
print(profiler.summary())
```

---

## 实验特性（通过环境变量控制）

以下功能当前仅通过环境变量控制，为实验特性。后续版本将通过正式 Python API 暴露。

| 环境变量 | 值 | 说明 | 引入版本 |
|----------|-----|------|----------|
| `FORGE_KV_STORAGE_MODE` | `"paged"` / 不设置 | 启用分页 KV 存储（默认 contiguous）。paged 模式支持 prefix cache、页级内存管理和批量调度。 | Phase 5 |
| `FORGE_KV_AUTO_POLICY` | `"1"` / 不设置 | 启用自动策略选择：CUDA decode → F16+paged，CPU decode → F16/Q8_0，长上下文 → paged。默认关闭，保持 FP32 contiguous。 | Phase 6 |

**与后续 `ContextConfig` 的对冲规则**：
- 环境变量仅在 `ContextConfig` 未显式指定对应字段时作为 fallback。
- 一旦 Python 侧通过 `ContextConfig` 指定 `kv_storage` / `kv_policy` 等参数，环境变量将被忽略。
- 环境变量仅作为开发环境快速切换入口，不建议在生产代码中依赖。
