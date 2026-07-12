# Python API

[English](python_api.md)

## 快速示例

```python
import nanoinfer

model = nanoinfer.Model("model.gguf", device=nanoinfer.DeviceType.CUDA)
tokenizer = nanoinfer.Tokenizer("model.gguf")

tokens = tokenizer.encode("你好，世界！")
result = model.generate(tokens, max_new_tokens=128)
print(tokenizer.decode(result))
```

## 核心类

### Tensor

支持 CPU 和 CUDA 存储的多维数组。

```python
# 创建
t = nanoinfer.Tensor([1, 2, 3, 4, 5, 6], shape=[2, 3], dtype=nanoinfer.DataType.FP32)
t = nanoinfer.Tensor.zeros([1024, 4096])                             # 零初始化
t = nanoinfer.Tensor.from_numpy(np_array, device=nanoinfer.DeviceType.CUDA)

# 属性
t.shape              # [2, 3]
t.dtype              # nanoinfer.DataType.FP32
t.device             # nanoinfer.DeviceType.CUDA
t.size               # 总元素数
t.nbytes             # 总字节数

# 操作
t.to_device(nanoinfer.DeviceType.CPU)   # 跨设备传输
t.zero_()                                # 原地置零
t.numpy()                                # 转为 numpy 数组（仅 CPU）
```

### Model

```python
model = nanoinfer.Model("model.gguf", device=nanoinfer.DeviceType.CUDA)
model = nanoinfer.Model.load_auto("model.gguf", device=nanoinfer.DeviceType.CUDA)
```

| 方法 | 说明 |
|------|------|
| `Model(path, device)` | 从 GGUF / NINF 文件加载模型 |
| `load_auto(path, device)` | 自动检测格式并加载 |
| `load_vision_weights(path)` | 加载多模态视觉编码器权重 |
| `config()` | 获取 `ModelConfig` |
| `is_loaded()` | 检查模型是否已加载 |
| `create_context(...)` | 创建 `InferenceContext` |
| `generate(tokens, ...)` | 生成文本（返回 `GenerationResult`） |
| `generate_stream(tokens, ...)` | 流式生成（逐个 yield token） |

**`generate()` 参数：**

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `max_new_tokens` | `256` | 最大生成 token 数 |
| `temperature` | `1.0` | 采样温度 |
| `top_k` | `0` | Top-K 采样（`0` = 禁用） |
| `top_p` | `1.0` | Top-P（核）采样 |
| `repeat_penalty` | `1.0` | 重复惩罚 |
| `do_sample` | `True` | 启用采样（False = 贪婪） |

**`GenerationResult`** 字段：`token_ids`、`text`、`num_prompt_tokens`、`num_generated_tokens`、`finished`。

### InferenceContext

```python
ctx = model.create_context(
    kv_cache_dtype=nanoinfer.KVCacheDType.FP32,
    gpu_layers=-1,
    batch_size=512
)
ctx.forward(input_tokens)              # 单次前向传播
ctx.generate(input_tokens, max_new=128) # 从 prompt 生成
ctx.reset()                            # 重置 KV 缓存
ctx.warmup()                           # 预热 CUDA kernel
```

### Tokenizer

```python
tokenizer = nanoinfer.Tokenizer("model.gguf")
# 或
tokenizer = nanoinfer.Tokenizer.load_from_gguf(model)
```

| 方法 / 属性 | 说明 |
|-------------|------|
| `encode(text)` | 将文本编码为 token ID |
| `decode(tokens)` | 将 token ID 解码为文本 |
| `vocab_size()` | 获取词表大小 |
| `bos_token_id` | BOS token ID |
| `eos_token_id` | EOS token ID |
| `chat_template` | GGUF 元数据中的 chat template 字符串 |

### Generator

```python
gen = nanoinfer.Generator(ctx)
result = gen.generate(tokens, max_new_tokens=128, temperature=0.8)
```

### MultimodalModel

```python
mm = nanoinfer.MultimodalModel("model.gguf", "mmproj.gguf")
result = mm.chat("描述这张图片", image_path="photo.jpg")
```

### Backend

```python
backend = nanoinfer.BackendManager.instance()
print(backend.available_backends())   # ['CPU', 'CUDA']
print(backend.has_cuda())             # True / False
```

### Logger

```python
log = nanoinfer.Logger.instance()
log.set_level(nanoinfer.LogLevel.DEBUG)  # NONE, ERROR, WARN, INFO, DEBUG, TRACE
```

### PerfProfiler

```python
profiler = nanoinfer.PerfProfiler.instance()
profiler.enable()
# ... 运行推理 ...
profiler.disable()
print(profiler.summary())
```

### RequestScheduler

使用分页 KV 缓存进行多请求批量调度：

```python
scheduler = nanoinfer.RequestScheduler(model)
req_id = scheduler.submit(input_tokens, max_new_tokens=128)
while True:
    scheduler.step()
    finished = scheduler.get_finished()
    if req_id in finished:
        result = finished[req_id]
        break
```

## 枚举

| 枚举 | 值 |
|------|-----|
| `DataType` | `FP32`, `FP16`, `Q4_0`, `Q4_1`, `Q4_K`, `Q6_K`, `Q8_0`, `INT8`, `INT32` |
| `DeviceType` | `CPU`, `CUDA` |
| `KVCacheDType` | `FP32`, `Q4_0` |
| `LogLevel` | `NONE`, `ERROR`, `WARN`, `INFO`, `DEBUG`, `TRACE` |

## 线程控制

```python
nanoinfer.set_num_threads(8)  # 设置 OpenMP 线程数
```
