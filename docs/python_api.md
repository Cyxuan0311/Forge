# Python API

[中文](python_api_zh.md)

## Quick Example

```python
import nanoinfer

model = nanoinfer.Model("model.gguf", device=nanoinfer.DeviceType.CUDA)
tokenizer = nanoinfer.Tokenizer("model.gguf")

tokens = tokenizer.encode("Hello, world!")
result = model.generate(tokens, max_new_tokens=128)
print(tokenizer.decode(result))
```

## Core Classes

### Tensor

Multi-dimensional array with CPU/CUDA storage.

```python
# Create
t = nanoinfer.Tensor([1, 2, 3, 4, 5, 6], shape=[2, 3], dtype=nanoinfer.DataType.FP32)
t = nanoinfer.Tensor.zeros([1024, 4096])                          # Zero-initialized
t = nanoinfer.Tensor.from_numpy(np_array, device=nanoinfer.DeviceType.CUDA)

# Properties
t.shape              # [2, 3]
t.dtype              # nanoinfer.DataType.FP32
t.device             # nanoinfer.DeviceType.CUDA
t.size               # total elements
t.nbytes             # total bytes

# Operations
t.to_device(nanoinfer.DeviceType.CPU)   # Cross-device transfer
t.zero_()                                # In-place zero
t.numpy()                                # Convert to numpy array (CPU only)
```

### Model

```python
model = nanoinfer.Model("model.gguf", device=nanoinfer.DeviceType.CUDA)
model = nanoinfer.Model.load_auto("model.gguf", device=nanoinfer.DeviceType.CUDA)
```

| Method | Description |
|--------|-------------|
| `Model(path, device)` | Load model from GGUF / NINF file |
| `load_auto(path, device)` | Auto-detect format and load |
| `load_vision_weights(path)` | Load multimodal vision encoder weights |
| `config()` | Get `ModelConfig` |
| `is_loaded()` | Check if model is loaded |
| `create_context(...)` | Create an `InferenceContext` |
| `generate(tokens, ...)` | Generate text (returns `GenerationResult`) |
| `generate_stream(tokens, ...)` | Streaming generation (yields tokens) |

**`generate()` parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_new_tokens` | `256` | Maximum tokens to generate |
| `temperature` | `1.0` | Sampling temperature |
| `top_k` | `0` | Top-K sampling (`0` = disabled) |
| `top_p` | `1.0` | Top-P (nucleus) sampling |
| `repeat_penalty` | `1.0` | Repetition penalty |
| `do_sample` | `True` | Enable sampling (False = greedy) |

**`GenerationResult`** fields: `token_ids`, `text`, `num_prompt_tokens`, `num_generated_tokens`, `finished`.

### InferenceContext

```python
ctx = model.create_context(
    kv_cache_dtype=nanoinfer.KVCacheDType.FP32,
    gpu_layers=-1,
    batch_size=512
)
ctx.forward(input_tokens)              # Single forward pass
ctx.generate(input_tokens, max_new=128) # Generate from prompt
ctx.reset()                            # Reset KV cache
ctx.warmup()                           # Warm up CUDA kernels
```

### Tokenizer

```python
tokenizer = nanoinfer.Tokenizer("model.gguf")
# or
tokenizer = nanoinfer.Tokenizer.load_from_gguf(model)
```

| Method / Property | Description |
|-------------------|-------------|
| `encode(text)` | Encode text to token IDs |
| `decode(tokens)` | Decode token IDs to text |
| `vocab_size()` | Get vocabulary size |
| `bos_token_id` | BOS token ID |
| `eos_token_id` | EOS token ID |
| `chat_template` | Chat template string from GGUF metadata |

### Generator

```python
gen = nanoinfer.Generator(ctx)
result = gen.generate(tokens, max_new_tokens=128, temperature=0.8)
```

### MultimodalModel

```python
mm = nanoinfer.MultimodalModel("model.gguf", "mmproj.gguf")
result = mm.chat("Describe this image", image_path="photo.jpg")
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
# ... run inference ...
profiler.disable()
print(profiler.summary())
```

### RequestScheduler

For multi-request batching with paged KV cache:

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

## Enums

| Enum | Values |
|------|--------|
| `DataType` | `FP32`, `FP16`, `Q4_0`, `Q4_1`, `Q4_K`, `Q6_K`, `Q8_0`, `INT8`, `INT32` |
| `DeviceType` | `CPU`, `CUDA` |
| `KVCacheDType` | `FP32`, `Q4_0` |
| `LogLevel` | `NONE`, `ERROR`, `WARN`, `INFO`, `DEBUG`, `TRACE` |

## Threading

```python
nanoinfer.set_num_threads(8)  # Set OpenMP thread count
```
