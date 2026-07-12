# 架构设计

[English](architecture.md)

## 概览

Forge 是一个基于 C++17 和 CUDA 构建的模块化 LLM 推理引擎，采用分层架构，通过可扩展的插件系统实现灵活性：

```
┌──────────────────────────────────────┐
│         Python 绑定 (pybind11)        │
├──────────────────────────────────────┤
│         CLI / 应用层                  │
├────────────┬──────────┬──────────────┤
│  推理引擎   │  模型加载 │   分词器     │
│            │          │  (SPM/BPE)   │
├────────────┴──────────┼──────────────┤
│      算子层           │   视觉编码器   │
│  (CPU + CUDA)        │              │
├───────────────────────┴──────────────┤
│  计算图 / 算子融合 / KV 缓存          │
├──────────────────────────────────────┤
│     核心层 (Tensor, Backend, Memory)  │
└──────────────────────────────────────┘
```

## 核心层

### Tensor

基础数据结构（`include/forge/tensor.h`）。支持 FP32、FP16、Q4_0、Q4_1、Q4_K、Q6_K、Q8_0、INT8、INT32 数据类型的多维数组。

- **存储**：CPU 使用 `malloc`，CUDA 使用 `cudaMalloc`
- **零拷贝**：`from_buffer()` 包装 mmap 指针，用于 GGUF 加载
- **传输**：`to_device()` 在 CPU 和 CUDA 之间复制
- **视图**：`view()`、`slice()` 创建不复制数据的张量视图

### Backend

抽象设备后端（`include/forge/backend.h`）：

| 实现 | 分配 | 拷贝 | 特性 |
|------|------|------|------|
| `CpuBackend` | `malloc` / `free` | `memcpy` | FP32、FP16、量化 |
| `CudaBackend` | `cudaMalloc` / `cudaFree` | `cudaMemcpy` | FP32、FP16、量化、流 |

`BackendManager`（单例）管理可用后端，支持基于插件的注册（`register_backend`）。

### MemoryPool

设备感知的内存池（`include/forge/memory_pool.h`），实现空闲列表复用和分配跟踪，减少推理过程中的 `cudaMalloc`/`cudaFree` 开销。

### 计算图

基于 DAG 的执行系统（`include/forge/compute_graph.h`）：

- `GraphNode`：算子类型、输入/输出张量、计算函数
- `GraphBuilder`：流畅接口构建计算图
- CPU 与 CUDA 节点间的自动设备传输
- 算子融合（`optimize_fusion`）合并相邻算子
- 中间张量释放优化

## 算子层

所有算子实现 `Op` 接口，通过模板分发（`dispatch.h`）同时支持 CPU 和 CUDA 执行。

| 算子 | CPU | CUDA | 说明 |
|------|-----|------|------|
| RMSNorm / LayerNorm | AVX2 + OpenMP | 自研 CUDA kernel | 归一化 |
| SiLU / GELU / GeGLU | AVX2 多项式 | 自研 CUDA kernel | 激活函数 |
| MatMul (FP32) | OpenBLAS 或 AVX2 GEMM | cuBLAS 或 tiled GEMM | 稠密矩阵乘 |
| MatMul (量化) | 融合去量化 + GEMV AVX2 | 融合去量化 + GEMV | 量化 GEMV（Q4_0、Q4_K、Q6_K 等） |
| Flash Attention | AVX2 + OpenMP | 自研 CUDA kernel | 多头 / GQA / MLA |
| RoPE | AVX2 | 自研 CUDA kernel | 旋转位置编码 |
| Embedding | 纯 C++ | 自研 CUDA kernel | Token 嵌入查找 |
| Elementwise (add, mul) | 纯 C++ | 自研 CUDA kernel | 逐元素运算 |
| Softmax | 纯 C++ | 自研 CUDA kernel | Softmax |

### CUDA Kernel

位于 `src/operators/cuda*.cu` 和 `src/operators/cuda/`：

| 文件 | Kernel |
|------|--------|
| `cuda_gemv.cu` | FP32、Q4_0、Q4_1、Q4_K、Q6_K 的 GEMV，融合 dual/FFN 变体 |
| `cuda_gemm.cu` | 纯 CUDA tiled GEMM（cuBLAS 回退） |
| `cuda_attention.cu` | Flash attention、GQA flash attention、decode GQA |
| `cuda_norm.cu` | 融合激活函数的 RMS norm |
| `cuda_embedding.cu` | FP32 和量化 embedding |
| `cuda_quant.cu` | Q4_0、Q4_1、Q4_K、Q6_K 的量化/反量化 |
| `cuda/cuda_rope.cu` | RoPE |
| `cuda/cuda_elementwise.cu` | 逐元素运算 |
| `cuda/cuda_activation.cu` | 激活函数 |
| `cuda/cuda_fused.cu` | 融合 QKV / FFN 操作 |

### CPU Kernel

位于 `src/operators/cpu/`：

- **gemv.h**：AVX2 优化的 GEMV，支持 FP32 和所有量化类型
- **gemm.h**：AVX2 优化的 GEMM
- **fused.h**：融合的 Q4_0/Q6_K FFN 下投影 + 残差
- **vec.h**：向量运算（点积等）
- **quant_helpers.h**：量化工具函数

## 模型层

### 模型加载器

双格式支持（`include/forge/model_loader.h`）：

| 格式 | 加载器 | 特性 |
|------|--------|------|
| **GGUF** | `GgufModel` | 基于 mmap 的零拷贝，直接从 GGUF 文件读取元数据和张量 |
| **NINF** | `NinfModel` | Forge 原生格式，打包二进制头，加载优化 |

通过 `detect_format()` 自动检测格式。

### 权重映射器

`WeightMapper`（`include/forge/weight_mapper.h`）将架构特定的权重名称映射到统一的规范命名方案。每种架构通过 `FORGE_REGISTER_ARCH` 注册自己的映射。这使得所有架构可以共享同一套 `ModelWeights` / `LayerWeights` 结构和推理代码路径。

## 推理层

### 引擎注册表

架构特定引擎通过 `FORGE_REGISTER_ENGINE` / `FORGE_REGISTER_ARCH` 宏注册。工厂模式在运行时根据模型架构选择正确的引擎。

```
InferenceEngine (虚基类)
  └── TransformerEngine
        ├── LlamaEngine        (LLaMA、Mistral、Qwen、Yi、Phi-3)
        ├── DeepSeekEngine     (DeepSeek V2/V3/R1)
        ├── Qwen35Engine       (Qwen3.5 混合 SSM+Attention)
        ├── FalconEngine       (Falcon)
        └── GemmaEngine        (Gemma 1/2)
```

### 图构建器

每种引擎架构有对应的 `GraphBuilder` 用于构建计算图：

| 图构建器 | 架构 |
|---------|------|
| `LlamaGraphBuilder` | LLaMA / Mistral / Qwen / Yi |
| `DeepSeekGraphBuilder` | DeepSeek V2/V3（GQA + MLA） |
| `Qwen35GraphBuilder` | Qwen3.5（Hybrid SSM + Attention） |

通过 `FORGE_REGISTER_GRAPH_BUILDER` 宏注册。

### KV 缓存

两种缓存实现（`include/forge/kv_cache.h`）：

| 类型 | 说明 |
|------|------|
| **连续 KVCache** | 固定大小预分配缓冲区，直接索引 |
| **分页 KVCache** | 基于块的分配，支持连续批处理 |

两者均支持 FP32 和 Q4_0 量化的 KV 缓存数据。

### Generator

高级文本生成（`include/forge/generator.h`）：

- 可配置采样：温度、top-k、top-p、重复惩罚
- 贪婪（argmax）模式
- 流式回调支持
- CUDA 加速采样

### Sampler

`Sampler`（`include/forge/sampler.h`）实现：

- 温度缩放
- Top-K 过滤
- Top-P（核）过滤
- 重复惩罚
- Argmax（CPU 和 CUDA）

### RequestScheduler

`RequestScheduler`（`include/forge/request_scheduler.h`）管理多个并发推理请求，使用分页 KV 缓存分配。支持 prefill/decode 调度以实现高效批处理。

## 视觉层

视觉编码器（`include/forge/vision_encoder.h`、`src/vision/`）实现了基于 CLIP 的 ViT，用于多模态模型：

```
图像 → 预处理（调整大小 + 归一化）
     → Patch 嵌入（卷积）
     → 位置嵌入（SigLIP 2D 桶）
     → ViT 块（LayerNorm → QKV → Attention → FFN）
     → Token 合并器（窗口注意力 + 2x2 下采样）
     → 投影器（下采样 → LN → FFN → 输出）
     → LLM 嵌入空间
```

支持：
- MiniCPM-V 4.6 架构
- SigLIP 风格位置编码（2D 桶）
- 2x2 空间下采样的窗口注意力
- 灵活的插入点（可先运行所有 ViT 块，或在 `insert_layer_id` 处分段执行）

## 注册表系统

项目广泛使用编译期自动注册：

| 宏 | 注册表 | 用途 |
|----|--------|------|
| `FORGE_REGISTER_ENGINE` | EngineRegistry | 按架构名称注册推理引擎 |
| `FORGE_REGISTER_ARCH` | Arch registry | 注册架构（引擎+配置+权重+能力） |
| `FORGE_REGISTER_LOADER` | ModelLoaderRegistry | 注册模型格式加载器 |
| `FORGE_REGISTER_CONFIG_PARSER` | ConfigParserRegistry | 架构特定配置解析 |
| `FORGE_REGISTER_WEIGHT_INIT` | WeightInitRegistry | 架构特定权重初始化 |
| `FORGE_REGISTER_ARCH_CAPABILITY` | Capability registry | 架构能力标志 |
| `FORGE_REGISTER_GRAPH_BUILDER` | GraphBuilderRegistry | 计算图构建器 |
| `FORGE_REGISTER_OP` | OpRegistry | 算子注册 |

## 性能分析器

`PerfProfiler`（`include/forge/perf_profiler.h`）提供：

- 基于 CUDA event 的 GPU 时序
- 线程本地记录
- RAII 作用域计时器（`PerfScopeTimer`）
- 带逐算子分解的摘要输出
