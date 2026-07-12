# Architecture

[中文](architecture_zh.md)

## Overview

Forge is a modular LLM inference engine built with C++17 and CUDA. It follows a layered architecture with extensible plugin systems:

```
┌──────────────────────────────────────┐
│         Python Bindings (pybind11)    │
├──────────────────────────────────────┤
│         CLI / Applications            │
├────────────┬──────────┬──────────────┤
│  Inference │  Model   │  Tokenizer   │
│  Engines   │  Loader  │  (SPM/BPE)   │
├────────────┴──────────┼──────────────┤
│      Operators        │    Vision    │
│  (CPU + CUDA)         │   Encoder    │
├───────────────────────┴──────────────┤
│  Compute Graph / OP Fusion / KV Cache│
├──────────────────────────────────────┤
│     Core (Tensor, Backend, Memory)    │
└──────────────────────────────────────┘
```

## Core Layer

### Tensor

The fundamental data structure (`include/forge/tensor.h`). A multi-dimensional array with shape/stride/strides, supporting FP32, FP16, Q4_0, Q4_1, Q4_K, Q6_K, Q8_0, INT8, and INT32 data types.

- **Storage**: CPU via `malloc`, CUDA via `cudaMalloc`
- **Zero-copy**: `from_buffer()` wraps mmap'd pointers for GGUF loading
- **Transfer**: `to_device()` copies between CPU and CUDA
- **Views**: `view()`, `slice()` create tensor views without data copy

### Backend

Abstract device backend (`include/forge/backend.h`):

| Implementation | Allocation | Copy | Features |
|---------------|-----------|------|----------|
| `CpuBackend` | `malloc` / `free` | `memcpy` | FP32, FP16, quantized |
| `CudaBackend` | `cudaMalloc` / `cudaFree` | `cudaMemcpy` | FP32, FP16, quantized, streams |

`BackendManager` (singleton) manages available backends with plugin-based registration (`register_backend`).

### MemoryPool

Device-aware memory pool (`include/forge/memory_pool.h`) with free-list reuse and allocation tracking. Reduces `cudaMalloc`/`cudaFree` overhead during inference.

### ComputeGraph

DAG-based execution system (`include/forge/compute_graph.h`):

- `GraphNode`: op type, input/output tensors, compute function
- `GraphBuilder`: fluent API for constructing compute graphs
- Auto device transfer between CPU and CUDA nodes
- Operator fusion (`optimize_fusion`) combines adjacent ops
- Intermediate tensor release optimization

## Operator Layer

All operators implement the `Op` interface and support both CPU and CUDA execution via template dispatch (`dispatch.h`).

| Operator | CPU | CUDA | Description |
|----------|-----|------|-------------|
| RMSNorm / LayerNorm | AVX2 + OpenMP | Custom CUDA kernel | Normalization |
| SiLU / GELU / GeGLU | AVX2 polynomial | Custom CUDA kernel | Activation |
| MatMul (FP32) | OpenBLAS or AVX2 GEMM | cuBLAS or tiled GEMM | Dense matrix multiply |
| MatMul (quantized) | Fused dequant + GEMV AVX2 | Fused dequant + GEMV | Quantized GEMV (Q4_0, Q4_K, Q6_K, etc.) |
| Flash Attention | AVX2 + OpenMP | Custom CUDA kernel | Multi-head / GQA / MLA |
| RoPE | AVX2 | Custom CUDA kernel | Rotary position embedding |
| Embedding | Pure C++ | Custom CUDA kernel | Token embedding lookup |
| Elementwise (add, mul, etc.) | Pure C++ | Custom CUDA kernel | Element-wise ops |
| Softmax | Pure C++ | Custom CUDA kernel | Softmax |

### CUDA Kernels

Located in `src/operators/cuda*.cu` and `src/operators/cuda/`:

| File | Kernels |
|------|---------|
| `cuda_gemv.cu` | GEMV for FP32, Q4_0, Q4_1, Q4_K, Q6_K, fused dual/FFN variants |
| `cuda_gemm.cu` | Pure CUDA tiled GEMM (cuBLAS fallback) |
| `cuda_attention.cu` | Flash attention, GQA flash attention, decode GQA |
| `cuda_norm.cu` | RMS norm fused with activation |
| `cuda_embedding.cu` | FP32 and quantized embedding |
| `cuda_quant.cu` | Quantize/dequantize Q4_0, Q4_1, Q4_K, Q6_K |
| `cuda/cuda_rope.cu` | RoPE |
| `cuda/cuda_elementwise.cu` | Element-wise operations |
| `cuda/cuda_activation.cu` | Activation functions |
| `cuda/cuda_fused.cu` | Fused QKV / FFN operations |

### CPU Kernels

Located in `src/operators/cpu/`:

- **gemv.h**: AVX2-optimized GEMV for FP32 and all quantized types
- **gemm.h**: AVX2-optimized GEMM
- **fused.h**: Fused Q4_0/Q6_K FFN down + residual
- **vec.h**: Vector operations (dot product, etc.)
- **quant_helpers.h**: Quantization utilities

## Model Layer

### Model Loader

Dual-format support (`include/forge/model_loader.h`):

| Format | Loader | Features |
|--------|--------|----------|
| **GGUF** | `GgufModel` | mmap-based zero-copy, reads metadata + tensors directly from GGUF files |
| **NINF** | `NinfModel` | Forge native format, packed binary header, optimized loading |

Auto-detection via `detect_format()`.

### Weight Mapper

`WeightMapper` (`include/forge/weight_mapper.h`) maps architecture-specific weight names to a unified canonical naming scheme. Each architecture registers its own mapping via `FORGE_REGISTER_ARCH`. This allows a single `ModelWeights` / `LayerWeights` structure and inference code path for all architectures.

## Inference Layer

### Engine Registry

Architecture-specific engines are registered via `FORGE_REGISTER_ENGINE` / `FORGE_REGISTER_ARCH` macros. The factory pattern selects the correct engine at runtime based on model architecture.

```
InferenceEngine (virtual base)
  └── TransformerEngine
        ├── LlamaEngine        (LLaMA, Mistral, Qwen, Yi, Phi-3)
        ├── DeepSeekEngine     (DeepSeek V2/V3/R1)
        ├── Qwen35Engine       (Qwen3.5 Hybrid SSM+Attention)
        ├── FalconEngine       (Falcon)
        └── GemmaEngine        (Gemma 1/2)
```

### Graph Builders

Each engine architecture has a corresponding `GraphBuilder` that constructs the ComputeGraph:

| Graph Builder | Architecture |
|--------------|-------------|
| `LlamaGraphBuilder` | LLaMA / Mistral / Qwen / Yi |
| `DeepSeekGraphBuilder` | DeepSeek V2/V3 (GQA + MLA) |
| `Qwen35GraphBuilder` | Qwen3.5 (Hybrid SSM + Attention) |

Registered via `FORGE_REGISTER_GRAPH_BUILDER` macro.

### KV Cache

Two cache implementations (`include/forge/kv_cache.h`):

| Type | Description |
|------|-------------|
| **Contiguous KVCache** | Fixed-size pre-allocated buffer, direct indexing |
| **PagedKVCache** | Block-based allocation, enables continuous batching |

Both support FP32 and Q4_0 quantization for the KV cache data.

### Generator

High-level text generation (`include/forge/generator.h`) with:

- Configurable sampling: temperature, top-k, top-p, repetition penalty
- Greedy (argmax) mode
- Streaming callback support
- CUDA-accelerated sampling

### Sampler

`Sampler` (`include/forge/sampler.h`) implements:

- Temperature scaling
- Top-K filtering
- Top-P (nucleus) filtering
- Repetition penalty
- Argmax (CPU and CUDA)

### RequestScheduler

`RequestScheduler` (`include/forge/request_scheduler.h`) manages multiple concurrent inference requests with paged KV cache allocation. Supports prefill/decode scheduling for efficient batching.

## Vision Layer

The vision encoder (`include/forge/vision_encoder.h`, `src/vision/`) implements a CLIP-based ViT for multimodal models:

```
Image → Preprocess (resize + normalize)
     → Patch Embedding (convolutional)
     → Position Embedding (SigLIP 2D buckets)
     → ViT Blocks (LayerNorm → QKV → Attention → FFN)
     → Token Merger (window attention + 2x2 downsample)
     → Projector (downsample → LN → FFN → output)
     → LLM embedding space
```

Supports:
- MiniCPM-V 4.6 architecture
- SigLIP-style position encoding (2D buckets)
- Window attention with 2x2 spatial downsample
- Flexible insertion point (run all ViT blocks first, or split at `insert_layer_id`)

## Registry Systems

The project extensively uses compile-time auto-registration:

| Macro | Registry | Purpose |
|-------|----------|---------|
| `FORGE_REGISTER_ENGINE` | EngineRegistry | Register inference engine by arch name |
| `FORGE_REGISTER_ARCH` | Arch registry | Register arch with engine + config + weights + capability |
| `FORGE_REGISTER_LOADER` | ModelLoaderRegistry | Register model format loader |
| `FORGE_REGISTER_CONFIG_PARSER` | ConfigParserRegistry | Architecture-specific config parsing |
| `FORGE_REGISTER_WEIGHT_INIT` | WeightInitRegistry | Architecture-specific weight initialization |
| `FORGE_REGISTER_ARCH_CAPABILITY` | Capability registry | Architecture capability flags |
| `FORGE_REGISTER_GRAPH_BUILDER` | GraphBuilderRegistry | Compute graph builder |
| `FORGE_REGISTER_OP` | OpRegistry | Operator registration |

## Performance Profiler

The `PerfProfiler` (`include/forge/perf_profiler.h`) provides:

- CUDA event-based GPU timing
- Thread-local recording
- RAII scoped timers (`PerfScopeTimer`)
- Summary output with per-op breakdown
