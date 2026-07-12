# 模型支持

[English](models.md)

## 支持的架构

| 架构 | 引擎 | 注意力 | 特殊特性 |
|------|------|--------|---------|
| LLaMA / LLaMA 2 / LLaMA 3 | LlamaEngine | GQA | NeoX RoPE, SiLU+GELU |
| Mistral | LlamaEngine | GQA | 滑动窗口 |
| Qwen / Qwen2 / Qwen2.5 | LlamaEngine | GQA | RoPE theta=1e6, 共享嵌入 |
| Yi / Yi 1.5 | LlamaEngine | GQA | NeoX RoPE |
| Phi-3 | LlamaEngine | GQA | NeoX RoPE |
| DeepSeek (V2) | DeepSeekEngine | MLA + GQA | KV LoRA, Q LoRA |
| DeepSeek (V3, R1) | DeepSeekEngine | MLA + GQA | KV LoRA, Q LoRA, MoE |
| Qwen3.5 (MoE) | Qwen35Engine | 混合（Full Attn + SSM） | MRoPE, Gated Delta Net, MoE |
| Falcon | FalconEngine | GQA | LayerNorm, 并行残差 |
| Gemma 1 / 2 | GemmaEngine | GQA | GeGLU, logit softcapping |

### 多模态

| 模型 | 视觉编码器 | 备注 |
|------|-----------|------|
| MiniCPM-V 4.6 | CLIP ViT (mmproj) | 需要单独的 mmproj 权重文件 |

### 模型规模

| 系列 | 支持规模 |
|------|---------|
| TinyLlama | 1.1B |
| Qwen2.5 | 0.5B – 72B |
| LLaMA 2/3 | 7B – 70B |
| DeepSeek V2/V3 | 236B – 671B |
| Phi-3 | 3.8B – 14B |
| Gemma 1/2 | 2B – 27B |
| Falcon | 7B – 180B |

## 引擎能力矩阵

| 能力 | LLaMA | DeepSeek | Qwen3.5 | Falcon | Gemma | MiniCPM-V |
|------|-------|----------|---------|--------|-------|-----------|
| GQA | ✅ | ✅ | — | ✅ | ✅ | ✅ |
| MLA | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| MoE | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
| SSM | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| 多模态 | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |

## 量化类型

| 类型 | 比特/权重 | 块大小 | 说明 |
|------|----------|--------|------|
| FP32 | 32 | — | 全精度 |
| FP16 | 16 | — | 半精度 |
| Q8_0 | 8.5 | 32 | 8-bit 量化 |
| Q6_K | 6.5 | 256 | 6-bit K-quant |
| Q4_K | 4.5 | 256 | 4-bit K-quant（更高质量） |
| Q4_1 | 4.5 | 32 | 4-bit 量化（带 FP16 scale） |
| Q4_0 | 4.5 | 32 | 4-bit 量化 |

## 模型格式

### GGUF

llama.cpp 生态的标准格式，在 HuggingFace 上广泛可用。支持基于 mmap 的零拷贝加载。

### .ninf

NanoInfer 原生格式，优化了加载速度。从 GGUF 转换：

```bash
python tools/convert_gguf_to_ninf.py input.gguf output.ninf
```

## 下载模型

```bash
# TinyLlama (Q4_0)
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf -P models/

# Qwen2.5 (Q4_0)
wget https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_0.gguf -P models/
```

## 权重映射

NanoInfer 使用 `WeightMapper` 系统将架构特定的权重名称映射到统一的规范命名方案。这使得所有支持的架构可以共享同一推理代码路径。映射示例：

```
model.layers.0.self_attn.q_proj.weight → layers.0.attention.wq.weight
model.layers.0.self_attn.k_proj.weight → layers.0.attention.wk.weight
```
