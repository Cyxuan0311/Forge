# Model Support

[中文](models_zh.md)

## Supported Architectures

| Architecture | Engine | Attention | Special Features |
|-------------|--------|-----------|------------------|
| LLaMA / LLaMA 2 / LLaMA 3 | LlamaEngine | GQA | NeoX RoPE, SiLU+GELU |
| Mistral | LlamaEngine | GQA | Sliding window |
| Qwen / Qwen2 / Qwen2.5 | LlamaEngine | GQA | RoPE theta=1e6, tied embeddings |
| Yi / Yi 1.5 | LlamaEngine | GQA | NeoX RoPE |
| Phi-3 | LlamaEngine | GQA | NeoX RoPE |
| DeepSeek (V2) | DeepSeekEngine | MLA + GQA | KV LoRA, Q LoRA |
| DeepSeek (V3, R1) | DeepSeekEngine | MLA + GQA | KV LoRA, Q LoRA, MoE |
| Qwen3.5 (MoE) | Qwen35Engine | Hybrid (Full Attn + SSM) | MRoPE, Gated Delta Net, MoE |
| Falcon | FalconEngine | GQA | LayerNorm, parallel residual |
| Gemma 1 / 2 | GemmaEngine | GQA | GeGLU, logit softcapping |

### Multimodal

| Model | Vision Encoder | Notes |
|-------|---------------|-------|
| MiniCPM-V 4.6 | CLIP ViT (mmproj) | Requires separate mmproj weights |

### Model Sizes

| Family | Supported Sizes |
|--------|----------------|
| TinyLlama | 1.1B |
| Qwen2.5 | 0.5B – 72B |
| LLaMA 2/3 | 7B – 70B |
| DeepSeek V2/V3 | 236B – 671B |
| Phi-3 | 3.8B – 14B |
| Gemma 1/2 | 2B – 27B |
| Falcon | 7B – 180B |

## Engine Architecture Matrix

| Capability | LLaMA | DeepSeek | Qwen3.5 | Falcon | Gemma | MiniCPM-V |
|-----------|-------|----------|---------|--------|-------|-----------|
| GQA | ✅ | ✅ | — | ✅ | ✅ | ✅ |
| MLA | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| MoE | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
| SSM | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| Multimodal | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |

## Quantization Types

| Type | Bits/Weight | Block Size | Description |
|------|------------|-----------|-------------|
| FP32 | 32 | — | Full precision |
| FP16 | 16 | — | Half precision |
| Q8_0 | 8.5 | 32 | 8-bit quantization |
| Q6_K | 6.5 | 256 | 6-bit K-quant |
| Q4_K | 4.5 | 256 | 4-bit K-quant (higher quality) |
| Q4_1 | 4.5 | 32 | 4-bit quantization (with FP16 scale) |
| Q4_0 | 4.5 | 32 | 4-bit quantization |

## Model Formats

### GGUF

The standard format from the llama.cpp ecosystem. Widely available on HuggingFace. Supports mmap-based zero-copy loading.

### .ninf

NanoInfer's native format, optimized for faster loading. Convert from GGUF:

```bash
python tools/convert_gguf_to_ninf.py input.gguf output.ninf
```

## Downloading Models

```bash
# TinyLlama (Q4_0)
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf -P models/

# Qwen2.5 (Q4_0)
wget https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_0.gguf -P models/
```

## Weight Mapping

NanoInfer uses a `WeightMapper` system that maps architecture-specific weight names to a unified canonical naming scheme. This enables a single inference code path across all supported architectures. The mapping supports patterns like:

```
model.layers.0.self_attn.q_proj.weight → layers.0.attention.wq.weight
model.layers.0.self_attn.k_proj.weight → layers.0.attention.wk.weight
```
