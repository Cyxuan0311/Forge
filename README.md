<p align="center">
  <img src="resource/logo.png" alt="Forge" width="45%">
</p>

<div align="center">

<b>Lightweight LLM inference in pure C++/CUDA</b>

[![GitHub Stars](https://img.shields.io/github/stars/Cyxuan0311/Forge?style=social)](https://github.com/Cyxuan0311/Forge/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/Cyxuan0311/Forge?style=social)](https://github.com/Cyxuan0311/Forge/network/members)
[![GitHub Issues](https://img.shields.io/github/issues/Cyxuan0311/Forge?style=social)](https://github.com/Cyxuan0311/Forge/issues)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[中文](README_zh.md)

</div>

## Quick start

Build from source:

```bash
git clone https://github.com/Cyxuan0311/Forge.git
cd Forge
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Run a model:

```bash
# Interactive chat in the terminal
./build/forge-cli -m model.gguf --stream

# One-shot generation
./build/forge-cli -m model.gguf -p "Hello, world!" -n 128

# OpenAI-compatible API server with the embedded Web UI
./build/forge-server -m model.gguf --port 8080 -ngl 99

# Chat UI with image input (multimodal models need the mmproj weights)
./build/forge-server -m model.gguf --mmproj mmproj.gguf --chat
```

<p align="center">
  <img src="resource/demo.png" alt="Forge Demo" width="90%">
</p>

## Description

The main goal of `Forge` is to enable LLM (and VLM) inference with minimal
dependencies, a transparent codebase, and competitive performance — from the
token API down to hand-written kernels.

- Pure C++17 / CUDA implementation, no PyTorch or other ML frameworks
- Custom CUDA kernels: GEMM/GEMV, quantized MMQ, Flash Attention, RMSNorm, RoPE, sampling
- AVX2 + OpenMP CPU backend; optional cuBLAS / OpenBLAS
- Integer quantization: Q3_K, Q4_0, Q4_1, Q4_K, Q5_K, Q6_K, Q8_0; FP16 / BF16 weights
- CPU+GPU hybrid inference to partially accelerate models larger than the VRAM capacity (`-ngl`)
- OpenAI-compatible HTTP server (`/v1/chat/completions`) with built-in dashboard and chat UI
- Multimodal inference for Qwen3-VL and MiniCPM-V 4.6 via `mmproj` vision encoders
- Speculative decoding (DFlash draft model, MTP), paged KV cache, KV cache quantization, and prompt prefix caching

## Architecture

Forge keeps the inference path compact, from the user-facing interfaces down to the optimized CPU/GPU kernels.

<p align="center">
  <img src="resource/arch.png" alt="Forge Architecture" width="100%">
</p>

```text
Model
  ↓
Inference Engine
  ↓
Operators
  ↓
CPU / CUDA Kernels
  ↓
Hardware
```

See the [Architecture overview](docs/architecture.md) for the detailed design.

## Supported models

| Family | Architectures | Structure | Multimodal |
| --- | --- | --- | --- |
| LLaMA | llama, mistral, yi, phi-3, falcon | GQA | ❌ |
| Qwen | qwen, qwen2 | GQA | ❌ |
| Qwen3-VL | qwen3vl | GQA + MRoPE + QK-Norm | ✅ |
| Qwen3.5 | qwen35 | Hybrid SSM + Gated Delta Net + Attention | ❌ |
| DeepSeek | deepseek_v2/v3 | MLA (+ MoE) | ❌ |
| Gemma | gemma, gemma2, gemma4 | GQA / SWA (+ MoE) | ❌ |
| MiniCPM-V | minicpm-v 4.6 | GQA + SigLIP vision | ✅ |
| Others | phimoe, spark2_5, lfm2, bailingmoe3, xing4_0 | MoE / hybrid attention | ❌ |

See [Model Support](docs/models.md) for the full matrix and [kv-quant.md](docs/kv-quant.md) for KV cache quantization.

## Documentation

#### Tools

- [forge-cli usage](docs/cli.md)
- [Python API](docs/python_api.md)

#### Development

- [How to build](docs/build.md)
- [Dependencies](docs/dependencies.md)
- [Architecture overview](docs/architecture.md)
- [Model support](docs/models.md)
- [KV cache quantization](docs/kv-quant.md)
- [LFM2 performance notes](docs/perf-lfm2.md)
- [Benchmark pipeline](report/README.md)

## Performance

Forge ships a reproducible benchmark pipeline under [`report/`](report/README.md):

```bash
python -m report.runner
```

Generated reports live under `resource/reports/`.

## Contributing

- Contributors can open PRs
- Any help with issues, PRs, and model support is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by the multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library - MIT License
