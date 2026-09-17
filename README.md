<p align="center">
  <img src="resource/logo.png" alt="Forge" width="45%">
</p>

<h1 align="center">Forge</h1>

<p align="center">
  <strong>Lightweight LLM Inference Engine</strong><br>
  Pure C++/CUDA · Minimal Dependencies · Custom Kernels
</p>

<p align="center">
  <a href="README_zh.md">中文</a>
</p>

<p align="center">
  <a href="https://github.com/Cyxuan0311/Forge/stargazers">
    <img src="https://img.shields.io/github/stars/Cyxuan0311/Forge?style=flat-square&logo=github&logoColor=white&label=Stars&labelColor=18181B&color=76B900" alt="GitHub Stars">
  </a>
  <a href="https://github.com/Cyxuan0311/Forge/network/members">
    <img src="https://img.shields.io/github/forks/Cyxuan0311/Forge?style=flat-square&logo=github&logoColor=white&label=Forks&labelColor=18181B&color=52525B" alt="GitHub Forks">
  </a>
  <a href="https://github.com/Cyxuan0311/Forge/issues">
    <img src="https://img.shields.io/github/issues/Cyxuan0311/Forge?style=flat-square&logo=github&logoColor=white&label=Issues&labelColor=18181B&color=52525B" alt="GitHub Issues">
  </a>
  <img src="https://img.shields.io/github/license/Cyxuan0311/Forge?style=flat-square&label=MIT&labelColor=18181B&color=A1A1AA" alt="MIT License">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/C%2B%2B-17-18181B?style=flat-square&logo=cplusplus&logoColor=white&labelColor=18181B" alt="C++17">
  <img src="https://img.shields.io/badge/CUDA-12.8%2B-76B900?style=flat-square&logo=nvidia&logoColor=white&labelColor=18181B" alt="CUDA 12.8+">
  <img src="https://img.shields.io/badge/Custom-Kernels-76B900?style=flat-square&labelColor=18181B" alt="Custom Kernels">
</p>



## Overview

**Forge** is a lightweight LLM inference engine written from scratch in **C++ and CUDA**.

It provides a minimal, transparent, and high-performance inference stack without PyTorch, TensorFlow, or other heavyweight ML frameworks.

### Highlights

* **Pure C++/CUDA** — minimal runtime dependencies
* **Custom CUDA kernels** — GEMM, GEMV, Flash Attention, RMSNorm, RoPE, sampling, and more
* **Optimized CPU backend** — AVX2 + OpenMP
* **Quantized inference** — Q4_0, Q4_1, Q4_K, Q6_K, Q8_0
* **Optional BLAS** — cuBLAS / OpenBLAS
* **Multiple architectures** — LLaMA, DeepSeek, Qwen3.5, Qwen3-VL, Gemma4, MiniCPM-V



## Architecture

Forge keeps the inference path compact, from user-facing interfaces down to optimized CPU/GPU kernels.

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

See [Architecture](docs/architecture.md) for the detailed design.



## Demo

<p align="center">
  <img src="resource/demo.png" alt="Forge Demo" width="90%">
</p>



## Quick Start

### Requirements

* CMake ≥ 3.18
* C++17 compiler
* CUDA Toolkit ≥ 12.8 for GPU inference
* Python 3 + NumPy for Python examples

### Build

```bash
git clone https://github.com/Cyxuan0311/Forge.git
cd Forge

cmake -B build
cmake --build build -j
```

### CLI

```bash
# Interactive chat
./build/forge-cli -m model.gguf --stream

# One-shot generation
./build/forge-cli \
    -m model.gguf \
    -p "Hello, world!" \
    -n 128
```

### Python

```bash
python examples/qwen_inference.py \
    --model-path model.gguf \
    --device cuda
```

For detailed build and usage instructions, see the [documentation](#documentation).



## Supported Models

| Architecture  | Example Models               | Structure                                | Multimodal |
| ------------- | ---------------------------- | ---------------------------------------- | ---------: |
| **LLaMA**     | TinyLlama, Qwen2.5, Qwen3-VL | GQA                                      |          ✅ |
| **DeepSeek**  | DeepSeek-V2/V3, DeepSeek-R1  | MLA                                      |          ❌ |
| **Qwen3.5**   | Qwen3.5-MoE                  | Hybrid SSM + Gated Delta Net + Attention |          ❌ |
| **Gemma4**    | Gemma4                       | SWA + Full Attention + MoE               |          ❌ |
| **MiniCPM-V** | MiniCPM-V 4.6                | GQA                                      |          ✅ |

See [Supported Models](docs/models.md) for details.



## Performance

Forge includes a benchmark pipeline under `report/`.

### Qwen2.5-7B-Instruct Q4_0

**RTX 4050 Laptop GPU**

<p align="center">
  <img
    src="resource/reports/qwen2.5-7b-instruct/performance_report.png"
    alt="Qwen2.5 Performance Report"
    width="70%"
  >
</p>

Benchmarks can be reproduced locally:

```bash
python -m report.runner
```

Detailed results and methodology are available under `resource/reports/`.



## Documentation

| Document                             | Description                      |
| ------------------------------------ | -------------------------------- |
| [Build Guide](docs/build.md)         | Build and configuration          |
| [Dependencies](docs/dependencies.md) | Dependencies and environment     |
| [Python API](docs/python_api.md)     | Python interface                 |
| [CLI Usage](docs/cli.md)             | Command-line inference           |
| [Supported Models](docs/models.md)   | Model architecture support       |
| [Architecture](docs/architecture.md) | Internal architecture and design |



## Contributing

Issues, optimizations, new operators, and model support are welcome.

See [CONTRIBUTING.md](CONTRIBUTING.md).



## License

[MIT License](LICENSE)
