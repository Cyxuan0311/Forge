<p align="center">
  <img src="resource/logo.png" alt="Forge" width="45%">
</p>

<div align="center">

<b>纯 C++/CUDA 的轻量级 LLM 推理引擎</b>

[![GitHub Stars](https://img.shields.io/github/stars/Cyxuan0311/Forge?style=social)](https://github.com/Cyxuan0311/Forge/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/Cyxuan0311/Forge?style=social)](https://github.com/Cyxuan0311/Forge/network/members)
[![GitHub Issues](https://img.shields.io/github/issues/Cyxuan0311/Forge?style=social)](https://github.com/Cyxuan0311/Forge/issues)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[English](README.md)

</div>

## 快速开始

从源码构建：

```bash
git clone https://github.com/Cyxuan0311/Forge.git
cd Forge
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

运行模型：

```bash
# 终端交互式对话
./build/forge-cli -m model.gguf --stream

# 单次生成
./build/forge-cli -m model.gguf -p "你好，世界！" -n 128

# OpenAI 兼容 API 服务器（内嵌 Web 控制台）
./build/forge-server -m model.gguf --port 8080 -ngl 99

# 带图片输入的 Chat UI（多模态模型需要 mmproj 权重）
./build/forge-server -m model.gguf --mmproj mmproj.gguf --chat
```

<p align="center">
  <img src="resource/demo.png" alt="Forge 演示" width="90%">
</p>

## 简介

`Forge` 的目标是在最少依赖、代码完全透明的前提下实现 LLM（及 VLM）推理——
从上层 Token API 到手写 Kernel，整条推理路径都保持紧凑可读。

- 纯 C++17 / CUDA 实现，不依赖 PyTorch 等任何 ML 框架
- 自研 CUDA Kernel：GEMM/GEMV、量化 MMQ、Flash Attention、RMSNorm、RoPE、Sampling
- AVX2 + OpenMP CPU 后端；可选 cuBLAS / OpenBLAS
- 整数量化：Q3_K、Q4_0、Q4_1、Q4_K、Q5_K、Q6_K、Q8_0；支持 FP16 / BF16 权重
- CPU+GPU 混合推理，显存放不下的模型可以部分卸载加速（`-ngl`）
- OpenAI 兼容 HTTP 服务器（`/v1/chat/completions`），内嵌 Dashboard 与 Chat UI
- 通过 `mmproj` 视觉编码器支持 Qwen3-VL 与 MiniCPM-V 4.6 多模态推理
- 投机解码（DFlash 草稿模型、MTP）、分页 KV Cache、KV Cache 量化、Prompt 前缀缓存

## 架构

Forge 将推理路径保持在一个紧凑的分层结构中，从用户接口一直到底层 CPU/GPU Kernel。

<p align="center">
  <img src="resource/arch.png" alt="Forge 架构图" width="100%">
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

详细设计请参考[架构设计](docs/architecture_zh.md)。

## 支持的模型

| 系列 | 架构 | 结构 | 多模态 |
| --- | --- | --- | --- |
| LLaMA | llama, mistral, yi, phi-3, falcon | GQA | ❌ |
| Qwen | qwen, qwen2 | GQA | ❌ |
| Qwen3-VL | qwen3vl | GQA + MRoPE + QK-Norm | ✅ |
| Qwen3.5 | qwen35 | 混合 SSM + Gated Delta Net + Attention | ❌ |
| DeepSeek | deepseek_v2/v3 | MLA（+ MoE） | ❌ |
| Gemma | gemma, gemma2, gemma4 | GQA / SWA（+ MoE） | ❌ |
| MiniCPM-V | minicpm-v 4.6 | GQA + SigLIP 视觉编码器 | ✅ |
| 其他 | phimoe, spark2_5, lfm2, bailingmoe3, xing4_0 | MoE / 混合注意力 | ❌ |

完整矩阵见[模型支持](docs/models_zh.md)，KV Cache 量化见 [kv-quant.md](docs/kv-quant.md)。

## 文档

#### 工具

- [forge-cli 使用](docs/cli_zh.md)
- [Python API](docs/python_api_zh.md)

#### 开发

- [构建指南](docs/build_zh.md)
- [依赖说明](docs/dependencies_zh.md)
- [架构设计](docs/architecture_zh.md)
- [模型支持](docs/models_zh.md)
- [KV Cache 量化](docs/kv-quant.md)
- [LFM2 性能笔记](docs/perf-lfm2.md)
- [Benchmark Pipeline](report/README.md)

## 性能

Forge 提供 [`report/`](report/README.md) 下的可复现 Benchmark Pipeline：

```bash
python -m report.runner
```

生成的报告位于 `resource/reports/`。

## 贡献

- 欢迎提交 PR
- Issue、PR 与模型支持的任何帮助都非常感谢！
- 更多信息见 [CONTRIBUTING.md](CONTRIBUTING.md)

## 致谢

- [nothings/stb](https://github.com/nothings/stb) - 单头文件图像格式解码库，用于多模态子系统 - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - 单头文件 JSON 库 - MIT License
