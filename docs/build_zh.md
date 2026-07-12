# 构建与安装

[English](build.md)

## 依赖

| 依赖 | 最低版本 | 是否必需 |
|-----|---------|---------|
| CMake | 3.18+ | 是 |
| C++ 编译器 | 支持 C++17 | 是 |
| CUDA Toolkit | 11.0+ | 是 |
| Python 3 | 3.8+ | 是（绑定需要） |
| NumPy | 任意 | 是（绑定需要） |
| OpenMP | 任意 | 可选（CPU 并行） |

## 从源码构建

```bash
git clone https://github.com/yourname/NanoInfer.git
cd NanoInfer

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

构建产物：
- `build/nanoinfer-cli` — CLI 推理可执行文件
- `build/nanoinfer*.so` — Python 绑定模块

## 编译选项

| 选项 | 默认值 | 说明 |
|------|-------|------|
| `CMAKE_BUILD_TYPE` | `Release` | 构建类型（`Release` / `Debug`） |
| `CMAKE_CUDA_ARCHITECTURES` | `75;86;89` | 目标 CUDA 架构（逗号分隔） |
| `USE_CUBLAS` | `ON` | 使用 cuBLAS 做 GPU GEMM；`OFF` 则用纯 CUDA tiled GEMM kernel |
| `USE_OPENBLAS` | `OFF` | 使用 OpenBLAS 做 CPU FP32 GEMM；默认用手调 AVX2 kernel |

### 常见编译变体

**完全自包含（无 cuBLAS、无 OpenBLAS）：**

```bash
cmake -B build -DUSE_CUBLAS=OFF -DUSE_OPENBLAS=OFF
cmake --build build -j
```

**启用 cuBLAS 加速：**

```bash
cmake -B build -DUSE_CUBLAS=ON
cmake --build build -j
```

**启用 OpenBLAS 加速 CPU FP32 矩阵乘：**

```bash
cmake -B build -DUSE_OPENBLAS=ON
cmake --build build -j
```

**Debug 构建：**

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
```

### CUDA 架构设置

根据你的 GPU 设置 `CMAKE_CUDA_ARCHITECTURES`：

| GPU | 架构 |
|-----|------|
| RTX 4090, 4080, 3090, 3080 | `86` |
| RTX 5090, 5080 | `89` |
| RTX 3070, 2070, 2080 | `75` |
| 多 GPU / 通用 | `75;86;89` |

## AVX2 检测

CMake 在配置时自动检测 AVX2 / FMA / F16C 支持。若支持，定义 `USE_AVX2` 并启用手调 AVX2 GEMV/GEMM kernel。在不支持 AVX2 的 CPU 上，使用标量回退实现。

## 安装 Python 包

构建完成后，可直接导入 `build/` 目录下的 `.so` 文件：

```python
import sys
sys.path.insert(0, "build")
import nanoinfer
```

或通过 pip 安装：

```bash
pip install .
```

## 验证安装

```python
import nanoinfer
print(nanoinfer.__version__)  # 0.5.0
```

## Docker

提供 Dockerfile 用于可重现构建：

```bash
docker build -t nanoinfer .
docker run --gpus all nanoinfer
```

## 常见问题

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| `CUDA not found` | CUDA Toolkit 未安装或不在 PATH 中 | 安装 CUDA Toolkit 11.0+，确保 `nvcc` 在 `PATH` 中 |
| `pybind11 not found` | 缺少子模块 | 执行 `git submodule update --init` 或设置 `FETCHCONTENT_FETCH_ONE` |
| CPU 推理慢 | 未检测到 AVX2 | 检查 CPU 是否支持 AVX2；在目标机器上构建 |
| 链接错误 | CUDA 架构不匹配 | 设置 `CMAKE_CUDA_ARCHITECTURES` 匹配你的 GPU |
