# 依赖说明

[English](dependencies.md)

## 编译期依赖

| 依赖 | 最低版本 | 是否必需 | 说明 |
|-----|---------|---------|------|
| CMake | 3.18+ | ✅ | 构建系统 |
| C++ 编译器 | C++17（GCC ≥ 9, Clang ≥ 10, MSVC ≥ 2019） | ✅ | |
| CUDA Toolkit | 11.0+ | ❌ | 通过 `FORGE_USE_CUDA=ON` 开启 |
| Python 3 | 3.8+ | ❌ | 仅 Python 绑定需要 |
| NumPy | 任意 | ❌ | 仅 Python 绑定需要 |
| cuBLAS | 随 CUDA Toolkit 提供 | ❌ | 通过 `USE_CUBLAS=ON` 开启 |
| OpenBLAS | 0.3+ | ❌ | 通过 `USE_OPENBLAS=ON` 开启 |
| OpenMP | 任意（编译器内置） | ❌ | 自动检测，用于 CPU 并行 |

### 各平台安装

**Ubuntu / Debian：**
```bash
sudo apt install cmake build-essential
# 可选：CUDA Toolkit — 从 https://developer.nvidia.com/cuda-downloads 下载
# 可选：OpenBLAS
sudo apt install libopenblas-dev
```

**Arch Linux：**
```bash
sudo pacman -S cmake base-devel
# 可选：CUDA Toolkit
sudo pacman -S cuda
# 可选：OpenBLAS
sudo pacman -S openblas
```

**macOS（仅 CPU）：**
```bash
brew install cmake
# 可选：OpenBLAS
brew install openblas
```

**验证工具链：**
```bash
cmake --version                 # ≥ 3.18
g++ --version                   # C++17
nvcc --version                  # CUDA ≥ 11.0（使用 GPU 时）
python3 --version               # ≥ 3.8
```

## 运行时依赖

### Python 绑定（使用 Python API 必需）

| 包 | 最低版本 | 说明 |
|----|---------|------|
| Python 3 | 3.8+ | |
| NumPy | 任意 | `pip install numpy` |

### 示例脚本（可选）

| 脚本 | 依赖包 | 安装命令 |
|------|--------|---------|
| `minicpmv_cli_inference.py` | Pillow | `pip install Pillow` |
| `minicpmv_cli_inference.py`（视频） | decord | `pip install decord` |
| `bench_backend.py` | tabulate | `pip install tabulate` |
| 报告模块 | matplotlib, pandas | `pip install matplotlib pandas` |

## 依赖关系图

```
┌──────────────────────────────────────────────────────────┐
│                    forge (.so / .exe)                 │
├─────────────┬──────────────┬──────────────┬──────────────┤
│ forge_  │ forge_   │ forge_   │ forge_   │
│ core        │ ops          │ model        │ tokenizer    │
├──────┬──────┼──────┬───────┤              │              │
│ C++  │ CUDA │ C++  │ CUDA  │              │              │
│ STL  │ rt   │      │ 内核  │              │              │
│      │      │      │ .cu   │              │              │
│      │      │      ├───────┤              │              │
│      │      │      │cuBLAS │              │              │
│      │      │      │(可选) │              │              │
├──────┴──────┴──────┴───────┴──────────────┴──────────────┤
│ OpenMP (可选) │ OpenBLAS (可选) │ AVX2 (自动) │ C++ STL  │
└──────────────────────────────────────────────────────────┘
```

## 构建场景

| 场景 | 命令 | USE_CUDA | USE_CUBLAS | USE_OPENBLAS |
|------|------|----------|------------|--------------|
| GPU + cuBLAS（默认） | `cmake -B build` | 自动检测 | ON | OFF |
| GPU，纯 CUDA 内核 | `cmake -B build -DUSE_CUBLAS=OFF` | 自动检测 | OFF | OFF |
| 仅 CPU | `cmake -B build -DFORGE_USE_CUDA=OFF` | OFF | — | OFF |
| CPU + OpenBLAS | `cmake -B build -DFORGE_USE_CUDA=OFF -DUSE_OPENBLAS=ON` | OFF | — | ON |

> 详细构建说明请参阅[构建指南](build_zh.md)。

## CUDA 架构自动检测

从 v0.5.0 起，CMake 配置时会自动检测当前 GPU 的架构：

```cmake
cmake -B build
# 输出: -- Detected CUDA architecture: 89
```

检测原理：编译一个迷你 CUDA 程序，调用 `cudaGetDeviceProperties` 获取实际 GPU 的计算能力。如果检测失败（如无 GPU 的服务器），默认使用 `86;89`。手动指定：

```bash
cmake -B build -DFORGE_CUDA_ARCH="90"         # RTX 5090
cmake -B build -DFORGE_CUDA_ARCH="75;86;89"   # 便携二进制
```

## 后端汇总

| 后端 | 类型 | 自动检测 | 启用条件 |
|------|------|---------|---------|
| CUDA | GPU 内核执行 | ❌（需 toolkit） | `CUDAToolkit_FOUND` |
| cuBLAS | GPU FP32 GEMM | ❌（需 CUDA） | `USE_CUBLAS=ON` |
| AVX2 | CPU SIMD 加速 | ✅ 编译时 | `COMPILER_SUPPORTS_AVX2` |
| OpenMP | CPU 多线程 | ✅ 编译时 | `OpenMP_CXX_FOUND` |
| OpenBLAS | CPU FP32 GEMM | ❌（需安装库） | `USE_OPENBLAS=ON` |
