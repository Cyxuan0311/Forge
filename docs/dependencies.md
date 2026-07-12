# Dependencies

[中文](dependencies_zh.md)

## Build-time Dependencies

| Dependency | Minimum Version | Required | Notes |
|-----------|----------------|----------|-------|
| CMake | 3.18+ | ✅ | Build system |
| C++ Compiler | C++17 (GCC ≥ 9, Clang ≥ 10, MSVC ≥ 2019) | ✅ | |
| CUDA Toolkit | 11.0+ | ❌ | opt-in via `NANOINFER_USE_CUDA=ON` |
| Python 3 | 3.8+ | ❌ | Only needed for Python bindings |
| NumPy | any | ❌ | Only needed for Python bindings |
| cuBLAS | bundled with CUDA Toolkit | ❌ | opt-in via `USE_CUBLAS=ON` |
| OpenBLAS | 0.3+ | ❌ | opt-in via `USE_OPENBLAS=ON` |
| OpenMP | any (compiler built-in) | ❌ | auto-detected for CPU parallelism |

### Platform-specific Install

**Ubuntu / Debian:**
```bash
sudo apt install cmake build-essential
# Optional: CUDA Toolkit — download from https://developer.nvidia.com/cuda-downloads
# Optional: OpenBLAS
sudo apt install libopenblas-dev
```

**Arch Linux:**
```bash
sudo pacman -S cmake base-devel
# Optional: CUDA Toolkit
sudo pacman -S cuda
# Optional: OpenBLAS
sudo pacman -S openblas
```

**macOS (CPU-only):**
```bash
brew install cmake
# Optional: OpenBLAS
brew install openblas
```

**Verify toolchain:**
```bash
cmake --version                 # ≥ 3.18
g++ --version                   # C++17
nvcc --version                  # CUDA ≥ 11.0 (if using GPU)
python3 --version               # ≥ 3.8
```

## Runtime Dependencies

### Python Bindings (always required for Python API)

| Package | Minimum Version | Notes |
|---------|----------------|-------|
| Python 3 | 3.8+ | |
| NumPy | any | `pip install numpy` |

### Python Examples (optional)

| Example Script | Package | Install |
|---------------|---------|---------|
| `minicpmv_cli_inference.py` | Pillow | `pip install Pillow` |
| `minicpmv_cli_inference.py` *(video)* | decord | `pip install decord` |
| `bench_backend.py` | tabulate | `pip install tabulate` |
| Report module | matplotlib, pandas | `pip install matplotlib pandas` |

## Dependency Graph

```
┌──────────────────────────────────────────────────────────┐
│                    nanoinfer (.so / .exe)                 │
├─────────────┬──────────────┬──────────────┬──────────────┤
│ nanoinfer_  │ nanoinfer_   │ nanoinfer_   │ nanoinfer_   │
│ core        │ ops          │ model        │ tokenizer    │
├──────┬──────┼──────┬───────┤              │              │
│ C++  │ CUDA │ C++  │ CUDA  │              │              │
│ STL  │ rt   │      │ kernels│              │              │
│      │      │      │ .cu   │              │              │
│      │      │      ├───────┤              │              │
│      │      │      │cuBLAS │              │              │
│      │      │      │(opt)  │              │              │
├──────┴──────┴──────┴───────┴──────────────┴──────────────┤
│ OpenMP (opt) │ OpenBLAS (opt) │ AVX2 (auto) │ C++ STL    │
└──────────────────────────────────────────────────────────┘
```

## Build Scenarios

| Scenario | Command | USE_CUDA | USE_CUBLAS | USE_OPENBLAS |
|----------|---------|----------|------------|--------------|
| GPU + cuBLAS (default) | `cmake -B build` | auto-detect | ON | OFF |
| GPU, pure CUDA kernels | `cmake -B build -DUSE_CUBLAS=OFF` | auto-detect | OFF | OFF |
| CPU-only | `cmake -B build -DNANOINFER_USE_CUDA=OFF` | OFF | — | OFF |
| CPU + OpenBLAS | `cmake -B build -DNANOINFER_USE_CUDA=OFF -DUSE_OPENBLAS=ON` | OFF | — | ON |

> See [Build Guide](build.md) for detailed build instructions.

## CUDA Architecture Detection

Since v0.5.0, CUDA architecture is auto-detected at configure time:

```cmake
cmake -B build
# Output: -- Detected CUDA architecture: 89
```

The detection compiles a small CUDA program that calls `cudaGetDeviceProperties` on the actual GPU. If detection fails (e.g., on a headless server), the default `86;89` is used. To override:

```bash
cmake -B build -DNANOINFER_CUDA_ARCH="90"         # RTX 5090
cmake -B build -DNANOINFER_CUDA_ARCH="75;86;89"   # portable binary
```

## Backend Summary

| Backend | Type | Auto-detected | Enabled By |
|---------|------|---------------|------------|
| CUDA | GPU kernel execution | ❌ (requires toolkit) | `CUDAToolkit_FOUND` |
| cuBLAS | GPU FP32 GEMM | ❌ (requires CUDA) | `USE_CUBLAS=ON` |
| AVX2 | CPU SIMD | ✅ at configure time | `COMPILER_SUPPORTS_AVX2` |
| OpenMP | CPU threading | ✅ at configure time | `OpenMP_CXX_FOUND` |
| OpenBLAS | CPU FP32 GEMM | ❌ (requires library) | `USE_OPENBLAS=ON` |
