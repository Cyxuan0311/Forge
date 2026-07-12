# Build & Install

[中文](build_zh.md)

## Dependencies

| Dependency | Minimum Version | Required |
|-----------|----------------|----------|
| CMake | 3.18+ | Yes |
| C++ Compiler | C++17 support | Yes |
| CUDA Toolkit | 11.0+ | Yes |
| Python 3 | 3.8+ | Yes (for bindings) |
| NumPy | Any | Yes (for bindings) |
| OpenMP | Any | Optional (CPU parallelism) |

## Build from Source

```bash
git clone https://github.com/yourname/Forge.git
cd Forge

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

After building:
- `build/forge-cli` — CLI inference executable
- `build/forge*.so` — Python binding module

## Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_BUILD_TYPE` | `Release` | Build type (`Release` / `Debug`) |
| `CMAKE_CUDA_ARCHITECTURES` | `75;86;89` | Target CUDA architectures (comma-separated) |
| `USE_CUBLAS` | `ON` | Use cuBLAS for GPU GEMM; `OFF` uses pure CUDA tiled GEMM kernel |
| `USE_OPENBLAS` | `OFF` | Use OpenBLAS for CPU FP32 GEMM; default uses hand-tuned AVX2 kernels |

### Common Build Variants

**Fully self-contained (no cuBLAS, no OpenBLAS):**

```bash
cmake -B build -DUSE_CUBLAS=OFF -DUSE_OPENBLAS=OFF
cmake --build build -j
```

**With cuBLAS for GEMM acceleration:**

```bash
cmake -B build -DUSE_CUBLAS=ON
cmake --build build -j
```

**With OpenBLAS for CPU FP32 matmul:**

```bash
cmake -B build -DUSE_OPENBLAS=ON
cmake --build build -j
```

**Debug build:**

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
```

### CUDA Architecture Targets

Set `CMAKE_CUDA_ARCHITECTURES` to match your GPU:

| GPU | Architecture |
|-----|-------------|
| RTX 4090, 4080, 3090, 3080 | `86` |
| RTX 5090, 5080 | `89` |
| RTX 3070, 2070, 2080 | `75` |
| Multi-GPU / portable | `75;86;89` |

## AVX2 Detection

CMake automatically detects AVX2 / FMA / F16C support at configure time. If supported, `USE_AVX2` is defined and the hand-tuned AVX2 GEMV/GEMM kernels are enabled. On CPUs without AVX2, scalar fallbacks are used.

## Python Package

After building, the `.so` file can be imported directly:

```python
import sys
sys.path.insert(0, "build")
import forge
```

Or install via pip:

```bash
pip install .
```

## Verify Installation

```python
import forge
print(forge.__version__)  # 0.5.0
```

## Docker

A Dockerfile is provided for reproducible builds:

```bash
docker build -t forge .
docker run --gpus all forge
```

## Troubleshooting

| Problem | Likely Cause | Solution |
|---------|-------------|----------|
| `CUDA not found` | CUDA Toolkit not installed or not in PATH | Install CUDA Toolkit 11.0+ and ensure `nvcc` is on `PATH` |
| `pybind11 not found` | Missing submodule | Run `git submodule update --init` or set `FETCHCONTENT_FETCH_ONE` |
| Slow CPU inference | AVX2 not detected | Check CPU supports AVX2; build on the target machine |
| Linking errors | CUDA architecture mismatch | Set `CMAKE_CUDA_ARCHITECTURES` to match your GPU |
