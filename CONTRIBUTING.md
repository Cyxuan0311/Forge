# Contributing to Forge

Welcome! We appreciate your interest in contributing. This document covers the development workflow, code style, and how to add new models or operators.

## Table of Contents

- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Workflow](#pull-request-workflow)
- [Adding a New Model Architecture](#adding-a-new-model-architecture)
- [Adding a New Operator](#adding-a-new-operator)
- [Documentation](#documentation)

---

## Development Setup

### Prerequisites

- CMake ≥ 3.18
- C++17 compatible compiler (GCC 11+, Clang 14+, MSVC 2022)
- CUDA Toolkit ≥ 11.0 (required for GPU backend)
- Python 3.8+ with NumPy

### Native Build

```bash
git clone https://github.com/yourname/Forge.git
cd Forge
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### Docker

```bash
docker build -t forge-dev .
docker run --gpus all -it --rm -v $(pwd):/workspace forge-dev
```

### VS Code

The repository includes `.clang-format` and `CMakeLists.txt` for IDE integration. Recommended extensions:
- C/C++ (ms-vscode.cpptools)
- CMake Tools (ms-vscode.cmake-tools)
- clangd (llvm-vs-code-extensions.vscode-clangd)
- Python (ms-python.python)
- Ruff (charliermarsh.ruff)

---

## Code Style

### C++

We follow the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) with modifications defined in [`.clang-format`](.clang-format):

```bash
# Check formatting
find . \( -name '*.cpp' -o -name '*.h' -o -name '*.cu' -o -name '*.cuh' \) \
  -not -path './build/*' -not -path './_deps/*' \
  -exec clang-format --dry-run --Werror {} +

# Auto-format
find . \( -name '*.cpp' -o -name '*.h' -o -name '*.cu' -o -name '*.cuh' \) \
  -not -path './build/*' -not -path './_deps/*' \
  -exec clang-format -i {} +
```

Key conventions:
- 4-space indentation, no tabs
- 100-column limit
- Left pointer alignment (`int* p`)
- `camelCase` for functions, `PascalCase` for classes, `snake_case` for variables

### Python

We use [ruff](https://docs.astral.sh/ruff/) for linting:

```bash
ruff check .
ruff format .
```

### CMake

We use [cmake-format](https://github.com/cheshirekow/cmake_format) for `.cmake` and `CMakeLists.txt` files:

```bash
cmake-format --check CMakeLists.txt
```

---

## Testing

### Running Tests

```bash
# All tests (requires a CUDA-capable GPU)
python -m pytest tests/ -v

# CPU-only tests
python -m pytest tests/ -v -k "not gpu"

# Quick unit tests (no external model)
python -m pytest tests/ -v -k "not tinyllama and not gpu" \
  --ignore=tests/test_tinyllama_integration.py

# Single test file
python -m pytest tests/test_tensor.py -v

# With timeout
python -m pytest tests/ -x --timeout=120
```

### Test Requirements

- **Unit tests** (`test_tensor.py`, `test_backend.py`, etc.): No external model needed. Uses a small synthetic NINF test model.
- **Integration tests** (`test_tinyllama_integration.py`): Require downloading a TinyLlama GGUF model.
- **GPU tests** (`test_gpu.py`, `test_gguf_gpu.py`): Require a CUDA-capable GPU.

### Adding Tests

- Place new tests in `tests/` following the existing naming convention.
- Use pytest fixtures from `tests/conftest.py`.
- For operator tests, compare CPU and CUDA outputs to verify correctness.

---

## Pull Request Workflow

1. **Fork** the repository on GitHub.
2. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feat/my-feature
   ```
3. **Make your changes**, following the code style guidelines.
4. **Run tests** locally:
   ```bash
   python -m pytest tests/ -v -k "not tinyllama and not gpu" --timeout=120
   ```
5. **Commit** with a clear message:
   ```
   feat(engine): add support for Llama 4 attention
   fix(operator): correct RoPE frequency calculation for Qwen2
   docs(build): update CUDA architecture table
   ```
6. **Push** and open a Pull Request against `main`.
7. **Ensure CI passes** — all lint and build checks must succeed.

### PR Checklist

Before submitting, ensure:
- [ ] Code follows the existing style (clang-format)
- [ ] Self-review completed
- [ ] Tests added for new functionality
- [ ] Existing tests pass (`python -m pytest tests/ -v`)
- [ ] Build completes on Linux with CUDA
- [ ] Documentation updated (if applicable)

---

## Adding a New Model Architecture

Forge's plugin system makes adding new architectures straightforward. Here's the step-by-step process:

### 1. Register the architecture

Add a call to `FORGE_REGISTER_ARCH` that ties together the architecture name, engine, config parser, and weight init:

```cpp
FORGE_REGISTER_ARCH("my_model", MyModelEngine, my_model_config_parser,
                         my_model_weight_init, { /* capabilities */ });
```

### 2. Implement the engine

Create a new engine class inheriting from `TransformerEngine` (or `InferenceEngine` for non-transformer models):

```cpp
class MyModelEngine : public TransformerEngine {
  bool forward(const Tensor& hidden, const LayerWeights& w,
               const KvCache& cache, int layer) override;
  std::string_view name() const override { return "my_model"; }
};
```

Register it:
```cpp
FORGE_REGISTER_ENGINE("my_model", MyModelEngine);
```

### 3. Implement the graph builder (optional)

For graph-based execution:
```cpp
class MyModelGraphBuilder : public LayerGraphBuilder {
  std::unique_ptr<ComputeGraph> build(const ModelConfig& config) override;
};
FORGE_REGISTER_GRAPH_BUILDER("my_model", MyModelGraphBuilder);
```

### 4. Implement config parser

Parse architecture-specific parameters from `ModelConfig`:
```cpp
void my_model_config_parser(ModelConfig& config, const WeightStore& store);
FORGE_REGISTER_CONFIG_PARSER("my_model", my_model_config_parser);
```

### 5. Implement weight init / mapping

Map weight names to the canonical scheme:
```cpp
void my_model_weight_init(LayerWeights& w, const WeightStore& store);
FORGE_REGISTER_WEIGHT_INIT("my_model", my_model_weight_init);
```

### 6. Add tests

- Add a unit test in `tests/test_engine.py`
- If possible, add an integration test with a real model
- Verify forward pass reproducibility

### 7. Update model support table

Add the new architecture to `docs/models.md` and `docs/models_zh.md`.

---

## Adding a New Operator

### 1. Define the op interface

```cpp
class MyOp : public Op {
  std::vector<Tensor> compute(const std::vector<Tensor>& inputs) override;
  std::string_view name() const override { return "my_op"; }
};
FORGE_REGISTER_OP("my_op", MyOp);
```

### 2. Implement CPU kernel

Place in `src/operators/` with AVX2 optimizations in `src/operators/cpu/`.

### 3. Implement CUDA kernel

Place in `src/operators/cuda*.cu` or `src/operators/cuda/`. Declare the kernel launcher in `include/forge/cuda_kernels.h`.

### 4. Add to dispatch

Wire the operator into the dispatch system in `dispatch.h`.

### 5. Add tests

Test correctness by running the operator on both CPU and CUDA and comparing outputs.

---

## Documentation

- All user-facing features must have both English and Chinese documentation.
- Documentation files live in `docs/` with the `_zh.md` suffix for Chinese.
- When adding a new model, update:
  - `docs/models.md` / `docs/models_zh.md`
  - `docs/architecture.md` / `docs/architecture_zh.md` (if new engine/mechanism)
  - `README.md` / `README_zh.md` (if listed in the main page)

---

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
