# CLI Usage

[中文](cli_zh.md)

## Basic Usage

```bash
./forge-cli -m <model_path> [options]
```

## Options

### Model & Input

| Option | Default | Description |
|--------|---------|-------------|
| `-m, --model PATH` | (required) | Path to GGUF / NINF model file |
| `-p, --prompt TEXT` | — | Input prompt (non-interactive mode) |
| `-n, --n-predict N` | `256` | Maximum tokens to generate |
| `--mmproj PATH` | — | Multimodal vision encoder path (for MiniCPM-V) |
| `--image PATH` | — | Input image for multimodal model |

### Performance

| Option | Default | Description |
|--------|---------|-------------|
| `-ngl, --n-gpu-layers N` | `-1` | Number of layers to offload to GPU (`-1` = all, `0` = CPU only) |
| `-t, --threads N` | auto | Number of CPU threads |
| `-b, --batch-size N` | `512` | Max tokens per prefill forward (sets `ContextParams::n_batch`) |
| `--kv-cache-dtype TYPE` | `f16` | KV dtype for **both** K and V: `fp32`, `f16`, `q8_0`, `q4_0`, `q4_k`, `fp8_e4m3`, `fp8_e5m2` |
-rkq, --rope-k-quant    Hadamard-rotate Q/K/V before a quantized KV cache (-nrkq to disable)
| `-ctk, --cache-type-k TYPE` | | KV dtype for **K only** (overrides `--kv-cache-dtype`) |
| `-ctv, --cache-type-v TYPE` | | KV dtype for **V only** (overrides `--kv-cache-dtype`) |

### Sampling

| Option | Default | Description |
|--------|---------|-------------|
| `--temp FLOAT` | `0.7` | Sampling temperature (`0` = greedy) |
| `--top-k N` | `40` | Top-K sampling (`0` = disabled) |
| `--top-p FLOAT` | `0.9` | Top-P (nucleus) sampling |
| `--repeat-penalty FLOAT` | `1.1` | Repetition penalty |
| `--seed N` | random | Random seed for reproducibility |

### Mode

| Option | Default | Description |
|--------|---------|-------------|
| `-i, --interactive` | off | Interactive chat mode |
| `--no-stream` | off | Disable streaming output |
| `--system-prompt TEXT` | — | System prompt for chat |

### Info & Debug

| Option | Description |
|--------|-------------|
| `--info` | Show model information and exit |
| `--bench` | Run performance benchmark |
| `-v, --verbose` | Enable verbose logging |
| `-h, --help` | Show help message |

### `forge-inspect`

A separate tool for GGUF analysis. It reads a model file and reports its
metadata, tensor layout, dtype histogram, operator pipeline, and an optional
KV/VRAM estimate.

```bash
# human-readable summary
forge-inspect model.gguf

# will it fit? assume 4K context and a 6 GiB card
forge-inspect -M --ctx 4096 --vram 6144 model.gguf

# q8_0 KV halves the cache versus f16
forge-inspect -M --ctx 4096 --kv-dtype q8_0 model.gguf

# machine-readable, for scripting
forge-inspect -M -f json model.gguf | jq .memory

# will this file even load? non-zero exit on errors, so CI can gate on it
forge-inspect -V model.gguf

# what changed between an F16 and a BF16 conversion of the same model?
forge-inspect -d model-f16.gguf model-bf16.gguf
```

| Option | Description |
|--------|-------------|
| `-M, --memory` | KV cache + VRAM estimate |
| `--ctx N` | Context length to assume (default: the model's trained length) |
| `--kv-dtype D` | `f16`, `fp32`, `q8_0`, `q4_0`, `q4_k`, `fp8_e4m3`, `fp8_e5m2` |
| `--vram MB` | Budget to check against; reports over/under and how many layers to offload |
| `-f, --format` | `text`, `table`, `markdown`, `json`, `yaml`, `csv`, `ini` |
| `-d, --tensor-diff` | Per-tensor diff against a second model (needs two paths) |
| `-V, --validate` | Cross-check the metadata against the tensors; exit 2 on errors |
| `--metadata` | Dump all metadata key/value pairs |
| `-t, --tree [D]` | Tensor tree to depth `D` |

The estimate follows the same accounting Forge itself uses, so the number
matches the engine's own `KV Cache` report. Quantized sizes include the
per-block scale and round each row up to a whole block, and the dtype list is
exactly the set the engine can store — there is no BF16 or K-quant KV cache.
Hybrid architectures (LFM2, Qwen3.5) also report how many layers actually hold
attention, since that is often far fewer than the layer count.

`--tensor-diff` compares the two models' tensor indexes: which tensors exist in
only one of them, and which shared tensors differ in dtype, shape or size. It
reads the GGUF index, not the weights, so two models with identical indexes but
different values are reported as identical. `--top N` bounds the listing (the
counts are unaffected).

`--validate` cross-checks the header against the tensors: the layer count against
the layers actually present, head geometry, duplicate tensor names, the vocabulary
against the embedding matrix, and the tensor extents against the file length. It
exits 2 if any of those is an error, so it can gate a download or a CI step.

## Examples

### Interactive Chat

```bash
./forge-cli -m model.gguf

# Type messages directly. Commands: /quit, /clear, /system, /help
```

### Single Prompt

```bash
./forge-cli -m model.gguf -p "What is machine learning?"
```

### CPU-Only Inference

```bash
./forge-cli -m model.gguf -ngl 0
```

### Custom Sampling Parameters

```bash
./forge-cli -m model.gguf -p "Write a poem about AI" \
  --temp 0.9 --top-p 0.95 --repeat-penalty 1.2 -n 512
```

### Multimodal (Vision-Language)

```bash
./forge-cli -m minicpmv.gguf --mmproj mmproj.gguf \
  --image photo.jpg -p "Describe this image"
```

### Benchmark Mode

```bash
./forge-cli -m model.gguf --bench
```

### Show Model Info

```bash
./forge-cli -m model.gguf --info
```

## Interactive Commands

| Command | Description |
|---------|-------------|
| `/quit` or `/exit` | Exit the chat |
| `/clear` | Clear conversation history |
| `/system TEXT` | Set system prompt |
| `/image PATH` | Load an image for multimodal models |
| `/save PATH` | Save conversation to file |
| `/help` | Show help |

## Performance Tuning

- **GPU**: Use `--n-gpu-layers -1` to offload all layers. Set `-ctk q8_0 -ctv q4_0` to cut KV memory ~2.5x vs f16 with far less quality loss than quantising K (see `kv-quant.md`).
- **CPU**: Increase `--threads` to match your CPU core count. Use `--batch-size 1024` for faster prefill.
- **Sampling**: Lower `--temp` for more deterministic output; higher for more creativity.
