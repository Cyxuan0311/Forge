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
