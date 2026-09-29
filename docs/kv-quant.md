# KV cache quantization

Forge can store the attention K and V caches at different precisions. This
document covers the flags, the recommended pairings, the constraints, and where
the throughput ceiling currently is.

## Flags

```
-ctk, --cache-type-k TYPE    KV cache dtype for K only
-ctv, --cache-type-v TYPE    KV cache dtype for V only
     --kv-cache-dtype TYPE   KV dtype for BOTH K and V (default: f16)
```

`-ctk` and `-ctv` win over `--kv-cache-dtype`; giving only one of them applies it
to both, so `-ctk q8_0` on its own means "q8_0 for everything".

Supported values, and only these — anything else is a hard error naming the
value, never a silent downgrade:

| name | bytes / element | block | notes |
|---|---|---|---|
| `fp32` | 4 | – | baseline |
| `f16` | 2 | – | the default |
| `q8_0` | ~1.06 | 32 | best quality per byte here |
| `q4_0` | ~0.56 | 32 | coarse; mainly useful on V |
| `q4_k` | ~0.56 | 256 | needs `head_dim % 256 == 0`, so it is unavailable on most models |
| `fp8_e4m3` | 1 | – | |
| `fp8_e5m2` | 1 | – | |

llama.cpp's `-ctk` also accepts `bf16`, `q4_1`, `q5_0`, `q5_1` and `iq4_nl`.
Forge has no dequantiser for those, so they are rejected with a message rather
than quietly becoming f16.

`forge-cli` and `forge-server` take the same flags and resolve them through the
same parser, so the two cannot disagree. Both report the resolved pair at
startup, e.g. `KV Cache: dtype=q8_0/q4_0 (K/V), size=68 MB`.

## Why K and V are not equivalent

At the **same bit width**, quantising K costs more quality than quantising V.

K feeds `softmax(Q·K/√d)`. An error there lands in an exponent, so it is
amplified and there is no mechanism downstream to absorb it. V enters the output
as a convex combination `Σᵢ pᵢ·Vᵢ` — a linear path, and an error shared
across the rows that combine tends to average out rather than compound.

Measured as KLD against an f16/f16 baseline (Qwen3.5-9B):

| K | V | KLD |
|---|---|---|
| f16 | f16 | 0.000782 |
| f16 | q4_0 | 0.003747 |
| q4_0 | f16 | 0.005426 |
| q5_0 | f16 | 0.002223 |
| q8_0 | f16 | 0.000923 |
| q8_0 | q5_0 | 0.001715 |
| q5_0 | q8_0 | 0.002310 |
| q4_0 | q4_0 | 0.007979 |

Note the second and third rows: `q4_0` on **V** is better than `q4_0` on **K**
(0.003747 vs 0.005426), and symmetric `q4_0` is worse than either. So the cheap
quantisation belongs on V, and the recommendation follows from that.

`forge-cli`/`forge-server` warn when K is quantised and V is not.

## Recommended pairings

`-ctk q8_0 -ctv q4_0` is the default suggestion. On MiniCPM5-2B (42 layers,
`kv_dim` 256, `-c 4096`):

| K / V | KV bytes |
|---|---|
| `f16` / `f16` | 168 MB |
| `q8_0` / `q8_0` | 89 MB |
| `f16` / `q4_0` | 107 MB |
| **`q8_0` / `q4_0`** | **68 MB** |
| `q4_0` / `q4_0` | 47 MB |

`q8_0/q4_0` is 2.5x smaller than f16 for about 2% throughput, and lands
between the two symmetric options in quality — it keeps K at the precision the
softmax actually cares about and spends the savings on V.

Going all the way to `q4_0/q4_0` saves another 21 MB but puts the
highest-sensitivity tensor at the coarsest setting; prefer `f16` or `q8_0` for K
unless the context simply does not fit.

## Constraints

- **Head dim must be a multiple of the block size**, checked per layer at startup
  for K and V separately. `q8_0`/`q4_0` need `head_dim % 32 == 0`; `q4_k` needs
  `% 256 == 0`, which excludes it on the common 128-head-dim models. This is not
  cosmetic: the row size is rounded **up** to a whole block, so a non-divisible
  head dim yields rows longer than the fused kernels assume and they read
  misaligned. The check exists to turn that into an error at startup.
- **MLA-style architectures** (DeepSeek-V2/V3/V4, Kimi, and the Gemma4 latent
  path here) store V inside the K tensor, so `type_k` and `type_v` must match.
  Asymmetric is rejected for those.

## Current throughput ceiling

All fused quantized attention kernels are monomorphic: one row size, one block
layout, one dtype for both K and V. An asymmetric pair therefore has no fused
kernel and falls back to materialising the filled rows to FP32 before the
attention. The result is **correct** — only rows `[dequantized_filled, filled)`
are re-materialised — but it costs memory traffic, and it is measurably noisier:

```
MiniCPM5-2B, -c 16384, ~1.5k-token prompt, 120 generated tokens, 3 runs
-ctk q8_0 -ctv q8_0   22.9 / 22.7 / 22.7 tok/s   (fused)
-ctk q8_0 -ctv q4_0   21.6 / 26.1 / 21.2 tok/s   (materialise fallback)
```

Comparable median, wider spread. Forge prints a one-line warning when this path is
taken so it is never mistaken for a regression:

```
[Forge WARN] asymmetric KV cache (K=q8_0, V=q4_0): no fused mixed-dtype
attention kernel yet, falling back to FP32 materialization of the filled rows.
```

Mixed-dtype fused kernels (`<TypeK, TypeV>` with independent row sizes) are the
follow-up that removes this branch.

## What is not implemented yet

llama.cpp improves quantised KV quality substantially by rotating activations
with an orthonormal Hadamard matrix before caching and rotating the attention
output back (upstream PR #21038). Because the rotation is a linear isometry it
does not change the exact result, but it flattens outliers before quantisation,
which matters a great deal at 4 bits:

| dtype | PPL before | PPL after |
|---|---|---|
| q8_0 | 13.91 | 13.67 |
| q5_1 | 61.70 | 14.15 |
| q4_1 | 212.48 | 22.28 |
| q4_0 | 62.02 | 46.25 |

(Qwen3-0.6B, f16 baseline 13.67.) The cost is four extra matrix multiplies in
attention: roughly 2–3% on prefill and 7–12% on decode.

Forge does not do this yet, so the aggressive settings above should be treated as
memory-first, quality-second. `q8_0` on K is the conservative choice and is
where most of the win is anyway.
