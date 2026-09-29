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

## Throughput

Asymmetric pairs run a fused mixed-dtype kernel: K and V are templated
independently, each with its own block geometry and its own row size, so the head
slice is read at the right offset on each side. Pairs outside `head_dim ∈
{64, 96, 128}` and the `{q8_0, q4_0, f16}` formats are refused by the launcher
and fall back to materialisation, with a warning — never silently.

```
MiniCPM5-2B, -c 16384, ~1.5k-token prompt, 120 generated tokens, 5 runs each
-ctk q8_0 -ctv q8_0   median 22.6 tok/s   spread 1%    (symmetric, unchanged)
-ctk q8_0 -ctv q4_0   median 21.2 tok/s   spread 1%
-ctk f16  -ctv q4_0   median 20.4 tok/s
```

Two things to be clear about:

- **Asymmetric is not faster.** It is ~6% slower than the symmetric fused path at
  equal configuration. The q4_0 unpacking costs more ALU than the 40% smaller V
  rows save in a launch-bound decode. Choose `q8_0/q4_0` for **memory**, not speed.
- The win is over the old behaviour. An asymmetric pair used to have no fused
  kernel at all and fell back to materialising the filled rows to FP32, which
  measured 21.2–26.1 tok/s across runs. The mixed kernel holds 1% spread. So it
  buys stability and the memory saving, not throughput.

Note also that generated-text hashes are **not** a valid regression signal for
any of this: greedy decoding on this GPU path is not bit-reproducible run to run
(the same binary and flags produce different output across runs). Correctness of
the kernels is verified numerically instead — `forge-mixed-attn-test` fills a real
KV cache through the same `quantize_row()` the engine uses and compares against
attention computed on the CPU from the dequantized rows.

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

## Hadamard rotation (`-rkq`)

Quantization error is set by the largest value in each block, so a few large
activations degrade all 31 of their neighbours. Rotation fixes that at the cost
of three small transforms per layer: with `R` orthonormal, `(Rq).(Rk) = q.k` so
caching `Rk` and rotating `q` the same way leaves every logit identical, and
`sum_i p_i (R v_i) = R (sum_i p_i v_i)` so the output is `R` applied to the true
output and `R^-1` recovers it. A Hadamard matrix is used because it is symmetric
(hence self-inverse) and spreads any single large component over all
coordinates instead of leaving it to dominate one block.

`-rkq` / `--rope-k-quant` enables it; it is off by default, and it only engages
when K or V is actually quantized, since an f16 cache has nothing to gain.

Quality, 8 factual prompts x 4 runs, scored on whether the expected answer
appears (greedy decoding here is not bit-reproducible, so a single run proves
nothing; 24/32 is the ceiling, since 8 prompts are missed by f16 too):

| K / V        | without `-rkq` | with `-rkq` |
|--------------|----------------|--------------|
| f16 (ref)    | 24/32          | –            |
| `q8_0/q8_0`  | 24/32          | 24/32        |
| `q8_0/q4_0`  | 20/32          | **24/32**    |
| `q4_0/q4_0`  | 24/32          | 24/32        |
| `f16/q4_0`   | –              | 24/32        |

So it closes the whole `q8_0/q4_0` gap to f16 and is neutral elsewhere — which
is the right shape, since a 4-bit V is the only case here that loses anything.
Cost is about 1% (median 21.6 → 21.4 tok/s at `-c 16384`), well under the
2–3% prefill / 7–12% decode llama.cpp reports, because this is a
Walsh-Hadamard butterfly at O(n log n) rather than a dense O(n²) product.

Rotation is **not** unconditionally good, and the test deliberately does not
assert that it reduces quantization error — it prints the measured numbers
instead. For a per-block-amax quantiser the effect changes sign with the outlier
ratio (8x outlier: 2.8x better; 1000x: 33x *worse*), because spreading a dominant
coordinate raises the scale of every block it touches. On this model's real K
the effect is favourable — measured 1.7x lower `q4_0` K reconstruction error
across all 44 layers, none worse — but that is an empirical property of the
model, not a theorem, which is why it is a flag.

## q4_0 has two different encodings — read this before touching a KV reader

Forge does not agree with itself on how a `q4_0` KV row is encoded:

| path | writer | nibble encoding |
|------|--------|-----------------|
| CPU cache | `quantize_q4_0_cpu()` | **sign-extended** int4, `d = amax / -8` |
| CUDA cache | `launch_quantize_q4_0_matrix()`, `quantize_row_q4_0()` | **unsigned, offset by 8**, `d = amax / -8` |

The scale is negative in both. The payload is not. The two are different
functions — they differ by 8 on *every* element. A CUDA layer is written by the
CUDA writer, so the fused attention kernels must decode with `(n & 0xF) - 8`.
`dequantize_q4_0_cpu()` will misread a CUDA-written row.

This has already cost one bad commit. The CPU/CUDA split was mistaken for a
single bug ("the CUDA kernels use the weights convention, that is wrong"),
"fixed" to sign extension, and that broke every `q4_0` KV cache end to end
(answer accuracy 24/32 → 4/32). The commit that introduced the break was
validated by a unit test that filled the cache with the CPU quantizer and
compared against the CPU dequantizer — a test of CPU-writer against CPU-reader,
which cannot detect a CPU/CUDA mismatch. `forge-mixed-attn-test` now writes
through the engine's actual writer and reads back through the decode the kernel
performs.

The split itself is still there and is still latent. Reconciling it is a
deliberate decision with its own migration story, not something to slip into a
bug fix.

## A note on verifying changes here

Do not validate KV-quantization work by diffing generated text against a previous
build. Greedy decoding on this GPU path is not bit-reproducible run to run — the
same binary and flags produce different text across runs — so such a diff
measures noise. Worse, a *broken* quantized path frequently still produces
fluent text, because softmax absorbs a bounded per-element error; every
regression in the history above passed a prose check.

Use instead:

- `forge-mixed-attn-test` — each kernel against a reference built from the
  dequantized rows, written through the engine's real writers.
- `forge-attn-rot-test` — orthonormality, self-inverse, the dot-product
  identity, the full attention identity, CPU-vs-CUDA agreement.
- A repeated-sample quality score, as in the `-rkq` table above. Sample it
  several times; a single run is noise.
