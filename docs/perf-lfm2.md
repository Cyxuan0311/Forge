# LFM2.5-2.6B inference performance

## Read this before trusting any number on this machine

The RTX 4050 Laptop **bistables between roughly 1755 MHz and 2400 MHz** under software
power capping -- `nvidia-smi -q -d PERFORMANCE` reports `SW Power Cap: Active` and
`SW Thermal Slowdown: Active` (a laptop dGPU power policy, visible here because the box
runs under WSL2). Decode is clock-bound, and throughput tracks the clock ratio exactly:

| SM clock during run | tok/s |
|---------------------|-------|
| 1755-1770 MHz       | 35.7 - 36.4 |
| 2265 MHz            | 49.1 |
| 2415 MHz            | 49.3 |

2400 / 1755 = 1.37, and 49.1 / 35.8 = 1.37. A single run can therefore read up to 1.7x
faster than its neighbour for reasons that have nothing to do with the code.

Practical rules:

- Compare **ratios**, not absolute tok/s. Absolute figures do not reproduce here.
- Interleave the two arms of an A/B within the same loop and report the median of
  several reps. A one-shot measurement of DSpark speculative decoding made it look 24%
  slower than plain decode; the controlled median is 0.97x, i.e. parity.
- Sanity-check with `nvidia-smi -q -d CLOCK,PERFORMANCE` (sampled mid-run) before
  believing a surprising result. Note `nvidia-smi --query-gpu=clocks.sm,...` is **not**
  supported by this driver; use `-q -d`.

Effects that survive interleaving and tight error bars (e.g. the speculative path being
1.9x slower with a 32768-token KV cache than a 2048-token one, with non-overlapping
ranges) are real and worth chasing. Effects that move run to run are the clock.

Measured on the reference box: **i5-13500H** (8C/16T, AVX2 + F16C, **no** AVX-512/AMX),
15 GB RAM, **RTX 4050 Laptop** 6 GB (sm_89, measured HBM read roof 164 GB/s), model on an
external NTFS volume sustaining ~138 MB/s. Model is `arch=lfm2`, 30 layers
(8 full-attention + 22 shortconv), dense FFN, **no MoE** -- so every generated token streams
essentially the whole model.

## Baseline, all three GGUFs

| Model | Size | GPU tok/s | CPU tok/s |
|-------|------|-----------|-----------|
| Q8_0  | 2.9 G | 43.1 | 9.4 |
| F16   | 5.4 G | 29.8 | 7.7 (was 4.9) |
| BF16  | 5.4 G | 29.5 | 5.4 (was 3.8) |

## GPU: already at the memory roof. Do not optimise this.

The original premise for this work was "F16 GPU decode runs at 5.5 tok/s while Q8_0 runs
at 43", a supposed 7x cliff. That 5.5 was a measurement artifact of a different
configuration. Re-measured back to back under identical flags, F16 is ~29.8.

Effective weight bandwidth:

| Model | GB / ms-per-token | GB/s | % of measured 164 GB/s read roof |
|-------|-------------------|------|----------------------------------|
| Q8_0  | 2.9 / 23.2        | 125  | 76% |
| F16   | 5.4 / 33.6        | 161  | **98%** |
| BF16  | 5.4 / 33.9        | 159  | **97%** |

The 2-byte path achieves a *higher* bandwidth than the 1-byte one, and the whole F16/Q8_0
speed gap is what the 1.86x data ratio predicts. There is nothing for a fused FP16/BF16
GEMV to win. `forge-bench-gpu_bandwidth` reproduces the roof:

```
read-only (decode's roof)   6.55 ms   164.0 GB/s
device-to-device copy      12.33 ms   174.2 GB/s (r+w)
theoretical peak                       192.0 GB/s   (85.4% achieved)
```

Two things that bench got wrong first, kept in the source because they are easy to get
wrong again: a fixed 2048-block grid does not saturate a 20-SM part (this understates the
roof and makes a memory-bound decode look like it has headroom), and "read-only" cannot be
inferred by halving a read+write figure.

**If GPU headroom is ever wanted, it is on the Q8_0 path (76%), not the 2-byte one.**

## CPU: F16/BF16 were paying 2x the memory traffic

An F16/BF16 GGUF was expanded to FP32 at load, so 5.4 GB of weights was read as 10.8 GB
per generated token -- on a 15 GB host, 10.8 GB of resident weights leaves nothing for
anything else.

`ops::matmul_transB` now has a widening GEMV for 2-byte weights (AVX2 + F16C, `_mm256_cvtph_ps`
for FP16; for BF16 a zero-extend plus a 16-bit shift, which is exact because bf16 is the
top half of a float32; FP32 accumulation either way). Only the 2D matmul weights keep their
native width -- see the scope note in `gguf_model.cpp`, because everything else is read
through a `float*` and that read fails silently.

| Model | before | after | speedup | peak RSS |
|-------|--------|-------|---------|----------|
| F16   | 4.9 tok/s | **7.3** | 1.49x | 12.8 GB -> **6.8 GB** |
| BF16  | 3.8 tok/s | **5.4** | 1.42x | -- |

Output is byte-identical either way. The RAM halving may matter more than the speed: at
12.8 GB the F16 model left ~2 GB of headroom, which is a hard ceiling on model size.

`FORGE_CPU_KEEP_2BYTE=0` restores expand-everything for any model whose 2D weights turn out
to be read as `float` somewhere the rule did not anticipate.

## CPU: what is actually left, and why it was not pursued

Q8_0 on CPU reaches 2.9 GB x 9.4 tok/s = **27 GB/s**, against 41.6 GB/s that the native-F16
path demonstrates on the same machine. That looks like ~1.5x of headroom in the Q8_0 kernel,
and two candidate fixes were implemented and measured:

1. **Hoisting the per-block activation sums out of the row loop.** `maddubs_epi16` is
   (unsigned x signed), so the weight is XORed with 0x80 and a `+128` bias is removed
   afterwards with `128 * sum(act)`. `sum(act)` depends only on the activation, so it looked
   like ~3 of the ~11 vector ops per 32-element block were being redone for each of the
   N rows (10752 for this model's FFN). **This is wrong on two counts and was reverted.**
   The correction is a *per-lane partial* sum, not a per-block total -- broadcasting the total
   produced garbage ("POT apro PRI miscalc RIS Mock..."). And hoisting it would not have paid
   anyway: it means storing 8 int32 per block (32 B against 36 B of activation), nearly
   doubling activation traffic, to save 3 ALU ops on a value already sitting in a register.
2. **4-row unroll of the Q8_0 GEMV**, matching the FP32 and native-F16 paths instead of
   `parallel_for(N)` with one row per task. Measured 8.5 vs 9.1 tok/s median over 7 runs
   each -- no gain, so it was reverted rather than committed.

The real observation is more interesting than either fix: **CPU per-token time is roughly
constant (~110-130 ms) whether the model is 2.9 GB or 5.4 GB.** So CPU decode is not
weight-bandwidth-bound the way GPU decode is, and the Q8_0 kernel is not the thing limiting
it. Something fixed-cost per token dominates -- 30 layers of non-matmul work (attention,
norms, sampling, 22 shortconv layers) plus thread-pool synchronisation across 90+ matmuls
per token on 8 threads.

**Methodology warning.** CPU numbers on this box have a p25-p75 spread of roughly 50% at
n=7, so micro-optimisations below ~20% cannot be validated from model runs. Use 48+ tokens
and medians, never single runs -- and prefer an in-process benchmark with many iterations
for anything smaller. GPU numbers are far tighter because there is no host scheduling in the
loop.

Also measured, not acted on: model load takes ~90 s while the disk sustains 138 MB/s
(5.4 GB in ~39 s), so ~50 s is loader overhead. That is a startup/UX cost, not decode.
