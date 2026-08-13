"""Phi-mini-MoE decode benchmark with fixed prompt/seed.

Measures decode tok/s for a fixed prompt across N generated tokens,
3 iterations, reporting mean/p50. Also runs a prompt prefill timing.

Usage:
    OMP_NUM_THREADS=16 python3 benchmarks/bench_phimini.py [--threads 16]
"""

import os
import sys
import time
import argparse
import statistics

build_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

import forge
import numpy as np

MODEL_PATH = "/mnt/g/AI/Phi-mini-MoE-instruct-IQ2_S/Phi-mini-MoE-instruct-IQ2_S.gguf"
PROMPT = "The capital of France is Paris and the largest city in the country. It is known for"


def bench_decode(model, tok, gen_len=64, iters=3, warmup=1, do_sample=False):
    prompt_ids = np.array(tok.encode(PROMPT, add_bos=True), dtype=np.int32)
    prompt_len = len(prompt_ids)

    ctx = model.create_context(kv_cache_dtype="fp32", gpu_layers=0)

    def run():
        ctx.reset_kv()
        ctx.forward(prompt_ids)
        last = prompt_ids[-1]
        pos = prompt_len
        for _ in range(gen_len):
            logits = ctx.forward(np.array([last], dtype=np.int32), start_pos=pos)
            if do_sample:
                # simple greedy for determinism
                last = int(np.argmax(logits))
            else:
                last = int(np.argmax(logits))
            pos += 1

    for _ in range(warmup):
        run()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        run()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    mean = statistics.mean(times)
    return {
        "prompt_len": prompt_len,
        "gen_len": gen_len,
        "mean_ms": mean * 1000,
        "p50_ms": statistics.median(times) * 1000,
        "decode_tok_s": gen_len / mean,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-len", type=int, default=64)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(args.threads)

    model = forge.Model()
    model.load_auto(MODEL_PATH, device="cpu")
    tok = forge.Tokenizer()
    tok.load_from_gguf(MODEL_PATH)

    stats = bench_decode(model, tok, gen_len=args.gen_len, iters=args.iters)
    print(f"[threads={args.threads}] prompt_len={stats['prompt_len']} "
          f"gen_len={stats['gen_len']} mean={stats['mean_ms']:.1f}ms "
          f"decode={stats['decode_tok_s']:.2f} tok/s")


if __name__ == "__main__":
    main()
