"""
FP8 KV-cache decode accuracy verification (Phase 0 acceptance).

Goal: confirm the fused FP8 decode kernel produces logits that match the
FP32 KV-cache baseline, so the cosine similarity of per-token logits
per decode step meets the acceptance bar (>=0.998 E4M3 / >=0.997 E5M2).

Because the test GPU is small (6GB), only one CUDA context can live on the
device at a time, so we split into two passes:

  # 1) FP32 baseline: greedy-generate a token sequence, store the per-step
  #    predicted logits + the generated token ids.
  python3 benchmarks/verify_fp8_kv_accuracy.py --mode fp32 \
      --model /mnt/g/AI/MiMo-7B-RL-Q4_K_M/MiMo-7B-RL-Q4_K_M.gguf \
      --out /tmp/fp8_kv_baseline.npz

  # 2) FP8 replay: feed the SAME token ids through an FP8 KV-cache context,
  #    capture per-step logits, compare against the baseline.
  python3 benchmarks/verify_fp8_kv_accuracy.py --mode fp8_e4m3 --out /tmp/fp8_kv_baseline.npz
  python3 benchmarks/verify_fp8_kv_accuracy.py --mode fp8_e5m2 --out /tmp/fp8_kv_baseline.npz

The FP8 context reuses the exact token history as the FP32 baseline, so any
divergence in logits isolates the KV-cache quantization (not generation drift).
"""

import os
import sys
import argparse
import numpy as np

BUILD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build")
if os.path.exists(BUILD_DIR):
    sys.path.insert(0, BUILD_DIR)

import forge  # noqa: E402

DEFAULT_MODEL = "/mnt/g/AI/MiMo-7B-RL-Q4_K_M/MiMo-7B-RL-Q4_K_M.gguf"
PROMPT = "The history of artificial intelligence begins with the idea that"


def _squeeze_logits(arr):
    """Flatten forward() output to a 1D (vocab,) logits vector."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        # (seq_len, vocab) -> use the last position
        arr = arr[-1]
    return arr.reshape(-1)


def _encode_prompt(model_path, prompt, use_tokenizer):
    if use_tokenizer:
        try:
            tok = forge.Tokenizer()
            tok.load_from_gguf(model_path)
            ids = tok.encode(prompt, add_bos=True)
            if len(ids) > 0:
                return np.asarray(ids, dtype=np.int32)
        except Exception as e:
            print(f"  [warn] tokenizer failed ({e}); falling back to synthetic prompt")
    # Synthetic prompt: deterministic ids within a typical vocab range.
    return (np.arange(32, dtype=np.int32) % 32000) + 1


def run_fp32(model_path, prompt, gen_len, use_tokenizer, out_path, device="cuda"):
    print(f"[fp32] loading model on {device}: {model_path}")
    model = forge.Model()
    model.load_auto(model_path, device=device)
    ctx = model.create_context(kv_cache_dtype="fp32",
                               gpu_layers=-1 if device == "cuda" else 0)

    stats = ctx.memory_stats()
    print(f"[fp32] kv_cache_dtype = {stats.get('kv_cache_dtype')}")

    prompt_ids = _encode_prompt(model_path, prompt, use_tokenizer)
    prompt_len = len(prompt_ids)
    print(f"[fp32] prompt_len = {prompt_len}, gen_len = {gen_len}")

    # Prefill
    prefill_logits = _squeeze_logits(ctx.forward(prompt_ids))
    tokens = [int(np.argmax(prefill_logits))]
    logits_seq = []

    start = prompt_len
    for i in range(gen_len - 1):
        lg = _squeeze_logits(ctx.forward(np.array([tokens[-1]], dtype=np.int32),
                                        start_pos=start))
        logits_seq.append(lg)
        nxt = int(np.argmax(lg))
        tokens.append(nxt)
        start += 1

    # save the FINAL predicted logits for the last generated token too
    lg_last = _squeeze_logits(ctx.forward(np.array([tokens[-1]], dtype=np.int32),
                                          start_pos=start))
    logits_seq.append(lg_last)

    logits_seq = np.stack(logits_seq, axis=0).astype(np.float32)  # (gen_len, vocab)
    tokens = np.asarray(tokens, dtype=np.int32)
    np.savez(out_path, logits=logits_seq, tokens=tokens, prompt_len=np.int32(prompt_len),
             prompt_ids=np.asarray(prompt_ids, dtype=np.int32))
    print(f"[fp32] saved baseline -> {out_path}")
    print(f"[fp32] first 16 tokens: {tokens[:16].tolist()}")


def run_fp8(model_path, mode, gen_len, use_tokenizer, out_path, device="cuda"):
    base = np.load(out_path)
    base_logits = base["logits"].astype(np.float32)  # (gen_len, vocab)
    base_tokens = base["tokens"].astype(np.int32)
    prompt_len = int(base["prompt_len"])
    gen_len = min(gen_len, base_logits.shape[0])

    kv_dtype = "fp8_e4m3" if mode == "fp8_e4m3" else "fp8_e5m2"
    print(f"[{mode}] loading model on {device}: {model_path}")
    model = forge.Model()
    model.load_auto(model_path, device=device)
    ctx = model.create_context(kv_cache_dtype=kv_dtype,
                               gpu_layers=-1 if device == "cuda" else 0)

    # Reuse the EXACT prompt ids from the baseline (no re-encode -> no drift).
    if "prompt_ids" in base.files:
        prompt_ids = base["prompt_ids"].astype(np.int32)
    else:
        prompt_ids = _encode_prompt(model_path, PROMPT, use_tokenizer)[:prompt_len]
    assert len(prompt_ids) == prompt_len, f"prompt_len {len(prompt_ids)} != {prompt_len}"

    # Prefill (this lazily initializes the KV cache with the requested dtype).
    ctx.forward(prompt_ids)

    # Now the KV cache is materialized: verify the dtype was actually applied.
    stats = ctx.memory_stats()
    print(f"[{mode}] kv_cache_dtype = {stats.get('kv_cache_dtype')} "
          f"(type_k={stats.get('kv_cache_type_k')}, type_v={stats.get('kv_cache_type_v')})")
    if stats.get("kv_cache_dtype") != kv_dtype:
        raise RuntimeError(
            f"FP8 KV cache was NOT applied! requested={kv_dtype} "
            f"actual={stats.get('kv_cache_dtype')}"
        )

    fp8_logits = []
    start = prompt_len
    for i in range(gen_len):
        lg = _squeeze_logits(ctx.forward(np.array([base_tokens[i]], dtype=np.int32),
                                         start_pos=start))
        fp8_logits.append(lg)
        start += 1
    fp8_logits = np.stack(fp8_logits, axis=0).astype(np.float32)

    # ---- Compare ----
    cosines = []
    top1_match = 0
    max_abs = 0.0
    for i in range(gen_len):
        a = base_logits[i]
        b = fp8_logits[i]
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        cos = float(np.dot(a, b) / (na * nb + 1e-12))
        cosines.append(cos)
        if int(np.argmax(b)) == int(np.argmax(a)):
            top1_match += 1
        max_abs = max(max_abs, float(np.max(np.abs(a - b))))

    cosines = np.asarray(cosines)
    mean_cos = float(cosines.mean())
    min_cos = float(cosines.min())
    top1 = top1_match / gen_len

    thr = 0.998 if mode == "fp8_e4m3" else 0.997
    print(f"\n===== FP8 decode vs FP32 baseline ({mode}) =====")
    print(f"  decode steps compared : {gen_len}")
    print(f"  mean cosine similarity : {mean_cos:.6f}")
    print(f"  min  cosine similarity : {min_cos:.6f}")
    print(f"  max |logits diff|      : {max_abs:.4f}")
    print(f"  top-1 argmax match     : {top1*100:.1f}%")
    print(f"  acceptance bar         : >= {thr}")
    ok = min_cos >= thr
    print(f"  RESULT                 : {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["fp32", "fp8_e4m3", "fp8_e5m2"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="/tmp/fp8_kv_baseline.npz")
    ap.add_argument("--gen-len", type=int, default=32)
    ap.add_argument("--no-tokenizer", action="store_true",
                    help="Use a synthetic prompt instead of real tokenizer ids")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    use_tok = not args.no_tokenizer
    if args.mode == "fp32":
        run_fp32(args.model, PROMPT, args.gen_len, use_tok, args.out, args.device)
    else:
        ok = run_fp8(args.model, args.mode, args.gen_len, use_tok, args.out, args.device)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
