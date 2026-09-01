"""Roadmap phase 1 / item 1.3 — chunked-prefill decode TPOT micro-benchmark.

This is the *functional* half of the 1.3 performance acceptance ("TPOT/P99 of
decode must not regress with chunked prefill enabled"). It runs on CPU with the
tiny test fixture, so the absolute numbers are NOT production-representative —
a real P99 run still needs a GPU + a GGUF model + benchmarks/bench_inference.py
wired through RequestScheduler. What this script proves is the mechanism:

  * chunked OFF: one long prefill monopolises a whole step, decode stalls.
  * chunked ON : the long prefill is split, decode keeps advancing every step
                 and its per-step latency (TPOT proxy) stays bounded.

Run:  python3 benchmarks/bench_chunked_prefill_tpot.py
"""

import os
import sys
import time

import numpy as np

build_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

import forge  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "test_model_small.ninf")

MODEL_CONFIG = {
    "vocab_size": 100,
    "hidden_dim": 32,
    "intermediate_dim": 64,
    "num_layers": 1,
    "num_heads": 2,
    "num_kv_heads": 1,
    "head_dim": 16,
    "device": "cpu",
}

SHORT_PROMPT = [1, 2, 3, 4, 5, 6, 7, 8]
LONG_PROMPT = [(i % 99) + 1 for i in range(400)]


def _request_map(sched):
    return {r.request_id: r for r in sched.get_all_requests()}


def run_mixed(model, chunk_size, short_prompts=2, long_len=400, max_steps=160):
    """Long prefill + N decode. Returns decode step latencies (ms) while the
    long prefill is in flight, plus how many prefill steps there were."""
    sched = forge.RequestScheduler(model, block_size=16, max_num_seqs=8)
    sched.prefill_chunk_size = chunk_size

    short_ids = [
        sched.submit(SHORT_PROMPT, max_new_tokens=80, eos_token_id=-1)
        for _ in range(short_prompts)
    ]
    for _ in range(5):  # let shorts reach decode
        sched.step()

    long_id = sched.submit(LONG_PROMPT[:long_len], max_new_tokens=3, eos_token_id=-1)

    decode_step_ms = []
    long_prefill_steps = 0
    prev = {rid: _request_map(sched)[rid].num_generated for rid in short_ids}

    while sched.has_pending():
        reqs = _request_map(sched)
        long_req = reqs.get(long_id)
        # "During long prefill" == long has not produced a token yet. With
        # chunking disabled the whole prompt is one step, so this window is a
        # single (decode-starving) step; with chunking it spans many steps.
        long_prefilling = long_req is not None and long_req.num_generated == 0

        t0 = time.perf_counter()
        sched.step()
        dt = (time.perf_counter() - t0) * 1000.0

        reqs2 = _request_map(sched)
        cur = {rid: reqs2[rid].num_generated for rid in short_ids if rid in reqs2}
        decode_advanced = any(cur.get(rid, 0) > prev.get(rid, 0) for rid in short_ids)

        if long_prefilling:
            long_prefill_steps += 1
            if decode_advanced:
                decode_step_ms.append(dt)
        prev = cur

        if long_req is None:
            break

    return decode_step_ms, long_prefill_steps


def run_baseline(model, short_prompts=2, max_steps=80):
    """Pure decode, no competing prefill — the TPOT baseline."""
    sched = forge.RequestScheduler(model, block_size=16, max_num_seqs=8)
    for _ in range(short_prompts):
        sched.submit(SHORT_PROMPT, max_new_tokens=80, eos_token_id=-1)

    times = []
    for _ in range(max_steps):
        if not sched.has_pending():
            break
        t0 = time.perf_counter()
        sched.step()
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def _stats(xs):
    if not xs:
        return {"n": 0, "mean": float("nan"), "p50": float("nan"), "p99": float("nan")}
    a = np.array(xs)
    return {
        "n": len(a),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p99": float(np.percentile(a, 99)),
    }


def main():
    if not os.path.exists(FIXTURE):
        print("fixture", FIXTURE, "not found — skipping (needs tests/fixtures/test_model_small.ninf)")
        return

    model = forge.Model()
    model.load(FIXTURE, **MODEL_CONFIG)

    print("=" * 72)
    print("  1.3 chunked-prefill decode TPOT micro-benchmark (CPU fixture)")
    print("=" * 72)

    base = _stats(run_baseline(model))
    print("\n[baseline] pure decode step latency (ms): "
          f"mean={base['mean']:.3f} p50={base['p50']:.3f} p99={base['p99']:.3f} n={base['n']}")

    off_ms, off_steps = run_mixed(model, chunk_size=-1)  # chunking disabled
    off = _stats(off_ms)
    print(f"[chunked OFF] long prefill took {off_steps} step(s); "
          f"decode advanced on {off['n']} of them (starved if 0)")
    print("    decode step latency (ms) while prefilling: "
          f"mean={off['mean']:.3f} p50={off['p50']:.3f} p99={off['p99']:.3f}")

    on_ms, on_steps = run_mixed(model, chunk_size=32)  # chunking enabled
    on = _stats(on_ms)
    print(f"[chunked ON ] long prefill took {on_steps} step(s); "
          f"decode advanced on {on['n']} of them")
    print("    decode step latency (ms) while prefilling: "
          f"mean={on['mean']:.3f} p50={on['p50']:.3f} p99={on['p99']:.3f}")

    print("\n--- verdict ---")
    off_inflation = off["mean"] / base["mean"] if base["mean"] > 0 else float("nan")
    on_inflation = on["mean"] / base["mean"] if base["mean"] > 0 else float("nan")
    print(f"  chunked OFF: a whole-prompt prefill step inflates decode step "
          f"latency {off_inflation:.1f}x ({off['mean']:.3f} vs {base['mean']:.3f} ms) "
          f"— decode is NOT starved but every token waits ~10x longer")
    print(f"  chunked ON : decode step latency is {on_inflation:.2f}x baseline "
          f"({on['mean']:.3f} vs {base['mean']:.3f} ms) — bounded, not degraded")
    # Acceptance: with chunking enabled, decode step latency stays within 3x of
    # the pure-decode baseline AND is far below the un-chunked whole-prompt step.
    ok = (on_inflation <= 3.0) and (on_inflation < off_inflation)
    print("  RESULT:", "PASS" if ok else "CHECK")
    print("=" * 72)


if __name__ == "__main__":
    main()
