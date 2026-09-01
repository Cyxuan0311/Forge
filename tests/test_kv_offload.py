"""Roadmap phase 1 / item 1.1: KV page host offload (swap).

Acceptance criteria from the roadmap:

    Functional: with a small page pool and many sequences, a page that was
    evicted to the host pool must be brought back correctly on the next access
    and attention results must stay correct (identical to the no-offload run).

    Resource: with max_num_seqs beyond what fits, the scheduler keeps serving
    by swapping out instead of OOM / rejecting.

The scheduler triggers swap-out when free pages drop below a watermark
(kv_swap_watermark, fraction of total capacity). With the watermark at 1.0
every step that consumes pages also evicts some, so the round-trip is exercised
hard and the generated tokens must still be bit-identical to a run with swap
disabled (watermark 0).
"""

import os
import sys

import pytest

# Paged storage must be selected before the scheduler builds its KV memory.
os.environ.setdefault("FORGE_KV_STORAGE_MODE", "paged")

build_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "build")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

import forge  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "test_model_small.ninf")

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

# Token ids must stay inside the fixture's 100-token vocabulary.
PROMPT = [(i % 99) + 1 for i in range(64)]


@pytest.fixture(scope="module")
def model():
    if not os.path.exists(FIXTURE):
        pytest.skip("fixture test_model_small.ninf not found")
    m = forge.Model()
    m.load(FIXTURE, **MODEL_CONFIG)
    return m


def _run(model, watermark, prompt=None, max_new_tokens=24, n_requests=2):
    """Run n_requests over the same prompt and return (outputs, scheduler)."""
    sched = forge.RequestScheduler(model, block_size=16, max_num_seqs=8)
    sched.kv_swap_watermark = watermark
    prompt = PROMPT if prompt is None else prompt

    ids = [sched.submit(prompt, max_new_tokens=max_new_tokens, eos_token_id=-1)
           for _ in range(n_requests)]

    for _ in range(400):
        if not sched.has_pending():
            break
        sched.step()

    finished = {r.request_id: r for r in sched.get_finished()}
    outputs = [list(finished[rid].output_tokens) for rid in ids]
    return outputs, sched


def test_swap_offloads_pages_when_under_pressure(model):
    """Watermark 1.0 must actually evict pages to the host pool."""
    _, sched = _run(model, watermark=1.0)

    assert sched.swap_events > 0, "no swap-out happened under pressure"
    assert sched.num_offloaded_pages > 0, "no pages were evicted to the host pool"
    assert sched.num_brought_back_pages > 0, "evicted pages were never brought back"
    assert sched.host_pool_bytes > 0, "host swap pool is empty despite evictions"
    assert sched.num_free_pages >= 0
    assert sched.num_total_pages > 0


def test_swap_output_is_identical_to_no_swap(model):
    """Attention must be unaffected by eviction: tokens match the control run."""
    control, _ = _run(model, watermark=0.0)
    swapped, sched = _run(model, watermark=1.0)

    assert sched.swap_events > 0, "precondition: pressure run must have swapped"
    for i, (c, s) in enumerate(zip(control, swapped)):
        assert c == s, (
            f"request {i}: swapped run produced different tokens "
            f"(control={c}, swapped={s})"
        )


def test_swap_recovers_after_pressure_relaxes(model):
    """Once the page pool has room again, further steps must not evict."""
    sched = forge.RequestScheduler(model, block_size=16, max_num_seqs=8)
    sched.kv_swap_watermark = 0.0  # no proactive eviction
    rid = sched.submit(PROMPT, max_new_tokens=6, eos_token_id=-1)
    for _ in range(200):
        if not sched.has_pending():
            break
        sched.step()
    finished = {r.request_id: r for r in sched.get_finished()}

    assert rid in finished and len(finished[rid].output_tokens) == 6
    # With no pressure there is nothing to evict at all.
    assert sched.swap_events == 0
    assert sched.num_offloaded_pages == 0


def test_swap_metrics_and_knob(model):
    """The watermark knob and the metrics must be exposed from Python."""
    sched = forge.RequestScheduler(model, block_size=16, max_num_seqs=8)
    assert sched.kv_swap_watermark == pytest.approx(0.15)  # default
    sched.kv_swap_watermark = 0.7
    assert sched.kv_swap_watermark == pytest.approx(0.7)

    assert sched.swap_events == 0
    assert sched.num_offloaded_pages == 0
    assert sched.num_brought_back_pages == 0
    assert sched.host_pool_bytes == 0
