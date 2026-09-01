"""Roadmap phase 1 / item 1.3: chunked prefill must not starve decode.

Acceptance criterion from the roadmap:

    Build a mixed load of one long prefill plus N decode requests and assert
    that decode keeps producing tokens while the long prefill is still in
    progress (i.e. it is not starved).

Before 1.3 a request's whole prompt was pushed through the model in a single
step, so one long prompt produced one very slow step during which no other
request advanced. With chunked prefill the prompt is spread over several steps
and decode requests ride along in the same batch.
"""

import os
import sys

import pytest

build_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "build")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

import forge  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "test_model_small.ninf")

# Matches tests/conftest.py's model_config so the tiny .ninf fixture loads.
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

# Keep every token id inside the fixture's 100-token vocabulary.
LONG_PROMPT = [(i % 99) + 1 for i in range(400)]
SHORT_PROMPT = [1, 2, 3, 4, 5, 6, 7, 8]


@pytest.fixture(scope="module")
def model():
    if not os.path.exists(FIXTURE):
        pytest.skip("fixture test_model_small.ninf not found")
    m = forge.Model()
    m.load(FIXTURE, **MODEL_CONFIG)
    return m


def _request_map(sched):
    return {r.request_id: r for r in sched.get_all_requests()}


def _run_mixed_load(model, chunk_size, short_prompts=2, max_steps=80):
    """Submit short requests, let them reach decode, then submit a long prompt.

    Returns (steps_where_long_prefilled, steps_where_decode_advanced,
             scheduler) so callers can assert on the interleaving.
    """
    sched = forge.RequestScheduler(model, block_size=16, max_num_seqs=8)
    sched.prefill_chunk_size = chunk_size

    short_ids = [
        sched.submit(SHORT_PROMPT, max_new_tokens=40, eos_token_id=-1)
        for _ in range(short_prompts)
    ]

    # Let the short prompts finish prefill and enter decode.
    for _ in range(3):
        sched.step()

    long_id = sched.submit(LONG_PROMPT, max_new_tokens=2, eos_token_id=-1)

    long_prefill_steps = 0
    decode_advanced_steps = 0
    prev_counts = {rid: 0 for rid in short_ids}

    for _ in range(max_steps):
        if not sched.has_pending():
            break
        sched.step()

        reqs = _request_map(sched)
        long_req = reqs.get(long_id)
        if long_req is None or long_req.status != forge.RequestStatus.Prefilling:
            # Long prefill is over; nothing left to interleave with.
            break

        long_prefill_steps += 1

        cur = {rid: reqs[rid].num_generated for rid in short_ids if rid in reqs}
        if any(cur.get(rid, 0) > prev_counts[rid] for rid in short_ids):
            decode_advanced_steps += 1
        prev_counts.update(cur)

    return long_prefill_steps, decode_advanced_steps, sched


def test_long_prompt_is_split_into_multiple_steps(model):
    """A prompt longer than the chunk size must span more than one step."""
    long_prefill_steps, _, _ = _run_mixed_load(model, chunk_size=32)

    # 400 tokens at 32 tokens/step needs >1 step to prefill.
    assert long_prefill_steps > 1, (
        f"long prompt should be chunked across steps, but prefilled in "
        f"{long_prefill_steps} step(s)"
    )


def test_decode_keeps_advancing_during_long_prefill(model):
    """Decode requests must produce tokens while a long prompt prefills."""
    long_prefill_steps, decode_advanced_steps, sched = _run_mixed_load(model, chunk_size=32)

    assert long_prefill_steps > 1, "precondition: long prompt must be chunked"
    assert decode_advanced_steps > 0, (
        "decode requests produced no tokens while the long prompt was still "
        f"prefilling ({long_prefill_steps} steps) — decode is starved"
    )
    # Not just once: it should advance on essentially every chunked step.
    assert decode_advanced_steps >= long_prefill_steps - 1, (
        f"decode advanced on only {decode_advanced_steps} of {long_prefill_steps} "
        "chunked prefill steps"
    )
    assert sched.interleaved_steps > 0, "scheduler recorded no interleaved steps"


def _count_prefill_chunks(model, chunk_size, prompt_len=400):
    """How many prefill chunks a single long prompt consumed.

    With chunking disabled the whole prompt is one chunk; with chunking it is
    ceil(prompt_len / chunk_size) chunks. Directly measures the mechanism.
    """
    sched = forge.RequestScheduler(model, block_size=16, max_num_seqs=8)
    sched.prefill_chunk_size = chunk_size
    sched.submit(LONG_PROMPT[:prompt_len], max_new_tokens=2, eos_token_id=-1)
    for _ in range(100):
        if not sched.has_pending():
            break
        sched.step()
    return sched.prefill_chunks_issued


def test_chunking_concentrates_work_when_disabled(model):
    """Control case: with chunking disabled the long prefill is one step.

    This is what makes the chunked assertions meaningful — they are not simply
    observing that everything shares a batch anyway.
    """
    unchunked_chunks = _count_prefill_chunks(model, chunk_size=-1)
    chunked_chunks = _count_prefill_chunks(model, chunk_size=32)

    assert unchunked_chunks == 1, (
        f"with chunking disabled the prompt should be one prefill chunk, got "
        f"{unchunked_chunks}"
    )
    assert chunked_chunks > unchunked_chunks, (
        f"chunking should split the prompt into more chunks "
        f"(chunked={chunked_chunks}, unchunked={unchunked_chunks})"
    )
    assert chunked_chunks == 13, (  # ceil(400 / 32)
        f"400 tokens at 32/chunk should be 13 chunks, got {chunked_chunks}"
    )


def test_metrics_are_exposed(model):
    """The interleaving metrics must be readable from Python."""
    sched = forge.RequestScheduler(model, block_size=16, max_num_seqs=8)
    sched.prefill_chunk_size = 32

    assert sched.prefill_chunk_size == 32
    assert sched.interleaved_steps == 0
    assert sched.max_step_latency_ms == 0.0
    assert sched.prefill_chunks_issued == 0

    sched.submit(LONG_PROMPT, max_new_tokens=2, eos_token_id=-1)
    for _ in range(20):
        if not sched.has_pending():
            break
        sched.step()

    assert sched.prefill_chunks_issued > 1, "a 400-token prompt at 32/step is >1 chunk"
    assert sched.max_step_latency_ms > 0.0, "step latency should be measured"

    # Ratio is a fraction of the last step's tokens that came from decode.
    assert 0.0 <= sched.last_step_decode_ratio <= 1.0
