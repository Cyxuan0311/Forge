"""Multi-level KV cache tests (Phases 1-3).

Exercises the sglang-style tiered prefix cache via the Python scheduler API:

  Phase 1  — two-tier (device + host blob pool) residency, write policies.
  Phase 2  — asynchronous device->host backup, restore prefetch / bounded wait.
  Phase 3  — L3 file backend: host blobs spilled to disk and fetched back.

All tests run with FORGE_KV_STORAGE_MODE=paged. The test model uses
page_size=16, so prompts must be >= 16 tokens for prefix caching to engage.
The hard correctness assertion everywhere is that greedy output is identical
to a from-scratch run, which proves the restored/refetched KV is bit-exact.
"""

import pytest

import forge


@pytest.fixture(autouse=True)
def _paged_kv_storage(monkeypatch):
    """Pin FORGE_KV_STORAGE_MODE=paged for this module (restored after each test)."""
    monkeypatch.setenv("FORGE_KV_STORAGE_MODE", "paged")


def _run_scheduler(scheduler, max_steps=400):
    for _ in range(max_steps):
        if not scheduler.has_pending():
            break
        scheduler.step()
    return scheduler.get_finished()


def _make_scheduler(model_path, model_config):
    model = forge.Model()
    model.load(model_path, arch_type="llama", **model_config)
    return forge.RequestScheduler(model, block_size=4, max_num_seqs=4)


# A 32-token prompt (two complete pages @ page_size=16).
PRIME = list(range(1, 33))


def _unique_prompt(i):
    """A constant-token 32-token prompt distinct from PRIME and from each other."""
    tok = (i * 7 + 3) % 97 + 1  # 1..97, avoids token 0 and vocab ceiling (100)
    return [tok] * 32


def _prime_then_evict(scheduler, greedy_cfg, n_unique):
    """Register PRIME, then register `n_unique` other prompts so PRIME (the oldest
    device node) is pushed out of the device tier into the host pool / L3.

    Returns PRIME's reference output tokens (from the first registration run).
    """
    scheduler.submit(PRIME, max_new_tokens=2, sampler_config=greedy_cfg)
    finished = _run_scheduler(scheduler)
    ref = finished[0].output_tokens

    for i in range(n_unique):
        scheduler.submit(_unique_prompt(i), max_new_tokens=1, sampler_config=greedy_cfg)
        _run_scheduler(scheduler)

    return ref


class TestWriteBackHostHit:
    """Phase 1/2 — write-back: evicted prefixes survive on the host pool and are
    restored on a later hit (avoiding re-prefill), with bit-exact output."""

    def test_sync_backup_host_hit(self, model_path, model_config, monkeypatch):
        monkeypatch.setenv("FORGE_KV_WRITE_POLICY", "write_back")
        monkeypatch.setenv("FORGE_KV_BACKUP_ASYNC", "0")  # deterministic sync backup
        s = _make_scheduler(model_path, model_config)
        cfg = forge.SamplerConfig(do_sample=False)

        ref = _prime_then_evict(s, cfg, n_unique=72)

        # PRIME was the oldest device node -> evicted to the host pool.
        before = s.prefix_cache_host_hits
        s.submit(PRIME, max_new_tokens=2, sampler_config=cfg)
        finished = _run_scheduler(s)
        assert len(finished) == 1

        assert s.prefix_cache_host_hits > before  # served from the host pool
        assert finished[0].output_tokens == ref    # restored KV is correct


class TestAsyncBackup:
    """Phase 2 — backup runs on a worker thread; no pending jobs may leak and the
    restored output must still be correct."""

    def test_async_backup_no_leak_correct(self, model_path, model_config, monkeypatch):
        monkeypatch.setenv("FORGE_KV_WRITE_POLICY", "write_back")
        monkeypatch.setenv("FORGE_KV_BACKUP_ASYNC", "1")  # worker-thread backup
        s = _make_scheduler(model_path, model_config)
        cfg = forge.SamplerConfig(do_sample=False)

        ref = _prime_then_evict(s, cfg, n_unique=72)

        # Drain the worker so PRIME's async backup is committed (host-resident)
        # before we re-submit — otherwise it is still in flight and would be
        # served as a device hit. This also proves no job leaks.
        s.prefix_cache_flush()
        assert s.prefix_cache_pending_jobs == 0

        before = s.prefix_cache_host_hits
        s.submit(PRIME, max_new_tokens=2, sampler_config=cfg)
        finished = _run_scheduler(s)
        assert len(finished) == 1

        # Drain the worker one more time: the re-submit run may have queued a
        # device->host backup on its final eviction. A single in-flight job is
        # benign; it must not be *stranded*.
        s.prefix_cache_flush()
        assert s.prefix_cache_pending_jobs == 0
        assert s.prefix_cache_host_hits > before
        assert finished[0].output_tokens == ref


class TestWriteThrough:
    """Phase 1 — write-through mirrors every registered prefix eagerly; eviction
    then only releases device pages. Restored output must be correct."""

    def test_write_through_host_hit(self, model_path, model_config, monkeypatch):
        monkeypatch.setenv("FORGE_KV_WRITE_POLICY", "write_through")
        monkeypatch.setenv("FORGE_KV_BACKUP_ASYNC", "0")
        s = _make_scheduler(model_path, model_config)
        cfg = forge.SamplerConfig(do_sample=False)

        ref = _prime_then_evict(s, cfg, n_unique=72)

        before = s.prefix_cache_host_hits
        s.submit(PRIME, max_new_tokens=2, sampler_config=cfg)
        finished = _run_scheduler(s)
        assert len(finished) == 1

        assert s.prefix_cache_host_hits > before
        assert finished[0].output_tokens == ref


class TestWritePolicyOff:
    """Phase 1 — off = legacy single-tier behavior: eviction destroys the node,
    so a re-submit is a full miss + recompute, but output is still correct."""

    def test_off_no_host_hits_recompute(self, model_path, model_config, monkeypatch):
        monkeypatch.setenv("FORGE_KV_WRITE_POLICY", "off")
        s = _make_scheduler(model_path, model_config)
        cfg = forge.SamplerConfig(do_sample=False)

        ref = _prime_then_evict(s, cfg, n_unique=72)

        before = s.prefix_cache_host_hits
        s.submit(PRIME, max_new_tokens=2, sampler_config=cfg)
        finished = _run_scheduler(s)
        assert len(finished) == 1

        # With policy off the prefix was destroyed, so no host hit occurred.
        assert s.prefix_cache_host_hits == before
        # Recompute must still yield identical greedy output.
        assert finished[0].output_tokens == ref


class TestRestorePrefetch:
    """Phase 2 — prefetch warms a host-resident prefix ahead of lookup. With
    FORGE_KV_PREFETCH=1 the re-submitted PRIME should still restore correctly
    (and at least the synchronous fallback keeps output bit-exact)."""

    def test_prefetch_host_hit(self, model_path, model_config, monkeypatch):
        monkeypatch.setenv("FORGE_KV_WRITE_POLICY", "write_back")
        monkeypatch.setenv("FORGE_KV_BACKUP_ASYNC", "0")
        monkeypatch.setenv("FORGE_KV_PREFETCH", "1")
        s = _make_scheduler(model_path, model_config)
        cfg = forge.SamplerConfig(do_sample=False)

        ref = _prime_then_evict(s, cfg, n_unique=72)

        before = s.prefix_cache_hits
        s.submit(PRIME, max_new_tokens=2, sampler_config=cfg)
        finished = _run_scheduler(s)
        assert len(finished) == 1

        # Prefetch restores the prefix ahead of the lookup, so the lookup hits the
        # device-resident (already-restored) node: a device hit, not a host hit.
        # The restored KV must still be bit-exact.
        assert s.prefix_cache_hits > before
        assert s.prefix_cache_pending_jobs == 0
        assert finished[0].output_tokens == ref


class TestL3FileBackend:
    """Phase 3 — when the host pool is over budget the oldest blobs are spilled to
    an L3 file backend; a later hit fetches them back and restores correctly."""

    def test_l3_spill_and_restore(self, model_path, model_config, monkeypatch, tmp_path):
        monkeypatch.setenv("FORGE_KV_WRITE_POLICY", "write_back")
        monkeypatch.setenv("FORGE_KV_BACKUP_ASYNC", "0")
        # Tiny host pool (far smaller than a single prefix blob) so every blob
        # spills to L3 instead of being retained in the host pool.
        monkeypatch.setenv("FORGE_KV_HOST_POOL_BYTES", "4096")
        monkeypatch.setenv("FORGE_KV_L3_BACKEND", "file")
        monkeypatch.setenv("FORGE_KV_L3_DIR", str(tmp_path / "l3"))
        monkeypatch.setenv("FORGE_KV_L3_BYTES", "1073741824")  # 1 GiB: holds everything

        s = _make_scheduler(model_path, model_config)
        cfg = forge.SamplerConfig(do_sample=False)

        ref = _prime_then_evict(s, cfg, n_unique=72)

        before = s.prefix_cache_host_hits
        s.submit(PRIME, max_new_tokens=2, sampler_config=cfg)
        finished = _run_scheduler(s)
        assert len(finished) == 1

        # PRIME's blob was spilled to L3 (host pool too small to retain it) and
        # must have been fetched back and restored — output is bit-exact.
        assert s.prefix_cache_host_hits > before
        assert finished[0].output_tokens == ref

        # L3 files were actually written to disk.
        l3_dir = tmp_path / "l3"
        assert l3_dir.exists()
        assert any(l3_dir.iterdir())
