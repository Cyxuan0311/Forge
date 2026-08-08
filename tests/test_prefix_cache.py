"""Phase 5: Page-level prefix cache tests.

Tests run with FORGE_KV_STORAGE_MODE=paged to exercise the PrefixCache path.
The test model uses page_size=16, so prompts must be >= 16 tokens for prefix
caching to engage (MIN_CACHE_PROMPT_LEN=16).

Acceptance criteria (from KVCache优化与渐进式集成方案.md Phase 5):
  - 0%, 50%, 100% prefix hit rate all correct.
  - prefix eviction → re-computation gives consistent results.
  - prefix page shared by multiple requests → no premature reclaim.
"""

import os

import pytest

# Ensure paged mode for this test module.
# The scheduler reads FORGE_KV_STORAGE_MODE in its constructor.
os.environ["FORGE_KV_STORAGE_MODE"] = "paged"

import forge


def _run_scheduler(scheduler, max_steps=200):
    """Run scheduler until no pending requests, return finished list."""
    for _ in range(max_steps):
        if not scheduler.has_pending():
            break
        scheduler.step()
    return scheduler.get_finished()


@pytest.fixture
def scheduler(model_path, model_config):
    model = forge.Model()
    model.load(model_path, arch_type="llama", **model_config)
    s = forge.RequestScheduler(model, block_size=4, max_num_seqs=4)
    return s


@pytest.fixture
def greedy_cfg():
    return forge.SamplerConfig(do_sample=False)


# A 32-token prompt (2 complete pages with page_size=16).
PROMPT_A = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

# A different 32-token prompt (no shared prefix with PROMPT_A).
PROMPT_B = [99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84,
            83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68]

# A 48-token prompt sharing the first 32 tokens with PROMPT_A (3 complete pages).
PROMPT_A_EXT = PROMPT_A + [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]


class TestPrefixCacheHitRates:
    """Acceptance: 0%, 50%, 100% prefix hit rate all correct."""

    def test_100_percent_hit_rate(self, scheduler, greedy_cfg):
        """Same prompt submitted twice → second request hits cache."""
        # First request: miss (registers prefix)
        scheduler.submit(PROMPT_A, max_new_tokens=2, sampler_config=greedy_cfg)
        finished = _run_scheduler(scheduler)
        assert len(finished) == 1
        first_output = finished[0].output_tokens

        # Second request: should hit
        scheduler.submit(PROMPT_A, max_new_tokens=2, sampler_config=greedy_cfg)
        finished = _run_scheduler(scheduler)
        assert len(finished) == 1
        second_output = finished[0].output_tokens

        # Verify hit was recorded
        assert scheduler.prefix_cache_hits >= 1
        assert scheduler.prefix_cache_misses >= 1

        # Output should be identical (greedy, same KV state for prefix)
        assert first_output == second_output

    def test_0_percent_hit_rate(self, scheduler, greedy_cfg):
        """Two completely different prompts → no hits."""
        scheduler.submit(PROMPT_A, max_new_tokens=2, sampler_config=greedy_cfg)
        finished = _run_scheduler(scheduler)
        assert len(finished) == 1

        scheduler.submit(PROMPT_B, max_new_tokens=2, sampler_config=greedy_cfg)
        finished = _run_scheduler(scheduler)
        assert len(finished) == 1

        # No hits (both are unique prompts)
        assert scheduler.prefix_cache_hits == 0
        assert scheduler.prefix_cache_misses >= 2

    def test_50_percent_hit_rate(self, scheduler, greedy_cfg):
        """Submit A, then A (hit) and B (miss) → 50% hit rate."""
        # Prime cache with A
        scheduler.submit(PROMPT_A, max_new_tokens=2, sampler_config=greedy_cfg)
        _run_scheduler(scheduler)

        # Submit A (should hit) and B (should miss)
        scheduler.submit(PROMPT_A, max_new_tokens=2, sampler_config=greedy_cfg)
        scheduler.submit(PROMPT_B, max_new_tokens=2, sampler_config=greedy_cfg)
        finished = _run_scheduler(scheduler)
        assert len(finished) == 2

        # At least 1 hit (from the second A) and at least 2 misses (first A + B)
        assert scheduler.prefix_cache_hits >= 1

    def test_partial_prefix_hit(self, scheduler, greedy_cfg):
        """Prompt sharing first 32 tokens with a cached 48-token prompt.

        Submit PROMPT_A_EXT (48 tokens, 3 pages), then submit PROMPT_A
        (32 tokens, 2 pages). The 32-token prefix should hit.
        """
        scheduler.submit(PROMPT_A_EXT, max_new_tokens=2, sampler_config=greedy_cfg)
        _run_scheduler(scheduler)

        # PROMPT_A's first 32 tokens match PROMPT_A_EXT's first 32 tokens
        scheduler.submit(PROMPT_A, max_new_tokens=2, sampler_config=greedy_cfg)
        finished = _run_scheduler(scheduler)
        assert len(finished) == 1

        # Should have at least 1 hit (the 2-page prefix of PROMPT_A matches)
        assert scheduler.prefix_cache_hits >= 1


class TestPrefixEviction:
    """Acceptance: prefix eviction → re-computation gives consistent results."""

    def test_eviction_then_recompute(self, scheduler, greedy_cfg):
        """Fill cache, evict, then re-submit same prompt → results still correct."""
        # Submit A to register its prefix
        scheduler.submit(PROMPT_A, max_new_tokens=2, sampler_config=greedy_cfg)
        finished = _run_scheduler(scheduler)
        original_output = finished[0].output_tokens

        # Fill cache with many unique prompts to trigger eviction
        # (DEFAULT_MAX_ENTRIES=64; each prompt creates up to 2 entries for 32 tokens)
        for i in range(80):
            prompt = [i % 100] * 32  # unique-ish prompts
            scheduler.submit(prompt, max_new_tokens=1, sampler_config=greedy_cfg)
            _run_scheduler(scheduler)

        # Re-submit A — prefix was likely evicted, so it's a miss and recomputed
        scheduler.submit(PROMPT_A, max_new_tokens=2, sampler_config=greedy_cfg)
        finished = _run_scheduler(scheduler)
        recomputed_output = finished[0].output_tokens

        # Results should be identical (greedy decoding is deterministic)
        assert original_output == recomputed_output


class TestSharedPrefixNoPrematureReclaim:
    """Acceptance: prefix page shared by multiple requests → no premature reclaim."""

    def test_shared_prefix_concurrent(self, scheduler, greedy_cfg):
        """Two requests share a prefix; finishing one shouldn't break the other.

        1. Submit A, finish it (registers prefix in cache).
        2. Submit two copies of A concurrently (both hit cache).
        3. Finish one (releases its prefix reference).
        4. The other should still complete correctly.
        """
        # Prime cache
        scheduler.submit(PROMPT_A, max_new_tokens=2, sampler_config=greedy_cfg)
        _run_scheduler(scheduler)

        # Submit two identical prompts concurrently
        scheduler.submit(PROMPT_A, max_new_tokens=3, sampler_config=greedy_cfg)
        scheduler.submit(PROMPT_A, max_new_tokens=3, sampler_config=greedy_cfg)

        # Run to completion — both should finish
        finished = _run_scheduler(scheduler, max_steps=300)
        assert len(finished) == 2

        # Both should have produced output
        for req in finished:
            assert req.num_generated >= 1
            assert req.status == forge.RequestStatus.Finished

        # Both should have hit the cache
        assert scheduler.prefix_cache_hits >= 2

    def test_shared_prefix_sequential(self, scheduler, greedy_cfg):
        """Sequential sharing: A registers, B hits, B finishes, C hits.

        C should still get correct results after B released its prefix reference.
        """
        # Prime cache
        scheduler.submit(PROMPT_A, max_new_tokens=2, sampler_config=greedy_cfg)
        finished = _run_scheduler(scheduler)
        ref_output = finished[0].output_tokens

        # B hits and finishes
        scheduler.submit(PROMPT_A, max_new_tokens=2, sampler_config=greedy_cfg)
        finished_b = _run_scheduler(scheduler)
        assert len(finished_b) == 1
        assert finished_b[0].output_tokens == ref_output

        # C hits after B released its prefix reference
        scheduler.submit(PROMPT_A, max_new_tokens=2, sampler_config=greedy_cfg)
        finished_c = _run_scheduler(scheduler)
        assert len(finished_c) == 1
        assert finished_c[0].output_tokens == ref_output

        # At least 2 hits (B and C)
        assert scheduler.prefix_cache_hits >= 2


class TestPrefixCacheContiguousMode:
    """Verify contiguous mode still works (old prompt_cache_ path)."""

    def test_contiguous_prefix_cache_still_works(self, model_path, model_config):
        """Contiguous mode should use the old prompt_cache_ path."""
        # Temporarily unset env var to get contiguous mode
        old = os.environ.pop("FORGE_KV_STORAGE_MODE", None)
        try:
            model = forge.Model()
            model.load(model_path, arch_type="llama", **model_config)
            scheduler = forge.RequestScheduler(model, block_size=4, max_num_seqs=4)
            cfg = forge.SamplerConfig(do_sample=False)

            scheduler.submit(PROMPT_A, max_new_tokens=2, sampler_config=cfg)
            _run_scheduler(scheduler)

            scheduler.submit(PROMPT_A, max_new_tokens=2, sampler_config=cfg)
            finished = _run_scheduler(scheduler)
            assert len(finished) == 1

            # Contiguous mode should also get a hit
            assert scheduler.prefix_cache_hits >= 1
        finally:
            if old is not None:
                os.environ["FORGE_KV_STORAGE_MODE"] = old
