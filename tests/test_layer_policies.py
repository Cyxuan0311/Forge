"""Phase 6: Per-layer memory policy tests.

Tests:
  - KVLayerPolicy enum is accessible from Python
  - SWA layers get smaller page pool in paged mode (core verification)
  - Paged mode with policies still produces correct output
  - Prefix cache still works with policies
  - Contiguous mode still works with policies
"""

import os

import pytest

# Ensure paged mode for this test module.
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
def model_path(model_path):
    return model_path


@pytest.fixture
def greedy_cfg():
    return forge.SamplerConfig(do_sample=False)


PROMPT_A = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]


class TestLayerPolicyEnum:
    """KVLayerPolicy enum is exposed to Python."""

    def test_enum_values(self):
        assert hasattr(forge, "KVLayerPolicy")
        assert forge.KVLayerPolicy.None_.value == 0
        assert forge.KVLayerPolicy.Full.value == 1
        assert forge.KVLayerPolicy.SlidingWindow.value == 2
        assert forge.KVLayerPolicy.Recurrent.value == 3


class TestSWAPagePoolIsolation:
    """Core Phase 6 verification: SWA layers get a smaller page pool.

    With n_swa=1024 and page_size=16:
      - SWA layer pool:  ceil(1024/16) + 1 = 65 pages
      - Full layer pool: min(4*4096/16, 256) = 256 pages

    This test verifies the per-layer pool sizing logic in PagedKVStorage.
    """

    def test_swa_layer_has_smaller_pool(self, model_path):
        """SWA layer gets fewer pages than a Full layer."""
        model = forge.Model()
        # 1-layer model, layer 0 is SWA with window=1024
        model.load(model_path, arch_type="llama",
                   vocab_size=100, hidden_dim=32, intermediate_dim=64,
                   num_layers=1, num_heads=2, num_kv_heads=1, head_dim=16,
                   device="cpu", n_swa=1024, swa_layers=[1])
        s = forge.RequestScheduler(model, block_size=4, max_num_seqs=4)

        # Trigger KV cache init by running one step
        cfg = forge.SamplerConfig(do_sample=False)
        s.submit(PROMPT_A, max_new_tokens=1, sampler_config=cfg)
        _run_scheduler(s)

        stats = s.memory_stats()
        assert "layer_pool_max_pages" in stats, "memory_stats must expose layer_pool_max_pages"

        pool_sizes = list(stats["layer_pool_max_pages"])
        assert len(pool_sizes) == 1, f"Expected 1 layer, got {len(pool_sizes)}"

        # SWA with window=1024, page_size=16: ceil(1024/16)+1 = 65 pages
        assert pool_sizes[0] == 65, \
            f"SWA layer pool should be 65 pages (window=1024/16+1), got {pool_sizes[0]}"

    def test_full_layer_has_larger_pool(self, model_path):
        """Without SWA, layer gets the full pool size (256 pages capped)."""
        model = forge.Model()
        model.load(model_path, arch_type="llama",
                   vocab_size=100, hidden_dim=32, intermediate_dim=64,
                   num_layers=1, num_heads=2, num_kv_heads=1, head_dim=16,
                   device="cpu")
        s = forge.RequestScheduler(model, block_size=4, max_num_seqs=4)

        cfg = forge.SamplerConfig(do_sample=False)
        s.submit(PROMPT_A, max_new_tokens=1, sampler_config=cfg)
        _run_scheduler(s)

        stats = s.memory_stats()
        pool_sizes = list(stats["layer_pool_max_pages"])
        assert len(pool_sizes) == 1

        # Full pool: min(4*4096/16, 256) = 256
        assert pool_sizes[0] == 256, \
            f"Full layer pool should be 256 pages (capped), got {pool_sizes[0]}"

    def test_swa_pool_is_smaller_than_full(self, model_path):
        """Direct comparison: SWA pool < Full pool."""
        # SWA model
        model_swa = forge.Model()
        model_swa.load(model_path, arch_type="llama",
                       vocab_size=100, hidden_dim=32, intermediate_dim=64,
                       num_layers=1, num_heads=2, num_kv_heads=1, head_dim=16,
                       device="cpu", n_swa=1024, swa_layers=[1])
        s_swa = forge.RequestScheduler(model_swa, block_size=4, max_num_seqs=4)
        s_swa.submit(PROMPT_A, max_new_tokens=1,
                     sampler_config=forge.SamplerConfig(do_sample=False))
        _run_scheduler(s_swa)
        swa_pool = list(s_swa.memory_stats()["layer_pool_max_pages"])[0]

        # Full model
        model_full = forge.Model()
        model_full.load(model_path, arch_type="llama",
                        vocab_size=100, hidden_dim=32, intermediate_dim=64,
                        num_layers=1, num_heads=2, num_kv_heads=1, head_dim=16,
                        device="cpu")
        s_full = forge.RequestScheduler(model_full, block_size=4, max_num_seqs=4)
        s_full.submit(PROMPT_A, max_new_tokens=1,
                      sampler_config=forge.SamplerConfig(do_sample=False))
        _run_scheduler(s_full)
        full_pool = list(s_full.memory_stats()["layer_pool_max_pages"])[0]

        assert swa_pool < full_pool, \
            f"SWA pool ({swa_pool}) should be smaller than Full pool ({full_pool})"

    def test_swa_nbytes_smaller_than_full(self, model_path):
        """SWA pool uses less memory than Full pool."""
        model_swa = forge.Model()
        model_swa.load(model_path, arch_type="llama",
                       vocab_size=100, hidden_dim=32, intermediate_dim=64,
                       num_layers=1, num_heads=2, num_kv_heads=1, head_dim=16,
                       device="cpu", n_swa=1024, swa_layers=[1])
        s_swa = forge.RequestScheduler(model_swa, block_size=4, max_num_seqs=4)
        s_swa.submit(PROMPT_A, max_new_tokens=1,
                     sampler_config=forge.SamplerConfig(do_sample=False))
        _run_scheduler(s_swa)
        swa_bytes = s_swa.memory_stats()["kv_cache_nbytes"]

        model_full = forge.Model()
        model_full.load(model_path, arch_type="llama",
                        vocab_size=100, hidden_dim=32, intermediate_dim=64,
                        num_layers=1, num_heads=2, num_kv_heads=1, head_dim=16,
                        device="cpu")
        s_full = forge.RequestScheduler(model_full, block_size=4, max_num_seqs=4)
        s_full.submit(PROMPT_A, max_new_tokens=1,
                      sampler_config=forge.SamplerConfig(do_sample=False))
        _run_scheduler(s_full)
        full_bytes = s_full.memory_stats()["kv_cache_nbytes"]

        assert swa_bytes < full_bytes, \
            f"SWA nbytes ({swa_bytes}) should be < Full nbytes ({full_bytes})"


class TestPagedModeWithPolicies:
    """Paged mode with layer policies still produces correct output."""

    def test_paged_inference_correct(self, model_path):
        """Basic inference in paged mode works with default policies."""
        model = forge.Model()
        model.load(model_path, arch_type="llama",
                   vocab_size=100, hidden_dim=32, intermediate_dim=64,
                   num_layers=1, num_heads=2, num_kv_heads=1, head_dim=16,
                   device="cpu")
        s = forge.RequestScheduler(model, block_size=4, max_num_seqs=4)
        s.submit(PROMPT_A, max_new_tokens=2,
                 sampler_config=forge.SamplerConfig(do_sample=False))
        finished = _run_scheduler(s)
        assert len(finished) == 1
        assert len(finished[0].output_tokens) >= 1

    def test_prefix_cache_with_policies(self, model_path):
        """Prefix cache still works with layer policies set."""
        model = forge.Model()
        model.load(model_path, arch_type="llama",
                   vocab_size=100, hidden_dim=32, intermediate_dim=64,
                   num_layers=1, num_heads=2, num_kv_heads=1, head_dim=16,
                   device="cpu")
        s = forge.RequestScheduler(model, block_size=4, max_num_seqs=4)
        s.submit(PROMPT_A, max_new_tokens=2,
                 sampler_config=forge.SamplerConfig(do_sample=False))
        _run_scheduler(s)
        s.submit(PROMPT_A, max_new_tokens=2,
                 sampler_config=forge.SamplerConfig(do_sample=False))
        finished = _run_scheduler(s)
        assert len(finished) == 1
        assert s.prefix_cache_hits >= 1


class TestContiguousModeWithPolicies:
    """Contiguous mode (no paging) still works with policies."""

    def test_contiguous_inference(self, model_path):
        old = os.environ.pop("FORGE_KV_STORAGE_MODE", None)
        try:
            model = forge.Model()
            model.load(model_path, arch_type="llama",
                       vocab_size=100, hidden_dim=32, intermediate_dim=64,
                       num_layers=1, num_heads=2, num_kv_heads=1, head_dim=16,
                       device="cpu")
            s = forge.RequestScheduler(model, block_size=4, max_num_seqs=4)
            s.submit(PROMPT_A, max_new_tokens=2,
                     sampler_config=forge.SamplerConfig(do_sample=False))
            finished = _run_scheduler(s)
            assert len(finished) == 1
            assert len(finished[0].output_tokens) >= 1
        finally:
            if old is not None:
                os.environ["FORGE_KV_STORAGE_MODE"] = old
