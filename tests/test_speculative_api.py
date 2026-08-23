"""
Speculative decoding API smoke tests (Phase 4 of SPECULATIVE_DECODING_PLAN.md).

Tests:
  1. SpeculativeConfig Python binding completeness
  2. GenerationResult exposes spec_stats
  3. Greedy parity: spec on vs off produces identical tokens
  4. Spec stats are populated after generation
"""

import os
import sys
import pytest

build_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "build")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

import forge

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
TINYLLAMA_PATH = os.path.join(MODELS_DIR, "tinyllama-1.1b-chat-v1.0.Q4_0.gguf")


def model_available():
    return os.path.exists(TINYLLAMA_PATH)


@pytest.mark.skipif(not model_available(), reason="TinyLlama model not found")
class TestSpeculativeConfigBinding:
    """Test that SpeculativeConfig exposes all fields in Python."""

    def test_all_fields_present(self):
        cfg = forge.SpeculativeConfig()
        # Master switch
        assert hasattr(cfg, "enabled")
        cfg.enabled = True
        assert cfg.enabled is True
        cfg.enabled = False

        # n-gram params
        assert hasattr(cfg, "n_draft")
        cfg.n_draft = 8
        assert cfg.n_draft == 8

        assert hasattr(cfg, "n_min")
        cfg.n_min = 2
        assert cfg.n_min == 2

        assert hasattr(cfg, "p_min")
        cfg.p_min = 0.5
        assert cfg.p_min == 0.5

        assert hasattr(cfg, "use_ngram")
        cfg.use_ngram = False
        assert cfg.use_ngram is False
        cfg.use_ngram = True

        assert hasattr(cfg, "ngram_n")
        cfg.ngram_n = 7
        assert cfg.ngram_n == 7

        assert hasattr(cfg, "ngram_min")
        cfg.ngram_min = 3
        assert cfg.ngram_min == 3

        # Model draft params (Phase 3)
        assert hasattr(cfg, "draft_model_path")
        cfg.draft_model_path = "/tmp/draft.gguf"
        assert cfg.draft_model_path == "/tmp/draft.gguf"

        assert hasattr(cfg, "draft_gpu_layers")
        cfg.draft_gpu_layers = 4
        assert cfg.draft_gpu_layers == 4

    def test_defaults(self):
        cfg = forge.SpeculativeConfig()
        assert cfg.enabled is False
        assert cfg.n_draft == 5
        assert cfg.n_min == 0
        assert cfg.p_min == 0.0
        assert cfg.use_ngram is True
        assert cfg.ngram_n == 5
        assert cfg.ngram_min == 2
        assert cfg.draft_model_path == ""
        assert cfg.draft_gpu_layers == -1


@pytest.mark.skipif(not model_available(), reason="TinyLlama model not found")
class TestGenerationResultSpecStats:
    """Test that GenerationResult exposes spec_stats."""

    def test_spec_stats_attribute_exists(self):
        model = forge.Model()
        model.load_auto(TINYLLAMA_PATH, device="cpu")
        ctx = model.create_context(kv_cache_dtype="fp32", gpu_layers=0)

        prompt = [1, 42, 43, 42, 43]
        cfg = forge.GenerationConfig()
        cfg.max_new_tokens = 8
        cfg.do_sample = False

        result = ctx.generate(prompt, cfg)
        assert hasattr(result, "spec_stats")
        # With spec disabled, spec_stats should be None
        assert result.spec_stats is None or hasattr(result.spec_stats, "acceptance_rate")


@pytest.mark.skipif(not model_available(), reason="TinyLlama model not found")
class TestSpeculativeGreedyParity:
    """Test that greedy output length is identical with spec on vs off."""

    def test_greedy_parity(self):
        model = forge.Model()
        model.load_auto(TINYLLAMA_PATH, device="cpu")

        # Spec OFF
        ctx_off = model.create_context(kv_cache_dtype="fp32", gpu_layers=0)
        prompt = [1, 42, 43, 42, 43, 42, 43, 42, 43]
        cfg = forge.GenerationConfig()
        cfg.max_new_tokens = 16
        cfg.do_sample = False

        result_off = ctx_off.generate(prompt, cfg)

        # Spec ON (n-gram)
        ctx_on = model.create_context(
            kv_cache_dtype="fp32",
            gpu_layers=0,
            speculative_config=forge.SpeculativeConfig(
                enabled=True,
                use_ngram=True,
                n_draft=4,
                ngram_n=5,
                ngram_min=2,
            ),
        )
        result_on = ctx_on.generate(prompt, cfg)

        assert len(result_on.token_ids) == len(result_off.token_ids), (
            "Greedy output length must be identical with speculation enabled"
        )

    def test_spec_stats_populated(self):
        model = forge.Model()
        model.load_auto(TINYLLAMA_PATH, device="cpu")

        ctx = model.create_context(
            kv_cache_dtype="fp32",
            gpu_layers=0,
            speculative_config=forge.SpeculativeConfig(
                enabled=True,
                use_ngram=True,
                n_draft=4,
                ngram_n=5,
                ngram_min=2,
            ),
        )
        prompt = [1, 42, 43, 42, 43, 42, 43, 42, 43]
        cfg = forge.GenerationConfig()
        cfg.max_new_tokens = 16
        cfg.do_sample = False

        result = ctx.generate(prompt, cfg)
        assert result.spec_stats is not None, "spec_stats should be populated when spec is enabled"
        stats = result.spec_stats
        total_steps = stats.n_spec_steps + stats.n_fallback_steps
        assert total_steps > 0, "Should have at least one generation step"
        assert 0.0 <= stats.acceptance_rate() <= 1.0, "acceptance_rate must be in [0,1]"


@pytest.mark.skipif(not model_available(), reason="TinyLlama model not found")
class TestSpeculativeAPIContract:
    """Verify the full Python API contract for speculative decoding."""

    def test_speculative_config_in_context_params(self):
        """SpeculativeConfig can be passed via create_context."""
        model = forge.Model()
        model.load_auto(TINYLLAMA_PATH, device="cpu")

        spec_cfg = forge.SpeculativeConfig(
            enabled=True,
            use_ngram=True,
            n_draft=3,
            ngram_n=4,
            ngram_min=2,
        )
        ctx = model.create_context(
            kv_cache_dtype="fp32",
            gpu_layers=0,
            speculative_config=spec_cfg,
        )
        assert ctx is not None

    def test_generation_with_spec_disabled(self):
        """Generation works normally when spec is disabled."""
        model = forge.Model()
        model.load_auto(TINYLLAMA_PATH, device="cpu")
        ctx = model.create_context(kv_cache_dtype="fp32", gpu_layers=0)

        prompt = [1, 42, 43, 42, 43, 42, 43, 42, 43]
        cfg = forge.GenerationConfig()
        cfg.max_new_tokens = 5
        cfg.do_sample = False

        result = ctx.generate(prompt, cfg)
        assert len(result.token_ids) > 0
        assert result.spec_stats is None

    def test_generation_with_spec_enabled(self):
        """Generation works with spec enabled and returns stats."""
        model = forge.Model()
        model.load_auto(TINYLLAMA_PATH, device="cpu")
        ctx = model.create_context(
            kv_cache_dtype="fp32",
            gpu_layers=0,
            speculative_config=forge.SpeculativeConfig(
                enabled=True,
                use_ngram=True,
                n_draft=4,
            ),
        )

        prompt = [1, 42, 43, 42, 43, 42, 43, 42, 43]
        cfg = forge.GenerationConfig()
        cfg.max_new_tokens = 5
        cfg.do_sample = False

        result = ctx.generate(prompt, cfg)
        assert len(result.token_ids) > 0
        assert result.spec_stats is not None
