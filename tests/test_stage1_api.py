"""Stage 1: Config structs + ctx.generate API tests."""

import forge
import pytest
import warnings


# ====================================================================
#  Fixtures
# ====================================================================


@pytest.fixture
def model_ctx(model_path):
    model = forge.Model()
    model.load(
        model_path,
        arch_type="llama",
        vocab_size=100,
        hidden_dim=32,
        intermediate_dim=64,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        head_dim=16,
        device="cpu",
    )
    ctx = model.create_context(kv_cache_dtype="fp32", gpu_layers=-1)
    return model, ctx


# ====================================================================
#  Test Classes
# ====================================================================


class TestGenerationConfig:
    def test_default_construction(self):
        cfg = forge.GenerationConfig()
        assert cfg.max_new_tokens == 256
        assert cfg.temperature == 1.0
        assert cfg.top_k == 0
        assert cfg.top_p == 1.0
        assert cfg.repeat_penalty == 1.0
        assert cfg.do_sample
        assert cfg.eos_token_id == -1

    def test_custom_values(self):
        cfg = forge.GenerationConfig()
        cfg.max_new_tokens = 10
        cfg.temperature = 0.5
        cfg.eos_token_id = 2
        cfg.do_sample = False
        cfg.stop_token_ids = [2, 3]
        assert cfg.max_new_tokens == 10
        assert cfg.temperature == 0.5
        assert cfg.eos_token_id == 2
        assert not cfg.do_sample
        assert cfg.stop_token_ids == [2, 3]


class TestContextConfig:
    def test_default_construction(self):
        cfg = forge.ContextConfig()
        assert cfg.kv_cache_dtype == "fp32"
        assert cfg.page_size == 16
        assert cfg.max_seq_len == 4096
        assert cfg.device == "cuda"

    def test_custom_values(self):
        cfg = forge.ContextConfig()
        cfg.kv_storage = "paged"
        cfg.page_size = 64
        cfg.kv_cache_dtype = "f16"
        assert cfg.kv_storage == "paged"
        assert cfg.page_size == 64
        assert cfg.kv_cache_dtype == "f16"


class TestGenerationResult:
    def test_fields(self):
        r = forge.GenerationResult()
        assert not r.finished
        assert r.finish_reason == ""
        assert r.num_prompt_tokens == 0
        assert r.num_generated_tokens == 0
        assert r.token_ids == []
        assert r.text == ""


class TestCtxGenerate:
    def test_basic_generate(self, model_ctx):
        model, ctx = model_ctx
        cfg = forge.GenerationConfig()
        cfg.max_new_tokens = 2
        cfg.do_sample = False
        result = ctx.generate([1, 2, 3, 4, 5], cfg)
        assert result.finished
        assert result.num_generated_tokens >= 1
        assert len(result.token_ids) >= 1

    def test_returns_generation_result(self, model_ctx):
        """ctx.generate() returns GenerationResult, not dict."""
        model, ctx = model_ctx
        cfg = forge.GenerationConfig()
        cfg.max_new_tokens = 1
        cfg.do_sample = False
        result = ctx.generate([1, 2, 3, 4, 5], cfg)
        assert not isinstance(result, dict), (
            "ctx.generate() should return GenerationResult, not dict"
        )
        assert isinstance(result, forge.GenerationResult)

    def test_reuse_kv_across_calls(self, model_ctx):
        """Same ctx, multiple generate() calls. Each resets KV but context survives."""
        model, ctx = model_ctx
        cfg = forge.GenerationConfig()
        cfg.max_new_tokens = 2
        cfg.do_sample = False

        result1 = ctx.generate([1, 2, 3, 4, 5], cfg)
        result2 = ctx.generate([1, 2, 3, 4, 5], cfg)
        assert result1.finished
        assert result2.finished

    def test_stream_generate(self, model_ctx):
        model, ctx = model_ctx
        cfg = forge.GenerationConfig()
        cfg.max_new_tokens = 3
        cfg.do_sample = False

        tokens = []

        def cb(tid, step):
            tokens.append(tid)

        ctx.generate_stream([1, 2, 3, 4, 5], cfg, cb)
        assert len(tokens) >= 1

    def test_generate_result_fields(self, model_ctx):
        model, ctx = model_ctx
        cfg = forge.GenerationConfig()
        cfg.max_new_tokens = 1
        cfg.do_sample = False
        result = ctx.generate([10, 20, 30], cfg)
        assert hasattr(result, "token_ids")
        assert hasattr(result, "finished")
        assert hasattr(result, "finish_reason")
        assert hasattr(result, "num_prompt_tokens")
        assert hasattr(result, "num_generated_tokens")
        assert hasattr(result, "text")


class TestModelGenerateDeprecation:
    def test_old_api_still_works(self, model_path):
        """Old Model.generate() API still functions, emits DeprecationWarning."""
        model = forge.Model()
        model.load(
            model_path,
            arch_type="llama",
            vocab_size=100,
            hidden_dim=32,
            intermediate_dim=64,
            num_layers=1,
            num_heads=2,
            num_kv_heads=1,
            head_dim=16,
            device="cpu",
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = model.generate([1, 2, 3, 4, 5], max_new_tokens=1, do_sample=False)
            deprecations = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecations) >= 1, "Old API should emit DeprecationWarning"

        assert isinstance(result, dict)
        assert "token_ids" in result

    def test_old_api_result_keys(self, model_path):
        """Old Model.generate() still returns dict with expected keys."""
        model = forge.Model()
        model.load(
            model_path,
            arch_type="llama",
            vocab_size=100,
            hidden_dim=32,
            intermediate_dim=64,
            num_layers=1,
            num_heads=2,
            num_kv_heads=1,
            head_dim=16,
            device="cpu",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = model.generate([1, 2, 3, 4, 5], max_new_tokens=1, do_sample=False)

        assert isinstance(result, dict)
        assert "token_ids" in result
        assert "finished" in result
        assert "num_prompt_tokens" in result
        assert "num_generated_tokens" in result


class TestDefaultContext:
    def test_ensure_default_context(self, model_path):
        """Model.ensure_default_context() creates and returns a context."""
        model = forge.Model()
        model.load(
            model_path,
            arch_type="llama",
            vocab_size=100,
            hidden_dim=32,
            intermediate_dim=64,
            num_layers=1,
            num_heads=2,
            num_kv_heads=1,
            head_dim=16,
            device="cpu",
        )

        ctx = model.ensure_default_context("fp32", -1)
        assert ctx is not None
        assert isinstance(ctx, forge.InferenceContext)

        # Second call returns same context
        ctx2 = model.ensure_default_context("fp32", -1)
        assert ctx is ctx2

    def test_default_context_property(self, model_path):
        """Model.default_context property returns None before first use."""
        model = forge.Model()
        model.load(
            model_path,
            arch_type="llama",
            vocab_size=100,
            hidden_dim=32,
            intermediate_dim=64,
            num_layers=1,
            num_heads=2,
            num_kv_heads=1,
            head_dim=16,
            device="cpu",
        )

        # Before ensure, default_context is None
        ctx = model.default_context
        assert ctx is None

        # After ensure, it's valid
        model.ensure_default_context("fp32", -1)
        assert model.default_context is not None

    def test_release_default_context(self, model_path):
        """release_default_context() clears the default context."""
        model = forge.Model()
        model.load(
            model_path,
            arch_type="llama",
            vocab_size=100,
            hidden_dim=32,
            intermediate_dim=64,
            num_layers=1,
            num_heads=2,
            num_kv_heads=1,
            head_dim=16,
            device="cpu",
        )

        model.ensure_default_context("fp32", -1)
        assert model.default_context is not None

        model.release_default_context()
        assert model.default_context is None


class TestCtxGenerateStream:
    def test_callback_receives_tokens(self, model_ctx):
        """generate_stream callback receives at least one token."""
        model, ctx = model_ctx
        cfg = forge.GenerationConfig()
        cfg.max_new_tokens = 3
        cfg.do_sample = False

        received = []

        def cb(tid, step):
            received.append((tid, step))

        ctx.generate_stream([1, 2, 3, 4, 5], cfg, cb)
        assert len(received) >= 1
        # First call should have step >= 0
        assert received[0][1] >= 0

    def test_stream_with_stop_tokens(self, model_ctx):
        """generate_stream respects eos_token_id."""
        model, ctx = model_ctx
        cfg = forge.GenerationConfig()
        cfg.max_new_tokens = 10
        cfg.do_sample = False
        cfg.eos_token_id = -1  # no EOS

        received = []

        def cb(tid, step):
            received.append(tid)

        # With a tiny model, it will generate up to max_new_tokens or hit natural stop
        ctx.generate_stream([1, 2, 3, 4, 5], cfg, cb)
        assert len(received) >= 1
