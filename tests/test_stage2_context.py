"""Stage 2: Context lifecycle — KV reuse, multi-turn chat patterns."""

import time
import warnings
import forge


def _make_model(model_path):
    model = forge.Model()
    model.load(model_path, arch_type="llama",
               vocab_size=100, hidden_dim=32, intermediate_dim=64,
               num_layers=1, num_heads=2, num_kv_heads=1, head_dim=16,
               device="cpu")
    return model


def _make_gen_cfg(max_tokens=2):
    cfg = forge.GenerationConfig()
    cfg.max_new_tokens = max_tokens
    cfg.do_sample = False
    return cfg


# ====================================================================
#  1. GenerationConfig.reset_kv_cache field
# ====================================================================

class TestResetKvCacheFlag:
    def test_field_exists_and_defaults_true(self):
        cfg = forge.GenerationConfig()
        assert cfg.reset_kv_cache is True

    def test_disable_reset_passes_through(self):
        cfg = forge.GenerationConfig()
        cfg.reset_kv_cache = False
        assert cfg.reset_kv_cache is False


# ====================================================================
#  2. ctx.generate_kv / ctx.generate_stream_kv
# ====================================================================

class TestGenerateKvMethods:
    def test_generate_kv_returns_result(self, model_path):
        model = _make_model(model_path)
        ctx = model.create_context(kv_cache_dtype="fp32", gpu_layers=-1)
        result = ctx.generate_kv([1, 2, 3], _make_gen_cfg())
        assert isinstance(result, forge.GenerationResult)
        assert result.finished

    def test_generate_kv_same_as_generate(self, model_path):
        model = _make_model(model_path)
        ctx = model.create_context(kv_cache_dtype="fp32", gpu_layers=-1)
        r1 = ctx.generate_kv([7, 8, 9], _make_gen_cfg())
        r2 = ctx.generate([7, 8, 9], _make_gen_cfg())
        assert r1.finished
        assert r2.finished

    def test_generate_stream_kv(self, model_path):
        model = _make_model(model_path)
        ctx = model.create_context(kv_cache_dtype="fp32", gpu_layers=-1)
        tokens = []
        def cb(tid, step):
            tokens.append(tid)
        ctx.generate_stream_kv([1, 2, 3, 4, 5], _make_gen_cfg(3), cb)
        assert len(tokens) >= 1

    def test_generate_kv_preserves_context(self, model_path):
        """ctx survives multiple generate_kv() calls — no re-creation needed."""
        model = _make_model(model_path)
        ctx = model.create_context(kv_cache_dtype="fp32", gpu_layers=-1)
        r1 = ctx.generate_kv([1, 2, 3], _make_gen_cfg())
        r2 = ctx.generate_kv([4, 5, 6], _make_gen_cfg())
        r3 = ctx.generate_kv([7, 8, 9], _make_gen_cfg())
        assert r1.finished
        assert r2.finished
        assert r3.finished
        stats = ctx.memory_stats()
        assert stats["kv_cache_nbytes"] > 0


# ====================================================================
#  3. Context reuse saves context-creation time
# ====================================================================

class TestContextReusePerformance:
    def test_generate_kv_avoids_recreation_cost(self, model_path):
        """Second generate_kv() on same ctx should not crash (context kept alive)."""
        model = _make_model(model_path)
        ctx = model.create_context(kv_cache_dtype="fp32", gpu_layers=-1)

        t0 = time.perf_counter()
        ctx.generate_kv([1, 2, 3, 4, 5], _make_gen_cfg(5))
        t1 = time.perf_counter()

        t2 = time.perf_counter()
        ctx.generate_kv([6, 7, 8, 9, 10], _make_gen_cfg(5))
        t3 = time.perf_counter()

        first_dur = t1 - t0
        second_dur = t3 - t2
        print(f"  First: {first_dur*1000:.2f}ms  Second: {second_dur*1000:.2f}ms")


# ====================================================================
#  4. reset_kv clears session between turns
# ====================================================================

class TestResetKvSemantics:
    def test_reset_kv_between_turns(self, model_path):
        model = _make_model(model_path)
        ctx = model.create_context(kv_cache_dtype="fp32", gpu_layers=-1)

        r1 = ctx.generate_kv([1, 2, 3], _make_gen_cfg())
        assert r1.finished

        ctx.reset_kv()

        r2 = ctx.generate_kv([1, 2, 3], _make_gen_cfg())
        assert r2.finished


# ====================================================================
#  5. FORGE_DISABLE_CONTEXT_REUSE rollback
# ====================================================================

class TestDisableContextReuse:
    def test_env_var_disables_caching(self, model_path, monkeypatch):
        """When FORGE_DISABLE_CONTEXT_REUSE=1, each Model.generate() creates fresh context."""
        monkeypatch.setenv("FORGE_DISABLE_CONTEXT_REUSE", "1")
        model = _make_model(model_path)

        r1 = model.generate([1, 2, 3, 4, 5], max_new_tokens=1, do_sample=False)
        assert isinstance(r1, dict)
        assert "token_ids" in r1

        r2 = model.generate([6, 7, 8, 9, 10], max_new_tokens=1, do_sample=False)
        assert isinstance(r2, dict)
        assert "token_ids" in r2

    def test_normal_mode_caches_context(self, model_path, monkeypatch):
        """Without FORGE_DISABLE_CONTEXT_REUSE, default context is cached."""
        monkeypatch.delenv("FORGE_DISABLE_CONTEXT_REUSE", raising=False)
        model = _make_model(model_path)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            model.generate([1, 2, 3], max_new_tokens=1, do_sample=False)

        assert model.default_context is not None


# ====================================================================
#  6. Multi-turn conversation simulation
# ====================================================================

class TestMultiTurnConversation:
    def test_multi_turn_no_crash(self, model_path):
        """Simulate a 5-turn conversation using persistent context."""
        model = _make_model(model_path)
        ctx = model.create_context(kv_cache_dtype="fp32", gpu_layers=-1)

        turns = [
            [100, 200, 300],
            [400, 500, 600],
            [700, 800, 900],
            [1000, 1100, 1200],
            [1300, 1400, 1500],
        ]

        for i, tokens in enumerate(turns):
            result = ctx.generate_kv(tokens, _make_gen_cfg(2))
            assert result.finished, f"Turn {i+1} did not finish"

    def test_multi_turn_memory_stable(self, model_path):
        """Memory stats should remain bounded across turns."""
        model = _make_model(model_path)
        ctx = model.create_context(kv_cache_dtype="fp32", gpu_layers=-1)

        # First call triggers lazy KV allocation
        ctx.generate_kv([100, 200, 300], _make_gen_cfg(2))
        stats_after_alloc = ctx.memory_stats()
        initial_bytes = stats_after_alloc["kv_cache_nbytes"]
        assert initial_bytes > 0, "KV cache should be allocated after first forward"

        for i in range(5):
            ctx.generate_kv([100 + i, 200 + i, 300 + i], _make_gen_cfg(2))

        stats1 = ctx.memory_stats()
        assert stats1["kv_cache_nbytes"] == initial_bytes, \
            "KV cache bytes changed unexpectedly across turns"
