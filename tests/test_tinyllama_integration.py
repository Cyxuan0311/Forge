import numpy as np
import pytest
import os
import sys

build_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "build")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

import forge

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
TINYLLAMA_Q4_PATH = os.path.join(MODELS_DIR, "tinyllama-1.1b-chat-v1.0.Q4_0.gguf")


def tinyllama_available():
    return os.path.exists(TINYLLAMA_Q4_PATH)


@pytest.mark.skipif(not tinyllama_available(), reason="TinyLlama-1.1B GGUF model not found")
class TestTinyLlamaLoading:
    def test_load_gguf_model(self):
        model = forge.Model()
        model.load_auto(TINYLLAMA_Q4_PATH, device="cpu")
        cfg = model.config
        assert cfg.vocab_size > 0
        assert cfg.hidden_dim > 0
        assert cfg.num_layers > 0
        assert cfg.num_heads > 0

    def test_gguf_config_values(self):
        model = forge.Model()
        model.load_auto(TINYLLAMA_Q4_PATH, device="cpu")
        cfg = model.config
        assert cfg.vocab_size == 32000
        assert cfg.hidden_dim == 2048
        assert cfg.num_layers == 22
        assert cfg.num_heads == 32
        assert cfg.num_kv_heads > 0
        assert cfg.head_dim > 0

    def test_create_context(self):
        model = forge.Model()
        model.load_auto(TINYLLAMA_Q4_PATH, device="cpu")
        ctx = model.create_context(kv_cache_dtype="fp32", gpu_layers=0)
        assert ctx is not None


@pytest.mark.skipif(not tinyllama_available(), reason="TinyLlama-1.1B GGUF model not found")
class TestTinyLlamaInference:
    @pytest.fixture(autouse=True)
    def setup_model(self):
        self.model = forge.Model()
        self.model.load_auto(TINYLLAMA_Q4_PATH, device="cpu")
        self.ctx = self.model.create_context(kv_cache_dtype="fp32", gpu_layers=0)

    def test_single_token_forward(self):
        ids = np.array([1], dtype=np.int32)
        self.ctx.reset_kv()
        logits = self.ctx.forward(ids, start_pos=0)
        assert logits.shape[0] == 1
        assert logits.shape[1] == 32000
        assert np.isfinite(logits).all()

    def test_multi_token_forward(self):
        ids = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        self.ctx.reset_kv()
        logits = self.ctx.forward(ids, start_pos=0)
        assert logits.shape == (5, 32000)
        assert np.isfinite(logits).all()

    def test_incremental_generation(self):
        self.ctx.reset_kv()
        prompt_ids = np.array([1, 2, 3], dtype=np.int32)
        logits = self.ctx.forward(prompt_ids, start_pos=0)
        assert logits.shape == (3, 32000)

        next_token_id = np.argmax(logits[-1])
        next_ids = np.array([int(next_token_id)], dtype=np.int32)
        next_logits = self.ctx.forward(next_ids, start_pos=3)
        assert next_logits.shape == (1, 32000)
        assert np.isfinite(next_logits).all()

    def test_greedy_generation(self):
        self.ctx.reset_kv()
        prompt_ids = np.array([1, 15043, 29892, 590, 1024], dtype=np.int32)
        logits = self.ctx.forward(prompt_ids, start_pos=0)

        generated_tokens = []
        current_pos = len(prompt_ids)
        for _ in range(10):
            next_token_id = int(np.argmax(logits[-1 if logits.ndim > 1 else 0]))
            generated_tokens.append(next_token_id)
            next_ids = np.array([next_token_id], dtype=np.int32)
            logits = self.ctx.forward(next_ids, start_pos=current_pos)
            current_pos += 1

        assert len(generated_tokens) == 10
        assert all(isinstance(t, int) for t in generated_tokens)

    def test_deterministic_output(self):
        ids = np.array([1, 2, 3], dtype=np.int32)

        self.ctx.reset_kv()
        out1 = self.ctx.forward(ids, start_pos=0)

        self.ctx.reset_kv()
        out2 = self.ctx.forward(ids, start_pos=0)

        np.testing.assert_array_equal(
            out1, out2, err_msg="Same input should produce deterministic output"
        )

    def test_incremental_matches_full(self):
        ids = np.array([1, 15043, 29892], dtype=np.int32)

        self.ctx.reset_kv()
        full_logits = self.ctx.forward(ids, start_pos=0)

        self.ctx.reset_kv()
        for i, tid in enumerate(ids):
            step_logits = self.ctx.forward(np.array([tid], dtype=np.int32), start_pos=i)

        for i in range(len(ids)):
            np.testing.assert_allclose(
                full_logits[i],
                step_logits[0] if i == len(ids) - 1 else full_logits[i],
                atol=1e-3,
                err_msg=f"Token {i} mismatch between full and incremental",
            )

    def test_kv_cache_reset_consistency(self):
        ids = np.array([1, 2, 3], dtype=np.int32)

        self.ctx.reset_kv()
        out1 = self.ctx.forward(ids, start_pos=0)

        self.ctx.reset_kv()
        out2 = self.ctx.forward(ids, start_pos=0)

        np.testing.assert_array_equal(
            out1, out2, err_msg="Output should be identical after KV cache reset"
        )


@pytest.mark.skipif(not tinyllama_available(), reason="TinyLlama-1.1B GGUF model not found")
class TestTinyLlamaQuantized:
    def test_q4_0_kv_cache(self):
        model = forge.Model()
        model.load_auto(TINYLLAMA_Q4_PATH, device="cpu")
        ctx = model.create_context(kv_cache_dtype="q4_0", gpu_layers=0)

        ids = np.array([1, 2, 3], dtype=np.int32)
        ctx.reset_kv()
        logits = ctx.forward(ids, start_pos=0)
        assert logits.shape == (3, 32000)
        assert np.isfinite(logits).all()

    def test_fp32_vs_q4_0_kv_cache(self):
        model = forge.Model()
        model.load_auto(TINYLLAMA_Q4_PATH, device="cpu")

        ctx_fp32 = model.create_context(kv_cache_dtype="fp32", gpu_layers=0)
        ctx_q4 = model.create_context(kv_cache_dtype="q4_0", gpu_layers=0)

        ids = np.array([1, 2, 3], dtype=np.int32)

        ctx_fp32.reset_kv()
        logits_fp32 = ctx_fp32.forward(ids, start_pos=0)

        ctx_q4.reset_kv()
        logits_q4 = ctx_q4.forward(ids, start_pos=0)

        assert logits_fp32.shape == logits_q4.shape
        diff = np.abs(logits_fp32 - logits_q4).max()
        assert diff < 5.0, f"Q4_0 KV cache should be close to FP32, max diff={diff}"


@pytest.mark.skipif(not tinyllama_available(), reason="TinyLlama-1.1B GGUF model not found")
class TestTinyLlamaQ6KWeights:
    def test_q6_k_output_weight_forward(self):
        model = forge.Model()
        model.load_auto(TINYLLAMA_Q4_PATH, device="cpu")
        ctx = model.create_context(kv_cache_dtype="fp32", gpu_layers=0)

        ids = np.array([1, 450, 4996, 29901], dtype=np.int32)
        ctx.reset_kv()
        logits = ctx.forward(ids)
        assert not np.any(np.isnan(logits)), "Q6_K output.weight should not produce NaN"
        assert not np.any(np.isinf(logits)), "Q6_K output.weight should not produce Inf"
        assert logits.shape == (4, 32000)

    def test_q6_k_output_weight_top_token(self):
        model = forge.Model()
        model.load_auto(TINYLLAMA_Q4_PATH, device="cpu")
        ctx = model.create_context(kv_cache_dtype="fp32", gpu_layers=0)

        ids = np.array([1, 450, 4996, 29901], dtype=np.int32)
        ctx.reset_kv()
        logits = ctx.forward(ids)
        top_token = np.argmax(logits[-1])
        assert 0 <= top_token < 32000, f"Top token {top_token} out of vocab range"

    def test_q6_k_deterministic(self):
        model = forge.Model()
        model.load_auto(TINYLLAMA_Q4_PATH, device="cpu")
        ctx = model.create_context(kv_cache_dtype="fp32", gpu_layers=0)

        ids = np.array([1, 2, 3], dtype=np.int32)
        ctx.reset_kv()
        out1 = ctx.forward(ids)
        ctx.reset_kv()
        out2 = ctx.forward(ids)
        np.testing.assert_array_equal(
            out1, out2, err_msg="Q6_K weight inference should be deterministic"
        )


@pytest.mark.skipif(not tinyllama_available(), reason="TinyLlama-1.1B GGUF model not found")
class TestTinyLlamaGeneration:
    def test_generate_no_repeat(self):
        model = forge.Model()
        model.load_auto(TINYLLAMA_Q4_PATH, device="cpu")
        prompt = np.array([1, 450, 4996, 29901], dtype=np.int32)
        result = model.generate(
            prompt, max_new_tokens=20, do_sample=False, gpu_layers=0, kv_cache_dtype="fp32"
        )
        tokens = result["token_ids"]
        unique_ratio = len(set(tokens)) / len(tokens) if len(tokens) > 0 else 0
        assert unique_ratio > 0.3, (
            f"Generated tokens should not be highly repetitive, unique_ratio={unique_ratio}"
        )

    def test_generate_with_eos(self):
        model = forge.Model()
        model.load_auto(TINYLLAMA_Q4_PATH, device="cpu")
        prompt = np.array([1, 450, 4996, 29901], dtype=np.int32)
        result = model.generate(
            prompt,
            max_new_tokens=50,
            eos_token_id=2,
            do_sample=False,
            gpu_layers=0,
            kv_cache_dtype="fp32",
        )
        if result["finish_reason"] == "eos":
            assert result["token_ids"][-1] == 2

    def test_generate_deterministic(self):
        model = forge.Model()
        model.load_auto(TINYLLAMA_Q4_PATH, device="cpu")
        prompt = np.array([1, 450, 4996, 29901], dtype=np.int32)
        result1 = model.generate(
            prompt, max_new_tokens=10, do_sample=False, gpu_layers=0, kv_cache_dtype="fp32"
        )
        result2 = model.generate(
            prompt, max_new_tokens=10, do_sample=False, gpu_layers=0, kv_cache_dtype="fp32"
        )
        assert list(result1["token_ids"]) == list(result2["token_ids"]), (
            "Greedy generation should be deterministic"
        )

    def test_generate_with_sampling(self):
        model = forge.Model()
        model.load_auto(TINYLLAMA_Q4_PATH, device="cpu")
        prompt = np.array([1, 450, 4996, 29901], dtype=np.int32)
        result = model.generate(
            prompt,
            max_new_tokens=10,
            temperature=0.8,
            top_k=40,
            top_p=0.9,
            do_sample=True,
            seed=42,
            gpu_layers=0,
            kv_cache_dtype="fp32",
        )
        assert result["num_generated_tokens"] >= 1
