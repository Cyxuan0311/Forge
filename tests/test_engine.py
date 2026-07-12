import os
import sys
import pytest
import numpy as np

build_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "build")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

import nanoinfer


@pytest.fixture
def model_path():
    path = os.path.join(os.path.dirname(__file__), "fixtures", "test_model_small.ninf")
    if not os.path.exists(path):
        pytest.skip("test_model_small.ninf not found")
    return path


@pytest.fixture
def model_config():
    return {
        "vocab_size": 100,
        "hidden_dim": 32,
        "intermediate_dim": 64,
        "num_layers": 1,
        "num_heads": 2,
        "num_kv_heads": 1,
        "head_dim": 16,
        "device": "cpu",
    }


class TestEngineRegistry:
    def test_registered_archs_not_empty(self):
        model = nanoinfer.Model()
        archs = model.registered_archs()
        assert len(archs) > 0

    def test_llama_arch_registered(self):
        model = nanoinfer.Model()
        archs = model.registered_archs()
        assert "llama" in archs

    def test_mistral_arch_registered(self):
        model = nanoinfer.Model()
        archs = model.registered_archs()
        assert "mistral" in archs

    def test_qwen_arch_registered(self):
        model = nanoinfer.Model()
        archs = model.registered_archs()
        assert "qwen" in archs

    def test_qwen2_arch_registered(self):
        model = nanoinfer.Model()
        archs = model.registered_archs()
        assert "qwen2" in archs


class TestArchLoading:
    def test_load_with_explicit_llama_arch(self, model_path, model_config):
        model = nanoinfer.Model()
        model.load(model_path, arch_type="llama", **model_config)
        ctx = model.create_context()
        ids = np.array([1, 2, 3], dtype=np.int32)
        out = ctx.forward(ids)
        assert out.shape == (3, model_config["vocab_size"])

    def test_load_with_mistral_arch(self, model_path, model_config):
        model = nanoinfer.Model()
        model.load(model_path, arch_type="mistral", **model_config)
        ctx = model.create_context()
        ids = np.array([1, 2, 3], dtype=np.int32)
        out = ctx.forward(ids)
        assert out.shape == (3, model_config["vocab_size"])

    def test_load_with_qwen_arch(self, model_path, model_config):
        model = nanoinfer.Model()
        model.load(model_path, arch_type="qwen", **model_config)
        ctx = model.create_context()
        ids = np.array([1, 2, 3], dtype=np.int32)
        out = ctx.forward(ids)
        assert out.shape == (3, model_config["vocab_size"])

    def test_load_with_yi_arch(self, model_path, model_config):
        model = nanoinfer.Model()
        model.load(model_path, arch_type="yi", **model_config)
        ctx = model.create_context()
        ids = np.array([1, 2, 3], dtype=np.int32)
        out = ctx.forward(ids)
        assert out.shape == (3, model_config["vocab_size"])

    def test_load_with_deepseek_arch(self, model_path, model_config):
        model = nanoinfer.Model()
        model.load(model_path, arch_type="deepseek", **model_config)
        ctx = model.create_context()
        ids = np.array([1, 2, 3], dtype=np.int32)
        out = ctx.forward(ids)
        assert out.shape == (3, model_config["vocab_size"])

    def test_unsupported_arch_raises(self, model_path, model_config):
        model = nanoinfer.Model()
        model.load(model_path, arch_type="nonexistent_arch", **model_config)
        with pytest.raises(RuntimeError, match="No engine registered"):
            model.create_context()

    def test_same_weights_different_arch_names(self, model_path, model_config):
        model_a = nanoinfer.Model()
        model_a.load(model_path, arch_type="llama", **model_config)
        model_b = nanoinfer.Model()
        model_b.load(model_path, arch_type="mistral", **model_config)

        ctx_a = model_a.create_context()
        ctx_b = model_b.create_context()
        ids = np.array([5, 10, 15], dtype=np.int32)
        out_a = ctx_a.forward(ids)
        out_b = ctx_b.forward(ids)
        np.testing.assert_allclose(out_a, out_b, atol=1e-5,
                                   err_msg="Same weights with different arch names should produce same output")


class TestModelConfigEnums:
    def test_norm_type_enum(self):
        assert nanoinfer.NormType.RMSNorm is not None
        assert nanoinfer.NormType.LayerNorm is not None

    def test_activation_type_enum(self):
        assert nanoinfer.ActivationType.SiLU_GELU is not None
        assert nanoinfer.ActivationType.GELU is not None
        assert nanoinfer.ActivationType.ReLU is not None

    def test_rope_type_enum(self):
        assert nanoinfer.RopeType.Standard is not None
        assert nanoinfer.RopeType.LinearScaling is not None
        assert nanoinfer.RopeType.NTK_Scaled is not None

    def test_load_with_layernorm_option(self, model_path, model_config):
        model = nanoinfer.Model()
        model.load(model_path, norm_type="layernorm", **model_config)
        ctx = model.create_context()
        ids = np.array([1, 2], dtype=np.int32)
        out = ctx.forward(ids)
        assert out.shape == (2, model_config["vocab_size"])

    def test_load_with_gelu_option(self, model_path, model_config):
        model = nanoinfer.Model()
        model.load(model_path, activation="gelu", **model_config)
        ctx = model.create_context()
        ids = np.array([1, 2], dtype=np.int32)
        out = ctx.forward(ids)
        assert out.shape == (2, model_config["vocab_size"])

    def test_load_with_tie_embeddings(self, model_path, model_config):
        model = nanoinfer.Model()
        model.load(model_path, tie_embeddings=True, **model_config)
        ctx = model.create_context()
        ids = np.array([1, 2], dtype=np.int32)
        out = ctx.forward(ids)
        assert out.shape == (2, model_config["vocab_size"])


class TestGenerate:
    def test_generate_basic(self, loaded_model):
        prompt = np.array([1, 2, 3], dtype=np.int32)
        result = loaded_model.generate(prompt, max_new_tokens=5, do_sample=False)
        assert "token_ids" in result
        assert "num_prompt_tokens" in result
        assert "num_generated_tokens" in result
        assert "finished" in result
        assert "finish_reason" in result
        assert result["num_prompt_tokens"] == 3
        assert result["num_generated_tokens"] <= 5
        assert result["num_generated_tokens"] >= 1
        assert result["finished"] is True

    def test_generate_greedy_deterministic(self, loaded_model):
        prompt = np.array([1, 2, 3], dtype=np.int32)
        result1 = loaded_model.generate(prompt, max_new_tokens=10, do_sample=False)
        result2 = loaded_model.generate(prompt, max_new_tokens=10, do_sample=False)
        assert list(result1["token_ids"]) == list(result2["token_ids"])

    def test_generate_with_temperature(self, loaded_model):
        prompt = np.array([1, 2, 3], dtype=np.int32)
        result = loaded_model.generate(prompt, max_new_tokens=5, temperature=0.8, do_sample=True, seed=42)
        assert result["num_generated_tokens"] >= 1

    def test_generate_max_tokens_limit(self, loaded_model):
        prompt = np.array([1, 2], dtype=np.int32)
        result = loaded_model.generate(prompt, max_new_tokens=3, do_sample=False)
        assert result["num_generated_tokens"] <= 3
        assert result["finish_reason"] == "length"

    def test_generate_single_token_prompt(self, loaded_model):
        prompt = np.array([5], dtype=np.int32)
        result = loaded_model.generate(prompt, max_new_tokens=4, do_sample=False)
        assert result["num_prompt_tokens"] == 1
        assert result["num_generated_tokens"] >= 1

    def test_generate_with_eos(self, loaded_model):
        prompt = np.array([1, 2, 3], dtype=np.int32)
        result = loaded_model.generate(prompt, max_new_tokens=10, eos_token_id=5, do_sample=False)
        assert result["finished"] is True
        if result["finish_reason"] == "eos":
            assert result["token_ids"][-1] == 5

    def test_generate_with_top_k(self, loaded_model):
        prompt = np.array([1, 2, 3], dtype=np.int32)
        result = loaded_model.generate(prompt, max_new_tokens=5, top_k=10, do_sample=True, seed=123)
        assert result["num_generated_tokens"] >= 1

    def test_generate_with_top_p(self, loaded_model):
        prompt = np.array([1, 2, 3], dtype=np.int32)
        result = loaded_model.generate(prompt, max_new_tokens=5, top_p=0.9, do_sample=True, seed=456)
        assert result["num_generated_tokens"] >= 1
