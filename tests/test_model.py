import numpy as np
import pytest
import nanoinfer


class TestTransformerModel:
    def test_load_and_forward(self, loaded_model, loaded_context):
        ids = np.array([1, 2, 3], dtype=np.int32)
        logits = loaded_context.forward(ids)
        assert logits.shape == (3, 100)
        assert np.isfinite(logits).all()

    def test_forward_single_token(self, loaded_model, loaded_context):
        ids = np.array([5], dtype=np.int32)
        logits = loaded_context.forward(ids)
        assert logits.shape == (1, 100)
        assert np.isfinite(logits).all()

    def test_forward_different_lengths(self, loaded_model, loaded_context):
        for sl in [1, 2, 3, 5]:
            ids = np.arange(sl, dtype=np.int32) % 100
            logits = loaded_context.forward(ids)
            assert logits.shape[0] == sl
            assert logits.shape[1] == 100

    def test_reproducibility(self, loaded_model, loaded_context):
        ids = np.array([1, 2, 3], dtype=np.int32)
        loaded_context.reset_kv()
        out1 = loaded_context.forward(ids)
        loaded_context.reset_kv()
        out2 = loaded_context.forward(ids)
        np.testing.assert_array_equal(out1, out2)

    def test_rope_position_sensitivity(self, loaded_model, loaded_context):
        ids = np.array([5, 10], dtype=np.int32)
        out0 = loaded_context.forward(ids, start_pos=0)
        out100 = loaded_context.forward(ids, start_pos=100)
        diff = np.abs(out0 - out100).max()
        assert diff > 0, "RoPE should produce different outputs for different positions"

    def test_invalid_path(self, model_config):
        model = nanoinfer.Model()
        with pytest.raises(RuntimeError):
            model.load("/nonexistent/model.ninf", **model_config)

    def test_load_auto(self, model_path, model_config):
        model = nanoinfer.Model()
        model.load_auto(model_path, device="cpu")
        ctx = model.create_context()
        ids = np.array([1, 2, 3], dtype=np.int32)
        logits = ctx.forward(ids)
        assert logits.shape[0] == 3

    def test_detect_format(self, model_path):
        fmt = nanoinfer.Model.detect_format(model_path)
        assert fmt in ("ninf", "gguf")

    def test_model_config_accessible(self, loaded_model):
        cfg = loaded_model.config
        assert cfg.vocab_size == 100
        assert cfg.hidden_dim == 32
        assert cfg.num_layers == 1
