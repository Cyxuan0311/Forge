import numpy as np


class TestForwardPass:
    def test_basic_forward(self, loaded_model, loaded_context):
        ids = np.array([1, 2, 3], dtype=np.int32)
        logits = loaded_context.forward(ids)
        assert logits.ndim == 2
        assert logits.shape[0] == 3
        assert logits.shape[1] == 100
        assert np.isfinite(logits).all()

    def test_logits_range(self, loaded_model, loaded_context):
        ids = np.array([1, 2, 3], dtype=np.int32)
        logits = loaded_context.forward(ids)
        assert logits.max() < 1e6, "Logits should be in reasonable range"
        assert logits.min() > -1e6, "Logits should be in reasonable range"

    def test_start_pos_zero(self, loaded_model, loaded_context):
        ids = np.array([1], dtype=np.int32)
        logits = loaded_context.forward(ids, start_pos=0)
        assert logits.shape == (1, 100)

    def test_start_pos_nonzero(self, loaded_model, loaded_context):
        ids = np.array([1], dtype=np.int32)
        logits = loaded_context.forward(ids, start_pos=50)
        assert logits.shape == (1, 100)
        assert np.isfinite(logits).all()

    def test_causal_attention_invariance(self, loaded_model, loaded_context):
        loaded_context.reset_kv()
        ids_short = np.array([5, 10], dtype=np.int32)
        out_short = loaded_context.forward(ids_short)
        loaded_context.reset_kv()
        ids_long = np.array([5, 10, 20], dtype=np.int32)
        out_long = loaded_context.forward(ids_long)
        np.testing.assert_allclose(
            out_short[1],
            out_long[1],
            atol=1e-5,
            err_msg="Causal: token at pos 1 should be invariant to tokens at pos >= 2",
        )

    def test_determinism(self, loaded_model, loaded_context):
        ids = np.array([3, 7, 15], dtype=np.int32)
        loaded_context.reset_kv()
        out1 = loaded_context.forward(ids)
        loaded_context.reset_kv()
        out2 = loaded_context.forward(ids)
        np.testing.assert_array_equal(out1, out2)
