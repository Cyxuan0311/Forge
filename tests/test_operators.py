import numpy as np
import os
import sys

build_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "build")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)



class TestEmbeddingOp:
    def test_embedding_shape(self, loaded_model, loaded_context):
        ids = np.array([0, 1, 2], dtype=np.int32)
        logits = loaded_context.forward(ids)
        assert logits.shape[0] == 3
        assert logits.shape[1] == 100
        assert np.isfinite(logits).all()

    def test_embedding_different_ids(self, loaded_model, loaded_context):
        ids1 = np.array([0], dtype=np.int32)
        ids2 = np.array([1], dtype=np.int32)
        loaded_context.reset_kv()
        out1 = loaded_context.forward(ids1)
        loaded_context.reset_kv()
        out2 = loaded_context.forward(ids2)
        assert not np.allclose(out1, out2), "Different token IDs should produce different outputs"

    def test_embedding_boundary_ids(self, loaded_model, loaded_context):
        ids = np.array([0, 99], dtype=np.int32)
        logits = loaded_context.forward(ids)
        assert np.isfinite(logits).all()


class TestNormOp:
    def test_rms_norm_output_finite(self, loaded_model, loaded_context):
        ids = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        logits = loaded_context.forward(ids)
        assert np.isfinite(logits).all(), "RMSNorm should produce finite outputs"

    def test_rms_norm_output_range(self, loaded_model, loaded_context):
        ids = np.array([1, 2, 3], dtype=np.int32)
        logits = loaded_context.forward(ids)
        assert np.abs(logits).max() < 1e6, "RMSNorm should keep output in reasonable range"


class TestMatmulOp:
    def test_matmul_produces_logits(self, loaded_model, loaded_context):
        ids = np.array([1], dtype=np.int32)
        logits = loaded_context.forward(ids)
        assert logits.shape[1] == 100, "Output projection should map to vocab_size"


class TestAttentionOp:
    def test_attention_causal_mask(self, loaded_model, loaded_context):
        ids = np.array([1, 2, 3], dtype=np.int32)
        loaded_context.reset_kv()
        out_full = loaded_context.forward(ids)

        loaded_context.reset_kv()
        out_single = loaded_context.forward(np.array([1], dtype=np.int32))
        out_single_next = loaded_context.forward(np.array([2], dtype=np.int32), start_pos=1)

        assert out_full.shape == (3, 100)
        assert out_single.shape == (1, 100)
        np.testing.assert_allclose(out_full[0], out_single[0], atol=1e-5,
                                    err_msg="First token logits should match between full and incremental")

    def test_attention_position_sensitivity(self, loaded_model, loaded_context):
        ids = np.arange(10, dtype=np.int32) % 100
        loaded_context.reset_kv()
        out_pos0 = loaded_context.forward(ids, start_pos=0)
        loaded_context.reset_kv()
        out_pos1000 = loaded_context.forward(ids, start_pos=1000)
        diff = np.abs(out_pos0 - out_pos1000).max()
        assert diff > 0, "RoPE should produce position-dependent outputs with multi-token input"

    def test_attention_incremental_vs_full(self, loaded_model, loaded_context):
        ids = np.array([1, 2, 3, 4, 5], dtype=np.int32)

        loaded_context.reset_kv()
        full_out = loaded_context.forward(ids)

        loaded_context.reset_kv()
        incremental_outputs = []
        for i, tid in enumerate(ids):
            out = loaded_context.forward(np.array([tid], dtype=np.int32), start_pos=i)
            incremental_outputs.append(out[0])

        for i in range(len(ids)):
            np.testing.assert_allclose(full_out[i], incremental_outputs[i], atol=1e-4,
                                        err_msg=f"Token {i} mismatch between full and incremental")


class TestRopeOp:
    def test_rope_different_positions(self, loaded_model, loaded_context):
        ids = np.arange(10, dtype=np.int32) % 100
        loaded_context.reset_kv()
        out0 = loaded_context.forward(ids, start_pos=0)
        loaded_context.reset_kv()
        out500 = loaded_context.forward(ids, start_pos=500)
        loaded_context.reset_kv()
        out1000 = loaded_context.forward(ids, start_pos=1000)

        diff_0_500 = np.abs(out0 - out500).max()
        diff_500_1000 = np.abs(out500 - out1000).max()
        assert diff_0_500 > 0, "RoPE should differentiate positions 0 and 500"
        assert diff_500_1000 > 0, "RoPE should differentiate positions 500 and 1000"

    def test_rope_incremental_consistency(self, loaded_model, loaded_context):
        ids = np.arange(5, dtype=np.int32) % 100
        loaded_context.reset_kv()
        full_out = loaded_context.forward(ids, start_pos=0)

        loaded_context.reset_kv()
        incremental_outputs = []
        for i, tid in enumerate(ids):
            out = loaded_context.forward(np.array([tid], dtype=np.int32), start_pos=i)
            incremental_outputs.append(out[0])

        for i in range(len(ids)):
            np.testing.assert_allclose(full_out[i], incremental_outputs[i], atol=1e-4,
                                        err_msg=f"Token {i} RoPE mismatch between full and incremental")


class TestFFNOp:
    def test_ffn_output_finite(self, loaded_model, loaded_context):
        ids = np.array([1, 2, 3], dtype=np.int32)
        logits = loaded_context.forward(ids)
        assert np.isfinite(logits).all(), "FFN (SiLU+GELU) should produce finite outputs"

    def test_ffn_nonlinear(self, loaded_model, loaded_context):
        ids = np.array([1], dtype=np.int32)
        loaded_context.reset_kv()
        out = loaded_context.forward(ids)
        assert not np.allclose(out, 0), "FFN should produce non-zero output"


class TestKVCachOp:
    def test_kv_cache_reset(self, loaded_model, loaded_context):
        ids = np.array([1, 2, 3], dtype=np.int32)
        loaded_context.reset_kv()
        out1 = loaded_context.forward(ids)
        loaded_context.reset_kv()
        out2 = loaded_context.forward(ids)
        np.testing.assert_array_equal(out1, out2, "Reset KV cache should give same results")

    def test_kv_cache_incremental(self, loaded_model, loaded_context):
        ids = np.array([1, 2, 3], dtype=np.int32)
        loaded_context.reset_kv()
        out1 = loaded_context.forward(ids)

        loaded_context.reset_kv()
        loaded_context.forward(np.array([1], dtype=np.int32), start_pos=0)
        loaded_context.forward(np.array([2], dtype=np.int32), start_pos=1)
        out2_last = loaded_context.forward(np.array([3], dtype=np.int32), start_pos=2)

        np.testing.assert_allclose(out1[2], out2_last[0], atol=1e-4,
                                    err_msg="Incremental KV cache should match full forward")

    def test_kv_cache_memory_stats(self, loaded_model, loaded_context):
        stats = loaded_context.memory_stats()
        assert "kv_cache_nbytes" in stats
        assert stats["kv_cache_nbytes"] >= 0

    def test_kv_cache_persistence(self, loaded_model, loaded_context):
        loaded_context.reset_kv()
        loaded_context.forward(np.array([1], dtype=np.int32), start_pos=0)
        out_a = loaded_context.forward(np.array([2], dtype=np.int32), start_pos=1)

        loaded_context.reset_kv()
        loaded_context.forward(np.array([1], dtype=np.int32), start_pos=0)
        loaded_context.forward(np.array([2], dtype=np.int32), start_pos=1)
        out_b = loaded_context.forward(np.array([3], dtype=np.int32), start_pos=2)

        assert out_b.shape == (1, 100), "Continued generation should work after KV cache persistence"
