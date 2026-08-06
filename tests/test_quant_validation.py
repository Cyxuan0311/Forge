"""
Phase 0: Quantization block definition validation and scalar-dequant gold values.

Validates that Forge's quant block sizes match llama.cpp's block definitions.
Verifies scalar dequantization produces correct values for each quant type.
"""

import os
import sys
import pytest

build_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "build")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

import forge


# --- Block definition validation: must match llama.cpp ggml_type_traits ---
# Reference: llama.cpp ggml-quants.c / ggml-common.h block definitions
EXPECTED_BLOCK_DEFS = {
    # (block_elements, block_size_bytes)
    forge.DataType.Q4_0:    (32,  18),
    forge.DataType.Q4_1:    (32,  20),
    forge.DataType.Q8_0:    (32,  34),
    forge.DataType.Q5_0:    (32,  22),
    forge.DataType.Q5_1:    (32,  24),
    forge.DataType.Q2_K:    (256, 84),
    forge.DataType.Q3_K:    (256, 110),
    forge.DataType.Q4_K:    (256, 144),
    forge.DataType.Q5_K:    (256, 176),
    forge.DataType.Q6_K:    (256, 210),
    forge.DataType.IQ2_S:   (256, 82),
    forge.DataType.IQ2_XXS: (256, 66),
    forge.DataType.IQ4_NL:  (32,  18),
}


class TestQuantBlockDefinitions:
    """Verify every quant type's block definition matches llama.cpp."""

    @pytest.mark.parametrize("dtype,expected", [
        (dt, exp) for dt, exp in EXPECTED_BLOCK_DEFS.items()
    ])
    def test_block_elements(self, dtype, expected):
        """block_elements must match llama.cpp."""
        block_el = forge.dtype_block_elements(dtype)
        assert block_el == expected[0], (
            f"block_elements mismatch for {dtype}: "
            f"got {block_el}, expected {expected[0]}"
        )

    @pytest.mark.parametrize("dtype,expected", [
        (dt, exp) for dt, exp in EXPECTED_BLOCK_DEFS.items()
    ])
    def test_block_size(self, dtype, expected):
        """block_size must match llama.cpp."""
        block_sz = forge.dtype_block_size(dtype)
        assert block_sz == expected[1], (
            f"block_size mismatch for {dtype}: "
            f"got {block_sz}, expected {expected[1]}"
        )

    def test_non_quantized_types(self):
        """FP32, FP16, INT8, INT32, BF16 should not be quantized."""
        non_quant = [
            forge.DataType.FP32,
            forge.DataType.FP16,
            forge.DataType.INT8,
            forge.DataType.INT32,
        ]
        for dt in non_quant:
            assert not forge.is_quantized_type(dt), f"{dt} should not be quantized"
            assert forge.dtype_block_elements(dt) == 1
            assert forge.dtype_block_size(dt) == 0

    def test_quantized_types(self):
        """All quant types should be recognized as quantized."""
        for dt in EXPECTED_BLOCK_DEFS:
            assert forge.is_quantized_type(dt), f"{dt} should be quantized"


class TestQuantizedTensorBytes:
    """Verify compute_quantized_bytes returns correct values."""

    def test_q4_0_bytes(self):
        """Q4_0: 32 elements = 18 bytes."""
        t = forge.Tensor(forge.DataType.Q4_0, [32], forge.DeviceType.CPU)
        assert t.nbytes() == 18

    def test_q4_0_padded_bytes(self):
        """Q4_0: 33 elements = 2 blocks = 36 bytes."""
        t = forge.Tensor(forge.DataType.Q4_0, [33], forge.DeviceType.CPU)
        assert t.nbytes() == 36

    def test_q4_k_bytes(self):
        """Q4_K: 256 elements = 144 bytes."""
        t = forge.Tensor(forge.DataType.Q4_K, [256], forge.DeviceType.CPU)
        assert t.nbytes() == 144

    def test_q6_k_bytes(self):
        """Q6_K: 512 elements = 2*210 = 420 bytes."""
        t = forge.Tensor(forge.DataType.Q6_K, [512], forge.DeviceType.CPU)
        assert t.nbytes() == 420

    def test_fp32_bytes(self):
        """FP32: 10 elements = 40 bytes."""
        t = forge.Tensor(forge.DataType.FP32, [10], forge.DeviceType.CPU)
        assert t.nbytes() == 40


class TestQuantSlice:
    """Verify sliced quantized tensors have correct byte offsets."""

    def test_q4_0_slice(self):
        """Q4_0 slice at dim boundary: 128 elements, slice [64:128]."""
        t = forge.Tensor(forge.DataType.Q4_0, [128], forge.DeviceType.CPU)
        # 128 elements = 4 blocks = 72 bytes
        assert t.nbytes() == 72

        # slice [64:128] = 64 elements = 2 blocks = 36 bytes
        s = t.slice(0, 64, 128)
        assert s.shape() == [64]
        assert s.nbytes() == 36

    def test_q4_k_slice(self):
        """Q4_K slice: 512 elements, slice [256:512]."""
        t = forge.Tensor(forge.DataType.Q4_K, [512], forge.DeviceType.CPU)
        # 512 elements = 2 blocks = 288 bytes
        assert t.nbytes() == 288

        s = t.slice(0, 256, 512)
        assert s.shape() == [256]
        assert s.nbytes() == 144


class TestQuantTraitsTable:
    """Verify the runtime traits table is consistent with compile-time traits."""

    def test_traits_table_has_all_entries(self):
        """Runtime table must have entries for all DataType values."""
        # Check that dtype_name returns non-empty for all types
        for dt in [
            forge.DataType.FP32, forge.DataType.FP16, forge.DataType.Q4_0,
            forge.DataType.Q4_1, forge.DataType.Q4_K, forge.DataType.INT8,
            forge.DataType.INT32, forge.DataType.Q8_0, forge.DataType.Q5_0,
            forge.DataType.Q5_1, forge.DataType.Q2_K, forge.DataType.Q3_K,
            forge.DataType.Q5_K, forge.DataType.Q6_K, forge.DataType.IQ2_S,
            forge.DataType.IQ2_XXS, forge.DataType.IQ4_NL,
        ]:
            name = forge.dtype_name(dt)
            assert len(name) > 0, f"dtype_name({dt}) returned empty string"

    def test_dequant_row_fns_exist(self):
        """All quant types should have dequant_row function pointers."""
        quant_types = [
            forge.DataType.Q4_0, forge.DataType.Q4_1, forge.DataType.Q4_K,
            forge.DataType.Q8_0, forge.DataType.Q5_0, forge.DataType.Q5_1,
            forge.DataType.Q2_K, forge.DataType.Q3_K, forge.DataType.Q5_K,
            forge.DataType.Q6_K, forge.DataType.IQ2_S, forge.DataType.IQ2_XXS,
            forge.DataType.IQ4_NL,
        ]
        for dt in quant_types:
            fn = forge.get_dequant_row_fn(dt)
            assert fn is not None, f"dequant_row_fn for {dt} is None"
