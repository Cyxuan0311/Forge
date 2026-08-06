"""
Phase 1: Tensor Storage/Layout unit tests.

Tests for:
  - Basic creation, zero, numpy roundtrip (existing)
  - view() with storage/layout metadata
  - slice() with byte_offset
  - from_buffer() with storage semantics
  - storage() and layout() accessors
  - allocation_bytes() vs nbytes()
  - byte_offset() correctness
  - Quantized tensor view/slice correctness
"""

import numpy as np
import forge


class TestTensorCreation:
    def test_create_fp32_cpu(self):
        t = forge.Tensor(forge.DataType.FP32, [3, 4], forge.DeviceType.CPU)
        assert t.shape() == [3, 4]
        assert t.numel() == 12
        assert t.dtype() == forge.DataType.FP32
        assert t.device() == forge.DeviceType.CPU

    def test_zero(self):
        t = forge.Tensor(forge.DataType.FP32, [2, 3], forge.DeviceType.CPU)
        t.zero_()
        arr = t.numpy()
        assert np.all(arr == 0)

    def test_numpy_roundtrip(self):
        t = forge.Tensor(forge.DataType.FP32, [2, 5], forge.DeviceType.CPU)
        arr = t.numpy()
        assert arr.shape == (2, 5)
        assert arr.dtype == np.float32

    def test_nbytes(self):
        t = forge.Tensor(forge.DataType.FP32, [4, 8], forge.DeviceType.CPU)
        assert t.nbytes() == 4 * 8 * 4

    def test_1d_tensor(self):
        t = forge.Tensor(forge.DataType.FP32, [10], forge.DeviceType.CPU)
        assert t.shape() == [10]
        assert t.numel() == 10

    def test_scalar_like_tensor(self):
        t = forge.Tensor(forge.DataType.FP32, [1], forge.DeviceType.CPU)
        assert t.numel() == 1


class TestTensorDevice:
    def test_to_device_cpu_to_cpu(self):
        t = forge.Tensor(forge.DataType.FP32, [2, 3], forge.DeviceType.CPU)
        t.to_device(forge.DeviceType.CPU)
        assert t.device() == forge.DeviceType.CPU


# ---- Phase 1: Storage/Layout Tests ----

class TestTensorStorageLayout:
    """Verify TensorStorage and TensorLayout are properly initialized."""

    def test_owned_tensor_has_storage(self):
        """Owned tensors should have storage with base == data, capacity == nbytes."""
        t = forge.Tensor(forge.DataType.FP32, [10, 20], forge.DeviceType.CPU)
        assert t.nbytes() == 10 * 20 * 4
        assert t.allocation_bytes() == t.nbytes()

    def test_owned_tensor_byte_offset_zero(self):
        """Owned tensors should have byte_offset == 0."""
        t = forge.Tensor(forge.DataType.FP32, [5, 5], forge.DeviceType.CPU)
        assert t.byte_offset() == 0

    def test_from_buffer_has_storage(self):
        """from_buffer() should set storage fields."""
        data = np.zeros(100, dtype=np.float32)
        t = forge.Tensor.from_buffer(
            data.ctypes.data, forge.DataType.FP32, [100],
            forge.DeviceType.CPU, own=False
        )
        assert t.nbytes() == 400
        assert t.byte_offset() == 0
        assert t.allocation_bytes() == 400

    def test_strides_are_correct(self):
        """strides should be computed correctly."""
        t = forge.Tensor(forge.DataType.FP32, [3, 4, 5], forge.DeviceType.CPU)
        s = t.strides()
        assert s == [20, 5, 1]  # row-major


class TestTensorView:
    """Verify view() carries proper storage/layout metadata."""

    def test_view_shares_storage(self):
        """view() should share the same storage base."""
        t = forge.Tensor(forge.DataType.FP32, [4, 6], forge.DeviceType.CPU)
        v = t.view([2, 12])
        # Both should have same nbytes
        assert v.nbytes() == t.nbytes()
        assert v.byte_offset() == 0

    def test_view_reshape_preserves_numel(self):
        """view() should preserve total element count."""
        t = forge.Tensor(forge.DataType.FP32, [3, 4], forge.DeviceType.CPU)
        v = t.view([2, 6])
        assert v.numel() == 12
        assert v.shape() == [2, 6]

    def test_view_1d(self):
        """view() to 1D should work."""
        t = forge.Tensor(forge.DataType.FP32, [3, 4], forge.DeviceType.CPU)
        v = t.view([12])
        assert v.shape() == [12]
        assert v.numel() == 12

    def test_view_from_quantized(self):
        """view() on quantized tensor should preserve quant metadata."""
        t = forge.Tensor(forge.DataType.Q4_0, [32], forge.DeviceType.CPU)
        v = t.view([32])  # same shape
        assert v.nbytes() == 18
        assert v.byte_offset() == 0

    def test_view_allocation_bytes(self):
        """view() allocation_bytes should equal nbytes."""
        t = forge.Tensor(forge.DataType.FP32, [100], forge.DeviceType.CPU)
        v = t.view([10, 10])
        assert v.allocation_bytes() == v.nbytes()


class TestTensorSlice:
    """Verify slice() carries proper byte_offset and storage metadata."""

    def test_slice_fp32_byte_offset(self):
        """slice() on FP32 should have correct byte_offset."""
        t = forge.Tensor(forge.DataType.FP32, [10], forge.DeviceType.CPU)
        s = t.slice(0, 2, 8)  # 6 elements
        # byte_offset = 2 * 4 = 8
        assert s.byte_offset() == 8
        assert s.shape() == [6]
        assert s.nbytes() == 24

    def test_slice_q4_0_byte_offset(self):
        """slice() on Q4_0 should have block-aligned byte_offset."""
        t = forge.Tensor(forge.DataType.Q4_0, [128], forge.DeviceType.CPU)
        # 128 elements = 4 blocks = 72 bytes
        # slice [64:128] = 64 elements = 2 blocks = 36 bytes
        # byte_offset = 2 blocks * 18 = 36
        s = t.slice(0, 64, 128)
        assert s.shape() == [64]
        assert s.nbytes() == 36
        assert s.byte_offset() == 36

    def test_slice_q4_k_byte_offset(self):
        """slice() on Q4_K should have correct byte_offset."""
        t = forge.Tensor(forge.DataType.Q4_K, [512], forge.DeviceType.CPU)
        # 512 elements = 2 blocks = 288 bytes
        # slice [256:512] = 256 elements = 1 block = 144 bytes
        s = t.slice(0, 256, 512)
        assert s.shape() == [256]
        assert s.nbytes() == 144
        assert s.byte_offset() == 144

    def test_slice_2d_fp32(self):
        """slice() on 2D tensor."""
        t = forge.Tensor(forge.DataType.FP32, [4, 8], forge.DeviceType.CPU)
        s = t.slice(0, 1, 3)  # rows 1,2
        assert s.shape() == [2, 8]
        assert s.numel() == 16
        assert s.nbytes() == 64
        assert s.byte_offset() == 32  # 1 row * 8 cols * 4 bytes

    def test_slice_allocation_bytes(self):
        """slice() allocation_bytes should equal nbytes."""
        t = forge.Tensor(forge.DataType.FP32, [100], forge.DeviceType.CPU)
        s = t.slice(0, 10, 50)
        assert s.allocation_bytes() == s.nbytes()


class TestTensorNestedViewSlice:
    """Verify view-of-slice and slice-of-view carry correct metadata."""

    def test_view_of_slice(self):
        """view() on a slice should preserve byte_offset."""
        t = forge.Tensor(forge.DataType.FP32, [20], forge.DeviceType.CPU)
        s = t.slice(0, 5, 15)  # 10 elements from offset 20
        assert s.byte_offset() == 20
        v = s.view([2, 5])
        assert v.shape() == [2, 5]
        assert v.byte_offset() == 20  # same offset as parent slice

    def test_slice_of_view(self):
        """slice() on a view should accumulate byte_offset."""
        t = forge.Tensor(forge.DataType.FP32, [8, 8], forge.DeviceType.CPU)
        v = t.view([4, 16])  # same data, different shape
        assert v.byte_offset() == 0
        s = v.slice(0, 1, 3)  # rows 1,2 → 2*16*4 = 128 bytes offset
        assert s.shape() == [2, 16]
        assert s.byte_offset() == 64  # 1 row * 16 cols * 4 bytes


class TestTensorFromBuffer:
    """Verify from_buffer() storage semantics."""

    def test_from_buffer_fp32(self):
        """from_buffer() with FP32 data."""
        data = np.zeros(50, dtype=np.float32)
        t = forge.Tensor.from_buffer(
            data.ctypes.data, forge.DataType.FP32, [50],
            forge.DeviceType.CPU, own=False
        )
        assert t.shape() == [50]
        assert t.nbytes() == 200
        assert t.byte_offset() == 0
        assert t.allocation_bytes() == 200

    def test_from_buffer_q4_0(self):
        """from_buffer() with Q4_0 data."""
        # 32 elements = 18 bytes
        data = np.zeros(18, dtype=np.uint8)
        t = forge.Tensor.from_buffer(
            data.ctypes.data, forge.DataType.Q4_0, [32],
            forge.DeviceType.CPU, own=False
        )
        assert t.nbytes() == 18
        assert t.byte_offset() == 0
        assert t.allocation_bytes() == 18
