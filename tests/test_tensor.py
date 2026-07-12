import numpy as np
import pytest
import nanoinfer


class TestTensorCreation:
    def test_create_fp32_cpu(self):
        t = nanoinfer.Tensor(nanoinfer.DataType.FP32, [3, 4], nanoinfer.DeviceType.CPU)
        assert t.shape() == [3, 4]
        assert t.numel() == 12
        assert t.dtype() == nanoinfer.DataType.FP32
        assert t.device() == nanoinfer.DeviceType.CPU

    def test_zero(self):
        t = nanoinfer.Tensor(nanoinfer.DataType.FP32, [2, 3], nanoinfer.DeviceType.CPU)
        t.zero_()
        arr = t.numpy()
        assert np.all(arr == 0)

    def test_numpy_roundtrip(self):
        t = nanoinfer.Tensor(nanoinfer.DataType.FP32, [2, 5], nanoinfer.DeviceType.CPU)
        arr = t.numpy()
        assert arr.shape == (2, 5)
        assert arr.dtype == np.float32

    def test_nbytes(self):
        t = nanoinfer.Tensor(nanoinfer.DataType.FP32, [4, 8], nanoinfer.DeviceType.CPU)
        assert t.nbytes() == 4 * 8 * 4

    def test_1d_tensor(self):
        t = nanoinfer.Tensor(nanoinfer.DataType.FP32, [10], nanoinfer.DeviceType.CPU)
        assert t.shape() == [10]
        assert t.numel() == 10

    def test_scalar_like_tensor(self):
        t = nanoinfer.Tensor(nanoinfer.DataType.FP32, [1], nanoinfer.DeviceType.CPU)
        assert t.numel() == 1


class TestTensorDevice:
    def test_to_device_cpu_to_cpu(self):
        t = nanoinfer.Tensor(nanoinfer.DataType.FP32, [2, 3], nanoinfer.DeviceType.CPU)
        t.to_device(nanoinfer.DeviceType.CPU)
        assert t.device() == nanoinfer.DeviceType.CPU
