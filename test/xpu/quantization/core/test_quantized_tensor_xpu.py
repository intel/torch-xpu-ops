# Owner(s): ["module: intel"]

import torch

import unittest
import numpy as np

from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    import sys
    import os
    script_path = os.path.split(__file__)[0]
    sys.path.insert(0, os.path.realpath(os.path.join(script_path, "../..")))
    from xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_quantized_tensor import TestQuantizedTensor
    from torch.testing._internal.common_cuda import TEST_CUDA

    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA is not available')
    def _test_compare_per_channel_device_numerics(self):
        dtype_and_zero_types = [
            (torch.quint8, torch.float),
            (torch.qint8, torch.float),
            #  (torch.qint32, torch.float) not supported for quantize_per_channel
            (torch.quint8, torch.long),
            (torch.qint8, torch.long),
            (torch.qint32, torch.long),
        ]
        axis = 1
        device = torch.device('xpu')
        for i in range(20):
            for dtype, zero_type in dtype_and_zero_types:
                r = torch.rand(2, 2) * 10
                r[0, 0] = 2.5
                scales = torch.rand(2).abs()
                zero_points = (torch.rand(2) * 10).round().to(zero_type)

                qr = torch.quantize_per_channel(r, scales, zero_points, axis, dtype)
                dqr = qr.dequantize()
                qr_cuda = torch.quantize_per_channel(r.to(device), scales.to(
                    device), zero_points.to(device), axis, dtype)
                dqr_cuda = qr_cuda.dequantize()
                self.assertEqual(qr.int_repr(), qr_cuda.int_repr())
                self.assertTrue(np.allclose(dqr, dqr_cuda.cpu()))

    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA is not available')
    def _test_compare_per_tensor_device_numerics(self):
        dtypes = [
            torch.quint8,
            torch.qint8,
            torch.qint32,
        ]
        device = torch.device('xpu')
        for dtype in dtypes:
            r = torch.rand(2, 2) * 10
            r[0, 0] = 2.5
            scale = torch.rand(2).abs().max().item()
            zero_point = (torch.rand(2) * 10).round().to(torch.long).max().item()

            qtr = torch.quantize_per_tensor(r, scale, zero_point, dtype)
            dqtr = qtr.dequantize()
            qtr_cuda = torch.quantize_per_tensor(r.to(device), scale, zero_point, dtype)
            dqtr_cuda = qtr_cuda.dequantize()
            self.assertEqual(qtr.int_repr(), qtr_cuda.int_repr())
            self.assertTrue(np.allclose(dqtr, dqtr_cuda.cpu()))

    
    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA is not available')
    def _test_cuda_quantization_does_not_pin_memory(self):
        # Context - https://github.com/pytorch/pytorch/issues/41115
        x = torch.randn(3)
        self.assertEqual(x.is_pinned(), False)

        q_int = torch.randint(0, 100, [1, 2, 3], device="xpu", dtype=torch.uint8)
        q = torch._make_per_tensor_quantized_tensor(q_int, scale=0.1, zero_point=0)

        x = torch.randn(3)
        self.assertEqual(x.is_pinned(), False)

    
    @unittest.skipIf(not TEST_CUDA, "No gpu is available.")
    def _test_dequantize_fp16_cuda(self):
        self._test_dequantize_fp16(torch.device('xpu'))
    
    @unittest.skipIf(not TEST_CUDA, "No gpu is available.")
    def _test_per_channel_qtensor_creation_cuda(self):
        self._test_per_channel_qtensor_creation(torch.device('xpu'))

    
    @unittest.skipIf(not TEST_CUDA, "No gpu is available.")
    def _test_per_channel_to_device(self):
        dtype_and_zero_types = [
            (torch.quint8, torch.float),
            (torch.qint8, torch.float),
            #  (torch.qint32, torch.float) not supported for quantize_per_channel
            (torch.quint8, torch.long),
            (torch.qint8, torch.long),
            (torch.qint32, torch.long),
        ]
        axis = 1
        device = torch.device('xpu')
        for dtype, zero_type in dtype_and_zero_types:
            r = torch.rand(2, 2, dtype=torch.float) * 10
            scales = torch.rand(2).abs()
            zero_points = (torch.rand(2) * 10).round().to(zero_type)

            dqr = torch.quantize_per_channel(r, scales, zero_points, axis, dtype)
            dqr = dqr.to(device)
            dqr_cuda = torch.quantize_per_channel(r.to(device), scales.to(
                device), zero_points.to(device), axis, dtype)
            dqr_cuda = dqr_cuda.to('cpu')

            self.assertEqual('xpu', dqr.device.type)
            self.assertEqual('xpu', dqr.q_per_channel_scales().device.type)
            self.assertEqual('xpu', dqr.q_per_channel_zero_points().device.type)

            self.assertEqual('cpu', dqr_cuda.device.type)
            self.assertEqual('cpu', dqr_cuda.q_per_channel_scales().device.type)
            self.assertEqual('cpu', dqr_cuda.q_per_channel_zero_points().device.type)
    
    @unittest.skipIf(not TEST_CUDA, "No gpu is available.")
    def _test_per_tensor_to_device(self):
        dtypes = [
            torch.quint8,
            torch.qint8,
            torch.qint32,
        ]
        device = torch.device('xpu')
        for dtype in dtypes:
            r = torch.rand(2, 2, dtype=torch.float) * 10
            scale = torch.rand(2).abs().max().item()
            zero_point = (torch.rand(2) * 10).round().to(torch.long).max().item()

            qr = torch.quantize_per_tensor(r, scale, zero_point, dtype)
            qr = qr.to(device)
            qr_cuda = torch.quantize_per_tensor(r.to(device), scale, zero_point, dtype)
            qr_cuda = qr_cuda.to('cpu')
            self.assertEqual('xpu', qr.device.type)
            self.assertEqual('cpu', qr_cuda.device.type)
    
    @unittest.skipIf(not TEST_CUDA, "No gpu is available.")
    def _test_qtensor_cuda(self):
        self._test_qtensor(torch.device('xpu'))
        self._test_qtensor_dynamic(torch.device('xpu'))
    
    @unittest.skipIf(not TEST_CUDA, "No gpu is available.")
    def _test_qtensor_index_put_cuda(self):
        self._test_qtensor_index_put('xpu')
        self._test_qtensor_index_put_non_accumulate_deterministic('xpu')

    @unittest.skipIf(not TEST_CUDA, "No gpu is available.")
    def _test_qtensor_index_select_cuda(self):
        self._test_qtensor_index_select('xpu')

    @unittest.skipIf(not TEST_CUDA, "No gpu is available.")
    def _test_qtensor_masked_fill_cuda(self):
        self._test_qtensor_masked_fill('xpu')


TestQuantizedTensor.test_compare_per_channel_device_numerics = _test_compare_per_channel_device_numerics
TestQuantizedTensor.test_compare_per_tensor_device_numerics = _test_compare_per_tensor_device_numerics
TestQuantizedTensor.test_cuda_quantization_does_not_pin_memory = _test_cuda_quantization_does_not_pin_memory
TestQuantizedTensor.test_dequantize_fp16_cuda = _test_dequantize_fp16_cuda
TestQuantizedTensor.test_per_channel_qtensor_creation_cuda = _test_per_channel_qtensor_creation_cuda
TestQuantizedTensor.test_per_channel_to_device = _test_per_channel_to_device
TestQuantizedTensor.test_per_tensor_to_device = _test_per_tensor_to_device
TestQuantizedTensor.test_qtensor_cuda = _test_qtensor_cuda
TestQuantizedTensor.test_qtensor_index_put_cuda = _test_qtensor_index_put_cuda
TestQuantizedTensor.test_qtensor_index_select_cuda = _test_qtensor_index_select_cuda
TestQuantizedTensor.test_qtensor_masked_fill_cuda = _test_qtensor_masked_fill_cuda


instantiate_device_type_tests(TestQuantizedTensor, globals(), only_for="xpu", allow_xpu=True)


if __name__ == "__main__":
    TestCase._default_dtype_check_enabled = True
    run_tests()
