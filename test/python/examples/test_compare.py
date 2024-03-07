import torch
from torch.testing._internal.common_utils import TestCase


class TestTorchMethod(TestCase):
    def _test_compare_tensor_fn(self, fn, dtype):
        x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
        x2 = torch.tensor([[1.0, 1.0], [4.0, 4.0]], dtype=dtype)
        x1_xpu = x1.xpu()
        x2_xpu = x2.xpu()
        self.assertEqual((fn(x1_xpu, x2_xpu)).cpu(), fn(x1, x2))

    def test_eq(self, dtype=torch.float):
        self._test_compare_tensor_fn(torch.eq, dtype)
    
    def test_ne(self, dtype=torch.float):
        self._test_compare_tensor_fn(torch.ne, dtype)
    
    def test_lt(self, dtype=torch.float):
        self._test_compare_tensor_fn(torch.lt, dtype)
    
    def test_le(self, dtype=torch.float):
        self._test_compare_tensor_fn(torch.le, dtype)
    
    def test_gt(self, dtype=torch.float):
        self._test_compare_tensor_fn(torch.gt, dtype)
    
    def test_ge(self, dtype=torch.float):
        self._test_compare_tensor_fn(torch.ge, dtype)
