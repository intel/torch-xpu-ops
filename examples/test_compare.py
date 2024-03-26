import torch
from torch.testing._internal.common_utils import TestCase


class TestTorchMethod(TestCase):
    def _test_binary_compare_fn(self, fn, dtype):
        # test tensor
        x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
        x2 = torch.tensor([[1.0, 1.0], [4.0, 4.0]], dtype=dtype)
        x1_xpu = x1.xpu()
        x2_xpu = x2.xpu()
        y = fn(x1, x2)
        y_xpu = fn(x1_xpu, x2_xpu)
        self.assertEqual(y_xpu.cpu(), y)
        y_xpu.zero_()
        fn(x1_xpu, x2_xpu, out=y_xpu)
        self.assertEqual(y_xpu.cpu(), y)
        # test scalar
        x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
        x2 = 2.0
        x1_xpu = x1.xpu()
        x2_xpu = x2
        y = fn(x1, x2)
        y_xpu = fn(x1_xpu, x2_xpu)
        self.assertEqual(y_xpu.cpu(), y)
        y_xpu.zero_()
        fn(x1_xpu, x2_xpu, out=y_xpu)
        self.assertEqual(y_xpu.cpu(), y)
    
    def test_eq(self, dtype=torch.float):
        self._test_binary_compare_fn(torch.eq, dtype)
    
    def test_ne(self, dtype=torch.float):
        self._test_binary_compare_fn(torch.ne, dtype)
    
    def test_lt(self, dtype=torch.float):
        self._test_binary_compare_fn(torch.lt, dtype)
    
    def test_le(self, dtype=torch.float):
        self._test_binary_compare_fn(torch.le, dtype)
    
    def test_gt(self, dtype=torch.float):
        self._test_binary_compare_fn(torch.gt, dtype)
    
    def test_ge(self, dtype=torch.float):
        self._test_binary_compare_fn(torch.ge, dtype)
    
    def test_clamp(self, dtype=torch.float):
        a = torch.randn(2049, dtype=dtype)
        a_xpu = a.xpu()
        a.clamp_(min=-2.0, max=2.0)
        a_xpu.clamp_(min=-2.0, max=2.0)
        self.assertEqual(a_xpu.cpu(), a)
        a = torch.randn(2049, dtype=dtype)
        a_xpu = a.xpu()
        a.clamp_(min=-2.0)
        a_xpu.clamp_(min=-2.0)
        self.assertEqual(a_xpu.cpu(), a)
        a = torch.randn(2049, dtype=dtype)
        a_xpu = a.xpu()
        a.clamp_(max=2.0)
        a_xpu.clamp_(max=2.0)
        self.assertEqual(a_xpu.cpu(), a)
        a = torch.randn(2049, dtype=dtype)
        a_xpu = a.xpu()
        a.clamp_max_(max=2.0)
        a_xpu.clamp_max_(max=2.0)
        self.assertEqual(a_xpu.cpu(), a)
        a = torch.randn(2049, dtype=dtype)
        a_xpu = a.xpu()
        a.clamp_min_(min=2.0)
        a_xpu.clamp_min_(min=2.0)
        self.assertEqual(a_xpu.cpu(), a)
    
    def test_bitwise_and(self, dtype=torch.short):
        a = torch.randint(0, 10000, (2049,), dtype=dtype)
        b = torch.randint(0, 10000, (2049,), dtype=dtype)
        a_xpu = a.xpu()
        b_xpu = b.xpu()
        c = torch.bitwise_and(a, b)
        c_xpu = torch.bitwise_and(a_xpu, b_xpu)
        self.assertEqual(c_xpu.cpu(), c)
    
    def test_bitwise_or(self, dtype=torch.short):
        a = torch.randint(0, 10000, (2049,), dtype=dtype)
        b = torch.randint(0, 10000, (2049,), dtype=dtype)
        a_xpu = a.xpu()
        b_xpu = b.xpu()
        c = torch.bitwise_or(a, b)
        c_xpu = torch.bitwise_or(a_xpu, b_xpu)
        self.assertEqual(c_xpu.cpu(), c)
    
    def test_bitwise_xor(self, dtype=torch.short):
        a = torch.randint(0, 10000, (2049,), dtype=dtype)
        b = torch.randint(0, 10000, (2049,), dtype=dtype)
        a_xpu = a.xpu()
        b_xpu = b.xpu()
        c = torch.bitwise_xor(a, b)
        c_xpu = torch.bitwise_xor(a_xpu, b_xpu)
        self.assertEqual(c_xpu.cpu(), c)

    def test_bitwise_nor(self, dtype=torch.short):
        a = torch.randint(0, 100, (2049,), dtype=dtype)
        a_xpu = a.xpu()
        c = torch.bitwise_not(a)
        c_xpu = torch.bitwise_not(a_xpu)
        self.assertEqual(c_xpu.cpu(), c)
    
    def test_isnan(self, dtype=torch.float):
        a = torch.randint(0, 10, (10,), dtype=dtype)
        a[5] = float('nan')
        a[8] = float('nan')
        a_xpu = a.xpu()
        c = torch.isnan(a)
        c_xpu = torch.isnan(a_xpu)
        self.assertEqual(c_xpu.cpu(), c)
