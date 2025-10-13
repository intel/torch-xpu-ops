# Owner(s): ["module: intel"]
import torch
from torch.testing._internal.common_utils import TestCase

class TestTorchMethod(TestCase):
    # Define float8 dtypes
    FLOAT8_DTYPES = (
        torch.float8_e5m2, 
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2fnuz,
        torch.float8_e8m0fnu
    )
    
    # Define the set of all dtypes to be tested
    TEST_DTYPES = (
        torch.float32, 
        torch.float64, 
        torch.half, 
        torch.bfloat16,
        torch.bool, 
    ) + FLOAT8_DTYPES

    def _test_compare_fn(self, fn, dtype):
        # --- Tensor Test ---
        x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
        x2 = torch.tensor([[1.0, 1.0], [4.0, 4.0]], dtype=dtype)
        
        # Handle boolean input
        if dtype == torch.bool:
             x1 = x1.bool()
             x2 = x2.bool()

        # Determine the golden reference tensor on CPU
        if dtype in self.FLOAT8_DTYPES:
            # For float8, use float32 as the CPU reference type
            x1_ref = x1.cpu().to(torch.float32)
            x2_ref = x2.cpu().to(torch.float32)
        else:
            # For other types, use the original dtype
            x1_ref = x1.cpu()
            x2_ref = x2.cpu()
            
        y_ref = fn(x1_ref, x2_ref)

        # XPU operation
        x1_xpu = x1.xpu()
        x2_xpu = x2.xpu()
        y_xpu = fn(x1_xpu, x2_xpu)
        
        # Compare XPU result and CPU golden reference (comparison ops yield exact boolean values)
        self.assertEqual(y_xpu.cpu(), y_ref) 
        
        # Test the version with out= argument
        # For comparison ops, the output is bool, which doesn't support zero_().
        # We must create a new out tensor.
        if y_xpu.dtype != torch.bool:
             y_xpu.zero_()
        else:
             y_xpu = torch.empty_like(y_xpu, dtype=torch.bool) 
             
        fn(x1_xpu, x2_xpu, out=y_xpu)
        self.assertEqual(y_xpu.cpu(), y_ref)

        # --- 2. Scalar Test ---
        x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
        x2 = 2.0
        
        if dtype == torch.bool:
             x1 = x1.bool()

        # Determine the golden reference tensor on CPU
        if dtype in self.FLOAT8_DTYPES:
            x1_ref = x1.cpu().to(torch.float32)
        else:
            x1_ref = x1.cpu()
            
        x2_ref = x2 # Scalar remains the same
        y_ref = fn(x1_ref, x2_ref)

        # XPU operation
        x1_xpu = x1.xpu()
        x2_xpu = x2
        y_xpu = fn(x1_xpu, x2_xpu)
        
        self.assertEqual(y_xpu.cpu(), y_ref)
        
        # Test the version with out= argument
        if y_xpu.dtype != torch.bool:
             y_xpu.zero_()
        else:
             y_xpu = torch.empty_like(y_xpu, dtype=torch.bool) 

        fn(x1_xpu, x2_xpu, out=y_xpu)
        self.assertEqual(y_xpu.cpu(), y_ref)

    # --- Test methods iterating over DTypes ---

    def test_eq(self):
        for dtype in self.TEST_DTYPES:
            with self.subTest(dtype=dtype):
                self._test_compare_fn(torch.eq, dtype)

    def test_ne(self):
        for dtype in self.TEST_DTYPES:
            with self.subTest(dtype=dtype):
                self._test_compare_fn(torch.ne, dtype)

    def test_lt(self):
        for dtype in self.TEST_DTYPES:
            with self.subTest(dtype=dtype):
                self._test_compare_fn(torch.lt, dtype)

    def test_le(self):
        for dtype in self.TEST_DTYPES:
            with self.subTest(dtype=dtype):
                self._test_compare_fn(torch.le, dtype)

    def test_gt(self):
        for dtype in self.TEST_DTYPES:
            with self.subTest(dtype=dtype):
                self._test_compare_fn(torch.gt, dtype)

    def test_ge(self):
        for dtype in self.TEST_DTYPES:
            with self.subTest(dtype=dtype):
                self._test_compare_fn(torch.ge, dtype)
