# Owner(s): ["module: intel"]
import torch
from torch.testing._internal.common_utils import TestCase


class TestTorchWhereMethod(TestCase):
    # Define float8 dtypes
    FLOAT8_DTYPES = (
        torch.float8_e5m2,
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2fnuz,
        torch.float8_e8m0fnu,
    )

    # Define the set of all dtypes to be tested
    TEST_DTYPES = (
        torch.float32,
        torch.float64,
        torch.half,
        torch.bfloat16,
    ) + FLOAT8_DTYPES

    def _test_where_fn(self, dtype):
        """Core function to test torch.where(condition, x, y) correctness."""

        # 1. Input Tensors (x and y)
        x = torch.tensor([[10.0, 20.0], [30.0, 40.0]], dtype=dtype)
        y = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]], dtype=dtype)
        # Condition must be bool
        condition = torch.tensor([[True, False], [False, True]], dtype=torch.bool)

        # --- 1. CPU Reference Calculation and Tolerance Setting ---

        if dtype in self.FLOAT8_DTYPES:
            # FP8: Use float32 as reference type for comparison
            x_ref = x.cpu().to(torch.float32)
            y_ref = y.cpu().to(torch.float32)
            rtol = 1e-2
            atol = 1e-2
        else:
            # Non-FP8: Use original dtype as reference type
            x_ref = x.cpu()
            y_ref = y.cpu()
            rtol = 1e-5
            atol = 1e-5

        condition_ref = condition.cpu()
        res_ref = torch.where(condition_ref, x_ref, y_ref)

        # --- 2. XPU Operation (Default) ---
        x_xpu = x.xpu()
        y_xpu = y.xpu()
        condition_xpu = condition.xpu()

        res_xpu = torch.where(condition_xpu, x_xpu, y_xpu)

        # Prepare XPU result for comparison (must match res_ref dtype)
        if dtype in self.FLOAT8_DTYPES:
            # FP8: Convert XPU result to float32
            res_xpu_to_compare = res_xpu.cpu().to(torch.float32)
        else:
            # Non-FP8: Pull to CPU, keeping original dtype
            res_xpu_to_compare = res_xpu.cpu()

        # Compare: res_ref vs res_xpu_to_compare
        self.assertEqual(res_ref, res_xpu_to_compare, rtol=rtol, atol=atol)

        # --- 3. Test the version with out= argument ---

        # Create output tensor on XPU
        res_xpu_out = torch.empty_like(res_xpu, dtype=dtype).xpu()
        torch.where(condition_xpu, x_xpu, y_xpu, out=res_xpu_out)

        # Prepare XPU 'out' result for comparison
        if dtype in self.FLOAT8_DTYPES:
            # FP8: Convert XPU result to float32
            res_xpu_out_to_compare = res_xpu_out.cpu().to(torch.float32)
        else:
            # Non-FP8: Pull to CPU, keeping original dtype
            res_xpu_out_to_compare = res_xpu_out.cpu()

        # Compare: res_ref vs res_xpu_out_to_compare
        self.assertEqual(res_ref, res_xpu_out_to_compare, rtol=rtol, atol=atol)

    def test_where(self):
        """Test torch.where() correctness across all supported dtypes, including float8."""
        for dtype in self.TEST_DTYPES:
            # Use string conversion for better subTest reporting
            dtype_name = str(dtype).split(".")[-1]
            with self.subTest(dtype=dtype_name):
                self._test_where_fn(dtype)
