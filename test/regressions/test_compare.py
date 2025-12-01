# Owner(s): ["module: intel"]
import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_dtype import float8_types_and
from torch.testing._internal.common_utils import run_tests, TestCase


class TestTorchMethod(TestCase):
    def _test_compare_fn(self, fn, dtype):
        # test tensor
        x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
        x2 = torch.tensor([[1.0, 1.0], [4.0, 4.0]], dtype=dtype)
        x1_xpu = x1.xpu()
        x2_xpu = x2.xpu()
        y = fn(x1, x2)
        y_xpu = fn(x1_xpu, x2_xpu)
        self.assertEqual(y_xpu.cpu(), y)

        # Comparison ops output bool; must create new out tensor
        y_out = torch.empty_like(y_xpu, dtype=torch.bool)
        fn(x1_xpu, x2_xpu, out=y_out)
        self.assertEqual(y_out.cpu(), y)

        # test scalar
        x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
        x2 = 2.0
        x1_xpu = x1.xpu()
        x2_xpu = x2
        y = fn(x1, x2)
        y_xpu = fn(x1_xpu, x2_xpu)
        self.assertEqual(y_xpu.cpu(), y)

        y_out = torch.empty_like(y_xpu, dtype=torch.bool)
        fn(x1_xpu, x2_xpu, out=y_out)
        self.assertEqual(y_out.cpu(), y)

    def test_eq(self, dtype=torch.float):
        self._test_compare_fn(torch.eq, dtype)

    def test_ne(self, dtype=torch.float):
        self._test_compare_fn(torch.ne, dtype)

    def test_lt(self, dtype=torch.float):
        self._test_compare_fn(torch.lt, dtype)

    def test_le(self, dtype=torch.float):
        self._test_compare_fn(torch.le, dtype)

    def test_gt(self, dtype=torch.float):
        self._test_compare_fn(torch.gt, dtype)

    def test_ge(self, dtype=torch.float):
        self._test_compare_fn(torch.ge, dtype)

    def _test_compare_float8_core(self, fn, dtype):
        """
        Core logic for float8 comparison. Uses rounded float32 values
        as the Golden Reference to account for lossy f32 -> f8 conversion.
        """
        # 1. Original high-precision inputs (f32)
        x1_f32 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        x2_f32 = torch.tensor([[1.0, 1.0], [4.0, 4.0]], dtype=torch.float32)

        # 2. Compute Rounded Reference Inputs: f32 -> f8 -> f32 to capture rounding
        x1_rounded = x1_f32.to(dtype).to(torch.float32)
        x2_rounded = x2_f32.to(dtype).to(torch.float32)

        # CPU Reference Result (using rounded f32 values)
        y_ref = fn(x1_rounded, x2_rounded)

        # --- 1. Tensor Test (Tensor vs Tensor) ---
        # Convert to target float8 dtype for XPU test
        x1 = x1_f32.to(dtype)
        x2 = x2_f32.to(dtype)

        # XPU operation
        x1_xpu = x1.xpu()
        x2_xpu = x2.xpu()
        y_xpu = fn(x1_xpu, x2_xpu)

        # Compare XPU float8 result against CPU rounded reference
        self.assertEqual(y_xpu.cpu(), y_ref)

        # Test out= argument
        y_out = torch.empty_like(y_xpu, dtype=torch.bool)
        fn(x1_xpu, x2_xpu, out=y_out)
        self.assertEqual(y_out.cpu(), y_ref)

        # --- 2. Scalar Test (Tensor vs Scalar) ---
        x2_scalar = 2.0

        # CPU Reference Result (using rounded f32 tensor vs f32 scalar)
        y_ref_scalar = fn(x1_rounded, x2_scalar)

        # Convert to target float8 dtype
        x1 = x1_f32.to(dtype)

        # XPU operation
        x1_xpu = x1.xpu()
        y_xpu_scalar = fn(x1_xpu, x2_scalar)

        self.assertEqual(y_xpu_scalar.cpu(), y_ref_scalar)

        # Test out= argument
        y_out_scalar = torch.empty_like(y_xpu_scalar, dtype=torch.bool)
        fn(x1_xpu, x2_scalar, out=y_out_scalar)
        self.assertEqual(y_out_scalar.cpu(), y_ref_scalar)

    FLOAT8_DTYPES = float8_types_and(torch.float8_e8m0fnu)

    @dtypes(*FLOAT8_DTYPES)
    def test_eq_float8(self, dtype):
        """Test torch.eq correctness across float8 dtypes using rounded reference."""
        self._test_compare_float8_core(torch.eq, dtype)

    @dtypes(*FLOAT8_DTYPES)
    def test_ne_float8(self, dtype):
        """Test torch.ne correctness across float8 dtypes using rounded reference."""
        self._test_compare_float8_core(torch.ne, dtype)

    @dtypes(*FLOAT8_DTYPES)
    def test_lt_float8(self, dtype):
        """Test torch.lt correctness across float8 dtypes using rounded reference."""
        self._test_compare_float8_core(torch.lt, dtype)

    @dtypes(*FLOAT8_DTYPES)
    def test_le_float8(self, dtype):
        """Test torch.le correctness across float8 dtypes using rounded reference."""
        self._test_compare_float8_core(torch.le, dtype)

    @dtypes(*FLOAT8_DTYPES)
    def test_gt_float8(self, dtype):
        """Test torch.gt correctness across float8 dtypes using rounded reference."""
        self._test_compare_float8_core(torch.gt, dtype)

    @dtypes(*FLOAT8_DTYPES)
    def test_ge_float8(self, dtype):
        """Test torch.ge correctness across float8 dtypes using rounded reference."""
        self._test_compare_float8_core(torch.ge, dtype)


instantiate_device_type_tests(
    TestTorchMethod, globals(), only_for="xpu", allow_xpu=True
)


if __name__ == "__main__":
    run_tests()
