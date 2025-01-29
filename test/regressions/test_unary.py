# Owner(s): ["module: intel"]
import torch
from torch.testing._internal.common_utils import TestCase

floating_types = [torch.float, torch.half, torch.bfloat16, torch.double]
integral_types = [
    torch.int8,
    torch.uint8,
    torch.short,
    torch.int,
    torch.long,
    torch.bool,
]
complex_types = [torch.cfloat, torch.cdouble]
floating_and_complex_types = floating_types + complex_types
all_basic_types = floating_types + integral_types
all_basic_and_complex_types = floating_types + integral_types + complex_types


class Dtypes:  # noqa: UP004
    def __init__(self, include_dtypes, exclude_dtypes=[]):  # noqa: B006
        self.include_dtypes = include_dtypes
        self.exclude_dtypes = exclude_dtypes

    def __call__(self, fn):
        def fn_out(*args, **kwargs):
            for dtype in self.include_dtypes:
                if dtype in self.exclude_dtypes:
                    continue
                kwargs["dtype"] = dtype
                fn(*args, **kwargs)

        return fn_out


class TestSimpleUnary(TestCase):
    def _test_unary_out_ops(self, fn_str, dtype):
        a_cpu = (torch.randn(2049) * 10).to(dtype)
        a_xpu = a_cpu.xpu()
        b_cpu = eval(f"torch.{fn_str}(a_cpu)")
        b_xpu = eval(f"torch.{fn_str}(a_xpu)")
        c_cpu = eval(f"a_cpu.{fn_str}()")
        c_xpu = eval(f"a_xpu.{fn_str}()")
        self.assertEqual(b_cpu, b_xpu.cpu(), atol=1e-4, rtol=1e-4)
        self.assertEqual(c_cpu, c_xpu.cpu(), atol=1e-4, rtol=1e-4)
        d_cpu = eval(f"torch.{fn_str}(a_cpu, out=c_cpu)")
        d_xpu = eval(f"torch.{fn_str}(a_xpu, out=c_xpu)")
        self.assertEqual(c_cpu, c_xpu.cpu(), atol=1e-4, rtol=1e-4)

    @Dtypes(floating_types)
    def test_abs_out(self, dtype):
        self._test_unary_out_ops("abs", dtype)

    @Dtypes(floating_and_complex_types)
    def test_sin_out(self, dtype):
        self._test_unary_out_ops("sin", dtype)

    @Dtypes(floating_and_complex_types)
    def test_cos_out(self, dtype):
        self._test_unary_out_ops("cos", dtype)

    @Dtypes(floating_and_complex_types)
    def test_log_out(self, dtype):
        self._test_unary_out_ops("log", dtype)

    @Dtypes(floating_and_complex_types)
    def test_sqrt_out(self, dtype):
        self._test_unary_out_ops("sqrt", dtype)

    @Dtypes(floating_and_complex_types)
    def test_rsqrt_out(self, dtype):
        self._test_unary_out_ops("rsqrt", dtype)

    @Dtypes(floating_and_complex_types)
    def test_tanh_out(self, dtype):
        self._test_unary_out_ops("tanh", dtype)

    @Dtypes(all_basic_and_complex_types, [torch.bool])
    def test_neg_out(self, dtype):
        self._test_unary_out_ops("neg", dtype)

    @Dtypes(floating_and_complex_types)
    def test_reciprocal_out(self, dtype):
        self._test_unary_out_ops("reciprocal", dtype)
