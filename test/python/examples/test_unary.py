import torch
from torch.testing._internal.common_utils import TestCase


floating_and_complex_types = [torch.float, torch.half, torch.bfloat16, torch.double, torch.cfloat, torch.double]
all_types = floating_and_complex_types + [torch.int8, torch.unit8, torch.short, torch.int, torch.long, torch.bool]


def for_each_floating_and_complex_types(fn):
    def fn_out(*args, **kwargs):
        for dtype in floating_and_complex_types:
            kwargs['dtype'] = dtype
            print(f"TestSimpleUnary:{fn__name__}:{dtype}", flush=True)
            fn(*args, **kwargs)
    return fn_out


class TestSimpleUnary(TestCase):
    def _test_unary_out_ops(self, fn_str, dtype):
        a_cpu = torch.randn(10001, dtype=dtype)
        a_xpu = a_cpu.xpu()
        b_cpu = eval(f"torch.{fn_str}(a_cpu)")
        b_xpu = eval(f"torch.{fn_str}(a_xpu)")
        c_cpu = eval(f"a_cpu.{fn_str}()")
        c_xpu = eval(f"a_xpu.{fn_str}()")
        self.assertEqual(b_cpu, b_xpu.cpu())
        self.assertEqual(b_cpu, b_xpu.cpu())
    
    @for_each_floating_and_complex_types
    def test_sin_out(self, dtype):
        print(dtype, flush=True)
        self._test_unary_out_ops('sin', dtype)
    
    @for_each_floating_and_complex_types
    def test_cos_out(self, dtype):
        self._test_unary_out_ops('cos', dtype)
    
    @for_each_floating_and_complex_types
    def test_abs_out(self, dtype):
        self._test_unary_out_ops('abs', dtype)
    
    @for_each_floating_and_complex_types
    def test_log_out(self, dtype):
        self._test_unary_out_ops('log', dtype)
    
    @for_each_floating_and_complex_types
    def test_sqrt_out(self, dtype):
        self._test_unary_out_ops('sqrt', dtype)

    @for_each_floating_and_complex_types
    def test_rsqrt_out(self, dtype):
        self._test_unary_out_ops('rsqrt', dtype)
    
    @for_each_floating_and_complex_types
    def test_tanh_out(self, dtype):
        self._test_unary_out_ops('tanh', dtype)
    
    @for_each_floating_and_complex_types
    def test_neg_out(self, dtype):
        self._test_unary_out_ops('neg', dtype)
