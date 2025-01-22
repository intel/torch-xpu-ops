# Owner(s): ["module: intel"]
import torch
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyNativeDeviceTypes,
)
from torch.testing._internal.common_utils import run_tests

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    import itertools
    import operator

    from test_type_promotion import float_double_default_dtype, TestTypePromotion
    from torch.testing._internal.common_dtype import get_all_dtypes

    @float_double_default_dtype
    def _test_mixed_type_backward(self, device):
        f = torch.ones([3, 3], dtype=torch.float, requires_grad=True, device=device)
        ten = torch.tensor([10.0], dtype=torch.double, device=device)
        tens = f * ten
        s = (tens + 2).sum()
        s.backward()
        expected = f.grad.to(torch.double)
        self.assertEqual(tens, expected)

        # If we don't convert the returned grad_input to the actual input type
        # we get an error like:
        # RuntimeError: Function SubBackward0 returned an invalid gradient at index 0 - expected type \
        # torch.FloatTensor but got torch.DoubleTensor
        f_dtypes = [torch.float, torch.double]
        if self.device_type == "cuda" or self.device_type == "xpu":
            f_dtypes = f_dtypes + [torch.half]
        i_dtypes = [torch.int, torch.long]
        for func in [torch.add, torch.sub, torch.rsub, torch.mul, torch.div]:
            for dtype1, dtype2 in itertools.product(f_dtypes, f_dtypes + i_dtypes):
                x = torch.ones(10, requires_grad=True, dtype=dtype1, device=device)
                y = torch.ones(10, dtype=dtype2, device=device)
                func(x, y).sum().backward()

    # XLA tests fail for self.assertRaises for complex dtypes
    @onlyNativeDeviceTypes
    def _test_complex_assertraises(self, device):
        comparison_ops = [
            dict(
                name="lt",
                compare_op=operator.lt,
            ),
            dict(
                name="le",
                compare_op=operator.le,
            ),
            dict(
                name="gt",
                compare_op=operator.gt,
            ),
            dict(
                name="ge",
                compare_op=operator.ge,
            ),
            dict(
                name="eq",
                compare_op=operator.eq,
            ),
            dict(
                name="ne",
                compare_op=operator.ne,
            ),
        ]
        for op in comparison_ops:
            is_cuda = (
                torch.device(device).type == "cuda"
                or torch.device(device).type == "xpu"
            )
            dtypes = get_all_dtypes(
                include_half=is_cuda,
                include_bfloat16=False,
                include_bool=False,
                include_complex32=True,
            )

            for dt1, dt2 in itertools.product(dtypes, dtypes):
                if (dt1.is_complex or dt2.is_complex) and not (
                    op["name"] == "eq" or op["name"] == "ne"
                ):
                    u = torch.tensor([1], dtype=dt1, device=device)
                    v = torch.tensor([2], dtype=dt2, device=device)
                    self.assertRaises(
                        RuntimeError,
                        lambda: torch.tensor(
                            [op["compare_op"](u, v)], dtype=torch.bool
                        ),
                    )

    TestTypePromotion.test_complex_assertraises = _test_complex_assertraises
    TestTypePromotion.test_mixed_type_backward = _test_mixed_type_backward

instantiate_device_type_tests(
    TestTypePromotion, globals(), only_for=("xpu"), allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
