# Owner(s): ["module: intel"]

import numpy as np
import torch
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.common_dtype import all_types_and

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_binary_ufuncs import TestBinaryUfuncs

    def _test_pow(self, base, exponent, np_exponent=None):
        if np_exponent is None:
            np_exponent = exponent

        def to_np(value):
            if isinstance(value, torch.Tensor):
                return value.cpu().numpy()
            return value

        try:
            np_res = np.power(to_np(base), to_np(np_exponent))
            expected = (
                torch.from_numpy(np_res)
                if isinstance(np_res, np.ndarray)
                else torch.tensor(np_res, dtype=base.dtype)
            )
        except ValueError as e:
            err_msg = "Integers to negative integer powers are not allowed."
            self.assertEqual(str(e), err_msg)
            out = torch.empty_like(base)
            test_cases = [
                lambda: base.pow(exponent),
                lambda: base.pow_(exponent),
                lambda: torch.pow(base, exponent),
                lambda: torch.pow(base, exponent, out=out),
            ]
            for test_case in test_cases:
                self.assertRaisesRegex(RuntimeError, err_msg, test_case)
        else:
            if isinstance(base, torch.Tensor):
                actual = base.pow(exponent)
                self.assertEqual(actual, expected.to(actual))
                actual = base.clone()
                # When base is a 0-dim cpu tensor and exp is a cuda tensor, we exp `pow` to work but `pow_` to fail, since
                # `pow` will try to create the output tensor on a cuda device, but `pow_` needs to use the cpu tensor as the output
                if (
                    isinstance(exponent, torch.Tensor)
                    and base.dim() == 0
                    and base.device.type == "cpu"
                    and exponent.device.type == "xpu"
                ):
                    regex = "Expected all tensors to be on the same device, but found at least two devices, xpu.* and cpu!"
                    self.assertRaisesRegex(RuntimeError, regex, base.pow_, exponent)
                elif torch.can_cast(torch.result_type(base, exponent), base.dtype):
                    actual2 = actual.pow_(exponent)
                    self.assertEqual(actual, expected)
                    self.assertEqual(actual2, expected)
                else:
                    self.assertRaisesRegex(
                        RuntimeError,
                        r"result type \w+ can't be cast to the desired output type \w+",
                        lambda: actual.pow_(exponent),
                    )

            actual = torch.pow(base, exponent)
            self.assertEqual(actual, expected.to(actual))

            actual2 = torch.pow(base, exponent, out=actual)
            self.assertEqual(actual, expected.to(actual))
            self.assertEqual(actual2, expected.to(actual))

    def cpu_tensor_pow_xpu_scalar_tensor(self, device):
        xpu_tensors = [
            torch.tensor(5.0, device=device),
            torch.tensor(-3, device=device),
        ]
        for exp in xpu_tensors:
            base = torch.randn((3, 3), device="cpu")
            regex = "Expected all tensors to be on the same device, but found at least two devices, xpu.* and cpu!"
            self.assertRaisesRegex(RuntimeError, regex, torch.pow, base, exp)
        for exp in xpu_tensors:
            # Binary ops with a cpu + cuda tensor are allowed if the cpu tensor has 0 dimension
            base = torch.tensor(3.0, device="cpu")
            self._test_pow(base, exp)

    @dtypes(*all_types_and(torch.half, torch.bfloat16))
    def div_rounding_modes(self, device, dtype):
        if dtype.is_floating_point:
            low, high = -10.0, 10.0
        else:
            info = torch.iinfo(dtype)
            low, high = info.min, info.max

        a = make_tensor((100,), dtype=dtype, device=device, low=low, high=high)
        b = make_tensor((100,), dtype=dtype, device=device, low=low, high=high)

        # Avoid division by zero so we can test (a / b) * b == a
        if dtype.is_floating_point:
            eps = 0.1
            b[(-eps < b) & (b < eps)] = eps
        else:
            b[b == 0] = 1

        if not dtype.is_floating_point:
            # floor(a / b) * b can be < a, so fixup slightly to avoid underflow
            a = torch.where(a < 0, a + b, a)

        d_true = torch.divide(a, b, rounding_mode=None)
        self.assertTrue(d_true.is_floating_point())
        self.assertEqual(d_true * b, a.to(d_true.dtype))

        d_floor = torch.divide(a, b, rounding_mode="floor")
        if dtype not in (torch.bfloat16, torch.half):
            self.assertEqual(d_floor * b + torch.remainder(a, b), a)
        else:
            self.assertEqual(
                d_floor * b + torch.remainder(a.float(), b.float()),
                a,
                exact_dtype=False,
            )

        d_trunc = torch.divide(a, b, rounding_mode="trunc")
        rounding_unsupported = (
            dtype == torch.half
            and device != "xpu"
            or dtype == torch.bfloat16
            and device != "cpu"
        )
        d_ref = d_true.float() if rounding_unsupported else d_true
        self.assertEqual(d_trunc, d_ref.trunc().to(dtype))

    TestBinaryUfuncs._test_pow = _test_pow
    TestBinaryUfuncs.test_cpu_tensor_pow_cuda_scalar_tensor = cpu_tensor_pow_xpu_scalar_tensor
    TestBinaryUfuncs.test_div_rounding_modes = div_rounding_modes

instantiate_device_type_tests(TestBinaryUfuncs, globals(), only_for="xpu", allow_xpu=True)


if __name__ == "__main__":
    run_tests()
