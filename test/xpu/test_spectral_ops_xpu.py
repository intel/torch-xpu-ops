# Owner(s): ["module: intel"]

import torch
import numpy as np
from packaging import version
from itertools import product

from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests, ops, onlyNativeDeviceTypes)
from torch.testing._internal.common_methods_invocations import (
    spectral_funcs, SpectralFuncType)
from torch.testing._internal.common_utils import run_tests

try:
    from .xpu_test_utils import XPUPatchForImport
except Exception as e:
    from ..xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_spectral_ops import TestFFT

has_scipy_fft = False
try:
    import scipy.fft
    has_scipy_fft = True
except ModuleNotFoundError:
    pass

REFERENCE_NORM_MODES = (
    (None, "forward", "backward", "ortho")
    if version.parse(np.__version__) >= version.parse('1.20.0') and (
        not has_scipy_fft or version.parse(scipy.__version__) >= version.parse('1.6.0'))
    else (None, "ortho"))

@ops([op for op in spectral_funcs if op.ndimensional == SpectralFuncType.OneD],
     allowed_dtypes=(torch.float, torch.cfloat))
def _test_reference_1d(self, device, dtype, op):
    if op.ref is None:
        raise unittest.SkipTest("No reference implementation")

    norm_modes = REFERENCE_NORM_MODES
    test_args = [
        *product(
            # input
            (torch.randn(67, device=device, dtype=dtype),
             torch.randn(80, device=device, dtype=dtype),
             torch.randn(12, 14, device=device, dtype=dtype),
             torch.randn(9, 6, 3, device=device, dtype=dtype)),
            # n
            (None, 50, 6),
            # dim
            (-1, 0),
            # norm
            norm_modes
        ),
        # Test transforming middle dimensions of multi-dim tensor
        *product(
            (torch.randn(4, 5, 6, 7, device=device, dtype=dtype),),
            (None,),
            (1, 2, -2,),
            norm_modes
        )
    ]

    for iargs in test_args:
        args = list(iargs)
        input = args[0]
        args = args[1:]

        expected = op.ref(input.cpu().numpy(), *args)
        exact_dtype = dtype in (torch.double, torch.complex128)
        actual = op(input, *args)
        self.assertEqual(actual, expected, exact_dtype=exact_dtype, atol=1e-4, rtol=1e-5)

TestFFT.test_reference_1d = _test_reference_1d

instantiate_device_type_tests(TestFFT, globals(), only_for=("xpu"), allow_xpu=True)


if __name__ == "__main__":
    run_tests()
