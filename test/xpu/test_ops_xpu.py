# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]


import contextlib
import unittest

import torch
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
    skip,
)
from torch.testing._internal.common_methods_invocations import python_ref_db
from torch.testing._internal.common_utils import run_tests, skipIfTorchInductor


# NOTE: only needed in this wrapper - in upstream use the original function
def skipOps(to_skip):
    def wrapped(fn):
        from torch.testing._internal.opinfo.core import DecorateInfo

        parts = fn.__qualname__.split(".")
        test_name = parts[-1].lstrip("_")
        overrides = getattr(fn, "_op_overrides", {})
        for skip_spec in to_skip:
            if hasattr(skip_spec, "op_name"):
                op_name = skip_spec.op_name
                variant_name = skip_spec.variant_name
                device_type = skip_spec.device_type
                dtypes = skip_spec.dtypes
                if hasattr(skip_spec, "decorator"):
                    decorator_callable = skip_spec.decorator
                else:
                    expected_failure = skip_spec.expected_failure
                    decorator_callable = (
                        unittest.expectedFailure
                        if expected_failure
                        else unittest.skip("Skipped!")
                    )
            else:
                op_name, variant_name, device_type, dtypes, expected_failure = skip_spec
                decorator_callable = (
                    unittest.expectedFailure
                    if expected_failure
                    else unittest.skip("Skipped!")
                )
            full_name = f"{op_name}.{variant_name}" if variant_name else op_name
            decorator = DecorateInfo(
                decorator_callable,
                None,  # cls_name=None to match any class
                test_name,
                device_type=device_type,
                dtypes=dtypes,
            )
            overrides.setdefault(full_name, []).append(decorator)
        fn._op_overrides = overrides
        return fn

    return wrapped


try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport
with XPUPatchForImport(False):
    from test_ops import (
        fake_autocast_device_skips,
        TestCommon,
        TestCompositeCompliance,
        TestFakeTensor,
        TestForwardADWithScalars,
        TestMathBits,
    )

fake_autocast_device_skips["xpu"] = {"linalg.pinv", "pinverse"}


# XPU stft with float16 returns complex64 instead of complex32 (ComplexHalf unsupported).
# XPU stft returns complex64 for float16 input; complex32 (ComplexHalf) is unsupported.
@skipOps((skip("_refs.stft", dtypes=[torch.float16]),))
@ops(python_ref_db)
@skipIfTorchInductor("Takes too long for inductor")
def _test_python_ref_torch_fallback(self, device, dtype, op):
    if op.full_name == "_refs.div.floor_rounding" and dtype == torch.bfloat16:
        self.skipTest(
            "Skipped _refs.div.floor_rounding with bfloat16"
            "Divide by 0: _refs produces NaN, torch produces +/-inf"
        )
    self._ref_test_helper(contextlib.nullcontext, device, dtype, op)


_test_python_ref_torch_fallback.__name__ = "test_python_ref_torch_fallback"
TestCommon.test_python_ref_torch_fallback = _test_python_ref_torch_fallback

instantiate_device_type_tests(TestCommon, globals(), only_for="xpu", allow_xpu=True)
instantiate_device_type_tests(TestMathBits, globals(), only_for="xpu", allow_xpu=True)
# in finegrand
instantiate_device_type_tests(
    TestCompositeCompliance, globals(), only_for="xpu", allow_xpu=True
)
# only CPU
# instantiate_device_type_tests(TestRefsOpsInfo, globals(), only_for="xpu", allow_xpu=True)
# not important
instantiate_device_type_tests(TestFakeTensor, globals(), only_for="xpu", allow_xpu=True)
instantiate_device_type_tests(
    TestForwardADWithScalars, globals(), only_for="xpu", allow_xpu=True
)
# instantiate_device_type_tests(TestTags, globals(), only_for="xpu", allow_xpu=True)


if __name__ == "__main__":
    run_tests()
