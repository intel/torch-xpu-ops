# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]


from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests
from torch._prims import utils as prims_utils
import torch._prims as prims
import torch._refs as refs
import builtins

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport
with XPUPatchForImport(False):
    from test_ops import (
        TestCommon,
        TestCompositeCompliance,
        TestFakeTensor,
        TestForwardADWithScalars,
        TestMathBits,
    )


_original_reshape_view_helper = refs._reshape_view_helper


def _xpu_refs_reshape_view_helper(a, *shape, allow_copy):
    shape = prims_utils.extract_shape_from_varargs(shape, validate=False)
    shape = prims_utils.infer_size(shape, a.numel())
    if a.numel() == 0:
        if len(shape) == a.ndim and builtins.all(d == s for d, s in zip(shape, a.shape)):
            return prims.view_of(a)
    return _original_reshape_view_helper(a, *shape, allow_copy=allow_copy)


refs._reshape_view_helper = _xpu_refs_reshape_view_helper

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
