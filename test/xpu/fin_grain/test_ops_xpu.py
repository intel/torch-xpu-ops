# Owner(s): ["module: intel"]

import sys
import unittest

import torch
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyXPU,
    OpDTypes,
    ops,
)
from torch.testing._internal.common_methods_invocations import ops_and_refs
from torch.testing._internal.common_utils import (
    NoTest,
    run_tests,
    suppress_warnings,
    TEST_WITH_UBSAN,
    TEST_XPU,
    TestCase,
    slowTest,
    unMarkDynamoStrictTest,
)

try:
    from xpu_test_utils import get_wrapped_fn, XPUPatchForImport
except Exception as e:
    from ..xpu_test_utils import get_wrapped_fn, XPUPatchForImport

with XPUPatchForImport():
    from test_ops import TestCommon as TestCommonBase

if not TEST_XPU:
    print("XPU not available, skipping tests", file=sys.stderr)
    TestCase = NoTest  # noqa: F811

TEST_MULTIXPU = torch.xpu.device_count() > 1

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")

any_common_cpu_xpu_one = OpDTypes.any_common_cpu_cuda_one
cpu_xpu_all = (torch.bfloat16, torch.complex128, torch.complex64, torch.float16, torch.float32, torch.float64, torch.int16, torch.int32, torch.int64, torch.int8, torch.uint8, torch.bool)
_ops_and_refs_with_no_numpy_ref = [op for op in ops_and_refs if op.ref is None]

_xpu_computation_op_list = [
    "fill",
    "zeros",
    "zeros_like",
    "clone",
    "view_as_real",
    "view_as_complex",
    "view",
    "resize_",
    "resize_as_",
    "add",
    "sub",
    "mul",
    "div",
    "abs",
    "bernoulli",
    "bitwise_and",
    "bitwise_not",
    "bitwise_or",
    "bitwise_xor",
    "clamp",
    "clamp_max",
    "clamp_min",
    "clone",
    "copy",
    "cos",
    "cumsum",
    "empty",
    "eq",
    "fill",
    "fmod",
    "gcd",
    "ge",
    "gelu",
    "gt",
    "index_add",
    "index_put",
    "index_select",
    "isnan",
    "le",
    "log",
    "lt",
    "masked_fill",
    "maximum",
    "minimum",
    "mul",
    "native_dropout_backward",
    "ne",
    "neg",
    "nn.functional.adaptive_avg_pool2d",
    "nn.functional.threshold",
    "nonzero",
    "normal",
    "pow",
    "reciprocal",
    "_refs.rsub",
    "relu",
    "remainder",
    "reshape",
    "rsqrt",
    "sin",
    "sqrt",
    "sum",
    "tanh",
    "unfold",
    "uniform",
    "view",
    "where",
    "zero",
    "add",
    "any",
    "arange",
    "as_strided",
    "sort",
    "flip",
]

_xpu_computation_ops = [
    op for op in ops_and_refs if op.name in _xpu_computation_op_list
]

_xpu_computation_ops_no_numpy_ref = [
    op for op in _ops_and_refs_with_no_numpy_ref if op.name in _xpu_computation_op_list
]

# NB: TestCommonProxy is a nested class. This prevents test runners from picking
# it up and running it.
class Namespace:
    # When we import TestCommon, we patch the TestCase as NoTest to prevent test runners
    # picking TestCommon up and running it. But we still need to reuse its test cases.
    # Therefore, we build TestCommonProxy by inheriting the TestCommon and TestCase to ensure
    # the same feature set as the TestCommon.
    class TestCommonProxy(TestCase, TestCommonBase):
        pass


class TestCommon(TestCase):
    @onlyXPU
    @suppress_warnings
    @slowTest
    #@ops(_xpu_computation_ops_no_numpy_ref, dtypes=any_common_cpu_xpu_all)
    @ops(_xpu_computation_ops, dtypes=cpu_xpu_all)
    def test_compare_cpu(self, device, dtype, op):
        self.proxy = Namespace.TestCommonProxy()

        test_common_test_fn = get_wrapped_fn(Namespace.TestCommonProxy.test_compare_cpu)
        test_common_test_fn(self.proxy, device, dtype, op)

    @onlyXPU
    @ops(_xpu_computation_ops, allowed_dtypes=(torch.bool,))
    @unittest.skipIf(TEST_WITH_UBSAN, "Test uses undefined behavior")
    def test_non_standard_bool_values(self, device, dtype, op):
        self.proxy = Namespace.TestCommonProxy()

        test_common_test_fn = get_wrapped_fn(
            Namespace.TestCommonProxy.test_non_standard_bool_values
        )
        test_common_test_fn(self.proxy, device, dtype, op)


instantiate_device_type_tests(TestCommon, globals(), only_for="xpu")


if __name__ == "__main__":
    run_tests()
