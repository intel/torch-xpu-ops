# Owner(s): ["module: intel"]

import sys
import pytest
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
    IS_FBCODE,
    IS_SANDCASTLE,
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
    from test_ops import TestCompositeCompliance as TestCompositeComplianceBase

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
    "lerp",
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
    "all",
    "any",
    "arange",
    "as_strided",
    # "sort", # Comparison with CPU is not feasible due to its unstable sorting algorithm
    "flip",
    "tril",
    "triu",
    "cat",
    "log_softmax",
    "softmax",
    "scatter",
    "gather",
    "max_pool2d_with_indices_backward",
    "nn.functional.embedding",
    "nn.functional.unfold",
    # "nn.functional.nll_loss", # Lack of XPU implementation of aten::nll_loss2d_forward. Will retrieve the case, only if the op is implemented.
    "sigmoid",
    "sgn",
    "nn.functional.embedding_bag",
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

    class TestCompositeComplianceProxy(TestCase, TestCompositeComplianceBase):
        pass


class TestCommon(TestCase):
    @onlyXPU
    @suppress_warnings
    @slowTest
    #@ops(_xpu_computation_ops_no_numpy_ref, dtypes=any_common_cpu_xpu_all)
    @ops(_xpu_computation_ops, dtypes=cpu_xpu_all)
    def test_compare_cpu(self, device, dtype, op):
        if dtype in op.supported_dtypes(device):
            self.proxy = Namespace.TestCommonProxy()
            test_common_test_fn = get_wrapped_fn(Namespace.TestCommonProxy.test_compare_cpu)
            test_common_test_fn(self.proxy, device, dtype, op)
        else:
            pytest.skip(f"{op.name} has not supported {dtype} yet for {device}")

    @onlyXPU
    @ops(_xpu_computation_ops, allowed_dtypes=(torch.bool,))
    @unittest.skipIf(TEST_WITH_UBSAN, "Test uses undefined behavior")
    def test_non_standard_bool_values(self, device, dtype, op):
        self.proxy = Namespace.TestCommonProxy()

        test_common_test_fn = get_wrapped_fn(
            Namespace.TestCommonProxy.test_non_standard_bool_values
        )
        test_common_test_fn(self.proxy, device, dtype, op)


@unMarkDynamoStrictTest
class TestCompositeCompliance(TestCase):
    # Checks if the operator (if it is composite) is written to support most
    # backends and Tensor subclasses. See "CompositeImplicitAutograd Compliance"
    # in aten/src/ATen/native/README.md for more details
    @unittest.skipIf(
        IS_FBCODE or IS_SANDCASTLE, "__torch_dispatch__ does not work in fbcode"
    )
    @ops(_xpu_computation_ops, allowed_dtypes=(torch.float,))
    def test_operator(self, device, dtype, op):
        if dtype in op.supported_dtypes(device):
            self.proxy = Namespace.TestCompositeComplianceProxy()

            test_composite_compliance_test_fn = get_wrapped_fn(
                Namespace.TestCompositeComplianceProxy.test_operator
            )
            test_composite_compliance_test_fn(self.proxy, device, dtype, op)

    @unittest.skipIf(
        IS_FBCODE or IS_SANDCASTLE, "__torch_dispatch__ does not work in fbcode"
    )
    @ops([op for op in _xpu_computation_ops if op.supports_autograd], allowed_dtypes=(torch.float,))
    def test_backward(self, device, dtype, op):
        if dtype in op.supported_dtypes(device):
            self.proxy = Namespace.TestCompositeComplianceProxy()

            test_composite_compliance_test_fn = get_wrapped_fn(
                Namespace.TestCompositeComplianceProxy.test_backward
            )
            test_composite_compliance_test_fn(self.proxy, device, dtype, op)

    @unittest.skipIf(
        IS_FBCODE or IS_SANDCASTLE, "__torch_dispatch__ does not work in fbcode"
    )
    @ops(_xpu_computation_ops, allowed_dtypes=(torch.float,))
    def test_forward_ad(self, device, dtype, op):
        if dtype in op.supported_dtypes(device):
            self.proxy = Namespace.TestCompositeComplianceProxy()

            test_composite_compliance_test_fn = get_wrapped_fn(
                Namespace.TestCompositeComplianceProxy.test_forward_ad
            )
            test_composite_compliance_test_fn(self.proxy, device, dtype, op)

    @ops(_xpu_computation_ops, allowed_dtypes=(torch.float,))
    def test_cow_input(self, device, dtype, op):
        if dtype in op.supported_dtypes(device):
            self.proxy = Namespace.TestCompositeComplianceProxy()

            test_composite_compliance_test_fn = get_wrapped_fn(
                Namespace.TestCompositeComplianceProxy.test_cow_input
            )
            test_composite_compliance_test_fn(self.proxy, device, dtype, op)

    @ops(_xpu_computation_ops, allowed_dtypes=(torch.float,))
    def test_view_replay(self, device, dtype, op):
        if dtype in op.supported_dtypes(device):
            self.proxy = Namespace.TestCompositeComplianceProxy()

            test_composite_compliance_test_fn = get_wrapped_fn(
                Namespace.TestCompositeComplianceProxy.test_view_replay
            )
            test_composite_compliance_test_fn(self.proxy, device, dtype, op)


instantiate_device_type_tests(TestCommon, globals(), only_for="xpu", allow_xpu=True)
instantiate_device_type_tests(TestCompositeCompliance, globals(), only_for="xpu", allow_xpu=True)


if __name__ == "__main__":
    run_tests()
