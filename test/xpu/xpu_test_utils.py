# Owner(s): ["module: intel"]

import copy
import os
import sys
from torch import bfloat16
from torch.testing._internal import common_device_type, common_methods_invocations, common_utils


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
    "erf",
    "erfc",
    "bernoulli",
    "bitwise_and",
    "bitwise_not",
    "bitwise_or",
    "bitwise_xor",
    "addcmul",
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
    "hardtanh",
    "hardswish",
    "index_add",
    "index_put",
    "index_select",
    "isnan",
    "le",
    "log",
    "lt",
    "logical_not",
    "masked_fill",
    "maximum",
    "minimum",
    "mul",
    "native_dropout_backward",
    "ne",
    "neg",
    "nn.functional.adaptive_avg_pool2d",
    "nn.functional.elu",
    "nn.functional.threshold",
    "nn.functional.silu",
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
    "amin",
    "amax",
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
    "nn.functional.interpolate",
    "nn.functional.upsample_nearest",
    # "nn.functional.nll_loss", # Lack of XPU implementation of aten::nll_loss2d_forward. Will retrieve the case, only if the op is implemented.
    "sigmoid",
    "sgn",
    "nn.functional.embedding_bag",
    "grid_sampler_2d",
    # "nn.functional.grid_sample", # Lack of XPU implementation of aten::grid_sampler_3d.
    "acos",
    "acosh",
    "addr",
    "nn.functional.group_norm",
    "bincount",
    'nn.functional.interpolate',
]


def get_wrapped_fn(fn):
    if hasattr(fn, "__wrapped__"):
        wrapped = fn.__wrapped__
        return get_wrapped_fn(wrapped)
    else:
        return fn


def DO_NOTHING(*args, **kwargs):
    # Do nothing
    pass


class XPUPatchForImport:
    def __init__(self, patch_test_case=True) -> None:
        self.test_package = (
            os.path.dirname(os.path.abspath(__file__)) + "/../../../../test",
            os.path.dirname(os.path.abspath(__file__)) + "/../../../../test/nn",
        )
        self.patch_test_case = patch_test_case
        self.original_path = sys.path.copy()
        self.test_case_cls = common_utils.TestCase
        self.only_cuda_fn = common_device_type.onlyCUDA
        self.dtypes_if_cuda_fn = common_device_type.dtypesIfCUDA
        self.only_native_device_types_fn = (
            common_device_type.onlyNativeDeviceTypes
        )
        self.instantiate_device_type_tests_fn = (
            common_device_type.instantiate_device_type_tests
        )
        self.instantiate_parametrized_tests_fn = (
            common_utils.instantiate_parametrized_tests
        )
        self.python_ref_db = common_methods_invocations.python_ref_db
        self.ops_and_refs = common_methods_invocations.ops_and_refs

    def __enter__(self):
        # Monkey patch until we have a fancy way
        common_device_type.onlyCUDA = (
            common_device_type.onlyXPU
        )
        class dtypesIfXPU(common_device_type.dtypes):
            def __init__(self, *args):
                super().__init__(*args, device_type='xpu')
        common_device_type.dtypesIfCUDA=dtypesIfXPU
        common_device_type.onlyNativeDeviceTypes = (
            common_device_type.onlyXPU
        )
        if self.patch_test_case:
            common_utils.TestCase = (
                common_utils.NoTest
            )
        common_device_type.instantiate_device_type_tests = (
            DO_NOTHING
        )
        common_utils.instantiate_parametrized_tests = (
            DO_NOTHING
        )
        for op in common_methods_invocations.op_db:
            if op.name not in _xpu_computation_op_list:
                op.dtypesIfXPU = op.dtypes
            else:
                backward_dtypes = set(op.backward_dtypesIfCUDA)
                backward_dtypes.add(bfloat16)
                op.backward_dtypes = tuple(backward_dtypes)
        common_methods_invocations.python_ref_db = [op for op in self.python_ref_db if op.torch_opinfo_name in _xpu_computation_op_list]
        common_methods_invocations.ops_and_refs = common_methods_invocations.op_db + common_methods_invocations.python_ref_db

        sys.path.extend(self.test_package)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.path = self.original_path
        common_device_type.onlyCUDA = self.only_cuda_fn
        common_device_type.dtypesIfCUDA=self.dtypes_if_cuda_fn
        common_device_type.onlyNativeDeviceTypes = (
            self.only_native_device_types_fn
        )
        common_device_type.instantiate_device_type_tests = (
            self.instantiate_device_type_tests_fn
        )
        common_utils.instantiate_parametrized_tests = (
            self.instantiate_parametrized_tests_fn
        )
        common_utils.TestCase = self.test_case_cls
        common_methods_invocations.python_ref_db = self.python_ref_db
        common_methods_invocations.ops_and_refs = self.ops_and_refs


# Copy the test cases from generic_base_class to generic_test_class.
# It serves to reuse test cases. Regarding some newly added hardware,
# they have to copy and paste code manually from some test files to reuse the test
# cases. The maintenance effort is non-negligible as the test file changing is
# always on its way.
# This function provides an auto mechanism by replacing manual copy-paste w/
# automatically copying the test member functions from the base class to the dest test
# class.
def copy_tests(
    generic_test_class, generic_base_class, applicable_list=None, bypass_list=None
):
    assert len(generic_base_class.__bases__) > 0
    generic_base_class_members = set(generic_base_class.__dict__.keys()) - set(
        generic_test_class.__dict__.keys()
    )
    assert not (
        applicable_list and bypass_list
    ), "Does not support setting both applicable list and bypass list."
    if applicable_list:
        generic_base_class_members = [
            x for x in generic_base_class_members if x in applicable_list
        ]
    if bypass_list:
        generic_base_class_members = [
            x for x in generic_base_class_members if x not in bypass_list
        ]

    generic_base_tests = [x for x in generic_base_class_members if x.startswith("test")]

    for name in generic_base_class_members:
        if name in generic_base_tests:  # Instantiates test member
            test = getattr(generic_base_class, name)
            setattr(generic_test_class, name, copy.deepcopy(test))
        else:  # Ports non-test member
            nontest = getattr(generic_base_class, name)
            setattr(generic_test_class, name, nontest)
