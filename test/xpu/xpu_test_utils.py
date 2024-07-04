# Owner(s): ["module: intel"]


import copy
import os
import sys

import torch
from torch import bfloat16, cuda
from torch.testing._internal import (
    common_cuda,
    common_device_type,
    common_methods_invocations,
    common_utils,
)
from torch.testing._internal.common_nn import CriterionTest, ModuleTest
from torch.testing._internal.common_utils import set_default_dtype


_xpu_computation_op_list = [
    "empty",
    "fill",
    "zeros",
    "zeros_like",
    "clone",
    "randperm",
    "view_as_real",
    "view_as_complex",
    "view",
    "trace",
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
    "nn.functional.glu",
    "nn.functional.pad",
    "nn.functional.leaky_relu",
    "nn.functional.threshold",
    "nn.functional.silu",
    "nn.functional.hardsigmoid",
    "nn.functional.softplus",
    "nn.functional.softshrink",
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
    "std",
    "std_mean",
    "var",
    "var_mean",
    "tanh",
    "hypot",
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
    "roll",
    "tril",
    "triu",
    "cat",
    "log_softmax",
    "softmax",
    "scatter",
    "gather",
    "nn.functional.max_pool2d",
    "max_pool2d_with_indices_backward",
    "nn.functional.avg_pool2d",
    "nn.functional.embedding",
    "nn.functional.unfold",
    "nn.functional.pad",
    "nn.functional.interpolate",
    "nn.functional.upsample_bilinear",
    "nn.functional.upsample_nearest",
    # "nn.functional.nll_loss", # Lack of XPU implementation of aten::nll_loss2d_forward. Will retrieve the case, only if the op is implemented.
    "nn.functional.mse_loss",
    "sigmoid",
    "sgn",
    "nn.functional.embedding_bag",
    "grid_sampler_2d",
    # "nn.functional.grid_sample", # Lack of XPU implementation of aten::grid_sampler_3d.
    "acos",
    "acosh",
    "addr",
    "cdist",
    "nn.functional.group_norm",
    "nn.functional.batch_norm",
    "native_batch_norm",
    "_native_batch_norm_legit",
    "_batch_norm_with_update",
    "bincount",
    "renorm",
    "lerp",
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


def to_xpu(obj, type_map=None):
    if type_map is None:
        type_map = {}
    if isinstance(obj, torch.Tensor):
        assert obj.is_leaf
        t = type_map.get(obj.dtype, obj.dtype)
        with torch.no_grad():
            res = obj.clone().to(dtype=t, device="xpu")
            res.requires_grad = obj.requires_grad
        return res
    elif torch.is_storage(obj):
        return obj.new().resize_(obj.size()).copy_(obj)
    elif isinstance(obj, list):
        return [to_xpu(o, type_map) for o in obj]
    elif isinstance(obj, tuple):
        return tuple(to_xpu(o, type_map) for o in obj)
    else:
        return copy.deepcopy(obj)


def ModuleTest_test_xpu(self, test_case):
    with set_default_dtype(self.default_dtype):
        cpu_input = self._get_input()

        type_map = {torch.double: torch.float}
        cpu_input_tuple = cpu_input if isinstance(cpu_input, tuple) else (cpu_input,)

        is_any_input_complex = any(
            isinstance(t, torch.Tensor) and t.dtype.is_complex for t in cpu_input_tuple
        )

        xpu_input_tuple = to_xpu(cpu_input_tuple, type_map=type_map)

        cpu_module = self.constructor(*self.constructor_args)
        xpu_module = self.constructor(*self.constructor_args).float().xpu()
        cpu_param = test_case._get_parameters(cpu_module)
        xpu_param = test_case._get_parameters(xpu_module)
        for cpu_p, xpu_p in zip(cpu_param[0], xpu_param[0]):
            xpu_p.data.copy_(cpu_p)
        test_case._zero_grad_input(cpu_input_tuple)
        test_case._zero_grad_input(xpu_input_tuple)
        test_case._zero_grad_parameters(cpu_module)
        test_case._zero_grad_parameters(xpu_module)
        cpu_output = test_case._forward(cpu_module, cpu_input_tuple)
        xpu_output = test_case._forward(xpu_module, xpu_input_tuple)
        test_case.assertEqual(
            cpu_input_tuple,
            xpu_input_tuple,
            atol=self.precision,
            rtol=0,
            exact_dtype=False,
        )
        if getattr(cpu_module, "return_indices", False):
            cpu_output = cpu_output[0]
            xpu_output = xpu_output[0]
        test_case.assertEqual(
            cpu_output, xpu_output, atol=self.precision, rtol=0, exact_dtype=False
        )

        # Run backwards on CPU and GPU and compare results

        for _ in range(5):
            cpu_gradOutput = cpu_output.clone().normal_()
            xpu_gradOutput = cpu_gradOutput.type_as(xpu_output)
            test_case.assertEqual(
                cpu_input_tuple,
                xpu_input_tuple,
                atol=self.precision,
                rtol=0,
                exact_dtype=False,
            )
            cpu_gradInput = test_case._backward(
                cpu_module, cpu_input_tuple, cpu_output, cpu_gradOutput
            )
            xpu_gradInput = test_case._backward(
                xpu_module, xpu_input_tuple, xpu_output, xpu_gradOutput
            )
            test_case.assertEqual(
                cpu_gradInput,
                xpu_gradInput,
                atol=self.precision,
                rtol=0,
                exact_dtype=False,
            )
            for cpu_d_p, xpu_d_p in zip(cpu_param[1], xpu_param[1]):
                test_case.assertEqual(cpu_d_p, xpu_d_p, atol=self.precision, rtol=0)
        # Run double-backwards on CPU and GPU and compare results

        if self.check_gradgrad and not self.FIXME_no_cuda_gradgrad_comparison:
            cpu_output = cpu_module(*cpu_input_tuple)
            xpu_output = xpu_module(*xpu_input_tuple)
            if getattr(cpu_module, "return_indices", False):
                cpu_output = cpu_output[0]
                xpu_output = xpu_output[0]
            cpu_gradOutput = torch.randn_like(cpu_output, requires_grad=True)
            xpu_gradOutput = cpu_gradOutput.type_as(xpu_output).detach()
            xpu_gradOutput.requires_grad = True

            cpu_gradInputs = torch.autograd.grad(
                cpu_output,
                cpu_input_tuple + tuple(cpu_module.parameters()),
                cpu_gradOutput,
                create_graph=True,
            )
            xpu_gradInputs = torch.autograd.grad(
                xpu_output,
                xpu_input_tuple + tuple(xpu_module.parameters()),
                xpu_gradOutput,
                create_graph=True,
            )

            for cpu_d_i, xpu_d_i in zip(cpu_gradInputs, xpu_gradInputs):
                test_case.assertEqual(
                    cpu_d_i, xpu_d_i, atol=self.precision, rtol=0, exact_dtype=False
                )
            # We mix output into the second backwards computation so that
            # torch.autograd.grad doesn't complain that some inputs
            # are unreachable (which can happen if you differentiate
            # only on the gradient.

            if is_any_input_complex:
                outputs_cpu = cpu_output.sum().abs() + sum(
                    x.sum().abs() for x in cpu_gradInputs
                )
                outputs_xpu = xpu_output.sum().abs() + sum(
                    x.sum().abs() for x in xpu_gradInputs
                )
            else:
                outputs_cpu = cpu_output.sum() + sum(x.sum() for x in cpu_gradInputs)
                outputs_xpu = xpu_output.sum() + sum(x.sum() for x in xpu_gradInputs)
            cpu_gg = torch.autograd.grad(
                outputs_cpu,
                cpu_input_tuple + (cpu_gradOutput,) + tuple(cpu_module.parameters()),
                retain_graph=True,
            )
            xpu_gg = torch.autograd.grad(
                outputs_xpu,
                xpu_input_tuple + (xpu_gradOutput,) + tuple(xpu_module.parameters()),
                retain_graph=True,
            )
            test_case.assertEqual(
                cpu_gradInput,
                xpu_gradInput,
                atol=self.precision,
                rtol=0,
                exact_dtype=False,
            )
            for cpu_d_p, xpu_d_p in zip(cpu_gg, xpu_gg):
                test_case.assertEqual(
                    cpu_d_p, xpu_d_p, atol=self.precision, rtol=0, exact_dtype=False
                )
        self.test_noncontig(test_case, xpu_module, xpu_input_tuple)


ModuleTest.test_cuda = ModuleTest_test_xpu


def CriterionTest_test_xpu(self, test_case, dtype, extra_args=None):
    def convert_dtype(obj, dtype, requires_grad=False):
        if isinstance(obj, torch.Tensor):
            return obj.detach().to(dtype=dtype).requires_grad_(requires_grad)
        elif isinstance(obj, tuple):
            return tuple(convert_dtype(o, dtype, requires_grad) for o in obj)
        else:
            return obj

    with set_default_dtype(self.default_dtype):
        cpu_input = self._get_input()
        cpu_target = self._get_target()
        cpu_module = self.constructor(*self.constructor_args)
        xpu_module = self.constructor(*self.constructor_args)

        # Convert input, target and module parameters to dtype

        cpu_input = convert_dtype(cpu_input, dtype, True)
        if cpu_target.is_floating_point() or cpu_target.is_complex():
            cpu_target = convert_dtype(cpu_target, dtype)
        cpu_module.type(dtype)
        xpu_module.type(dtype)

        # GPU setup

        xpu_input = to_xpu(cpu_input)
        xpu_target = to_xpu(cpu_target)
        xpu_module.xpu()

        # torch.HalfTensor doesn't support most operations, converting back to default

        if dtype in {torch.half, torch.bfloat16}:
            cpu_input = self._get_input()
            cpu_target = self._get_target()
            # Loss modules with weights require consistent input/module weight types

            cpu_module = self.constructor(*self.constructor_args)
        cpu_output = test_case._forward_criterion(
            cpu_module, cpu_input, cpu_target, extra_args=extra_args
        )
        xpu_output = test_case._forward_criterion(
            xpu_module, xpu_input, xpu_target, extra_args=extra_args
        )
        # dtype used to be able to be None, so set precision in this way instead of a precision map

        test_case.assertEqual(
            cpu_output,
            xpu_output,
            atol=1e-1 if dtype in {torch.half, torch.bfloat16} else 4e-4,
            rtol=0,
            exact_dtype=False,
        )

        cpu_gradInput = test_case._backward_criterion(
            cpu_module, cpu_input, cpu_output, cpu_target, extra_args=extra_args
        )
        xpu_gradInput = test_case._backward_criterion(
            xpu_module, xpu_input, xpu_output, xpu_target, extra_args=extra_args
        )
        # dtype used to be able to be None, so set precision in this way instead of a precision map

        test_case.assertEqual(
            cpu_gradInput,
            xpu_gradInput,
            atol=1e-1 if dtype in {torch.half, torch.bfloat16} else 4e-4,
            rtol=0,
            exact_dtype=False,
        )


CriterionTest.test_cuda = CriterionTest_test_xpu


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
        self.only_native_device_types_fn = common_device_type.onlyNativeDeviceTypes
        self.instantiate_device_type_tests_fn = (
            common_device_type.instantiate_device_type_tests
        )
        self.instantiate_parametrized_tests_fn = (
            common_utils.instantiate_parametrized_tests
        )
        self.python_ref_db = common_methods_invocations.python_ref_db
        self.ops_and_refs = common_methods_invocations.ops_and_refs
        self.largeTensorTest = common_device_type.largeTensorTest
        self.TEST_CUDA = common_cuda.TEST_CUDA
        self.TEST_CUDNN = common_cuda.TEST_CUDNN
        self.cuda_is_available = cuda.is_available
        self.cuda_is_bf16_supported = cuda.is_bf16_supported

    def __enter__(self):
        # Monkey patch until we have a fancy way

        common_device_type.onlyCUDA = common_device_type.onlyXPU

        class dtypesIfXPU(common_device_type.dtypes):
            def __init__(self, *args):
                super().__init__(*args, device_type="xpu")

        common_device_type.dtypesIfCUDA = dtypesIfXPU
        common_device_type.onlyNativeDeviceTypes = common_device_type.onlyXPU
        if self.patch_test_case:
            common_utils.TestCase = common_utils.NoTest
        common_device_type.instantiate_device_type_tests = DO_NOTHING
        common_utils.instantiate_parametrized_tests = DO_NOTHING
        common_device_type.largeTensorTest = (
            lambda size, device=None: self.largeTensorTest(
                size, device if device and device != "cuda" else "xpu"
            )
        )
        for op in common_methods_invocations.op_db:
            if op.decorators and len(op.decorators) > 0:
                if self.only_cuda_fn in op.decorators:
                    tmp = list(op.decorators)
                    for i in range(len(op.decorators)):
                        if op.decorators[i] == self.only_cuda_fn:
                            tmp[i] = common_device_type.onlyCUDA
                    op.decorators = tuple(tmp)
            if op.name not in _xpu_computation_op_list:
                op.dtypesIfXPU = op.dtypes
            else:
                backward_dtypes = set(op.backward_dtypesIfCUDA)
                backward_dtypes.add(bfloat16)
                op.backward_dtypes = tuple(backward_dtypes)
        common_methods_invocations.python_ref_db = [
            op
            for op in self.python_ref_db
            if op.torch_opinfo_name in _xpu_computation_op_list
        ]
        common_methods_invocations.ops_and_refs = (
            common_methods_invocations.op_db + common_methods_invocations.python_ref_db
        )
        common_cuda.TEST_CUDA = True
        common_cuda.TEST_CUDNN = True
        cuda.is_available = lambda: True
        cuda.is_bf16_supported = lambda: True

        sys.path.extend(self.test_package)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.path = self.original_path
        common_device_type.onlyCUDA = self.only_cuda_fn
        common_device_type.dtypesIfCUDA = self.dtypes_if_cuda_fn
        common_device_type.onlyNativeDeviceTypes = self.only_native_device_types_fn
        common_device_type.instantiate_device_type_tests = (
            self.instantiate_device_type_tests_fn
        )
        common_utils.instantiate_parametrized_tests = (
            self.instantiate_parametrized_tests_fn
        )
        common_utils.TestCase = self.test_case_cls
        common_methods_invocations.python_ref_db = self.python_ref_db
        common_methods_invocations.ops_and_refs = self.ops_and_refs
        common_device_type.largeTensorTest = self.largeTensorTest
        common_cuda.TEST_CUDA = self.TEST_CUDA
        common_cuda.TEST_CUDNN = self.TEST_CUDNN
        cuda.is_available = self.cuda_is_available
        cuda.is_bf16_supported = self.cuda_is_bf16_supported


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
