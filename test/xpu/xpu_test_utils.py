# Owner(s): ["module: intel"]


import copy
import os
import pytest
import sys
import unittest

import torch
from torch import bfloat16, cuda
from torch.testing._internal import (
    common_cuda,
    common_device_type,
    common_methods_invocations,
    common_utils,
)
from torch.testing._internal.common_device_type import tol, toleranceOverride
from torch.testing._internal.common_modules import module_db
from torch.testing._internal.common_nn import CriterionTest, ModuleTest
from torch.testing._internal.common_utils import set_default_dtype
from torch.testing._internal.opinfo.core import (
    BinaryUfuncInfo,
    DecorateInfo,
    OpInfo,
    ReductionOpInfo,
    ShapeFuncInfo,
    SpectralFuncInfo,
    UnaryUfuncInfo,
)

_xpu_computation_op_list = [
    "empty",
    "eye",
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
    "erfinv",
    "bernoulli",
    "bitwise_and",
    "bitwise_not",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_left_shift",
    "bitwise_right_shift",
    "addcmul",
    "addcdiv",
    "clamp",
    "clamp_max",
    "clamp_min",
    "clone",
    "copy",
    "cumprod",
    "cumsum",
    "cummax",
    "cummin",
    "equal",
    "eq",
    "exp",
    "exp2",
    "expm1",
    "exponential",
    "fill",
    "fmod",
    "__rmod__",
    "gcd",
    "ge",
    "gelu",
    "gt",
    "hardtanh",
    "hardswish",
    "nn.functional.hardshrink",
    "nn.functional.mish",
    "i0",
    "index_add",
    "index_reduce",
    "index_fill",
    "index_put",
    "index_select",
    "masked_scatter",
    "masked_select",
    "isin",
    "isnan",
    "kthvalue",
    "lcm",
    "le",
    "log",
    "log10",
    "log1p",
    "log2",
    "logaddexp",
    "logaddexp2",
    "logcumsumexp",
    "logit",
    "lt",
    "logical_and",
    "logical_or",
    "logical_xor",
    "logical_not",
    "masked_fill",
    "maximum",
    "minimum",
    "mul",
    "median",
    "nanmedian",
    "native_dropout_backward",
    "nn.functional.dropout",
    "ne",
    "neg",
    "nn.functional.elu",
    "nn.functional.glu",
    "nn.functional.pad",
    "nn.functional.leaky_relu",
    "nn.functional.prelu",
    "nn.functional.rrelu",
    "nn.functional.threshold",
    "nn.functional.silu",
    "nn.functional.hardsigmoid",
    "nn.functional.softplus",
    "nn.functional.softshrink",
    "nn.functional.local_response_norm",
    "nextafter",
    "heaviside",
    "nonzero",
    "normal",
    "pow",
    "reciprocal",
    "_refs.rsub",
    "relu",
    "remainder",
    "reshape",
    "rsqrt",
    "cos",
    "cosh",
    "acos",
    "acosh",
    "sin",
    "sinc",
    "sinh",
    "asin",
    "asinh",
    "tan",
    "tanh",
    "atan",
    "atan2",
    "atanh",
    "sqrt",
    "sum",
    "prod",
    "nansum",
    "amin",
    "amax",
    "std",
    "std_mean",
    "var",
    "var_mean",
    "norm",
    "hypot",
    "unfold",
    "uniform",
    "view",
    "where",
    "zero",
    "add",
    "all",
    "any",
    "isposinf",
    "isneginf",
    "arange",
    "as_strided",
    # "sort", # Comparison with CPU is not feasible due to its unstable sorting algorithm
    # "topk", # Comparison with CPU is not feasible due to its unstable sorting algorithm
    "flip",
    "roll",
    "tril",
    "triu",
    "cat",
    "log_softmax",
    "softmax",
    "_softmax_backward_data",
    "scatter",
    "gather",
    "nn.functional.adaptive_max_pool2d",
    "nn.functional.adaptive_max_pool3d",
    "nn.functional.max_pool2d",
    "nn.functional.fractional_max_pool2d",
    "nn.functional.fractional_max_pool3d",
    "max_pool2d_with_indices_backward",
    "nn.functional.max_pool3d",
    "nn.functional.adaptive_avg_pool2d",
    "nn.functional.adaptive_avg_pool3d",
    "nn.functional.avg_pool1d",
    "nn.functional.avg_pool2d",
    "nn.functional.avg_pool3d",
    "nn.functional.embedding",
    "nn.functional.unfold",
    "nn.functional.pad",
    "nn.functional.interpolate",
    "nn.functional.upsample_bilinear",
    "_upsample_bilinear2d_aa",
    "nn.functional.upsample_nearest",
    "nn.functional.nll_loss",
    "nn.functional.smooth_l1_loss",
    "nn.functional.mse_loss",
    "nn.functional.binary_cross_entropy",
    "nn.functional.multilabel_margin_loss",
    "nn.functional.huber_loss",
    "nn.functional.multi_margin_loss",
    "nn.functional.max_unpool2d",
    "nn.functional.max_unpool3d",
    "nn.functional.ctc_loss",
    "nn.functional.channel_shuffle",
    "nn.functional.multi_head_attention_forward",
    "nn.GRUCell",
    "nn.LSTMCell",
    "sigmoid",
    "logsigmoid",
    "sgn",
    "sign",
    "signbit",
    "round",
    "trunc",
    "xlogy",
    "nn.functional.embedding_bag",
    "bucketize",
    "searchsorted",
    "grid_sampler_2d",
    "nn.functional.grid_sample",
    "addr",
    "cdist",
    "nn.functional.pdist",
    "nn.functional.group_norm",
    "nn.functional.batch_norm",
    "native_batch_norm",
    "_native_batch_norm_legit",
    "_batch_norm_with_update",
    "bincount",
    "cross",
    "renorm",
    "igamma",
    "igammac",
    "digamma",
    "polygamma",
    "lgamma",
    "linspace",
    "logspace",
    "unique_consecutive",
    "unique",
    "multinomial",
    "lerp",
    "polar",
    "frac",
    "aminmax",
    "argmin",
    "angle",
    "conj_physical",
    "histogram",
    "histc",
    "repeat_interleave",
    "fmax",
    "fmin",
    "max",
    "min",
    "floor",
    "floor_divide",
    "frexp",
    "copysign",
    "count_nonzero",
    "nan_to_num",
    "scatter_reduce",
    "nanmean",
    "native_layer_norm",
    "native_layer_norm_backward",
    "square",
    "heaviside",
    "argsort",
    "tril_indices",
    "triu_indices",
    "index_copy",
    "cauchy",
    "geometric",
    "mode",
    "log_normal",
    "take",
    "put",
    "_segment_reduce",
    "_chunk_cat",
    "split_with_sizes_copy",
]

_ops_without_cuda_support = [
    "histogram",
    "histogramdd",
]

# some case fail in cuda becasue of cuda's bug, so cuda set xfail in opdb
# but xpu can pass these case, and assert 'unexpected success'
# the list will pass these case.


_cuda_xfail_xpu_pass = [
    ("rsqrt", "test_reference_numerics_large"),
    ("_batch_norm_with_update", "test_noncontiguous_samples"),
    ("_batch_norm_with_update", "test_dispatch_symbolic_meta_outplace_all_strides"),
    ("histc", "test_out"),
    ("_refs.mul", "test_python_ref"),
    ("_refs.mul", "test_python_ref_torch_fallback"),
    ("nn.AvgPool2d", "test_memory_format"),
    ("narrow_copy","test_meta_outplace"),
    ("narrow_copy","test_dispatch_meta_outplace"),
    ("narrow_copy","test_dispatch_symbolic_meta_outplace"),
]

# some case should adjust tolerance to pass.
# The new threshold is at the same order of magnitude as cuda's or cpu's.
# format hint:{op_name:{(cls_name,test_name):{dtype:tol(atol, rtol)}}

_xpu_tolerance_override = {
    "nn.functional.grid_sample": {
        ("TestCommon", "test_compare_cpu"): {
            torch.float32: tol(atol=0.002, rtol=0.008),
        }
    },
    "nn.functional.tanhshrink": {
        ("TestUnaryUfuncs", "test_reference_numerics_normal"): {
            torch.complex64: tol(atol=2e-05, rtol=9e-06),
            torch.bfloat16: tol(atol=1e-02, rtol=1.6e-02),
        }
    },
    "atan2": {
        ("TestCommon", "test_compare_cpu"): {
            torch.bfloat16: tol(atol=0.008, rtol=0.005),
        }
    },
    "cumprod": {
        ("TestCommon", "test_compare_cpu"): {
            torch.bfloat16: tol(atol=0.002, rtol=0.008),
        }
    },
    "nanmean": {
        ("TestCommon", "test_compare_cpu"): {
            torch.bfloat16: tol(atol=0.002, rtol=0.008),
        }
    },
    "nansum": {
        ("TestCommon", "test_compare_cpu"): {
            torch.bfloat16: tol(atol=0.008, rtol=0.006),
        }
    },
    "nn.functional.batch_norm": {
        ("TestCommon", "test_compare_cpu"): {
            torch.float16: tol(atol=0.003, rtol=0.004),
        }
    },
    "nn.functional.embedding_bag": {
        ("TestCommon", "test_compare_cpu"): {
            torch.float16: tol(atol=0.005, rtol=0.007),
        }
    },
    "nn.functional.group_norm": {
        ("TestCommon", "test_compare_cpu"): {
            torch.float16: tol(atol=0.002, rtol=0.006),
        }
    },
    "prod": {
        ("TestCommon", "test_compare_cpu"): {
            torch.bfloat16: tol(atol=0.002, rtol=0.005),
        }
    },
    "rsqrt": {
        ("TestCommon", "test_compare_cpu"): {
            torch.bfloat16: tol(atol=0.004, rtol=0.007),
        }
    },
    "std_mean": {
        ("TestCommon", "test_compare_cpu"): {
            torch.bfloat16: tol(atol=0.008, rtol=0.005),
        }
    },
    "var_mean": {
        ("TestCommon", "test_compare_cpu"): {
            torch.bfloat16: tol(atol=0.008, rtol=0.005),
        }
    },
    "nn.LazyConvTranspose3d": {
        ("TestModule", "test_non_contiguous_tensors"): {
            torch.float32: tol(atol=2e-5, rtol=5e-5),
        }
    },
    "histogram": {
        ("TestCommon", "test_out"):{
            torch.float32: tol(atol=3e-5, rtol=5e-5),
        }
    }
}


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
    if not self.should_test_cuda:
        raise unittest.SkipTest("Excluded from XPU tests")
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
        if getattr(cpu_module, "return_indices", False):
            cpu_output = cpu_output[0]
            xpu_output = xpu_output[0]
        test_case.assertEqual(
            cpu_output, xpu_output, atol=self.precision, rtol=0, exact_dtype=False
        )

        # Run backwards on CPU and xpu and compare results

        for _ in range(5):
            cpu_gradOutput = cpu_output.clone().normal_()
            xpu_gradOutput = cpu_gradOutput.type_as(xpu_output)
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
        # Run double-backwards on CPU and xpu and compare results

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

    if not self.should_test_cuda:
        raise unittest.SkipTest("Excluded from XPU tests")
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

from torch.testing._internal.common_methods_invocations import sample_inputs_cat_concat, S, M
from torch.testing._internal.common_methods_invocations import make_tensor, mask_not_all_zeros
from functools import partial
from torch.testing._internal.opinfo.core import SampleInput

def reference_inputs_cat_nofp64(op, device, dtype, requires_grad, **kwargs):
    yield from sample_inputs_cat_concat(op, device, dtype, requires_grad, **kwargs)

    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # Noncontiguous type promoting tensors
    a = make_arg((3, 4, 2))
    #b = make_arg((3, 2, 2), noncontiguous=True, dtype=torch.double)
    # for platform without fp64 support
    b = make_arg((3, 2, 2), noncontiguous=True, dtype=torch.float)
    c = make_arg((3, 3, 2), dtype=torch.float16).permute(1, 0, 2)

    yield SampleInput((a, b, c), kwargs={'dim': 1})

    # Special 1D tensor with dim length of 0 case
    a = make_arg((0,))
    b = make_arg((3, 2, 2))

    yield SampleInput((a, b, a))
    yield SampleInput((a, a, a))


def index_variable_nofp64(shape, max_indices, device=torch.device('cpu')):
    if not isinstance(shape, tuple):
        shape = (shape,)
    #index = torch.rand(*shape, dtype=torch.double, device=device).mul_(max_indices).floor_().long()
    # for platform without fp64 support
    index = torch.rand(*shape, dtype=torch.float32, device=device).mul_(max_indices).floor_().long()
    return index

def sample_inputs_index_put_nofp64(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    for accumulate in [False, True]:
        # Test with indices arg
        yield SampleInput(
            make_arg((S, S,)),
            (index_variable_nofp64(2, S, device=device),),
            make_arg((2, S)),
            accumulate=accumulate)

        # Test with mask arg
        mask = torch.zeros(S, dtype=torch.bool) if accumulate else mask_not_all_zeros((S,))
        yield SampleInput(
            make_arg((S, S)), (mask, ), make_arg((S,)), accumulate=accumulate)

def sample_inputs_softmax_variant_nofp64(
    op_info,
    device,
    dtype,
    requires_grad,
    with_dtype=False,
    use_zero_dimensions=True,
    **kwargs,
):
    make_arg = partial(
        make_tensor, device=device, dtype=dtype, requires_grad=requires_grad
    )
    cases = [
        ((S,), (0,)),
        ((S, S), (0,)),
        ((S, S), (1,)),
        ((S, S), (-1,)),
        ((S, M, S), (2,)),
        *([((S, 0, 0), (-1,))] if use_zero_dimensions else []),
    ]
    #kwargs = dict(dtype=torch.float64) if with_dtype else None
    # for platform without fp64 support
    kwargs = dict(dtype=torch.float32) if with_dtype else None

    # PyTorch on XLA throws an error when passed with dim argument for 0d tensor.
    # See https://github.com/pytorch/xla/issues/3061 for more details.
    if torch.device(device).type != "xla":
        cases.append(((), (0,)))

    return (
        SampleInput(make_arg(shape), args=dim, kwargs=kwargs) for shape, dim in cases
    )

def sample_inputs_like_fns_nofp64(self, device, dtype, requires_grad, **kwargs):

    inputs = [
        ((), {}),
        ((S, S), {}),
        ((0, S, 0), {}),
        ((S,), {'dtype': dtype, 'device': device}),
        # Hard-code some dtypes/devices. We want to test cases where the
        # (dtype, device) is different from the input's (dtype, device)
        # disabled for ARC
        # ((S,), {'dtype': torch.double}),
        ((S,), {'device': 'cpu'}),
        # disabled for ARC
        #((S,), {'dtype': torch.double, 'device': 'cpu'}),
    ]
    if torch.cuda.is_available():
        inputs.append(((S,), {'device': 'cuda'}))

    for shape, kwargs in inputs:
        t = make_tensor(shape, dtype=dtype, device=device,
                        low=None, high=None,
                        requires_grad=requires_grad)
        yield SampleInput(t, **kwargs)

class XPUPatchForImport:
    def __init__(self, patch_test_case=True) -> None:
        test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../test")
        self.test_package = (
            test_dir,
            os.path.join(test_dir, "nn"),
            os.path.join(test_dir, "distributions"),
            os.path.join(test_dir, "quantization/core"),
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
        self.foreach_unary_op_db = common_methods_invocations.foreach_unary_op_db
        self.foreach_binary_op_db = common_methods_invocations.foreach_binary_op_db
        self.foreach_pointwise_op_db = (
            common_methods_invocations.foreach_pointwise_op_db
        )
        self.foreach_reduce_op_db = common_methods_invocations.foreach_reduce_op_db
        self.foreach_other_op_db = common_methods_invocations.foreach_other_op_db
        self.python_ref_db = common_methods_invocations.python_ref_db
        self.ops_and_refs = common_methods_invocations.ops_and_refs
        self.largeTensorTest = common_device_type.largeTensorTest
        self.TEST_CUDA = common_cuda.TEST_CUDA
        self.TEST_CUDNN = common_cuda.TEST_CUDNN
        self.cuda_is_available = cuda.is_available
        self.cuda_is_bf16_supported = cuda.is_bf16_supported


    def align_db_decorators(self, db):
        def gen_xpu_wrappers(op_name, wrappers):
            wrapper_xpu = []
            replaced = False
            for wrapper in wrappers:
                if type(wrapper) == DecorateInfo:
                    if wrapper.device_type == "cuda":
                        if (
                            unittest.expectedFailure in wrapper.decorators
                            and (op_name, wrapper.test_name) in _cuda_xfail_xpu_pass
                        ):
                            pass
                        else:
                            wrapper.device_type = "xpu"
                            replaced = True
                    wrapper_xpu.append(wrapper)
                elif self.only_cuda_fn == wrapper:
                    wrapper_xpu.append(common_device_type.onlyCUDA)
                    replaced = True
            return replaced, wrapper_xpu

        for info in db:
            if hasattr(info, "decorators"):
                replaced, decorator_xpu = gen_xpu_wrappers(info.name, info.decorators)

                # the latter decorator will override the former.
                if info.name in _xpu_tolerance_override:
                    replaced = True
                    for case, tolerance in _xpu_tolerance_override[info.name].items():
                        decorator_xpu.append(
                            DecorateInfo(
                                toleranceOverride(tolerance),
                                case[0],  # cls_name
                                case[1],  # test_name
                                device_type="xpu",
                            )
                        )
                if replaced:
                    info.decorators = tuple(decorator_xpu)
            if hasattr(info, "skips"):
                replaced, skip_xpu = gen_xpu_wrappers(info.name, info.skips)
                if replaced:
                    info.skips = tuple(skip_xpu)

    def align_supported_dtypes(self, db):
        for opinfo in db:
            if ( opinfo.name not in _xpu_computation_op_list and (opinfo.torch_opinfo.name not in _xpu_computation_op_list 
                if db == common_methods_invocations.python_ref_db else True)) or opinfo.name in _ops_without_cuda_support:
                opinfo.dtypesIfXPU = opinfo.dtypes
            else:
                backward_dtypes = set(opinfo.backward_dtypesIfCUDA)
                if bfloat16 in opinfo.dtypesIfXPU:
                    backward_dtypes.add(bfloat16)
                opinfo.backward_dtypes = tuple(backward_dtypes)

            if "has_fp64=0" in str(torch.xpu.get_device_properties(0)):
                fp64_dtypes = [ torch.float64, torch.complex128, torch.double, ]
                opinfo.dtypesIfXPU = set(filter(lambda x: (x not in fp64_dtypes), list(opinfo.dtypesIfXPU)))
                opinfo.backward_dtypes = tuple(filter(lambda x: (x not in fp64_dtypes), list(opinfo.backward_dtypes)))

    def filter_fp64_sample_input(self, db):
        # Only for platform without fp64 support
        if "has_fp64=0" in str(torch.xpu.get_device_properties(0)):
            for opinfo in db:
                if opinfo.name in _xpu_computation_op_list:
                    if opinfo.variant_test_name == "with_dtype" and \
                        opinfo.name in ["log_softmax", "softmax", "nn.functional.softmin", ] and \
                        get_wrapped_fn(opinfo.sample_inputs_func) != opinfo.sample_inputs_func and \
                        get_wrapped_fn(opinfo.sample_inputs_func).func.__name__ == common_methods_invocations.sample_inputs_softmax_variant.__name__:
                            opinfo.sample_inputs_func = torch.no_grad()(partial(sample_inputs_softmax_variant_nofp64, with_dtype=True))
                    elif opinfo.sample_inputs_func.__name__ == common_methods_invocations.sample_inputs_softmax_variant.__name__:
                        opinfo.sample_inputs_func = sample_inputs_softmax_variant_nofp64
                    elif opinfo.sample_inputs_func.__name__ == common_methods_invocations.sample_inputs_like_fns.__name__:
                        opinfo.sample_inputs_func = sample_inputs_like_fns_nofp64
                    elif opinfo.sample_inputs_func.__name__ == common_methods_invocations.sample_inputs_index_put.__name__:
                        opinfo.sample_inputs_func = sample_inputs_index_put_nofp64

                    if opinfo.reference_inputs_func != None and opinfo.reference_inputs_func.__name__ == common_methods_invocations.reference_inputs_cat.__name__:
                        opinfo.reference_inputs_func = reference_inputs_cat_nofp64

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
        for db in [
            common_methods_invocations.foreach_unary_op_db,
            common_methods_invocations.foreach_binary_op_db,
            common_methods_invocations.foreach_pointwise_op_db,
            common_methods_invocations.foreach_reduce_op_db,
            common_methods_invocations.foreach_other_op_db,
            common_methods_invocations.python_ref_db,
            common_methods_invocations.op_db,
        ]:
            self.align_supported_dtypes(db)
            self.align_db_decorators(db)
            self.filter_fp64_sample_input(db)
        self.align_db_decorators(module_db)
        common_methods_invocations.python_ref_db = [
            op
            for op in self.python_ref_db
            if op.torch_opinfo_name in _xpu_computation_op_list
        ]
        common_methods_invocations.ops_and_refs = (
            common_methods_invocations.op_db + common_methods_invocations.python_ref_db
        )
        common_methods_invocations.unary_ufuncs = [
            op
            for op in common_methods_invocations.ops_and_refs
            if isinstance(op, UnaryUfuncInfo)
        ]
        common_methods_invocations.binary_ufuncs = [
            op
            for op in common_methods_invocations.ops_and_refs
            if isinstance(op, BinaryUfuncInfo)
        ]
        common_methods_invocations.binary_ufuncs_and_refs = tuple(
            op
            for op in common_methods_invocations.ops_and_refs
            if isinstance(op, BinaryUfuncInfo)
        )
        common_methods_invocations.spectral_funcs = [
            op
            for op in common_methods_invocations.ops_and_refs
            if isinstance(op, SpectralFuncInfo)
        ]
        common_methods_invocations.sparse_unary_ufuncs = [
            op
            for op in common_methods_invocations.op_db
            if isinstance(op, UnaryUfuncInfo) and op.supports_sparse
        ]
        common_methods_invocations.sparse_csr_unary_ufuncs = [
            op
            for op in common_methods_invocations.op_db
            if isinstance(op, UnaryUfuncInfo) and op.supports_sparse_csr
        ]
        common_methods_invocations.sparse_reduction_ops = [
            op
            for op in common_methods_invocations.op_db
            if isinstance(op, ReductionOpInfo) and op.supports_sparse
        ]
        common_methods_invocations.shape_funcs = [
            op
            for op in common_methods_invocations.ops_and_refs
            if isinstance(op, ShapeFuncInfo)
        ]
        common_methods_invocations.reduction_ops = [
            op
            for op in common_methods_invocations.ops_and_refs
            if isinstance(op, ReductionOpInfo)
        ]
        common_methods_invocations.reference_filtered_ops = [
            op for op in common_methods_invocations.reduction_ops if op.ref is not None
        ]
        common_methods_invocations.reference_masked_ops = [
            op
            for op in common_methods_invocations.reference_filtered_ops
            if op.name.startswith("masked.")
        ]
        common_methods_invocations.sparse_masked_reduction_ops = [
            op
            for op in common_methods_invocations.sparse_reduction_ops
            if op.name.startswith("masked.")
        ]
        common_cuda.TEST_CUDA = True
        common_cuda.TEST_CUDNN = True
        common_cuda.TEST_CUDNN_VERSION = 0
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


def launch_test(test_case, skip_list=None, exe_list=None):
    os.environ["PYTORCH_ENABLE_XPU_FALLBACK"]="1"
    os.environ["PYTORCH_TEST_WITH_SLOW"]="1"
    if skip_list != None:
        skip_options = "not " + skip_list[0]
        for skip_case in skip_list[1:]:
            skip_option = " and not " + skip_case
            skip_options += skip_option
        if test_case == "test_dataloader_xpu.py":
            test_command = ["-k", skip_options, "-n", "1", test_case, "-v"]
        else:
            test_command = ["-k", skip_options, test_case, "-v"]
    elif exe_list != None:
        exe_options = exe_list[0]
        for exe_case in exe_list[1:]:
            exe_option = " or " + exe_case
            exe_options += exe_option
        if test_case == "test_dataloader_xpu.py":
            test_command = ["-k", exe_options, "-n", "1", test_case, "-v"]
        else:
            test_command = ["-k", exe_options, test_case, "-v"]
    else:
        if test_case == "test_dataloader_xpu.py":
            test_command = ["-n", "1", test_case, "-v"]
        else:
            test_command = [test_case, "-v"]
    return pytest.main(test_command)
