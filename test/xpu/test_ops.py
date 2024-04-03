# Owner(s): ["module: intel"]

import sys
import unittest
from collections.abc import Sequence

import torch
from torch.testing._internal.common_dtype import (
    floating_and_complex_types_and,
    all_types_and_complex_and,
    integral_types_and,
)
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyXPU,
    OpDTypes,
    ops,
)
from torch.testing._internal.common_methods_invocations import (
    op_db,
    UnaryUfuncInfo,
    ReductionOpInfo,
    SpectralFuncInfo,
    ops_and_refs,
    BinaryUfuncInfo,
    python_ref_db,
)
from torch.testing._internal.common_utils import (
    NoTest,
    run_tests,
    set_default_dtype,
    suppress_warnings,
    TEST_WITH_TORCHINDUCTOR,
    TEST_WITH_UBSAN,
    TEST_XPU,
    TestCase,
)

# Get names of all the operators which have ref in their entry in OpInfo (testing infra)
#   except for elementwise unary operators (separately implemented in test/test_unary_ufuncs.py),
#   elementwise binary operators (separately implemented in test_binary_ufuncs.py),
#   reduction operations (separately impelemented in test_reductions.py),
#   and Spectral Functions (separately implemented for only 1D as of now, in test/test_spectral_ops.py)
_ref_test_ops = tuple(
    filter(
        lambda op: not isinstance(
            op, (UnaryUfuncInfo, ReductionOpInfo, SpectralFuncInfo, BinaryUfuncInfo)
        )
        and op.ref is not None,
        op_db,
    )
)

if not TEST_XPU:
    print("XPU not available, skipping tests", file=sys.stderr)
    TestCase = NoTest  # noqa: F811

any_common_cpu_xpu_one = OpDTypes.any_common_cpu_cuda_one
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
    "rsub",
    "remainder",
    "fmod",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "sin",
    "cos",
    "log",
    "sqrt",
    "rsqrt",
    "tanh",
    "neg",
    "reciprocal",
    "pow",
    "unfold",
    "nn.functional.threshold",
    "nn.functional.relu",
    "nn.functional.gelu",
    "arange",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_not",
    "where",
    "clamp_min",
    "clamp_max",
    "clamp",
    "masked_fill",
]
_xpu_tensor_factory_op_list = [
    "normal",
    "uniform",
    "as_strided",
    "empty",
    "empty_strided",
]
_xpu_all_op_list = _xpu_computation_op_list + _xpu_tensor_factory_op_list
_xpu_all_ops = [op for op in ops_and_refs if op.name in _xpu_all_op_list]

# Exclusive ops list
_xpu_not_test_dtype_op_list = [
    "resize_",  # Skipped by CPU
    "resize_as_",  # Skipped by CPU
]
_xpu_float_only_op_list = [
    "reciprocal", # Align with CUDA impl. Only float and complex supported in CUDA native.
]
_xpu_not_test_numpy_ref_op_list = [
    "clone", # Numpy copy cannot pass memory_format.
]

# test_numpy_ref
_xpu_numpy_ref_op_list = _xpu_computation_op_list.copy()
for op in _xpu_not_test_numpy_ref_op_list:
    _xpu_numpy_ref_op_list.remove(op)
_xpu_numpy_ref_ops = [
    op for op in _ref_test_ops if op.name in _xpu_numpy_ref_op_list
]

# test_compare_cpu
_xpu_computation_ops = [
    op for op in ops_and_refs if op.name in _xpu_computation_op_list
]

# test_non_standard_bool_values
_xpu_non_standard_bool_values_op_list = _xpu_computation_op_list.copy()
for op in _xpu_float_only_op_list:
    _xpu_non_standard_bool_values_op_list.remove(op)
_xpu_non_standard_bool_values_ops = [op for op in ops_and_refs if op.name in _xpu_non_standard_bool_values_op_list]

# test_dtypes
_xpu_dtype_op_list = _xpu_all_op_list.copy()
for op in _xpu_not_test_dtype_op_list:
    _xpu_dtype_op_list.remove(op)
for op in _xpu_float_only_op_list:
    _xpu_dtype_op_list.remove(op)
_xpu_dtype_ops = [op for op in ops_and_refs if op.name in _xpu_dtype_op_list]

# test_promotes_int_to_float
_xpu_promotes_int_to_float_list = _xpu_all_op_list.copy()
for op in _xpu_float_only_op_list:
    _xpu_promotes_int_to_float_list.remove(op)
_xpu_promotes_int_to_float_ops = [op for op in ops_and_refs if op.name in _xpu_promotes_int_to_float_list]


class TestXpu(TestCase):

    # Tests that the function and its (ndarray-accepting) reference produce the same
    #   values on the tensors from sample_inputs func for the corresponding op.
    # This test runs in double and complex double precision because
    # NumPy does computation internally using double precision for many functions
    # resulting in possible equality check failures.
    @onlyXPU
    @suppress_warnings
    @ops(_xpu_numpy_ref_ops, allowed_dtypes=(torch.float64, torch.long, torch.complex128))
    def test_numpy_ref(self, device, dtype, op):
        if (
            TEST_WITH_TORCHINDUCTOR and
            op.formatted_name in ('signal_windows_exponential', 'signal_windows_bartlett') and
            dtype == torch.float64 and 'cuda' in device
           ):   # noqa: E121
            raise unittest.SkipTest("XXX: raises tensor-likes are not close.")

        # Sets the default dtype to NumPy's default dtype of double
        with set_default_dtype(torch.double):
            for sample_input in op.reference_inputs(device, dtype):
                print(sample_input)
                self.compare_with_reference(
                    op, op.ref, sample_input, exact_dtype=(dtype is not torch.long)
                )

    @onlyXPU
    @suppress_warnings
    @ops(_xpu_computation_ops, dtypes=any_common_cpu_xpu_one)
    def test_compare_cpu(self, device, dtype, op):
        def to_cpu(arg):
            if isinstance(arg, torch.Tensor):
                return arg.to(device="cpu")
            return arg

        samples = op.reference_inputs(device, dtype)

        for sample in samples:
            cpu_sample = sample.transform(to_cpu)
            xpu_results = op(sample.input, *sample.args, **sample.kwargs)
            cpu_results = op(cpu_sample.input, *cpu_sample.args, **cpu_sample.kwargs)

            xpu_results = sample.output_process_fn_grad(xpu_results)
            cpu_results = cpu_sample.output_process_fn_grad(cpu_results)

            # Lower tolerance because we are running this as a `@slowTest`
            # Don't want the periodic tests to fail frequently
            self.assertEqual(xpu_results, cpu_results, atol=1e-4, rtol=1e-4)

    @onlyXPU
    @ops(_xpu_non_standard_bool_values_ops, allowed_dtypes=(torch.bool,))
    @unittest.skipIf(TEST_WITH_UBSAN, "Test uses undefined behavior")
    def test_non_standard_bool_values(self, device, dtype, op):
        # Test boolean values other than 0x00 and 0x01 (gh-54789)
        def convert_boolean_tensors(x):
            if not isinstance(x, torch.Tensor) or x.dtype != torch.bool:
                return x

            # Map False -> 0 and True -> Random value in [2, 255]
            true_vals = torch.randint(
                2, 255, x.shape, dtype=torch.uint8, device=x.device
            )
            false_vals = torch.zeros((), dtype=torch.uint8, device=x.device)
            x_int = torch.where(x, true_vals, false_vals)

            ret = x_int.view(torch.bool)
            self.assertEqual(ret, x)
            return ret

        for sample in op.sample_inputs(device, dtype):
            expect = op(sample.input, *sample.args, **sample.kwargs)

            transformed = sample.transform(convert_boolean_tensors)
            actual = op(transformed.input, *transformed.args, **transformed.kwargs)

            self.assertEqual(expect, actual)

    @onlyXPU
    @ops(_xpu_dtype_ops, dtypes=OpDTypes.none)
    def test_dtypes(self, device, op):
        # Check complex32 support only if the op claims.
        # TODO: Once the complex32 support is better, we should add check for complex32 unconditionally.
        device_type = torch.device(device).type
        include_complex32 = (
            (torch.complex32,)
            if op.supports_dtype(torch.complex32, device_type)
            else ()
        )

        # dtypes to try to backward in
        allowed_backward_dtypes = floating_and_complex_types_and(
            *((torch.half, torch.bfloat16) + include_complex32)
        )

        # lists for (un)supported dtypes
        supported_dtypes = set()
        unsupported_dtypes = set()
        supported_backward_dtypes = set()
        unsupported_backward_dtypes = set()
        dtype_error: Dict[torch.dtype, Exception] = dict()

        def unsupported(dtype, e):
            dtype_error[dtype] = e
            unsupported_dtypes.add(dtype)
            if dtype in allowed_backward_dtypes:
                unsupported_backward_dtypes.add(dtype)

        for dtype in all_types_and_complex_and(
            *((torch.half, torch.bfloat16, torch.bool) + include_complex32)
        ):
            # tries to acquire samples - failure indicates lack of support
            requires_grad = dtype in allowed_backward_dtypes
            try:
                samples = tuple(
                    op.sample_inputs(device, dtype, requires_grad=requires_grad)
                )
            except Exception as e:
                unsupported(dtype, e)
                continue

            for sample in samples:
                # tries to call operator with the sample - failure indicates
                #   lack of support
                try:
                    result = op(sample.input, *sample.args, **sample.kwargs)
                    supported_dtypes.add(dtype)
                except Exception as e:
                    # NOTE: some ops will fail in forward if their inputs
                    #   require grad but they don't support computing the gradient
                    #   in that type! This is a bug in the op!
                    unsupported(dtype, e)
                    continue

                # Checks for backward support in the same dtype, if the input has
                # one or more tensors requiring grad
                def _tensor_requires_grad(x):
                    if isinstance(x, dict):
                        for v in x.values():
                            if _tensor_requires_grad(v):
                                return True
                    if isinstance(x, (list, tuple)):
                        for a in x:
                            if _tensor_requires_grad(a):
                                return True
                    if isinstance(x, torch.Tensor) and x.requires_grad:
                        return True

                    return False

                requires_grad = (
                    _tensor_requires_grad(sample.input)
                    or _tensor_requires_grad(sample.args)
                    or _tensor_requires_grad(sample.kwargs)
                )
                if not requires_grad:
                    continue

                try:
                    result = sample.output_process_fn_grad(result)
                    if isinstance(result, torch.Tensor):
                        backward_tensor = result
                    elif isinstance(result, Sequence) and isinstance(
                        result[0], torch.Tensor
                    ):
                        backward_tensor = result[0]
                    else:
                        continue

                    # Note: this grad may not have the same dtype as dtype
                    # For functions like complex (float -> complex) or abs
                    #   (complex -> float) the grad tensor will have a
                    #   different dtype than the input.
                    #   For simplicity, this is still modeled as these ops
                    #   supporting grad in the input dtype.
                    grad = torch.randn_like(backward_tensor)
                    backward_tensor.backward(grad)
                    supported_backward_dtypes.add(dtype)
                except Exception as e:
                    dtype_error[dtype] = e
                    unsupported_backward_dtypes.add(dtype)

        # Checks that dtypes are listed correctly and generates an informative
        #   error message

        supported_forward = supported_dtypes - unsupported_dtypes
        partially_supported_forward = supported_dtypes & unsupported_dtypes
        unsupported_forward = unsupported_dtypes - supported_dtypes
        supported_backward = supported_backward_dtypes - unsupported_backward_dtypes
        partially_supported_backward = (
            supported_backward_dtypes & unsupported_backward_dtypes
        )
        unsupported_backward = unsupported_backward_dtypes - supported_backward_dtypes

        device_type = torch.device(device).type

        claimed_forward = set(op.supported_dtypes(device_type))
        supported_but_unclaimed_forward = supported_forward - claimed_forward
        claimed_but_unsupported_forward = claimed_forward & unsupported_forward

        claimed_backward = set(op.supported_backward_dtypes(device_type))
        supported_but_unclaimed_backward = supported_backward - claimed_backward
        claimed_but_unsupported_backward = claimed_backward & unsupported_backward

        # Partially supporting a dtype is not an error, but we print a warning
        if (len(partially_supported_forward) + len(partially_supported_backward)) > 0:
            msg = f"Some dtypes for {op.name} on device type {device_type} are only partially supported!\n"
            if len(partially_supported_forward) > 0:
                msg = (
                    msg
                    + "The following dtypes only worked on some samples during forward: {}.\n".format(
                        partially_supported_forward
                    )
                )
            if len(partially_supported_backward) > 0:
                msg = (
                    msg
                    + "The following dtypes only worked on some samples during backward: {}.\n".format(
                        partially_supported_backward
                    )
                )
            print(msg)

        if (
            len(supported_but_unclaimed_forward)
            + len(claimed_but_unsupported_forward)
            + len(supported_but_unclaimed_backward)
            + len(claimed_but_unsupported_backward)
        ) == 0:
            return

        # Reference operators often support additional dtypes, and that's OK
        if op in python_ref_db:
            if (
                len(claimed_but_unsupported_forward)
                + len(claimed_but_unsupported_backward)
            ) == 0:
                return

        # Generates error msg
        msg = f"The supported dtypes for {op.name} on device type {device_type} are incorrect!\n"
        if len(supported_but_unclaimed_forward) > 0:
            msg = (
                msg
                + "The following dtypes worked in forward but are not listed by the OpInfo: {}.\n".format(
                    supported_but_unclaimed_forward
                )
            )
        if len(supported_but_unclaimed_backward) > 0:
            msg = (
                msg
                + "The following dtypes worked in backward but are not listed by the OpInfo: {}.\n".format(
                    supported_but_unclaimed_backward
                )
            )
        if len(claimed_but_unsupported_forward) > 0:
            msg = (
                msg
                + "The following dtypes did not work in forward but are listed by the OpInfo: {}.\n".format(
                    claimed_but_unsupported_forward
                )
            )
        if len(claimed_but_unsupported_backward) > 0:
            msg = (
                msg
                + "The following dtypes did not work in backward but are listed by the OpInfo: {}.\n".format(
                    claimed_but_unsupported_backward
                )
            )

        all_claimed_but_unsupported = set.union(
            claimed_but_unsupported_backward, claimed_but_unsupported_forward
        )
        if all_claimed_but_unsupported:
            msg += "Unexpected failures raised the following errors:\n"
            for dtype in all_claimed_but_unsupported:
                msg += f"{dtype} - {dtype_error[dtype]}\n"

        self.fail(msg)

    # Validates that each OpInfo that sets promotes_int_to_float=True does as it says
    @onlyXPU
    @ops(
        (op for op in _xpu_promotes_int_to_float_ops if op.promotes_int_to_float),
        allowed_dtypes=integral_types_and(torch.bool),
    )
    def test_promotes_int_to_float(self, device, dtype, op):
        for sample in op.sample_inputs(device, dtype):
            output = op(sample.input, *sample.args, **sample.kwargs)
            if not output.dtype.is_floating_point:
                self.fail(
                    f"The OpInfo sets `promotes_int_to_float=True`, but {dtype} was promoted to {output.dtype}."
                )


instantiate_device_type_tests(TestXpu, globals(), only_for="xpu")


if __name__ == "__main__":
    run_tests()
