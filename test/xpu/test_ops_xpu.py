# Owner(s): ["module: intel"]


import torch
import torch._prims as prims
import torch.utils._pytree as pytree
import warnings
from torch.testing._internal.common_device_type import instantiate_device_type_tests, onlyXPU, OpDTypes, ops
from torch.testing._internal.common_utils import run_tests, slowTest, suppress_warnings

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport
with XPUPatchForImport(False):
    from test_ops import (
        _ops_and_refs_with_no_numpy_ref,
        TestCommon,
        TestMathBits,
    )

    # Tests that the cpu and gpu results are consistent
    # We add the logs for the test results to help debug, will remove them after the test is stable
    @onlyXPU
    @suppress_warnings
    @slowTest
    @ops(_ops_and_refs_with_no_numpy_ref, dtypes=OpDTypes.any_common_cpu_cuda_one)
    def _compare_cpu(self, device, dtype, op):
        def to_cpu(arg):
            if isinstance(arg, torch.Tensor):
                return arg.to(device="cpu")
            return arg
        samples = op.reference_inputs(device, dtype)
        for sample in samples:
            cpu_sample = sample.transform(to_cpu)
            cuda_results = op(sample.input, *sample.args, **sample.kwargs)
            cpu_results = op(cpu_sample.input, *cpu_sample.args, **cpu_sample.kwargs)
            # output_process_fn_grad has a very unfortunate name
            # We use this function in linalg extensively to postprocess the inputs of functions
            # that are not completely well-defined. Think svd and muliplying the singular vectors by -1.
            # CPU and CUDA implementations of the SVD can return valid SVDs that are different.
            # We use this function to compare them.
            cuda_results = sample.output_process_fn_grad(cuda_results)
            cpu_results = cpu_sample.output_process_fn_grad(cpu_results)
            # Lower tolerance because we are running this as a `@slowTest`
            # Don't want the periodic tests to fail frequently
            try:
                self.assertEqual(cuda_results, cpu_results, atol=1e-3, rtol=1e-3)
            except AssertionError as e:
                raise AssertionError(f"Failed with {sample.input}, {e} \
                                     \nthe results are {cuda_results} \nthe expect results are {cpu_results}.")

    # We add the logs for the test results to help debug, will remove them after the test is stable
    def _ref_test_helper(
        self,
        ctx,
        device,
        dtype,
        op,
        skip_zero_numel=False,
        skip_zero_dim=False,
        skip_bfloat=False,
        skip_view_consistency=False,
    ):
        # NOTE: this test works by comparing the reference
        ex = None
        for sample in op.reference_inputs(device, dtype, requires_grad=False):
            if (
                isinstance(sample.input, torch.Tensor)
                and sample.input.numel() == 0
                and skip_zero_numel
            ):
                continue
            if (
                isinstance(sample.input, torch.Tensor)
                and sample.input.ndim == 0
                and skip_zero_dim
            ):
                continue

            if skip_bfloat and (
                (
                    isinstance(sample.input, torch.Tensor)
                    and sample.input.dtype == torch.bfloat16
                )
                or any(
                    isinstance(arg, torch.Tensor) and arg.dtype == torch.bfloat16
                    for arg in sample.args
                )
            ):
                continue
            with ctx():
                ref_result = op(sample.input, *sample.args, **sample.kwargs)
            torch_result = op.torch_opinfo(sample.input, *sample.args, **sample.kwargs)

            for a, b in zip(
                pytree.tree_leaves(ref_result), pytree.tree_leaves(torch_result)
            ):
                if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
                    prims.utils.compare_tensor_meta(a, b)
                    if (
                        getattr(op, "validate_view_consistency", True)
                        and not skip_view_consistency
                    ):
                        msg = (
                            f"The torch implementation {'returns' if b._is_view() else 'does not return'} "
                            f"a view, while the reference {'does' if a._is_view() else 'does not'}"
                        )
                        try:
                            self.assertEqual(a._is_view(), b._is_view(), msg)
                        except AssertionError as e:
                            raise AssertionError(f"Failed with {sample.input}, {e} \
                                                 \nthe results are {b} \nthe expect results are {a}.")

            # Computes the dtype the more precise computatino would occur in
            precise_dtype = torch.bool
            if prims.utils.is_integer_dtype(dtype):
                # Note: bool and integer dtypes do not have more
                # precise dtypes -- they simply must be close
                precise_dtype = dtype
            if prims.utils.is_float_dtype(dtype):
                precise_dtype = torch.double
            if prims.utils.is_complex_dtype(dtype):
                precise_dtype = torch.cdouble

            # Checks if the results are close
            try:
                self.assertEqual(
                    ref_result,
                    torch_result,
                    exact_stride=False,
                    exact_device=True,
                    exact_layout=True,
                    exact_is_coalesced=True,
                )
            except AssertionError as e:
                # Raises the error if the precise dtype comparison wouldn't be
                # different
                if dtype is precise_dtype:
                    raise AssertionError(f"Failed with {sample.input}, {e} \
                                                 \nthe results are {torch_result} \nthe expect results are {ref_result}.")

                ex = e

            # Goes to next sample if these results are close
            if not ex:
                continue

            # If the results are not close, checks that the
            # reference is more accurate than the torch op
            def _make_precise(x):
                if isinstance(x, torch.dtype):
                    return precise_dtype
                if isinstance(x, torch.Tensor) and x.dtype is dtype:
                    return x.to(precise_dtype)
                return x

            precise_sample = sample.transform(_make_precise)
            precise_result = op.torch_opinfo(
                precise_sample.input, *precise_sample.args, **precise_sample.kwargs
            )

            def _distance(a, b):
                # Special-cases boolean comparisons
                if prims.utils.is_boolean_dtype(a.dtype):
                    assert b.dtype is torch.bool
                    return (a ^ b).sum()

                same = a == b
                if prims.utils.is_float_dtype(a.dtype) or prims.utils.is_complex_dtype(
                    a.dtype
                ):
                    same = torch.logical_or(
                        same, torch.logical_and(torch.isnan(a), torch.isnan(b))
                    )

                actual_error = torch.where(same, 0, torch.abs(a - b)).sum()
                return actual_error

            ref_distance = 0
            for a, b in zip(
                pytree.tree_leaves(ref_result), pytree.tree_leaves(precise_result)
            ):
                ref_distance = ref_distance + _distance(a, b)

            torch_distance = 0
            for a, b in zip(
                pytree.tree_leaves(torch_result), pytree.tree_leaves(precise_result)
            ):
                torch_distance = torch_distance + _distance(a, b)

            # TODO: consider adding some tolerance to this comparison
            msg = (
                f"Reference result was farther ({ref_distance}) from the precise "
                f"computation than the torch result was ({torch_distance})!"
            )
            try:
                self.assertTrue(ref_distance <= torch_distance, msg=msg)
            except AssertionError as e:
                raise AssertionError(f"Failed with {sample.input}, {e} \
                                     \nthe results are {torch_result} \nthe expect results are {precise_result}.")

        # Reports numerical accuracy discrepancies
        if ex is not None:
            msg = "Test passed because the reference was more accurate than the torch operator."
            warnings.warn(msg)

    TestCommon.test_compare_cpu = _compare_cpu
    TestCommon._ref_test_helper = _ref_test_helper

instantiate_device_type_tests(TestCommon, globals(), only_for="xpu", allow_xpu=True)
instantiate_device_type_tests(TestMathBits, globals(), only_for="xpu", allow_xpu=True)
# in finegrand
# instantiate_device_type_tests(TestCompositeCompliance, globals(), only_for="xpu", allow_xpu=True)
# only CPU
# instantiate_device_type_tests(TestRefsOpsInfo, globals(), only_for="xpu", allow_xpu=True)
# not important
# instantiate_device_type_tests(TestFakeTensor, globals(), only_for="xpu", allow_xpu=True)
# instantiate_device_type_tests(TestTags, globals(), only_for="xpu", allow_xpu=True)


if __name__ == "__main__":
    run_tests()
