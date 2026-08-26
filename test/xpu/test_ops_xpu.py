# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]


import unittest

from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests

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


# Skip XPU op tests that currently fail in p0; tracked upstream.
# Each entry maps an exact generated test name to its tracking issue.
_xpu_skip_cases = {
    "TestFakeTensorXPU": {
        "test_fake_crossref_backward_amp_fft_hfft2_xpu_float32": "https://github.com/intel/torch-xpu-ops/issues/4994",
        "test_fake_crossref_backward_amp_fft_hfftn_xpu_float32": "https://github.com/intel/torch-xpu-ops/issues/4994",
        "test_fake_crossref_backward_amp_fft_irfft2_xpu_float32": "https://github.com/intel/torch-xpu-ops/issues/4994",
        "test_fake_crossref_backward_amp_fft_irfftn_xpu_float32": "https://github.com/intel/torch-xpu-ops/issues/4994",
        "test_fake_crossref_backward_amp_nn_functional_bilinear_xpu_float32": "https://github.com/intel/torch-xpu-ops/issues/2333",
        "test_fake_crossref_backward_amp_nn_functional_max_unpool2d_grad_xpu_float32": "https://github.com/intel/torch-xpu-ops/issues/4057",
        "test_fake_crossref_backward_amp_nn_functional_max_unpool2d_xpu_float32": "https://github.com/intel/torch-xpu-ops/issues/4057",
        "test_fake_crossref_backward_amp_torch_ops_aten__efficient_attention_forward_xpu_float32": "https://github.com/intel/torch-xpu-ops/issues/2285",
        "test_fake_crossref_backward_no_amp_fft_hfft2_xpu_float32": "https://github.com/intel/torch-xpu-ops/issues/4995",
        "test_fake_crossref_backward_no_amp_fft_hfftn_xpu_float32": "https://github.com/intel/torch-xpu-ops/issues/4995",
        "test_fake_crossref_backward_no_amp_fft_irfft2_xpu_float32": "https://github.com/intel/torch-xpu-ops/issues/4995",
        "test_fake_crossref_backward_no_amp_fft_irfftn_xpu_float32": "https://github.com/intel/torch-xpu-ops/issues/4995",
        "test_fake_crossref_backward_no_amp_nn_functional_max_unpool2d_grad_xpu_float32": "https://github.com/intel/torch-xpu-ops/issues/4057",
        "test_fake_crossref_backward_no_amp_nn_functional_max_unpool2d_xpu_float32": "https://github.com/intel/torch-xpu-ops/issues/4057",
        "test_fake_crossref_backward_no_amp_torch_ops_aten__efficient_attention_forward_xpu_float32": "https://github.com/intel/torch-xpu-ops/issues/2285",
    },
    "TestCommonXPU": {
        "test_dtypes_addmm_decomposed_xpu": "https://github.com/intel/torch-xpu-ops/issues/2253",
        "test_dtypes_addmm_xpu": "https://github.com/intel/torch-xpu-ops/issues/2253",
        "test_dtypes_addmv_xpu": "https://github.com/intel/torch-xpu-ops/issues/2253",
        "test_dtypes_baddbmm_xpu": "https://github.com/intel/torch-xpu-ops/issues/2253",
        "test_dtypes_histogram_xpu": "https://github.com/intel/torch-xpu-ops/issues/2253",
        "test_dtypes_histogramdd_xpu": "https://github.com/intel/torch-xpu-ops/issues/2253",
        "test_dtypes_linalg_multi_dot_xpu": "https://github.com/intel/torch-xpu-ops/issues/2333",
        "test_dtypes_mv_xpu": "https://github.com/intel/torch-xpu-ops/issues/2253",
        "test_dtypes_nn_functional_conv1d_xpu": "https://github.com/intel/torch-xpu-ops/issues/2253",
        "test_dtypes_nn_functional_conv2d_xpu": "https://github.com/intel/torch-xpu-ops/issues/2253",
        "test_dtypes_nn_functional_conv3d_xpu": "https://github.com/intel/torch-xpu-ops/issues/2253",
        "test_dtypes_nn_functional_conv_transpose1d_xpu": "https://github.com/intel/torch-xpu-ops/issues/2482",
        "test_dtypes_nn_functional_conv_transpose2d_xpu": "https://github.com/intel/torch-xpu-ops/issues/3574",
        "test_dtypes_nn_functional_conv_transpose3d_xpu": "https://github.com/intel/torch-xpu-ops/issues/3574",
        "test_dtypes_torch_ops_aten__efficient_attention_forward_xpu": "https://github.com/intel/torch-xpu-ops/issues/4995",
        "test_dtypes_torch_ops_aten__flash_attention_forward_xpu": "https://github.com/intel/torch-xpu-ops/issues/4995",
        "test_errors_arange_xpu": "https://github.com/intel/torch-xpu-ops/issues/4106",
        "test_noncontiguous_samples_histogramdd_xpu_float32": "https://github.com/intel/torch-xpu-ops/issues/2254",
        "test_out_cholesky_inverse_xpu_float32": "https://github.com/intel/torch-xpu-ops/issues/1951",
        "test_out_geqrf_xpu_float32": "https://github.com/intel/torch-xpu-ops/issues/1951",
        "test_out_histc_xpu_float32": "https://github.com/intel/torch-xpu-ops/issues/2234",
        "test_out_histogramdd_xpu_float32": "https://github.com/intel/torch-xpu-ops/issues/2254",
        "test_out_ormqr_xpu_float32": "https://github.com/intel/torch-xpu-ops/issues/1951",
        "test_out_torch_ops_aten__efficient_attention_forward_xpu_float32": "https://github.com/intel/torch-xpu-ops/issues/2285",
        "test_out_torch_ops_aten__flash_attention_forward_xpu_float16": "(no tracking issue)",
        "test_out_triangular_solve_xpu_float32": "https://github.com/intel/torch-xpu-ops/issues/2167",
        "test_out_warning_histogramdd_xpu": "https://github.com/intel/torch-xpu-ops/issues/2254",
        "test_out_warning_torch_ops_aten__efficient_attention_forward_xpu": "https://github.com/intel/torch-xpu-ops/issues/2285",
        "test_out_warning_torch_ops_aten__flash_attention_forward_xpu": "https://github.com/intel/torch-xpu-ops/issues/2442",
        "test_python_ref__refs_true_divide_xpu_complex32": "https://github.com/intel/torch-xpu-ops/issues/2301",
        "test_python_ref_executor__refs_fft_hfft2_executor_aten_xpu_complex32": "https://github.com/intel/torch-xpu-ops/issues/4994",
        "test_python_ref_executor__refs_fft_hfft2_executor_aten_xpu_float16": "https://github.com/intel/torch-xpu-ops/issues/4268",
        "test_variant_consistency_eager_histogramdd_xpu_float32": "https://github.com/intel/torch-xpu-ops/issues/2254",
        "test_variant_consistency_eager_torch_ops_aten__efficient_attention_forward_xpu_float32": "https://github.com/intel/torch-xpu-ops/issues/2285",
    },
    "TestCompositeComplianceXPU": {
        "test_cow_input_histogramdd_xpu_float32": "https://github.com/intel/torch-xpu-ops/issues/2249",
        "test_cow_input_nn_functional_conv1d_xpu_float32": "https://github.com/intel/torch-xpu-ops/issues/2248",
        "test_cow_input_nn_functional_conv2d_xpu_float32": "https://github.com/intel/torch-xpu-ops/issues/2248",
        "test_cow_input_nn_functional_conv3d_xpu_float32": "https://github.com/intel/torch-xpu-ops/issues/2248",
        "test_cow_input_nn_functional_conv_transpose1d_xpu_float32": "https://github.com/intel/torch-xpu-ops/issues/2248",
        "test_cow_input_nn_functional_conv_transpose2d_xpu_float32": "https://github.com/intel/torch-xpu-ops/issues/2248",
        "test_cow_input_nn_functional_conv_transpose3d_xpu_float32": "https://github.com/intel/torch-xpu-ops/issues/2248",
        "test_cow_input_torch_ops_aten__efficient_attention_forward_xpu_float32": "https://github.com/intel/torch-xpu-ops/issues/2285",
        "test_operator_histogramdd_xpu_float32": "https://github.com/intel/torch-xpu-ops/issues/2249",
        "test_view_replay_histogramdd_xpu_float32": "https://github.com/intel/torch-xpu-ops/issues/2249",
        "test_view_replay_torch_ops_aten__efficient_attention_forward_xpu_float32": "https://github.com/intel/torch-xpu-ops/issues/2285",
    },
    "TestMathBitsXPU": {
        "test_neg_view_histogramdd_xpu_float64": "https://github.com/intel/torch-xpu-ops/issues/2249",
    },
}


def _apply_xpu_skips(_skip_cases):
    for _cls_name, _cases in _skip_cases.items():
        _cls = globals().get(_cls_name)
        if _cls is None:
            continue
        for _name, _issue in _cases.items():
            _method = getattr(_cls, _name, None)
            if _method is not None:
                setattr(
                    _cls,
                    _name,
                    unittest.skip(f"Skipped on XPU, see {_issue}")(_method),
                )


_apply_xpu_skips(_xpu_skip_cases)


if __name__ == "__main__":
    run_tests()
