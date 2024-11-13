# Owner(s): ["module: intel"]

import torch
import torch.testing._internal.hypothesis_utils as hu

import unittest

from hypothesis import given, settings
from hypothesis import strategies as st

from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

from torch.testing._internal.common_quantized import (
    to_tensor
)

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    import sys
    import os
    script_path = os.path.split(__file__)[0]
    sys.path.insert(0, os.path.realpath(os.path.join(script_path, "../..")))
    from xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_workflow_ops import TestFusedObsFakeQuant, TestFakeQuantizeOps
    from test_workflow_ops import NP_RANDOM_SEED
    from torch.testing._internal.common_cuda import TEST_CUDA


@unittest.skipIf(not TEST_CUDA, "No gpu is not available.")
def _test_forward_per_tensor_cachemask_cuda(self):
    device = torch.device('xpu')
    self._test_forward_per_tensor_cachemask_impl(device)


@unittest.skipIf(not TEST_CUDA, "No gpu is not available.")
def _test_backward_per_tensor_cachemask_cuda(self):
    device = torch.device('xpu')
    self._test_backward_per_tensor_cachemask_impl(device)


@given(X=hu.tensor(shapes=hu.array_shapes(1, 5,),
                    elements=hu.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
                    qparams=hu.qparams(dtypes=torch.quint8)))
@unittest.skipIf(not TEST_CUDA, "No gpu is not available.")
def _test_learnable_forward_per_tensor_cuda(self, X):
    X, (_, _, _) = X
    scale_base = torch.normal(mean=0, std=1, size=(1,)).clamp(1e-4, 100)
    zero_point_base = torch.normal(mean=0, std=128, size=(1,))
    self._test_learnable_forward_per_tensor(
        X, 'xpu', scale_base, zero_point_base)


@given(X=hu.tensor(shapes=hu.array_shapes(1, 5,),
                    elements=hu.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
                    qparams=hu.qparams(dtypes=torch.quint8)))
@unittest.skipIf(not TEST_CUDA, "No gpu is not available.")
def _test_learnable_backward_per_tensor_cuda(self, X):
    torch.random.manual_seed(NP_RANDOM_SEED)
    X, (_, _, _) = X
    scale_base = torch.normal(mean=0, std=1, size=(1,)).clamp(1e-4, 100)
    zero_point_base = torch.normal(mean=0, std=128, size=(1,))
    self._test_learnable_backward_per_tensor(
        X, 'xpu', scale_base, zero_point_base)


@unittest.skipIf(not TEST_CUDA, "No gpu is not available.")
def _test_forward_per_channel_cachemask_cuda(self):
    self._test_forward_per_channel_cachemask_impl('xpu')


@given(X=hu.per_channel_tensor(shapes=hu.array_shapes(1, 5,),
                                qparams=hu.qparams(dtypes=torch.quint8)))
@unittest.skipIf(not TEST_CUDA, "No gpu is not available.")
def _test_learnable_forward_per_channel_cuda(self, X):
    torch.random.manual_seed(NP_RANDOM_SEED)
    X, (_, _, axis, _) = X
    X_base = torch.tensor(X).to('xpu')
    channel_size = X_base.size(axis)
    scale_base = torch.normal(mean=0, std=1, size=(channel_size,)).clamp(1e-4, 100)
    zero_point_base = torch.normal(mean=0, std=128, size=(channel_size,))
    self._test_learnable_forward_per_channel(
        X_base, 'xpu', scale_base, zero_point_base, axis)


@unittest.skipIf(not TEST_CUDA, "No gpu is not available.")
def _test_backward_per_channel_cachemask_cuda(self):
    self._test_backward_per_channel_cachemask_impl('xpu')


@given(X=hu.per_channel_tensor(shapes=hu.array_shapes(2, 5,),
                                qparams=hu.qparams(dtypes=torch.quint8)))
@unittest.skipIf(not TEST_CUDA, "No gpu is not available.")
def _test_learnable_backward_per_channel_cuda(self, X):
    torch.random.manual_seed(NP_RANDOM_SEED)
    X, (scale, zero_point, axis, torch_type) = X
    X_base = torch.tensor(X).to('xpu')
    scale_base = to_tensor(scale, 'xpu')
    zero_point_base = to_tensor(zero_point, 'xpu')
    self._test_learnable_backward_per_channel(
        X_base, 'xpu', scale_base, zero_point_base, axis)


def rewrap_hypothesis_test(test, extra_given_kwargs=None, additional_wrapper=None):
    innert_test = test.hypothesis.inner_test
    given_kwargs = {
        "device": st.sampled_from(
            ["cpu", "xpu"] if torch.xpu.is_available() else ["cpu"]
        )
    }
    if extra_given_kwargs is not None:
        for k, v in extra_given_kwargs.items():
            given_kwargs[k] = v

    if additional_wrapper is not None:
        return given(**given_kwargs)(additional_wrapper(innert_test))
    
    return given(**given_kwargs)(innert_test)


given_kwargs_dict1 = {
    "X": hu.tensor(shapes=hu.array_shapes(1, 5,), qparams=hu.qparams(dtypes=torch.quint8)),
}
given_kwargs_dict2 = {
    "X": hu.per_channel_tensor(shapes=hu.array_shapes(1, 5,), qparams=hu.qparams(dtypes=torch.quint8)),
}
given_kwargs_dict3 = {
    "symmetric_quant": st.booleans(),
}

TestFakeQuantizeOps.test_forward_per_tensor = rewrap_hypothesis_test(TestFakeQuantizeOps.test_forward_per_tensor, extra_given_kwargs=given_kwargs_dict1)
TestFakeQuantizeOps.test_backward_per_tensor = rewrap_hypothesis_test(TestFakeQuantizeOps.test_backward_per_tensor, extra_given_kwargs=given_kwargs_dict1, additional_wrapper=unittest.skip("temporarily disable the test"))
TestFakeQuantizeOps.test_fq_module_per_tensor = rewrap_hypothesis_test(TestFakeQuantizeOps.test_fq_module_per_tensor, extra_given_kwargs=given_kwargs_dict1)
TestFakeQuantizeOps.test_forward_per_channel = rewrap_hypothesis_test(TestFakeQuantizeOps.test_forward_per_channel, extra_given_kwargs=given_kwargs_dict2)
TestFakeQuantizeOps.test_backward_per_channel = rewrap_hypothesis_test(TestFakeQuantizeOps.test_backward_per_channel, extra_given_kwargs=given_kwargs_dict2, additional_wrapper=unittest.skip(
        "this is broken without changes to any relevant code, "
        "we need to remove hypothesis testing in CI"))

TestFakeQuantizeOps.test_forward_per_tensor_cachemask_cuda = _test_forward_per_tensor_cachemask_cuda
TestFakeQuantizeOps.test_backward_per_tensor_cachemask_cuda = _test_backward_per_tensor_cachemask_cuda
TestFakeQuantizeOps.test_learnable_forward_per_tensor_cuda = _test_learnable_forward_per_tensor_cuda
TestFakeQuantizeOps.test_learnable_backward_per_tensor_cuda = _test_learnable_backward_per_tensor_cuda
TestFakeQuantizeOps.test_forward_per_channel_cachemask_cuda = _test_forward_per_channel_cachemask_cuda
TestFakeQuantizeOps.test_learnable_forward_per_channel_cuda = _test_learnable_forward_per_channel_cuda
TestFakeQuantizeOps.test_backward_per_channel_cachemask_cuda = _test_backward_per_channel_cachemask_cuda
TestFakeQuantizeOps.test_learnable_backward_per_channel_cuda = _test_learnable_backward_per_channel_cuda
TestFakeQuantizeOps.test_fixed_qparams_fq_module = rewrap_hypothesis_test(TestFakeQuantizeOps.test_fixed_qparams_fq_module, extra_given_kwargs=given_kwargs_dict1)

TestFusedObsFakeQuant.test_fused_obs_fake_quant_moving_avg = rewrap_hypothesis_test(TestFusedObsFakeQuant.test_fused_obs_fake_quant_moving_avg, extra_given_kwargs=given_kwargs_dict3)
TestFusedObsFakeQuant.test_fused_obs_fake_quant_moving_avg_per_channel = rewrap_hypothesis_test(TestFusedObsFakeQuant.test_fused_obs_fake_quant_moving_avg_per_channel, extra_given_kwargs=given_kwargs_dict3)
TestFusedObsFakeQuant.test_fused_obs_fake_quant_backward_op = rewrap_hypothesis_test(TestFusedObsFakeQuant.test_fused_obs_fake_quant_backward_op)
TestFusedObsFakeQuant.test_fused_backward_op_fake_quant_off = rewrap_hypothesis_test(TestFusedObsFakeQuant.test_fused_backward_op_fake_quant_off)


instantiate_device_type_tests(TestFusedObsFakeQuant, globals(), only_for="xpu", allow_xpu=True)
instantiate_device_type_tests(TestFakeQuantizeOps, globals(), only_for="xpu", allow_xpu=True)


if __name__ == "__main__":
    TestCase._default_dtype_check_enabled = True
    run_tests()
