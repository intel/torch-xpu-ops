# Owner(s): ["module: intel"]

import torch
import torch.testing._internal.hypothesis_utils as hu

from hypothesis import given, settings
from hypothesis import strategies as st

from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    import sys
    import os
    script_path = os.path.split(__file__)[0]
    sys.path.insert(0, os.path.realpath(os.path.join(script_path, "../..")))
    from xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_workflow_module import TestDistributed, TestFakeQuantize, TestFusedObsFakeQuantModule, TestHistogramObserver, TestObserver, TestRecordHistogramObserver


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


TestFakeQuantize.test_fq_module_per_channel = rewrap_hypothesis_test(TestFakeQuantize.test_fq_module_per_channel, extra_given_kwargs={
    "X": hu.per_channel_tensor(shapes=hu.array_shapes(2, 5,), qparams=hu.qparams(dtypes=torch.qint8))
})
TestFusedObsFakeQuantModule.test_compare_fused_obs_fq_oss_module = rewrap_hypothesis_test(TestFusedObsFakeQuantModule.test_compare_fused_obs_fq_oss_module)
TestFusedObsFakeQuantModule.test_fused_obs_fq_module = rewrap_hypothesis_test(TestFusedObsFakeQuantModule.test_fused_obs_fq_module)
TestFusedObsFakeQuantModule.test_fused_obs_fq_moving_avg_module = rewrap_hypothesis_test(TestFusedObsFakeQuantModule.test_fused_obs_fq_moving_avg_module)

instantiate_device_type_tests(TestDistributed, globals(), only_for="xpu", allow_xpu=True)
instantiate_device_type_tests(TestFakeQuantize, globals(), only_for="xpu", allow_xpu=True)
instantiate_device_type_tests(TestFusedObsFakeQuantModule, globals(), only_for="xpu", allow_xpu=True)
instantiate_device_type_tests(TestHistogramObserver, globals(), only_for="xpu", allow_xpu=True)
instantiate_device_type_tests(TestObserver, globals(), only_for="xpu", allow_xpu=True)
instantiate_device_type_tests(TestRecordHistogramObserver, globals(), only_for="xpu", allow_xpu=True)

if __name__ == "__main__":
    TestCase._default_dtype_check_enabled = True
    run_tests()
