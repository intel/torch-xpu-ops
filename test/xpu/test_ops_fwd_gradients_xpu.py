# Owner(s): ["module: intel"]

from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_ops_fwd_gradients import TestFwdGradients
TestFwdGradients._default_dtype_check_enabled = True
instantiate_device_type_tests(
    TestFwdGradients, globals(), only_for=("xpu"), allow_xpu=True
)
if __name__ == "__main__":
    TestCase._default_dtype_check_enabled = True
    run_tests()
