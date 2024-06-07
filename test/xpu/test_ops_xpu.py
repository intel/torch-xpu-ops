# Owner(s): ["module: intel"]

from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_ops import TestCommon
    from test_ops import TestMathBits
    from test_ops import TestFakeTensor
    from test_ops import TestCompositeCompliance
    from test_ops import TestTags


instantiate_device_type_tests(TestCommon, globals(), only_for="xpu")
instantiate_device_type_tests(TestMathBits, globals(), only_for="xpu")
instantiate_device_type_tests(TestFakeTensor, globals(), only_for="xpu")
instantiate_device_type_tests(TestCompositeCompliance, globals(), only_for="xpu")
instantiate_device_type_tests(TestTags, globals(), only_for="xpu")


if __name__ == "__main__":
    run_tests()
