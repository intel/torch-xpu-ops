# Owner(s): ["module: intel"]


from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport
with XPUPatchForImport(False):
    from test_ops import TestCommon, TestMathBits
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
