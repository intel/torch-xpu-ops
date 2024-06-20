
# Owner(s): ["module: intel"]

from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests,instantiate_parametrized_tests

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_maskedtensor import TestBasics,TestOperators,TestUnary,TestBinary,TestReductions

instantiate_device_type_tests(TestBasics, globals(), only_for=("xpu"), allow_xpu=True)

instantiate_device_type_tests(TestOperators, globals(), only_for=("xpu"), allow_xpu=True)
instantiate_parametrized_tests(TestUnary)
instantiate_parametrized_tests(TestBinary)
instantiate_parametrized_tests(TestReductions)

if __name__ == '__main__':
    run_tests()
