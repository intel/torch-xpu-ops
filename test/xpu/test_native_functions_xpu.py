# Owner(s): ["module: intel"]

from torch.testing._internal.common_utils import run_tests

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_native_functions import TestNativeFunctions  # noqa: F401`

if __name__ == "__main__":
    run_tests()
