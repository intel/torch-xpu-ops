# Owner(s): ["module: intel"]

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
)

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_dynamic_shapes import TestSymNumberMagicMethods

instantiate_parametrized_tests(TestSymNumberMagicMethods)


if __name__ == "__main__":
    run_tests()
