# Owner(s): ["module: intel"]

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
)

try:
    from .xpu_test_utils import XPUPatchForImport
except Exception as e:
    from ..xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_pruning import TestPruningNN


instantiate_parametrized_tests(TestPruningNN)


if __name__ == "__main__":
    run_tests()
