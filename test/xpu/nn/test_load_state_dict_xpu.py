# Owner(s): ["module: intel"]

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
    TestCase,
)

try:
    from .xpu_test_utils import XPUPatchForImport
except Exception as e:
    from ..xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_load_state_dict import TestLoadStateDict, TestLoadStateDictSwap


instantiate_parametrized_tests(TestLoadStateDict)
instantiate_parametrized_tests(TestLoadStateDictSwap)


if __name__ == "__main__":
    TestCase._default_dtype_check_enabled = True
    run_tests()
