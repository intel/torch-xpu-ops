# Owner(s): ["module: intel"]

from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
)

try:
    from .xpu_test_utils import XPUPatchForImport
except Exception as e:
    from ..xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_parametrization import TestNNParametrization, TestNNParametrizationDevice


instantiate_device_type_tests(
    TestNNParametrizationDevice, globals(), only_for="xpu", allow_xpu=True
)
instantiate_parametrized_tests(TestNNParametrization)


if __name__ == "__main__":
    run_tests()
