# Owner(s): ["module: intel"]

from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, instantiate_parametrized_tests

try:
    from .xpu_test_utils import XPUPatchForImport
except Exception as e:
    from ..xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_pooling import TestPoolingNNDeviceType, TestPoolingNN


instantiate_device_type_tests(TestPoolingNNDeviceType, globals(), only_for="xpu")
instantiate_parametrized_tests(TestPoolingNN)


if __name__ == "__main__":
    run_tests()
