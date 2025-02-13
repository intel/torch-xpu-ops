# Owner(s): ["module: intel"]

from torch.testing._internal.common_utils import run_tests

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport


def select_cuda(self):
    self._test_select("xpu")


def as_strided_cuda(self):
    self._test_as_strided("xpu")


with XPUPatchForImport(False):
    from test_namedtensor import TestNamedTensor

TestNamedTensor.test_select_cuda = select_cuda
TestNamedTensor.test_as_strided_cuda = as_strided_cuda

if __name__ == "__main__":
    run_tests()
