# Owner(s): ["module: intel"]
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_sparse_csr import TestSparseCompressed, TestSparseCSR

instantiate_device_type_tests(TestSparseCompressed, globals(), only_for="xpu", allow_xpu=True)

instantiate_device_type_tests(TestSparseCSR, globals(), only_for="xpu", allow_xpu=True)

if __name__ == "__main__":
    run_tests()