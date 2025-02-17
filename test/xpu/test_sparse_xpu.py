# Owner(s): ["module: intel"]
import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_dtype import all_types_and_complex_and
from torch.testing._internal.common_utils import run_tests

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_sparse import TestSparse

    # @skipIfTorchDynamo()
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def sparse_csr_from_dense(self, device, dtype):
        dense = torch.tensor(
            [[4, 5, 0], [0, 0, 0], [1, 0, 0]], dtype=dtype, device=device
        )
        sparse = dense.to_sparse_csr()
        self.assertEqual(
            torch.tensor([0, 2, 2, 3], dtype=torch.int64), sparse.crow_indices()
        )
        self.assertEqual(
            torch.tensor([0, 1, 0], dtype=torch.int64), sparse.col_indices()
        )
        self.assertEqual(torch.tensor([4, 5, 1], dtype=dtype), sparse.values())

        dense = torch.tensor(
            [[0, 0, 0], [0, 0, 1], [1, 0, 0]], dtype=dtype, device=device
        )
        sparse = dense.to_sparse_csr()
        self.assertEqual(
            torch.tensor([0, 0, 1, 2], dtype=torch.int64), sparse.crow_indices()
        )
        self.assertEqual(torch.tensor([2, 0], dtype=torch.int64), sparse.col_indices())
        self.assertEqual(torch.tensor([1, 1], dtype=dtype), sparse.values())

        dense = torch.tensor(
            [[2, 2, 2], [2, 2, 2], [2, 2, 2]], dtype=dtype, device=device
        )
        sparse = dense.to_sparse_csr()
        self.assertEqual(
            torch.tensor([0, 3, 6, 9], dtype=torch.int64), sparse.crow_indices()
        )
        self.assertEqual(
            torch.tensor([0, 1, 2] * 3, dtype=torch.int64), sparse.col_indices()
        )
        self.assertEqual(torch.tensor([2] * 9, dtype=dtype), sparse.values())

    TestSparse.test_sparse_csr_from_dense = sparse_csr_from_dense

instantiate_device_type_tests(TestSparse, globals(), only_for="xpu", allow_xpu=True)

if __name__ == "__main__":
    run_tests()
