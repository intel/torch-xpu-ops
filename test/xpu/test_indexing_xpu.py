# Owner(s): ["module: intel"]

from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import DeterministicGuard, run_tests

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    import torch
    from test_indexing import NumpyTests, TestIndexing

    torch.Tensor.is_cuda = torch.Tensor.is_xpu

    def __test_index_put_deterministic_with_optional_tensors(self, device):
        def func(x, i, v):
            with DeterministicGuard(True):
                x[..., i] = v
            return x

        def func1(x, i, v):
            with DeterministicGuard(True):
                x[i] = v
            return x

        n = 4
        t = torch.arange(n * 2, dtype=torch.float32).reshape(n, 2)
        t_dev = t.to(device)
        indices = torch.tensor([1, 0])
        indices_dev = indices.to(device)
        value0d = torch.tensor(10.0)
        value1d = torch.tensor([1.0, 2.0])
        values2d = torch.randn(n, 1)

        for val in (value0d, value1d, values2d):
            out_cuda = func(t_dev, indices_dev, val.to(device))
            out_cpu = func(t, indices, val)
            self.assertEqual(out_cuda.cpu(), out_cpu)

        t = torch.zeros((5, 4))
        t_dev = t.to(device)
        indices = torch.tensor([1, 4, 3])
        indices_dev = indices.to(device)
        val = torch.randn(4)
        out_cuda = func1(t_dev, indices_dev, val.xpu())
        out_cpu = func1(t, indices, val)
        self.assertEqual(out_cuda.cpu(), out_cpu)

        t = torch.zeros(2, 3, 4)
        ind = torch.tensor([0, 1])
        val = torch.randn(6, 2)
        with self.assertRaisesRegex(RuntimeError, "shape mismatch"):
            func(t, ind, val)

        with self.assertRaisesRegex(RuntimeError, "must match"):
            func(t.to(device), ind.to(device), val.to(device))

        val = torch.randn(2, 3, 1)
        out_cuda = func1(t.to(device), ind.to(device), val.to(device))
        out_cpu = func1(t, ind, val)
        self.assertEqual(out_cuda.cpu(), out_cpu)

    TestIndexing.test_index_put_deterministic_with_optional_tensors = (
        __test_index_put_deterministic_with_optional_tensors
    )

instantiate_device_type_tests(NumpyTests, globals(), only_for=("xpu"), allow_xpu=True)

instantiate_device_type_tests(TestIndexing, globals(), only_for=("xpu"), allow_xpu=True)
if __name__ == "__main__":
    run_tests()
