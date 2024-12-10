
# Owner(s): ["module: intel"]

from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_indexing import NumpyTests,TestIndexing
    import torch

    torch.Tensor.is_cuda = torch.Tensor.is_xpu
    
    def __test_index_put_accumulate_with_optional_tensors(self, device):
        # TODO: replace with a better solution.
        # Currently, here using torchscript to put None into indices.
        # on C++ it gives indices as a list of 2 optional tensors: first is null and
        # the second is a valid tensor.
        @torch.jit.script
        def func(x, i, v):
            idx = [None, i]
            x.index_put_(idx, v, accumulate=True)
            return x
        
        n = 4
        t = torch.arange(n * 2, dtype=torch.float32).reshape(n, 2)
        t_dev = t.to(device)
        indices = torch.tensor([1, 0])
        indices_dev = indices.to(device)
        value0d = torch.tensor(10.0)
        value1d = torch.tensor([1.0, 2.0])

        out_cuda = func(t_dev, indices_dev, value0d.xpu())
        out_cpu = func(t, indices, value0d)
        self.assertEqual(out_cuda.cpu(), out_cpu)

        out_cuda = func(t_dev, indices_dev, value1d.xpu())
        out_cpu = func(t, indices, value1d)
        self.assertEqual(out_cuda.cpu(), out_cpu)

    TestIndexing.test_index_put_accumulate_with_optional_tensors = __test_index_put_accumulate_with_optional_tensors

instantiate_device_type_tests(NumpyTests, globals(), only_for=("xpu"), allow_xpu=True)

instantiate_device_type_tests(TestIndexing, globals(), only_for=("xpu"), allow_xpu=True)
if __name__ == "__main__":
    run_tests()
