# Owner(s): ["module: intel"]
import torch
from torch.nn.modules.utils import _pair
from torch.testing._internal.common_utils import (
    run_tests,
    instantiate_parametrized_tests,
)

from hypothesis import assume, given
from hypothesis import strategies as st
import torch.testing._internal.hypothesis_utils as hu

try:
    from .xpu_test_utils import XPUPatchForImport
except Exception as e:
    from ..xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from core.test_quantized_op import TestQuantizedOps, pool_output_shape    

@given(X=hu.tensor(shapes=hu.array_shapes(min_dims=3, max_dims=4,
                                          min_side=1, max_side=10)),
       kernel=st.sampled_from((3, 5, 7)),
       stride=st.sampled_from((None, 1, 2)),
       dilation=st.integers(1, 2),
       padding=st.integers(0, 2),
       ceil_mode=st.booleans())
def _test_max_pool2d_xpu(self, X, kernel, stride, dilation, padding, ceil_mode):
    X = X[0] 
    assume(kernel // 2 >= padding)  # Kernel cannot be overhanging!
    iH, iW = X.shape[-2:]
    oH = pool_output_shape(iH, kernel, padding, stride, dilation, ceil_mode)
    assume(oH > 0)
    oW = pool_output_shape(iW, kernel, padding, stride, dilation, ceil_mode)
    assume(oW > 0)

    a = torch.from_numpy(X).to(torch.uint8)
    if a.dim() < 4:
        # Add nbatch dim
        a = torch.unsqueeze(a, dim=0)

    a_cpu_result = torch.ops.quantized.max_pool2d(
        a, kernel_size=_pair(kernel),
        stride=_pair(kernel if stride is None else stride),
        padding=_pair(padding), dilation=_pair(dilation), ceil_mode=ceil_mode)

    a_xpu = a.to('xpu:0')
    a_xpu_result = torch.ops.quantized.max_pool2d(
        a_xpu, kernel_size=_pair(kernel),
        stride=_pair(kernel if stride is None else stride),
        padding=_pair(padding), dilation=_pair(dilation), ceil_mode=ceil_mode)

    self.assertEqual(a_cpu_result, a_xpu_result,
                     msg="ops.quantized.max_pool2d results are off")
TestQuantizedOps.test_max_pool2d_xpu = _test_max_pool2d_xpu   

instantiate_parametrized_tests(TestQuantizedOps)

if __name__ == "__main__":
    run_tests()
