# Owner(s): ["module: intel"]

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests, onlyXPU
from torch.testing._internal.common_utils import run_tests

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_unary_ufuncs import TestUnaryUfuncs

    @onlyXPU
    def _nonzero_static_large(self, device):
        # large enough to have multiple iters per SM even on H100
        # with 132 sms
        size_inp = 1024 * 16 * 132 + 1024 * 16
        x = torch.zeros(size_inp, device=device)
        # unique indices
        indices = torch.randperm(size_inp, device=device)[: size_inp // 2]
        sorted, _ = torch.sort(indices)
        x[sorted] = 1
        res = torch.nonzero_static(x, size=size_inp // 2).view(-1)
        self.assertEqual(res, sorted)
        # no oob writes
        out = torch.full((size_inp,), 10, device=device, dtype=torch.int64)
        res = torch.nonzero_static(x, size=size_inp // 4, out=out[: size_inp // 2])
        self.assertEqual(out[: size_inp // 4], sorted[: size_inp // 4])
        self.assertEqual(
            out[size_inp // 4 :],
            torch.tensor(10, device="xpu").expand_as(out[size_inp // 4 :]),
        )
        # correct fill for 2d
        x = x.view(2, size_inp // 2)
        ref = x.nonzero()
        res = x.nonzero_static(size=size_inp // 2 + 2)
        self.assertEqual(res.shape, [size_inp // 2 + 2, 2])
        self.assertEqual(ref, res[: size_inp // 2])
        self.assertEqual(
            res[size_inp // 2 :],
            torch.tensor(-1, device="xpu").expand_as(res[size_inp // 2 :]),
        )
    TestUnaryUfuncs.test_nonzero_static_large = _nonzero_static_large

instantiate_device_type_tests(TestUnaryUfuncs, globals(),only_for=("xpu"), allow_xpu=True)

if __name__ == "__main__":
    run_tests()
