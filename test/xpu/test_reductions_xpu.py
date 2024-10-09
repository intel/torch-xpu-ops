# Owner(s): ["module: intel"]

import torch

from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_reductions import TestReductions


def _test_mode_wrong_device(self, device):
    # CPU Input Tensor
    x = torch.ones(2)

    with self.assertRaisesRegex(RuntimeError,
                                "Expected all tensors to be on the same device, but found at least two devices"):
        values = torch.tensor([], device=device)
        torch.mode(x, -1, True, out=(values, torch.tensor([], dtype=torch.long)))

    with self.assertRaisesRegex(RuntimeError,
                                "Expected all tensors to be on the same device, but found at least two devices"):
        indices = torch.tensor([], device=device)
        torch.mode(x, -1, True, out=(torch.tensor([]), indices))
    
TestReductions.test_mode_wrong_device=_test_mode_wrong_device

instantiate_device_type_tests(TestReductions, globals(), only_for="xpu", allow_xpu=True)


if __name__ == "__main__":
    run_tests()
