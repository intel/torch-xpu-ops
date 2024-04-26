# Owner(s): ["module: intel"]

import torch
from torch.testing._internal.common_utils import run_tests

from ..xpu_test_utils import XPUTestPatch, instantiate_device_type_tests

with XPUTestPatch():
    from test_embedding import TestEmbeddingNNDeviceType

instantiate_device_type_tests(TestEmbeddingNNDeviceType, globals(), only_for="xpu")


if __name__ == "__main__":
    run_tests()
