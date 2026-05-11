# Owner(s): ["module: dynamo"]
import sys
import unittest

import torch
import torch._dynamo
import torch._dynamo.backends
import torch._dynamo.test_case

sys.path.insert(0, "../../../../test/dynamo")

from test_backends import TestOptimizations
from torch.testing._internal.common_device_type import instantiate_device_type_tests


class TestOptimizationsXPU(TestOptimizations):
    # Skip test_aot_cudagraphs because cudagraph is not supported on XPU
    # Feature gap in xpugraph: XPUGraph is not fully supported
    # See https://github.com/intel/torch-xpu-ops/issues/3594
    @unittest.skip("xpugraph feature gap: XPUGraph is not fully supported")
    def test_aot_cudagraphs(self, device):
        pass


instantiate_device_type_tests(TestOptimizationsXPU, globals(), only_for="xpu", allow_xpu=True)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
