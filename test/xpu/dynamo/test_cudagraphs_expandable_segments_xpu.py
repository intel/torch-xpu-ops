# Owner(s): ["module: cuda graphs"]
# This file is a placeholder for the XPU equivalent of test_cudagraphs_expandable_segments.py.
# All tests are skipped because cudagraph (XPUGraph) is not fully supported on XPU.
# Feature gap in xpugraph: XPUGraph is not fully supported
# See https://github.com/intel/torch-xpu-ops/issues/3594

import sys
import unittest

import torch
import torch._dynamo.test_case

sys.path.insert(0, "../../../../test/dynamo")

from test_cudagraphs import TestAotCudagraphs


@unittest.skip("xpugraph feature gap: XPUGraph is not fully supported on XPU")
class TestAotCudagraphsExpandableSegmentsXPU(TestAotCudagraphs):
    pass


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
