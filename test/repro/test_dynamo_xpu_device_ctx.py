# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
"""
Regression test for the torch.xpu.device context manager used inside
torch.compile(fullgraph=True).

Root cause
----------
``torch.xpu`` is listed in ``MOD_SKIPLIST`` in
``torch/_dynamo/trace_rules.py``.  When TorchDynamo traces a function
that calls ``torch.xpu.device(...)`` with ``fullgraph=True``, it tries
to inline ``device.__init__``, whose source file is
``torch/xpu/__init__.py``.  Because that file matches MOD_SKIPLIST,
Dynamo raises::

    torch._dynamo.exc.Unsupported: Attempted to inline function marked
    as skipped
      qualname: device.__init__, skip reason: skipped according
      trace_rules.lookup MOD_SKIPLIST

Upstream fix
------------
Apply ``@torch._dynamo.dont_skip_tracing`` to ``torch.xpu.device.__init__``
in ``torch/xpu/__init__.py`` (PyTorch upstream change required).

Workaround applied here
-----------------------
Until the upstream fix lands, this module monkey-patches
``torch.xpu.device.__init__`` with the ``@dont_skip_tracing`` decorator
so that Dynamo can trace into the method without raising.

Reproducer
----------
cd <pytorch> && PYTORCH_TEST_WITH_SLOW=1 pytest -v \\
    test/dynamo/test_ctx_manager.py -k test_cuda_device
"""

import unittest

import torch
import torch._dynamo

# ---------------------------------------------------------------------------
# Workaround: decorate torch.xpu.device.__init__ with dont_skip_tracing so
# that TorchDynamo is allowed to trace into it.  This mirrors the upstream
# fix that should land in torch/xpu/__init__.py.
# ---------------------------------------------------------------------------
if torch.xpu.is_available():
    torch.xpu.device.__init__ = torch._dynamo.dont_skip_tracing(
        torch.xpu.device.__init__
    )


@unittest.skipIf(not torch.xpu.is_available(), "requires XPU")
@unittest.skipIf(not torch._dynamo.is_dynamo_supported(), "dynamo is not supported")
class TestDynamoXpuDeviceCtx(unittest.TestCase):
    """Verify torch.xpu.device context manager is traceable under fullgraph=True."""

    def setUp(self):
        torch._dynamo.reset()

    def tearDown(self):
        torch._dynamo.reset()

    def test_xpu_device_ctx_fullgraph(self):
        """torch.xpu.device used as ctx-manager must compile without graph-break."""

        def fn(x):
            with torch.xpu.device(x.device.index):
                x = torch.sin(x + 1)
            return x

        x = torch.randn((2, 2), device="xpu")
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        torch.testing.assert_close(ref, res)

    def test_xpu_device_ctx_via_get_device_module(self):
        """Same as above but accessed via torch.get_device_module() (matches
        the pattern used in PyTorch's test_ctx_manager.py::test_cuda_device)."""

        device_type = "xpu"

        def fn(x):
            with torch.get_device_module(device_type).device(x.device.index):
                x = torch.sin(x + 1)
            return x

        x = torch.randn((2, 2), device=device_type)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        torch.testing.assert_close(ref, res)


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
