# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Portions of this file are derived from PyTorch
# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: BSD-3-Clause

# Owner(s): ["module: intel"]

import os
import sys
from unittest import skipIf as skip

import torch
import torch._numpy as w
from torch._numpy.testing import assert_allclose
from torch.testing._internal.common_utils import run_tests

try:
    from xpu_test_utils import XPUPatchForImport
except Exception:
    from .xpu_test_utils import XPUPatchForImport

# torch_np test lives under pytorch/test/torch_np, which is not in the default
# XPUPatchForImport search path. Add it here so the hook-override import works.
_TORCH_NP_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../test/torch_np")
)
if _TORCH_NP_DIR not in sys.path:
    sys.path.insert(0, _TORCH_NP_DIR)

with XPUPatchForImport(False):
    from test_basic import TestMisc


@skip(not torch.xpu.is_available(), reason="requires xpu")
def _test_f16_on_cuda(self):
    # make sure operations with float16 tensors give same results on XPU and on CPU
    t = torch.arange(5, dtype=torch.float16)
    assert_allclose(w.vdot(t.xpu(), t.xpu()), w.vdot(t, t))
    assert_allclose(w.inner(t.xpu(), t.xpu()), w.inner(t, t))
    assert_allclose(w.matmul(t.xpu(), t.xpu()), w.matmul(t, t))
    assert_allclose(w.einsum("i,i", t.xpu(), t.xpu()), w.einsum("i,i", t, t))

    assert_allclose(w.mean(t.xpu()), w.mean(t))

    assert_allclose(w.cov(t.xpu(), t.xpu()), w.cov(t, t).tensor.xpu())
    assert_allclose(w.corrcoef(t.xpu()), w.corrcoef(t).tensor.xpu())


TestMisc.test_f16_on_cuda = _test_f16_on_cuda


if __name__ == "__main__":
    run_tests()
