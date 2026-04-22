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

import unittest
from sys import platform

import torch
import torch.multiprocessing as mp
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    run_tests,
    TestCase,
)

try:
    from xpu_test_utils import XPUPatchForImport
except Exception:
    from .xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_multiprocessing import queue_get_exception, TestMultiprocessing


@unittest.skipIf(IS_WINDOWS, "not applicable to Windows (only fails with fork)")
@unittest.skipIf(not torch.xpu.is_available(), "XPU not available")
def _test_cuda_bad_call(self):
    # Initialize XPU
    t = torch.zeros(5, 5).xpu().cpu()
    inq = mp.Queue()
    outq = mp.Queue()
    p = mp.Process(target=queue_get_exception, args=(inq, outq))
    p.start()
    inq.put(t)
    p.join()
    self.assertIsInstance(outq.get(), RuntimeError)


@unittest.skipIf(IS_WINDOWS, "not applicable to Windows (only fails with fork)")
@unittest.skipIf(not torch.xpu.is_available(), "XPU not available")
def _test_wrong_cuda_fork(self):
    stderr = TestCase.runWithPytorchAPIUsageStderr(
        """\
import torch
from torch.multiprocessing import Process
def run(rank):
    torch.xpu.set_device(rank)
if __name__ == "__main__":
    size = 2
    processes = []
    for rank in range(size):
        # it would work fine without the line below
        x = torch.rand(20, 2).xpu()
        p = Process(target=run, args=(rank,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
"""
    )
    self.assertRegex(stderr, "Cannot re-initialize XPU in forked subprocess.")


@unittest.skipIf(not torch.xpu.is_available(), "XPU not available")
def _test_empty_tensor_sharing_cuda(self):
    self._test_empty_tensor_sharing(torch.float32, torch.device("xpu"))
    self._test_empty_tensor_sharing(torch.int64, torch.device("xpu"))


@unittest.skipIf(
    platform == "darwin", "file descriptor strategy is not supported on macOS"
)
@unittest.skipIf(not torch.xpu.is_available(), "XPU not available")
def _test_is_shared_cuda(self):
    t = torch.randn(5, 5).xpu()
    self.assertTrue(t.is_shared())


TestMultiprocessing.test_cuda_bad_call = _test_cuda_bad_call
TestMultiprocessing.test_wrong_cuda_fork = _test_wrong_cuda_fork
TestMultiprocessing.test_empty_tensor_sharing_cuda = _test_empty_tensor_sharing_cuda
TestMultiprocessing.test_is_shared_cuda = _test_is_shared_cuda


if __name__ == "__main__":
    run_tests()
