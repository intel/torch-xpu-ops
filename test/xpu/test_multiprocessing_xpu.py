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
import unittest

import torch
import torch.multiprocessing as mp
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import IS_WINDOWS, run_tests, TestCase

try:
    from xpu_test_utils import ensure_pytorch_test_path, XPUPatchForImport
except Exception:
    from .xpu_test_utils import ensure_pytorch_test_path, XPUPatchForImport

with XPUPatchForImport(False) as patcher:
    from test_multiprocessing import TestMultiprocessing, TestMultiprocessingDeviceType


test_dir = os.path.abspath(patcher.test_package[0])
ensure_pytorch_test_path(test_dir)


def queue_get_exception(inqueue, outqueue):
    os.close(2)
    try:
        torch.zeros(5, 5).xpu()
    except Exception as e:
        outqueue.put(e)
    else:
        outqueue.put("no exception")


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


TestMultiprocessing.test_cuda_bad_call = _test_cuda_bad_call
TestMultiprocessing.test_wrong_cuda_fork = _test_wrong_cuda_fork

TestMultiprocessingDeviceType.test_integer_parameter_serialization = unittest.skip(
    "XPU storage serialization is not supported"
)(TestMultiprocessingDeviceType.test_integer_parameter_serialization)
TestMultiprocessingDeviceType.test_leaf_variable_sharing = unittest.skip(
    "XPU storage sharing is not supported"
)(TestMultiprocessingDeviceType.test_leaf_variable_sharing)
TestMultiprocessingDeviceType.test_parameter_sharing = unittest.skip(
    "XPU storage sharing is not supported"
)(TestMultiprocessingDeviceType.test_parameter_sharing)
TestMultiprocessingDeviceType.test_variable_sharing = unittest.skip(
    "XPU storage sharing is not supported"
)(TestMultiprocessingDeviceType.test_variable_sharing)
TestMultiprocessingDeviceType.test_simple_sharing = unittest.skip(
    "XPU storage sharing is not supported"
)(TestMultiprocessingDeviceType.test_simple_sharing)
TestMultiprocessingDeviceType.test_empty_tensor_sharing = unittest.skipIf(
    IS_WINDOWS, "XPU empty tensor sharing is not supported on Windows"
)(TestMultiprocessingDeviceType.test_empty_tensor_sharing)

instantiate_device_type_tests(
    TestMultiprocessingDeviceType, globals(), only_for="xpu", allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
