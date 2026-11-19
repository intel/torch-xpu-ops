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
# ruff: noqa F401

import os
import sys

import torch
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    skipXPUIf,
)
from torch.testing._internal.common_utils import parametrize, run_tests

try:
    from xpu_test_utils import XPUImportCtx
except Exception as e:
    from .xpu_test_utils import XPUImportCtx


with XPUImportCtx(False):
    from test_dataloader import (
        supported_multiprocessing_contexts,
        TestCustomPinFn,
        TestDataLoader,
        TestDataLoaderCUDA,
        TestDataLoaderCUDAPersistentWorkers,
        TestDataLoaderDevice,
        TestDataLoaderPersistentWorkers,
        TestDictDataLoader,
        TestDictDataLoaderDevice,
        TestStringDataLoader,
    )

    def _without_cuda_ipc_skip(fn):
        # unittest resolves the TEST_CUDA_IPC guard at import and replaces the method
        # with a wrapper that always raises SkipTest, so the flag cannot be flipped
        # afterwards; recover the undecorated body instead.
        while getattr(fn, "__unittest_skip_why__", None) == "CUDA IPC not available":
            fn = fn.__wrapped__
        return fn

    def _drop_cuda_ipc_skip(cls, *names):
        for name in names:
            setattr(cls, name, _without_cuda_ipc_skip(getattr(cls, name)))

    _XPU_IPC_ISSUE = "https://github.com/intel/torch-xpu-ops/issues/3000"

    _nested_tensor_multiprocessing = _without_cuda_ipc_skip(
        TestDataLoaderDevice.test_nested_tensor_multiprocessing
    )
    _sparse_tensor_multiprocessing = _without_cuda_ipc_skip(
        TestDataLoaderDevice.test_sparse_tensor_multiprocessing
    )

    # Upstream gates these on TEST_CUDA_IPC, which reads as "CUDA IPC not available"
    # on XPU. Re-skip against the XPU blocker instead so the files stay tracked:
    # reduce_storage() has no XPU branch, so no start method works.
    @parametrize(
        "context",
        [ctx for ctx in supported_multiprocessing_contexts if ctx is not None],
    )
    @skipXPUIf(True, _XPU_IPC_ISSUE)
    def nested_tensor_multiprocessing(self, device, context):
        _nested_tensor_multiprocessing(self, device, context)

    @parametrize(
        "context",
        [ctx for ctx in supported_multiprocessing_contexts if ctx is not None],
    )
    @skipXPUIf(True, _XPU_IPC_ISSUE)
    def sparse_tensor_multiprocessing(self, device, context):
        _sparse_tensor_multiprocessing(self, device, context)

    TestDataLoaderDevice.test_nested_tensor_multiprocessing = (
        nested_tensor_multiprocessing
    )
    TestDataLoaderDevice.test_sparse_tensor_multiprocessing = (
        sparse_tensor_multiprocessing
    )

    # These bodies are device-agnostic; only the CUDA IPC guard keeps them off XPU.
    # TestDataLoaderCUDAPersistentWorkers inherits them from TestDataLoaderCUDA.
    _drop_cuda_ipc_skip(
        TestDataLoaderCUDA,
        "test_multiple_dataloaders",
        "test_multiprocessing_contexts",
        "test_multiprocessing_iterdatapipe",
    )


instantiate_device_type_tests(
    TestDataLoaderDevice, globals(), only_for="xpu", allow_xpu=True
)
sys.path.extend(
    (
        os.path.dirname(os.path.abspath(__file__)) + "/../../../../test",
        os.path.dirname(os.path.abspath(__file__)) + "/../../../../test/nn",
    )
)


if __name__ == "__main__":
    run_tests()
