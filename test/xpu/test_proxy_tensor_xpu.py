# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
# ruff: noqa: F401

from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests

try:
    from xpu_test_utils import XPUImportCtx
except Exception:
    from .xpu_test_utils import XPUImportCtx

with XPUImportCtx(False):
    from test_proxy_tensor import (
        TestGenericProxyTensorDevice,
        TestGenericProxyTensorFake,
        TestGenericProxyTensorReal,
        TestGenericProxyTensorSymbolic,
        TestSymbolicTracing,
        TestSymbolicTracingDevice,
        TestUnbackedSymbolicTracingDevice,
    )


instantiate_device_type_tests(
    TestGenericProxyTensorDevice, globals(), only_for="xpu", allow_xpu=True
)
instantiate_device_type_tests(
    TestSymbolicTracingDevice, globals(), only_for="xpu", allow_xpu=True
)
instantiate_device_type_tests(
    TestUnbackedSymbolicTracingDevice, globals(), only_for="xpu", allow_xpu=True
)


if __name__ == "__main__":
    run_tests()
