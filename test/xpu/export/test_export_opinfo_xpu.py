# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
)
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_utils import run_tests

try:
    from xpu_test_utils import XPUImportCtx
except Exception:
    from ..xpu_test_utils import XPUImportCtx

with XPUImportCtx(False):
    from test_export_opinfo import TestExportOnFakeDevice

# Upstream marks the without_cudnn variant onlyCUDA; XPU dtypes already inherit from CUDA.
for op in op_db:
    if (
        op.name == "nn.functional.batch_norm"
        and op.variant_test_name == "without_cudnn"
    ):
        op.decorators = tuple(d for d in op.decorators if d is not onlyCUDA)

instantiate_device_type_tests(
    TestExportOnFakeDevice, globals(), only_for="xpu", allow_xpu=True
)


if __name__ == "__main__":
    run_tests()
