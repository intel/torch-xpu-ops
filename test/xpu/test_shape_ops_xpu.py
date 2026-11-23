# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

import torch
from torch.testing._internal.common_device_type import (
    dtypesIfXPU,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_dtype import all_passthru_types_and
from torch.testing._internal.common_utils import run_tests

try:
    from xpu_test_utils import XPUImportCtx
except Exception:
    from .xpu_test_utils import XPUImportCtx

with XPUImportCtx(False):
    from test_shape_ops import TestShapeOps


# ======================================================================
# dtypesIfXPU decorator additions
# ======================================================================

TestShapeOps.test_flip = dtypesIfXPU(*all_passthru_types_and(torch.chalf))(
    TestShapeOps.test_flip
)

instantiate_device_type_tests(TestShapeOps, globals(), only_for="xpu", allow_xpu=True)


if __name__ == "__main__":
    run_tests()
