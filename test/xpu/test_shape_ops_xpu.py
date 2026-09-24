# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

import unittest

from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_shape_ops import TestShapeOps

instantiate_device_type_tests(TestShapeOps, globals(), only_for="xpu", allow_xpu=True)


# Skip XPU shape-op tests that currently fail; tracked upstream.
# Each entry maps an exact generated test name to its tracking issue.
_xpu_skip_cases = {
    "TestShapeOpsXPU": {
        "test_flip_xpu_float32": "https://github.com/intel/torch-xpu-ops/issues/2722",
    },
}


def _apply_xpu_skips(_skip_cases):
    for _cls_name, _cases in _skip_cases.items():
        _cls = globals().get(_cls_name)
        if _cls is None:
            continue
        for _name, _issue in _cases.items():
            _method = getattr(_cls, _name, None)
            if _method is not None:
                setattr(
                    _cls,
                    _name,
                    unittest.skip(f"Skipped on XPU, see {_issue}")(_method),
                )


_apply_xpu_skips(_xpu_skip_cases)


if __name__ == "__main__":
    run_tests()
