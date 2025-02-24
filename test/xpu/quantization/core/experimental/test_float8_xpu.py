# Owner(s): ["module: intel"]
import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_dtype import all_types_and_complex_and
from torch.testing._internal.common_utils import coalescedonoff, run_tests

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    import os
    import sys
    script_path = os.path.split(__file__)[0]
    sys.path.insert(0, os.path.realpath(os.path.join(script_path, "../../..")))
    from xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_float8 import TestFloat8Dtype


instantiate_device_type_tests(TestFloat8Dtype, globals(), only_for="xpu", allow_xpu=True)

if __name__ == "__main__":
    run_tests()
