# Owner(s): ["module: intel"]

from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_view_ops import TestOldViewOps, TestViewOps

    def is_view_of(self, base, other):
        if (
            not other._is_view()
            or other is base
            or other._base is not base
            or base.device != other.device
        ):
            return False

        # Note: only validates storage on native device types
        # because some accelerators, like XLA, do not expose storage
        if base.device.type == "cpu" or base.device.type == "xpu":
            if base.untyped_storage().data_ptr() != other.untyped_storage().data_ptr():
                return False

        return True

    TestViewOps.is_view_of = is_view_of


instantiate_device_type_tests(
    TestViewOps, globals(), include_lazy=True, only_for="xpu", allow_xpu=True
)
instantiate_device_type_tests(TestOldViewOps, globals(), only_for="xpu", allow_xpu=True)

if __name__ == "__main__":
    run_tests()
