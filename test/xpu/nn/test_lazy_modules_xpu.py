# Owner(s): ["module: intel"]

from torch.nn.parameter import UninitializedParameter
from torch.testing._internal.common_utils import run_tests, suppress_warnings

try:
    from .xpu_test_utils import XPUPatchForImport
except Exception as e:
    from ..xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_lazy_modules import LazyModule, TestLazyModules


@suppress_warnings
def materialize_device(self):
    module = LazyModule()
    module.register_parameter("test_param", UninitializedParameter())
    module.test_param.materialize(10)
    self.assertTrue(module.test_param.device.type == "cpu")
    device = "xpu"
    module = LazyModule()
    module.register_parameter("test_param", UninitializedParameter())
    module.to(device)
    module.test_param.materialize(10)
    self.assertTrue(module.test_param.device.type == device)


TestLazyModules.test_materialize_device = materialize_device

if __name__ == "__main__":
    run_tests()
