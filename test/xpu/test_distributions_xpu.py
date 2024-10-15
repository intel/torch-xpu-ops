# Owner(s): ["module: intel"]


from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport
with XPUPatchForImport(False):
    from test_distributions import (
        TestDistributions,
        TestRsample,
        TestDistributionShapes,
        TestKL,
        TestConstraints,
        TestNumericalStability,
        TestLazyLogitsInitialization,
        TestAgainstScipy,
        TestFunctors,
        TestValidation,
        TestJit,
    )
instantiate_device_type_tests(TestDistributions, globals(), only_for="xpu", allow_xpu=True)
instantiate_device_type_tests(TestRsample, globals(), only_for="xpu", allow_xpu=True)
instantiate_device_type_tests(TestDistributionShapes, globals(), only_for="xpu", allow_xpu=True)
instantiate_device_type_tests(TestKL, globals(), only_for="xpu", allow_xpu=True)
instantiate_device_type_tests(TestConstraints, globals(), only_for="xpu", allow_xpu=True)
instantiate_device_type_tests(TestNumericalStability, globals(), only_for="xpu", allow_xpu=True)
instantiate_device_type_tests(TestLazyLogitsInitialization, globals(), only_for="xpu", allow_xpu=True)
instantiate_device_type_tests(TestAgainstScipy, globals(), only_for="xpu", allow_xpu=True)
instantiate_device_type_tests(TestFunctors, globals(), only_for="xpu", allow_xpu=True)
instantiate_device_type_tests(TestValidation, globals(), only_for="xpu", allow_xpu=True)
instantiate_device_type_tests(TestJit, globals(), only_for="xpu", allow_xpu=True)


if __name__ == "__main__":
    run_tests()
