# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]


from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests

try:
    from xpu_test_utils import (
        patch_matrix_op_samples_with_well_conditioned_inputs,
        XPUPatchForImport,
    )
except Exception as e:
    from .xpu_test_utils import (
        patch_matrix_op_samples_with_well_conditioned_inputs,
        XPUPatchForImport,
    )

with XPUPatchForImport(False):
    from test_ops import (
        TestCommon,
        TestCompositeCompliance,
        TestFakeTensor,
        TestForwardADWithScalars,
        TestMathBits,
    )
    from torch.testing._internal.common_methods_invocations import op_db


patch_matrix_op_samples_with_well_conditioned_inputs(
    op_db, "linalg.cond", "_xpu_linalg_cond_sample_patch"
)

instantiate_device_type_tests(TestCommon, globals(), only_for="xpu", allow_xpu=True)
instantiate_device_type_tests(TestMathBits, globals(), only_for="xpu", allow_xpu=True)
# in finegrand
instantiate_device_type_tests(
    TestCompositeCompliance, globals(), only_for="xpu", allow_xpu=True
)
# only CPU
# instantiate_device_type_tests(TestRefsOpsInfo, globals(), only_for="xpu", allow_xpu=True)
# not important
instantiate_device_type_tests(TestFakeTensor, globals(), only_for="xpu", allow_xpu=True)
instantiate_device_type_tests(
    TestForwardADWithScalars, globals(), only_for="xpu", allow_xpu=True
)
# instantiate_device_type_tests(TestTags, globals(), only_for="xpu", allow_xpu=True)


if __name__ == "__main__":
    run_tests()
