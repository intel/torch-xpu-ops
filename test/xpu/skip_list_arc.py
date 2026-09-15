# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

from skip_list_common import PYTORCH_TEST_DIR

skip_dict = {
    f"{PYTORCH_TEST_DIR}/test_indexing.py": (
        "test_index_put_accumulate_large_tensor_xpu",
    ),
    "test_nn_xpu.py": ("test_grid_sample_large_xpu",),
    "test_tensor_creation_ops_xpu.py": (
        "test_float_to_int_conversion_finite_xpu_int64",
    ),
}
