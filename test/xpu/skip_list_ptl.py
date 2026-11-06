# Copyright 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

from skip_list_common import PYTORCH_TEST_DIR

skip_dict = {
    # torch-xpu-ops tests, not upstreamed yet, must be sorted alphabetically
    # upstreamed tests, must be sorted alphabetically
    f"{PYTORCH_TEST_DIR}/test_modules.py": (
        # ~800 s on PTL-H, over the 600 s UT timeout
        # https://github.com/intel/torch-xpu-ops/issues/5770
        "test_gradgrad_nn_Transformer_xpu_float64",
    ),
}
