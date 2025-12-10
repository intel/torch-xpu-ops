# Copyright 2020-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

skip_dict = {
    "test_ops_xpu.py": (
        "test_compare_cpu_pow_xpu_bfloat16",  # https://github.com/intel/torch-xpu-ops/pull/764
        "test_compare_cpu_argmin_xpu_int",
    ),
}
