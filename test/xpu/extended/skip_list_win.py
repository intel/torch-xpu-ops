# Copyright (c) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

skip_dict = {
    "test_ops_xpu.py": (
        "test_compare_cpu_pow_xpu_bfloat16",  # https://github.com/intel/torch-xpu-ops/pull/764
        "test_compare_cpu_argmin_xpu_int",
    ),
}
