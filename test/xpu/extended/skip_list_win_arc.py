# Copyright 2020-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

skip_dict = {
    # SYCL Compiler on Windows removed the following operations when '-cl-poison-unsupported-fp64-kernels' is on
    # Hence, skip the following windows specific errors
    "test_ops_xpu.py": (
        "test_compare_cpu_sqrt_xpu_complex64",
        "test_backward_nn_functional_adaptive_avg_pool2d_xpu_float32",
    ),
}
