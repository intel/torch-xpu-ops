# Copyright (c) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

skip_dict = {
    # failed on MTL windows, skip first for Preci
    "test_xpu.py": ("test_mem_get_info_xpu",),
}
