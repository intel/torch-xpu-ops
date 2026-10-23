# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""
Window specific skip list for unit tests
Using pytest -k filtering syntax
"""

from skip_list_common import PYTORCH_TEST_DIR

skip_dict = {
    # main-thread stack (1-2 MB) overflows at the c recursion limit this test needs (#5726)
    f"{PYTORCH_TEST_DIR}/dynamo/test_repros.py": (
        "(test_dynamo_set_recursion_limit and not usage)",
    ),
    # main-thread stack (1-2 MB) overflows at the c recursion limit this test needs (#5726)
    f"{PYTORCH_TEST_DIR}/dynamo/test_dynamic_shapes.py": (
        "(test_dynamo_set_recursion_limit and not usage)",
    ),
    # Windows: Skip entire files using *.py:: pattern
    "test_decomp_xpu.py": [
        "test_decomp_xpu.py::",  # Skip entire file on Windows
    ],
    # Files where Windows only needs to skip specific tests (will merge with Linux defaults)
    # "test_linalg": [
    #     "test_cholesky_windows_bug",  # Only skip specific Windows issues
    #     "test_qr_windows_memory",     # Will be merged with Linux skip list
    # ],
    # New test groups only needed on Windows
    # "windows_specific_issues": [
    #     "test_dll_loading",
    #     "test_path_length",
    # ],
}
