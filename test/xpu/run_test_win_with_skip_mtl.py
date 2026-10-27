# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

import re
import subprocess
import sys
from io import StringIO

import pytest
from skip_list_win_mtl import skip_dict

IS_WINDOWS = sys.platform == "win32"

skip_list = skip_dict["test_xpu.py"]

skip_options = "not " + skip_list[0]
for skip_case in skip_list[1:]:
    skip_option = " and not " + skip_case
    skip_options += skip_option

original_stdout = sys.stdout
sys.stdout = StringIO()

test_command = ["-k", skip_options, "../../../../test/test_xpu.py", "-v"]
test_command.extend(["--junit-xml", "./test_xpu.xml"])
res = pytest.main(test_command)

output = sys.stdout.getvalue()
sys.stdout = original_stdout

cleaned_output = re.sub(
    r"\.\.(\/|\\)\.\.(\/|\\)\.\.(\/|\\)\.\.(\/|\\)test(\/|\\)", "", output
)
print(cleaned_output, end="", flush=True)

# TestXpuTrace in test_xpu.py enables GPU trace for the rest of the process and
# the trace hook swallows pending Python exceptions, so run GEMM tests separately.
# The "test_xpu." prefix makes check-ut.py count these cases as test_xpu.
gemm_options = (
    "test__int_mm or test__int4_mm or test_compile_int4_mm or test__int_mm_errors"
)
res_gemm = subprocess.run(
    [
        sys.executable,
        "-m",
        "pytest",
        "-k",
        gemm_options,
        "test_linalg_xpu.py",
        "-v",
        "--junit-xml",
        "./test_xpu.test_linalg_xpu.xml",
    ]
).returncode
sys.exit(res or res_gemm)
