# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

import os
import re
import sys
from io import StringIO

import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from skip_list_win_mtl import skip_dict

IS_WINDOWS = sys.platform == "win32"

skip_list = skip_dict["test_xpu.py"]

skip_options = "not " + skip_list[0]
for skip_case in skip_list[1:]:
    skip_option = " and not " + skip_case
    skip_options += skip_option

xml_path = os.path.abspath("./test_xpu.xml")
pytorch_test_dir = os.path.abspath(os.path.join(_THIS_DIR, "../../../../test"))
original_cwd = os.getcwd()
original_stdout = sys.stdout
captured_stdout = StringIO()
sys.stdout = captured_stdout
try:
    os.chdir(pytorch_test_dir)
    test_command = ["-k", skip_options, "test_xpu.py", "-v", "--junit-xml", xml_path]
    res = pytest.main(test_command)
finally:
    sys.stdout = original_stdout
    os.chdir(original_cwd)

output = captured_stdout.getvalue()

cleaned_output = re.sub(
    r"\.\.(\/|\\)\.\.(\/|\\)\.\.(\/|\\)\.\.(\/|\\)test(\/|\\)", "", output
)
print(cleaned_output, end="")
sys.exit(res)
