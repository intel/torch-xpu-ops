import os
import sys

import pytest
from skip_list_win_mtl import skip_dict

IS_WINDOWS = sys.platform == "win32"

skip_list = skip_dict["test_xpu_xpu.py"]

skip_options = "not " + skip_list[0]
for skip_case in skip_list[1:]:
    skip_option = " and not " + skip_case
    skip_options += skip_option

test_command = ["-k", skip_options, "test_xpu_xpu.py", "-v"]
res = pytest.main(test_command)
sys.exit(res)
