import os
import pytest
import sys
from skip_list_common import skip_dict
from skip_list_win import skip_dict as skip_dict_win
from skip_list_win_lnl import skip_dict as skip_dict_win_lnl

IS_WINDOWS = sys.platform == "win32"

skip_list = skip_dict["test_ops_xpu.py"]
if IS_WINDOWS:
    skip_list += skip_dict_win["test_ops_xpu.py"] + skip_dict_win_lnl["test_ops_xpu.py"]

skip_options = "not " + skip_list[0]
for skip_case in skip_list[1:]:
    skip_option = " and not " + skip_case
    skip_options += skip_option

os.environ["PYTORCH_TEST_WITH_SLOW"]="1"
test_command = ["-k", skip_options, "test_ops_xpu.py", "-v"]
res = pytest.main(test_command)
sys.exit(res)
