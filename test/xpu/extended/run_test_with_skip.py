import os
import sys
from os.path import abspath, join, dirname
sys.path.insert(0, join(abspath(dirname(__file__)), ".."))
from extended.skip_list_common import skip_dict
from extended.skip_list_win import skip_dict as skip_dict_win
from xpu_test_utils import get_device_name

IS_WINDOWS = sys.platform == "win32"
dev_name = get_device_name()
if dev_name == "ARC":
    from extended.skip_list_arc import skip_dict as skip_dict_specifical
    from extended.skip_list_win_arc import skip_dict as skip_dict_win_arc

skip_list = skip_dict["test_ops_xpu.py"]
if dev_name == "ARC":
    skip_list += skip_dict_specifical["test_ops_xpu.py"]
    if IS_WINDOWS:
        skip_list += skip_dict_win_arc["test_ops_xpu.py"]
if IS_WINDOWS:
    skip_list += skip_dict_win["test_ops_xpu.py"]

skip_options = " -k \"not " + skip_list[0]
for skip_case in skip_list[1:]:
    skip_option = " and not " + skip_case
    skip_options += skip_option
skip_options += "\""

os.environ["PYTORCH_TEST_WITH_SLOW"]="1"
test_command = "pytest -v test_ops_xpu.py"
test_command += skip_options
res = os.system(test_command)
sys.exit(res)
