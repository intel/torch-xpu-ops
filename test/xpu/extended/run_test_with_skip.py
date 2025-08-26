import os
import sys

import torch
from skip_list_common import skip_dict
from skip_list_win import skip_dict as skip_dict_win

IS_WINDOWS = sys.platform == "win32"

skip_list = skip_dict["test_ops_xpu.py"]
if IS_WINDOWS:
    skip_list += skip_dict_win["test_ops_xpu.py"]

skip_options = ' -k "not ' + skip_list[0]
for skip_case in skip_list[1:]:
    skip_option = " and not " + skip_case
    skip_options += skip_option
skip_options += '"'

# pytest options
xpu_num = torch.xpu.device_count()
parallel_options = (
    " --dist worksteal "
    + " ".join([f"--tx popen//env:ZE_AFFINITY_MASK={x}" for x in range(xpu_num)])
    if xpu_num > 1
    else " -n 1 "
)
test_options = f" --timeout 600 --timeout_method=thread {parallel_options} "

os.environ["PYTORCH_TEST_WITH_SLOW"] = "1"
test_command = (
    f" pytest {test_options} --junit-xml=./ut_extended.xml test_ops_xpu.py "
)
test_command += skip_options
res = os.system(test_command)
sys.exit(res)
