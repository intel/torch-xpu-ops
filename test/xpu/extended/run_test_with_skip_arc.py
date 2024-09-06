import os
import sys
from skip_list_common import skip_dict
from skip_list_arc import skip_dict as skip_dict_specifical
from skip_list_win import skip_dict as skip_dict_win
from skip_list_win_arc import skip_dict as skip_dict_win_arc
from xpu_test_utils import launch_test

IS_WINDOWS = sys.platform == "win32"

skip_list = skip_dict["test_ops_xpu.py"] + skip_dict_specifical["test_ops_xpu.py"]
if IS_WINDOWS:
    skip_list += skip_dict_win["test_ops_xpu.py"] + skip_dict_win_arc["test_ops_xpu.py"]

return_code, count_buf, fails = launch_test("test_ops_xpu.py", skip_list)

if fails:
    print("="*10," failures list ","="*10)
    for fail in fails:
        print(fail)
print("="*10," case count ","="*10)
print(count_buf)

if os.name == "nt":
    sys.exit(return_code)
else:    
    exit_code = os.WEXITSTATUS(return_code)
    sys.exit(exit_code)