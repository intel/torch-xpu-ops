import os
import sys

from skip_list_common import skip_dict
from xpu_test_utils import get_device_name, launch_test

dev_name = get_device_name()
if dev_name == "ARC":
    from skip_list_arc import skip_dict as skip_dict_specifical
    from skip_list_win import skip_dict as skip_dict_win
    from skip_list_win_arc import skip_dict as skip_dict_win_arc
elif dev_name == "MTL":
    from skip_list_mtl import skip_dict as skip_dict_specifical
    from skip_list_win import skip_dict as skip_dict_win

res = 0
fail_test = []
IS_WINDOWS = sys.platform == "win32"

for key in skip_dict:
    skip_list = skip_dict[key]
    if dev_name == "ARC":
        if key in skip_dict_specifical:
            skip_list += skip_dict_specifical[key]
        if IS_WINDOWS and key in skip_dict_win:
            skip_list += skip_dict_win[key]
        if IS_WINDOWS and key in skip_dict_win_arc:
            skip_list += skip_dict_win_arc[key]
    elif dev_name == "MTL":
        if key in skip_dict_specifical:
            skip_list += skip_dict_specifical[key]
        if IS_WINDOWS and key in skip_dict_win:
            skip_list += skip_dict_win[key]
    fail = launch_test(key, skip_list)
    res += fail
    if fail:
        fail_test.append(key)
if fail_test:
    print(",".join(fail_test) + " have failures")

if os.name == "nt":
    sys.exit(res)
else:    
    exit_code = os.WEXITSTATUS(res)
    sys.exit(exit_code)
