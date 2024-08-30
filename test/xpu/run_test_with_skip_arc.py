import os
import sys
from skip_list_common import skip_dict
from skip_list_arc import skip_dict as skip_dict_specifical
from skip_list_win import skip_dict as skip_dict_win
from skip_list_win_arc import skip_dict as skip_dict_win_arc
from xpu_test_utils import launch_test


res = 0
IS_WINDOWS = sys.platform == "win32"

for key in skip_dict:
    skip_list = skip_dict[key]
    if key in skip_dict_specifical:
        skip_list += skip_dict_specifical[key]
    if IS_WINDOWS and key in skip_dict_win:
        skip_list += skip_dict_win[key]
    if IS_WINDOWS and key in skip_dict_win_arc:
        skip_list += skip_dict_win_arc[key]
    res += launch_test(key, skip_list)

exit_code = os.WEXITSTATUS(res)
sys.exit(exit_code)
