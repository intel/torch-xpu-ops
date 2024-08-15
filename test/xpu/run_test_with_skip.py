import os
import sys
from skip_list_common import skip_dict
from xpu_test_utils import launch_test


res = 0

for key in skip_dict:
    skip_list = skip_dict[key]
    res += launch_test(key, skip_list)

exit_code = os.WEXITSTATUS(res)
sys.exit(exit_code)
