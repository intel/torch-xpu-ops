import os
import sys
from skip_list_common import skip_dict
from skip_list_mtl import skip_dict as skip_dict_specifical
from xpu_test_utils import launch_test

res = 0

for key in skip_dict:
    skip_list = skip_dict[key] if key not in skip_dict_specifical else skip_dict[key] + skip_dict_specifical[key]
    res += launch_test(key, skip_list)

exit_code = os.WEXITSTATUS(res)
sys.exit(exit_code)
