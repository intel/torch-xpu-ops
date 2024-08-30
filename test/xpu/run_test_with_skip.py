import os
import sys

from skip_list_common import skip_dict
from xpu_test_utils import launch_test


res = 0
fail_test = []

for key in skip_dict:
    skip_list = skip_dict[key]
    fail = launch_test(key, skip_list)
    res += fail
    if fail:
        fail_test.append(key)
if fail_test:
    print(",".join(fail_test) + " have failures")

exit_code = os.WEXITSTATUS(res)
sys.exit(exit_code)
