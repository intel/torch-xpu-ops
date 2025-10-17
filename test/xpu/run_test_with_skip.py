import os
import sys
import argparse

from skip_list_common import skip_dict
from xpu_test_utils import launch_test


parser = argparse.ArgumentParser(description='Run specific unit tests')
# By default, run the cases without the skipped cases
parser.add_argument('--run', choices=['selected', 'skipped', 'all'], default='selected', help='Test cases scope')
args = parser.parse_args()


res = 0
fail_test = []

for key in skip_dict:
    skip_list = skip_dict[key]
    exe_list = None
    if args.run == "skipped":
        skip_list = None
        exe_list = skip_list
    elif args.run == "all":
        skip_list = None
    fail = launch_test(key, skip_list=skip_list, exe_list=exe_list)
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
