import os
import sys

from skip_list_common import skip_dict
from xpu_test_utils import launch_test


res = 0
fails = []
count = ""

for key in skip_dict:
    skip_list = skip_dict[key]
    return_code, count_buf, fail = launch_test(key, skip_list)
    res += return_code
    count=count+count_buf+"\n"
    if return_code:
        fails.extend(fail)
if fails:
    print("="*10," failures list ","="*10)
    for fail in fails:
        print(fail)
print("="*10," case count ","="*10)
print(count)


if os.name == "nt":
    sys.exit(res)
else:    
    exit_code = os.WEXITSTATUS(res)
    sys.exit(exit_code)
