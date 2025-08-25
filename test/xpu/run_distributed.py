import os
import subprocess
import sys

from skip_list_dist import skip_dict
from xpu_test_utils import launch_test

res = 0
res2 = 0
fail_test = []

# libfabric WA to avoid hang issue
os.environ["FI_PROVIDER"] = "tcp"
# os.environ["ZE_AFFINITY_MASK"] = "0,1,2,3"


# run python test
def run(test_command):
    result = subprocess.run(test_command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    if "FAILED" in result.stdout or "FAILED" in result.stderr:
        fail_test.append(" ".join(test_command))
    return result.returncode


test_command = ["python", "distributed/test_c10d_ops_xccl.py"]
res += run(test_command)

# run pytest with skiplist
for key in skip_dict:
    skip_list = skip_dict[key]
    fail = launch_test(key, skip_list)
    res2 += fail
    if fail:
        fail_test.append(key)

if fail_test:
    print(",".join(fail_test) + " have failures")

exit_code = os.WEXITSTATUS(res2)
if exit_code == 0:
    sys.exit(res)
else:
    sys.exit(exit_code)
