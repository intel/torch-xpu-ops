import os
import subprocess
import sys

from skip_list_dist import skip_dict, skip_dict_python
from xpu_test_utils import launch_test

res = 0
res2 = 0
fail_test = []


# run python test
def run(test_command):
    result = subprocess.run(test_command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    return result


os.environ["PYTHONPATH"] = "$PYTHONPATH:../../../../test/distributed/pipelining"
for key in skip_dict_python:
    skip_list = skip_dict_python[key]
    test_command = ["python", key]
    fail = run(test_command)
    if fail.returncode:
        for line in fail.stderr.split("\n"):
            if "FAIL: " in line:
                is_error = True
                for skip_case in skip_list:
                    if skip_case in line:
                        print("Skiped error: ", key + " " + skip_case)
                        is_error = False
                if is_error:
                    res += fail.returncode
                    fail_test.append("".join(key + " " + line))

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
