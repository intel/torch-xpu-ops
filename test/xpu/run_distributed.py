# Copyright 2020-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

import subprocess
import sys

from skip_list_dist import skip_dict
from xpu_test_utils import launch_test

res = 0
res2 = 0
fail_test = []


# run python test
def run(test_command):
    result = subprocess.run(test_command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    if "FAILED" in result.stdout or "FAILED" in result.stderr:
        fail_test.append(" ".join(test_command))
    return result.returncode


test_command = ["python", "../../../../test/distributed/pipelining/test_backward.py"]
res += run(test_command)
test_command = ["python", "../../../../test/distributed/pipelining/test_microbatch.py"]
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

sys.exit(res)
