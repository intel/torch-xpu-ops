import os
import subprocess
import sys

from skip_list_dist_local import skip_dict, skip_dict_python
from xpu_test_utils import launch_test

res = 0
res2 = 0
fail_test = []

os.environ["CCL_ATL_TRANSPORT"] = "ofi"
os.environ["CCL_SEND"] = "direct"
os.environ["CCL_RECV"] = "direct"
os.environ["PYTHONPATH"] = "$PYTHONPATH:../../../../test/distributed/pipelining"
# Get the xelink group card affinity
ret = os.system("xpu-smi topology -m 2>&1|tee topology.log")
if ret == 0:
    gpu_dict = {}
    with open("topology.log") as file:
        lines = file.readlines()
        for line in lines:
            if "CPU Affinity" in line:
                continue
            line = line.strip()
            if line.startswith("GPU "):
                items = line.split(" ")
                items = [x for x in items if x]
                gpu_id = items[1]
                i = gpu_id.split("/")[0]
                affinity = ""
                for j, item in enumerate(items):
                    if "SYS" not in item and ("XL" in item or "S" in item):
                        if len(affinity) == 0:
                            affinity = str(j - 2)
                        else:
                            affinity = affinity + "," + str(j - 2)
                gpu_dict[i] = affinity

    max_affinity = ""
    for key, value in gpu_dict.items():
        if len(value) > len(max_affinity):
            max_affinity = value

    os.environ["ZE_AFFINITY_MASK"] = str(max_affinity)
    print(str("ZE_AFFINITY_MASK=" + os.environ.get("ZE_AFFINITY_MASK")))

else:
    print("xpu-smi topology failed")
    sys.exit(255)


# run python test
def run(test_command):
    result = subprocess.run(test_command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    return result


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
                    res2 += fail.returncode
                    fail_test.append("".join(key + " " + line))

# run pytest with skiplist
for key in skip_dict:
    skip_list = skip_dict[key]
    fail = launch_test(key, skip_list)
    res += fail
    if fail:
        fail_test.append(key)

if fail_test:
    print(",".join(fail_test) + " have failures")

exit_code = os.WEXITSTATUS(res)
if exit_code == 0:
    sys.exit(res2)
else:
    sys.exit(exit_code)
