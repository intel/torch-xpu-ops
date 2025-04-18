import os
import subprocess
import sys

from skip_list_dist_local import skip_dict, skip_dict_python

res = 0
res2 = 0
fail_test = []
error_log = ""

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


from xpu_test_utils import launch_test

# run python test
def run(test_command):
    result = subprocess.run(test_command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    return result


for key in skip_dict_python:
    skip_list = skip_dict_python[key] if skip_dict_python[key] else []
    test_command = ["python", key]
    fail = run(test_command)
    num_skipped = 0
    num_err = 0
    if fail.returncode:
        for i, err in enumerate(fail.stderr.split("FAIL: ")):
            if i == 0 and len(err) > 0:
                error_log += err
                continue
            is_skipped = False
            for skip_case in skip_list:
                if skip_case in err:
                    print("Skipped error: ", key + " " + skip_case)
                    num_skipped += 1
                    is_skipped = True
                    break
            if not is_skipped:
                num_err += 1
                res2 += fail.returncode
                if i == len(fail.stderr.split("FAIL: ")) - 1:
                    error_log += "FAIL: "
                    for line in err.split("\n"):
                        if line.startswith("FAILED (failures="):
                            num_errs = line.split("=")[1].split(")")[0].strip()
                            error_log += ("FAILED (failures=" + str(int(num_errs) - num_skipped) + f" skipped {num_skipped} cases" + ")\n")
                        else:
                            error_log += (line + "\n")
                else:
                    error_log += ("FAIL: " + err)
            else:
                if i == len(fail.stderr.split("FAIL: ")) - 1:
                    error_log += "FAIL: "
                    for line in err.split("\n"):
                        if line.startswith("FAILED (failures="):
                            num_errs = line.split("=")[1].split(")")[0].strip()
                            error_log += ("FAILED (failures=" + str(int(num_errs) - num_skipped) + f" skipped {num_skipped} cases" + ")\n")

    renamed_key = key.replace("../../../../", "").replace("/", "_")
    if num_err > 0:
        fail_test.append(key)
        with open(f"op_ut_with_skip_{renamed_key}.log", "w") as f:
            f.write(error_log)
    else:
        with open(f"op_ut_with_skip_{renamed_key}.log", "w") as f:
            f.write(fail.stdout)
            f.write(fail.stderr)

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
