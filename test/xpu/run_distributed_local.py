import os
import subprocess
import sys

from skip_list_dist_local import skip_dict
from xpu_test_utils import launch_test

res = 0
fail_test = []

os.environ["CCL_ATL_TRANSPORT"] = "ofi"
os.environ["CCL_SEND"] = "direct"
os.environ["CCL_RECV"] = "direct" 
# Get the xelink group card affinity
ret = os.system("xpu-smi topology -m 2>&1|tee topology.log")
if ret == 0:
    gpu_dict = {}
    with open("topology.log", "r") as file:
        lines = file.readlines()
        for line in lines:
           if "CPU Affinity" in line:
              continue
           line = line.strip()
           if line.startswith("GPU "):
               items = line.split(' ')
               items = [x for x in items if x]
               gpu_id = items[1]
               i = gpu_id.split('/')[0]
               affinity = ""
               for j, item in enumerate(items):
                   if "SYS" not in item and ( "XL" in item or "S" in item ):
                      if len(affinity) == 0:
                          affinity = str(j-2)
                      else:
                          affinity = affinity + ',' + str(j-2)
               gpu_dict[i] = affinity
    
    
    max_affinity = ""
    for key, value in gpu_dict.items():
        if  len(value) > len(max_affinity):
            max_affinity = value
    
    os.environ["ZE_AFFINITY_MASK"] = str(max_affinity)
    print(str("ZE_AFFINITY_MASK=" + os.environ.get("ZE_AFFINITY_MASK")))

else:
    print("xpu-smi topology failed")
    sys.exit(255)

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
    sys.exit(res)
else:
    sys.exit(exit_code)
