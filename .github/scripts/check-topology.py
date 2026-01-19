import os
import sys

# Get the xelink group card affinity
ret = os.system("xpu-smi topology -m 2>&1|tee topology.log > /dev/null")
if ret == 0:
    gpu_dict = {}
    cpu_dict = {}
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
                cpu_affinity = items[-1].split(",")[0]
                i = gpu_id.split("/")[0]
                affinity = ""
                for j, item in enumerate(items):
                    if "SYS" not in item and ("XL" in item or "S" in item):
                        if len(affinity) == 0:
                            affinity = str(j - 2)
                        else:
                            affinity = affinity + "," + str(j - 2)
                gpu_dict[i] = affinity
                cpu_dict[i] = cpu_affinity

    value_to_keys = {}
    gpu_cpu_dict = {}
    for key, value in gpu_dict.items():
        if value not in value_to_keys:
            value_to_keys[value] = []
        value_to_keys[value].append(key)
    dist_group = []
    for key, value in value_to_keys.items():
        if key == ','.join(value_to_keys[key]):
            dist_group.append(key)
    for group in dist_group:
        cpu_aff = []
        for i in group.split(","):
            if cpu_dict[i] not in cpu_aff:
                cpu_aff.append(cpu_dict[i])
        if len(cpu_aff) == 1:
            gpu_cpu_dict[group] = ','.join(cpu_aff)
    # if len(gpu_cpu_dict) == 0:
    #     print("No Xelink detected")
    #     sys.exit(255)
    pytest_extra_args = ""
    for key, value in gpu_cpu_dict.items():
        start_cpu = int(value.split("-")[0])
        end_cpu = int(value.split("-")[1])
        threads = end_cpu - start_cpu + 1
        pytest_extra_args = pytest_extra_args + \
            ' --tx popen//env:ZE_AFFINITY_MASK=%s//env:OMP_NUM_THREADS=%d//python="numactl -l -C %s python"'\
            %(key, threads, value)
    print(pytest_extra_args)

else:
    print("xpu-smi topology failed")

    sys.exit(255)
