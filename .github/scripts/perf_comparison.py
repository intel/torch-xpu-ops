
import re
import os
import sys
import fnmatch
import argparse
import pandas as pd
from statistics import geometric_mean

parser = argparse.ArgumentParser(description="Analysis", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-xpu", default=None, help="XPU file")
parser.add_argument("-cuda", default=None, help="CUDA file")
args = parser.parse_args()

# the_two = next((x for x in primitive_list[args.files_path[1]] if x.name == p), None)
def multiple_replace(text):
    REGEX_REPLACEMENTS = [
        (r".*inductor_", ""),
        (r"_xpu_performance.csv", ""),
    ]
    for old, new in REGEX_REPLACEMENTS:
        text = re.sub(old, new, text, flags=re.IGNORECASE)
    return text

def find_files(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

# comparison result output
output_header = ["Category", "Model",
                 "Target eager", "Target inductor", "Inductor vs. Eager [Target]",
                 "Baseline eager", "Baseline inductor", "Inductor vs. Eager [Baseline]",
                 "Target vs. Baseline [Eager]", "Target vs. Baseline [Inductor]"]
output_data = []
xpu_files = find_files("*_xpu_performance.csv", args.xpu)
for xpu_file in xpu_files:
    xpu_data = pd.read_csv(xpu_file)
    # xpu_data = xpu_data.reset_index()  # make sure indexes pair with number of rows
    xpu_names = [row["name"] for index, row in xpu_data.iterrows()]
    cuda_file = re.sub(args.xpu, args.cuda + "/", xpu_file, flags=re.IGNORECASE)
    if os.path.isfile(cuda_file):
        cuda_data= pd.read_csv(cuda_file)
        # cuda_data = cuda_data.reset_index()  # make sure indexes pair with number of rows
        cuda_names = [row["name"] for index, row in cuda_data.iterrows()]
        names = xpu_names + cuda_names
        names = set(names)
        names = sorted(names)
        for name in names:
            # xpu info
            xpu_value = next((row for index, row in xpu_data.iterrows() if row["name"] == name), None)
            xpu_eager_latency = xpu_value["speedup"] * xpu_value["abs_latency"] if xpu_value is not None else -1
            xpu_inductor_latency = xpu_value["abs_latency"] if xpu_value is not None else -1
            xpu_indcutor_vs_eager = xpu_value["speedup"] if xpu_value is not None else -1 # higher is better
            # cuda info
            cuda_value = next((row for index, row in cuda_data.iterrows() if row["name"] == name), None)
            cuda_eager_latency = float(cuda_value["speedup"]) * float(cuda_value["abs_latency"]) if cuda_value is not None else -1
            cuda_inductor_latency = cuda_value["abs_latency"] if cuda_value is not None else -1
            cuda_indcutor_vs_eager = cuda_value["speedup"] if cuda_value is not None else -1 # higher is better
            # xpu vs. cuda
            xpu_vs_cuda_eager = cuda_eager_latency / xpu_eager_latency  if xpu_value is not None and cuda_value is not None and xpu_eager_latency > 0 else 0 # higher is better
            xpu_vs_cuda_inductor = float(cuda_value["abs_latency"]) / xpu_value["abs_latency"] if xpu_value is not None and cuda_value is not None and xpu_value["abs_latency"] > 0 else 0 # higher is better
            # output data
            output_data.append([multiple_replace(xpu_file), name, xpu_eager_latency, xpu_inductor_latency, xpu_indcutor_vs_eager, cuda_eager_latency, cuda_inductor_latency, cuda_indcutor_vs_eager, xpu_vs_cuda_eager, xpu_vs_cuda_inductor])
    else:
        names = set(xpu_names)
        names = sorted(names)
        for name in names:
            xpu_value = next((row for index, row in xpu_data.iterrows() if row["name"] == name), "")
            xpu_eager_latency = xpu_value["speedup"] * xpu_value["abs_latency"]
            output_data.append([multiple_replace(xpu_file), name, xpu_eager_latency, xpu_value["abs_latency"], xpu_value["speedup"], -1, -1, -1, -1, -1])
cuda_files = find_files("*_xpu_performance.csv", args.cuda)
for cuda_file in cuda_files:
    cuda_data = pd.read_csv(cuda_file)
    # cuda_data = cuda_data.reset_index()  # make sure indexes pair with number of rows
    cuda_names = [row["name"] for index, row in cuda_data.iterrows()]
    xpu_file = re.sub(args.cuda, args.xpu + "/", cuda_file, flags=re.IGNORECASE)
    if not os.path.isfile(xpu_file):
        names = set(cuda_names)
        names = sorted(names)
        for name in names:
            cuda_value = next((row for index, row in cuda_data.iterrows() if row["name"] == name), "")
            cuda_eager_latency = cuda_value["speedup"] * cuda_value["abs_latency"]
            output_data.append([multiple_replace(cuda_file), name, -1, -1, -1, cuda_eager_latency, cuda_value["abs_latency"], cuda_value["speedup"], -1, -1])

# summary
output_data = pd.DataFrame(output_data, columns=output_header)
geomean_list = {"Category": "Geomean"}
for column_name in ["Inductor vs. Eager [Target]", "Target vs. Baseline [Eager]", "Target vs. Baseline [Inductor]"]:
    data = [row[column_name] for index, row in output_data.iterrows() if row[column_name] > 0]
    if len(data) > 0:
        geomean_list[column_name + "all"] = geometric_mean(data)
    for model_name in ["huggingface", "timm_models", "torchbench"]:
        data = [row[column_name] for index, row in output_data.iterrows() if row[column_name] > 0 and re.match(model_name, row["Category"])]
        if len(data) > 0:
            geomean_list[column_name + model_name] = geometric_mean(data)
# get output
output_sum = pd.DataFrame.from_dict([geomean_list]).T
# output_sum = pd.DataFrame(output_sum, columns=["Category", "Geomean"])
output = output_sum.to_html(header=False)
print(output)
output = output_data.to_html(index=False)
print("\n", output)
