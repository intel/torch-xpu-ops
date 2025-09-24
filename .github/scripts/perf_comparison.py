# To compare the performance diff
# Usage:
#   python perf_comparison.py --xpu /path/to/xpu/performance/result/dir --refer /path/to/reference/dir

import re
import os
import json
import fnmatch
import argparse
import pandas as pd
from statistics import geometric_mean

parser = argparse.ArgumentParser(description="Analysis", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--xpu", default=None, help="XPU performance result csv files dir")
parser.add_argument("--refer", default=None, help="XPU refrerence result csv files dir")
parser.add_argument("--pr", action="store_true", help="Only show results xpu has")
args = parser.parse_args()


ci_config_file = os.environ.get('GITHUB_WORKSPACE') + "/.github/ci_expected_accuracy/models_list.json"

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
with open(ci_config_file) as f:
    config = json.load(f)

xpu_files = find_files("*_xpu_performance.csv", args.xpu)
for xpu_file in xpu_files:
    xpu_data = pd.read_csv(xpu_file)
    xpu_names = xpu_data["name"].tolist()
    if args.pr:
        if 'timm_models' in xpu_file and config.get("timm_models"):
            xpu_names = config.get("timm_models")
        elif 'torchbench' in xpu_file and config.get("torchbench"):
            xpu_names = config.get("torchbench")
    refer_file = re.sub(args.xpu, args.refer + "/", xpu_file, flags=re.IGNORECASE)
    if os.path.isfile(refer_file):
        refer_data= pd.read_csv(refer_file)
        refer_names = [row["name"] for index, row in refer_data.iterrows()]
        names = set(xpu_names)
        names = sorted(names)
        for name in names:
            # xpu info
            xpu_value = next((row for index, row in xpu_data.iterrows() if row["name"] == name), None)
            xpu_eager_latency = xpu_value["speedup"] * xpu_value["abs_latency"] if xpu_value is not None else -1
            xpu_inductor_latency = xpu_value["abs_latency"] if xpu_value is not None else -1
            xpu_indcutor_vs_eager = xpu_value["speedup"] if xpu_value is not None else -1 # higher is better
            # refer info
            refer_value = next((row for index, row in refer_data.iterrows() if row["name"] == name), None)
            refer_eager_latency = float(refer_value["speedup"]) * float(refer_value["abs_latency"]) if refer_value is not None else -1
            refer_inductor_latency = refer_value["abs_latency"] if refer_value is not None else -1
            refer_indcutor_vs_eager = refer_value["speedup"] if refer_value is not None else -1 # higher is better
            # xpu vs. refer
            xpu_vs_refer_eager = refer_eager_latency / xpu_eager_latency  if xpu_value is not None and refer_value is not None and xpu_eager_latency > 0 else 0 # higher is better
            xpu_vs_refer_inductor = float(refer_value["abs_latency"]) / xpu_value["abs_latency"] if xpu_value is not None and refer_value is not None and xpu_value["abs_latency"] > 0 else 0 # higher is better
            # output data
            output_data.append([multiple_replace(xpu_file), name, xpu_eager_latency, xpu_inductor_latency, xpu_indcutor_vs_eager, refer_eager_latency, refer_inductor_latency, refer_indcutor_vs_eager, xpu_vs_refer_eager, xpu_vs_refer_inductor])
    else:
        names = set(xpu_names)
        names = sorted(names)
        for name in names:
            xpu_value = next((row for index, row in xpu_data.iterrows() if row["name"] == name), "")
            xpu_eager_latency = xpu_value["speedup"] * xpu_value["abs_latency"]
            output_data.append([multiple_replace(xpu_file), name, xpu_eager_latency, xpu_value["abs_latency"], xpu_value["speedup"], -1, -1, -1, -1, -1])
if not args.pr:
    refer_files = find_files("*_xpu_performance.csv", args.refer)
    for refer_file in refer_files:
        refer_data = pd.read_csv(refer_file)
        refer_names = refer_data["name"].tolist()
        xpu_file = re.sub(args.refer, args.xpu + "/", refer_file, flags=re.IGNORECASE)
        if not os.path.isfile(xpu_file):
            names = set(refer_names)
            names = sorted(names)
            for name in names:
                refer_value = next((row for index, row in refer_data.iterrows() if row["name"] == name), "")
                refer_eager_latency = refer_value["speedup"] * refer_value["abs_latency"]
                output_data.append([multiple_replace(refer_file), name, -1, -1, -1, refer_eager_latency, refer_value["abs_latency"], refer_value["speedup"], -1, -1])

# summary
output_data = pd.DataFrame(output_data, columns=output_header)
geomean_list = {"Category": "Geomean"}
for column_name in ["Inductor vs. Eager [Target]", "Target vs. Baseline [Eager]", "Target vs. Baseline [Inductor]"]:
    data = [row[column_name] for index, row in output_data.iterrows() if row[column_name] > 0]
    if len(data) > 0:
        geomean_list[column_name + " | all"] = geometric_mean(data)
    for model_name in ["huggingface", "timm_models", "torchbench"]:
        data = [row[column_name] for index, row in output_data.iterrows() if row[column_name] > 0 and re.match(model_name, row["Category"])]
        if len(data) > 0:
            geomean_list[column_name + " | " + model_name] = geometric_mean(data)

# get output
output_sum = pd.DataFrame.from_dict([geomean_list]).T
output = output_sum.to_html(header=False)
print(output)
output_data = output_data.sort_values(['Target vs. Baseline [Inductor]', 'Target vs. Baseline [Eager]'], ascending=[True, True])
output = output_data.to_html(index=False)
print("\n", output)

# get comparison result
criteria = 0.95
comparison = output_data.loc[(output_data['Target vs. Baseline [Inductor]'] < criteria) | (output_data['Target vs. Baseline [Eager]'] < criteria)]
output = comparison.to_html(index=False)
print("\n", output)
