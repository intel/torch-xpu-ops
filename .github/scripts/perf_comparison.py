# To compare the performance diff
# Usage:
#   python perf_comparison.py --target /path/to/xpu/performance/result/dir --baseline /path/to/reference/dir

import re
import os
import fnmatch
import argparse
import subprocess
import pandas as pd
from statistics import geometric_mean

parser = argparse.ArgumentParser(description="Analysis", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--target", default=None, help="XPU performance result csv files dir")
parser.add_argument("--baseline", default=None, help="XPU refrerence result csv files dir")
parser.add_argument("--pr", action="store_true", help="Only show results xpu has")
parser.add_argument("--criteria", default=0.90, type=float, help="Criteria for comparison")
args = parser.parse_args()


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

def color_result(criteria, input):
    if input == -1:
        output = input
    elif input < criteria:
        output = f"🔴{input}"
    elif input > 1 - criteria + 1:
        output = f"🟢{input}"
    else:
        output = input
    return output

def get_filtered_data(owner, repo, issue_number, data):
    """Get skipped models list from issue body"""
    try:
        result = subprocess.run([
            'gh', '--repo', f'{owner}/{repo}',
            'issue', 'view', f'{issue_number}'
        ], capture_output=True, text=True, check=True)
        lines = result.stdout.splitlines()
        models = next((line.replace(" ", "") for line in lines if "Skipped models:" in line), data)
        at_index = models.find(":")
        models_list = models[at_index+1:].split(",")
        output = data[~data['Model'].isin(models_list) if models_list else True]
        return output
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return data

# comparison result output
output_header = ["Category", "Model",
                 "Target eager", "Target inductor", "Inductor vs. Eager [Target]",
                 "Baseline eager", "Baseline inductor", "Inductor vs. Eager [Baseline]",
                 "Target vs. Baseline [Eager]", "Target vs. Baseline [Inductor]"]
output_data = []

xpu_files = find_files("*_xpu_performance.csv", args.target)
for xpu_file in xpu_files:
    xpu_data = pd.read_csv(xpu_file)
    xpu_names = xpu_data["name"].tolist()
    refer_file = re.sub(args.target, args.baseline + "/", xpu_file, flags=re.IGNORECASE, count=1)
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
            eager_comparison = str(color_result(args.criteria, xpu_vs_refer_eager))
            inductor_comparison = str(color_result(args.criteria, xpu_vs_refer_inductor))
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
    refer_files = find_files("*_xpu_performance.csv", args.baseline)
    for refer_file in refer_files:
        refer_data = pd.read_csv(refer_file)
        refer_names = refer_data["name"].tolist()
        xpu_file = re.sub(args.baseline, args.target + "/", refer_file, flags=re.IGNORECASE, count=1)
        if not os.path.isfile(xpu_file):
            names = set(refer_names)
            names = sorted(names)
            for name in names:
                refer_value = next((row for index, row in refer_data.iterrows() if row["name"] == name), "")
                refer_eager_latency = refer_value["speedup"] * refer_value["abs_latency"]
                output_data.append([multiple_replace(refer_file), name, -1, -1, -1, refer_eager_latency, refer_value["abs_latency"], refer_value["speedup"], -1, -1])

# result
output_data = pd.DataFrame(output_data, columns=output_header)
output_data = output_data.sort_values(['Target vs. Baseline [Inductor]', 'Target vs. Baseline [Eager]'], ascending=[True, True])

# summary
geomean_sum = {"all": [], "huggingface": [], "timm_models": [], "torchbench": []}
for column_name in ["Target vs. Baseline [Inductor]", "Target vs. Baseline [Eager]", "Inductor vs. Eager [Target]"]:
    data = [row[column_name] for index, row in output_data.iterrows() if row[column_name] > 0]
    if len(data) > 0:
        geomean_sum["all"].append(color_result(args.criteria, geometric_mean(data)))
    else:
        geomean_sum["all"].append("🔴")
    for model_name in ["huggingface", "timm_models", "torchbench"]:
        data = [row[column_name] for index, row in output_data.iterrows() if row[column_name] > 0 and re.match(model_name, row["Category"])]
        if os.path.exists(args.target + "/" + model_name):
            if len(data) > 0:
                geomean_sum[model_name].append(color_result(args.criteria, geometric_mean(data)))
            else:
                geomean_sum[model_name].append("🔴")
geomean_sum = {k: v for k, v in geomean_sum.items() if v}
output_sum = pd.DataFrame(geomean_sum, index=["Target vs. Baseline [Inductor]", "Target vs. Baseline [Eager]", "Inductor vs. Eager [Target]"]).T
output = output_sum.to_html(header=True)
with open('performance.summary.html', 'w', encoding='utf-8') as f:
    f.write("\n\n#### performance\n\n" + output)

# details
output = output_data.to_html(index=False)
with open('performance.details.html', 'w', encoding='utf-8') as f:
    f.write("\n\n#### performance\n\n" + output)

# regression
filtered_data = get_filtered_data("intel", "torch-xpu-ops", "1645", output_data)
comparison = filtered_data.loc[
    ((filtered_data['Target vs. Baseline [Inductor]'] < args.criteria) | (filtered_data['Target vs. Baseline [Eager]'] < args.criteria))
    & (filtered_data['Baseline inductor'] > 0)
    & (filtered_data['Target inductor'] >= 0)
]
if not comparison.empty:
    output = comparison.to_html(index=False)
    with open('performance.regression.html', 'w', encoding='utf-8') as f:
        f.write("\n\n#### performance\n\n" + output)
