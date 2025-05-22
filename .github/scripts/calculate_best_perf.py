
# To get the best performance number
# Usage:
#   python calculate_best_perf.py --best /path/to/best.csv
#               --new /path/to/new/measured/performance/result/dir
#               --device <Used device name>
#               --os <Used OS version>
#               --driver <Used driver version>
#               --oneapi <Used oneapi version>
#               --gcc <Used gcc version>
#               --python <Used python version>
#               --pytorch <Used pytorch version>
#               --torch-xpu-ops <Used torch-xpu-ops version>

import re
import os
import fnmatch
import argparse
import pandas as pd
from datetime import date

parser = argparse.ArgumentParser(description="Get Best Performance",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--best", required=True, help="Saved best performance file")
parser.add_argument("--new", required=True, help="New round launch")
parser.add_argument("--device", default=None, type=str, help="Device name, such as PVC1100")
parser.add_argument("--os", default=None, type=str, help="OS version, such as Ubuntu 22.04")
parser.add_argument("--driver", default=None, type=str, help="Driver version, such as 25.05.32567")
parser.add_argument("--oneapi", default=None, type=str, help="OneAPI version, such as 2025.1")
parser.add_argument("--gcc", default=None, type=str, help="GCC version, such as 11")
parser.add_argument("--python", default=None, type=str, help="Python version, such as 3.10")
parser.add_argument("--pytorch", default=None, type=str, help="PyTorch version")
parser.add_argument("--torch-xpu-ops", default=None, type=str, help="Torch XPU Ops version")
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

# comparison result output
best_header = ["Category", "Model", "Value Type", "Value",
               "Device", "OS", "Driver", "OneAPI", "GCC", "Python",
               "PyTorch", "Torch XPU Ops", "Date"]
best_data = pd.read_csv(args.best) if os.path.isfile(args.best) else pd.DataFrame(columns=best_header)
best_data = best_data.reset_index(drop=True)
new_files = find_files("*_xpu_performance.csv", args.new)
for new_file in new_files:
    category = multiple_replace(new_file)
    new_data = pd.read_csv(new_file)
    new_data = new_data.reset_index(drop=True)
    for index, row in new_data.iterrows():
        # eager
        new_eager = row["abs_latency"] * row["speedup"]
        eager_row = next(([i, line] for i, line in best_data.iterrows()
                          if (line["Category"] == category
                              and line["Model"] == row["name"]
                              and line["Value Type"] == "eager")), "N/A")
        best_eager_value = best_data.loc[
            (best_data["Category"] == category) &
            (best_data["Model"] == row["name"]) &
            (best_data["Value Type"] == "eager")]
        if eager_row != "N/A":
            if new_eager < best_eager_value["Value"].values[0]:
                best_data.loc[eager_row[0]] = [category, row["name"], "eager", new_eager,
                    args.device, args.os, args.driver, args.oneapi, args.gcc, args.python,
                    args.pytorch, args.torch_xpu_ops, date.today().strftime('%F')]
        else:
            best_data.loc[len(best_data), :] = None
            best_data.loc[len(best_data) - 1] = [category, row["name"], "eager", new_eager,
                args.device, args.os, args.driver, args.oneapi, args.gcc, args.python,
                args.pytorch, args.torch_xpu_ops, date.today().strftime('%F')]
        # inductor
        inductor_row = next(([i, line] for i, line in best_data.iterrows()
                             if (line["Category"] == category
                                 and line["Model"] == row["name"]
                                 and line["Value Type"] == "inductor")), "N/A")
        best_inductor_value = best_data.loc[
            (best_data["Category"] == category) &
            (best_data["Model"] == row["name"]) &
            (best_data["Value Type"] == "inductor")]
        if inductor_row != "N/A":
            if row["abs_latency"] < best_inductor_value["Value"].values[0]:
                best_data.loc[inductor_row[0]] = [category, row["name"], "inductor", row["abs_latency"],
                    args.device, args.os, args.driver, args.oneapi, args.gcc, args.python,
                    args.pytorch, args.torch_xpu_ops, date.today().strftime('%F')]
        else:
            best_data.loc[len(best_data), :] = None
            best_data.loc[len(best_data) - 1] = [category, row["name"], "inductor", row["abs_latency"],
                args.device, args.os, args.driver, args.oneapi, args.gcc, args.python,
                args.pytorch, args.torch_xpu_ops, date.today().strftime('%F')]

best_data.to_csv(args.best, sep=',', encoding='utf-8', index=False)
