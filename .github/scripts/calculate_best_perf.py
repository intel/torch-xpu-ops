
import re
import os
import fnmatch
import argparse
import pandas as pd
from datetime import date

parser = argparse.ArgumentParser(description="Get Best Performance",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--best", default=None, required=True, help="Saved best performance file")
parser.add_argument("--new", default=None, required=True, help="New round launch")
parser.add_argument("--device", default=None, type=str, help="Device")
parser.add_argument("--os", default=None, type=str, help="OS")
parser.add_argument("--driver", default=None, type=str, help="Driver")
parser.add_argument("--oneapi", default=None, type=str, help="OneAPI")
parser.add_argument("--gcc", default=None, type=str, help="GCC")
parser.add_argument("--python", default=None, type=str, help="Python")
parser.add_argument("--pytorch", default=None, type=str, help="PyTorch")
parser.add_argument("--torch-xpu-ops", default=None, type=str, help="Torch XPU Ops")
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
        new_eager = row["abs_latency"] * row["speedup"]
        # eager
        eager_row = next(([i, line] for i, line in best_data.iterrows()
                          if (line["Category"] == category
                              and line["Model"] == row["name"]
                              and line["Value Type"] == "eager")), "N/A")
        best_eager_value = best_data.loc[
            (best_data["Category"] == category) &
            (best_data["Model"] == row["name"]) &
            (best_data["Value Type"] == "eager")]
        if eager_row != "N/A":
            if new_eager < best_eager_value["Value"].values:
                best_data.loc[eager_row[0]] = category, row["name"], "eager", new_eager, \
                    args.device, args.os, args.driver, args.oneapi, args.gcc, args.python, \
                    args.pytorch, args.torch_xpu_ops, date.today().strftime('%F')
        else:
            best_data.loc[len(best_data), :] = None
            best_data.loc[len(best_data) - 1] = category, row["name"], "eager", new_eager, \
                args.device, args.os, args.driver, args.oneapi, args.gcc, args.python, \
                args.pytorch, args.torch_xpu_ops, date.today().strftime('%F')
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
            if row["abs_latency"] < best_inductor_value["Value"].values:
                best_data.at[inductor_row[0]] = category, row["name"], "inductor", row["abs_latency"], \
                    args.device, args.os, args.driver, args.oneapi, args.gcc, args.python, \
                    args.pytorch, args.torch_xpu_ops, date.today().strftime('%F')
        else:
            best_data.loc[len(best_data), :] = None
            best_data.loc[len(best_data) - 1] = category, row["name"], "inductor", row["abs_latency"], \
                args.device, args.os, args.driver, args.oneapi, args.gcc, args.python, \
                args.pytorch, args.torch_xpu_ops, date.today().strftime('%F')

best_data.to_csv(args.best, sep=',', encoding='utf-8', index=False)
