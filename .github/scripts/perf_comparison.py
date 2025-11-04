#!/usr/bin/env python3
"""
Performance comparison script.

To compare the performance diff
Usage:
    python perf_comparison.py --target /path/to/xpu/performance/result/dir --baseline /path/to/reference/dir
"""

import re
import os
import fnmatch
import argparse
import pandas as pd
from statistics import geometric_mean

parser = argparse.ArgumentParser(
    description="Analysis",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "--target",
    default=None,
    help="XPU performance result csv files dir"
)
parser.add_argument(
    "--baseline",
    default=None,
    help="XPU reference result csv files dir"
)
parser.add_argument(
    "--pr",
    action="store_true",
    help="Only show results xpu has"
)
args = parser.parse_args()


def multiple_replace(text):
    """Apply multiple regex replacements to text."""
    regex_replacements = [
        (r".*inductor_", ""),
        (r"_xpu_performance.csv", ""),
    ]
    for old, new in regex_replacements:
        text = re.sub(old, new, text, flags=re.IGNORECASE)
    return text


def find_files(pattern, path):
    """Find files matching pattern in directory tree."""
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


def color_result(input_val):
    """Add color coding to performance results."""
    if input_val == -1:
        output = input_val
    elif input_val < 0.8:
        output = f"🔴{input_val}"
    elif input_val < 0.9:
        output = f"🟡{input_val}"
    elif input_val > 1.1:  # Fixed: 1 + 0.1 -> 1.1
        output = f"🟢{input_val}"
    else:
        output = input_val
    return output


def write_report(cases, filename, message, method):
    """Helper function to write reports."""
    if not cases.empty:
        output = cases.to_html(index=False)
        with open(filename, method, encoding='utf-8') as file:
            file.write(f"\n\n{message}\n\n{output}")


def calculate_latencies(value):
    """Calculate eager and inductor latencies from value row."""
    if value is None:
        return -1, -1, -1
    eager_latency = value["speedup"] * value["abs_latency"]
    inductor_latency = value["abs_latency"]
    inductor_vs_eager = value["speedup"]
    return eager_latency, inductor_latency, inductor_vs_eager


def process_comparison_data():
    """Process and compare performance data between target and baseline."""
    # Comparison result output
    output_header = [
        "Category", "Model", "Target eager", "Target inductor",
        "Inductor vs. Eager [Target]", "Baseline eager", "Baseline inductor",
        "Inductor vs. Eager [Baseline]", "Target vs. Baseline [Eager]",
        "Target vs. Baseline [Inductor]"
    ]
    output_data = []

    xpu_files = find_files("*_xpu_performance.csv", args.target)
    for xpu_file in xpu_files:
        xpu_data = pd.read_csv(xpu_file)
        xpu_names = xpu_data["name"].tolist()
        refer_file = re.sub(
            args.target,
            args.baseline + "/",
            xpu_file,
            flags=re.IGNORECASE,
            count=1
        )

        if os.path.isfile(refer_file):
            refer_data = pd.read_csv(refer_file)
            refer_names = [row["name"] for index, row in refer_data.iterrows()]
            names = set(xpu_names)
            names = sorted(names)

            for name in names:
                # XPU info
                xpu_value = next(
                    (row for index, row in xpu_data.iterrows()
                     if row["name"] == name),
                    None
                )
                (xpu_eager_latency, xpu_inductor_latency,
                 xpu_inductor_vs_eager) = calculate_latencies(xpu_value)

                # Reference info
                refer_value = next(
                    (row for index, row in refer_data.iterrows()
                     if row["name"] == name),
                    None
                )
                (refer_eager_latency, refer_inductor_latency,
                 refer_inductor_vs_eager) = calculate_latencies(refer_value)

                # XPU vs reference comparisons
                if (xpu_value is not None and refer_value is not None
                    and xpu_eager_latency > 0):
                    xpu_vs_refer_eager = (refer_eager_latency / xpu_eager_latency)
                else:
                    xpu_vs_refer_eager = 0

                if (xpu_value is not None and refer_value is not None
                    and xpu_inductor_latency > 0):
                    xpu_vs_refer_inductor = (float(refer_value["abs_latency"]) /
                                            xpu_value["abs_latency"])
                else:
                    xpu_vs_refer_inductor = 0

                # Output data
                output_data.append([
                    multiple_replace(xpu_file), name,
                    xpu_eager_latency, xpu_inductor_latency,
                    xpu_inductor_vs_eager,
                    refer_eager_latency, refer_inductor_latency,
                    refer_inductor_vs_eager,
                    xpu_vs_refer_eager, xpu_vs_refer_inductor
                ])
        else:
            names = set(xpu_names)
            names = sorted(names)
            for name in names:
                xpu_value = next(
                    (row for index, row in xpu_data.iterrows()
                     if row["name"] == name),
                    None
                )
                if xpu_value is not None:
                    xpu_eager_latency = (xpu_value["speedup"] *
                                        xpu_value["abs_latency"])
                    output_data.append([
                        multiple_replace(xpu_file), name,
                        xpu_eager_latency, xpu_value["abs_latency"],
                        xpu_value["speedup"], -1, -1, -1, -1, -1
                    ])

    if not args.pr:
        refer_files = find_files("*_xpu_performance.csv", args.baseline)
        for refer_file in refer_files:
            refer_data = pd.read_csv(refer_file)
            refer_names = refer_data["name"].tolist()
            xpu_file = re.sub(
                args.baseline,
                args.target + "/",
                refer_file,
                flags=re.IGNORECASE,
                count=1
            )
            if not os.path.isfile(xpu_file):
                names = set(refer_names)
                names = sorted(names)
                for name in names:
                    refer_value = next(
                        (row for index, row in refer_data.iterrows()
                         if row["name"] == name),
                        None
                    )
                    if refer_value is not None:
                        refer_eager_latency = (refer_value["speedup"] *
                                             refer_value["abs_latency"])
                        output_data.append([
                            multiple_replace(refer_file), name,
                            -1, -1, -1,
                            refer_eager_latency, refer_value["abs_latency"],
                            refer_value["speedup"], -1, -1
                        ])

    return output_data, output_header


def generate_summary(output_data):
    """Generate performance summary statistics."""
    geomean_sum = {
        "all": [],
        "huggingface": [],
        "timm_models": [],
        "torchbench": []
    }

    columns = [
        "Target vs. Baseline [Inductor]",
        "Target vs. Baseline [Eager]",
        "Inductor vs. Eager [Target]"
    ]

    for column_name in columns:
        data = [
            row[column_name] for index, row in output_data.iterrows()
            if row[column_name] > 0
        ]
        if data:
            geomean_sum["all"].append(color_result(geometric_mean(data)))
        else:
            geomean_sum["all"].append("🔴")

        for model_name in ["huggingface", "timm_models", "torchbench"]:
            data = [
                row[column_name] for index, row in output_data.iterrows()
                if (row[column_name] > 0 and
                    re.match(model_name, row["Category"]))
            ]
            if os.path.exists(os.path.join(args.target, model_name)):
                if data:
                    geomean_sum[model_name].append(
                        color_result(geometric_mean(data))
                    )
                else:
                    geomean_sum[model_name].append("🔴")

    geomean_sum = {k: v for k, v in geomean_sum.items() if v}
    output_sum = pd.DataFrame(
        geomean_sum,
        index=columns
    ).T
    return output_sum


def main():
    """Main function to run performance comparison."""
    output_data, output_header = process_comparison_data()

    # Create DataFrame and sort
    output_data = pd.DataFrame(output_data, columns=output_header)
    output_data = output_data.sort_values(
        ['Target vs. Baseline [Inductor]', 'Target vs. Baseline [Eager]'],
        ascending=[True, True]
    )

    # Generate summary
    output_sum = generate_summary(output_data)
    output = output_sum.to_html(header=True)
    with open('performance.summary.html', 'w', encoding='utf-8') as f:
        f.write("\n\n#### performance\n\n" + output)

    # Generate details
    output = output_data.to_html(index=False)
    with open('performance.details.html', 'w', encoding='utf-8') as f:
        f.write("\n\n#### performance\n\n" + output)

    # Regression analysis
    CRITERIA_HIGH = 0.8
    CRITERIA_MEDIUM = 0.9
    PERFORMANCE_FILE = 'performance.regression.html'
    PR_FILE = 'performance.regression.pr.html'

    # Regression cases
    cases_regression = output_data.loc[
        ((output_data['Target vs. Baseline [Inductor]'] < CRITERIA_MEDIUM)
         | (output_data['Target vs. Baseline [Eager]'] < CRITERIA_MEDIUM))
        & (output_data['Baseline inductor'] > 0)
    ]
    write_report(cases_regression, PERFORMANCE_FILE, "#### performance", "w")

    # Highlight in PR
    if args.pr:
        filtered_data = output_data.loc[(output_data['Baseline inductor'] > 0)]
        filtered_data = filtered_data[[
            "Category", "Model", "Target vs. Baseline [Eager]",
            "Target vs. Baseline [Inductor]"
        ]]
        cases_h = filtered_data.loc[
            ((filtered_data['Target vs. Baseline [Inductor]'] < CRITERIA_HIGH)
             | (filtered_data['Target vs. Baseline [Eager]'] < CRITERIA_HIGH))
        ]
        cases_m = filtered_data.loc[
            ((filtered_data['Target vs. Baseline [Inductor]'] < CRITERIA_MEDIUM)
             | (filtered_data['Target vs. Baseline [Eager]'] < CRITERIA_MEDIUM))
            & ((filtered_data['Target vs. Baseline [Inductor]'] >= CRITERIA_HIGH)
               & (filtered_data['Target vs. Baseline [Eager]'] >= CRITERIA_HIGH))
        ]
        if not cases_h.empty or not cases_m.empty:
            with open(PR_FILE, 'w', encoding='utf-8') as f:
                f.write("\n### Performance check outliers, please check!\n")
            write_report(
                cases_h, PR_FILE, "- 🔴 [-1, 80%), should be regression", 'a'
            )
            write_report(
                cases_m, PR_FILE, "- 🟡 [80%, 90%), may be fluctuations", 'a'
            )


if __name__ == "__main__":
    main()
