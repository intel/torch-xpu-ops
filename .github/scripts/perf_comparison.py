#!/usr/bin/env python3
"""
Performance comparison script.

Compare performance differences between target and baseline results.
Usage:
    python perf_comparison.py --target /path/to/xpu/performance/result/dir
                             --baseline /path/to/reference/dir
"""

import re
import os
import fnmatch
import argparse
import pandas as pd
from statistics import geometric_mean
from typing import List, Tuple, Any, Optional


def multiple_replace(text: str) -> str:
    """
    Apply multiple regex replacements to text.

    Args:
        text: Input text to process

    Returns:
        Processed text with replacements applied
    """
    regex_replacements = [
        (r".*inductor_", ""),
        (r"_xpu_performance.csv", ""),
    ]
    for old_pattern, new_pattern in regex_replacements:
        text = re.sub(old_pattern, new_pattern, text, flags=re.IGNORECASE)
    return text


def find_files(pattern: str, search_path: str) -> List[str]:
    """
    Find files matching pattern in directory tree.

    Args:
        pattern: File pattern to match
        search_path: Directory path to search

    Returns:
        List of matching file paths
    """
    matched_files = []
    for root, _, files in os.walk(search_path):
        for filename in files:
            if fnmatch.fnmatch(filename, pattern):
                matched_files.append(os.path.join(root, filename))
    return matched_files


def color_result(input_val: float) -> str:
    """
    Add color coding to performance results.

    Args:
        input_val: Performance ratio value

    Returns:
        Colored string representation
    """
    if input_val == -1:
        return str(input_val)
    elif input_val < 0.8:
        return f"🔴{input_val:.3f}"
    elif input_val < 0.9:
        return f"🟡{input_val:.3f}"
    elif input_val > 1.1:
        return f"🟢{input_val:.3f}"
    else:
        return f"{input_val:.3f}"


def write_report(cases: pd.DataFrame, filename: str,
                 message: str, write_mode: str) -> None:
    """
    Helper function to write reports to HTML files.

    Args:
        cases: DataFrame to write
        filename: Output filename
        message: Header message
        write_mode: File write mode ('w' or 'a')
    """
    if not cases.empty:
        output = cases.to_html(index=False)
        with open(filename, write_mode, encoding='utf-8') as file:
            file.write(f"\n\n{message}\n\n{output}")


def calculate_latencies(value: Optional[pd.Series]) -> Tuple[float, float, float]:
    """
    Calculate eager and inductor latencies from value row.

    Args:
        value: DataFrame row with performance data

    Returns:
        Tuple of (eager_latency, inductor_latency, inductor_vs_eager)
    """
    if value is None:
        return -1.0, -1.0, -1.0

    eager_latency = value["speedup"] * value["abs_latency"]
    inductor_latency = value["abs_latency"]
    inductor_vs_eager = value["speedup"]

    return eager_latency, inductor_latency, inductor_vs_eager


def find_matching_row(dataframe: pd.DataFrame, model_name: str) -> Optional[pd.Series]:
    """
    Find row for specific model in dataframe.

    Args:
        dataframe: DataFrame to search
        model_name: Model name to find

    Returns:
        Matching row or None if not found
    """
    matches = dataframe[dataframe["name"] == model_name]
    return matches.iloc[0] if not matches.empty else None


def calculate_comparison_ratios(xpu_value: Optional[pd.Series],
                               refer_value: Optional[pd.Series]) -> Tuple[float, float]:
    """
    Calculate performance comparison ratios between target and baseline.

    Args:
        xpu_value: Target performance data
        refer_value: Baseline performance data

    Returns:
        Tuple of (eager_ratio, inductor_ratio)
    """
    if xpu_value is None or refer_value is None:
        return 0.0, 0.0

    # Calculate eager comparison
    xpu_eager = xpu_value["speedup"] * xpu_value["abs_latency"]
    refer_eager = refer_value["speedup"] * refer_value["abs_latency"]
    eager_ratio = refer_eager / xpu_eager if xpu_eager > 0 else 0.0

    # Calculate inductor comparison
    inductor_ratio = (refer_value["abs_latency"] / xpu_value["abs_latency"]
                     if xpu_value["abs_latency"] > 0 else 0.0)

    return eager_ratio, inductor_ratio


def process_comparison_data(args: argparse.Namespace) -> Tuple[List[List[Any]], List[str]]:
    """
    Process and compare performance data between target and baseline.

    Args:
        args: Command line arguments

    Returns:
        Tuple of (output_data, output_header)
    """
    output_header = [
        "Category", "Model", "Target eager", "Target inductor",
        "Inductor vs. Eager [Target]", "Baseline eager", "Baseline inductor",
        "Inductor vs. Eager [Baseline]", "Target vs. Baseline [Eager]",
        "Target vs. Baseline [Inductor]"
    ]
    output_data = []

    # Process target files
    xpu_files = find_files("*_xpu_performance.csv", args.target)

    for xpu_file in xpu_files:
        try:
            xpu_data = pd.read_csv(xpu_file)
            category = multiple_replace(xpu_file)

            # Find corresponding baseline file
            refer_file = xpu_file.replace(args.target, args.baseline)

            if os.path.isfile(refer_file):
                refer_data = pd.read_csv(refer_file)
                process_matching_models(xpu_data, refer_data, category, output_data)
            else:
                process_target_only_models(xpu_data, category, output_data)

        except Exception as e:
            print(f"Error processing {xpu_file}: {e}")
            continue

    # Process baseline-only files if not in PR mode
    if not args.pr:
        process_baseline_only_models(args, output_data)

    return output_data, output_header


def process_matching_models(xpu_data: pd.DataFrame, refer_data: pd.DataFrame,
                           category: str, output_data: List[List[Any]]) -> None:
    """
    Process models that exist in both target and baseline.
    """
    xpu_names = set(xpu_data["name"].tolist())
    refer_names = set(refer_data["name"].tolist())
    all_names = sorted(xpu_names | refer_names)

    for model_name in all_names:
        xpu_value = find_matching_row(xpu_data, model_name)
        refer_value = find_matching_row(refer_data, model_name)

        (xpu_eager, xpu_inductor, xpu_ratio) = calculate_latencies(xpu_value)
        (refer_eager, refer_inductor, refer_ratio) = calculate_latencies(refer_value)
        eager_ratio, inductor_ratio = calculate_comparison_ratios(xpu_value, refer_value)

        output_data.append([
            category, model_name,
            xpu_eager, xpu_inductor, xpu_ratio,
            refer_eager, refer_inductor, refer_ratio,
            eager_ratio, inductor_ratio
        ])


def process_target_only_models(xpu_data: pd.DataFrame, category: str,
                              output_data: List[List[Any]]) -> None:
    """
    Process models that only exist in target data.
    """
    for model_name in sorted(xpu_data["name"].tolist()):
        xpu_value = find_matching_row(xpu_data, model_name)
        if xpu_value is not None:
            xpu_eager = xpu_value["speedup"] * xpu_value["abs_latency"]
            output_data.append([
                category, model_name,
                xpu_eager, xpu_value["abs_latency"], xpu_value["speedup"],
                -1, -1, -1, -1, -1
            ])


def process_baseline_only_models(args: argparse.Namespace,
                                output_data: List[List[Any]]) -> None:
    """
    Process models that only exist in baseline data.
    """
    refer_files = find_files("*_xpu_performance.csv", args.baseline)

    for refer_file in refer_files:
        try:
            # Find corresponding target file
            xpu_file = refer_file.replace(args.baseline, args.target)
            if os.path.isfile(xpu_file):
                continue

            refer_data = pd.read_csv(refer_file)
            category = multiple_replace(refer_file)

            for model_name in sorted(refer_data["name"].tolist()):
                refer_value = find_matching_row(refer_data, model_name)
                if refer_value is not None:
                    refer_eager = refer_value["speedup"] * refer_value["abs_latency"]
                    output_data.append([
                        category, model_name,
                        -1, -1, -1,
                        refer_eager, refer_value["abs_latency"],
                        refer_value["speedup"], -1, -1
                    ])
        except Exception as e:
            print(f"Error processing baseline file {refer_file}: {e}")
            continue


def generate_summary(output_data: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """
    Generate performance summary statistics.

    Args:
        output_data: Processed performance data
        args: Command line arguments

    Returns:
        DataFrame with summary statistics
    """
    geomean_results = {
        "all": [],
        "huggingface": [],
        "timm_models": [],
        "torchbench": []
    }

    comparison_columns = [
        "Target vs. Baseline [Inductor]",
        "Target vs. Baseline [Eager]",
        "Inductor vs. Eager [Target]"
    ]

    for column_name in comparison_columns:
        # Overall geometric mean
        valid_data = [
            row[column_name] for _, row in output_data.iterrows()
            if row[column_name] > 0
        ]
        geomean_results["all"].append(
            color_result(geometric_mean(valid_data)) if valid_data else "🔴"
        )

        # Per-category geometric means
        for category in ["huggingface", "timm_models", "torchbench"]:
            category_data = [
                row[column_name] for _, row in output_data.iterrows()
                if (row[column_name] > 0 and
                    re.match(category, row["Category"]))
            ]
            geomean_results[category].append(
                color_result(geometric_mean(category_data)) if category_data else "🔴"
            )

    # Filter out empty categories
    geomean_results = {k: v for k, v in geomean_results.items() if any(v)}

    return pd.DataFrame(geomean_results, index=comparison_columns).T


def generate_regression_reports(output_data: pd.DataFrame, args: argparse.Namespace) -> None:
    """
    Generate regression analysis reports.

    Args:
        output_data: Processed performance data
        args: Command line arguments
    """
    criteria_high = 0.8
    criteria_medium = 0.9

    # Regression cases for full report
    regression_cases = output_data.loc[
        ((output_data['Target vs. Baseline [Inductor]'] < criteria_medium) |
         (output_data['Target vs. Baseline [Eager]'] < criteria_medium)) &
        (output_data['Baseline inductor'] > 0)
    ]

    write_report(regression_cases, 'performance.regression.html',
                 "#### Performance Regression", "w")

    # PR-specific reports
    if args.pr:
        generate_pr_report(output_data, criteria_high, criteria_medium)


def generate_pr_report(output_data: pd.DataFrame, criteria_high: float,
                      criteria_medium: float) -> None:
    """
    Generate PR-specific performance report.

    Args:
        output_data: Processed performance data
        criteria_high: High regression threshold
        criteria_medium: Medium regression threshold
    """
    pr_data = output_data.loc[output_data['Baseline inductor'] > 0]
    pr_data = pr_data[[
        "Category", "Model", "Target vs. Baseline [Eager]",
        "Target vs. Baseline [Inductor]"
    ]]

    # High regression cases
    high_regression = pr_data.loc[
        (pr_data['Target vs. Baseline [Inductor]'] < criteria_high) |
        (pr_data['Target vs. Baseline [Eager]'] < criteria_high)
    ]

    # Medium regression cases
    medium_regression = pr_data.loc[
        ((pr_data['Target vs. Baseline [Inductor]'] < criteria_medium) |
         (pr_data['Target vs. Baseline [Eager]'] < criteria_medium)) &
        (pr_data['Target vs. Baseline [Inductor]'] >= criteria_high) &
        (pr_data['Target vs. Baseline [Eager]'] >= criteria_high)
    ]

    if not high_regression.empty or not medium_regression.empty:
        with open('performance.regression.pr.html', 'w', encoding='utf-8') as f:
            f.write("\n### Performance outliers, please check!\n")

        write_report(high_regression, 'performance.regression.pr.html',
                    "- 🔴 [-1, 80%), should be regression", 'a')
        write_report(medium_regression, 'performance.regression.pr.html',
                    "- 🟡 [80%, 90%), may be fluctuations", 'a')


def main():
    """Main function to run performance comparison."""
    args = argparse.ArgumentParser(
        description="Performance Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    args.add_argument(
        "--target",
        required=True,
        help="XPU performance result csv files directory"
    )
    args.add_argument(
        "--baseline",
        required=True,
        help="XPU reference result csv files directory"
    )
    args.add_argument(
        "--pr",
        action="store_true",
        help="Only show results that XPU has"
    )
    args = args.parse_args()

    # Process comparison data
    output_data, output_header = process_comparison_data(args)
    output_df = pd.DataFrame(output_data, columns=output_header)
    print(output_df)

    # Sort by performance ratios
    output_df = output_df.sort_values(
        ['Target vs. Baseline [Inductor]', 'Target vs. Baseline [Eager]'],
        ascending=[True, True]
    )


    # Generate summary report
    summary_df = generate_summary(output_df, args)
    with open('performance.summary.html', 'w', encoding='utf-8') as f:
        f.write("\n\n#### Performance Summary\n\n" + summary_df.to_html(header=True))

    # Generate detailed report
    with open('performance.details.html', 'w', encoding='utf-8') as f:
        f.write("\n\n#### Performance Details\n\n" + output_df.to_html(index=False))

    # Generate regression reports
    generate_regression_reports(output_df, args)

    print("Performance comparison completed!")
    print(f"Processed {len(output_df)} model comparisons")


if __name__ == "__main__":
    main()
