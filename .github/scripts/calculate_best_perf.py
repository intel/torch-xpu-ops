#!/usr/bin/env python3
"""
Performance Benchmark Analyzer

Calculate and update best performance metrics by comparing new results
with historical best performance data.

Usage:
    python calculate_best_perf.py --best /path/to/best.csv
            --new /path/to/new/measured/performance/result/dir
            --device <Used device name>
            --os <Used OS version>
            --driver <Used driver version>
            --oneapi <Used oneapi version>
            --gcc <Used gcc version>
            --python <Used python version>
            --pytorch <Used pytorch version>
            --torch-xpu-ops <Used torch-xpu-ops version>
"""

import re
import os
import fnmatch
import argparse
import pandas as pd
from datetime import date


def multiple_replace(text):
    """
    Apply regex replacements to extract category from filename.

    Args:
        text: Input filename to process

    Returns:
        Cleaned category name
    """
    regex_replacements = [
        (r".*inductor_", ""),
        (r"_xpu_performance.csv", ""),
    ]
    for old_pattern, new_pattern in regex_replacements:
        text = re.sub(old_pattern, new_pattern, text, flags=re.IGNORECASE)
    return text


def find_files(pattern, search_path):
    """
    Recursively find files matching pattern in directory.

    Args:
        pattern: File pattern to match (e.g., "*.csv")
        search_path: Directory path to search in

    Returns:
        List of matching file paths
    """
    matched_files = []
    for root, _, files in os.walk(search_path):
        for filename in files:
            if fnmatch.fnmatch(filename, pattern):
                matched_files.append(os.path.join(root, filename))
    return matched_files


def find_best_row(dataframe, category, model, value_type):
    """
    Find the best performance row for given criteria.

    Args:
        dataframe: DataFrame to search in
        category: Test category
        model: Model name
        value_type: 'eager' or 'inductor'

    Returns:
        Tuple of (row_index, row_data) or None if not found
    """
    matches = dataframe[
        (dataframe["Category"] == category) &
        (dataframe["Model"] == model) &
        (dataframe["Value Type"] == value_type)
    ]
    if not matches.empty:
        return matches.index[0], matches.iloc[0]
    return None


def create_new_row(category, model, value_type, value, args_dict):
    """
    Create a new row for best performance data.

    Args:
        category: Test category
        model: Model name
        value_type: 'eager' or 'inductor'
        value: Performance value
        args_dict: Dictionary of system configuration arguments

    Returns:
        Dictionary representing the new row
    """
    return {
        "Category": category,
        "Model": model,
        "Value Type": value_type,
        "Value": value,
        "Device": args_dict["device"],
        "OS": args_dict["os"],
        "Driver": args_dict["driver"],
        "OneAPI": args_dict["oneapi"],
        "GCC": args_dict["gcc"],
        "Python": args_dict["python"],
        "PyTorch": args_dict["pytorch"],
        "Torch XPU Ops": args_dict["torch_xpu_ops"],
        "Date": date.today().strftime('%F')
    }


def update_best_performance(best_data, category, model, value_type,
                          new_value, args_dict):
    """
    Update best performance data with new value if better.

    Args:
        best_data: DataFrame with best performance data
        category: Test category
        model: Model name
        value_type: 'eager' or 'inductor'
        new_value: New performance value to compare
        args_dict: System configuration arguments

    Returns:
        Updated DataFrame
    """
    best_row = find_best_row(best_data, category, model, value_type)

    # For performance metrics, lower values are better
    current_best = best_row[1]["Value"] if best_row else float('inf')
    is_better = new_value < current_best

    if best_row and is_better:
        # Update existing row
        best_data.loc[best_row[0]] = create_new_row(
            category, model, value_type, new_value, args_dict
        )
    elif not best_row:
        # Add new row
        new_row = create_new_row(category, model, value_type, new_value, args_dict)
        best_data = pd.concat([
            best_data,
            pd.DataFrame([new_row])
        ], ignore_index=True)

    return best_data


def main():
    """Main function to calculate and update best performance metrics."""
    parser = argparse.ArgumentParser(
        description="Get Best Performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--best", required=True,
                       help="Saved best performance file")
    parser.add_argument("--new", required=True,
                       help="New performance results directory")
    parser.add_argument("--device", type=str,
                       help="Device name, such as PVC1100")
    parser.add_argument("--os", type=str,
                       help="OS version, such as Ubuntu 22.04")
    parser.add_argument("--driver", type=str,
                       help="Driver version, such as 25.05.32567")
    parser.add_argument("--oneapi", type=str,
                       help="OneAPI version, such as 2025.1")
    parser.add_argument("--gcc", type=str,
                       help="GCC version, such as 11")
    parser.add_argument("--python", type=str,
                       help="Python version, such as 3.10")
    parser.add_argument("--pytorch", type=str,
                       help="PyTorch version")
    parser.add_argument("--torch-xpu-ops", type=str,
                       help="Torch XPU Ops version")

    args = parser.parse_args()

    # Prepare system configuration dictionary
    system_config = {
        "device": args.device,
        "os": args.os,
        "driver": args.driver,
        "oneapi": args.oneapi,
        "gcc": args.gcc,
        "python": args.python,
        "pytorch": args.pytorch,
        "torch_xpu_ops": getattr(args, 'torch-xpu-ops', None)
    }

    # Define output columns
    best_columns = [
        "Category", "Model", "Value Type", "Value",
        "Device", "OS", "Driver", "OneAPI", "GCC", "Python",
        "PyTorch", "Torch XPU Ops", "Date"
    ]

    # Load or initialize best performance data
    if os.path.isfile(args.best):
        best_data = pd.read_csv(args.best)
    else:
        best_data = pd.DataFrame(columns=best_columns)

    best_data = best_data.reset_index(drop=True)

    # Find and process new performance files
    new_files = find_files("*_xpu_performance.csv", args.new)

    if not new_files:
        print(f"No performance files found in {args.new}")
        return

    print(f"Processing {len(new_files)} performance files...")

    for new_file in new_files:
        category = multiple_replace(new_file)
        # print(f"Processing category: {category}")

        try:
            new_data = pd.read_csv(new_file)
            new_data = new_data.reset_index(drop=True)

            for _, row in new_data.iterrows():
                model_name = row["name"]

                # Process eager performance
                eager_perf = row["abs_latency"] * row["speedup"]
                best_data = update_best_performance(
                    best_data, category, model_name, "eager",
                    eager_perf, system_config
                )

                # Process inductor performance
                inductor_perf = row["abs_latency"]
                best_data = update_best_performance(
                    best_data, category, model_name, "inductor",
                    inductor_perf, system_config
                )

        except Exception as e:
            print(f"Error processing {new_file}: {e}")
            continue

    # Save updated best performance data
    best_data.to_csv(args.best, sep=',', encoding='utf-8', index=False)
    print(f"Best performance data saved to: {args.best}")
    print(f"Total records: {len(best_data)}")


if __name__ == "__main__":
    main()
