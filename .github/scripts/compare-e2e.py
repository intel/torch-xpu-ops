#!/usr/bin/env python3
"""
Enhanced comparison tool for PyTorch inductor test results (target vs baseline).
Recursively finds all *_performance.csv and *_accuracy.csv files,
validates known suites, data types, and modes,
merges data by suite, data_type, mode, model,
and writes comparison to Excel (three sheets) or CSV (combined file).
All missing cells are filled with empty strings.

If performance files are missing, the combined output excludes performance columns.
If accuracy files are missing, the combined output excludes accuracy columns.
"""

import os
import argparse
import pandas as pd
from glob import glob

# Known valid values
KNOWN_SUITES = {"huggingface", "timm_models", "torchbench"}
KNOWN_DATA_TYPES = {"float32", "bfloat16", "float16", "amp_bf16", "amp_fp16"}
KNOWN_MODES = {"inference", "training"}
PERFORMANCE_THRESHHOLD = 0.1


def find_result_files(root_dir):
    """
    Recursively find all files ending with _performance.csv or _accuracy.csv.
    Returns a list of absolute file paths.
    """
    perf_files = glob(os.path.join(root_dir, "**", "*_performance.csv"), recursive=True)
    acc_files = glob(os.path.join(root_dir, "**", "*_accuracy.csv"), recursive=True)
    return perf_files + acc_files


def parse_filename(filepath):
    """
    Extract suite, data_type, mode, result_type from a filename.
    Expected format: inductor_<suite>_<data_type>_<mode>_xpu_<result_type>.csv
    Example: inductor_timm_models_bfloat16_training_xpu_accuracy.csv
    Returns (suite, data_type, mode, result_type) or raises ValueError.
    Also validates against known keys (warning only).
    """
    basename = os.path.basename(filepath)
    if not basename.endswith(".csv"):
        raise ValueError("Not a CSV file")

    # Remove .csv extension
    base = basename[:-4]

    # Ensure it starts with "inductor_"
    if not base.startswith("inductor_"):
        raise ValueError(f"Filename does not start with 'inductor_': {basename}")
    rest = base[len("inductor_"):]  # everything after "inductor_"

    # Identify suite by checking which known suite the rest starts with
    suite = None
    for s in sorted(KNOWN_SUITES, key=len, reverse=True):  # check longer first
        if rest.startswith(s + "_"):
            suite = s
            rest = rest[len(s) + 1:]  # remove suite and the following underscore
            break
    if suite is None:
        raise ValueError(f"Unknown suite in {basename}")

    # Now rest should be: <data_type>_<mode>_xpu_<result_type>
    parts = rest.split('_')
    # Find the index of the mode (inference/training)
    mode_index = None
    for i, part in enumerate(parts):
        if part in KNOWN_MODES:
            mode_index = i
            break
    if mode_index is None:
        raise ValueError(f"Could not find mode (inference/training) in {basename}")
    mode = parts[mode_index]

    # Data type is everything before mode_index
    data_type = "_".join(parts[:mode_index])
    if data_type not in KNOWN_DATA_TYPES:
        print(f"Warning: Unknown data_type '{data_type}' in {basename}")

    # After mode, we expect "xpu" then result_type
    if mode_index + 1 >= len(parts) or parts[mode_index + 1] != "xpu":
        raise ValueError(f"Missing 'xpu' after mode in {basename}")
    if mode_index + 2 >= len(parts):
        raise ValueError(f"Missing result type in {basename}")
    result_type = parts[mode_index + 2]  # after "xpu"
    if result_type not in ("accuracy", "performance"):
        raise ValueError(f"Result type not recognized in {basename}")

    return suite, data_type, mode, result_type


def load_results(file_list, result_type_filter):
    """
    Load all files of a given result_type (accuracy or performance) and
    return a list of dictionaries with extracted data.
    """
    records = []
    for fpath in file_list:
        try:
            suite, data_type, mode, res_type = parse_filename(fpath)
        except ValueError as e:
            print(f"Skipping {fpath}: {e}")
            continue

        if res_type != result_type_filter:
            continue

        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            print(f"Error reading {fpath}: {e}")
            continue

        if result_type_filter == "accuracy":
            # Expected columns: name, batch_size, accuracy
            for _, row in df.iterrows():
                records.append({
                    "suite": suite,
                    "data_type": data_type,
                    "mode": mode,
                    "model": row["name"],
                    "batch_size": row["batch_size"],
                    "accuracy": row["accuracy"]
                })
        elif result_type_filter == "performance":
            # Expected columns: name, batch_size, speedup, abs_latency
            for _, row in df.iterrows():
                speedup = row.get("speedup")
                abs_latency = row.get("abs_latency")
                if pd.isna(speedup) or pd.isna(abs_latency):
                    print(f"Warning: Missing speedup/abs_latency for {suite}/{data_type}/{mode}/{row.get('name')} in {fpath}")
                    continue
                eager = speedup * abs_latency   # baseline eager time
                inductor = abs_latency
                records.append({
                    "suite": suite,
                    "data_type": data_type,
                    "mode": mode,
                    "model": row["name"],
                    "batch_size": row["batch_size"],
                    "inductor": inductor,
                    "eager": eager
                })
    return records


def merge_accuracy(target_records, baseline_records):
    """Merge accuracy records and add comparison column."""
    target_df = pd.DataFrame(target_records)
    baseline_df = pd.DataFrame(baseline_records)

    if target_df.empty and baseline_df.empty:
        return pd.DataFrame()

    merge_keys = ["suite", "data_type", "mode", "model"]
    merged = pd.merge(target_df, baseline_df, on=merge_keys, how="outer",
                      suffixes=("_target", "_baseline"), indicator=True)

    # Convert batch_size columns to Int64 (nullable integer)
    for col in ["batch_size_target", "batch_size_baseline"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors='coerce').astype('Int64')

    # Add comparison column
    def compare_acc(row):
        if pd.isna(row.get("accuracy_target")) and pd.isna(row.get("accuracy_baseline")):
            return ""
        elif pd.isna(row.get("accuracy_target")):
            return "new_failed"
        elif pd.isna(row.get("accuracy_baseline")):
            return "new_passed"
        elif 'pass' not in row["accuracy_target"] and 'pass' in row["accuracy_baseline"]:
            return "new_failed"
        elif 'fail_accuracy' not in row["accuracy_target"] and 'fail_accuracy' in row["accuracy_baseline"]:
            return "new_failed"
        elif 'pass' in row["accuracy_target"] and 'pass' not in row["accuracy_baseline"]:
            return "new_passed"
        elif 'fail_accuracy' in row["accuracy_target"] and 'fail_accuracy' not in row["accuracy_baseline"]:
            return "new_passed"
        else:
            return "no_changed"

    merged["comparison_acc"] = merged.apply(compare_acc, axis=1)

    # Select and order columns
    cols = ["suite", "data_type", "mode", "model",
            "batch_size_target", "accuracy_target",
            "batch_size_baseline", "accuracy_baseline",
            "comparison_acc"]
    # Ensure all columns exist (fill missing if needed)
    for c in cols:
        if c not in merged.columns:
            merged[c] = None
    return merged[cols].sort_values(by=["suite", "data_type", "mode", "model"])


def merge_performance(target_records, baseline_records):
    """Merge performance records and compute ratios."""
    target_df = pd.DataFrame(target_records)
    baseline_df = pd.DataFrame(baseline_records)

    if target_df.empty and baseline_df.empty:
        return pd.DataFrame()

    merge_keys = ["suite", "data_type", "mode", "model"]
    merged = pd.merge(target_df, baseline_df, on=merge_keys, how="outer",
                      suffixes=("_target", "_baseline"), indicator=True)

    # Convert batch_size columns to Int64
    for col in ["batch_size_target", "batch_size_baseline"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors='coerce').astype('Int64')

    # Compute ratios (target / baseline)
    merged["inductor_ratio"] = merged["inductor_baseline"] / merged["inductor_target"]
    merged["eager_ratio"] = merged["eager_baseline"] / merged["eager_target"]

    # Round performance values to 2 decimals, ratios to 3 decimals
    for col in ["inductor_target", "inductor_baseline", "eager_target", "eager_baseline"]:
        if col in merged.columns:
            merged[col] = merged[col].round(2)
    for col in ["inductor_ratio", "eager_ratio"]:
        if col in merged.columns:
            merged[col] = merged[col].round(3)

    # Add comparison column
    def compare_perf(row):
        if pd.isna(row.get("inductor_target")) and pd.isna(row.get("inductor_baseline")):
            return ""
        elif row["inductor_ratio"] < 0 and row["inductor_baseline"] < 0:
            return ""
        elif pd.isna(row.get("inductor_target")) or row["inductor_ratio"] < 0:
            return "new_failed"
        elif pd.isna(row.get("inductor_baseline")) or row["inductor_baseline"] < 0:
            return "new_passed"
        elif row["inductor_ratio"] < 1 - PERFORMANCE_THRESHHOLD or row["eager_ratio"] < 1 - PERFORMANCE_THRESHHOLD:
            return "new_dropped"
        elif row["inductor_ratio"] > 1 + PERFORMANCE_THRESHHOLD or row["eager_ratio"] > 1 + PERFORMANCE_THRESHHOLD:
            return "new_improved"
        else:
            return "no_changed"

    merged["comparison_perf"] = merged.apply(compare_perf, axis=1)

    # Select and order columns
    cols = ["suite", "data_type", "mode", "model",
            "batch_size_target", "inductor_target", "eager_target",
            "batch_size_baseline", "inductor_baseline", "eager_baseline",
            "inductor_ratio", "eager_ratio", "comparison_perf"]
    for c in cols:
        if c not in merged.columns:
            merged[c] = None
    return merged[cols].sort_values(by=["suite", "data_type", "mode", "model"])


def combine_results(acc_merged, perf_merged):
    """
    Combine accuracy and performance merged dataframes into one wide-format dataframe.
    If one type is missing, return only the available data (with renamed batch size columns).
    """
    # Prepare renamed versions if data exists
    acc_renamed = None
    perf_renamed = None

    if not acc_merged.empty:
        acc_renamed = acc_merged.rename(columns={
            "batch_size_target": "batch_size_accuracy_target",
            "batch_size_baseline": "batch_size_accuracy_baseline"
        })

    if not perf_merged.empty:
        perf_renamed = perf_merged.rename(columns={
            "batch_size_target": "batch_size_performance_target",
            "batch_size_baseline": "batch_size_performance_baseline"
        })

    # Decide what to return
    if acc_renamed is not None and perf_renamed is not None:
        # Both exist: merge on keys
        merge_keys = ["suite", "data_type", "mode", "model"]
        combined = pd.merge(acc_renamed, perf_renamed, on=merge_keys, how="outer")
        def compare_result(row):
            if pd.isna(row.get("comparison_acc")) and pd.isna(row.get("comparison_perf")):
                return ""
            elif pd.isna(row.get("comparison_acc")):
                return row.get("comparison_perf")
            elif pd.isna(row.get("comparison_perf")):
                return row.get("comparison_acc")
            elif ('new_dropped' in [row.get("comparison_acc"), row.get("comparison_perf")] or
                'new_failed' in [row.get("comparison_acc"), row.get("comparison_perf")]):
                return "new_failed"
            elif ('new_improved' in [row.get("comparison_acc"), row.get("comparison_perf")] or
                'new_passed' in [row.get("comparison_acc"), row.get("comparison_perf")]):
                return "new_improved"
            else:
                return "no_changed"
        combined["comparison"] = combined.apply(compare_result, axis=1)
        # combined.drop(columns=['comparison_acc', 'comparison_perf'], inplace=True)
        return combined.sort_values(by=merge_keys)
    elif acc_renamed is not None:
        # Only accuracy exists
        return acc_renamed.sort_values(by=["suite", "data_type", "mode", "model"])
    elif perf_renamed is not None:
        # Only performance exists
        return perf_renamed.sort_values(by=["suite", "data_type", "mode", "model"])
    else:
        # Neither exists
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(
        description="Compare PyTorch inductor test results with baseline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s target_dir baseline_dir comparison.xlsx   # Creates Excel with three sheets
  %(prog)s target_dir baseline_dir comparison.csv    # Creates single combined CSV
        """
    )
    parser.add_argument("target_dir", help="Directory containing target (test) result CSV files (searched recursively)")
    parser.add_argument("baseline_dir", help="Directory containing baseline (reference) result CSV files (searched recursively)")
    parser.add_argument("output", help="Output file name. Use .xlsx for Excel with three sheets, "
                                        "or .csv for a single combined CSV file.")
    args = parser.parse_args()

    # Recursively find all relevant files
    target_files = find_result_files(args.target_dir)
    baseline_files = find_result_files(args.baseline_dir)

    print(f"Found {len(target_files)} CSV files in target directory.")
    print(f"Found {len(baseline_files)} CSV files in baseline directory.")

    # Load accuracy and performance records
    target_acc = load_results(target_files, "accuracy")
    target_perf = load_results(target_files, "performance")
    baseline_acc = load_results(baseline_files, "accuracy")
    baseline_perf = load_results(baseline_files, "performance")

    print(f"Target accuracy records: {len(target_acc)}")
    print(f"Target performance records: {len(target_perf)}")
    print(f"Baseline accuracy records: {len(baseline_acc)}")
    print(f"Baseline performance records: {len(baseline_perf)}")

    # Merge data
    acc_merged = merge_accuracy(target_acc, baseline_acc)
    perf_merged = merge_performance(target_perf, baseline_perf)

    if args.output.endswith(".xlsx"):
        # Write Excel with three sheets (no need to fillna, blanks are automatic)
        with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
            # Combined sheet
            combined = combine_results(acc_merged, perf_merged)
            if not combined.empty:
                combined.to_excel(writer, sheet_name="Details", index=False)
            else:
                pd.DataFrame({"Info": ["No data to combine"]}).to_excel(writer, sheet_name="Details", index=False)

        print(f"Comparison written to {args.output} (Excel with three sheets)")

    elif args.output.endswith(".csv"):
        # Write a single combined CSV file with empty strings for missing values
        combined = combine_results(acc_merged, perf_merged)
        if not combined.empty:
            combined.to_csv(args.output, index=False, na_rep='')
            print(f"Combined comparison written to {args.output}")
        else:
            print("No data to write.")
    else:
        print("Output file name must end with .xlsx or .csv")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
