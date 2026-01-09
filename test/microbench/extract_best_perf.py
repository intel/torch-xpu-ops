# Copyright 2020-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

import argparse
import sys

import pandas as pd


def process_csv(input_file, output_file):
    """
    For rows identical except for 'time(us)' and 'E2E total time(us)',
    keep the row with the minimum 'E2E total time(us)' in each group.
    """
    # Read CSV with semicolon separator
    df = pd.read_csv(input_file, sep=";")

    time_col = "time(us)"
    e2e_col = "E2E total time(us)"

    # Validate required columns
    for col in [time_col, e2e_col]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: '{col}'")

    # Grouping columns: all except the two time columns
    group_cols = [col for col in df.columns if col not in [time_col, e2e_col]]

    # Group and select the row with min E2E total time in each group
    # Use idxmin to get index of min E2E row per group, then loc to fetch full rows
    grouped = df.groupby(group_cols, dropna=False)
    min_e2e_indices = grouped[e2e_col].idxmin()
    result_df = df.loc[min_e2e_indices].reset_index(drop=True)

    # Preserve original column order
    result_df = result_df[df.columns]

    # Save with semicolon separator
    result_df.to_csv(output_file, sep=";", index=False)

    # Statistics
    print(f"Original rows: {len(df)}")
    print(f"Output rows: {len(result_df)}")
    print(f"Groups merged: {len(grouped)}")
    print(f"Saved to: {output_file}")

    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Keep the row with min 'E2E total time(us)' for each group of identical non-time rows."
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Input CSV file (semicolon-separated)"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Output CSV file (semicolon-separated)"
    )

    args = parser.parse_args()

    try:
        result = process_csv(args.input, args.output)
        print("\nSample output:")
        print(result.head())
    except FileNotFoundError:
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
