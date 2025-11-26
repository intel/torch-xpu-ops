#!/usr/bin/env python3
"""
Accuracy Check Script
Compares test results against reference data and calculates pass rates.
Reference last updated: https://github.com/intel/torch-xpu-ops/pull/1223
"""

import re
import json
import argparse
import pandas as pd
from pathlib import Path


def load_data(csv_file):
    """Load CSV file with comment support."""
    return pd.read_csv(csv_file, comment='#')


def find_model_row(dataframe, model_name):
    """Find row for a specific model in dataframe."""
    matches = dataframe[dataframe['name'] == model_name]
    return matches.iloc[0] if not matches.empty else None


def get_test_result(data, suite, dtype, mode, model):
    """
    Get test result for specific test configuration.

    Args:
        data: JSON data containing test results
        suite: Test suite name
        dtype: Data type
        mode: Inference or training mode
        model: Model name

    Returns:
        Test result or "N/A" if not found
    """
    for issue in data:
        for row in issue.get('table_rows', []):
            if len(row) >= 6 and row[:4] == [suite, dtype, mode, model]:
                return row[4]
    return "N/A"


def parse_file_name(filename):
    """
    Parse benchmark file name to extract suite, dtype, and mode.

    Args:
        filename: Input filename to parse

    Returns:
        tuple: (suite, dtype, mode) or ("N/A", "N/A", "N/A") if pattern not found
    """
    pattern = (
        r"_(huggingface|timm_models|torchbench)_"
        r"(float32|bfloat16|float16|amp_bf16|amp_fp16)_"
        r"(inference|training)_"
    )
    match = re.search(pattern, filename)
    return match.groups() if match else ("N/A", "N/A", "N/A")


def load_known_data(issue_file):
    """Load known test data from JSON file."""
    try:
        with open(issue_file, encoding='utf-8') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading known data from {issue_file}: {e}")
        return []


def update_reference_dataframe(refer_data, model_name, dtype, accuracy):
    """
    Update reference dataframe with new or updated model results.

    Args:
        refer_data: Reference dataframe to update
        model_name: Name of the model to update
        dtype: Data type column to update
        accuracy: Accuracy value to set

    Returns:
        Updated dataframe
    """
    mask = refer_data['name'] == model_name
    if mask.any():
        refer_data.loc[mask, dtype] = accuracy
    else:
        new_row = {'name': model_name, dtype: accuracy}
        refer_data = pd.concat([refer_data, pd.DataFrame([new_row])], ignore_index=True)
    return refer_data


def categorize_model(test_accuracy, refer_accuracy, known_accuracy):
    """
    Categorize a model based on its test results.

    Returns:
        tuple: (category, should_update_reference)
    """
    if test_accuracy == "N/A":
        return "lost", False
    elif 'pass' in test_accuracy:
        if refer_accuracy == "N/A":
            return "new", True
        elif 'pass' not in refer_accuracy:
            return "new_pass", True
        return "passed", False
    elif 'timeout' in test_accuracy:
        if refer_accuracy == "N/A":
            return "new", True
        return "timeout", False
    else:  # Failed cases
        if refer_accuracy == "N/A":
            return "expected_failed", True
        elif "pass" in refer_accuracy and known_accuracy != test_accuracy:
            return "real_failed", False
        else:
            if test_accuracy != refer_accuracy:
                return "expected_failed", True
            return "expected_failed", False


def print_results_summary(suite, dtype, mode, categories):
    """Print formatted summary of results."""
    print(f"============ Summary for {suite} {dtype} {mode} accuracy ============")
    print(f"Total models: {len(categories['all_models'])}")
    print(f"Passed models: {len(categories['passed'])}")
    print(f"Real failed models: {len(categories['real_failed'])} , {categories['real_failed']}")
    print(f"Expected failed models: {len(categories['expected_failed'])} , {categories['expected_failed']}")
    print(f"Warning timeout models: {len(categories['timeout'])} , {categories['timeout']}")
    print(f"New models: {len(categories['new'])} , {categories['new']}")
    print(f"Failed to passed models: {len(categories['new_pass'])} , {categories['new_pass']}")
    print(f"Not run/in models: {len(categories['lost'])} , {categories['lost']}")

    total_models = len(categories['all_models'])
    if total_models > 0:
        pass_rate = len(categories['passed']) / total_models * 100
        print(f"Pass rate: {pass_rate:.2f}%")


def main():
    """Main function to run accuracy comparison."""
    parser = argparse.ArgumentParser(
        description="Accuracy Check",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--driver", type=str, default="rolling", help="rolling or lts")
    parser.add_argument("--category", type=str, default="inductor", help="inductor")
    parser.add_argument("--suite", type=str, required=True, help="huggingface, timm_models or torchbench")
    parser.add_argument("--mode", type=str, required=True, help="inference or training")
    parser.add_argument("--dtype", type=str, required=True, help="float32, bfloat16, float16, amp_bf16 or amp_fp16")
    parser.add_argument("--csv_file", type=str, required=True, help="Test results CSV file path")
    parser.add_argument("--issue_file", type=str, required=True, help="Known test data JSON file path")
    parser.add_argument('--update', action='store_true', help="Whether to update new pass and new failed info")

    args = parser.parse_args()

    # Load data files
    test_data = load_data(args.csv_file)
    test_known_data = load_known_data(args.issue_file)
    suite, dtype, mode = parse_file_name(args.csv_file)

    # Load reference data
    current_path = Path(__file__).parent.resolve()
    refer_filename = f"{args.category}_{args.suite}_{args.mode}.csv"
    refer_file = current_path / args.driver / refer_filename
    refer_data = load_data(refer_file)

    # Get model names
    test_names = test_data['name'].tolist()
    refer_names = refer_data['name'].tolist()
    model_names = set(refer_names + test_names)

    # Initialize result categories
    categories = {
        'all_models': list(model_names),
        'passed': [],
        'real_failed': [],
        'expected_failed': [],
        'new': [],
        'new_pass': [],
        'lost': [],
        'timeout': []
    }

    needs_update = False

    # Process each model
    for model_name in model_names:
        test_row = find_model_row(test_data, model_name)
        refer_row = find_model_row(refer_data, model_name)

        test_accuracy = str(test_row['accuracy']) if test_row is not None else "N/A"
        refer_accuracy = str(refer_row[args.dtype]) if refer_row is not None else "N/A"
        known_accuracy = get_test_result(test_known_data, suite, dtype, mode, model_name)

        # Debug print (optional)
        # print(f"{model_name}: test={test_accuracy}, ref={refer_accuracy}, known={known_accuracy}")

        # Categorize model and determine if reference needs update
        category, should_update = categorize_model(
            test_accuracy, refer_accuracy, known_accuracy
        )

        categories[category].append([model_name, test_accuracy])

        # Update reference data if needed
        if should_update and args.update:
            refer_data = update_reference_dataframe(
                refer_data, model_name, args.dtype, test_accuracy
            )
            needs_update = True

    # Print summary
    print_results_summary(args.suite, args.dtype, args.mode, categories)

    # Update reference CSV if requested
    if needs_update:
        refer_data.to_csv(refer_file, sep=',', encoding='utf-8', index=False)
        print(f"Reference file updated: {refer_file}")


if __name__ == "__main__":
    main()
