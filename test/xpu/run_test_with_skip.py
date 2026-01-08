# Copyright 2020-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

import argparse
import os
import sys

from skip_list_common import skip_dict
from xpu_test_utils import launch_test

parser = argparse.ArgumentParser(description="Run specific unit tests")
# By default, run the cases without the skipped cases
parser.add_argument(
    "--test-cases",
    choices=["selected", "skipped", "all"],
    default="selected",
    help="Test cases scope",
)
# Add skip-cases parameter to import window skip dictionary
parser.add_argument(
    "--skip-cases",
    action="store_true",
    default=False,
    help="Use window skip dictionary for test cases",
)
args = parser.parse_args()


def should_skip_entire_file(skip_list):
    """Check if the skip list contains any entire file skip pattern (*.py::)"""
    if not skip_list:
        return False
    return any(item.endswith(".py::") for item in skip_list)


# Import window skip dictionary if skip-cases is True
if args.skip_cases:
    try:
        # Import the window skip dictionary module
        from window_skip_dict import skip_dict as window_skip_dict

        # Merge the window skip dictionary with the default one using intelligent strategy
        merged_skip_dict = {}

        # First, copy all keys from default skip_dict
        for key in skip_dict:
            merged_skip_dict[key] = skip_dict[key].copy() if skip_dict[key] else []

        # Then merge with window_skip_dict using intelligent strategy
        for key in window_skip_dict:
            window_skip_list = window_skip_dict[key]

            if key in merged_skip_dict:
                default_skip_list = merged_skip_dict[key]

                # Intelligent merge strategy:
                if should_skip_entire_file(window_skip_list):
                    # If Windows wants to skip entire file, use ONLY Windows skip list
                    merged_skip_dict[key] = window_skip_list
                    print(
                        f"Windows entire file skip detected for {key}, using: {window_skip_list}"
                    )
                else:
                    # Otherwise, merge both lists and remove duplicates
                    combined_list = default_skip_list + [
                        item
                        for item in window_skip_list
                        if item not in default_skip_list
                    ]
                    merged_skip_dict[key] = combined_list
                    print(f"Windows merging skip lists for {key}: {combined_list}")
            else:
                # Add new key-value pair from window_skip_dict
                merged_skip_dict[key] = window_skip_list
                print(f"Windows adding new skip key: {key} with {window_skip_list}")

        print("Using intelligently merged skip dictionary")

    except ImportError:
        print(
            "Warning: window_skip_dict module not found, using default skip dictionary"
        )
        merged_skip_dict = skip_dict
    except Exception as e:
        print(f"Error importing window skip dictionary: {e}")
        merged_skip_dict = skip_dict
else:
    merged_skip_dict = skip_dict
    print("Using default skip dictionary")

res = 0
fail_test = []

for key in merged_skip_dict:
    skip_list = merged_skip_dict[key]
    exe_list = None

    if args.test_cases == "skipped":
        # When running only skipped cases, use skip_list as exe_list
        exe_list = skip_list
        skip_list = None
        if not exe_list:  # Check if exe_list is empty
            print(f"Skipping {key} as no tests to execute")
            continue
    elif args.test_cases == "all":
        # When running all cases, don't skip any
        skip_list = None
    # For "selected" case, use the skip_list as is

    print(f"Running test case: {key}")
    if skip_list:
        print(f"Skip list: {skip_list}")
    if exe_list:
        print(f"Execute list: {exe_list}")

    fail = launch_test(key, skip_list=skip_list, exe_list=exe_list)
    res += fail
    if fail:
        fail_test.append(key)

if fail_test:
    print(",".join(fail_test) + " have failures")
else:
    print("All tests passed!")

if os.name == "nt":
    sys.exit(res)
else:
    exit_code = os.WEXITSTATUS(res)
    sys.exit(exit_code)
