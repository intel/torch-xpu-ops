# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

import argparse
import os
import sys
from enum import StrEnum
from pathlib import Path

from skip_list_common import skip_dict
from skip_list_win import skip_dict as skip_dict_win
from xpu_test_utils import launch_test

IS_WINDOWS: bool = sys.platform == "win32"


class TestScope(StrEnum):
    SELECTED = "selected"
    """All test cases, excluding the skipped ones."""
    SKIPPED = "skipped"
    """Only the skipped test cases."""
    ALL = "all"
    """All test cases, including the skipped ones."""


parser = argparse.ArgumentParser(description="Run specific unit tests")
parser.add_argument(
    "--test-cases",
    type=TestScope,
    choices=list(TestScope),
    default=TestScope.SELECTED,
    help="Test cases scope",
)
parser.add_argument("tests", nargs="*", help="Specific test cases to execute")
args = parser.parse_args()


failed_test_files: list[str] = []
launch_files: list[str] = args.tests or list(skip_dict)
print(f"Running tests on the platform: {os.name}")
for file in launch_files:
    file = Path(file).as_posix()
    skip_list: list[str] = list(skip_dict.get(file) or ())
    if IS_WINDOWS:
        skip_list += skip_dict_win.get(file) or ()

    failed: int = 0
    if args.test_cases == TestScope.ALL:
        failed = launch_test(file)
    elif args.test_cases == TestScope.SKIPPED:
        if not skip_list:
            continue
        failed = launch_test(file, exe_list=skip_list)
    elif args.test_cases == TestScope.SELECTED:
        failed = launch_test(file, skip_list=skip_list)
    else:
        raise AssertionError("Unknown TestScope value!")

    if failed:
        failed_test_files.append(file)

if failed_test_files:
    print(",".join(failed_test_files) + " have failures")
    sys.exit(1)
else:
    print("All tests passed!")
    sys.exit(0)
