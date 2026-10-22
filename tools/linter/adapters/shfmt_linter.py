# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "shfmt-py==3.12.0.2",
# ]
# ///

# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Shell formatter (shfmt -i 4 -ci) for lintrunner."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
from enum import Enum
from typing import NamedTuple

LINTER_CODE = "SHFMT"
SHFMT_ARGS = ["-i", "4", "-ci"]


class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


class LintMessage(NamedTuple):
    path: str | None
    line: int | None
    char: int | None
    code: str
    severity: LintSeverity
    name: str
    original: str | None
    replacement: str | None
    description: str | None


def shfmt_binary() -> str:
    local = os.path.join(sys.prefix, "bin", "shfmt")
    return local if os.path.exists(local) else (shutil.which("shfmt") or "shfmt")


def check_file(binary: str, path: str) -> LintMessage | None:
    with open(path, encoding="utf-8") as f:
        original = f.read()
    proc = subprocess.run(
        [binary, *SHFMT_ARGS, "--filename", path],
        input=original,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return LintMessage(
            path=path,
            line=None,
            char=None,
            code=LINTER_CODE,
            severity=LintSeverity.ERROR,
            name="shfmt failure",
            original=None,
            replacement=None,
            description=proc.stderr.strip(),
        )
    if proc.stdout == original:
        return None
    return LintMessage(
        path=path,
        line=None,
        char=None,
        code=LINTER_CODE,
        severity=LintSeverity.WARNING,
        name="format",
        original=original,
        replacement=proc.stdout,
        description="Run `lintrunner -a` to apply this patch.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="shell formatter linter", fromfile_prefix_chars="@"
    )
    parser.add_argument("filenames", nargs="+", help="paths to lint")
    args = parser.parse_args()

    binary = shfmt_binary()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for msg in executor.map(lambda p: check_file(binary, p), args.filenames):
            if msg is not None:
                print(json.dumps(msg._asdict()), flush=True)
