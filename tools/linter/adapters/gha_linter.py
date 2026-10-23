# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "ruamel.yaml==0.18.10",
# ]
# ///

# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""
TODO
"""

from __future__ import annotations

import argparse
import json
from enum import Enum
from typing import NamedTuple

import ruamel.yaml  # type: ignore[import]


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="github actions linter",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )

    args = parser.parse_args()

    for fn in args.filenames:
        with open(fn) as f:
            contents = f.read()

        yaml = ruamel.yaml.YAML()  # type: ignore[attr-defined]
        try:
            r = yaml.load(contents)
        except Exception as err:
            msg = LintMessage(
                path=None,
                line=None,
                char=None,
                code="GHA",
                severity=LintSeverity.ERROR,
                name="YAML load failure",
                original=None,
                replacement=None,
                description=f"Failed due to {err.__class__.__name__}:\n{err}",
            )

            print(json.dumps(msg._asdict()), flush=True)
            continue

        for job_name, job in r.get("jobs", {}).items():
            # Least privilege (zizmor secrets-inherit): reusable workflows get only the secrets they declare.
            if job.get("uses") and job.get("secrets") == "inherit":
                msg = LintMessage(
                    path=fn,
                    line=job.lc.line,
                    char=None,
                    code="GHA",
                    severity=LintSeverity.ERROR,
                    name="secrets: inherit",
                    original=None,
                    replacement=None,
                    description=(
                        f"GitHub actions job '{job_name}' uses 'secrets: inherit'; "
                        "pass only the secrets the called workflow declares"
                    ),
                )

                print(json.dumps(msg._asdict()), flush=True)
