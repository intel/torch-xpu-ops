# /// script
# requires-python = ">=3.11"
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

"""Syntax checker for structured data and script files (JSON, YAML, TOML, JS)."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tomllib
from enum import Enum
from typing import NamedTuple

from ruamel.yaml import YAML

LINTER_CODE = "SYNTAX"


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


def error(path: str, name: str, desc: str, line: int | None = None) -> LintMessage:
    return LintMessage(
        path=path,
        line=line,
        char=None,
        code=LINTER_CODE,
        severity=LintSeverity.ERROR,
        name=name,
        original=None,
        replacement=None,
        description=desc,
    )


def check_file(path: str) -> LintMessage | None:
    try:
        if path.endswith(".json"):
            with open(path, encoding="utf-8") as f:
                json.load(f)
        elif path.endswith(".toml"):
            with open(path, "rb") as f:
                tomllib.load(f)
        elif path.endswith((".yml", ".yaml")):
            with open(path, encoding="utf-8") as f:
                # Round-trip loader rejects duplicate keys and tolerates custom tags.
                list(YAML(typ="rt").load_all(f))
        elif path.endswith((".js", ".mjs", ".cjs")):
            node = shutil.which("node")
            if node is None:
                return None
            proc = subprocess.run(
                [node, "--check", path], capture_output=True, text=True
            )
            if proc.returncode != 0:
                return error(path, "JS syntax error", proc.stderr.strip())
    except json.JSONDecodeError as e:
        return error(path, "JSON syntax error", e.msg, e.lineno)
    except tomllib.TOMLDecodeError as e:
        return error(path, "TOML syntax error", str(e))
    except Exception as e:  # ruamel raises several unrelated error types
        line = getattr(getattr(e, "problem_mark", None), "line", None)
        return error(
            path,
            "YAML syntax error",
            f"{type(e).__name__}: {e}",
            line + 1 if line is not None else None,
        )
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="structured file syntax linter", fromfile_prefix_chars="@"
    )
    parser.add_argument("filenames", nargs="+", help="paths to lint")
    args = parser.parse_args()

    for fn in args.filenames:
        msg = check_file(fn)
        if msg is not None:
            print(json.dumps(msg._asdict()), flush=True)
