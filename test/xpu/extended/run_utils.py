# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

import os
import subprocess
from collections.abc import Iterable
from pathlib import Path


def launch_test(test_file: str, skip_list: Iterable[str]) -> int:
    """Runs test_file with pytest from torch-xpu-ops/test/, filtering skip_list.

    Returns:
        int: The pytest exit code, 0 when all selected tests passed.
    """
    os.environ["PYTORCH_TEST_WITH_SLOW"] = "1"
    module_name: str = Path(test_file).stem
    ops_test_dir: Path = Path(__file__).resolve().parent.parent.parent
    norm_file: str = test_file.replace("\\", "/")
    run_target: str = (
        norm_file
        if norm_file.startswith("xpu/extended/")
        else f"xpu/extended/{norm_file}"
    )
    xml_path: Path = Path(f"./op_extended.{module_name}.xml").resolve()
    cmd_args = ["pytest", f"--junit-xml={xml_path}", run_target]
    if skip_list:
        cmd_args += ["-k", "not " + " and not ".join(skip_list)]
    # Use subprocess to avoid cmd.exe 8191-char command line limit on Windows
    return subprocess.run(cmd_args, cwd=ops_test_dir).returncode
