# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Run the torch-xpu-ops distributed unit tests on the largest XeLink GPU group."""

import os
import subprocess
import sys

from skip_list_dist import skip_dict
from ut_launcher import main, merge_skip_dicts

PIPELINING_TEST_DIR = "../../../../test/distributed/pipelining"


def xelink_group():
    """GPU ids of the largest XeLink group in `xpu-smi topology -m`, or None."""
    proc = subprocess.run(["xpu-smi", "topology", "-m"], capture_output=True, text=True)
    print(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        return None
    largest = ""
    for line in proc.stdout.splitlines():
        items = line.split()
        if not items or items[0] != "GPU" or "CPU Affinity" in line:
            continue
        # Link columns start at index 2, one per GPU
        group = ",".join(
            str(j - 2)
            for j, item in enumerate(items)
            if "SYS" not in item and ("XL" in item or "S" in item)
        )
        if len(group) > len(largest):
            largest = group
    return largest


if __name__ == "__main__":
    group = xelink_group()
    if group is None:
        print("xpu-smi topology failed")
        sys.exit(255)
    os.environ["ZE_AFFINITY_MASK"] = group
    print(f"ZE_AFFINITY_MASK={group}")
    # -rA prints PASSED lines, which the UT check needs to collect passing cases.
    os.environ.setdefault("PYTEST_ADDOPTS", "-rA")
    os.environ["PYTHONPATH"] = os.pathsep.join(
        filter(None, [os.environ.get("PYTHONPATH"), PIPELINING_TEST_DIR])
    )
    sys.exit(main(merge_skip_dicts(skip_dict), __doc__, xml_prefix="xpu_distributed"))
