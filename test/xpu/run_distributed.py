# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

import os
import re
import shlex
import subprocess
import sys

from skip_list_dist import skip_dict
from xpu_test_utils import launch_test

res = 0
res2 = 0
fail_test = []

# xdist options that take a value, as in detect-devices.sh's PYTEST_ADDOPTS
XDIST_VALUE_OPTS = ("-n", "--numprocesses", "--maxprocesses", "--tx", "--dist")


def strip_xdist(addopts):
    """addopts tokens without the xdist worker options."""
    tokens, rest = shlex.split(addopts or ""), []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in XDIST_VALUE_OPTS:
            i += 2
            continue
        if tok == "-d" or tok.startswith(tuple(f"{o}=" for o in XDIST_VALUE_OPTS)):
            i += 1
            continue
        if re.fullmatch(r"-n\S+", tok):
            i += 1
            continue
        rest.append(tok)
        i += 1
    return rest


def distributed_group(mask):
    """First complete 4-GPU group (0-3, 4-7, ...) in mask."""
    gpus = list(
        dict.fromkeys(
            g.strip() for g in (mask or "0,1,2,3,4,5,6,7").split(",") if g.strip()
        )
    )
    bad = [g for g in gpus if not g.isdigit()]
    if bad:
        raise RuntimeError(f"distributed needs integer GPU ids, got {','.join(bad)}")
    ids = {int(g) for g in gpus}
    groups = sorted(
        {i // 4 for i in ids if all(i // 4 * 4 + k in ids for k in range(4))}
    )
    if not groups:
        raise RuntimeError(
            f"distributed needs a complete 4-GPU group (0-3, 4-7, ...), got {','.join(gpus)}"
        )
    base = groups[0] * 4
    return [str(base + k) for k in range(4)]


# -rA prints PASSED lines when PYTEST_ADDOPTS is not set; always run in one xdist worker (-n 1).
addopts = os.environ.get("PYTEST_ADDOPTS") or "-rA"
os.environ["PYTEST_ADDOPTS"] = shlex.join([*strip_xdist(addopts), "-n", "1"])

pipelining = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../test/distributed/pipelining")
)
os.environ["PYTHONPATH"] = os.pathsep.join(
    p for p in (os.environ.get("PYTHONPATH"), pipelining) if p
)

try:
    gpus = distributed_group(os.environ.get("ZE_AFFINITY_MASK"))
except RuntimeError as e:
    print(f"[ERROR] {e}", file=sys.stderr)
    sys.exit(2)

os.environ["ZE_AFFINITY_MASK"] = ",".join(gpus)
print(f"ZE_AFFINITY_MASK={os.environ['ZE_AFFINITY_MASK']}")


# run python test
def run(test_command):
    result = subprocess.run(test_command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    if "FAILED" in result.stdout or "FAILED" in result.stderr:
        fail_test.append(" ".join(test_command))
    return result.returncode


# run pytest with skiplist
for key in skip_dict:
    skip_list = skip_dict[key]
    fail = launch_test(key, skip_list, xml_prefix="xpu_distributed")
    res2 += fail
    if fail:
        fail_test.append(key)

if fail_test:
    print(",".join(fail_test) + " have failures")

sys.exit(res)
