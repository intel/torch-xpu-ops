# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""torch-xpu-ops operator microbenchmark framework (see README.md).

Only measure.py and the case ``setup``/``fwd`` bodies import torch. Everything
else runs on a plain Python with numpy and pyyaml, so case listing and
selection work without a wheel.
"""

import json
import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = PACKAGE_DIR / "config"
REPO_ROOT = PACKAGE_DIR.parent.parent
CANARY_ID = "__canary__"

# cost estimate of a case without history (seconds per worker process)
DEFAULT_CASE_SECONDS = 4.0
WORKER_START_SECONDS = 10.0


def load_json(path, default=None):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def load_yaml(path, default=None):
    if not path or not os.path.exists(path):
        return default
    import yaml

    with open(path) as f:
        return yaml.safe_load(f) or default


def read_jsonl(path):
    """Records of a JSONL file; a line cut off by a crashed writer is skipped."""
    out = []
    if os.path.exists(path):
        with open(path) as f:
            for line in filter(None, map(str.strip, f)):
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def load_history(path=None):
    """config/history.json: noise_stats (A/A calibration), case_cost, canary."""
    data = load_json(path or CONFIG_DIR / "history.json", {}) or {}
    return {k: data.get(k) or {} for k in ("noise_stats", "case_cost", "canary")}
