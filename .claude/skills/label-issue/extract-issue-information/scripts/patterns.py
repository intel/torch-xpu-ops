# Copyright 2020-2025 Intel Corporation
# Licensed under the Apache License, Version 2.0

import json
import os
import sys


def _load_patterns():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'patterns.json')
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        sys.exit(f"FATAL: cannot load {path}: {exc}")


PATTERNS = _load_patterns()


KNOWN_TEST_TYPES = PATTERNS['known_test_types']


_BENCHMARK_LIST_FILES = PATTERNS['benchmark_list_files']


PYTORCHXPU_FIELD_MAP = PATTERNS['pytorchxpu_field_map']


_PLATFORM_KEYWORDS = [(code, kws) for code, kws in PATTERNS['platform_keywords']]


_ISSUE_TYPE = PATTERNS['issue_type']


_E2E = PATTERNS['e2e']


_TEST_MODULE = PATTERNS['test_module']


_MODULE = PATTERNS['module']


_DEPENDENCY = PATTERNS['dependency']


_REPRO_SIMPLE_STARTS = tuple(PATTERNS['reproduce_steps']['simple_starts'])


_E2E_CMD_PATTERNS = PATTERNS['e2e_reproducer']['cmd_patterns']


_E2E_DTYPE_PATTERNS = [(p, d) for p, d in PATTERNS['e2e_dtype_patterns']]
