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


# Ordered: Windows is tested before Linux, since a Windows CI log often also
# names a linux path. Values are lowercase substrings.
_OS_KEYWORDS = [(name, kws) for name, kws in PATTERNS['os_keywords'].items()]


_BENCHMARK_LIST_FILES = PATTERNS['benchmark_list_files']


PYTORCHXPU_FIELD_MAP = PATTERNS['pytorchxpu_field_map']


# The live PyTorchXPU Priority field uses P0-P3; named forms are also accepted
# so a migration to GitHub's Urgent/High/Medium/Low template needs no code change.
PRIORITY_MAP = PATTERNS['priority_map']


_PLATFORM_KEYWORDS = [(code, kws) for code, kws in PATTERNS['platform_keywords']]


_ISSUE_TYPE = PATTERNS['issue_type']


_E2E = PATTERNS['e2e']


_TEST_MODULE = PATTERNS['test_module']


# module.keywords order is semantic. `inductor` precedes `dynamo` because a
# torch.compile failure passes through torch/_dynamo frames even when the defect
# is in the inductor backend, so the backend signal must claim it first.
_MODULE = PATTERNS['module']


MODULE_LABELS = PATTERNS['module_labels']


# dependency.keywords order is semantic too: oneCCL/oneDNN/oneMKL precede oneAPI
# because a collective or library failure usually also names SYCL, and IGC and
# Level_Zero precede driver for the same reason.
_DEPENDENCY = PATTERNS['dependency']


DEPENDENCY_LABELS = PATTERNS['dependency_labels']


_REPRO_SIMPLE_STARTS = tuple(PATTERNS['reproduce_steps']['simple_starts'])


# Whole-token English function words. A real shell command uses operators
# (`&&`, `|`, `;`), never a bare `and`/`the`, so one of these as a standalone
# token marks the line as prose that merely begins with a command name.
_REPRO_PROSE_WORDS = frozenset(PATTERNS['reproduce_steps']['prose_function_words'])


_E2E_CMD_PATTERNS = PATTERNS['e2e_reproducer']['cmd_patterns']


_E2E_DTYPE_PATTERNS = [(p, d) for p, d in PATTERNS['e2e_dtype_patterns']]
