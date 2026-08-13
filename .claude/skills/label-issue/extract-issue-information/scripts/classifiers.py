# Copyright 2020-2025 Intel Corporation
# Licensed under the Apache License, Version 2.0

import os
import re
import subprocess

from benchmarks import mentions_benchmark_model
from patterns import (
    _DEPENDENCY,
    _E2E,
    _ISSUE_TYPE,
    _MODULE,
    _PLATFORM_KEYWORDS,
    _TEST_MODULE,
)


def generate_summary(body, title):
    # Summary based on issue title
    return title[:150]


def classify_issue_type_canonical(github_type, classifier_type, labels):
    """Map to canonical issue type: Bug, Task, Feature, Epic.

    Priority: github_type (native GitHub issue type from Projects) > labels > classifier heuristic.
    """
    # 1. GitHub native issue type (authoritative when set)
    if github_type:
        gt = github_type.strip().lower()
        if gt in ('bug',):
            return 'Bug'
        if gt in ('task',):
            return 'Task'
        if gt in ('feature', 'feature request'):
            return 'Feature'
        if gt in ('epic',):
            return 'Epic'

    # 2. Labels
    for label in labels:
        ln = (label.get('name', '') if isinstance(label, dict) else str(label)).lower()
        if ln == 'bug':
            return 'Bug'
        if ln in ('task', 'internal task'):
            return 'Task'
        if ln in ('feature', 'feature request', 'enhancement'):
            return 'Feature'
        if ln == 'epic':
            return 'Epic'

    # 3. Infer from classifier_type
    return _ISSUE_TYPE['type_map'].get(classifier_type, 'Bug')


def classify_issue_type(body, title, labels):
    text = f"{title} {body}".lower()
    
    for label in labels:
        ln = label.get('name', '').lower()
        if 'task' == ln or 'internal task' in ln:
            return 'internal task'
    
    has_performance_keyword = any(k in text for k in _ISSUE_TYPE['performance_keywords'])
    has_bug_keyword = any(k in text for k in _ISSUE_TYPE['bug_keywords'])
    # Accuracy issues are a specialized subtype of bug (numerical correctness).
    # They must be checked before the generic bug catch-all below, because
    # bug_keywords contains broad terms ('fail', 'error', 'wrong', 'incorrect')
    # that would otherwise shadow every accuracy issue.
    has_accuracy_keyword = any(k in text for k in _ISSUE_TYPE['accuracy_keywords'])
    has_feature_keyword = any(k in text for k in _ISSUE_TYPE['feature_keywords'])
    
    if has_feature_keyword:
        return 'feature request'
    
    if has_performance_keyword:
        return 'performance issue'

    if has_accuracy_keyword:
        return 'accuracy issue'
    
    if has_bug_keyword:
        return 'functionality bug'
    
    return 'unknown'


def is_e2e_issue(body, title, labels):
    """Check if issue is related to E2E benchmark"""
    text = f"{title} {body}".lower()
    
    # Check labels first - only exact 'e2e' label
    for label in labels:
        ln = label.get('name', '').lower()
        if ln == 'e2e':
            return True
    
    for pattern in _E2E['patterns']:
        if re.search(pattern, text):
            return True
    
    # Check for model names from benchmark model lists with explicit benchmark framework mention
    # Only for specific benchmark prefixes
    has_model = False
    has_benchmark_context = False
    
    for prefix in _E2E['model_prefixes']:
        if prefix in text:
            has_model = True
            break

    if not has_model and mentions_benchmark_model(text):
        has_model = True

    # Must have explicit benchmark framework mention (as test framework)
    if has_model:
        for kw in _E2E['paths']:
            if kw in text:
                has_benchmark_context = True
                break
    
    if has_model and has_benchmark_context:
        return True
    
    return False


def classify_test_module(body, title, labels):
    text = f"{title} {body}".lower()
    
    # Check if it's an E2E issue first
    if is_e2e_issue(body, title, labels):
        return 'e2e'
    
    has_test_pattern = False
    for pattern in _TEST_MODULE['pytest_patterns']:
        if re.search(pattern, text):
            has_test_pattern = True
            break
    
    has_build = any(
        re.search(p, text, re.IGNORECASE) for p in _TEST_MODULE['build_patterns'])
    
    has_infra = any(re.search(p, text) for p in _TEST_MODULE['infra_patterns'])
    
    for label in labels:
        ln = label.get('name', '').lower()
        if 'infrastructure' in ln and ('ci' in ln or 'workflow' in ln or 'action' in ln):
            has_infra = True
            break
    
    if has_build:
        return 'build'
    
    if has_infra:
        return 'infrastructure'
    
    if has_test_pattern:
        if 'benchmarks/dynamo/' in text or 'benchmark' in text:
            return 'e2e'
        return 'ut'
    
    return 'ut'


def classify_module(body, title, labels):
    text = f"{title} {body}".lower()
    
    # Check labels first
    for label in labels:
        ln = label.get('name', '').lower()
        if 'module: distributed' in ln:
            return 'distributed'
        if 'module: inductor' in ln:
            return 'inductor'
        if 'module: ao' in ln:
            return 'AO'
        if 'module: ut' in ln:
            return 'aten_ops'
        if 'module: quant' in ln:
            return 'low_precision'
        if 'module: profiler' in ln:
            return 'profiling'
        if 'module: dynamo' in ln:
            return 'dynamo'
        if 'module: op impl' in ln:
            return 'aten_ops'
    
    # Special case - "Torch not compiled with CUDA enabled" means test configuration issue, not inductor
    if 'torch not compiled with cuda enabled' in text:
        return 'unknown'
    
    # Random failures are not module-specific
    if 'random failure' in text or 'random failures' in text:
        return 'unknown'
    
    # Torch operations (from PyTorch docs)
    for op in _MODULE['torch_ops']:
        if re.search(rf'\b{re.escape(op)}\b', text):
            return 'aten_ops'
    
    for m, kw in _MODULE['keywords']:
        if any(k in text for k in kw):
            return m
    
    return 'unknown'


def get_dependency_from_body(body, labels=None):
    if labels is None:
        labels = []
    
    labels_str = ', '.join([l.get('name', '') for l in labels]).lower()
    
    # Check labels first for 'dependency component:'
    if 'dependency component: onednn' in labels_str or 'dependency component: mkl-dnn' in labels_str or 'dependency component: dnnl' in labels_str:
        return 'oneDNN'
    if 'dependency component: onemkl' in labels_str or 'dependency component: mkl' in labels_str:
        return 'oneMKL'
    if 'dependency component: triton' in labels_str:
        return 'Triton'
    if 'dependency component: torchao' in labels_str:
        return 'AO'
    if 'dependency component: transformers' in labels_str or 'dependency component: huggingface' in labels_str:
        return 'transformers'
    if 'dependency component: oneapi' in labels_str or 'dependency component: sycl' in labels_str:
        return 'oneAPI'
    if 'dependency component: driver' in labels_str:
        return 'driver'
    if 'dependency component: oneccl' in labels_str or 'dependency component: ccl' in labels_str or 'dependency component: xccl' in labels_str:
        return 'oneCCL'
    
    # Filter out version/environment sections
    if not body:
        return ''
    
    text = body.lower()
    
    # Remove version/environment sections
    for header in _DEPENDENCY['version_headers']:
        match = re.search(header, text, re.IGNORECASE)
        if match:
            text = text[:match.start()]
            break
    
    # Check for actual dependency in body (require context like "caused by", "issue", "depend on")
    for d, kw in _DEPENDENCY['keywords']:
        if any(k in text for k in kw):
            return d
    
    return ''


def extract_os(body):
    """Classify OS as 'Windows'/'Linux'/'' from the whole body.

    Prefer an explicit collect_env 'OS:' line if present.
    """
    if not body:
        return ""

    def classify(text):
        t = text.lower()
        if any(k in t for k in ('windows', ' win ', '[win]', 'win32', 'msvc')):
            return "Windows"
        if any(k in t for k in
               ('linux', 'ubuntu', 'wsl', 'debian', 'centos', 'rhel', 'fedora')):
            return "Linux"
        return ""

    os_line = re.search(r'OS:\s*(.+)', body)
    if os_line:
        result = classify(os_line.group(1))
        if result:
            return result
    return classify(body)


def extract_platform(body, title="", labels=None):
    """Return canonical platform code from labels, title, and body (most specific first).

    Priority order:
    1. Labels matching 'hw: <CODE>' (e.g. 'hw: BMG', 'hw: PVC')
    2. Title text (keyword/regex match)
    3. Body text (keyword/regex match)
    """
    # 1. Check labels for 'hw: <CODE>' pattern
    if labels:
        for label in labels:
            ln = label.get('name', '') if isinstance(label, dict) else str(label)
            m = re.match(r'hw:\s*(\w+)', ln, re.IGNORECASE)
            if m:
                candidate = m.group(1).upper()
                # Validate it's a known platform code
                for code, _ in _PLATFORM_KEYWORDS:
                    if candidate == code:
                        return code

    # 2. Check title + body text
    text = f"{title}\n{body}" if title else (body or "")
    if not text:
        return ""
    for code, keywords in _PLATFORM_KEYWORDS:
        for kw in keywords:
            if kw.startswith(r'\b') or kw.endswith(r'\b'):
                if re.search(kw, text, re.IGNORECASE):
                    return code
            elif kw in text.lower():
                return code
    return ""


def detect_local_platform_code():
    """Detect local GPU and return its canonical platform code, or ""."""
    local_str = None
    try:
        result = subprocess.run(
            ["sycl-ls"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "gpu" in line.lower() and "intel" in line.lower():
                    local_str = line.strip()
                    break
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    if not local_str:
        try:
            result = subprocess.run(
                ["xpu-smi", "discovery"], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if "device name" in line.lower():
                        parts = line.split(":", 1)
                        if len(parts) == 2:
                            local_str = parts[1].strip()
                            break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    if not local_str:
        return ""

    for code, keywords in _PLATFORM_KEYWORDS:
        for kw in keywords:
            if kw.startswith(r'\b') or kw.endswith(r'\b'):
                if re.search(kw, local_str, re.IGNORECASE):
                    return code
            elif kw in local_str.lower():
                return code
    return ""


def check_platform_specific(issue_platform_code):
    """Return True if issue platform differs from local GPU, False otherwise.

    Empty issue_platform_code -> False (no constraint).
    Local detection failure -> True (conservative: assume mismatch).
    """
    if not issue_platform_code:
        return False
    local_code = detect_local_platform_code()
    if not local_code:
        return True
    return issue_platform_code != local_code
