# Copyright 2020-2025 Intel Corporation
# Licensed under the Apache License, Version 2.0

import re
import subprocess

from benchmarks import mentions_benchmark_model
from patterns import (
    _DEPENDENCY,
    _E2E,
    _ISSUE_TYPE,
    _MODULE,
    _OS_KEYWORDS,
    _PLATFORM_KEYWORDS,
    _TEST_MODULE,
)


def label_names(labels):
    """Lowercased label names. Accepts REST dicts or plain strings."""
    return [
        (label.get('name', '') if isinstance(label, dict) else str(label)).lower()
        for label in (labels or [])
    ]


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
    
    for ln in label_names(labels):
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
    # 'implement' is a substring of the bug signals 'not implemented' and
    # 'notimplementederror', so those must win over a feature request.
    has_unimplemented_bug = any(
        k in text for k in _ISSUE_TYPE['bug_keywords'] if 'implement' in k
    )

    if has_feature_keyword and not has_unimplemented_bug:
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
    for ln in label_names(labels):
        if ln == 'e2e':
            return True

    for pattern in _E2E['patterns']:
        if re.search(pattern, text):
            return True

    # Model names come only from the authoritative .ci/benchmarks lists. A bare
    # hf_/timm_ prefix is not accepted: that would be a hardcoded fallback that
    # silently mis-detects e2e when the lists are missing.
    has_model = mentions_benchmark_model(text)
    has_benchmark_context = False

    # Must have explicit benchmark framework mention (as test framework)
    if has_model:
        for kw in _E2E['paths']:
            if kw in text:
                has_benchmark_context = True
                break

    return has_model and has_benchmark_context


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
    
    for ln in label_names(labels):
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


_ENV_SECTION_RE = re.compile(
    r'(?im)^(?:#+\s*versions\b|collecting environment|pytorch version:)'
)


def strip_env_section(text):
    """Drop the collect_env/Versions dump from an issue body.

    That dump lists every installed package (onemkl-sycl-sparse, torchao, ...),
    so keyword matching against it misclassifies almost every issue.
    """
    m = _ENV_SECTION_RE.search(text)
    return text[:m.start()] if m else text


_MODULE_LABEL_BUCKETS = [
    ('module: distributed', 'distributed'),
    ('module: sdpa', 'sdpa'),
    ('module: sparse', 'sparse'),
    ('module: profiler', 'profiler'),
    ('module: inductor', 'inductor'),
    ('module:inductor', 'inductor'),
    ('module: dynamo', 'dynamo'),
    ('module: ao', 'torchAO'),
    ('module: quant', 'torchAO'),
    ('module: torch-ops-gemm', 'torch-ops-gemm'),
    ('module: torch-ops-eltwise', 'torch-ops-eltwise'),
    ('module: torch-ops-reduction', 'torch-ops-reduction'),
    ('module: torch-ops-others', 'torch-ops-others'),
    ('module: op impl', 'torch-ops-others'),
    ('module: core', 'torch-runtime'),
    ('module: others', 'others'),
]


_OP_FAMILY_BUCKETS = [
    ('torch_ops_sdpa', 'sdpa'),
    ('torch_ops_gemm', 'torch-ops-gemm'),
    ('torch_ops_reduction', 'torch-ops-reduction'),
    ('torch_ops_eltwise', 'torch-ops-eltwise'),
    ('torch_ops_others', 'torch-ops-others'),
]


def classify_module(body, title, labels):
    """Return one of the 13 canonical category buckets.

    `module: ut` is deliberately absent from the label map: it is a test-module
    signal carried by the test_module axis, not a category.
    """
    text = f"{title} {strip_env_section(body)}".lower()

    names = label_names(labels)
    # Bucket priority must win over the order GitHub happens to return labels in,
    # so iterate buckets outermost: an issue labelled both `module: inductor` and
    # `module: dynamo` always resolves to inductor.
    for needle, bucket in _MODULE_LABEL_BUCKETS:
        if any(needle in ln for ln in names):
            return bucket

    if 'torch not compiled with cuda enabled' in text:
        return 'others'
    if 'random failure' in text or 'random failures' in text:
        return 'others'

    # Keywords precede op names: op names like `view`, `backward` and `call` are
    # generic enough to appear in almost any issue, so matching them first would
    # shadow the specific buckets.
    for bucket, kw in _MODULE['keywords']:
        if any(k in text for k in kw):
            return bucket

    for key, bucket in _OP_FAMILY_BUCKETS:
        for op in _MODULE[key]:
            if re.search(rf'\b{re.escape(op)}\b', text):
                return bucket

    return 'others'


_DEPENDENCY_LABEL_VALUES = [
    ('dependency component: onednn', 'oneDNN'),
    ('dependency component: mkl-dnn', 'oneDNN'),
    ('dependency component: dnnl', 'oneDNN'),
    ('dependency component: onemkl', 'oneMKL'),
    ('dependency component: oneccl', 'oneCCL'),
    ('dependency component: xccl', 'oneCCL'),
    ('dependency component: ccl', 'oneCCL'),
    ('dependency component: level_zero', 'Level_Zero'),
    ('dependency component: level zero', 'Level_Zero'),
    ('dependency component: igc', 'IGC'),
    ('dependency component: msvc', 'MSVC'),
    ('dependency component: triton', 'Triton'),
    ('dependency component: community', 'community'),
    ('dependency: third_party packages', 'third_party_packages'),
    ('dependency component: transformers', 'third_party_packages'),
    ('dependency component: huggingface', 'third_party_packages'),
    ('dependency component: oneapi', 'oneAPI'),
    ('dependency component: sycl', 'oneAPI'),
    ('dependency component: driver', 'driver'),
    ('dependency component: mkl', 'oneMKL'),
]


def get_dependency_from_body(body, labels=None):
    """Return one canonical dependency value, or '' when nothing is evidenced.

    `AO` is deliberately not a value: torchao is a PyTorch-ecosystem component
    owned by the module axis (`module: ao`), not an external dependency. A
    transformers/huggingface failure maps to `third_party_packages`.
    """
    if labels is None:
        labels = []

    names = label_names(labels)
    # Value priority must beat GitHub's label order, and the longer needles must
    # be tried first: 'dependency component: mkl' is a substring of the oneMKL
    # and mkl-dnn labels, so it is checked last.
    for needle, value in _DEPENDENCY_LABEL_VALUES:
        if any(needle in ln for ln in names):
            return value

    if not body:
        return ''

    text = strip_env_section(body).lower()

    for header in _DEPENDENCY['version_headers']:
        match = re.search(header, text, re.IGNORECASE)
        if match:
            text = text[:match.start()]
            break

    for value, kw in _DEPENDENCY['keywords']:
        if any(k in text for k in kw):
            return value

    return ''


def extract_os(body):
    """Classify OS as 'Windows'/'Linux'/'' from the whole body.

    Prefer an explicit collect_env 'OS:' line if present.
    """
    if not body:
        return ""

    def classify(text):
        t = text.lower()
        for name, keywords in _OS_KEYWORDS:
            if any(k in t for k in keywords):
                return name
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

    # 2. Title, then 3. body - as separate passes. Merging them would let a body
    # match for an earlier table entry beat a title match for a later one.
    for source in (title, body):
        if not source:
            continue
        lowered = source.lower()
        for code, keywords in _PLATFORM_KEYWORDS:
            for kw in keywords:
                if kw.startswith(r'\b') or kw.endswith(r'\b'):
                    if re.search(kw, source, re.IGNORECASE):
                        return code
                elif kw in lowered:
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
