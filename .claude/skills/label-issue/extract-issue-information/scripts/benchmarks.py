# Copyright 2020-2025 Intel Corporation
# Licensed under the Apache License, Version 2.0

import os
import re
import sys

from patterns import _BENCHMARK_LIST_FILES


def _benchmark_dir_candidates(pytorch_folder=None):
    roots = []
    if pytorch_folder:
        roots.append(pytorch_folder)
    env_pf = os.environ.get('PYTORCH_FOLDER')
    if env_pf:
        roots.append(env_pf)
    roots.append(os.getcwd())
    seen = set()
    for root in roots:
        for rel in ('third_party/torch-xpu-ops/.ci/benchmarks', '.ci/benchmarks'):
            path = os.path.join(os.path.expanduser(root), rel)
            if path not in seen:
                seen.add(path)
                yield path


def _parse_model_list_file(path):
    models = []
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            # Lines are "ModelName,batch" or "ModelName batch"; take the name only.
            name = re.split(r'[,\s]', line.strip(), maxsplit=1)[0]
            if name:
                models.append(name)
    return models


def _load_benchmark_models(pytorch_folder=None):
    result = {}
    for bench, filename in _BENCHMARK_LIST_FILES.items():
        models = []
        seen = set()
        for bench_dir in _benchmark_dir_candidates(pytorch_folder):
            # The top-level list holds the default set; p0/p1/p2 hold the
            # priority tiers, and names such as hf_Bert appear ONLY in a tier
            # file. Read every tier so detection covers the full model set.
            found_here = False
            for sub in ('', 'p0', 'p1', 'p2'):
                path = os.path.join(bench_dir, sub, filename) if sub else os.path.join(bench_dir, filename)
                if not os.path.isfile(path):
                    continue
                try:
                    names = _parse_model_list_file(path)
                except OSError:
                    continue
                for name in names:
                    if name not in seen:
                        seen.add(name)
                        models.append(name)
                found_here = True
            if found_here and models:
                break
        if models:
            result[bench] = models
    return result


def set_benchmark_models(pytorch_folder=None):
    """Load the authoritative model lists and rebuild the derived regex.

    Missing lists are NOT substituted with hardcoded names: a stale list
    silently mis-detects e2e models. The affected buckets stay empty and a
    warning names them, so label- and path-based e2e signals still work.
    """
    global HUGGINGFACE_MODELS, TIMM_MODELS, TORCHBENCH_MODELS, _BENCHMARK_MODEL_RE
    loaded = _load_benchmark_models(pytorch_folder)
    HUGGINGFACE_MODELS = loaded.get('huggingface', [])
    TIMM_MODELS = loaded.get('timm', [])
    TORCHBENCH_MODELS = loaded.get('torchbench', [])
    _BENCHMARK_MODEL_RE = _build_benchmark_model_regex()
    missing = [b for b in _BENCHMARK_LIST_FILES if not loaded.get(b)]
    if missing:
        print(
            f"WARNING: benchmark model lists not found for {', '.join(sorted(missing))}; "
            f"e2e model-name detection is disabled for those. Looked for "
            f"{'/'.join(_BENCHMARK_LIST_FILES[b] for b in sorted(missing))} under "
            f"<root>/.ci/benchmarks and <root>/third_party/torch-xpu-ops/.ci/benchmarks "
            f"(roots: --pytorch-folder, $PYTORCH_FOLDER, cwd). Pass --pytorch-folder "
            f"to point at a checkout.",
            file=sys.stderr,
        )
    return loaded


def _build_benchmark_model_regex():
    names = {m.lower() for m in HUGGINGFACE_MODELS + TIMM_MODELS + TORCHBENCH_MODELS}
    names = sorted(names, key=len, reverse=True)
    if not names:
        return None
    alternation = '|'.join(re.escape(n) for n in names)
    # (?<![\w-]) / (?![\w-]) give whole-token matching so short names like
    # 'sage' or 'moco' don't match inside unrelated words.
    return re.compile(r'(?<![\w-])(?:' + alternation + r')(?![\w-])')


HUGGINGFACE_MODELS = []


TIMM_MODELS = []


TORCHBENCH_MODELS = []


_BENCHMARK_MODEL_RE = None


def mentions_benchmark_model(text):
    return bool(_BENCHMARK_MODEL_RE and _BENCHMARK_MODEL_RE.search(text))


def identify_benchmark(model_name):
    """Identify benchmark from model name using exact matching"""
    model_lower = model_name.lower()

    # Check torchbench models first (includes hf_* and timm_* wrapped versions)
    for m in TORCHBENCH_MODELS:
        m_lower = m.lower()
        if m_lower == model_lower or m_lower.replace('_', '') == model_lower.replace('_', ''):
            return 'torchbench'

    # Check huggingface models (official class names)
    for m in HUGGINGFACE_MODELS:
        m_lower = m.lower()
        if m_lower == model_lower or m_lower.replace('_', '') == model_lower.replace('_', ''):
            return 'huggingface'

    # Check timm models
    for m in TIMM_MODELS:
        m_lower = m.lower()
        if m_lower == model_lower or m_lower.replace('_', '') == model_lower.replace('_', ''):
            return 'timm'

    return 'unknown'
