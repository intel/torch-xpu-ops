# Copyright 2020-2025 Intel Corporation
# Licensed under the Apache License, Version 2.0

import re

import benchmarks
from benchmarks import identify_benchmark
from patterns import KNOWN_TEST_TYPES, _E2E_CMD_PATTERNS, _E2E_DTYPE_PATTERNS


def test_case_source(test_file):
    """Return 'torch-xpu-ops' if the file is an XPU test, else 'pytorch'; '' if empty."""
    if not test_file:
        return ""
    base = test_file.replace('\\', '/').rsplit('/', 1)[-1]
    stem = base[:-3] if base.endswith('.py') else base
    if base.endswith('_xpu.py') or stem.endswith('_xpu'):
        return 'torch-xpu-ops'
    return 'pytorch'


def extract_e2e_reproducer(body, title):
    """Extract reproducer command from issue body"""
    text = f"{title} {body}"

    reproducer_lines = []

    # Look for code blocks with commands (between ``` and ```)
    if '```' in text:
        parts = text.split('```')
        for i, part in enumerate(parts):
            # Code blocks are odd-indexed (1, 3, 5, ...)
            if i % 2 == 1:  # This is a code block content
                part_stripped = part.strip()
                if part_stripped:
                    lines = part_stripped.split('\n')
                    for line in lines:
                        line_stripped = line.strip()
                        # Look for actual commands (python, pytest, etc.)
                        if line_stripped and (line_stripped.startswith(('python', 'pytest', 'XPU_', './')) or 'python' in line_stripped.lower()):
                            if not line_stripped.startswith('#'):
                                reproducer_lines.append(line_stripped)
                    # If we found a command, use it
                    if reproducer_lines:
                        break

    # Also look for command patterns without code blocks
    if not reproducer_lines:
        # Look for python or pytest command patterns
        for pattern in _E2E_CMD_PATTERNS:
            matches = re.findall(pattern, text)
            for match in matches:
                reproducer_lines.append(match.strip())

    if not reproducer_lines:
        # Generic reproducer from title
        return title[:200]

    # Join and limit to 3 lines
    return '\n'.join(reproducer_lines[:3])


def parse_e2e_info(body, title):
    """Parse e2e benchmark information from issue body"""
    e2e_info = []

    text = f"{title} {body}"

    # Get reproducer
    reproducer = extract_e2e_reproducer(body, title)

    # Check for model names in title or body
    all_model_names = (benchmarks.HUGGINGFACE_MODELS + benchmarks.TIMM_MODELS
                       + benchmarks.TORCHBENCH_MODELS)

    # Extract phase (training/inference)
    phase = 'inference'
    if 'training' in text.lower():
        phase = 'training'
    elif 'train' in text.lower():
        phase = 'training'

    # Extract dtype
    dtype = 'float32'
    for pattern, dt in _E2E_DTYPE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            dtype = dt
            break

    # Word boundary is required: a bare 'amp' substring also matches 'sample',
    # 'example', and 'clamp', which appear in almost every issue body.
    amp = bool(re.search(r'--amp\b|\bamp\b', text, re.IGNORECASE))

    # Extract test type
    test_type = 'accuracy'
    if 'throughputs' in text.lower() or 'performance' in text.lower() or 'latency' in text.lower():
        test_type = 'performance'

    # Extract backend
    backend = 'inductor'
    if '--backend=' in text:
        match = re.search(r'--backend=(\w+)', text)
        if match:
            backend = match.group(1)
    elif 'eager' in text.lower():
        backend = 'eager'
    elif 'inductor' in text.lower():
        backend = 'inductor'

    # Extract disable-cudagraphs
    disable_cudagraphs = 'no'
    if 'disable-cudagraphs' in text.lower() or 'disable_cudagraphs' in text.lower():
        disable_cudagraphs = 'yes'

    # Find model in body - need exact model name, not partial match
    found_models = set()
    for model in all_model_names:
        # Use word boundary to avoid partial matches
        if re.search(r'\b' + re.escape(model.lower()) + r'\b', text.lower()):
            benchmark = identify_benchmark(model)
            if benchmark != 'unknown' and model not in found_models:
                found_models.add(model)
                e2e_info.append({
                    'reproducer': reproducer,
                    'benchmark': benchmark,
                    'model': model,
                    'phase': phase,
                    'dtype': dtype,
                    'amp': amp,
                    'test_type': test_type,
                    'backend': backend,
                    'disable_cudagraphs': disable_cudagraphs,
                })

    # If no specific model found but looks like e2e issue
    if not e2e_info:
        if 'benchmark' in text.lower() or 'huggingface' in text.lower() or 'timm' in text.lower() or 'torchbench' in text.lower():
            # Try to identify benchmark from context
            if 'hf_' in text.lower() or 'huggingface' in text.lower():
                benchmark = 'huggingface'
            elif 'timm_' in text.lower() or 'timm.' in text.lower():
                benchmark = 'timm'
            elif 'torchbench' in text.lower():
                benchmark = 'torchbench'
            else:
                benchmark = 'unknown'

            e2e_info.append({
                'reproducer': reproducer,
                'benchmark': benchmark,
                'model': 'unknown',
                'phase': phase,
                'dtype': dtype,
                'amp': amp,
                'test_type': test_type,
                'backend': backend,
                'disable_cudagraphs': disable_cudagraphs,
            })

    return e2e_info


def map_origin_test_file(test_file):
    if not test_file:
        return ""
    match = re.search(r'test/xpu/(.+?)(?:_xpu)?\.py$', test_file)
    if match:
        return f"test/{match.group(1)}.py"
    if 'benchmarks/' in test_file:
        return test_file
    return test_file


def resolve_test_file(test_path):
    """Map a dotted test path to (test_file_rel, class_suffix, origin_file_rel).

    String-only reconstruction: this variant does no on-disk verification.
    A dotted path is split into leading path components (the directory/file
    part) and a trailing run of PascalCase segments (treated as a dotted
    class chain). The remaining leading segments become the module file, to
    which '.py' is appended. This guarantees a non-empty test_file for any
    non-empty input so downstream cases are never dropped for an unresolved
    path. The origin file is computed via map_origin_test_file.

    Returns ("", "", "") only when test_path is empty.
    """
    if not test_path:
        return "", "", ""
    parts = test_path.split('.')

    # Pop trailing PascalCase tokens as the class chain (e.g. ['ReproTests']).
    def _split_class_suffix(rel_parts):
        cls = []
        while rel_parts and rel_parts[-1] and rel_parts[-1][:1].isupper():
            cls.insert(0, rel_parts.pop())
        return rel_parts, '.'.join(cls)

    if 'torch-xpu-ops' in parts:
        try:
            i = parts.index('torch-xpu-ops')
            sub = parts[i + 1:]
            if sub and sub[0] == 'test':
                rel = list(sub[1:])
                rel, class_suffix = _split_class_suffix(rel)
                if rel:
                    fp_rel = 'torch-xpu-ops/test/' + '/'.join(rel) + '.py'
                    return fp_rel, class_suffix, map_origin_test_file(fp_rel)
        except ValueError:
            pass

    rel = list(parts[1:] if parts and parts[0] == 'test' else parts)
    rel, class_suffix = _split_class_suffix(rel)
    if rel:
        fp_rel = 'test/' + '/'.join(rel) + '.py'
        return fp_rel, class_suffix, map_origin_test_file(fp_rel)
    return "", "", ""


def parse_test_cases_from_body(body):
    cases = []

    if 'Cases:' in body:
        cases_section = body.split('Cases:')[1]

        end_markers = ['\n###', '\nVersions', '\n```']
        min_end = len(cases_section)
        for marker in end_markers:
            idx = cases_section.find(marker)
            if idx > 0 and idx < min_end:
                min_end = idx
        cases_section = cases_section[:min_end]

        lines = cases_section.split('\n')

        for line in lines:
            line = line.strip()

            if not line:
                continue

            if line.startswith('###') or line.startswith('...'):
                continue

            if line.startswith('~~') and line.endswith('~~'):
                continue

            parts = line.split(',')
            if len(parts) < 3:
                continue

            test_type = parts[0].strip()
            if test_type not in KNOWN_TEST_TYPES:
                continue

            field1 = parts[1].strip()
            field2 = parts[2].strip()

            # Two formats observed in the wild:
            #   A) op_ut,<dotted.module[.Class]>,<test_case>
            #   B) op_ut,,<dotted.module>            (module-level import error)
            # In (B) field1 is empty and field2 is the module path with no case.
            if field1:
                test_path = field1
                test_case = field2
                module_level = False
            else:
                test_path = field2
                test_case = ''
                module_level = True

            if not test_path:
                continue
            if not module_level:
                if not test_case or len(test_case) < 3:
                    continue
                if ' ' in test_case:
                    continue

            test_file, class_suffix, origin_file = resolve_test_file(test_path)
            test_class = class_suffix

            if not module_level and not test_class and '.' in test_case:
                head, _, tail = test_case.rpartition('.')
                if head and tail:
                    test_class = head
                    test_case = tail

            cases.append({
                'test_type': test_type,
                'test_file': test_file,
                'origin_test_file': origin_file,
                'test_class': test_class,
                'test_case': test_case,
                'module_level': module_level,
            })

    if 'test_cases:' in body:
        idx = body.find('test_cases:')
        cases_section = body[idx + len('test_cases:'):]
        end_markers = ['\n###', '\n## ', '\nVersions', '\n```']
        min_end = len(cases_section)
        for em in end_markers:
            ei = cases_section.find(em)
            if ei > 0 and ei < min_end:
                min_end = ei
        cases_section = cases_section[:min_end]
        for line in cases_section.split('\n'):
            stripped = line.strip()
            if not stripped or not stripped.startswith('- '):
                continue
            csv_part = stripped[2:].strip()
            parts = csv_part.split(',')
            if len(parts) < 3:
                continue
            test_type = parts[0].strip()
            if test_type not in KNOWN_TEST_TYPES:
                continue
            field1 = parts[1].strip()
            field2 = parts[2].strip()
            if field1:
                test_path, test_method = field1, field2
                module_level = False
            else:
                test_path, test_method = field2, ''
                module_level = True
            if not test_path:
                continue
            if not module_level:
                if not test_method or len(test_method) < 3 or ' ' in test_method:
                    continue
            test_file, class_suffix, origin_file = resolve_test_file(test_path)
            test_class = class_suffix
            if not module_level and not test_class and '.' in test_method:
                head, _, tail = test_method.rpartition('.')
                if head and tail:
                    test_class, test_method = head, tail
            cases.append({
                'test_type': test_type, 'test_file': test_file,
                'origin_test_file': origin_file, 'test_class': test_class,
                'test_case': test_method, 'module_level': module_level,
            })

    # Extract from pytest code blocks (format: pytest -v test/test_ops.py -k test_name)
    if '```' in body:
        code_blocks = body.split('```')
        for block in code_blocks:
            # Look for pytest patterns with test path and test method
            # Handles formats: test/test_ops.py or test/distributed/test_c10d_xccl.py::ClassName::method
            pytest_pattern = r'pytest\s+-v\s+(test[/a-zA-Z0-9_/.]+\.py(?:::[a-zA-Z0-9_]+)*)'
            matches = re.findall(pytest_pattern, block)
            for match in matches:
                test_path = match.strip()
                if '::' in test_path:
                    parts = test_path.split('::')
                    file_path = parts[0]
                    test_class = parts[1] if len(parts) > 1 else ""
                    # Only emit test_case when an explicit ::method segment is present.
                    # With just file::Class, the -k handler below produces the real
                    # method row; emitting test_method=class here yields a degenerate
                    # row where test_class == test_case.
                    test_method = parts[2] if len(parts) > 2 else ""
                    if test_method:
                        cases.append({
                            'test_type': 'ut',
                            'test_file': file_path,
                            'origin_test_file': file_path,
                            'test_class': test_class,
                            'test_case': test_method
                        })
                else:
                    # No class/method, just file
                    cases.append({
                        'test_type': 'ut',
                        'test_file': test_path,
                        'origin_test_file': test_path,
                        'test_class': '',
                        'test_case': ''
                    })

            # Also look for test_xpu,...,... format in code blocks
            test_xpu_pattern = r'(test_xpu),([a-zA-Z0-9_\.]+),([a-zA-Z0-9_]+)'
            matches = re.findall(test_xpu_pattern, block)
            for match in matches:
                test_type, test_path, test_method = match[0], match[1], match[2]
                test_class = ""
                if '.test_' in test_path:
                    # e.g., test.test_xpu.TestXpuAutocast -> TestXpuAutocast
                    class_parts = test_path.split('.test_')
                    if len(class_parts) > 1:
                        class_name = class_parts[1]
                        test_class = class_name.rsplit('.', 1)[-1]
                cases.append({
                    'test_type': test_type,
                    'test_file': test_path.replace('.', '/') + '.py',
                    'origin_test_file': test_path.replace('.', '/') + '.py',
                    'test_class': test_class,
                    'test_case': test_method
                })

            # Also handle pytest commands with -k pattern (extract test method from -k value)
            # Look for: pytest ... -k test_python_ref__refs_logspace_tensor_overload_xpu_float64
            k_pattern_matches = re.findall(r'-k\s+([a-zA-Z0-9_]+)', block)
            for test_name in k_pattern_matches:
                # Try to find associated test file in the same block
                pytest_v_match = re.search(r'pytest\s+-v\s+(test[/a-zA-Z0-9_]+\.py)', block)
                if pytest_v_match:
                    file_path = pytest_v_match.group(1)
                    cases.append({
                        'test_type': 'ut',
                        'test_file': file_path,
                        'origin_test_file': file_path,
                        'test_class': '',
                        'test_case': test_name
                    })

    # Extract from pytest commands outside code blocks
    # Look for patterns like: pytest -v test/test_ops.py -k test_name
    re_pattern = r'pytest\s+-v\s+(test[/a-zA-Z0-9_]+\.py)\s*-k\s+([a-zA-Z0-9_]+)'
    matches = re.findall(re_pattern, body)
    for file_path, test_name in matches:
        cases.append({
            'test_type': 'ut',
            'test_file': file_path,
            'origin_test_file': file_path,
            'test_class': '',
            'test_case': test_name
        })

    if 'benchmarks/dynamo/' in body:
        matches = re.findall(r'(python\s+benchmarks/dynamo/[^\s]+)', body)
        for match in matches:
            test_file = match.replace('python ', '').strip()
            cases.append({
                'test_type': 'e2e',
                'test_file': test_file,
                'origin_test_file': test_file,
                'test_class': '',
                'test_case': match.strip()
            })

    if 'pytest' in body:
        k_match = re.search(r'pytest[^-]*(-k\s+[^\s]+)?', body)
        if k_match and k_match.group(1):
            cases.append({
                'test_type': 'ut',
                'test_file': '',
                'origin_test_file': '',
                'test_class': '',
                'test_case': k_match.group(1).strip()
            })

    return cases


def dedup_test_cases(cases):
    # Dedup preserving first-occurrence order. A dict with a "benchmark" key is
    # e2e shape; everything else is unit-test shape. For UT-shape, an empty
    # test_case row is dropped only when another row for the same test_file
    # carries a non-empty test_case (empty rows survive as sole file info).
    ut_files_with_case = set()
    for c in cases:
        if "benchmark" not in c and c.get("test_case", ""):
            ut_files_with_case.add(c.get("test_file", ""))

    seen = set()
    result = []
    for c in cases:
        if "benchmark" in c:
            key = (
                c.get("benchmark", ""),
                c.get("model", ""),
                c.get("phase", ""),
                c.get("dtype", ""),
                c.get("backend", ""),
                c.get("test_type", ""),
            )
        else:
            test_file = c.get("test_file", "")
            test_case = c.get("test_case", "")
            if not test_case and test_file in ut_files_with_case:
                continue
            key = (test_file, c.get("test_class", ""), test_case)
        if key in seen:
            continue
        seen.add(key)
        result.append(c)
    return result


def is_unittest_issue(body, title, labels, test_cases):
    """Heuristic: is this issue about a unit test? True if ANY signal holds."""
    for label in labels:
        if 'module: ut' in (label.get('name', '') or '').lower():
            return True
    for tc in test_cases:
        if 'benchmark' in tc:
            continue
        tf = tc.get('test_file', '') or ''
        base = tf.rsplit('/', 1)[-1]
        if (tf.startswith('test/') or tf.startswith('test/xpu/')
                or '/test/' in tf or tf.startswith('test_')
                or base.startswith('test_')):
            return True
        if (tc.get('test_class', '') or '').startswith('Test'):
            return True
        if (tc.get('test_case', '') or '').startswith('test_'):
            return True
    return False
