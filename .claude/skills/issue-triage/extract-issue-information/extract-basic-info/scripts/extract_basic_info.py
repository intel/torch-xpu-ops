# Copyright 2020-2025 Intel Corporation
# Licensed under the Apache License, Version 2.0

# pyright: reportUnusedImport=false, reportUnusedParameter=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnusedVariable=false, reportImplicitStringConcatenation=false

import re
import json
import sys
import argparse
import subprocess


def generate_summary(body, title):
    # Summary based on issue title
    return title[:150]


def classify_issue_type(body, title, labels):
    text = f"{title} {body}".lower()
    
    for label in labels:
        ln = label.get('name', '').lower()
        if 'task' == ln or 'internal task' in ln:
            return 'internal task'
    
    performance_keywords = [
        'performance regression', 'performance dropped', 'performance issue',
        'latency', 'throughput', 'slow performance', 'performance slow',
        'execution time', 'runtime performance', 'performance fail'
    ]
    
    has_performance_keyword = any(k in text for k in performance_keywords)
    
    bug_keywords = [
        'assertionerror', 'runtimeerror', 'valueerror', 'typeerror', 'indexerror',
        'keyerror', 'importerror', 'notimplementederror', 'attributeerror',
        'inductorerror', 'crash', 'fail', 'bug', 'error', 'not implemented',
        'not supported', 'missing', 'incorrect', 'wrong', 'unexpected'
    ]
    
    has_bug_keyword = any(k in text for k in bug_keywords)
    
    # Accuracy issues are a specialized subtype of bug (numerical correctness).
    # They must be checked before the generic bug catch-all below, because
    # bug_keywords contains broad terms ('fail', 'error', 'wrong', 'incorrect')
    # that would otherwise shadow every accuracy issue.
    accuracy_keywords = [
        'accuracy', 'accurate', 'inaccurate', 'accuracy fail',
        'tolerance', 'atol', 'rtol', 'within tolerance', 'exceeds tolerance',
        'not close', 'allclose', 'assert_close', 'assertallclose',
        'tensor-likes are not close', 'element mismatch', 'numerical mismatch',
        'numerical error', 'numerical difference',
        'eager_two_runs_differ', 'reference_in_float',
        'wrong result', 'wrong output', 'incorrect result', 'incorrect output',
        'result differs', 'output differs', 'does not match', 'do not match',
    ]
    has_accuracy_keyword = any(k in text for k in accuracy_keywords)
    
    feature_keywords = ['feature request', 'support for', 'implement', 'add support', 'need feature']
    has_feature_keyword = any(k in text for k in feature_keywords)
    
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
    
    # Check for specific E2E benchmark paths (not just the word 'benchmark')
    e2e_patterns = [
        r'benchmarks/dynamo/',           # torch-xpu-ops benchmark scripts
        r'benchmarks/timm/',             # timm benchmark
        r'benchmarks/huggingface/',     # huggingface benchmark
        r'benchmarks/torchbench/',      # torchbench benchmark
        r'run_benchmark\.py',            # torchbenchmark runner
    ]
    
    for pattern in e2e_patterns:
        if re.search(pattern, text):
            return True
    
    # Check for model names from benchmark model lists with explicit benchmark framework mention
    # Only for specific benchmark prefixes
    benchmark_model_prefixes = ['hf_', 'timm_']  # e.g., hf_Albert, timm_resnet50
    
    has_model = False
    has_benchmark_context = False
    
    for prefix in benchmark_model_prefixes:
        if prefix in text:
            has_model = True
            break
    
    # Must have explicit benchmark framework mention (as test framework)
    if has_model:
        benchmark_paths = ['benchmarks/dynamo', 'run_benchmark', 'torchbenchmark', 'benchmark.py']
        for kw in benchmark_paths:
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
    
    pytest_patterns = [
        r'pytest\s+.*test[/._]',
        r'python\s+.*test[/._]',
        r'test/test_',
        r'test/xpu/test_',
    ]
    
    has_test_pattern = False
    for pattern in pytest_patterns:
        if re.search(pattern, text):
            has_test_pattern = True
            break
    
    build_patterns = [
        r'\[win\]\[build\]',
        r'build from source',
        r'compile from source', 
        r'source build',
        r'build script',
        r'BUILD_SEPARATE',
        r'BUILD_SHARED',
        r'cmake build',
        r'cmake error',
        r'cmake fail',
        r'setup\.py install',
        r'pip install -e \.',
        r'python setup\.py develop',
    ]
    
    has_build = any(re.search(p, text, re.IGNORECASE) for p in build_patterns)
    
    infra_patterns = [
        r'workflow\s+(error|fail|issue|problem)',
        r'github\s+action\s+(error|fail|issue)',
        r'azure\s+pipeline\s+(error|fail)',
        r'ci\s+(runner|config|setup)\s+(error|fail)',
        r'runner\s+(error|fail|timeout)\s+in\s+ci',
        r'checkout\s+(error|fail)\s+in\s+(workflow|ci)',
        r'githubaction',
    ]
    
    has_infra = any(re.search(p, text) for p in infra_patterns)
    
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
    labels_str = ', '.join([l.get('name', '') for l in labels]).lower()
    
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
    torch_ops = [
        'add', 'sub', 'mul', 'div', 'matmul', 'mm', 'dot', 'vdot', 'bmm',
        'addmm', 'addmv', 'addbmm', 'smm', 'spmm', 'mm', 'mv', 'vecdot',
        'conv', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose',
        'batch_norm', 'layer_norm', 'group_norm', 'instance_norm',
        'dropout', 'embedding', 'linear', 'lstm', 'gru', 'rnn',
        'softmax', 'log_softmax', 'sigmoid', 'tanh', 'relu', 'leaky_relu',
        'pool', 'avg_pool', 'max_pool', 'adaptive_pool',
        'fft', 'ifft', 'fft2', 'ifft2',
        'chunk', 'split', 'view', 'reshape', 'transpose', 'permute',
        'cat', 'stack', 'gather', 'scatter', 'index', 'where',
        'sum', 'mean', 'std', 'var', 'min', 'max', 'argmin', 'argmax',
        'norm', 'linalg.norm', 'linalg.matrix_norm', 'linalg.vector_norm',
        'eig', 'svd', 'qr', 'cholesky', 'solve', 'inverse',
        'det', 'logdet', 'slogdet', 'trace',
        'clone', 'copy_', 'to', 'cuda', 'cpu', 'xpu', 'device',
        'zeros', 'ones', 'empty', 'full', 'arange', 'linspace', 'logspace',
        'tensor', 'scalar_tensor', 'tensor.tensor',
        'getitem', 'setitem', 'call', 'forward', 'backward',
        'relu', 'gelu', 'silu', 'mish', 'softplus', 'elu', 'selu', 'celu',
        'flash_attention', 'scaled_dot_product_attention', 'sdpa',
        'interpolate', 'grid_sample', 'affine_grid',
        'grid_sampler', 'grid_sampler_2d',
        'bernoulli', 'normal', 'uniform', 'randn', 'rand', 'randint',
        'multinomial', ' poisson', 'exponential', 'geometric',
        'lerp', 'lerp_', 'fmod', 'remainder', 'nextafter',
        'linspace', 'logspace', 'geomspace',
        'complex', 'real', 'imag', 'angle',
        'conj', 'view_as_real', 'view_as_complex',
    ]
    
    module_keywords = [
        ('distributed', ['distributed', 'device_mesh', 'ProcessGroup', 'FSDP', 'DDP', 'c10d', 'tensor parallel']),
        ('inductor', ['inductor', 'inductor error', 'compile error', 'lower', 'kernel code']),
        ('dynamo', ['dynamo', 'torch.compile', '_dynamo', 'dynamo']),
        ('autograd', ['autograd', 'backward', 'grad', 'gradient']),
        ('aten_ops', ['aten::', 'torch.ops.aten', 'test_ops']),
        ('low_precision', ['quantization', 'int8', 'fp8', 'int4', 'amp', 'bf16', 'fp16', 'float8']),
        ('optimizer', ['optimizer', 'lr_scheduler', 'adam', 'sgd']),
        ('profiling', ['profiling', 'profile', 'benchmark']),
        ('fx', ['torch.fx', 'fx.', 'symbolic']),
        ('export', ['torch.export', 'exported']),
    ]
    
    # Check torch ops first
    for op in torch_ops:
        if re.search(rf'\b{re.escape(op)}\b', text):
            return 'aten_ops'
    
    for m, kw in module_keywords:
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
    version_headers = [
        r'###\s*version',
        r'###\s*versions',
        r'###\s*environment',
        r'###\s*reproduction',
        r'###\s*steps?\s+to\s+reproduce',
        r'###\s*additional\s*context',
    ]
    
    for header in version_headers:
        match = re.search(header, text, re.IGNORECASE)
        if match:
            text = text[:match.start()]
            break
    
    # Check for actual dependency in body (require context like "caused by", "issue", "depend on")
    dep_keywords = [
        ('transformers', [
            'caused by transformers', 'transformers issue', 'transformers bug',
            'depends on transformers', 'need transformers fix', 'waiting for transformers',
            'huggingface issue', 'huggingface bug', 'depends on huggingface'
        ]),
        ('AO', [
            'caused by torchao', 'torchao issue', 'torchao bug',
            'depends on torchao', 'need torchao fix', 'waiting for torchao'
        ]),
        ('oneDNN', [
            'caused by onednn', 'onednn issue', 'onednn bug', 'oneDNN issue',
            'depends on onednn', 'need onednn fix', 'waiting for onednn',
            'mkl-dnn issue', 'dnnl issue'
        ]),
        ('oneCCL', [
            'caused by oneccl', 'oneccl issue', 'oneccl bug',
            'depends on oneccl', 'need oneccl fix', 'waiting for oneccl',
            'xccl issue', 'ccl issue', 'depends on ccl'
        ]),
        ('oneMKL', [
            'caused by onemkl', 'onemkl issue', 'onemkl bug',
            'depends on onemkl', 'need onemkl fix', 'waiting for onemkl',
            'caused by mkl', 'mkl issue'
        ]),
        ('driver', [
            'caused by driver', 'driver issue', 'driver bug',
            'depends on driver', 'need driver fix', 'waiting for driver'
        ]),
        ('Triton', [
            'caused by triton', 'triton issue', 'triton bug',
            'depends on triton', 'need triton fix', 'waiting for triton',
            'triton-xpu issue', 'tl\\. issue'
        ]),
        ('oneAPI', [
            'caused by oneapi', 'oneapi issue', 'oneapi bug', 'sycl issue',
            'depends on oneapi', 'need oneapi fix', 'waiting for oneapi',
            'icpx issue', 'dpcpp issue', 'sycl compiler issue'
        ]),
    ]
    
    for d, kw in dep_keywords:
        if any(k in text for k in kw):
            return d
    
    return ''


# Patterns for extracting error/traceback when an issue has no parsed test cases.
ERROR_LINE_RE = re.compile(
    r'^\s*(?:[A-Za-z_][\w\.]*(?:Error|Exception|Warning)|RuntimeError|AssertionError|'
    r'ValueError|TypeError|IndexError|KeyError|ImportError|NotImplementedError|'
    r'AttributeError|InductorError):\s*.+',
    re.MULTILINE,
)
TRACEBACK_RE = re.compile(
    r'Traceback \(most recent call last\):.*?(?=\n\s*\n|\n###|\n```|\Z)',
    re.DOTALL,
)
# Headerless traceback: an error/exception line followed (possibly after a
# blank-line gap) by one or more `File "...", line N` frames plus their
# indented source lines. Handles reporters that show the error above the
# frames without the canonical "Traceback (most recent call last):" header.
HEADERLESS_TB_RE = re.compile(
    r'^[ \t]*(?:[A-Za-z_][\w\.]*(?:Error|Exception|Warning)):[^\n]*\n'
    r'(?:[ \t]*\n)*'
    r'(?:[ \t]*File\s+"[^"]+",\s+line\s+\d+[^\n]*\n(?:[ \t]+[^\n]*\n)*)+',
    re.MULTILINE,
)


def extract_error_message(body):
    if not body:
        return ""
    match = ERROR_LINE_RE.search(body)
    if match:
        return match.group(0).strip()
    return ""


def extract_traceback(body):
    if not body:
        return ""
    match = TRACEBACK_RE.search(body)
    if match:
        return match.group(0).strip()
    match = HEADERLESS_TB_RE.search(body)
    if match:
        return match.group(0).strip()
    return ""


def extract_reproduce_steps(body, title):
    """Extract shell command lines from the issue body (commands only).

    Scans the whole body (fenced code blocks and inline). Returns every
    matching shell command line in first-occurrence order, de-duplicated,
    joined by newlines. No cap, no title fallback. Returns "" if none found.
    Prose lines are never included.
    """
    if not body:
        return ""

    # Leading markdown list markers to strip before matching (e.g. "- ", "* ",
    # "1. ", "> ") plus any surrounding whitespace/backticks.
    list_marker_re = re.compile(r'^\s*(?:[-*+]\s+|\d+\.\s+|>\s*)?`*\s*')
    # Environment-prefixed invocation: one or more VAR=value tokens then python
    # (or a bare ZE_AFFINITY_MASK=... prefix on any command).
    env_python_re = re.compile(
        r'^(?:XPU_\w*|PYTORCH_\w*)=\S+\s+.*\bpython\b'
    )
    ze_prefix_re = re.compile(r'^ZE_AFFINITY_MASK=\S+\s+\S+')
    # Generic environment-variable-prefixed command: VAR=value followed by a
    # command token (e.g. "FOO=1 git clone x").
    env_prefix_re = re.compile(r'^[A-Z_][A-Z0-9_]*=\S+\s+\S+')
    simple_starts = (
        'pytest', 'python3', 'python', './', 'pip install',
        'cd ', 'export ', 'bash ', 'sh ', 'source ', 'git ',
        'cmake', 'make', 'ninja', 'wget ', 'curl ', 'conda ', 'pip ',
    )

    seen = set()
    ordered = []
    for raw in body.split('\n'):
        line = list_marker_re.sub('', raw).strip().rstrip('`').strip()
        if not line or line.startswith('#'):
            continue
        matched = False
        if line.startswith(simple_starts):
            matched = True
        elif (env_python_re.search(line) or ze_prefix_re.search(line)
              or env_prefix_re.search(line)):
            matched = True
        if matched and line not in seen:
            seen.add(line)
            ordered.append(line)

    return '\n'.join(ordered)


# Stack-frame line inside a Python traceback, e.g. File "x.py", line 5.
_TB_FRAME_RE = re.compile(r'File\s+"[^"]+",\s+line\s+\d+')
# Fallback error/exception/warning token when ERROR_LINE_RE does not hit.
_TB_ERROR_TOKEN_RE = re.compile(r'\w*(?:Error|Exception|Warning)\b')


def traceback_has_stack_and_error(tb):
    """True only if tb has a stack-frame line AND an error/exception message."""
    if not tb:
        return False
    has_frame = bool(_TB_FRAME_RE.search(tb))
    has_error = bool(ERROR_LINE_RE.search(tb)) or bool(_TB_ERROR_TOKEN_RE.search(tb))
    return has_frame and has_error


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


# Platform code -> keyword list, checked in order (most specific first).
# Short/ambiguous codes use word-boundary regex to avoid false hits.
_PLATFORM_KEYWORDS = [
    ('PVC', ['ponte vecchio', 'data center gpu max', 'gpu max 1550',
             'gpu max 1100', 'max 1550', 'max 1100', r'\bpvc\b',
             r'\b1550\b', r'\b1100\b']),
    ('BMG', ['battlemage', r'\bb580\b', r'\bb570\b', r'\bbmg\b']),
    ('ARC', ['alchemist', r'\ba770\b', r'\ba750\b', r'\ba380\b',
             r'\barc\b', 'arc a', 'arc graphics']),
    ('ARL', ['arrow lake', r'\barl\b']),
    ('LNL', ['lunar lake', r'\blnl\b']),
    ('MTL', ['meteor lake', r'\bmtl\b']),
    ('CRI', ['crescent island', r'\bcri\b']),
]


def extract_platform(body):
    """Return canonical platform code from the whole body, most specific first."""
    if not body:
        return ""
    for code, keywords in _PLATFORM_KEYWORDS:
        for kw in keywords:
            if kw.startswith(r'\b') or kw.endswith(r'\b'):
                if re.search(kw, body, re.IGNORECASE):
                    return code
            elif kw in body.lower():
                return code
    return ""


def test_case_source(test_file):
    """Return 'torch-xpu-ops' if the file is an XPU test, else 'pytorch'; '' if empty."""
    if not test_file:
        return ""
    base = test_file.replace('\\', '/').rsplit('/', 1)[-1]
    stem = base[:-3] if base.endswith('.py') else base
    if base.endswith('_xpu.py') or stem.endswith('_xpu'):
        return 'torch-xpu-ops'
    return 'pytorch'


# Known test types accepted in Cases:/test_cases: strategies.
KNOWN_TEST_TYPES = ['op_ut', 'op_extend', 'op_extended', 'e2e', 'benchmark', 'ut', 'test_xpu']

# Model lists from benchmarks
HUGGINGFACE_MODELS = [
    'AlbertForMaskedLM', 'AlbertForQuestionAnswering', 'AllenaiLongformerBase',
    'BartForCausalLM', 'BartForConditionalGeneration', 'BertForMaskedLM',
    'BertForQuestionAnswering', 'BlenderbotForCausalLM', 'BlenderbotForConditionalGeneration',
    'BlenderbotSmallForCausalLM', 'BlenderbotSmallForConditionalGeneration', 'CamemBert',
    'DebertaV2ForMaskedLM', 'DebertaV2ForQuestionAnswering', 'DistilBertForMaskedLM',
    'DistilBertForQuestionAnswering', 'DistillGPT2', 'ElectraForCausalLM',
    'ElectraForQuestionAnswering', 'GoogleFnet', 'google/gemma-2-2b', 'google/gemma-3-4b-it',
    'GPT2ForSequenceClassification', 'GPTJForCausalLM', 'GPTJForQuestionAnswering', 'GPTNeoForCausalLM',
    'GPTNeoForSequenceClassification', 'LayoutLMForMaskedLM', 'LayoutLMForSequenceClassification',
    'M2M100ForConditionalGeneration', 'MBartForCausalLM', 'MBartForConditionalGeneration',
    'MegatronBertForCausalLM', 'MegatronBertForQuestionAnswering', 'meta-llama/Llama-3.2-1B',
    'mistralai/Mistral-7B-Instruct-v0.3', 'MobileBertForMaskedLM', 'MobileBertForQuestionAnswering',
    'MT5ForConditionalGeneration', 'openai/gpt-oss-20b', 'openai/whisper-tiny', 'OPTForCausalLM',
    'PegasusForCausalLM', 'PegasusForConditionalGeneration', 'PLBartForCausalLM',
    'PLBartForConditionalGeneration', 'Qwen/Qwen3-0.6B', 'RobertaForCausalLM', 'RobertaForQuestionAnswering',
    'T5ForConditionalGeneration', 'T5Small', 'TrOCRForCausalLM', 'XGLMForCausalLM',
    'XLNetLMHeadModel', 'YituTechConvBert'
]

TIMM_MODELS = [
    'adv_inception_v3', 'beit_base_patch16_224', 'botnet26t_256', 'cait_m36_384',
    'coat_lite_mini', 'convit_base', 'convmixer_768_32', 'convnext_base',
    'convnextv2_nano.fcmae_ft_in22k_in1k', 'crossvit_9_240', 'cspdarknet53', 'deit_base_distilled_patch16_224',
    'deit_tiny_patch16_224.fb_in1k', 'dla102', 'dm_nfnet_f0', 'dpn107', 'eca_botnext26ts_256',
    'eca_halonext26ts', 'ese_vovnet19b_dw', 'fbnetc_100', 'fbnetv3_b', 'gernet_l',
    'ghostnet_100', 'gluon_inception_v3', 'gmixer_24_224', 'gmlp_s16_224', 'hrnet_w18',
    'inception_v3', 'jx_nest_base', 'lcnet_050', 'levit_128', 'mixer_b16_224',
    'mixnet_l', 'mnasnet_100', 'mobilenetv2_100', 'mobilenetv3_large_100', 'mobilevit_s',
    'nfnet_l0', 'pit_b_224', 'pnasnet5large', 'poolformer_m36', 'regnety_002',
    'repvgg_a2', 'res2net101_26w_4s', 'res2net50_14w_8s', 'res2next50', 'resmlp_12_224',
    'resnest101e', 'rexnet_100', 'sebotnet33ts_256', 'selecsls42b', 'spnasnet_100',
    'swin_base_patch4_window7_224', 'swsl_resnext101_32x16d', 'tf_efficientnet_b0',
    'tf_mixnet_l', 'tinynet_a', 'tnt_s_patch16_224', 'twins_pcpvt_base', 'visformer_small',
    'vit_base_patch14_dinov2.lvd142m', 'vit_base_patch16_224', 'vit_base_patch16_siglip_256',
    'volo_d1_224', 'xcit_large_24_p8_224'
]

TORCHBENCH_MODELS = [
    'alexnet', 'Background_Matting', 'basic_gnn_edgecnn', 'basic_gnn_gcn', 'basic_gnn_gin',
    'basic_gnn_sage', 'BERT_pytorch', 'cm3leon_generate', 'dcgan', 'demucs', 'densenet121',
    'detectron2_fasterrcnn_r_101_c4', 'detectron2_fasterrcnn_r_101_dc5', 'detectron2_fasterrcnn_r_101_fpn',
    'detectron2_fasterrcnn_r_50_c4', 'detectron2_fasterrcnn_r_50_dc5', 'detectron2_fasterrcnn_r_50_fpn',
    'detectron2_fcos_r_50_fpn', 'detectron2_maskrcnn', 'detectron2_maskrcnn_r_101_c4', 'detectron2_maskrcnn_r_101_fpn',
    'detectron2_maskrcnn_r_50_c4', 'detectron2_maskrcnn_r_50_fpn', 'dlrm', 'doctr_det_predictor',
    'doctr_reco_predictor', 'drq', 'fastNLP_Bert', 'functorch_dp_cifar10', 'functorch_maml_omniglot',
    'hf_Albert', 'hf_Bart', 'hf_Bert', 'hf_Bert_large', 'hf_BigBird', 'hf_clip', 'hf_DistilBert',
    'hf_distil_whisper', 'hf_GPT2', 'hf_GPT2_large', 'hf_Longformer', 'hf_Reformer', 'hf_Roberta_base',
    'hf_T5', 'hf_T5_base', 'hf_T5_generate', 'hf_T5_large', 'hf_Whisper',
    'LearningToPaint', 'lennard_jones', 'llama', 'llama_v2_7b_16h', 'llava', 'maml', 'maml_omniglot',
    'microbench_unbacked_tolist_sum', 'mnasnet1_0', 'mobilenet_v2', 'mobilenet_v2_quantized_qat',
    'mobilenet_v3_large', 'moco', 'modded_nanogpt', 'moondream', 'nanogpt', 'nvidia_deeprecommender',
    'opacus_cifar10', 'phlippe_densenet', 'phlippe_resnet', 'pyhpc_equation_of_state',
    'pyhpc_isoneutral_mixing', 'pyhpc_turbulent_kinetic_energy', 'pytorch_CycleGAN_and_pix2pix',
    'pytorch_stargan', 'pytorch_unet', 'resnet152', 'resnet18', 'resnet50', 'resnet50_quantized_qat',
    'resnext50_32x4d', 'sam', 'sam_fast', 'shufflenet_v2_x1_0', 'simple_gpt', 'simple_gpt_tp_manual',
    'soft_actor_critic', 'speech_transformer', 'squeezenet1_1', 'stable_diffusion_text_encoder',
    'stable_diffusion_unet', 'Super_SloMo', 'tacotron2', 'timm_efficientdet', 'timm_efficientnet',
    'timm_nfnet', 'timm_regnet', 'timm_resnest', 'timm_vision_transformer', 'timm_vision_transformer_large',
    'timm_vovnet', 'torch_multimodal_clip', 'tts_angular', 'vgg16', 'vision_maskrcnn', 'yolov3',
    'codellama', 'DALLE2_pytorch', 'diffuser_instruct_pix2pix', 'fambench_dlrm', 'fambench_xlmr',
    'gat', 'gcn', 'hf_GPT2_generate', 'hf_mixtral', 'hf_MPT_7b_instruct', 'hf_Yi', 'lit_llama',
    'lit_llama_generate', 'lit_llama_lora', 'llama_v2_13b', 'llama_v2_70b', 'llama_v31_8b',
    'mistral_7b_instruct', 'orca_2', 'phi_1_5', 'phi_2', 'sage', 'stable_diffusion_xl', 'torchrec_dlrm'
]


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
        cmd_patterns = [
            r'(pytest\s+[^\n]+)',
            r'(python\s+test/[^\n]+)',
            r'(python\s+-m\s+pytest[^\n]+)',
            r'(XPU_QUANT_CONFIG=[^\n]+python[^\n]+)',
            r'(python\s+benchmarks/dynamo/[^\n]+)',
            r'(python\s+[^\n]+run_benchmark[^\n]+)',
        ]

        for pattern in cmd_patterns:
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
    all_model_names = HUGGINGFACE_MODELS + TIMM_MODELS + TORCHBENCH_MODELS

    # Extract phase (training/inference)
    phase = 'inference'
    if 'training' in text.lower():
        phase = 'training'
    elif 'train' in text.lower():
        phase = 'training'

    # Extract dtype
    dtype = 'float32'
    dtype_patterns = [
        (r'bfloat16|bf16', 'bfloat16'),
        (r'float16|fp16', 'float16'),
        (r'float32|fp32', 'float32'),
        (r'int8|int\s*8', 'int8'),
    ]
    for pattern, dt in dtype_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            dtype = dt
            break

    # Extract AMP (automatic mixed precision)
    amp = False
    if '--amp' in text.lower() or 'amp' in text.lower():
        amp = True

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
        idx = body.lower().find('test_cases:')
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
                        if '.' in class_name:
                            test_class = class_name.rsplit('.', 1)[1] if '.' in class_name else class_name
                        else:
                            test_class = class_name
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


def parse_issue_ref(ref, default_owner="intel", default_repo="torch-xpu-ops"):
    """Parse an issue reference into (owner, repo, number).

    A full GitHub issue URL supplies its own owner/repo. A bare issue number
    uses the provided defaults.
    """
    if isinstance(ref, int):
        return default_owner, default_repo, ref
    if not isinstance(ref, str):
        raise ValueError(f"Invalid issue reference: {ref!r}")

    ref = ref.strip()
    if ref.isdigit():
        return default_owner, default_repo, int(ref)

    # Full GitHub issue URL: https://github.com/OWNER/REPO/issues/N
    m = re.search(r"github\.com/([^/]+)/([^/]+)/issues/(\d+)(?:[/?#].*)?$", ref)
    if m:
        return m.group(1), m.group(2), int(m.group(3))

    raise ValueError(f"Invalid issue reference: {ref!r}")


def fetch_issue(owner, repo, number):
    cmd = ["gh", "api", f"repos/{owner}/{repo}/issues/{number}"]
    result = subprocess.run(cmd, capture_output=True, check=False, text=True)

    if result.returncode != 0:
        message = (result.stderr or result.stdout or "").strip()
        if "Not Found" in message:
            raise RuntimeError(f"Issue {number} not found in {owner}/{repo}")
        raise RuntimeError(f"Failed to fetch issue {number} from {owner}/{repo}: {message or 'gh api returned a non-zero exit code'}")

    stdout = (result.stdout or "").strip()
    if not stdout:
        raise RuntimeError(f"Failed to fetch issue {number} from {owner}/{repo}: empty response from gh api")

    try:
        issue = json.loads(stdout)  # pyright: ignore[reportAny]
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to fetch issue {number} from {owner}/{repo}: non-JSON response from gh api") from exc

    if not isinstance(issue, dict):
        raise RuntimeError(f"Failed to fetch issue {number} from {owner}/{repo}: unexpected JSON response")

    if issue.get("pull_request") is not None:
        raise SystemExit(f"{owner}/{repo}#{number} is a pull request, not an issue")

    return issue


def rest_to_core(issue):
    assignee = ""
    if issue.get("assignee"):
        assignee = issue["assignee"].get("login", "")
    elif issue.get("assignees"):
        first_assignee = issue["assignees"][0] or {}
        assignee = first_assignee.get("login", "")

    milestone = ""
    if issue.get("milestone"):
        milestone = issue["milestone"].get("title", "")

    return {
        "issue_id": issue["number"],
        "title": issue.get("title") or "",
        "status": issue.get("state") or "",
        "assignee": assignee,
        "reporter": issue.get("user", {}).get("login", ""),
        "labels": [label.get("name", "") for label in issue.get("labels", [])],
        "created_time": issue.get("created_at") or "",
        "updated_time": issue.get("updated_at") or "",
        "milestone": milestone,
    }


# Bare field names in PyTorchXPU project -> output key.
PYTORCHXPU_FIELD_MAP = {
    "Priority": "priority",
    "Status": "project_status",
    "Estimate": "project_estimate",
    "Depending": "project_depending",
    "Short Comment": "project_short_comments",   # singular per project schema
    "Short Comments": "project_short_comments",   # tolerate plural alias
}


def fetch_project_and_type(owner, repo, number):
    """Fetch native issueType and PyTorchXPU project field values for one issue.

    Uses `gh api graphql` (not requests) because gh CLI's auth handles the
    read:project scope correctly. On any failure the function degrades
    gracefully: it prints a warning to stderr and returns an all-empty dict.
    It never raises.
    """
    result = {
        "github_type": "",
        "priority": "",
        "project_status": "",
        "project_estimate": "",
        "project_depending": "",
        "project_short_comments": "",
    }

    query = """
    query($owner: String!, $name: String!, $number: Int!) {
      repository(owner: $owner, name: $name) {
        issue(number: $number) {
          issueType { name }
          projectItems(first: 20) {
            nodes {
              project { title number }
              fieldValues(first: 50) {
                nodes {
                  ... on ProjectV2ItemFieldTextValue        { text   field { ... on ProjectV2FieldCommon { name } } }
                  ... on ProjectV2ItemFieldSingleSelectValue { name   field { ... on ProjectV2FieldCommon { name } } }
                  ... on ProjectV2ItemFieldNumberValue      { number field { ... on ProjectV2FieldCommon { name } } }
                }
              }
            }
          }
        }
      }
    }
    """

    args = [
        "gh",
        "api",
        "graphql",
        "-f",
        f"query={query}",
        "-f",
        f"owner={owner}",
        "-f",
        f"name={repo}",
        "-F",
        f"number={int(number)}",
    ]

    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=120)
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        print(f"PyTorchXPU project fetch failed for issue {number}: {exc}", file=sys.stderr)
        return result

    if proc.returncode != 0:
        print(
            f"PyTorchXPU project fetch failed for issue {number}: {proc.stderr.strip()}",
            file=sys.stderr,
        )
        return result

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        print(
            f"PyTorchXPU project fetch returned non-JSON for issue {number}: {exc}",
            file=sys.stderr,
        )
        return result

    if data.get("errors"):
        print(
            f"PyTorchXPU project GraphQL errors for issue {number}: {data['errors']}",
            file=sys.stderr,
        )
        return result

    issue = ((data.get("data") or {}).get("repository") or {}).get("issue") or {}

    result["github_type"] = ((issue.get("issueType") or {}).get("name")) or ""

    nodes = (issue.get("projectItems") or {}).get("nodes") or []

    # Prefer the PyTorchXPU project item if present, but the field-name map is
    # the real filter, so process all items if no titled match exists.
    preferred = [n for n in nodes if ((n.get("project") or {}).get("title") == "PyTorchXPU")]
    items = preferred if preferred else nodes

    for item in items:
        for fv in (item.get("fieldValues") or {}).get("nodes") or []:
            field = fv.get("field") or {}
            fname = str(field.get("name") or "").strip()
            key = PYTORCHXPU_FIELD_MAP.get(fname)
            if key is None:
                continue
            raw = ""
            for candidate in (
                str(fv.get("name") or "").strip(),
                str(fv.get("text") or "").strip(),
                str(fv.get("number") or "").strip(),
            ):
                if candidate:
                    raw = candidate
                    break
            if key == "priority":
                m = re.search(r"\bP[0-3]\b", raw.upper())
                result["priority"] = m.group(0) if m else ""
            else:
                result[key] = raw

    return result


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


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Extract basic information for one torch-xpu-ops issue as JSON."
    )
    parser.add_argument("issue", help="Issue number or GitHub issue URL")
    parser.add_argument("--repo", help="owner/name for a bare issue number (default intel/torch-xpu-ops); ignored when a full URL is given")
    parser.add_argument("--output", help="Optional path to also write the JSON output")
    args = parser.parse_args(argv)

    default_owner, default_repo = "intel", "torch-xpu-ops"
    if args.repo:
        if "/" not in args.repo:
            print(f"Invalid --repo value: {args.repo!r} (expected owner/name)", file=sys.stderr)
            sys.exit(2)
        default_owner, default_repo = args.repo.split("/", 1)

    try:
        owner, repo, number = parse_issue_ref(args.issue, default_owner, default_repo)
    except ValueError as err:
        print(err, file=sys.stderr)
        sys.exit(2)

    # fetch_issue raises RuntimeError (fatal) or SystemExit (PR guard). Let the
    # SystemExit propagate; it carries a message and exits nonzero.
    try:
        issue = fetch_issue(owner, repo, number)
    except RuntimeError as err:
        print(err, file=sys.stderr)
        sys.exit(1)

    core = rest_to_core(issue)

    # Classifiers run on the RAW issue fields (labels is the list of dicts).
    body = issue.get("body") or ""
    title = issue.get("title") or ""
    labels = issue.get("labels") or []

    summary = generate_summary(body, title)
    itype = classify_issue_type(body, title, labels)
    module = classify_module(body, title, labels)
    test_module = classify_test_module(body, title, labels)
    dependency = get_dependency_from_body(body, labels)

    pt = fetch_project_and_type(owner, repo, number)

    # Build the test_cases list (all cases in the issue). For e2e issues use
    # the benchmark/model extractor; otherwise use the unit-test parser.
    if test_module == "e2e":
        test_cases = parse_e2e_info(body, title)
    else:
        test_cases = parse_test_cases_from_body(body)
    test_cases = dedup_test_cases(test_cases)

    # Tag each unit-test case with its source repo. A test file ending in
    # _xpu belongs to torch-xpu-ops; otherwise it is an upstream pytorch test.
    # E2E cases (dicts with a "benchmark" key) are not tagged.
    for tc in test_cases:
        if "benchmark" not in tc:
            tc["source"] = test_case_source(tc.get("test_file", ""))

    reproduce_steps = extract_reproduce_steps(body, title)
    traceback = extract_traceback(body)
    os_name = extract_os(body)
    platform = extract_platform(body)

    # Primary unit-test case: first UT-shape case (dict without a "benchmark"
    # key). Top-level test_file/test_class/test_case mirror it for convenience;
    # the full list remains in test_cases.
    primary_tf = primary_tc_class = primary_tc_case = ""
    for tc in test_cases:
        if "benchmark" not in tc:
            primary_tf = tc.get("test_file", "")
            primary_tc_class = tc.get("test_class", "")
            primary_tc_case = tc.get("test_case", "")
            break

    unittest_issue = is_unittest_issue(body, title, labels, test_cases)

    result = {
        "issue_id": core["issue_id"],
        "repo": f"{owner}/{repo}",
        "title": core["title"],
        "status": core["status"],
        "assignee": core["assignee"],
        "reporter": core["reporter"],
        "labels": core["labels"],
        "created_time": core["created_time"],
        "updated_time": core["updated_time"],
        "milestone": core["milestone"],
        "summary": summary,
        "type": itype,
        "github_type": pt["github_type"],
        "module": module,
        "test_module": test_module,
        "dependency": dependency,
        "priority": pt["priority"],
        "pytorchxpu_status": pt["project_status"],
        "pytorchxpu_estimate": pt["project_estimate"],
        "pytorchxpu_depending": pt["project_depending"],
        "pytorchxpu_short_comments": pt["project_short_comments"],
        "os": os_name,
        "platform": platform,
        "traceback": traceback,
        "reproduce_steps": reproduce_steps,
        "test_file": primary_tf,
        "test_class": primary_tc_class,
        "test_case": primary_tc_case,
        "test_cases": test_cases,
    }

    low_confidence = []
    # reproduce_steps: flag when no shell command found, UNLESS this is a unit
    # test issue (the test_file/test_case itself is the reproducer).
    if not reproduce_steps and not unittest_issue:
        low_confidence.append("reproduce_steps")
    # test_cases: flag when none parsed but the issue looks test-related.
    if not test_cases and test_module in ("ut", "e2e"):
        low_confidence.append("test_cases")
    result["low_confidence"] = low_confidence

    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")

    sys.exit(0)


if __name__ == "__main__":
    main()
