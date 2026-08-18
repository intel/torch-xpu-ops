# Copyright 2020-2025 Intel Corporation
# Licensed under the Apache License, Version 2.0

import re

from patterns import _REPRO_PROSE_WORDS, _REPRO_SIMPLE_STARTS


# Patterns for extracting the traceback when an issue has no parsed test cases.
# A chained traceback ("The above exception ...", "During handling ...") is one
# logical traceback split by blank lines, so a blank line only terminates the
# match when what follows is not another link, frame, or nested header.
_TB_CONTINUATION = (
    r'(?:The above exception was the direct cause of the following exception'
    r'|During handling of the above exception, another exception occurred'
    r'|Traceback \(most recent call last\):'
    r'|File\s+")'
)

TRACEBACK_RE = re.compile(
    r'Traceback \(most recent call last\):'
    r'(?:'
    r'(?!\n(?:[ \t]*\n)+(?![ \t]*' + _TB_CONTINUATION + r'))'
    r'(?!\n###)(?!\n```)'
    r'.'
    r')*',
    re.DOTALL,
)


# Headerless traceback: an error/exception line followed (possibly after a
# blank-line gap) by one or more `File "...", line N` frames plus their
# indented source lines. Handles reporters that show the error above the
# frames without the canonical "Traceback (most recent call last):" header.
# The name alternation covers PEP 8 `*Error`/`*Exception`/`*Warning` plus the
# stdlib exceptions that follow neither convention.
_BARE_EXC_NAMES = (
    r'KeyboardInterrupt|SystemExit|StopIteration|StopAsyncIteration'
    r'|GeneratorExit|MemoryError|RecursionError|SystemError'
)

HEADERLESS_TB_RE = re.compile(
    r'^[ \t]*(?:[A-Za-z_][\w\.]*(?:Error|Exception|Warning)|(?:\w+\.)*(?:'
    + _BARE_EXC_NAMES + r')):?[^\n]*\n'
    r'(?:[ \t]*\n)*'
    r'(?:[ \t]*File\s+"[^"]+",\s+line\s+\d+[^\n]*\n(?:[ \t]+[^\n]*\n)*)+',
    re.MULTILINE,
)


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


def _looks_like_prose(line):
    """True when a command-prefixed line is really an English sentence.

    Shell joins with operators (`&&`, `|`, `;`), so a standalone function word
    such as `and` or `the` means the line is prose, not a command. Trailing
    punctuation is NOT a signal: `pip install -e .` legitimately ends in a dot.
    """
    for tok in re.split(r'\s+', line)[1:]:
        if tok.strip('.,;:()[]`"\'').lower() in _REPRO_PROSE_WORDS:
            return True
    return False


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
    seen = set()
    ordered = []
    for raw in body.split('\n'):
        line = list_marker_re.sub('', raw).strip().rstrip('`').strip()
        if not line or line.startswith('#'):
            continue
        matched = False
        if line.startswith(_REPRO_SIMPLE_STARTS):
            matched = True
        elif (env_python_re.search(line) or ze_prefix_re.search(line)
              or env_prefix_re.search(line)):
            matched = True
        if matched and _looks_like_prose(line):
            matched = False
        if matched and line not in seen:
            seen.add(line)
            ordered.append(line)

    return '\n'.join(ordered)


def extract_pr_link(body, title):
    """Return the PR URL the issue is tied to, or "" when there is none.

    Only an explicit /pull/ URL is trusted here. An `owner/repo#N` shorthand is
    ambiguous (issue or PR), so it is returned by extract_pr_candidate() for the
    caller to resolve against the API instead of being guessed at.
    """
    if not body and not title:
        return ""

    text = f"{title}\n{body}" if title else (body or "")

    # https://github.com/<owner>/<repo>/pull/<number>
    m = re.search(
        r'https?://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/pull/(\d+)',
        text,
    )
    if m:
        return f"https://github.com/{m.group(1)}/pull/{m.group(2)}"

    return ""


def extract_pr_candidate(body, title):
    """Return (owner, repo, number) for the first `owner/repo#N`, else None.

    Ambiguous by nature (issue or PR); github.resolve_ref_kind() decides.
    """
    if not body and not title:
        return None

    text = f"{title}\n{body}" if title else (body or "")

    # Lookbehind drops a bare #N: no owner/repo means a same-repo issue ref.
    m = re.search(
        r'(?<![/\w])([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)#(\d+)',
        text,
    )
    if not m:
        return None
    return m.group(1), m.group(2), int(m.group(3))


# Signals that an issue is PR/branch-related even when no explicit PR URL is
# found. Used for low_confidence flagging.
_PR_CONTEXT_SIGNALS_RE = re.compile(
    r'this PR|my branch|cherry.?pick|backport|/actions/runs/|CI failure on',
    re.IGNORECASE,
)
