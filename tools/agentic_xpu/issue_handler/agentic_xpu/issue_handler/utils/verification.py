# Copyright 2024-2026 Intel Corporation
# Co-authored with GitHub Copilot
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Shared test-command extraction and execution helpers.

Previously these lived in three different forms:

* ``verify_existence._get_raw_test_reference`` — verbatim text for the
  test-verification LLM agent.
* ``verify_fix._get_test_command`` — re-exported alias of the above.
* ``code_fix._get_test_command`` — runnable form (strips ``cd ...&&``
  prefix, builds a ``pytest`` command from a Failed Tests section).

…and two different ``_run_test`` / ``_run_verification`` flavours with
subtly different definitions of "the test passed".  Consolidating them
here removes the drift and gives a single place to evolve the
verification semantics.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from .body_templates import parse_sections
from .logger import log
from .xpu_env import ENV_SETUP


# ---------------------------------------------------------------------------
# Test-command extraction
# ---------------------------------------------------------------------------

# Pulled out as constants so the regex isn't re-compiled on every call and so
# the two extraction modes use exactly the same patterns.
_REPRODUCER_FENCE_RE = re.compile(
    r"```(?:bash|sh|python|python3|py)?\s*\n(.+?)```", re.DOTALL)
_FAILED_TEST_BACKTICK_RE = re.compile(r"`([^`]+(?:::|-k\s+)[^`]*)`")
_FAILED_TEST_LIST_RE = re.compile(r"[-*]\s+`?(\S+::\S+)`?")
_CD_PREFIX_RE = re.compile(r"^cd\s+(?:<[^>]+>|/\S+)\s*&&\s*")


def extract_test_command(body: str, *, executable: bool = True) -> str | None:
    """Extract the reproducer / failing-test command from a formatted issue body.

    Args:
        body: Markdown body of the issue (already formatted by
            ``format_agent``).
        executable: When True (default) the returned string is meant to be
            run as-is via ``bash -c``: a leading ``cd <pytorch> &&`` is
            stripped from every line and a Failed Tests section is turned
            into a ``pytest -v "<test>"...`` command.  When False the raw
            text is returned verbatim — that's what the test-verification
            LLM agent wants so it can resolve paths itself.

    Returns:
        The extracted command string, or None if no command could be found.
    """
    sections = parse_sections(body)

    # --- 1. Reproducer section --------------------------------------------
    reproducer = sections.get("Reproducer", "").strip()
    if reproducer and not reproducer.startswith("_Pending"):
        fence = _REPRODUCER_FENCE_RE.search(reproducer)
        if fence:
            cmd = fence.group(1).strip()
            # Strip nested fence markers (LLMs sometimes double-wrap)
            cmd = re.sub(r"^```(?:bash|sh)?\s*\n", "", cmd)
            cmd = re.sub(r"\n```\s*$", "", cmd).strip()
            if cmd:
                if executable and not _is_python_script(cmd):
                    cmd = _strip_cd_prefix(cmd)
                return cmd
        # No fence: fall back to non-comment, non-fence lines.
        # If the body looks like Python, preserve indentation verbatim;
        # otherwise strip per-line (legacy shell-reproducer behavior).
        raw_lines = [
            l for l in reproducer.splitlines()
            if l.strip()
            and not l.lstrip().startswith("```")
        ]
        if _is_python_script("\n".join(raw_lines)):
            return "\n".join(raw_lines)
        lines = [
            l.strip() for l in raw_lines
            if not l.strip().startswith("#")
        ]
        if lines:
            return "\n".join(lines)

    # --- 2. Failed Tests section ------------------------------------------
    failed_tests = sections.get("Failed Tests", "").strip()
    if failed_tests:
        tests = _FAILED_TEST_BACKTICK_RE.findall(failed_tests)
        if not tests:
            tests = _FAILED_TEST_LIST_RE.findall(failed_tests)
        if tests:
            if executable:
                args = " ".join(f'"{t}"' for t in tests)
                return f"pytest -v {args}"
            return "\n".join(tests)

    return None


def _strip_cd_prefix(cmd: str) -> str:
    """Drop ``cd <pytorch> &&`` / ``cd /abs/path &&`` from every line."""
    cleaned = []
    for line in cmd.splitlines():
        cleaned_line = _CD_PREFIX_RE.sub("", line).strip()
        if cleaned_line:
            cleaned.append(cleaned_line)
    return "\n".join(cleaned)


# ---------------------------------------------------------------------------
# Test execution
# ---------------------------------------------------------------------------

DEFAULT_TEST_TIMEOUT = 600  # 10 minutes


def _is_python_script(cmd: str) -> bool:
    """Heuristic: does this look like a Python reproducer (not a shell command)?"""
    lines = [l.strip() for l in cmd.splitlines() if l.strip()]
    if not lines:
        return False
    first = lines[0]
    if first.startswith(("import ", "from ", '"""', "'''", "# -*-")):
        return True
    shell_indicators = (
        "pytest", "python ", "python3 ", "bash ", "cd ", "pip ", "sh ",
    )
    if any(first.startswith(s) for s in shell_indicators):
        return False
    return any(l.startswith(("import ", "from ")) for l in lines)


def _materialise_command(cmd: str, issue: int | None) -> tuple[str, Path | None]:
    """Turn a multi-line / inline-python reproducer into a single shell command.

    Returns (shell_command, temp_path_to_cleanup).  ``temp_path`` is non-None
    when we wrote a Python script to disk and the caller is responsible for
    removing it.
    """
    import tempfile

    if _is_python_script(cmd):
        # Use a process-wide unique path so concurrent invocations don't clobber.
        fd, path_str = tempfile.mkstemp(
            prefix=f"repro_{issue or 'na'}_", suffix=".py")
        with os.fdopen(fd, "w") as fh:
            fh.write(cmd)
        path = Path(path_str)
        log("INFO", f"Wrote Python reproducer to {path}", issue=issue)
        return f"python {path}", path

    if "\n" in cmd:
        joined = " && ".join(
            line.strip() for line in cmd.splitlines() if line.strip())
        return joined, None

    return cmd, None


def _interpret_result(result: subprocess.CompletedProcess[str], output: str,
                      issue: int | None) -> tuple[bool, str]:
    """Apply strict pass/fail heuristics to a finished pytest/unittest run.

    Returns (passed, output).  "Passed" means *at least one test ran and all
    that ran genuinely passed* — xfail / all-skipped / 0-collected are
    treated as "not fixed" so we don't falsely close issues.
    """
    rc = result.returncode

    # rc=4: file/directory not found; rc=5: no tests collected.
    if rc in (4, 5):
        if re.search(
            r"ImportError|ModuleNotFoundError|AssertionError.*not compiled",
            output,
        ):
            log("INFO", "Tests failed to import (env issue, not fixed)",
                issue=issue)
            return False, output
        if re.search(r"no tests ran|collected 0 items", output):
            log("INFO", "No tests collected — cannot verify", issue=issue)
            return False, output

    if rc != 0:
        return False, output

    # rc == 0 path — be paranoid about pseudo-passes.
    if re.search(r"\d+ xfailed", output):
        log("INFO", "Tests xfailed (expected failure) — bug still exists",
            issue=issue)
        return False, output
    if re.search(r"\d+ skipped", output) and not re.search(r"\d+ passed", output):
        log("INFO", "All tests skipped — cannot confirm fix", issue=issue)
        return False, output
    if (re.search(r"OK \(skipped=\d+\)", output)
            and not re.search(r"Ran [1-9]\d* test", output)):
        log("INFO", "All tests skipped (unittest) — cannot confirm fix",
            issue=issue)
        return False, output
    ran_match = re.search(r"Ran (\d+) test", output)
    skip_match = re.search(r"skipped=(\d+)", output)
    if ran_match and skip_match and ran_match.group(1) == skip_match.group(1):
        log("INFO", "All tests skipped (unittest) — cannot confirm fix",
            issue=issue)
        return False, output
    return True, output


def run_test(workdir: Path, test_cmd: str, *, issue: int | None = None,
             timeout: int = DEFAULT_TEST_TIMEOUT) -> tuple[bool, str]:
    """Run a reproducer command under the XPU environment.

    A single concrete pass/fail definition (see ``_interpret_result``) is
    used everywhere so the discovery, fix and post-PR stages can't disagree
    about what "the test passed" means.

    Returns (passed, output_tail). ``output_tail`` is the last ~5 KB of
    combined stdout+stderr (also populated on timeout when the OS gave us
    partial output).

    .. warning::
       ``test_cmd`` is taken verbatim from the (formatted) issue body and
       executed via ``bash -c``.  That is the whole point of the function —
       we have to be able to run whatever reproducer the bug-reporter
       supplied — so there is *no* injection mitigation here.  The trust
       boundary is "anyone who can write into the issue body can execute
       arbitrary commands on the build host".  Callers must keep this in
       mind when widening who is allowed to file/edit issues.
    """
    log("INFO", f"Running test: {test_cmd[:200]}", issue=issue)
    shell_cmd, tmp_path = _materialise_command(test_cmd, issue)
    # Pre-validate Python reproducers: a SyntaxError here means the
    # extraction pipeline mangled the script (or the issue body is broken),
    # NOT that the fix failed. Surface it clearly so the caller can route
    # to agent:needs-human instead of looping on rebuild+retry.
    if tmp_path is not None and tmp_path.suffix == ".py":
        try:
            import ast
            ast.parse(tmp_path.read_text())
        except SyntaxError as exc:
            msg = (
                f"EXTRACTION_BUG: reproducer failed to parse as Python "
                f"({exc.__class__.__name__}: {exc.msg} at line {exc.lineno}). "
                f"The issue body's code fence is malformed or extract_test_command "
                f"mangled it. Do NOT treat this as a fix failure."
            )
            log("ERROR", msg, issue=issue)
            if tmp_path is not None:
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            return False, msg
    full_cmd = ENV_SETUP + shell_cmd
    try:
        result = subprocess.run(
            full_cmd, cwd=str(workdir),
            capture_output=True, text=True,
            timeout=timeout, shell=True, executable="/bin/bash",
        )
        output = ((result.stdout or "") + (result.stderr or ""))[-5000:]
        return _interpret_result(result, output, issue)
    except subprocess.TimeoutExpired as e:
        # subprocess gives us whatever the child printed before the timeout —
        # surface it so the issue thread has actionable context.
        partial = ""
        for stream in (e.stdout, e.stderr):
            if not stream:
                continue
            partial += stream.decode("utf-8", errors="replace") if isinstance(
                stream, (bytes, bytearray)) else stream
        partial = partial[-4000:]
        msg = f"Test timed out after {timeout}s"
        if partial:
            msg = f"{msg}\n--- partial output ---\n{partial}"
        log("WARN", msg, issue=issue)
        return False, msg
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except OSError:
                pass
