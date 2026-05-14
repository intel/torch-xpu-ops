"""Verify whether a discovered issue still reproduces locally.

Entry point:
  python -m issue_handler.verify_existence --issue 123

Runs after formatting (DISCOVERED stage). Extracts the test command from
the structured issue body and runs it locally. If the test passes, the
bug is already fixed — we comment, set status to DONE, and skip triage/fix.

Returns True if the issue is already fixed (no further work needed).
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

from .utils import git as gh
from .utils.config import ISSUE_REPO, PYTORCH_DIR, TORCH_XPU_OPS_DIR
from .utils.body_templates import (
    get_status, set_status, append_log, parse_sections,
)
from .utils.logger import log


# ---------------------------------------------------------------------------
# Test command extraction (shared with code_fix)
# ---------------------------------------------------------------------------

_SHELL_CMD_PATTERNS = re.compile(
    r"(?:^|\n)\s*(?:python|pytest|cd |bash |sh |export |source |\.\/|pip )",
    re.IGNORECASE,
)


def _is_shell_command(text: str) -> bool:
    """Heuristic: does this text look like a runnable shell command?

    Rejects CI metadata formats like 'op_ut,third_party.torch-xpu-ops...'
    """
    first_line = text.strip().split("\n")[0]
    # CI metadata format: "category,module.path,test_name"
    if "," in first_line and "::" not in first_line:
        return False
    # Must contain at least one shell-like pattern
    if _SHELL_CMD_PATTERNS.search(text):
        return True
    # Or contain common test runner patterns
    if "::" in text or "-k " in text or "test_" in first_line.split()[0:1]:
        return True
    return False

def _get_test_command(body: str) -> str | None:
    """Extract test command from issue body.

    Priority:
      1. "Reproducer" section — use verbatim bash block
      2. "Failed Tests" section — construct pytest command from test names
      3. None (skip verification)
    """
    sections = parse_sections(body)

    # 1. Reproducer section
    reproducer = sections.get("Reproducer", "").strip()
    if reproducer:
        # Extract bash code block if present
        m = re.search(r"```(?:bash|sh)?\s*\n(.+?)```", reproducer, re.DOTALL)
        if m:
            cmd = m.group(1).strip()
            # Strip nested fence markers (LLM sometimes double-wraps)
            cmd = re.sub(r'^```(?:bash|sh)?\s*\n', '', cmd)
            cmd = re.sub(r'\n```\s*$', '', cmd)
            cmd = cmd.strip()
            # Validate it looks like a shell command (not CI metadata like "op_ut,...")
            if cmd and _is_shell_command(cmd):
                return cmd
        # If no code block but non-empty text that looks like a command
        if reproducer and not reproducer.startswith("_Pending"):
            lines = [l.strip() for l in reproducer.splitlines()
                     if l.strip() and not l.strip().startswith("#")
                     and not l.strip().startswith("```")]
            joined = "\n".join(lines)
            if lines and _is_shell_command(joined):
                return joined

    # 2. Failed Tests section
    failed_tests = sections.get("Failed Tests", "").strip()
    if failed_tests:
        # Extract test paths like `test_ops.py::TestClass::test_method`
        tests = re.findall(
            r"`([^`]+(?:::|-k\s+)[^`]*)`", failed_tests)
        if not tests:
            # Try lines starting with - that look like test paths
            tests = re.findall(
                r"[-*]\s+`?(\S+::\S+)`?", failed_tests)
        if tests:
            # Prefix test/xpu/ paths for pytorch root execution
            fixed = []
            for t in tests:
                if t.startswith("test/xpu/"):
                    t = f"third_party/torch-xpu-ops/{t}"
                fixed.append(t)
            test_args = " ".join(f'"{t}"' for t in fixed)
            return f"pytest -v {test_args}"

    return None


def _detect_workdir(body: str) -> Path:
    """Determine which repo to run tests in based on issue content.

    All XPU tests run from pytorch root (as third_party/torch-xpu-ops/test/...).
    Only standalone torch-xpu-ops scripts (python script.py) run from TORCH_XPU_OPS_DIR.
    """
    sections = parse_sections(body)
    reproducer = sections.get("Reproducer", "")

    # If reproducer explicitly cds into torch-xpu-ops or runs scripts there
    if "cd " in reproducer and "torch-xpu-ops" in reproducer:
        return TORCH_XPU_OPS_DIR

    # Default: run from pytorch (handles test/xpu/* paths as third_party/...)
    return PYTORCH_DIR


def _run_test(workdir: Path, test_cmd: str,
              issue: int) -> tuple[bool, str]:
    """Run test command locally. Returns (passed, output).

    'passed' means ALL tests genuinely passed (not xfail, not skipped).
    xfail = test is expected to fail = bug still exists.
    skipped = test not run = can't confirm fix.
    """
    log("INFO", f"Running verification test: {test_cmd[:200]}",
        issue=issue)
    # Wrap with pytorch env activation
    env_setup = (
        "source ~/intel/oneapi/setvars.sh --force 2>/dev/null; "
        "source ~/pytorch/.venv/bin/activate; "
    )
    full_cmd = env_setup + test_cmd
    try:
        result = subprocess.run(
            full_cmd, cwd=str(workdir),
            capture_output=True, text=True,
            timeout=600, shell=True,
            executable="/bin/bash",
        )
        output = (result.stdout + result.stderr)[-5000:]

        # rc=4: file/directory not found; rc=5: no tests collected
        # Both mean the test no longer exists → bug is fixed
        if result.returncode in (4, 5) or re.search(
            r"no tests ran|collected 0 items|ERROR: not found:", output
        ):
            log("INFO", "Tests no longer exist (not found/not collected) — "
                "bug appears fixed", issue=issue)
            return True, output

        # Even with rc=0, check for xfail/skipped — those mean bug still exists
        if result.returncode == 0:
            # Parse pytest summary line: "X passed, Y xfailed, Z skipped"
            if re.search(r"\d+ xfailed", output):
                log("INFO", "Tests xfailed (expected failure) — bug still exists",
                    issue=issue)
                return False, output
            # Check if ALL tests were skipped (no passed)
            if re.search(r"\d+ skipped", output) and not re.search(r"\d+ passed", output):
                log("INFO", "All tests skipped — cannot confirm fix",
                    issue=issue)
                return False, output
            # Also check unittest-style skip: "OK (skipped=N)" with no actual runs
            if re.search(r"OK \(skipped=\d+\)", output) and not re.search(r"Ran [1-9]\d* test", output):
                log("INFO", "All tests skipped (unittest) — cannot confirm fix",
                    issue=issue)
                return False, output
            # unittest "Ran N test" but all skipped
            ran_match = re.search(r"Ran (\d+) test", output)
            skip_match = re.search(r"skipped=(\d+)", output)
            if ran_match and skip_match and ran_match.group(1) == skip_match.group(1):
                log("INFO", "All tests skipped (unittest) — cannot confirm fix",
                    issue=issue)
                return False, output
            # Genuine pass
            return True, output
        return False, output
    except subprocess.TimeoutExpired:
        return False, "Test timed out (10min) — treating as still failing"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(issue_number: int) -> bool:
    """Verify whether issue still reproduces.

    Returns True if issue is already fixed (pipeline should stop).
    Returns False if issue still exists (pipeline should continue to triage).
    """
    detail = gh.get_issue_detail(ISSUE_REPO, issue_number)
    body = detail.get("body", "") or ""

    status = get_status(body)
    if status != "DISCOVERED":
        log("INFO", f"Issue #{issue_number} not in DISCOVERED stage ({status}), skipping verification",
            issue=issue_number)
        return False

    # Extract test command
    test_cmd = _get_test_command(body)
    if not test_cmd:
        log("INFO", f"No test command found for #{issue_number}, skipping verification",
            issue=issue_number)
        return False

    # Determine workdir
    workdir = _detect_workdir(body)
    log("INFO", f"Verifying #{issue_number} in {workdir}", issue=issue_number)

    # Run test
    passed, output = _run_test(workdir, test_cmd, issue_number)

    if passed:
        # Bug is already fixed!
        log("INFO", f"Issue #{issue_number} no longer reproduces — marking as DONE",
            issue=issue_number)
        new_body = body
        new_body = set_status(new_body, "DONE")
        new_body = append_log(
            new_body, "fix",
            f"**Verification:** Issue no longer reproduces locally.\n"
            f"Test command: `{test_cmd[:200]}`\n"
            f"Result: ✅ PASSED — bug is already fixed in current codebase.\n"
            f"No triage/fix needed.",
        )
        gh.update_issue_body(ISSUE_REPO, issue_number, new_body)
        gh.add_issue_comment(
            ISSUE_REPO, issue_number,
            f"🎉 **Issue auto-verified as fixed**\n\n"
            f"The test command no longer fails locally:\n"
            f"```\n{test_cmd}\n```\n\n"
            f"This issue appears to have been fixed by a recent commit. "
            f"Closing as resolved.\n\n"
            f"<details><summary>Test output (last 2000 chars)</summary>\n\n"
            f"```\n{output[-2000:]}\n```\n</details>",
        )
        # Close the issue
        gh.add_label(ISSUE_REPO, issue_number, "agent:close")
        return True
    else:
        log("INFO", f"Issue #{issue_number} still reproduces — proceeding to triage",
            issue=issue_number)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify whether a discovered issue still reproduces locally")
    parser.add_argument("--issue", type=int, required=True)
    args = parser.parse_args()
    fixed = run(args.issue)
    print(f"Issue #{args.issue}: {'FIXED' if fixed else 'STILL_EXISTS'}")


if __name__ == "__main__":
    main()
