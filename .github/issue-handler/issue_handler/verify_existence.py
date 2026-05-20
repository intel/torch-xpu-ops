"""Verify whether a discovered issue still reproduces locally.

Entry point:
  python -m issue_handler.verify_existence --issue 123

Runs after formatting (DISCOVERED stage). Extracts the raw test reference
from the structured issue body, delegates path resolution and test execution
to an LLM agent (OpenCode) with the test-verification skill, then interprets
the result.

Returns True if the issue is already fixed (no further work needed).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from .utils import git as gh
from .utils.config import ISSUE_REPO
from .utils.xpu_env import ensure_xpu_ready
from .utils.body_templates import (
    get_status, set_status, append_log, parse_sections,
)
from .utils.agent_backend import get_backend
from .utils.logger import log


# ---------------------------------------------------------------------------
# Raw test reference extraction (no path fixing — agent does that)
# ---------------------------------------------------------------------------

def _get_raw_test_reference(body: str) -> str | None:
    """Extract raw test reference from issue body.

    Returns the verbatim text from Reproducer or Failed Tests sections.
    No path rewriting — the agent skill handles that.
    """
    sections = parse_sections(body)

    # 1. Reproducer section
    reproducer = sections.get("Reproducer", "").strip()
    if reproducer and not reproducer.startswith("_Pending"):
        # Extract bash code block if present
        m = re.search(r"```(?:bash|sh)?\s*\n(.+?)```", reproducer, re.DOTALL)
        if m:
            cmd = m.group(1).strip()
            # Strip nested fence markers
            cmd = re.sub(r'^```(?:bash|sh)?\s*\n', '', cmd)
            cmd = re.sub(r'\n```\s*$', '', cmd)
            cmd = cmd.strip()
            if cmd:
                return cmd
        # Non-code-block text
        lines = [l.strip() for l in reproducer.splitlines()
                 if l.strip() and not l.strip().startswith("#")
                 and not l.strip().startswith("```")]
        if lines:
            return "\n".join(lines)

    # 2. Failed Tests section
    failed_tests = sections.get("Failed Tests", "").strip()
    if failed_tests:
        # Extract test paths like `test_ops.py::TestClass::test_method`
        tests = re.findall(
            r"`([^`]+(?:::|-k\s+)[^`]*)`", failed_tests)
        if not tests:
            tests = re.findall(
                r"[-*]\s+`?(\S+::\S+)`?", failed_tests)
        if tests:
            return "\n".join(tests)

    return None


# Backward compatibility alias for verify_fix.py
_get_test_command = _get_raw_test_reference


def _run_test(workdir: Path, test_cmd: str,
              issue: int) -> tuple[bool, str]:
    """Run test command locally. Returns (passed, output).

    'passed' means ALL tests genuinely passed (not xfail, not skipped).
    Used by verify_fix.py for post-PR verification.
    """
    from .utils.xpu_env import ENV_SETUP

    log("INFO", f"Running verification test: {test_cmd[:200]}",
        issue=issue)
    full_cmd = ENV_SETUP + test_cmd
    try:
        result = subprocess.run(
            full_cmd, cwd=str(workdir),
            capture_output=True, text=True,
            timeout=600, shell=True,
            executable="/bin/bash",
        )
        output = (result.stdout + result.stderr)[-5000:]

        # rc=4: file/directory not found; rc=5: no tests collected
        if result.returncode in (4, 5):
            has_import_error = re.search(
                r"ImportError|ModuleNotFoundError|AssertionError.*not compiled",
                output)
            if has_import_error:
                log("INFO", "Tests failed to import (env issue, not fixed)",
                    issue=issue)
                return False, output
            if re.search(r"no tests ran|collected 0 items", output):
                # 0 collected = can't verify, NOT fixed
                log("INFO", "No tests collected — cannot verify",
                    issue=issue)
                return False, output

        if result.returncode == 0:
            # Check for xfail — bug still exists
            if re.search(r"\d+ xfailed", output):
                log("INFO", "Tests xfailed (expected failure) — bug still exists",
                    issue=issue)
                return False, output
            # All tests skipped — can't confirm
            if re.search(r"\d+ skipped", output) and not re.search(r"\d+ passed", output):
                log("INFO", "All tests skipped — cannot confirm fix",
                    issue=issue)
                return False, output
            # unittest-style all-skipped
            if re.search(r"OK \(skipped=\d+\)", output) and not re.search(r"Ran [1-9]\d* test", output):
                log("INFO", "All tests skipped (unittest) — cannot confirm fix",
                    issue=issue)
                return False, output
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
# Agent-based verification
# ---------------------------------------------------------------------------

def _build_verification_prompt(issue_number: int, body: str,
                               raw_ref: str) -> str:
    """Build the prompt for the verification agent."""
    title_detail = gh.get_issue_detail(ISSUE_REPO, issue_number)
    title = title_detail.get("title", f"Issue #{issue_number}")

    return (
        f"Verify whether issue #{issue_number} still reproduces.\n\n"
        f"## Issue Title\n{title}\n\n"
        f"## Raw Test Reference\n```\n{raw_ref}\n```\n\n"
        f"## Issue Body\n{body[:3000]}\n\n"
        f"Follow the test-verification skill instructions. "
        f"Resolve the path, run the test, and output the JSON result."
    )


def _parse_agent_result(output: str) -> dict | None:
    """Parse the agent's JSON output.

    Looks for the last JSON block in the output.
    """
    # Find all JSON blocks (```json ... ``` or bare { ... })
    json_blocks = re.findall(r'```json\s*\n(.*?)```', output, re.DOTALL)
    if json_blocks:
        raw = json_blocks[-1].strip()
    else:
        # Try to find bare JSON object with "status" key — allow nested braces
        # by matching from { to the last } in the output
        matches = re.findall(r'\{[^{}]*"status"\s*:.*\}', output, re.DOTALL)
        if matches:
            raw = matches[-1]
        else:
            return None

    try:
        result = json.loads(raw)
        if "status" in result:
            return result
    except json.JSONDecodeError:
        # Try to fix truncated JSON — find the last complete field
        # by adding closing brace
        for suffix in ['"}', '"\n}', '" }']:
            try:
                result = json.loads(raw + suffix)
                if "status" in result:
                    return result
            except json.JSONDecodeError:
                continue
    return None


def _update_reproducer_section(body: str, refined_cmd: str,
                               original_cmd: str) -> str:
    """Update the Reproducer section with refined command and original in details."""
    new_reproducer = (
        f"```bash\n{refined_cmd}\n```\n\n"
        f"<details><summary>Original command</summary>\n\n"
        f"```\n{original_cmd}\n```\n\n"
        f"</details>"
    )

    # Replace existing Reproducer section content
    # Pattern: ### Reproducer\n<content>\n### NextSection
    pattern = r'(### Reproducer\s*\n).*?(?=\n### |\n<!-- |\Z)'
    replacement = rf'\g<1>{new_reproducer}\n'
    new_body, count = re.subn(pattern, replacement, body, count=1, flags=re.DOTALL)

    if count == 0:
        # Section not found — don't modify
        log("WARN", "Could not find Reproducer section to update")
        return body

    return new_body


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

    # Extract raw test reference
    raw_ref = _get_raw_test_reference(body)
    if not raw_ref:
        log("INFO", f"No test reference found for #{issue_number}, skipping verification",
            issue=issue_number)
        return False

    log("INFO", f"Raw test reference for #{issue_number}: {raw_ref[:200]}",
        issue=issue_number)

    # Ensure XPU environment is ready before running tests
    if not ensure_xpu_ready(issue=issue_number):
        log("ERROR", f"XPU environment not available for #{issue_number}, skipping verification",
            issue=issue_number)
        return False

    # Call agent with test-verification skill
    prompt = _build_verification_prompt(issue_number, body, raw_ref)
    backend = get_backend()

    try:
        agent_output, log_path, session_id, token_usage = backend.run(
            prompt, skill="test-verification", timeout=300,
            issue=issue_number, stage="verify",
        )
    except Exception as e:
        log("ERROR", f"Verification agent failed for #{issue_number}: {e}",
            issue=issue_number, exc=e)
        return False

    # Parse agent result
    result = _parse_agent_result(agent_output)
    if not result:
        log("WARN", f"Could not parse agent result for #{issue_number}",
            issue=issue_number)
        return False

    agent_status = result.get("status", "UNKNOWN")
    refined_cmd = result.get("refined_command", "")
    original_cmd = result.get("original_command", raw_ref)
    reason = result.get("reason", "")
    output_tail = result.get("output_tail", "")

    log("INFO", f"Verification result for #{issue_number}: {agent_status} — {reason}",
        issue=issue_number)

    # Safety net: if agent says PASSED but output shows 0 collected, reject
    if agent_status == "PASSED" and output_tail:
        if re.search(r"collected 0 items|no tests ran", output_tail):
            log("WARN", f"Agent said PASSED but output shows 0 tests collected for #{issue_number}",
                issue=issue_number)
            agent_status = "CANNOT_VERIFY"
            reason = "Agent reported PASSED but 0 tests were collected (safety net)"

    # Update Reproducer section with refined command (if we have one)
    if refined_cmd and refined_cmd != raw_ref:
        new_body = _update_reproducer_section(body, refined_cmd, original_cmd)
    else:
        new_body = body

    if agent_status == "PASSED":
        # Bug is already fixed
        log("INFO", f"Issue #{issue_number} no longer reproduces — marking as DONE",
            issue=issue_number)
        new_body = set_status(new_body, "DONE")
        new_body = append_log(
            new_body, "fix",
            f"**Verification:** Issue no longer reproduces locally.\n"
            f"Test command: `{refined_cmd or raw_ref}`\n"
            f"Result: ✅ PASSED — bug is already fixed in current codebase.\n"
            f"No triage/fix needed.",
        )
        gh.update_issue_body(ISSUE_REPO, issue_number, new_body)
        gh.add_issue_comment(
            ISSUE_REPO, issue_number,
            f"🎉 **Issue auto-verified as fixed**\n\n"
            f"The test command no longer fails locally:\n"
            f"```\n{refined_cmd or raw_ref}\n```\n\n"
            f"This issue appears to have been fixed by a recent commit. "
            f"Closing as resolved.\n\n"
            f"<details><summary>Test output (last 2000 chars)</summary>\n\n"
            f"```\n{output_tail[-2000:]}\n```\n</details>",
        )
        gh.add_label(ISSUE_REPO, issue_number, "agent:close")
        return True
    else:
        # Still failing or cannot verify — update body with refined command if available
        if new_body != body:
            gh.update_issue_body(ISSUE_REPO, issue_number, new_body)
            log("INFO", f"Updated Reproducer section for #{issue_number} with resolved command",
                issue=issue_number)
        log("INFO", f"Issue #{issue_number} still reproduces or cannot verify — proceeding to triage",
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
