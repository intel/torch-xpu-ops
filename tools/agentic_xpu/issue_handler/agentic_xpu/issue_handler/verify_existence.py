# Copyright 2024-2026 Intel Corporation
# Co-authored with GitHub Copilot
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

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
from json import JSONDecoder

from .utils import git as gh
from .utils.build import incremental_build
from .utils.config import ISSUE_REPO, PYTORCH_DIR, UPSTREAM_REMOTE
from .utils.xpu_env import ensure_xpu_ready, sync_pytorch  # noqa: F401  (legacy)
from .utils.body_templates import (
    get_status, set_status, append_log, update_section,
)
from .utils.agent_backend import get_backend
from .utils.locks import pytorch_lock
from .utils.logger import log
from .utils.stages import Skill, Stage
from .utils.verification import extract_test_command


# ---------------------------------------------------------------------------
# Raw test reference extraction
# ---------------------------------------------------------------------------

def _get_raw_test_reference(body: str) -> str | None:
    """Verbatim test reference for the test-verification LLM agent.

    The agent resolves paths itself, so we hand it the body content as-is
    (no ``cd <pytorch>&&`` stripping, no pytest synthesis).
    """
    return extract_test_command(body, executable=False)


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
        f"## PyTorch Directory\n`{PYTORCH_DIR}`\n\n"
        f"## Issue Body\n{body[:3000]}\n\n"
        f"Follow the test-verification skill instructions. "
        f"Resolve the path, run the test, and output the JSON result."
    )


def _parse_agent_result(output: str) -> dict | None:
    """Parse the agent's JSON output.

    Strategy: prefer the last ```json fenced block; otherwise scan the
    text for a top-level JSON object containing a "status" key using
    ``json.JSONDecoder().raw_decode`` (handles nested braces correctly,
    unlike the previous regex which silently failed on nested objects).
    """
    blocks = re.findall(r'```json\s*\n(.*?)```', output, re.DOTALL)
    for raw in reversed(blocks):
        try:
            result = json.loads(raw.strip())
            if isinstance(result, dict) and "status" in result:
                return result
        except json.JSONDecodeError:
            continue

    # Scan for bare JSON objects.
    decoder = JSONDecoder()
    last_valid: dict | None = None
    for idx in range(len(output)):
        if output[idx] != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(output, idx)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "status" in obj:
            last_valid = obj  # keep scanning for a later, possibly better match

    return last_valid


def _update_reproducer_section(body: str, refined_cmd: str,
                               original_cmd: str) -> str:
    """Update the Reproducer section with refined command and original in details."""
    new_reproducer = (
        f"```bash\n{refined_cmd}\n```\n\n"
        f"<details><summary>Original command</summary>\n\n"
        f"```\n{original_cmd}\n```\n\n"
        f"</details>"
    )

    new_body = update_section(body, "Reproducer", new_reproducer)
    if new_body == body:
        log("WARN", "Could not find Reproducer section to update")
    return new_body


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _prepare_main_branch(issue: int) -> bool:
    """Fast-forward PYTORCH_DIR to ``{UPSTREAM_REMOTE}/main`` and run an
    incremental rebuild only for the diff against the previously-built
    ref (Part-1b of the upstream-PR-verify prebuild plan).

    Returns True on success, False on any fetch/checkout/build failure
    (the caller logs the higher-level "skipping verification" message).
    """
    main_ref = f"{UPSTREAM_REMOTE}/main"
    try:
        # Fetch latest main (best-effort: a transient network blip
        # should not stop us from testing against the already-checked-out
        # tree, but a hard failure to fetch is still surfaced as False).
        rv = gh.git(
            "fetch", UPSTREAM_REMOTE, "main",
            workdir=PYTORCH_DIR, check=False, issue=issue,
        )
        if rv.returncode != 0:
            log("ERROR", f"git fetch {UPSTREAM_REMOTE} main failed (rc={rv.returncode})",
                issue=issue)
            return False

        # Hard-checkout main; any local detached/agentic branch state is
        # discarded — pytorch_lock guarantees no concurrent agent owns it.
        rv = gh.git(
            "checkout", "-f", "-B", "main", main_ref,
            workdir=PYTORCH_DIR, check=False, issue=issue,
        )
        if rv.returncode != 0:
            log("ERROR", f"git checkout -B main {main_ref} failed (rc={rv.returncode})",
                issue=issue)
            return False

        # Best-effort submodule sync — non-fatal, mirrors verify_upstream_pr.
        try:
            gh.git(
                "submodule", "update", "--init", "--recursive",
                workdir=PYTORCH_DIR, check=False, issue=issue,
            )
        except Exception as sm_exc:
            log("WARN", f"submodule update failed (non-fatal): {sm_exc!r}",
                issue=issue)

        ok, msg = incremental_build(
            workdir=PYTORCH_DIR, base_ref=main_ref, issue=issue,
        )
        if not ok:
            log("ERROR", f"incremental_build failed: {msg}", issue=issue)
        return ok
    except Exception as exc:
        log("ERROR", f"_prepare_main_branch unexpected error: {exc!r}",
            issue=issue)
        return False


def run(issue_number: int) -> bool:
    """Verify whether issue still reproduces.

    Returns True if issue is already fixed (pipeline should stop).
    Returns False if issue still exists (pipeline should continue to triage).
    """
    detail = gh.get_issue_detail(ISSUE_REPO, issue_number)
    body = detail.get("body", "") or ""

    status = get_status(body)
    if status != Stage.DISCOVERED:
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

    # Pytorch repo state and the build artefacts under it are shared between
    # all agents; serialize against concurrent issue runs.
    with pytorch_lock(issue=issue_number):
        # Ensure XPU environment is ready before running tests
        if not ensure_xpu_ready(issue=issue_number):
            log("ERROR", f"XPU environment not available for #{issue_number}, skipping verification",
                issue=issue_number)
            return False

        # Update pytorch to current upstream/main and incrementally
        # rebuild only the diff against the previously-built ref (per
        # plan Part-1b). Replaces the legacy sync_pytorch() call which
        # would full-rebuild whenever main advanced past the installed
        # binary.
        if not _prepare_main_branch(issue=issue_number):
            log("ERROR", f"Failed to prepare pytorch main for #{issue_number}, skipping verification",
                issue=issue_number)
            return False

        # Call agent with test-verification skill
        prompt = _build_verification_prompt(issue_number, body, raw_ref)
        backend = get_backend()

        try:
            agent_output, log_path, session_id, token_usage = backend.run(
                prompt, skill=Skill.TEST_VERIFICATION, timeout=300,
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
        new_body = set_status(new_body, Stage.DONE)
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
        gh.sync_labels(ISSUE_REPO, issue_number, Stage.DONE)
        # Additional signal: distinguish auto-verified-fixed (bug no longer
        # reproduces locally) from generic pipeline completion. `agent:close`
        # is intentionally additive — sync_labels handles the agent:* status,
        # this tracks "ready to close as resolved".
        gh.add_label(ISSUE_REPO, issue_number, "agent:close")
        return True
    else:
        # Still failing or cannot verify — log result into body for triage
        new_body = append_log(
            new_body, "verify",
            f"**Local verification:** Bug still reproduces.\n"
            f"Test command: `{refined_cmd or raw_ref}`\n"
            f"Result: ❌ {agent_status} — {reason}\n"
            f"Nightly tested: current local build.",
        )
        gh.update_issue_body(ISSUE_REPO, issue_number, new_body)
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
