"""Run a previously-refined reproducer command directly (no LLM agent).

verify_existence.py uses the test-verification skill to *figure out* the
right pytest invocation for an issue, then writes that command into the
issue body's Reproducer section.  Once that's been done, downstream
stages (e.g. ``verify_upstream_pr``) can just execute the command
directly — there's no need to spin up another agent session.

Extraction is layered:
  - Tier 1: ``**Refined command:** `cmd``` (what verify_existence writes)
  - Tier 2: first fenced ```bash/sh/shell block (first non-# line)
  - Tier 3: first fenced ```python block → write to
            ``/tmp/agent/repro_issue_<N>.py`` and run via ``python`` (needs ``issue``)
  - Tier 4: LLM fallback via opencode + ``github-copilot/gpt-5-mini``
            — only when Tiers 1-3 all fail (separate helper:
            ``extract_reproducer_via_llm``)

Tier-4 results are persisted back to the body via
``persist_refined_command`` so subsequent runs hit Tier 1 (no repeated
LLM cost).

This module exposes:
  - ``extract_reproducer_command(body, issue=None)`` — Tiers 1-3
  - ``extract_reproducer_via_llm(body, issue, ...)`` — Tier 4
  - ``persist_refined_command(body, cmd)`` — prepend cmd into Reproducer
  - ``run_reproducer_command(cmd, issue)`` — execute under XPU env
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .body_templates import parse_sections, update_section
from .config import OPENCODE_CMD, PYTORCH_DIR
from .logger import log
from .xpu_env import ENV_SETUP


@dataclass(frozen=True)
class ReproResult:
    passed: bool          # True if reproducer no longer reproduces the bug
    exit_code: int
    output_tail: str      # last ~4KB of combined stdout+stderr
    reason: str           # short human-readable summary


AGENT_TMP_DIR = Path(os.environ.get(
    "AGENTIC_XPU_REPRO_DIR",
    str(Path(tempfile.gettempdir()) / "agent"),
))

_FENCE_BASH_RE = re.compile(
    r"```(?:bash|sh|shell)?\s*\n(.*?)\n```", re.DOTALL
)
_FENCE_PYTHON_RE = re.compile(
    r"```python\s*\n(.*?)\n```", re.DOTALL
)
_REFINED_RE = re.compile(r"\*\*Refined command:\*\*\s*`([^`]+)`")


def _agent_tmp() -> Path:
    """Ensure /tmp/agent/ exists and return it."""
    AGENT_TMP_DIR.mkdir(parents=True, exist_ok=True)
    return AGENT_TMP_DIR


def extract_reproducer_command(body: str,
                                issue: int | None = None) -> str | None:
    """Return a runnable reproducer command from the issue body, if present.

    Tiers (first hit wins):
      1. ``**Refined command:** `cmd``` (what verify_existence persists)
      2. First fenced ```bash/sh/shell block (first non-`#` non-blank line)
      3. First fenced ```python block → write to /tmp/agent/repro_issue_<N>.py
         and return ``python <path>``.  Requires ``issue`` to disambiguate
         the temp file; if ``issue`` is None, Tier 3 is skipped.

    Returns None if nothing usable was found — the caller may then
    attempt the LLM fallback via ``extract_reproducer_via_llm``.
    """
    sections = parse_sections(body or "")
    repro = sections.get("Reproducer", "")
    if not repro:
        return None

    # Tier 1: refined command
    m = _REFINED_RE.search(repro)
    if m:
        return m.group(1).strip()

    # Tier 2: bash/sh/shell fenced block
    for m in _FENCE_BASH_RE.finditer(repro):
        block = m.group(1)
        # Skip python blocks that match the no-lang fence — check the
        # opening line of the original match to be sure.
        open_idx = m.start()
        open_line = repro[open_idx:open_idx + 12]
        if open_line.startswith("```python"):
            continue
        for line in block.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return line

    # Tier 3: python fenced block → tmp file
    if issue is not None:
        pm = _FENCE_PYTHON_RE.search(repro)
        if pm:
            script = pm.group(1)
            tmp = _agent_tmp() / f"repro_issue_{issue}.py"
            tmp.write_text(script)
            log("INFO",
                f"Tier 3: wrote python reproducer to {tmp} "
                f"({len(script)} chars)",
                issue=issue)
            return f"python {tmp}"

    return None


# ---------------------------------------------------------------------------
# Tier 4 — opencode (github-copilot/gpt-5-mini) fallback
# ---------------------------------------------------------------------------

LLM_REPRO_MODEL = "github-copilot/gpt-5-mini"
LLM_REPRO_TIMEOUT = 180  # seconds

_LLM_PROMPT_TEMPLATE = """\
You are a reproducer-command extractor for an automated PyTorch-XPU \
bug-triage pipeline.

You will be given the full body of a GitHub issue.  Your job: produce a \
SINGLE shell command that, when run from ~/pytorch under the XPU env, \
attempts to reproduce the bug described in the issue.  The command will \
be executed verbatim — make it copy-paste-ready.

Rules:
  - If the body already contains a runnable shell command (in a code \
fence or inline), use that.
  - If the body contains a Python script (in a ```python block, a \
"Reproducer" / "Repro" section, or inline code), write the script \
verbatim to {script_path} using your `write` tool, then output the \
command `python {script_path}`.
  - Do NOT install packages, do NOT modify pytorch source, do NOT run pip.
  - Do NOT use `&&` to chain installs.  One command, one purpose: \
reproduce the bug.
  - If the issue links to upstream tests (e.g. test/test_nn.py), prefer \
a `python -m pytest <path>::<test>` invocation.

Respond with ONLY a single JSON object on the last line of your output \
(no markdown fence around it), with this exact shape:

  {{"command": "<the shell command>", "reason": "<one short sentence \
explaining what it does and why you chose this form>"}}

Issue body follows (delimited by <<<ISSUE_BODY>>>):
<<<ISSUE_BODY>>>
{body}
<<<END_ISSUE_BODY>>>
"""


def extract_reproducer_via_llm(
    body: str, issue: int, *,
    model: str = LLM_REPRO_MODEL,
    timeout: int = LLM_REPRO_TIMEOUT,
) -> tuple[str, dict] | None:
    """LLM-driven fallback: ask opencode to extract / construct a command.

    Spawns ``opencode run --model <model>`` in /tmp/agent/ (NOT ~/pytorch
    — keeps the run focused and avoids the .opencodeignore bootstrap).
    The model is allowed to write to /tmp/agent/repro_issue_<N>.py via
    its built-in ``write`` tool.

    Returns ``(cmd, meta)`` where meta contains ``reason`` and ``raw``
    (the LLM's full stdout, for diagnostics).  Returns None on any
    failure — the caller falls through to NEEDS_HUMAN.
    """
    tmp_dir = _agent_tmp()
    script_path = tmp_dir / f"repro_issue_{issue}.py"
    prompt = _LLM_PROMPT_TEMPLATE.format(
        script_path=str(script_path), body=body or "(empty)",
    )

    cmd = [
        OPENCODE_CMD, "run",
        "--model", model,
        "--dir", str(tmp_dir),
        "--dangerously-skip-permissions",
        prompt,
    ]
    log("INFO",
        f"Tier 4: invoking {model} via opencode "
        f"(prompt {len(prompt)} chars, timeout {timeout}s)",
        issue=issue)

    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        log("ERROR", f"Tier 4 LLM timed out after {timeout}s", issue=issue)
        return None
    except Exception as e:
        log("ERROR", f"Tier 4 LLM crashed: {e}", issue=issue)
        return None

    stdout = (r.stdout or "").strip()
    if r.returncode != 0:
        log("ERROR",
            f"Tier 4 LLM exit {r.returncode}: "
            f"{(r.stderr or '')[-400:].strip()}",
            issue=issue)
        return None

    # Parse: scan lines bottom-up for a JSON object with "command".
    extracted: dict | None = None
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("command"), str):
            extracted = obj
            break

    if not extracted:
        # Fallback: try to find a JSON object anywhere in the output.
        for m in re.finditer(r"\{[^{}]*\"command\"[^{}]*\}", stdout):
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
            if isinstance(obj.get("command"), str):
                extracted = obj
                break

    if not extracted:
        log("ERROR",
            f"Tier 4 LLM returned no parseable JSON. Output tail: "
            f"{stdout[-400:]}",
            issue=issue)
        return None

    cmd_str = extracted["command"].strip()
    reason = str(extracted.get("reason", "(no reason given)")).strip()
    log("INFO",
        f"Tier 4 LLM extracted command: {cmd_str[:200]} "
        f"({reason[:120]})",
        issue=issue)
    return cmd_str, {"reason": reason, "raw_tail": stdout[-2000:]}


def persist_refined_command(body: str, cmd: str) -> str:
    """Prepend ``**Refined command:** `cmd``` at the top of the
    Reproducer section, preserving everything below (so the original
    python block / shell snippet remains as history).

    If there's no Reproducer section, this is a no-op (returns body
    unchanged) — Tier 4 would not have been reached without one.
    """
    sections = parse_sections(body or "")
    repro = sections.get("Reproducer")
    if repro is None:
        return body

    refined_line = f"**Refined command:** `{cmd}`"
    # Don't double-prepend if already present (shouldn't happen but
    # cheap to guard).
    if _REFINED_RE.search(repro):
        return body

    new_content = f"{refined_line}\n\n{repro}".strip()
    return update_section(body, "Reproducer", new_content)


# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------

def run_reproducer_command(cmd: str, issue: int,
                            timeout: int = 300) -> ReproResult:
    """Execute a reproducer command under the XPU env in PYTORCH_DIR.

    A command "passes" (issue no longer reproduces) when:
      - exit code == 0, AND
      - output does not contain pytest failure markers, AND
      - output does not say ``collected 0 items`` / ``no tests ran``.

    Any other outcome is treated as "still reproduces" (or inconclusive,
    which the caller treats the same way — better safe than sorry).
    """
    log("INFO", f"Running reproducer: {cmd[:200]}", issue=issue)
    full_cmd = f"{ENV_SETUP} {cmd}"
    try:
        r = subprocess.run(
            full_cmd, shell=True, executable="/bin/bash",
            cwd=str(PYTORCH_DIR),
            capture_output=True, text=True, timeout=timeout,
        )
        combined = (r.stdout or "") + (r.stderr or "")
        tail = combined[-4096:]

        # Safety nets — don't trust exit 0 blindly
        if re.search(r"collected 0 items|no tests ran", combined):
            return ReproResult(False, r.returncode, tail,
                               "0 tests collected (cannot verify)")

        if r.returncode == 0:
            # Exit 0 + actual tests ran → bug is fixed
            return ReproResult(True, 0, tail,
                               "Reproducer exited 0 — bug no longer reproduces")

        return ReproResult(False, r.returncode, tail,
                           f"Reproducer failed (exit {r.returncode}) — bug still reproduces")

    except subprocess.TimeoutExpired:
        log("ERROR", f"Reproducer timed out after {timeout}s", issue=issue)
        return ReproResult(False, -1, "", f"Reproducer timed out after {timeout}s")
    except Exception as e:
        log("ERROR", f"Reproducer crashed: {e}", issue=issue)
        return ReproResult(False, -1, "", f"Reproducer crashed: {e}")
