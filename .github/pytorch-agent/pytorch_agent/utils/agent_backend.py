"""Agent backend abstraction — pluggable LLM agent dispatch.

OpenCode CLI notes (from `opencode run --help`):
  - Message is positional (not --prompt)
  - Use --dir for working directory (not cwd kwarg)
  - Use --dangerously-skip-permissions for autonomous operation
"""
from abc import ABC, abstractmethod
from datetime import datetime
import json
from pathlib import Path
import subprocess

from .config import OPENCODE_CMD, PYTORCH_DIR, SKILLS_DIR, LOG_DIR
from .logger import log as pipeline_log


def parse_opencode_events(raw_output: str) -> str:
    """Parse OpenCode JSON event stream and return concatenated text output."""
    text_parts = []
    for line in raw_output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "text":
            part = event.get("part", {})
            text_parts.append(part.get("text", "") or event.get("content", ""))
    return "".join(text_parts)


class AgentBackend(ABC):
    @abstractmethod
    def run(self, prompt: str, workdir: str | None = None,
            skill: str | None = None, timeout: int | None = None,
            issue: int | None = None, stage: str | None = None,
            on_session_start: 'Callable[[str], None] | None' = None) -> tuple[str, Path, str | None]:
        """Run LLM agent with prompt.

        Returns (agent_output_text, log_file_path, session_id_or_None).
        on_session_start: optional callback(session_id) called as soon as
            the agent session ID is known (while still running).
        """
        ...


class OpenCodeBackend(AgentBackend):
    def run(self, prompt: str, workdir: str | None = None,
            skill: str | None = None, timeout: int | None = None,
            issue: int | None = None, stage: str | None = None,
            on_session_start: 'Callable[[str], None] | None' = None) -> tuple[str, Path, str | None]:
        workdir = workdir or str(PYTORCH_DIR)
        timeout = timeout or 1800

        # Point OpenCode to XPU agent skills in torch-xpu-ops.
        # OpenCode runs in ~/pytorch, so it won't auto-discover them.
        # We inline a short pointer — OpenCode reads the files itself.
        skills_hint = (
            "\n\n## Context\n"
            f"XPU agent skills and instructions are in {SKILLS_DIR} "
            f"and {SKILLS_DIR.parent / 'instructions'}. "
            "Read the relevant SKILL.md before starting work."
        )
        if skill:
            skill_path = SKILLS_DIR / skill / "SKILL.md"
            if skill_path.exists():
                skills_hint += f"\nThe skill for this task is: {skill_path}"

        full_prompt = prompt + skills_hint

        # OpenCode CLI: --format json streams structured events to stdout
        cmd = [OPENCODE_CMD, "run", "--format", "json", "--dir", workdir,
               "--dangerously-skip-permissions", full_prompt]

        # Log the command (redact the full prompt, just show length)
        pipeline_log("INFO", f"Running: {' '.join(cmd[:6])} <prompt {len(full_prompt)} chars> "
                     f"(timeout={timeout}s)", issue=issue)

        # Stream output to log file in real-time (not buffered until exit)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        prefix = f"issue-{issue}" if issue else "unknown"
        stage_str = f"-{stage.lower()}" if stage else ""
        log_path = LOG_DIR / f"agent-{prefix}{stage_str}-{ts}.log"

        with open(log_path, "w") as log_f:
            log_f.write(f"=== COMMAND ===\n{' '.join(cmd[:6])} <prompt>\n\n")
            log_f.write(f"=== OUTPUT (real-time) ===\n")
            log_f.flush()

            # stdout = JSON events; discard stderr to avoid pipe buffer deadlock
            # Log full prompt to file for debugging
            prompt_log = log_path.with_suffix('.prompt.txt')
            with open(prompt_log, 'w') as pf:
                pf.write(full_prompt)
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL, text=True,
            )
            log_f.write(f"=== PID ===\n{proc.pid}\n\n")

            session_id = None
            text_parts = []
            try:
                for line in proc.stdout:
                    log_f.write(line)
                    log_f.flush()
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    # Extract session ID from first event
                    if session_id is None and event.get("sessionID"):
                        session_id = event["sessionID"]
                        pipeline_log("INFO",
                                     f"OpenCode session: {session_id}  "
                                     f"(attach: cd {workdir} && opencode -s {session_id})",
                                     issue=issue)
                        log_f.write(f"\n=== SESSION ID ===\n{session_id}\n\n")
                        log_f.flush()
                        if on_session_start:
                            try:
                                on_session_start(session_id)
                            except Exception:
                                pass
                    # Collect text output from assistant
                    if event.get("type") == "text":
                        part = event.get("part", {})
                        text_parts.append(part.get("text", ""))
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                log_f.write("\n=== TIMEOUT ===\n")
                raise

            log_f.write(f"\n=== EXIT CODE ===\n{proc.returncode}\n")

        full_output = "".join(text_parts)

        pipeline_log("INFO", f"Agent finished (rc={proc.returncode}), "
                     f"log: {log_path}", issue=issue)

        if proc.returncode != 0:
            raise RuntimeError(
                f"OpenCode failed (rc={proc.returncode}), "
                f"log: {log_path}\n{full_output[-500:]}"
            )
        return full_output, log_path, session_id


class CopilotBackend(AgentBackend):
    """Future: GitHub Copilot cloud agent.
    Placeholder — implement when migrating to cloud.
    """
    def run(self, prompt: str, workdir: str | None = None,
            skill: str | None = None, timeout: int | None = None,
            issue: int | None = None, stage: str | None = None,
            on_session_start: 'Callable[[str], None] | None' = None) -> tuple[str, Path, str | None]:
        raise NotImplementedError("Copilot backend not yet implemented")


def get_backend() -> AgentBackend:
    from .config import AGENT_BACKEND
    if AGENT_BACKEND == "copilot":
        return CopilotBackend()
    return OpenCodeBackend()
