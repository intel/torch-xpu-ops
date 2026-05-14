"""Agent backend abstraction — pluggable LLM agent dispatch.

OpenCode CLI notes (from `opencode run --help`):
  - Message is positional (not --prompt)
  - Use --dir for working directory (not cwd kwarg)
  - Use --dangerously-skip-permissions for autonomous operation
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
import select
import time
from collections.abc import Callable
from pathlib import Path
import subprocess

from .config import OPENCODE_CMD, PYTORCH_DIR, SKILLS_DIR, LOG_DIR, CONFIG_DIR, AGENT_MODEL
from .logger import log as pipeline_log


@dataclass
class TokenUsage:
    """Accumulated token usage across all steps in one agent run."""
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    total: int = 0
    cost: float = 0.0

    def add_step(self, tokens: dict) -> None:
        """Accumulate from a step_finish event's 'tokens' dict."""
        self.input += tokens.get("input", 0)
        self.output += tokens.get("output", 0)
        cache = tokens.get("cache", {})
        self.cache_read += cache.get("read", 0)
        self.cache_write += cache.get("write", 0)
        self.total += tokens.get("total", 0)

    def add_cost(self, cost: float) -> None:
        self.cost += cost or 0.0

    def estimated_cost(self) -> float:
        """Estimate cost using Claude Sonnet 4 pricing ($/M tokens).
        Input: $3, Output: $15, Cache read: $0.30, Cache write: $3.75.
        """
        return (
            self.input * 3.0 / 1_000_000
            + self.output * 15.0 / 1_000_000
            + self.cache_read * 0.30 / 1_000_000
            + self.cache_write * 3.75 / 1_000_000
        )

    def summary(self) -> str:
        """Human-readable summary for Action Items log."""
        def _fmt(n: int) -> str:
            if n >= 1_000_000:
                return f"{n / 1_000_000:.1f}M"
            if n >= 1_000:
                return f"{n / 1_000:.1f}K"
            return str(n)
        parts = [f"model: {AGENT_MODEL}", f"tokens: {_fmt(self.total)}"]
        parts.append(f"in: {_fmt(self.input)}")
        parts.append(f"out: {_fmt(self.output)}")
        if self.cache_read:
            parts.append(f"cache_read: {_fmt(self.cache_read)}")
        if self.cache_write:
            parts.append(f"cache_write: {_fmt(self.cache_write)}")
        # Use reported cost if available, otherwise estimate
        cost = self.cost if self.cost > 0 else self.estimated_cost()
        parts.append(f"cost: ${cost:.4f}")
        return " | ".join(parts)


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
            on_session_start: Callable[[str], None] | None = None) -> tuple[str, Path, str | None, TokenUsage]:
        """Run LLM agent with prompt.

        Returns (agent_output_text, log_file_path, session_id_or_None, token_usage).
        on_session_start: optional callback(session_id) called as soon as
            the agent session ID is known (while still running).
        """
        ...


OPENCODEIGNORE_TEMPLATE = CONFIG_DIR / "opencodeignore.default"


def _ensure_opencodeignore(workdir: Path) -> None:
    """Copy .opencodeignore into workdir if it doesn't exist yet.

    OpenCode's file watcher hangs on inotify init for large repos
    (200K+ files like pytorch). The ignore file excludes third_party/,
    build artifacts, and .git/ to keep the watcher fast.
    """
    target = workdir / ".opencodeignore"
    if target.exists():
        return
    if not workdir.is_dir():
        return
    if OPENCODEIGNORE_TEMPLATE.exists():
        import shutil
        shutil.copy2(OPENCODEIGNORE_TEMPLATE, target)
        pipeline_log("INFO", f"Created {target} from template")
    else:
        pipeline_log("WARN", f"No opencodeignore template at {OPENCODEIGNORE_TEMPLATE}")


class OpenCodeBackend(AgentBackend):
    def run(self, prompt: str, workdir: str | None = None,
            skill: str | None = None, timeout: int | None = None,
            issue: int | None = None, stage: str | None = None,
            on_session_start: Callable[[str], None] | None = None) -> tuple[str, Path, str | None, TokenUsage]:
        workdir = workdir or str(PYTORCH_DIR)
        timeout = timeout or 1800

        # Ensure .opencodeignore exists in workdir to prevent file watcher
        # from hanging on large repos (e.g. pytorch with 200K+ files)
        _ensure_opencodeignore(Path(workdir))

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
            log_f.write("=== OUTPUT (real-time) ===\n")
            log_f.flush()

            # stdout = JSON events; discard stderr to avoid pipe buffer deadlock
            # Log full prompt to file for debugging
            prompt_log = log_path.with_suffix('.prompt.txt')
            with open(prompt_log, 'w') as pf:
                pf.write(full_prompt)
            stderr_path = log_path.with_suffix('.stderr.log')
            stderr_f = open(stderr_path, 'w')
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=stderr_f,
                stdin=subprocess.DEVNULL, text=True,
            )
            log_f.write(f"=== PID ===\n{proc.pid}\n\n")

            session_id = None
            text_parts = []
            token_usage = TokenUsage()
            start_time = time.monotonic()
            effective_timeout = timeout or 600
            try:
                # Use select() to enforce timeout even when no output arrives.
                # opencode --format json may not emit events during long tool
                # calls, so `for line in proc.stdout` would block indefinitely.
                has_fileno = hasattr(proc.stdout, 'fileno')
                if has_fileno:
                    fd = proc.stdout.fileno()
                buf = ""
                idle_timeout = 300  # kill if no output for 5 minutes
                last_output_time = time.monotonic()

                def _read_lines():
                    """Yield lines from stdout, using select() when available."""
                    nonlocal buf, last_output_time
                    if not has_fileno:
                        # Fallback for mocked/non-fd streams
                        for raw_line in proc.stdout:
                            yield raw_line
                        return
                    while True:
                        remaining = effective_timeout - (time.monotonic() - start_time)
                        if remaining <= 0:
                            proc.kill()
                            proc.wait()
                            log_f.write("\n=== TIMEOUT (wall-clock) ===\n")
                            raise subprocess.TimeoutExpired(
                                cmd, effective_timeout)
                        # Idle timeout: kill if no output for too long
                        idle_elapsed = time.monotonic() - last_output_time
                        if idle_elapsed > idle_timeout:
                            proc.kill()
                            proc.wait()
                            log_f.write(f"\n=== IDLE TIMEOUT ({idle_timeout}s no output) ===\n")
                            raise subprocess.TimeoutExpired(
                                cmd, effective_timeout)
                        ready, _, _ = select.select([fd], [], [], min(remaining, 30))
                        if not ready:
                            # Check if process exited while we were waiting
                            if proc.poll() is not None:
                                # Drain any remaining data
                                while True:
                                    r, _, _ = select.select([fd], [], [], 0)
                                    if not r:
                                        break
                                    leftover = os.read(fd, 65536)
                                    if not leftover:
                                        break
                                    buf += leftover.decode("utf-8", errors="replace")
                                    last_output_time = time.monotonic()
                                    while "\n" in buf:
                                        line, buf = buf.split("\n", 1)
                                        yield line + "\n"
                                # Process exited and pipe drained
                                log_f.write("\n=== PROCESS EXITED (rc=%d) ===\n" % proc.returncode)
                                break
                            continue
                        chunk = os.read(fd, 65536)
                        if not chunk:
                            break
                        # If the main process exited, drain remaining and stop
                        # (child processes may keep pipe open indefinitely)
                        if proc.poll() is not None:
                            buf += chunk.decode("utf-8", errors="replace")
                            last_output_time = time.monotonic()
                            # Drain with short timeout
                            while True:
                                r, _, _ = select.select([fd], [], [], 0.5)
                                if not r:
                                    break
                                leftover = os.read(fd, 65536)
                                if not leftover:
                                    break
                                buf += leftover.decode("utf-8", errors="replace")
                            while "\n" in buf:
                                line, buf = buf.split("\n", 1)
                                yield line + "\n"
                            log_f.write("\n=== PROCESS EXITED (rc=%d, drained) ===\n" % proc.returncode)
                            return
                        buf += chunk.decode("utf-8", errors="replace")
                        last_output_time = time.monotonic()
                        while "\n" in buf:
                            line, buf = buf.split("\n", 1)
                            yield line + "\n"

                for line in _read_lines():
                    # Enforce timeout (for fallback path)
                    if not has_fileno and time.monotonic() - start_time > effective_timeout:
                        proc.kill()
                        proc.wait()
                        log_f.write("\n=== TIMEOUT (wall-clock) ===\n")
                        raise subprocess.TimeoutExpired(cmd, effective_timeout)
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
                    # Accumulate token usage from step_finish events
                    if event.get("type") == "step_finish":
                        part = event.get("part", {})
                        tokens = part.get("tokens", {})
                        if tokens:
                            token_usage.add_step(tokens)
                        token_usage.add_cost(part.get("cost", 0))
                # Ensure the process is dead after stdout EOF
                # (opencode may close stdout while still waiting on API)
                if proc.poll() is None:
                    log_f.write("\n=== PROCESS STILL ALIVE AFTER EOF — killing ===\n")
                    proc.kill()
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                log_f.write("\n=== TIMEOUT ===\n")
                raise

            log_f.write(f"\n=== EXIT CODE ===\n{proc.returncode}\n")
            stderr_f.close()

        full_output = "".join(text_parts)

        pipeline_log("INFO", f"Agent finished (rc={proc.returncode}), "
                     f"log: {log_path} | {token_usage.summary()}", issue=issue)

        if proc.returncode != 0:
            raise RuntimeError(
                f"OpenCode failed (rc={proc.returncode}), "
                f"log: {log_path}\n{full_output[-500:]}"
            )
        return full_output, log_path, session_id, token_usage


class CopilotBackend(AgentBackend):
    """Future: GitHub Copilot cloud agent.
    Placeholder — implement when migrating to cloud.
    """
    def run(self, prompt: str, workdir: str | None = None,
            skill: str | None = None, timeout: int | None = None,
            issue: int | None = None, stage: str | None = None,
            on_session_start: Callable[[str], None] | None = None) -> tuple[str, Path, str | None, TokenUsage]:
        raise NotImplementedError("Copilot backend not yet implemented")


def get_backend() -> AgentBackend:
    from .config import AGENT_BACKEND
    if AGENT_BACKEND == "copilot":
        return CopilotBackend()
    return OpenCodeBackend()
