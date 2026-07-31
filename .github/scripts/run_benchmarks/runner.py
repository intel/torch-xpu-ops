"""Benchmark orchestrator: single-task execution and multi-worker dispatch."""

import os
import queue
import shlex
import subprocess
import tempfile
import threading
import time
from contextlib import suppress
from pathlib import Path

from . import config
from .config import IS_WINDOWS, TestTask
from .log import fmt_duration, log
from .suites import get_suite
from .monitor import (
    get_gpu_memory_utilization,
    get_host_memory_utilization,
    parse_memory_threshold,
)


def _build_cmd_list(cmd: str, cmd_prefix: str) -> tuple[list[str], str]:
    """Split command (with optional prefix) into an argv list and display string."""
    posix = not IS_WINDOWS
    if cmd_prefix:
        cmd_list = shlex.split(cmd_prefix, posix=posix) + shlex.split(cmd, posix=posix)
        cmd_str = f"{cmd_prefix} {cmd}"
    else:
        cmd_list = shlex.split(cmd, posix=posix)
        cmd_str = cmd
    return cmd_list, cmd_str


def run_single(
    task: TestTask,
    card: int,
    cmd_prefix: str,
    env_vars: dict,
    log_dir: Path,
    worker_id: int,
    device: str,
    shape: str,
    dataset_dir: str = "",
) -> tuple[int, bool, str, str | None]:
    """Run a single benchmark and return (exit_code, success, test_result, kill_reason)."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"logs-{task.model.replace('/', '_')}-worker{worker_id}-card{card}.log"
    log_csv = log_dir / f"inductor-{task.suite}-{task.dt}-{task.mode}-{device}-{task.scenario}.csv"

    suite = get_suite(task)
    tmp_log_csv = None
    if suite.uses_temp_csv:
        tmp_fd, tmp_log_csv = tempfile.mkstemp(prefix='tmp_', suffix='.csv')
        os.close(tmp_fd)

    suite.prepare(task)
    cmd = suite.build_command(task, device, shape, dataset_dir, tmp_log_csv)
    full_cmd_list, full_cmd_str = _build_cmd_list(cmd, cmd_prefix)
    log(f"Running: {full_cmd_str[:200]}...", worker=worker_id)

    env = {**os.environ, **env_vars}
    if device != "cpu":
        env["ZE_AFFINITY_MASK"] = str(card)
    env.update(suite.env_overrides(task))

    # Shared state for threads
    log_buffer: list[str] = []
    log_buffer_lock = threading.Lock()
    kill_reason: list[str | None] = [None]
    kill_reason_lock = threading.Lock()

    popen_kwargs = {
        "shell": False, "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT,
        "text": True, "env": env, "bufsize": 1,
    }
    if IS_WINDOWS:
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    log_f = None
    try:
        log_f = open(log_file, "w")
        run_start = time.monotonic()
        proc = subprocess.Popen(full_cmd_list, **popen_kwargs)
    except Exception as e:
        if log_f is not None and not log_f.closed:
            log_f.close()
        raise RuntimeError(f"Failed to start benchmark for {task.model}: {e}") from e

    def _output_reader():
        if proc.stdout is None:
            return
        try:
            for line in iter(proc.stdout.readline, ""):
                formatted = f"{time.strftime('[%Y-%m-%d %H:%M:%S]')} {line.rstrip()}"
                print(formatted, flush=True)
                with log_buffer_lock:
                    log_buffer.append(formatted)
                try:
                    log_f.write(formatted + "\n")
                    log_f.flush()
                except (ValueError, OSError):
                    pass
        except (ValueError, OSError):
            pass

    def _error_monitor():
        scan_pos = 0
        memory_threshold = parse_memory_threshold()
        deadline = time.time() + config.task_timeout_seconds

        while proc.poll() is None and time.time() < deadline:
            with log_buffer_lock:
                new_lines = log_buffer[scan_pos:]
                scan_pos = len(log_buffer)
            if new_lines:
                content = "\n".join(new_lines).lower()
                for pattern in config.error_patterns:
                    if pattern.startswith("Memory>"):
                        continue
                    if pattern.lower() in content:
                        log(f"Detected '{pattern}' — killing process", level="WARN", worker=worker_id)
                        with kill_reason_lock:
                            kill_reason[0] = pattern
                        proc.kill()
                        return

            if memory_threshold is not None:
                try:
                    if device == "cpu":
                        mem_util = get_host_memory_utilization(memory_threshold, task)
                    else:
                        mem_util = get_gpu_memory_utilization(card, memory_threshold, task)
                except Exception as e:
                    log(f"Memory poll error: {e}", level="WARN", worker=worker_id)
                    mem_util = None
                if mem_util is not None and mem_util >= memory_threshold:
                    mem_kind = "Host" if device == "cpu" else "GPU"
                    log(
                        f"{mem_kind} memory {mem_util:.2%} >= threshold {memory_threshold:.0%} — killing process",
                        level="WARN", worker=worker_id,
                    )
                    with kill_reason_lock:
                        kill_reason[0] = f"Memory>{memory_threshold}"
                    proc.kill()
                    return

            time.sleep(3)

        if proc.poll() is None:
            log("Monitor deadline reached — killing process", level="WARN", worker=worker_id)
            with kill_reason_lock:
                kill_reason[0] = "timeout"
            proc.kill()

    reader_thread = threading.Thread(target=_output_reader, daemon=True)
    monitor_thread = threading.Thread(target=_error_monitor, daemon=True)
    reader_thread.start()
    monitor_thread.start()

    try:
        exit_code = proc.wait()
    except Exception:
        exit_code = -1
    elapsed = time.monotonic() - run_start
    reader_thread.join(timeout=10)
    if reader_thread.is_alive():
        log("Output reader thread still running after join; some log lines may be lost",
            level="WARN", worker=worker_id)
    if log_f is not None and not log_f.closed:
        log_f.close()
    monitor_thread.join(timeout=1)

    with kill_reason_lock:
        matched_pattern = kill_reason[0]
    test_result, success = "exception", False
    try:
        test_result, success = suite.collect_results(
            log_csv, log_file, tmp_log_csv, device, task, matched_pattern, elapsed)
    except Exception as e:
        log(f"CSV collection error for {task.model}: {e}", level="ERROR", worker=worker_id)
    finally:
        if tmp_log_csv is not None:
            with suppress(OSError):
                os.unlink(tmp_log_csv)

    return exit_code, success, test_result, matched_pattern


def run_all(
    tasks: list[TestTask],
    workers: list[tuple[int, str, dict]],
    device: str,
    shape: str,
    dataset_dir: str = "",
) -> None:
    """Dispatch tasks across workers and print a summary."""
    total = len(tasks)
    task_queue: queue.Queue[TestTask] = queue.Queue()
    for t in tasks:
        task_queue.put(t)

    results: queue.Queue[tuple[TestTask, bool, str, str | None]] = queue.Queue()
    completed = 0
    completed_lock = threading.Lock()
    base_log_dir = Path.cwd().resolve() / "inductor_log"
    wall_start = time.monotonic()

    all_threads: list[threading.Thread] = []
    threads_lock = threading.Lock()

    def _spawn_worker(worker_id: int, card: int, cmd_prefix: str, env_vars: dict) -> None:
        t = threading.Thread(
            target=_worker, args=(worker_id, card, cmd_prefix, env_vars), daemon=True,
        )
        with threads_lock:
            all_threads.append(t)
        t.start()

    def _worker(worker_id: int, card: int, cmd_prefix: str, env_vars: dict) -> None:
        nonlocal completed
        while True:
            try:
                task = task_queue.get_nowait()
            except queue.Empty:
                return

            task_start = time.monotonic()
            log_dir = base_log_dir / task.suite / task.dt / task.mode / task.scenario
            try:
                exit_code, success, test_result, kill_reason = run_single(
                    task=task, card=card, cmd_prefix=cmd_prefix, env_vars=env_vars,
                    log_dir=log_dir, worker_id=worker_id, device=device, shape=shape,
                    dataset_dir=dataset_dir,
                )
            except Exception as e:
                log(f"Exception running {task.model}: {e}", level="ERROR", worker=worker_id)
                exit_code, success, test_result, kill_reason = -1, False, "exception", None

            elapsed = fmt_duration(time.monotonic() - task_start)
            with completed_lock:
                completed += 1
                n = completed

            pct = n * 100 // total
            progress = f"[{n}/{total} {pct}%]"
            model_info = f"{task.suite}/{task.model} ({task.dt}, {task.mode}, {task.scenario})"
            if success:
                log(f"{progress} PASS  {model_info} ({elapsed}) → {test_result}", worker=worker_id)
            elif kill_reason:
                log(f"{progress} KILL  {model_info} ({elapsed}) → {kill_reason}", level="WARN", worker=worker_id)
            elif exit_code == 0:
                log(f"{progress} FAIL  {model_info} ({elapsed}) → {test_result}", level="WARN", worker=worker_id)
            else:
                log(
                    f"{progress} FAIL  {model_info} ({elapsed}) → exit code {exit_code}",
                    level="ERROR",
                    worker=worker_id,
                )

            results.put((task, success, test_result, kill_reason))

            # A kill (error pattern / OOM / timeout) may leave this worker's
            # device or process state dirty. Retire this worker and start a
            # fresh replacement with the same id/card to drain remaining tasks.
            if kill_reason:
                log(
                    f"Worker killed on '{kill_reason}' — starting replacement worker",
                    level="WARN", worker=worker_id,
                )
                _spawn_worker(worker_id, card, cmd_prefix, env_vars)
                return

    for idx, (card, pfx, ev) in enumerate(workers):
        _spawn_worker(idx, card, pfx, ev)

    # Join all workers, including replacements spawned after a kill.
    while True:
        with threads_lock:
            pending = [t for t in all_threads if t.is_alive()]
        if not pending:
            break
        for t in pending:
            t.join(timeout=0.5)

    # Summary
    from .log import banner
    wall_time = fmt_duration(time.monotonic() - wall_start)
    all_results = []
    while True:
        try:
            all_results.append(results.get_nowait())
        except queue.Empty:
            break
    failed = [(task, tr, kr) for task, ok, tr, kr in all_results if not ok]
    passed = total - len(failed)

    banner("Summary")
    log(f"Total:     {total}")
    log(f"Passed:    {passed} ({passed * 100 // total}%)")
    log(f"Failed:    {len(failed)} ({len(failed) * 100 // total}%)")
    log(f"Wall time: {wall_time}")
    if failed:
        print(flush=True)
        log("Failed tasks:")
        for task, test_result, kill_reason in failed:
            reason = kill_reason or test_result
            log(f"  ✗ {task.suite}/{task.model} ({task.dt}, {task.mode}, {task.scenario}) → {reason}")
    else:
        print(flush=True)
        log("All tasks completed successfully.")

    if failed:
        import sys
        sys.exit(1)
