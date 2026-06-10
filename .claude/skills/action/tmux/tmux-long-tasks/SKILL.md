---
name: tmux-long-tasks
description: Use tmux to launch and manage long-running jobs that must survive beyond the bash tool timeout. Use this skill when the user mentions long-running commands, background tasks, training jobs, large builds, or any command expected to exceed the bash tool timeout.
license: MIT
metadata:
  tmux: long-running jobs
  bash: background tasks
---

# Run Long Background Tasks with tmux

This SKILL is used when a command or workload may run longer than the bash tool timeout and must continue running after the tool call returns.

The bash tool runs commands inside a temporary shell. When the tool times out, the shell's process group is killed, including background processes started with `&` or `nohup`. Therefore, long-running jobs must be launched through `tmux` instead of normal shell backgrounding.

## When to Use This SKILL

Use this SKILL when the task involves any long-running command, such as:

- Multi-hour model training
- Long benchmark runs
- Large build or compilation jobs
- Long test suites
- Data preprocessing jobs
- Server processes that need to keep running
- Any command expected to exceed the bash tool timeout

## Core Rule

Do not use `&`, `nohup`, or normal shell backgrounding for long-running jobs.

Instead, launch the job inside a dedicated `tmux` window.

## Required Inputs

Before launching the job, make sure the following inputs are available:

- The target `tmux` session name
- The job/window name
- The working directory
- The command to run
- The log file path, if the user requires a specific one

If the command or working directory is missing, ask the user first and stop.

If the `tmux` session name is missing, inspect available sessions if possible. If no suitable session can be inferred, ask the user and stop.

## Instructions

### Step 1: Check available tmux sessions and windows

Before creating a new window, check the available `tmux` sessions and windows.

Use:

```bash
tmux list-sessions
```

Then list windows in the target session:

```bash
tmux list-windows -t <session>
```

This helps avoid creating duplicate windows or using the wrong session.

### Step 2: Create a dedicated tmux window

Create a dedicated detached `tmux` window for the job.

Use:

```bash
tmux new-window -d -n <job_name> -t <session>
```

Example:

```bash
tmux new-window -d -n train_llama -t my_session
```

Use a clear job name that reflects the workload.

### Step 3: Send the command to the tmux window

Send the long-running command to the dedicated window.

Use:

```bash
tmux send-keys -t <session>:<job_name> 'cd /work/dir || exit 1; CMD 2>&1 | tee run.log' Enter
```

Example:

```bash
tmux send-keys -t my_session:train_llama 'cd /work/dir || exit 1; python train.py 2>&1 | tee run.log' Enter
```

Use `cd <dir> || exit 1` instead of `cd <dir> && CMD`. If the directory does not exist or is inaccessible, `|| exit 1` terminates the shell immediately and makes the failure visible in the pane output. With `&&`, the shell stays open silently in the wrong directory.

The command should:

1. Change to the correct working directory (abort on failure).
2. Run the target command.
3. Redirect both stdout and stderr.
4. Save logs using `tee`.

### Step 3.1: Verify the job started

After sending the command, wait briefly and check that the job is running:

```bash
sleep 2
tmux capture-pane -p -t <session>:<job_name> | tail -5
```

Look for signs of immediate failure:

- Shell prompt reappeared (command exited immediately)
- Python traceback or error message visible
- `No such file or directory` or permission errors

If the job failed immediately, report the error to the user before continuing. Do not assume the job is running without checking.

### Step 4: Check job progress

To check progress without disturbing the running job, there are two methods:

#### Quick check: tmux pane buffer

```bash
tmux capture-pane -p -t <session>:<job_name> | tail -20
```

Note: `capture-pane` only shows the pane's scrollback buffer (typically the last ~2000 lines). For long-running jobs, early output may no longer be in the buffer.

#### Full history: log file

The log file written by `tee` contains the complete output from the start of the job. Use it for full history or searching past output:

```bash
tail -20 /work/dir/run.log
```

For searching specific events:

```bash
grep -i "error\|exception\|warning" /work/dir/run.log | tail -10
```

Prefer the log file over `capture-pane` when checking for errors or reviewing output from earlier in the run.

### Step 5: Stop the job cleanly

To stop the job cleanly, send `Ctrl-C` to the tmux window.

Use:

```bash
tmux send-keys -t <session>:<job_name> C-c
```

Example:

```bash
tmux send-keys -t my_session:train_llama C-c
```

Do not kill the process abruptly unless the clean stop fails.

### Step 6: Avoid unsafe long-running patterns

Do not use the following patterns for long-running jobs:

```bash
CMD &
```

```bash
nohup CMD &
```

```bash
CMD > run.log 2>&1 &
```

These jobs may be killed when the bash tool times out because they remain in the same temporary shell process group.

### Step 7: Report the launched job information

After launching the job, report the following information to the user:

```text
The long-running job has been launched in tmux.

Session: <session>
Window: <job_name>
Working directory: <work_dir>
Log file: <log_file>

To check progress:
tmux capture-pane -p -t <session>:<job_name> | tail -20

To stop the job:
tmux send-keys -t <session>:<job_name> C-c
```

## Example

```bash
tmux list-windows -t my_session

tmux new-window -d -n train_llama -t my_session

tmux send-keys -t my_session:train_llama 'cd /work/dir || exit 1; python train.py 2>&1 | tee run.log' Enter

tmux capture-pane -p -t my_session:train_llama | tail -20
```

To stop:

```bash
tmux send-keys -t my_session:train_llama C-c
```

## Failure Handling

### tmux is not installed

If `tmux` is not available, stop and report:

```text
tmux is not installed or not available in PATH. Long-running jobs should not be launched with &, nohup, or normal shell backgrounding because they may be killed when the bash tool times out.
```

### Target session does not exist

If the target session does not exist, list available sessions and ask the user which session to use.

If creating a new session is allowed by the user's workflow, create one explicitly:

```bash
tmux new-session -d -s <session>
```

Then create the job window under that session.

### Duplicate window name

If a window with the same job name already exists, do not overwrite or reuse it without checking.

Either ask the user or choose a clearly distinct name, such as:

```text
<job_name>_2
<job_name>_YYYYMMDD_HHMM
```

## Summary

For any job that must survive beyond the bash tool call, use `tmux`.

Do not rely on `&`, `nohup`, or shell backgrounding for long-running workloads.
