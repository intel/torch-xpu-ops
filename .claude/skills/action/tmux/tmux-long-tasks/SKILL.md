---
name: tmux-long-tasks
description: Use tmux to launch and manage long-running jobs that must survive beyond the bash tool timeout. Use this skill when the user mentions long-running commands, background tasks, training jobs, large builds, or any command expected to exceed the bash tool timeout.
license: MIT
metadata:
  tmux: long-running jobs
  bash: background tasks
---

# Run Long Tasks with tmux

The bash tool kills all processes (including `&` and `nohup`) when it times out. Use `tmux` for any job that may exceed the timeout.

Do NOT use `&`, `nohup`, or shell backgrounding for long-running jobs.

## Required Inputs

Before launching, you need:
- tmux session name
- Job/window name
- Working directory
- Command to run

If any are missing, ask the user and stop.

## Instructions

### Step 1: Check tmux sessions

```bash
tmux list-sessions 2>/dev/null || echo "NO_TMUX_SESSIONS"
```

If tmux is not installed, stop and report.

If the target session does not exist, create it:
```bash
tmux new-session -d -s <session>
```

### Step 2: Create a window and launch the job

```bash
tmux new-window -d -n <job_name> -t <session>
tmux send-keys -t <session>:<job_name> 'cd /work/dir || exit 1; CMD 2>&1 | tee run.log' Enter
```

Use `cd <dir> || exit 1` (not `&&`) so failures are visible in pane output.

If a window with the same name already exists, append a suffix (e.g., `_2`).

### Step 3: Verify the job started

```bash
sleep 2
tmux capture-pane -p -t <session>:<job_name> | tail -5
```

If the shell prompt reappeared or an error is visible, the job failed immediately. Report to user.

### Step 4: Check progress

```bash
tmux capture-pane -p -t <session>:<job_name> | tail -20
```

For full history, use the log file:
```bash
tail -20 /work/dir/run.log
```

### Step 5: Stop the job

```bash
tmux send-keys -t <session>:<job_name> C-c
```

### Step 6: Report to user

After launching, report:
```text
Job launched in tmux.
Session: <session> | Window: <job_name>
Log: <log_file>
Check: tmux capture-pane -p -t <session>:<job_name> | tail -20
Stop:  tmux send-keys -t <session>:<job_name> C-c
```
