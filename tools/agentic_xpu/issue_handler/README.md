# Issue Handler Agent

An automated pipeline that processes GitHub issues end-to-end: formatting → triage → fix → verification.

## Pipeline Stages

```
  Raw Issue
      │
      ▼
  ┌────────┐   Extracts sections, classifies bug/non-bug,
  │ format │   rewrites body from template, sets status DISCOVERED
  └────┬───┘
       ▼
  ┌─────────────────┐   Runs the reproducer against current PyTorch HEAD.
  │ verify_existence │   If already fixed → DONE. Otherwise continues.
  └────────┬────────┘
           ▼
  ┌────────────────────┐   If issue links an upstream pytorch/pytorch PR,
  │ verify_upstream_pr │   LLM classifies the diff as DEVICE_AGNOSTIC vs
  └─────────┬──────────┘   DEVICE_SPECIFIC. Agnostic → checkout PR, rebuild,
            │              re-run reproducer:
            │                · PR merged + repro PASS → DONE (close issue)
            │                · PR open   + repro PASS → WAITING_UPSTREAM
            │                · repro FAIL or unverifiable → NEEDS_HUMAN
            │              Specific → fall through to triage.
            ▼
  ┌────────┐   LLM reads reproducer + error log + PyTorch source,
  │ triage │   identifies root cause, target repo, fix strategy → TRIAGED
  └────┬───┘
       ▼
  ┌─────┐   Spawns coding agent (OpenCode/Copilot) to implement fix,
  │ fix │   creates branch + PR on review repo → IN_REVIEW
  └──┬──┘
     ▼
  ┌────────────┐   Checks out fix branch, rebuilds PyTorch,
  │ verify_fix │   runs reproducer → DONE or NEEDS_HUMAN
  └────────────┘
```

![Issue Handler](../assets/issue-handler.png)

Each stage reads the issue body status marker (`<!-- agent:status:STAGE -->`) and writes the next one on completion.

### Stage → Status Mapping

| CLI stage name | Agent module        | Output status                  |
|----------------|---------------------|--------------------------------|
| `format`       | `format_agent.py`   | `DISCOVERED`                   |
| (auto)         | `verify_existence.py` | `DONE` (if already fixed) or continues |
| (auto)         | `verify_upstream_pr.py` | `DONE` / `WAITING_UPSTREAM` / `NEEDS_HUMAN`, or falls through to triage |
| `triage`       | `triage_agent.py`   | `TRIAGED` or `NEEDS_HUMAN`     |
| `fix`          | `code_fix.py`       | `IMPLEMENTING` → `IN_REVIEW`   |
| `verify_fix`   | `verify_fix.py`     | `DONE` or `NEEDS_HUMAN`        |

### Terminal Stages

The pipeline stops advancing when an issue reaches: `DONE`, `SKIPPED`, or `NEEDS_HUMAN`. `WAITING_UPSTREAM` is non-terminal: the poller re-checks every 12 hours (throttle) until the upstream PR merges or the repro stops passing.

---

## Reproducer command pipeline

The reproducer command for an issue is *discovered once* and then *replayed many times* across downstream stages. Two modules cooperate:

- **Discovery (LLM-mediated)** — `verify_existence.py` spawns an OpenCode agent with the `test-verification` skill. The agent reads the body, figures out the right pytest/python invocation, runs it once, and returns a JSON `{status, refined_command, output_tail, …}`. `verify_existence` then writes `**Refined command:** \`<cmd>\`` into the issue body's `Reproducer` section.
- **Replay (direct subprocess)** — `utils/reproducer.py` exposes `extract_reproducer_command(body)` (regex-parses the refined-command line back out) and `run_reproducer_command(cmd)` (executes it under `ENV_SETUP` from `utils/xpu_env.py`, returns a `ReproResult`). This is what `verify_upstream_pr.py` and any future stages use to re-run the cached command without burning a $0.12 LLM call per check.

This split keeps reasoning (which command? did it find tests?) on the LLM path and pure execution on the cheap, deterministic path. The refined-command line in the body is the authoritative cache — both stages read/write through it rather than passing state in memory, so each stage can run in a fresh Python process.

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/intel/torch-xpu-ops.git ~/torch-xpu-ops
cd ~/torch-xpu-ops
```

### 2. Install dependencies

```bash
pip install pyyaml   # only hard dependency for the pipeline itself
```

The coding agent backend requires [OpenCode](https://github.com/nicepkg/opencode) or GitHub Copilot CLI installed and configured separately.

### 3. Configure environment

Copy the shared env template and fill in your tokens and repo settings:

```bash
cd /path/to/torch-xpu-ops/tools/agentic_xpu
cp .env.example .env
# Edit .env — at minimum set GITHUB_TOKEN and PRIVATE_REVIEW_REPO
chmod 600 .env
```

Key variables:

| Variable              | Description                                          | Required |
|-----------------------|------------------------------------------------------|----------|
| `GITHUB_TOKEN`        | GitHub PAT with write access to issue + review repos | ✅       |
| `PRIVATE_REVIEW_REPO` | Private review fork (e.g. `yourorg/pytorch`)         | ✅       |
| `PYTORCH_DIR`         | Path to local PyTorch checkout (default: `~/pytorch`)| Optional |
| `AGENT_BACKEND`       | `opencode` (default) or `copilot`                    | Optional |
| `OPENCODE_CMD`        | Path to the opencode binary (default: `opencode`)    | Optional |

Source the env file before running:
```bash
set -a && source tools/agentic_xpu/.env && set +a
```

### 4. Configure repos (optional)

Edit `tools/agentic_xpu/issue_handler/agentic_xpu/issue_handler/config/agent_config.yml` to change target repos, labels, timeouts, or git settings. Environment variables override YAML values:

The pipeline interacts with multiple GitHub repos for different purposes:

- **`ISSUE_REPO`** — Where issues are filed and tracked. The pipeline reads issues from here, rewrites their bodies, posts comments, and manages labels.
- **`PRIVATE_REVIEW_REPO`** — A private fork (e.g. `chuanqi129/pytorch`) where the agent pushes fix branches and opens PRs for review before submitting upstream. This keeps experimental agent PRs out of the public `pytorch/pytorch` repo.
- **`PUBLIC_TARGET_REPO`** — The upstream `pytorch/pytorch` repo. Used as a reference for diffing and eventually submitting verified fixes.
- **`TRACKING_REPO`** — Where the pipeline posts E2E dashboard reports (a tracking issue that summarizes batch run results). Can be the same as `ISSUE_REPO`.

| Env Variable          | YAML key                | Default                        |
|-----------------------|-------------------------|--------------------------------|
| `ISSUE_REPO`          | `repos.xpu_ops_issue`   | `intel/torch-xpu-ops`          |
| `PRIVATE_REVIEW_REPO` | `repos.pytorch_private` | `chuanqi129/pytorch`           |
| `PUBLIC_TARGET_REPO`  | `repos.pytorch_public`  | `pytorch/pytorch`              |
| `TRACKING_REPO`       | `repos.tracking`        | `intel/torch-xpu-ops`          |

### 5. PyTorch checkout

The `verify_existence`, `fix`, and `verify_fix` stages execute Python test code locally to check whether an issue reproduces or a fix works. This requires a local PyTorch build with XPU (Intel GPU) support:

```bash
cd ~/pytorch
git submodule sync && git submodule update --init --recursive
pip install -e . -v --no-build-isolation   # with USE_XPU=1, oneAPI sourced
```

The pipeline automatically syncs (`git pull` + submodule update) and rebuilds if the binary is stale before each verification run.

**Auto-stash safety net.** Before `git pull`, the pipeline checks the main `~/pytorch` worktree with `git status --porcelain --ignore-submodules=all` and auto-cleans unconditionally so the pipeline stays self-healing across runs:

- If dirty → all changes are stashed as `agent-autoclean-<UTC-timestamp>-issue-<N>`. The stash ref + recovery command (`git stash apply 'stash@{0}'`) are appended to the issue body's `<!-- agent:env-log -->` section so any auto-stashed work is recoverable.
- If HEAD is on an `agent/*` branch (leftover from a prior fix-agent run) → after stashing, the pipeline checks out `main` so the pull operates on a valid tracked branch. The previous branch name is recorded in the env-log note. The `agent/*` branch itself is preserved (not deleted) — `git checkout agent/issue-<N>` recovers it.
- If `git status` / `git stash push` / `git checkout main` fail → `RuntimeError` is raised, `sync_pytorch` returns `False`, and the stage soft-fails (logged, no body update).

A pruner keeps the stash list bounded: autoclean stashes older than **7 days** or beyond the **10 most recent** (autoclean stashes only — user/manual stashes are untouched) are dropped after each new stash. Inspect with:

```bash
cd ~/pytorch && git stash list | grep agent-autoclean
```

---

## Usage

```bash
# Source environment
cd ~/torch-xpu-ops
set -a && source tools/agentic_xpu/.env && set +a


# Run all stages on a specific issue
python tools/agentic_xpu/issue_handler/run_pipeline.py --once --issues 12

# Run specific stages only
python tools/agentic_xpu/issue_handler/run_pipeline.py --once --issues 12 --stages format triage

# Run fix and verify on an already-triaged issue
python tools/agentic_xpu/issue_handler/run_pipeline.py --once --issues 12 --stages fix verify_fix

# Run all open agent issues (fetches issues with the ai_generated label)
python tools/agentic_xpu/issue_handler/run_pipeline.py --once

# Multiple issues
python tools/agentic_xpu/issue_handler/run_pipeline.py --once --issues 12 15 23
```

### CLI Options

| Flag         | Description                                                |
|--------------|------------------------------------------------------------|
| `--once`     | Run one cycle and exit (required)                          |
| `--issues N` | Process specific issue number(s). Default: all open agent issues |
| `--stages`   | Run only named stages: `format`, `triage`, `fix`, `verify_fix`. Default: all |
| `--batch`    | Label for the dashboard report batch                       |

