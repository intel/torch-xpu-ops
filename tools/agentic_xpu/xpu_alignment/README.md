# xpu-alignment — Usage Guide

Scan `pytorch/pytorch` for backend bug-fix issues, PRs, and commits that may also affect XPU,
adapt reproducers for XPU, validate locally, route confirmed bugs, and file tracking issues to
[intel/torch-xpu-ops](https://github.com/intel/torch-xpu-ops).

# TL;DR

## Workflow
```
daily_scan.sh / batch_scan.sh
        │
        ▼
  common.sh (env, venv, skill sync)
        │
        ├─► opencode + SKILL.md ─► raw scan JSON
        │
        ▼
  audit_scan_report.sh (validate reproducers)
        │
        ▼
  render_issue_ready_report.py ─► full_scan.md
  render_issue_drafts.py       ─► issue_drafts.md
```

## QuickStart

```Bash
cd tools/agentic_xpu
cp .env.example .env && chmod 600 .env
# Edit .env — set GITHUB_TOKEN

cd xpu_alignment
bash daily_scan.sh          # scan today's upstream commits
```

---

## Table of Contents

- [xpu-alignment — Usage Guide](#xpu-alignment--usage-guide)
- [TL;DR](#tldr)
  - [Workflow](#workflow)
  - [QuickStart](#quickstart)
  - [Table of Contents](#table-of-contents)
  - [1. Prerequisites](#1-prerequisites)
  - [2. One-time Setup](#2-one-time-setup)
    - [2a. Configure the GitHub token](#2a-configure-the-github-token)
  - [3. Running a Scan](#3-running-a-scan)
    - [Single day](#single-day)
    - [Custom date range](#custom-date-range)
    - [Last N days](#last-n-days)
    - [Specifying opencode binary explicitly](#specifying-opencode-binary-explicitly)
  - [4. Output Structure](#4-output-structure)
  - [5. Reading the Report](#5-reading-the-report)
    - [Bucket meanings](#bucket-meanings)
  - [6. Issue Drafts (dry-run)](#6-issue-drafts-dry-run)
    - [Issue body format specification](#issue-body-format-specification)
    - [Pre-filing checklist](#pre-filing-checklist)
  - [7. Cron / Automated Daily Scans](#7-cron--automated-daily-scans)
  - [8. Resuming an Interrupted Scan](#8-resuming-an-interrupted-scan)
  - [9. Environment Variables](#9-environment-variables)
  - [10. File Structure](#10-file-structure)

---

## 1. Prerequisites

| Requirement | Notes |
|-------------|-------|
| Linux x86_64 | Ubuntu 22.04 tested |
| Intel GPU (Data Center Max / Arc / Flex) | Verified on Max 1550 |
| Intel oneAPI base toolkit | Level-Zero, compute-runtime must be installed |
| `python3` (3.10+) | Used to bootstrap the venv |
| `opencode` CLI | Installed and connected to an LLM provider |
| `gh` CLI, authenticated | `gh auth login` |
| `jq` | `apt install jq` |

Check all at once:

```bash
command -v opencode && echo "opencode ok"
gh auth status
command -v jq && echo "jq ok"
python3 --version
```

---

## 2. One-time Setup

All steps run once per machine (or after a fresh clone).

### 2a. Configure the GitHub token

> **The `SKILL.md` is loaded directly from this repo — no manual skill copy needed.**
> The scan scripts auto-sync it into the workspace before every run.
> The XPU nightly venv is also created and refreshed automatically on every run.

Copy the example env file and fill in your GitHub token:

```bash
cd /path/to/torch-xpu-ops/tools/agentic_xpu
cp .env.example .env
# Edit .env — set GITHUB_TOKEN (and optionally XPU_ALIGNMENT_ISSUE_REPO)
chmod 600 .env
```

This workflow does not file issues automatically. Generated `gh issue create` commands use
`XPU_ALIGNMENT_ISSUE_REPO` (default: `intel/torch-xpu-ops`) if you want to override the target.

Alternatively, export `GITHUB_TOKEN` (and optionally `XPU_ALIGNMENT_ISSUE_REPO`) in your shell before running.

---

## 3. Running a Scan

All commands run from this directory:

```bash
cd /path/to/torch-xpu-ops/tools/agentic_xpu/xpu_alignment/
```

### Single day

```bash
bash daily_scan.sh 2026-05-21
```

If no date is given, yesterday is used:

```bash
bash daily_scan.sh
```

### Custom date range

```bash
bash batch_scan.sh 2026-05-18 2026-05-21
```

### Last N days

```bash
bash batch_scan.sh 7d     # last 7 days
bash batch_scan.sh 30d    # last 30 days
```

### Specifying opencode binary explicitly

```bash
OPENCODE_BIN=~/.opencode/bin/opencode bash batch_scan.sh 2026-05-18 2026-05-21
```

---

## 4. Output Structure

Each run writes to `runs/<date-or-range>/`:

```
runs/2026-05-18_to_2026-05-21/
├── run.log                          — full opencode session log
├── artifacts/
│   ├── raw_candidates.json          — deduplicated search results (786 entries)
│   ├── candidate_ledger.jsonl       — per-candidate state machine (title/deep/local status)
│   ├── collect_env.txt              — torch.utils.collect_env snapshot
│   ├── details/
│   │   └── <id>.json                — fetched issue/PR body or commit diff
│   └── output_<id>.log              — repro script stdout+stderr
├── scripts/
│   └── repro_<id>.py                — XPU-adapted reproducer
└── reports/
    ├── full_scan.md                 — complete scan report (all 344 validated entries)
    └── issue_drafts.md              — dry-run issue drafts for confirmed bugs
```

Both `full_scan.md` and `issue_drafts.md` are generated automatically at the end of every scan run by `render_issue_ready_report.py` and `render_issue_drafts.py` respectively (called from `run_post_scan` in `common.sh`).

To regenerate either report manually from an existing run:

```bash
# Regenerate full_scan.md
python3 render_issue_ready_report.py runs/2026-05-26

# Regenerate issue_drafts.md
python3 render_issue_drafts.py runs/2026-05-26
```

---

## 5. Reading the Report

`reports/full_scan.md` contains:

| Section | Content |
|---------|---------|
| Executive Summary | Counts: confirmed / not-reproduced / blocked / routed |
| Action Board | Table of all confirmed + related-failure entries with upstream links |
| Blockers | Entries that could not be validated (missing env, distributed setup, etc.) |
| Artifact Index | Absolute paths to ledger, env snapshot, raw candidates |
| Triage / Validation Stats | Funnel numbers at each filter stage |
| Tested Candidates | Full detail entries with repro paths, output excerpts, `Local XPU result` |
| Final Summary | Filter → validation → routing statistics |

The `Local XPU result: \`<bucket>\`` line in each entry is the canonical outcome.

### Bucket meanings

| Bucket | Meaning |
|--------|---------|
| `confirmed` | Same bug reproduces on XPU |
| `related-failure` | XPU fails differently on the same scenario |
| `not-reproduced` | Upstream failure does not reproduce on XPU |
| `blocked-env` | Missing dependency (e.g. `transformers`) |
| `blocked-platform` | XPU lacks required code path |
| `blocked-script-error` | Repro failed before reaching the oracle |
| `needs-performance-harness` | Perf-only regression, needs benchmark |
| `not-applicable` | Rejected during title or deep triage |

---

## 6. Issue Drafts (dry-run)

After a scan, `issue_drafts.md` is automatically generated alongside `full_scan.md` in
`runs/<date-or-range>/reports/`. It contains GitHub-ready issue bodies for every
`confirmed` or `related-failure` candidate, following the
[pytorch/pytorch bug report template](https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml).

If no actionable candidates were found the file is still written with a short notice.
To regenerate it manually from an existing run:

```bash
python3 render_issue_drafts.py runs/2026-05-26
```

Each draft contains exactly two sections:

**`🐛 Describe the bug`** — includes:
- A source header line identifying the upstream reference and scan metadata
- Clear description of the bug with root-cause analysis where available
- A minimal, self-contained Python reproducer (XPU-adapted, copy-pasteable)
- Actual output / error message block

**`Versions`** — the full `torch.utils.collect_env` output captured during the scan run
(`artifacts/collect_env.txt`), pasted verbatim.

### Issue body format specification

```markdown
**Upstream source:** <upstream URL> (upstream-issue | upstream-pr)
**Scan date:** <YYYY-MM-DD> to <YYYY-MM-DD>
**Local XPU result:** confirmed on torch <version>, <GPU model>

---

### 🐛 Describe the bug

<description>

\```python
# minimal reproducer
\```

\```
<actual output / error>
\```

---

### Versions

\```
<full collect_env output>
\```
```

### Pre-filing checklist

Before turning a draft into a real issue:

1. Verify the upstream issue/PR is not already fixed in the current nightly.
2. Re-run the reproducer against the latest nightly to confirm the bug persists.
3. Update the `Versions` block to reflect the version you tested.

```bash
# Re-run a specific reproducer against the current venv
# WORKSPACE is the repo root (where .conda-xpu-fix-alignment lives)
$WORKSPACE/.conda-xpu-fix-alignment/bin/python \
  runs/2026-05-18_to_2026-05-21/scripts/repro_184340.py
```

---

## 7. Cron / Automated Daily Scans

To run a daily scan automatically at 05:31:

```bash
ENTRY=/path/to/torch-xpu-ops/tools/agentic_xpu/xpu_alignment
OPENCODE=$(command -v opencode)
WORKSPACE=/path/to/torch-xpu-ops

(crontab -l 2>/dev/null; echo "31 5 * * * OPENCODE_BIN=$OPENCODE WORKSPACE=$WORKSPACE \
  /bin/bash -lc 'mkdir -p $ENTRY/runs/logs && \
  exec /usr/bin/flock -n $ENTRY/runs/.daily.lock \
  /bin/bash $ENTRY/daily_scan.sh >> $ENTRY/runs/logs/cron.log 2>&1'") | crontab -
```

Verify the cron entry:
```bash
crontab -l
```

Cron logs: `runs/logs/cron.log`

---

## 8. Resuming an Interrupted Scan

`batch_scan.sh` automatically resumes if it detects a previous session:

```bash
# Just re-run the same command — the session ID and ledger are detected
bash batch_scan.sh 2026-05-18 2026-05-21
```

The ledger (`artifacts/candidate_ledger.jsonl`) tracks per-candidate progress.
Rows with `local_status: "done"` are skipped on resume.

To force a fresh scan, remove the run directory:

```bash
rm -rf runs/2026-05-18_to_2026-05-21
bash batch_scan.sh 2026-05-18 2026-05-21
```

---

## 9. Environment Variables

All optional — auto-detected if unset.

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENCODE_BIN` | Path to `opencode` binary | `opencode` in `$PATH` |
| `WORKSPACE` | Workspace root (where `.opencode/` lives) | Auto-detected upward from entry dir |
| `ENV_FILE` | Path to `.env` file with `GITHUB_TOKEN` | `<entry-dir>/../.env` |
| `GITHUB_TOKEN` | GitHub personal access token | Inherited from env or `.env` |
| `GH_TOKEN` | Alias for `GITHUB_TOKEN` (used by `gh` CLI) | Set from `GITHUB_TOKEN` |
| `XPU_ALIGNMENT_ISSUE_REPO` | Target issue repo for generated filing commands | `intel/torch-xpu-ops` |

---

## 10. File Structure

```
.claude/skills/xpu-alignment/SKILL.md      ← agent skill (loaded by opencode)
tools/agentic_xpu/
├── .env.example                            ← shared env config (all scenarios)
└── xpu_alignment/
    ├── common.sh                           ← shared helpers (env, venv, gh)
    ├── daily_scan.sh                       ← single-day scan entry point
    ├── batch_scan.sh                       ← multi-day scan entry point
    ├── audit_scan_report.sh                ← validate reproducers
    ├── render_issue_ready_report.py        ← generate full scan report
    ├── render_issue_drafts.py              ← generate issue drafts
    └── README.md
```
