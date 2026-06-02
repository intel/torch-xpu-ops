# pytorch-cuda-fix-xpu-alignment — Usage Guide

Scan `pytorch/pytorch` for backend bug-fix issues, PRs, and commits that may also affect XPU,
adapt reproducers for XPU, validate locally, route confirmed bugs, and file tracking issues to
[intel/torch-xpu-ops](https://github.com/intel/torch-xpu-ops).

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [One-time Setup](#2-one-time-setup)
3. [Running a Scan](#3-running-a-scan)
4. [Output Structure](#4-output-structure)
5. [Reading the Report](#5-reading-the-report)
6. [Issue Drafts (dry-run)](#6-issue-drafts-dry-run)
7. [Filing Issues to torch-xpu-ops](#7-filing-issues-to-torch-xpu-ops)
8. [Cron / Automated Daily Scans](#8-cron--automated-daily-scans)
9. [Resuming an Interrupted Scan](#9-resuming-an-interrupted-scan)
10. [Environment Variables](#10-environment-variables)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Prerequisites

| Requirement | Notes |
|-------------|-------|
| Linux x86_64 | Ubuntu 22.04 tested |
| Intel GPU (Data Center Max / Arc / Flex) | Verified on Max 1550 |
| Intel oneAPI base toolkit | Level-Zero, compute-runtime must be installed |
| `python3` (3.10+) | Used to bootstrap the venv |
| `opencode` CLI | Installed and connected to an LLM provider (see [INSTALL_AGENTS.md](./INSTALL_AGENTS.md)) |
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

> **Prerequisites:** OpenCode must be installed and connected to an LLM provider before running this scenario.
> See [INSTALL_AGENTS.md](./INSTALL_AGENTS.md) for one-time OpenCode setup instructions.

### 2a. Configure the GitHub token

> **The `SKILL.md` is loaded directly from this repo — no manual skill copy needed.**
> The scan scripts auto-sync it into the workspace before every run.
> The XPU nightly venv is also created and refreshed automatically on every run.

Create `tools/agentic_xpu/tokens.env` with your GitHub token.
**Never commit this file.**

```bash
cd /path/to/torch-xpu-ops
mkdir -p tools/agentic_xpu
# Edit tools/agentic_xpu/tokens.env to add GITHUB_TOKEN=ghp_YOUR_TOKEN_HERE
chmod 600 tools/agentic_xpu/tokens.env
```

Alternatively, export `GITHUB_TOKEN` in your shell before running.

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

## 7. Filing Issues to torch-xpu-ops

Confirmed bugs are tracked as issues in **[intel/torch-xpu-ops](https://github.com/intel/torch-xpu-ops)** by default. Override the target with `XPU_ALIGNMENT_ISSUE_REPO` when needed.

### Naming and label conventions

| Field | Convention |
|-------|-----------|
| **Title** | `[cuda_xpu_alignment] <original bug title>` |
| **Labels** | Always: `xpu-alignment`; plus `upstream-issue` or `upstream-pr` depending on source |
| **Description** | Start with the source header, then the full issue body as defined in §6 |

### Available labels in torch-xpu-ops

| Label | When to use |
|-------|-------------|
| `xpu-alignment` | All cuda-xpu alignment scan issues (always applied) |
| `upstream-issue` | Sourced from a pytorch/pytorch GitHub issue |
| `upstream-pr` | Sourced from a pytorch/pytorch GitHub PR (typically a fix PR where the bug was confirmed still present) |
| `confirmed` | Local XPU result is `confirmed` |
| `related-failure` | Local XPU result is `related-failure` |
| `daily-scan` | Optional: tag issues that came from an automated daily scan |

### Filing with gh CLI

```bash
REPO=${XPU_ALIGNMENT_ISSUE_REPO:-intel/torch-xpu-ops}

# For a bug sourced from an upstream issue:
gh issue create \
  --repo "$REPO" \
  --title "[cuda_xpu_alignment] <title>" \
  --label "xpu-alignment,upstream-issue" \
  --body "$(cat runs/<date>/reports/issue_drafts.md | <extract relevant section>)"

# For a bug sourced from an upstream PR:
gh issue create \
  --repo "$REPO" \
  --title "[cuda_xpu_alignment] <title>" \
  --label "xpu-alignment,upstream-pr" \
  --body "..."
```

### Filing from issue_drafts.md in bulk

Each issue in `reports/issue_drafts.md` is delimited by `## Issue N` headings.
The body of each issue starts after the metadata lines and ends before the next `## Issue` heading.

Example bulk filing (adapt the body extraction to your editor or scripting preference):

```bash
REPO=${XPU_ALIGNMENT_ISSUE_REPO:-intel/torch-xpu-ops}
VENV=$WORKSPACE/.conda-xpu-fix-alignment

# Re-verify before filing
"$VENV/bin/python" runs/2026-05-18_to_2026-05-21/scripts/repro_184340.py

# Then file
gh issue create \
  --repo "$REPO" \
  --title "[cuda_xpu_alignment] torch.compile wraps IndexError as BackendCompilerFailed, breaking except IndexError handlers" \
  --label "xpu-alignment,upstream-issue" \
  --body-file <(sed -n '/^## Issue 2$/,/^## Issue 3$/{ /^## Issue [0-9]/!p }' \
      runs/2026-05-18_to_2026-05-21/reports/issue_drafts.md)
```

## 8. Cron / Automated Daily Scans

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

## 9. Resuming an Interrupted Scan

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

## 10. Environment Variables

All optional — auto-detected if unset.

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENCODE_BIN` | Path to `opencode` binary | `opencode` in `$PATH` |
| `WORKSPACE` | Workspace root (where `.opencode/` lives) | Auto-detected upward from entry dir |
| `ENV_FILE` | Path to `tokens.env` file with `GITHUB_TOKEN` | `<entry-dir>/../tokens.env` |
| `GITHUB_TOKEN` | GitHub personal access token | Inherited from env or `tokens.env` |
| `GH_TOKEN` | Alias for `GITHUB_TOKEN` (used by `gh` CLI) | Set from `GITHUB_TOKEN` |
| `XPU_ALIGNMENT_ISSUE_REPO` | Target issue repo for generated filing commands | `intel/torch-xpu-ops` |

---

## 11. Troubleshooting

### `ERROR: SKILL.md not found`

`SKILL.md` is expected at `.github/skills/xpu-alignment/SKILL.md`.
Verify the repo is fully cloned:

```bash
git rev-parse --show-toplevel
ls "$(git rev-parse --show-toplevel)/.github/skills/xpu-alignment/SKILL.md"
```

If missing, re-clone or restore the file from the repo.

### `ERROR: opencode not found`

Set `OPENCODE_BIN` explicitly:

```bash
OPENCODE_BIN=~/.opencode/bin/opencode bash daily_scan.sh
```

### `xpu: False` after venv install

oneAPI runtime is not loaded. Source the environment:

```bash
source /opt/intel/oneapi/setvars.sh
# then re-verify:
$WORKSPACE/.conda-xpu-fix-alignment/bin/python \
  -c "import torch; print(torch.xpu.is_available())"
```

### GitHub API rate limit errors

The scan uses the authenticated `gh` CLI. If you see 403 / rate-limit errors:

```bash
gh auth status          # verify token is valid
gh api rate_limit       # check remaining quota
```

A full batch scan (4 days) consumes approximately 800–1200 API calls.
A GitHub PAT with `public_repo` scope and no extra permissions is sufficient.

### Audit reports `STATUS: INCOMPLETE`

The audit script checks that every ledger row with `title_status=pass` and
`deep_status=pass-to-repro` has `local_status=done`. Re-run the scan
(it will resume from the ledger):

```bash
bash batch_scan.sh 2026-05-18 2026-05-21
```

### SyntaxError in repro scripts (many `blocked-script-error` entries)

Some repro scripts contain raw upstream issue body text that was not cleanly extracted.
These are expected and counted in `blocked-script-error`. They do not affect confirmed results.
