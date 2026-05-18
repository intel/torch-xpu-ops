# Plan: Dashboard per-batch table support

**Date:** 2026-05-18  
**Branch:** `agent/pipeline-redesign-v2`

---

## Goal

Replace the single flat table in dashboard #1694 with named per-batch sections that accumulate additively. Old sections are never rewritten.

---

## Design

### Dashboard structure (new)

```markdown
# E2E Pipeline Test Dashboard

**Last updated:** 2026-05-18 ...

---

## [Run 1] Pre-flight test — 2026-05-18
| # | Title | Format | Triage | Fix | PR | Review | Model | Tokens | Cost | Result | Failure Reason |
...rows...

**Summary:** 4 issues · 2 TRIAGED · 2 NEEDS_HUMAN · cost $0.48

---

## [Run 2] Batch run — 35 issues — 2026-05-18
...rows...

**Summary:** ...

---
```

### Rules
1. Each call to `update_tracking_issue()` appends a **new** `## [Run N]` section.
2. Existing sections are **never modified** — preserved verbatim.
3. Run number N = count of existing `## [Run` sections + 1.
4. `build_report()` now returns just one section (header + table + summary), not the full dashboard.
5. `update_tracking_issue()` assembles full body = existing content + new section.

---

## Tasks

### Task 1 — `e2e_report.py`: split build/assemble responsibilities
- `build_section(results, batch_name, repo)` → returns one `## [Run N] ...` markdown block
  - Header: `## [Run N] {batch_name} — {date}`
  - Table rows (same columns as today)
  - Summary line: `**Summary:** N issues · X TRIAGED · Y NEEDS_HUMAN · Z IN PROGRESS · cost $C`
- Remove old `build_report()` or keep as thin wrapper for backward compat
- `update_tracking_issue(repo, results, batch_name)`:
  - Fetch existing body
  - Count existing `## [Run` sections → derive N
  - Append `---\n\n{new_section}`
  - Push updated body
  - If issue doesn't exist yet: create with just the new section + top header

### Task 2 — `run_pipeline.py`: pass batch_name through
- Add `--batch NAME` optional arg (default: `"Run"`)
- Pass to `update_tracking_issue()`

### Task 3 — migrate existing rows to [Run 1]
- Fetch current dashboard body
- Wrap existing `## Issue Status` table as `## [Run 1] Pre-flight test — 2026-05-18`
- Push updated body (one-shot migration, not a code change)

### Task 4 — verify
- Dry-run `e2e_report.py --dry-run` with a known issue to check output shape
- Confirm dashboard body on #1694 looks correct

---

## File paths
- `~/torch-xpu-ops/.github/issue-handler/scripts/e2e_report.py`
- `~/torch-xpu-ops/.github/issue-handler/scripts/run_pipeline.py`
