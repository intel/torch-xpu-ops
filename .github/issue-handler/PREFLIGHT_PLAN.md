# Private Repo Pipeline Pre-flight Plan

> **Status:** In progress — Phase 0 complete

**Goal:** Validate the pytorch-agent pipeline end-to-end against `intel-sandbox/torch-xpu-ops-exp`, confirm issue routing, and produce a pre-flight report with dashboard update.

**Architecture:** Pipeline runs via `run_pipeline.py --once --issues <N>`. Config permanently points to private repo. Fix routing: `torch-xpu-ops` issues → Copilot, `pytorch` issues → opencode. Dashboard is private repo issue #1694.

**Branch:** `agent/pipeline-redesign-v2` on `~/torch-xpu-ops`

---

## Phase 0 — Config & Label Setup ✅ DONE

Already completed:
- [x] `agent_config.yml`: `xpu_ops_issue` + `tracking` → `intel-sandbox/torch-xpu-ops-exp`
- [x] Both body templates: added `🎯 Triage complete` action item aligned to `agent:triaged` label
- [x] `triage_agent.py`: checks `"Triage complete"` when setting TRIAGED
- [x] Labels `agent:triaged` + `agent:done` confirmed present on private repo
- [x] Config verified: Python import shows correct values
- [x] Committed: `ed41f184`

---

## Phase 1 — Issue Roster Pre-flight

**Objective:** Read all private issues from the 6 category files, fetch their current state from GitHub, classify each as eligible/in-progress/terminal/closed, and produce a pre-flight table.

**Files to read:**
- `/home/stonepia/torch-xpu-ops/issues-for-torch-ops/conv_clean.md`
- `/home/stonepia/torch-xpu-ops/issues-for-torch-ops/eltwise_clean.md`
- `/home/stonepia/torch-xpu-ops/issues-for-torch-ops/gemm_clean.md`
- `/home/stonepia/torch-xpu-ops/issues-for-torch-ops/multi_clean.md`
- `/home/stonepia/torch-xpu-ops/issues-for-torch-ops/other_clean.md`
- `/home/stonepia/torch-xpu-ops/issues-for-torch-ops/reduction_clean.md`

**Step 1:** Parse all 6 files → extract `(private_issue_num, upstream_issue_num, title, category)` tuples.

**Step 2:** For each private issue, call `gh.get_issue_detail("intel-sandbox/torch-xpu-ops-exp", num)` and classify:
- `fresh` — no `agent:*` label and no `<!-- agent:status:... -->` in body
- `in-progress` — has `agent:active` label → read stage from body
- `terminal` — has `agent:needs-human`, `agent:skipped`, `agent:done`, or `agent:triaged` → skip
- `closed` — issue state is closed → skip

**Step 3:** Output pre-flight table:

| Private # | Upstream # | Title (truncated) | Category | State | Current Stage | Eligible |
|---|---|---|---|---|---|---|
| #N | #M | ... | conv | open | fresh | ✅ |

**Command (no pipeline, read-only Python script):**
```bash
cd ~/torch-xpu-ops/.github/issue-handler
set -a && source .env && set +a
python3 - <<'EOF'
# inline scan script — see execution notes
EOF
```

**Verification:** Table printed to stdout, count of eligible issues confirmed.

---

## Phase 2 — Routing Audit

**Objective:** Confirm the Copilot vs opencode routing logic works correctly for private repo issues.

**Step 1:** Read `triage_agent.py` `target_repo` inference logic (lines ~100–115) — confirm `torch-xpu-ops` → Copilot path, `pytorch` → opencode path.

**Step 2:** Read `orchestrator.py` `TRIAGED` case — confirm it stops (does not dispatch fix agent). Confirm `IMPLEMENTING` case routes via `code_fix.py`.

**Step 3:** Check `code_fix.py` for Copilot assignment call — confirm it calls `_assign_copilot()` for torch-xpu-ops target.

**Step 4:** Verify `_assign_copilot()` in `orchestrator.py` uses `ISSUE_REPO` (now private repo) for the assignment comment.

**No pipeline call needed — read-only audit.**

---

## Phase 3 — Test Run

**Objective:** Run 2 issues through FORMAT → TRIAGED on the private repo, confirm label sync and dashboard update.

**Step 1: Select issues**
- Pick one `torch-xpu-ops`-scoped issue (Copilot path) from Phase 1 eligible list
- Pick one `pytorch`-scoped issue (opencode path) from Phase 1 eligible list
- Prefer issues with `agent_dependency: upstream-pytorch` label for pytorch path

**Step 2: Reset each issue**
```bash
cd ~/torch-xpu-ops/.github/issue-handler
set -a && source .env && set +a
python -m issue_handler.format_agent --issue <N> --reset
python -m issue_handler.format_agent --issue <M> --reset
```
Expected: body reverted to original raw content, agent labels removed.

**Step 3: Cycle 1 — FORMAT**
```bash
python scripts/run_pipeline.py --once --issues <N> <M>
```
Expected: both issues advance `NONE → DISCOVERED`, body rendered with template, `agent:active` label set, `🔍 Issue formatted [x]`.

**Step 4: Cycle 2 — TRIAGE**
```bash
python scripts/run_pipeline.py --once --issues <N> <M>
```
Expected:
- `DISCOVERED → TRIAGED` (for fixable issues): `agent:triaged` label, `🎯 Triage complete [x]`
- OR `DISCOVERED → NEEDS_HUMAN` (for hardware/upstream-only): `agent:needs-human` label

**Step 5: Verify dashboard**
Check issue #1694 on `intel-sandbox/torch-xpu-ops-exp` was updated by `update_tracking_issue()`.
```bash
gh issue view 1694 --repo intel-sandbox/torch-xpu-ops-exp --json body | python3 -c "import json,sys; print(json.load(sys.stdin)['body'][:2000])"
```

**Verification:** Both issues at terminal or TRIAGED stage; dashboard #1694 shows updated report.

---

## Phase 4 — Pre-flight Report

**Objective:** Write `PREFLIGHT_REPORT_PRIVATE.md` with the full issue roster table plus test run results.

**Step 1:** Combine Phase 1 table (all issues) with Phase 3 results (observed stage transitions).

**Step 2:** Flag any pipeline errors or unexpected NEEDS_HUMAN verdicts for follow-up.

**Step 3:** Write report to `/home/stonepia/torch-xpu-ops/.github/issue-handler/PREFLIGHT_REPORT_PRIVATE.md`

**Report columns:**
| Private # | Upstream # | Title | Category | Pre-flight Status | Test Run Result | Notes |

---

## Execution Order

```
Phase 1 → Phase 2 (parallel, both read-only)
         ↓
       Phase 3 (test run, 2 issues)
         ↓
       Phase 4 (report)
```

---

## Key Constants

- Private repo: `intel-sandbox/torch-xpu-ops-exp`
- Upstream repo: `intel/torch-xpu-ops`
- Dashboard issue: `intel-sandbox/torch-xpu-ops-exp#1694`
- Pipeline entry: `scripts/run_pipeline.py --once --issues <N>`
- Reset: `python -m issue_handler.format_agent --issue <N> --reset`
- Env: `set -a && source .env && set +a` (must use this form, not plain `source`)
- Branch: `agent/pipeline-redesign-v2`
