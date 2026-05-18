# Private Repo Pipeline Pre-flight Plan

**Goal:** Validate the pytorch-agent pipeline end-to-end against `intel-sandbox/torch-xpu-ops-exp`, confirm routing logic, run a small test batch, and produce a pre-flight report.

**Architecture:** Read-only audit (Phase 2) → 2-issue test run (Phase 3) → report (Phase 4). Config already updated and labels cleaned up. All pipeline invocations go through `run_pipeline.py --once`.

**Tech Stack:** `run_pipeline.py`, `format_agent.py`, `triage_agent.py`, `agent_config.yml`, `gh` CLI, `intel-sandbox/torch-xpu-ops-exp`

---

## ✅ Done (Phase 0 + Phase 1)

- `agent_config.yml`: all three repo keys → `intel-sandbox/torch-xpu-ops-exp`
- Body templates: `🎯 Triage complete` action item added; `triage_agent.py` checks it on TRIAGED
- Labels: `agent:triaged` + `agent:done` confirmed present
- Label anomalies cleaned: #312, #335, #342, #336, #341, #338
- Issue roster: 42 issues, all open — 35 fresh, 7 at DISCOVERED

---

## Phase 2 — Routing Audit

**Objective:** Confirm the pipeline correctly routes `torch-xpu-ops` issues to Copilot and `pytorch` issues to opencode, with no code that accidentally calls `intel/torch-xpu-ops`.

### Task 2.1: Verify no stale `intel/torch-xpu-ops` references in Python code

**Files:** `~/torch-xpu-ops/.github/issue-handler/issue_handler/` (all `.py`)

**Step 1:** Search for hardcoded `intel/torch-xpu-ops` strings
```bash
grep -rn "intel/torch-xpu-ops" ~/torch-xpu-ops/.github/issue-handler/issue_handler/ --include="*.py"
```
Expected: zero hits (all repo refs should come from `config.py` constants).

**Step 2:** Confirm the two constants used at call sites are `ISSUE_REPO` (not `UPSTREAM_ISSUE_REPO`)
```bash
grep -rn "UPSTREAM_ISSUE_REPO" ~/torch-xpu-ops/.github/issue-handler/issue_handler/ --include="*.py"
```
Expected: only defined in `config.py`, never imported elsewhere.

**Verification:** Zero stale hardcoded strings; `UPSTREAM_ISSUE_REPO` unused.

---

### Task 2.2: Trace routing logic — Copilot path vs opencode path

**Files:** `orchestrator.py`, `triage_agent.py`, `fixing_steps/code_fix.py`

**Step 1:** Read routing decision in `triage_agent.py` (target_repo inference)
```bash
grep -n "target_repo\|torch-xpu-ops\|pytorch" \
  ~/torch-xpu-ops/.github/issue-handler/issue_handler/triage_agent.py
```

**Step 2:** Read the TRIAGED handler in `orchestrator.py` — confirm it stops and does NOT auto-dispatch to fix agent
```bash
grep -A 5 "TRIAGED" ~/torch-xpu-ops/.github/issue-handler/issue_handler/orchestrator.py
```

**Step 3:** Read `code_fix.py` to confirm it reads `target_repo` from issue body metadata and dispatches accordingly
```bash
grep -n "target_repo\|copilot\|opencode\|_assign_copilot" \
  ~/torch-xpu-ops/.github/issue-handler/issue_handler/fixing_steps/code_fix.py | head -30
```

**Expected routing table:**
| `target_repo` value | Path |
|---|---|
| `torch-xpu-ops` | `_assign_copilot()` → comment + assign on `intel-sandbox/torch-xpu-ops-exp` |
| `pytorch` | opencode agent on `chuanqi129/pytorch` |

**Verification:** Document exact routing logic; flag any gaps.

---

### Task 2.3: Verify dashboard update path

**Files:** `scripts/e2e_report.py`

**Step 1:** Confirm `update_tracking_issue` targets `TRACKING_REPO` (now `intel-sandbox/torch-xpu-ops-exp`) and updates issue #1694
```bash
grep -n "TRACKING_REPO\|tracking_num\|update_tracking" \
  ~/torch-xpu-ops/.github/issue-handler/scripts/e2e_report.py | head -20
```

**Step 2:** Verify it doesn't hardcode issue number — check how it finds #1694
```bash
grep -n "1694\|dashboard\|tracking" \
  ~/torch-xpu-ops/.github/issue-handler/scripts/e2e_report.py
```

**Verification:** Dashboard update writes to `intel-sandbox/torch-xpu-ops-exp#1694` via `TRACKING_REPO`.

---

## Phase 3 — Test Run (2 issues)

**Objective:** Run 2 issues through FORMAT → DISCOVERED → TRIAGED (or NEEDS_HUMAN), verify labels and body are correct, and confirm dashboard #1694 updates.

**Issue selection criteria:**
- One `torch-xpu-ops` category issue (Copilot path) — prefer a fresh, simple bug
- One `pytorch` dependency issue (opencode path) — prefer one with `agent_dependency: upstream-pytorch` label

**Candidates (from Phase 1 roster):**
- **#32** (eltwise, fresh) — "channel last hardswish_ extra copy" — likely torch-xpu-ops kernel path
- **#94** (multi, fresh) — "test_cow failures" — good routing test (target unknown until format)

*(#198 and #312 disqualified — both already at terminal TRIAGED/NEEDS_HUMAN from prior sessions)*

### Task 3.1: Reset and verify issue state

**Step 1:** Reset #198 if it has any prior pipeline stage (it's fresh — likely skip)
```bash
cd ~/torch-xpu-ops/.github/issue-handler && set -a && source .env && set +a
python -m issue_handler.format_agent --issue 198 --reset 2>&1 | tail -5
```

**Step 2:** Confirm #312 is at DISCOVERED (already verified in Phase 1)
```bash
gh issue view 312 --repo intel-sandbox/torch-xpu-ops-exp --json body --jq '.body' | head -3
```

**Verification:** #198 has no `agent:status` marker; #312 shows `<!-- agent:status:DISCOVERED -->`.

---

### Task 3.2: Run cycle 1 — FORMAT #198, TRIAGE #312

```bash
cd ~/torch-xpu-ops/.github/issue-handler && set -a && source .env && set +a
python scripts/run_pipeline.py --once --issues 198 312 2>&1
```

**Expected output:**
```
#198: stage=NONE, advancing...  → DISCOVERED
#312: stage=DISCOVERED, advancing... → TRIAGED or NEEDS_HUMAN
E2E report updated: #1694
```

**Step 2:** Check #198 body for correct `<!-- agent:status:DISCOVERED -->` and `agent:active` label
```bash
gh issue view 198 --repo intel-sandbox/torch-xpu-ops-exp --json body,labels \
  --jq '{stage: (.body | match("agent:status:(\\w+)").captures[0].string), labels: [.labels[].name | select(startswith("agent:"))]}'
```

**Step 3:** Check #312 body — confirm stage + `🎯 Triage complete [x]` + `target_repo` set
```bash
gh issue view 312 --repo intel-sandbox/torch-xpu-ops-exp --json body \
  --jq '.body' | grep -E "agent:status|Triage complete|target_repo"
```

**Verification:** #198 at DISCOVERED with `agent:active`; #312 at TRIAGED with checkbox checked and `target_repo` visible.

---

### Task 3.3: Run cycle 2 — FORMAT #198 second pass → TRIAGE

```bash
cd ~/torch-xpu-ops/.github/issue-handler && set -a && source .env && set +a
python scripts/run_pipeline.py --once --issues 198 2>&1
```

**Expected:** #198 advances DISCOVERED → TRIAGED or NEEDS_HUMAN.

**Verification:** #198 reaches a terminal stage; no Python errors.

---

### Task 3.4: Verify dashboard #1694

```bash
gh issue view 1694 --repo intel-sandbox/torch-xpu-ops-exp --json body --jq '.body' | head -60
```

**Expected:** Dashboard body updated with both issues listed.

---

## Phase 4 — Pre-flight Report

**Objective:** Write a markdown report summarising the full pre-flight run.

### Task 4.1: Write `PREFLIGHT_REPORT.md`

**File:** `~/torch-xpu-ops/.github/issue-handler/PREFLIGHT_REPORT_PRIVATE.md`

**Contents:**
- Config changes summary (Phase 0)
- Full issue roster table (42 issues, from Phase 1)
- Routing audit findings (Phase 2)
- Test run results (Phase 3): stages reached, labels, any errors
- Label health: before/after anomaly cleanup
- Issues to watch (any NEEDS_HUMAN from test run — document why)
- Recommendation: ready to run full batch? Any blockers?

**Format:** Flat table per section, clickable links, additive only (never delete prior content).

---

## Remaining Work (after pre-flight)

- Run full batch: all 35 fresh + 7 DISCOVERED issues through FORMAT + TRIAGE
- Phase 2 of pipeline: fix dispatch (Copilot for torch-xpu-ops, opencode for pytorch)
- Cron setup once pipeline is stable
