---
name: xpu-alignment
description: Proof-ladder XPU triage for upstream PyTorch changes. Use when scanning a date window, checking whether an XPU reproduction is a real bug that needs an independent XPU fix, reviewing alignment issues or PRs, or preparing canonical cases for issue-handler.
---

# XPU Alignment

Find upstream PyTorch behavior that may affect XPU and leave a closed-world,
canonical outcome. `issue-handler` owns implementation and PR creation.

## Inputs

- Scan window; default to yesterday.
- Run directory under `agent_space_xpu/runs/<scan-window>/`.
- Workspace-local XPU interpreter and ambient GitHub credentials.
- Tracking repository: default `intel/torch-xpu-ops`; an experimental run must
  receive an explicit tracking-repository override from the caller.

## Proof Ladder

Apply all four gates before calling a case a `confirmed-xpu-issue`:

1. **Runtime:** a faithful, isolated repro reaches the intended oracle.
2. **Bug:** supported, deterministic behavior proves a defect.
3. **XPU fix:** code-path analysis proves XPU needs an independent change.
4. **Canonical:** live state shows no existing XPU tracker and records fix PRs.

A matched repro is only a runtime result, never an issue verdict.

## Steps

### 1. Isolate the environment

Read [environment setup](references/xpu-alignment-environment-setup.md). Pass
required controls and isolate every attempt. **Complete when:**
`environment.json` records passing controls or one incident explains the block.

### 2. Collect and account for the window

Read [scan execution](references/xpu-alignment-scan.md) and
[case contracts](references/xpu-alignment-buckets-and-routing.md). Preserve one
source row per raw source and one row per merged case. **Complete when:** shards
are exhausted, source id sets match, and coverage agrees with both ledgers.

### 3. Establish runtime proof

Read [scan execution](references/xpu-alignment-scan.md) and
[evidence contract](references/xpu-alignment-evidence-contract.md). Execute ready
cases serially and isolate shared incidents. **Complete when:** every ready case
has audited evidence and every construction gap is structured; never downgrade
a ready case to avoid execution.

### 4. Establish issue, XPU-fix, and canonical proof

Read [case contracts](references/xpu-alignment-buckets-and-routing.md) and
[report and filing](references/xpu-alignment-report-and-issue-format.md).
Challenge validity, execution path, fix scope, and live state. **Complete when:**
each executed case has one audited assessment and verdict. A case-level `PASS`
may be filed with separate authorization while the run is partial.

### 5. Review and hand off

Delegate review exactly as required by
[review](references/xpu-alignment-review.md), then read
[handler handoff](references/xpu-alignment-handoff.md) and prepare only eligible
XPU-fix work. **Complete when:** `review_status=PASS`, each reviewed object has
one canonical verdict, the batch review dashboard is written, and each eligible
case has an evidence-backed next action.

## Run Completion

Write `STATUS=completed`, the final dashboard, and batch handoff only when the
run audit and independent review both record `PASS`. Otherwise write
`STATUS=partial|blocked` and the unmet conditions. All GitHub writes require
action-specific authorization.

## Outputs

- Environment, raw sources, source/case ledgers, coverage, audit, and details.
- Repros, attempt logs, runtime evidence, and case assessments.
- Scan, issue-draft, review-conclusion, review-dashboard, comment-draft, and
  handler-handoff reports.
