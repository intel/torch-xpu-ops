---
name: xpu-alignment
description: Evidence-backed XPU triage for upstream PyTorch changes. Use when scanning a date window, deciding whether observed behavior is a real XPU bug, independently reviewing alignment cases or GitHub objects, or preparing canonical XPU work for issue-handler.
---

# XPU Alignment

Apply a proof ladder to upstream behavior and leave every in-scope item with one
auditable outcome. `issue-handler` exclusively owns implementation and PR
creation.

## Modes and inputs

Choose one mode:

- **Full-window scan:** receive a UTC start/end date, a run directory under
  `agent_space_xpu/runs/<window>/`, a workspace-local XPU interpreter, and a
  tracking repository. Scan source repository `pytorch/pytorch`. Default the
  window to yesterday UTC and the tracking repository to
  `intel/torch-xpu-ops`.
- **Direct review:** receive exact issue/PR URLs and a review run directory.
  Default the directory to
  `agent_space_xpu/runs/review-<UTC-timestamp>/`. Start at Step 5. Do not invent
  local runtime claims when scan evidence is absent.

Keep every generated artifact inside the run directory. Use GitHub credentials
for collection and publishing only; never expose them to a repro process. Treat
issue creation, comments, reviews, labels, closures, handler execution, and PR
creation as separately authorized writes.

## Proof ladder

Call a case `confirmed-xpu-issue` only when all four gates pass:

1. **Runtime:** a faithful isolated repro reaches the intended oracle.
2. **Bug:** supported deterministic behavior proves a defect.
3. **XPU fix:** code-path evidence proves XPU needs an independent change.
4. **Canonical:** live state proves whether an XPU tracker exists and identifies
   all fix PRs.

A matched repro is a runtime result, not an issue verdict. Creating or finding
the canonical tracker does not change a proved XPU issue into `track-upstream`.

## Full-window scan

### 1. Isolate the environment

Read [environment setup](references/xpu-alignment-environment-setup.md). Establish
tested-build provenance, pass controls, and provide a credential-free repro
sandbox. **Complete when:** controls and sandbox checks pass, or coverage records
the run as `blocked` with one causal incident.

### 2. Collect and account for the window

Read [scan execution](references/xpu-alignment-scan.md) and
[case contracts](references/xpu-alignment-buckets-and-routing.md). Preserve one
source row per raw source and one case row per selected case. **Complete when:**
all UTC shards are exhausted and the raw/source-ledger sets match exactly.

### 3. Establish runtime proof

Read [scan execution](references/xpu-alignment-scan.md) and
[evidence contract](references/xpu-alignment-evidence-contract.md). Execute ready
cases serially in the repro sandbox. **Complete when:** every selected case has
terminal audited evidence or a structured construction blocker; blockers prevent
workflow completion.

### 4. Establish issue, XPU-fix, and canonical proof

Read [case contracts](references/xpu-alignment-buckets-and-routing.md) and
[report and filing](references/xpu-alignment-report-and-issue-format.md).
Derive each assessment from its evidence and live state. **Complete when:** every
runtime-terminal case has a valid assessment and the scan report agrees with the
ledgers. Generate drafts, but do not file before independent review.

## Review and handoff

### 5. Review and hand off

Both modes enter here. Build the review manifest and delegate review exactly as
required by [independent review](references/xpu-alignment-review.md). For
full-window scans, then apply [handler handoff](references/xpu-alignment-handoff.md).
**Complete when:** `review_status=pass`, every manifest unit has one conclusion,
and all assessment changes have been re-audited.

## Completion

`artifacts/coverage.json.workflow_status` is the sole full-window status:
`partial`, `blocked`, or `completed`. Set `completed` only after scan audit and
independent review both pass, no selected case remains pending or blocked, and
all report counts agree. Reports display this value; they do not define another
status.

For direct review, `artifacts/review_state.json.review_status` is authoritative.
A direct review cannot produce a scan completion claim or handler handoff.
