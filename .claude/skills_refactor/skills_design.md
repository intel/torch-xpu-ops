# Skills Refactor — Design Document

## Problem statement

The original skills had three structural issues:

1. **Mixed responsibilities** — `test-verification`, `issue-fix`, and
   `xpu-issues-triaging` each contained pipeline-mode GitHub issue body write
   operations (status markers, log slots). These belong to the orchestrator,
   not the leaf skills.

2. **Duplicated logic** — the failure category table, skip decorator handling,
   and run-test logic appeared independently in multiple skills.

3. **No shared fix primitives** — `issue-handler` and `nightly-ci-fix` both
   handled triage, implementation, and verification inline, with no shared
   layer between them.

---

## Design principles

### Orchestrators own scheduling; leaf skills own logic

Leaf skills (`fix/reproduce`, `fix/triage`, `fix/implement`, `fix/verify`):
- Do one thing
- Accept inputs, return outputs
- Do not know which orchestrator called them
- Do not write to GitHub issue bodies or commit code

Orchestrators (`issue-handler`, `nightly-ci-fix`):
- Own the pipeline sequence and branching
- Interpret leaf skill outputs and decide next steps
- Own all GitHub/git side effects (write to issue body, commit, PR)
- Pass scenario-specific parameters to leaf skills

### Scenario differences expressed as parameters

Where the two orchestrators need different behavior from the same leaf skill,
the difference is expressed as an input parameter rather than a separate skill:

- `fix/implement`: `allow_skip=false` (issue-handler) vs `allow_skip=true`
  (nightly-ci-fix)
- `fix/verify`: `run_before_after_diff`, `run_lint`

### Shared references for duplicated content

Content that was duplicated across skills is extracted to reference docs:
- `fix/references/run-test.md` — path resolution, test commands, result
  interpretation
- `fix/references/environment-setup.md` — source build, xpu.txt pin workflow
- `fix/references/failure-categories.md` — root cause taxonomy

---

## Layer model

```
intel-gpu/          source-oneapi, device-selection, unitrace/setup
                    (Intel oneAPI / Level Zero — no PyTorch dependency)
                    referenced by fix/references/environment-setup.md

fix/                shared leaf skills (scenario-independent)
  reproduce/        three-stage: nightly wheel → source build → CI env
  triage/           root cause analysis, IMPLEMENTING/NEEDS_HUMAN verdict
  implement/        apply the fix, allow_skip parameter
  verify/           source build verification, before/after diff
  references/       run-test.md, environment-setup.md, failure-categories.md

issue-handler/      orchestrator: single GitHub issue
  → fix/reproduce (allow nightly fallback, NO_REPRODUCER → triage only)
  → fix/triage
  → fix/implement (allow_skip=false)
  → fix/verify (run_before_after_diff=false, run_lint=false)
  → xpu-ops-pr-creation

nightly-ci-fix/     orchestrator: batch CI failure report
  → fix/reproduce (ci_commit required, batch per failure)
  → fix/triage
  → fix/implement (allow_skip=true, commit_message_template)
  → fix/verify (run_before_after_diff=true, run_lint=true)
  → commit per fix
```

---

## Reproduce: three-stage verification

The reproduce stage uses nightly wheel first because most CI failures reproduce
there, making it the fastest path to confirming a bug.

```
Stage 1: Nightly wheel
  pip3 install --pre torch ... --index-url .../nightly/xpu
  FAILED                           → REPRODUCED (fast path, most cases)
  CANNOT_VERIFY (env problem)      → stop, report to user
  PASSED + nightly older than CI   → stage 2 (nightly too stale)
  PASSED + nightly same/newer      → stage 3 (inconclusive)

Stage 2: Source build at CI commit
  Only when nightly is too old to be conclusive.
  FAILED                           → REPRODUCED
  PASSED                           → stage 3
  CANNOT_VERIFY                    → stop

Stage 3: CI environment alignment
  Local passes; check if CI-environment-specific.
  FAILED                           → REPRODUCED (with env_diff)
  PASSED                           → NOT_REPRODUCED; triage collects reason
  CANNOT_VERIFY                    → stop
```

Nightly CANNOT_VERIFY means the environment itself is broken (wheel install
failed, oneAPI missing). Source build would have the same environment problem,
so skipping to stage 2 is not helpful.

---

## Issue types and which stages run

| Issue type | reproduce | triage | implement | verify |
|---|---|---|---|---|
| Has reproducer, bug confirmed | ✓ | ✓ | ✓ | ✓ |
| No reproducer (static analysis only) | — | ✓ | ✓ | ✓ |
| Reproduces → already fixed | ✓ | ✓ (collect reason) | — | — |
| Nonbug (task, feature, question) | — | — | — | — |
| NEEDS_HUMAN | ✓ | ✓ | — | — |

---

## implement: allow_skip parameter

| Scenario | allow_skip | Behavior |
|---|---|---|
| issue-handler | `false` | Never add skip decorators. Must unskip and really fix. Stale skips must be removed. |
| nightly-ci-fix | `true` | May add `@skipIfXpu` + tracking issue when implementation is out of scope for a nightly fix. Stale skips must still be removed. |

The goal of nightly-ci-fix is to unblock CI quickly. The goal of issue-handler
is to actually fix the bug. This single parameter captures the difference.

---

## execution-modes.md scope

`issue-handler/references/execution-modes.md` is **not** shared with
`nightly-ci-fix`. It contains the pipeline-mode contract for writing to GitHub
issue bodies (agent:status markers, label map, log slots), which is specific to
the issue-handler workflow. `nightly-ci-fix` never writes to GitHub issue
bodies; it writes to `agent_space_xpu/summary_<date>.md` instead.

---

## Files not changed

The following original skills are unchanged in this refactor. They remain in
`.claude/skills/` and are not duplicated here:

- `pr-review/` — XPU PR review, no overlap with fix flow
- `xpu-ops-pr-creation/` — called by issue-handler after verify; no changes
- `issue-format/` — called by issue-handler as Stage 1; no changes
- `auto-label/` — CI label automation; independent
- `release-branching/` — release workflow; independent
- `xpu-build-pytorch/` — standalone build skill; independent
- `xpu-alignment/` — upstream scanning; independent
- `oob-perf-analysis/` — performance analysis; independent
- `action/` skills — environment setup actions; independent
