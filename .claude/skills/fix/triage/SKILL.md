---
name: fix/triage
description: >
  Analyze a failure and determine root cause, fix strategy, target repo,
  domain, and verdict (IMPLEMENTING or NEEDS_HUMAN). Analysis-only — no code
  changes. Used by both issue-handler and nightly-ci-fix orchestrators.
---

# Triage — Root Cause Analysis

Analysis-only. You may run read-only inspection commands (`read`/`grep`,
`git clone`, `git show`) to inspect source, but do not run tests or edit
files. After returning `IMPLEMENTING`, the orchestrator hands off to
`fix/implement`.

## Inputs

- Failure description: error log, reproducer command or test name, context.
- Read-only codebase access (`read`/`grep`).
- If a runnable test command is available, the orchestrator should have already
  run `fix/reproduce` before calling this skill. Do NOT run tests yourself.

## Your task

Determine:
1. **Root cause** — what exactly is failing and why.
2. **Fix strategy** — what files/functions to change.
3. **Target repo** — `pytorch` or `torch-xpu-ops`.
4. **Domain** — which domain knowledge pack applies (see Step 1).
5. **Verdict** — `IMPLEMENTING` (agent can fix) or `NEEDS_HUMAN`.

## Step 0: Quick classification

Skip deep analysis if any of these apply:

- Already triaged with a root-cause analysis you trust → confirm existing
  verdict and stop.
- Labeled `task` / `[Task]` / `[Feature]`, or describes broad alignment work →
  `NEEDS_HUMAN`: "Umbrella/task issue, not a single fixable bug."
- Describes a "feature gap" or "blocked by missing feature" → `NEEDS_HUMAN`.
- Performance issue with no specific failing test → `NEEDS_HUMAN`:
  "Performance optimization requires human design decision."
- Clear error message/stack trace → proceed to Step 1.

## Step 1: Classify the failure type and domain

- **kernel/operator bug** (`domain: xpu-kernel`) — failure in backend-specific
  operator or kernel code.
- **test porting issue** (`domain: cuda-porting`) — a test ported from another
  backend (CUDA) fails due to porting gaps: wrong tolerances, missing kernel,
  incorrect device assumptions.
- **core framework bug** (`domain: upstream-pytorch`) — failure in
  device-agnostic framework code that surfaces on XPU.

Check which repo you're in: `basename $(git rev-parse --show-toplevel)`

## Step 2: Obtain upstream source for cross-reference

If fixing code in an external submodule repo (e.g. torch-xpu-ops), clone the
upstream project to compare kernel implementations and check for existing fixes.
Reuse the checkout in `agent_space_xpu/pytorch/` if it already exists:

```bash
if [[ ! -d agent_space_xpu/pytorch/.git ]]; then
    git clone --depth 1 https://github.com/pytorch/pytorch.git agent_space_xpu/pytorch
fi
```

See domain skill (loaded by orchestrator) for upstream path mappings.

## Step 3: Investigate

1. **Read the failure carefully** — error log, reproducer, context.
2. **Identify what changed.** For a regression, ask: which component changed
   between the working and broken versions? Root cause belongs to the thing
   that changed, not just where the error fires.
3. **Check if already fixed upstream.** Search for recent commits touching the
   relevant file(s)/function(s). If a fix already exists, report it and do NOT
   duplicate it.
4. **Trace the failing code path** with `read`/`grep`. Stop when you have
   enough to make a call.
5. **Determine root cause by where the fix must be made**, not by keywords:
   - A symbol named `nan` is not a NaN bug unless the bug is about NaN propagation.
   - A stack trace through `autograd` does not make it an autograd bug.
   - A tolerance failure is a test/tolerance issue, not necessarily a kernel bug.
6. **Skip/xfail decorators are NOT fixes.** Their presence confirms the issue
   exists. Do NOT conclude "already fixed" because a skip decorator exists.

### NEEDS_HUMAN signals

- Hardware-specific failure with no self-contained repro script.
- Depends on a non-public model/checkpoint/dataset, or a distributed setup.
- Version-upgrade breakage with no minimal script and no identifiable changed
  component.

## Step 4: Decide the right repo

- Root cause in **device-agnostic/framework code** → fix belongs in **pytorch**.
- Root cause in **backend-specific kernel/dispatch code** → fix belongs in the
  backend repo (e.g. **torch-xpu-ops**).

See domain skill (loaded by orchestrator) for path conventions.

## Step 5: Assess fixability

- Fix is clearly within source → `IMPLEMENTING`.
- Requires hardware, complex redesign, or genuinely unresolvable statically →
  `NEEDS_HUMAN`.

## Step 5.5: Sanity check

Before emitting output, confirm all three:

1. **Root cause and fix strategy are consistent** — the fix location is where
   the bug originates, not just where the error fires.
2. **`target_repo` matches the fix location** — if the fix is in pytorch core
   code, `target_repo` must be `"pytorch"`, not `"torch-xpu-ops"`.
3. **Not concluding "already fixed" from a skip decorator** — a skip confirms
   the issue exists; it is not a fix.

If any check fails, revise before emitting.

## Fix-strategy principles

- **Minimal changes** — fix only what's broken.
- **Align with upstream** — match upstream logic, tolerances, and behavior
  unless the feature depends on hardware-specific details.
- **Never skip tests** — the strategy must FIX the test, never add skip
  decorators. Exception: `fix/implement` with `allow_skip=true` may add a skip
  with tracking issue when explicitly requested by the orchestrator.
- **Issue-driven** — address the root cause, not merely make one reproducer pass.

See Step 1 above for domain routing.

## Output

Return to the orchestrator:

```
### Triage Result
- **Issue type:** <kernel/operator bug | test porting issue | core framework bug>
- **Fix repo:** <pytorch | torch-xpu-ops | N/A>
- **Root cause:** <2-3 sentences>
- **Fix strategy:** <files/functions to change, or "None">
- **Verdict:** <IMPLEMENTING / NEEDS_HUMAN> — <one-line reason>
```

```json
{
  "root_cause": "2-3 sentences",
  "fix_strategy": "specific files/functions to change",
  "target_repo": "pytorch or torch-xpu-ops",
  "domain": "xpu-kernel or cuda-porting or upstream-pytorch",
  "verdict": "IMPLEMENTING or NEEDS_HUMAN",
  "reason": "one-line reason"
}
```

## HARD RULES
- NEVER make code changes or create PRs for `task`-labeled issues.
- NEVER submit a torch-xpu-ops PR for a bug whose root cause is in pytorch.
- NEVER recommend adding skip decorators as the fix strategy.
- NEVER conclude "already fixed" solely because a skip decorator exists.
