---
name: fix/triage
description: >
  Analyze a XPU/PyTorch failure and determine root cause, fix strategy, target
  repo, and verdict (IMPLEMENTING or NEEDS_HUMAN). Analysis-only — no code
  changes. Used by both issue-handler and nightly-ci-fix orchestrators.
---

# Triage — Root Cause Analysis

Analysis-only. Read code and reason about it; do not execute code or edit
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
4. **Verdict** — `IMPLEMENTING` (agent can fix) or `NEEDS_HUMAN`.

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

## Step 1: Classify the failure type

- **A) XPU kernel / operator bug** — failure in XPU-specific operator code.
- **B) PyTorch core bug** — failure in device-agnostic/framework code that
  surfaces on XPU.
- **C) CUDA UT porting issue** — a CUDA test ported to XPU fails due to porting
  gaps (see [Step 3b](#step-3b-cuda-ut-porting-issues)).

Check which repo you're in: `basename $(git rev-parse --show-toplevel)`
- `torch-xpu-ops` → XPU kernel/operator code (files under `src/`).
- `pytorch` → core code (files under `torch/`, `aten/`, `test/`, `c10/`).

## Step 2: Obtain PyTorch source for cross-reference

If in `torch-xpu-ops`, clone PyTorch to inspect upstream code:
```bash
git clone --depth 1 https://github.com/pytorch/pytorch.git /tmp/pytorch
```
Use `/tmp/pytorch/` to compare CUDA kernels, check upstream fixes, and verify
device-agnostic paths.

## Step 3: Investigate

1. **Read the failure carefully** — error log, reproducer, context.
2. **Identify what changed.** For a regression, ask: which component changed
   between the working and broken versions? Root cause belongs to the thing
   that changed, not just where the error fires.
3. **Check if already fixed upstream.** Search PyTorch main for recent commits
   touching the relevant file(s)/function(s). If a fix already exists, report
   it and do NOT duplicate it.
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

### Step 3b: CUDA UT porting issues

If the failure is a CUDA unit test ported to XPU:
1. **Locate the original CUDA test** in the PyTorch repo (under `test/`).
   Identify CUDA-specific assertions, tolerances, dtypes, or device assumptions.
2. **Sketch a reproducer** mirroring the CUDA test's logic but targeting
   `device="xpu"`.
3. **Diff CUDA vs XPU behavior** — dtypes, `atol`/`rtol`, dispatch paths,
   device-specific decorators.
4. **Root cause** — missing XPU kernel (fix in `src/`), incorrect test porting
   (fix in `test/`), or genuine behavioral difference.

## Step 4: Decide the right repo

- Root cause in **device-agnostic/framework code** (`torch/`, `aten/src/ATen/`,
  `c10/`) → fix belongs in **pytorch**.
- Root cause in **XPU-specific kernel/dispatch code** (`src/ATen/native/xpu/`,
  or `third_party/torch-xpu-ops/` inside the pytorch tree) → fix belongs in
  **torch-xpu-ops**.

### target_repo rules
- `"torch-xpu-ops"` — fix in `src/ATen/native/xpu/sycl/`, `src/ATen/native/xpu/`,
  or any path relative to torch-xpu-ops root. Includes files under
  `third_party/torch-xpu-ops/` in the pytorch tree.
- `"pytorch"` — fix in `torch/`, `aten/src/ATen/`, `test/`, `c10/`,
  `torch/_dynamo/`, `torch/_inductor/`, or any top-level pytorch path (but NOT
  `third_party/torch-xpu-ops/`).

## Step 5: Assess fixability

- Fix is clearly within pytorch or torch-xpu-ops source → `IMPLEMENTING`.
- Requires hardware, complex redesign, or genuinely unresolvable statically →
  `NEEDS_HUMAN`.

## Fix-strategy principles

- **Minimal changes** — fix only what's broken.
- **Align with CUDA** — match CUDA logic, tolerances, and behavior unless the
  feature depends on hardware-specific details.
- **Never skip tests** — the strategy must FIX the test, never add skip
  decorators. Exception: `fix/implement` with `allow_skip=true` may add skip
  with tracking issue when explicitly requested by the orchestrator.
- **Issue-driven** — address the root cause, not merely make one reproducer pass.

See [../references/failure-categories.md](../references/failure-categories.md)
for the full category taxonomy.

## Output

Return to the orchestrator:

```
### Triage Result
- **Issue type:** <kernel bug / pytorch core bug / CUDA UT porting / task>
- **Fix repo:** <pytorch / torch-xpu-ops / N/A>
- **Root cause:** <2-3 sentences>
- **Fix strategy:** <files/functions to change, or "None">
- **CUDA alignment:** <how the strategy aligns with CUDA, or "N/A">
- **Verdict:** <IMPLEMENTING / NEEDS_HUMAN> — <one-line reason>
```

```json
{
  "root_cause": "2-3 sentences",
  "fix_strategy": "specific files/functions to change",
  "target_repo": "pytorch or torch-xpu-ops",
  "verdict": "IMPLEMENTING or NEEDS_HUMAN",
  "reason": "one-line reason"
}
```

## HARD RULES
- NEVER make code changes or create PRs for `task`-labeled issues.
- NEVER submit a torch-xpu-ops PR for a bug whose root cause is in pytorch.
- NEVER recommend adding skip decorators as the fix strategy.
- NEVER conclude "already fixed" solely because a skip decorator exists.
