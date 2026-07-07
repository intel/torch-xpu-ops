---
name: xpu-issues-triaging
description: >
  Triaging the pytorch or torch-xpu-ops issue. Use to determine root cause and fix strategy.
---

# Triage XPU / PyTorch Issue

> **Scope note:** This skill is **analysis-only**. Implementation belongs to the
> `fix/implement` skill. Any existing workflow that expected this skill to produce a
> code fix should delegate to `fix/implement` after this skill returns a verdict of
> `IMPLEMENTING`.

> **Execution mode:** this skill behaves differently in interactive (default)
> vs pipeline mode (e.g. whether it comments on the issue). See
> `issue-handler/SKILL.md` "Pipeline mode: issue body contract".

You **analyze**; you do not execute code or edit files. Implementation belongs
to the `fix/implement` skill. In this triage stage, you could only do modification for
unskip the `skip` decorator or just do some prints for triaging. DO NOT commit
any code changes!

## Inputs

- One GitHub issue — structured or raw — including its error log, reproducer (if
any), surrounding context, and labels. You have read-only access to the
codebase (`read`/`grep`).
- If user offered the raw test command, delegate to the `fix/reproduce` skill
to run it and confirm the failure before triaging; do NOT run it yourself.


## Your Task

Determine:
1. **Root cause** — what exactly is failing and why.
2. **Fix strategy** — what files/functions to change.
3. **Target repo** —  The bug belongs to `pytorch` or `torch-xpu-ops`.
4. **Verdict** — can an agent fix this (`IMPLEMENTING`) or does it need a human
   (`NEEDS_HUMAN`).

## Step 0: Quick classification (BEFORE deep analysis)

- If the issue is already at the `agent:triaged` status (or already carries a
  root-cause analysis and fix strategy you trust), do NOT re-triage. Confirm the
  existing verdict and stop.
- If labeled `task` / `[Task]` / `[Feature]`, or it describes broad
  alignment/enablement work -> `NEEDS_HUMAN`, reason "Umbrella/task issue, not a
  single fixable bug". Do NOT make code changes for `task`-labeled issues. In
  **interactive mode** (default) tell the user why you stopped; in **pipeline
  mode** leave a comment on the issue explaining why you stopped. (See
  `issue-handler/SKILL.md` "Execution modes".)
- If it describes a "feature gap" or "blocked by missing feature" ->
  `NEEDS_HUMAN`.
- If category is "performance" with no specific failing test -> `NEEDS_HUMAN`,
  reason "Performance optimization requires human design decision".
- If it has a clear error message/stack trace -> proceed to Step 1.

## Step 1: Classify the failure type

Determine which category the issue falls into:
- **A) XPU kernel / operator bug** — failure in XPU-specific operator code.
- **B) PyTorch core bug** — failure in device-agnostic/framework code that
  surfaces on XPU.
- **C) CUDA UT porting issue** — a CUDA test ported to XPU fails due to porting
  gaps (see [Step 3b](#step-3b-cuda-ut-porting-issues)).

Check which repo you're in: `basename $(git rev-parse --show-toplevel)`
- `torch-xpu-ops` -> XPU kernel/operator code (files under `src/`).
- `pytorch` -> core code (files under `torch/`, `aten/`, `test/`, `c10/`).

## Step 2: Obtain PyTorch source for cross-reference

If you are in `torch-xpu-ops`, fetch the PyTorch repo to inspect upstream code:
```bash
git clone --depth 1 https://github.com/pytorch/pytorch.git /tmp/pytorch
```
Use `/tmp/pytorch/` to compare CUDA kernels, check upstream fixes, and verify
device-agnostic paths.

## Step 3: Investigate before deciding

1. **Read the issue body carefully** — error log, reproducer, context, labels.
2. **Identify what changed.** For a regression (worked on version X, broke on
   Y), ask *which component changed between those versions?* The root cause
   belongs to **the thing that changed**, not just where the error fires. If an
   external library broke because PyTorch changed behavior, the fix lives in
   PyTorch.
3. **Check if already fixed upstream.** Search PyTorch main for recent commits
   touching the relevant file(s)/function(s), and search GitHub issues/PRs for
   the same error/test name. If a fix already exists on PyTorch main, report it
   and do NOT duplicate it in torch-xpu-ops.
4. **Trace the failing code path** with `read`/`grep`. Stop when you have enough
   to make a call, not after counting files.
5. **Determine root cause** by exhausting static analysis (you cannot execute).
   Read the full call chain. If the root cause and a specific fix are
   identifiable from code alone, output `IMPLEMENTING` — even without running
   the reproducer. Only conclude "needs hardware to reproduce" if static
   analysis genuinely cannot determine what's wrong.
6. **Skip/xfail decorators are NOT fixes.** If the issue describes tests with
   `@skipIfXpu`, `@xfailIfXpu`, or similar and wants them removed so the tests
   pass on XPU, the decorator's presence **confirms the issue EXISTS** — it IS
   the problem, not a fix. Do NOT conclude "already fixed" because skip
   decorators exist.

### Root cause, not keywords

Label the failure by **where the fix must be made**, not by keywords that
happen to appear in the title, error, or stack trace. A keyword tells you what
failed, not why.

- A symbol mentioning `nan` in a parameter name is not a NaN/Inf bug unless the
  bug is actually about NaN propagation.
- A stack trace passing through `autograd` does not make it an autograd bug —
  check whether the bug is in autograd itself or just on the call path.
- A test failure tripping a tolerance threshold is a test/tolerance issue, not
  necessarily a kernel numerics bug.

The single question that decides root cause (and `target_repo`): **"where would
the fix need to be made?"**

### Needs reproduction (lean toward NEEDS_HUMAN / defer)

If the issue cannot be reduced to something an agent can reproduce or analyze
statically, lean toward `NEEDS_HUMAN` (or defer to the `fix/reproduce`
skill). Signals:

- Hardware-specific failure on a particular GPU model with no self-contained
  repro script.
- Depends on a non-public model/checkpoint/dataset, or a distributed/training
  setup that is not runnable in a few lines.
- Version-upgrade breakage described only at a high level, with no minimal
  script and no identifiable changed component.

### Step 3b: CUDA UT porting issues

If the failure is a CUDA unit test ported to XPU:
1. **Locate the original CUDA test** in the PyTorch repo (usually under
   `test/`). Identify CUDA-specific assertions, tolerances, dtypes, or device
   assumptions.
2. **Sketch a reproducer** mirroring the CUDA test's logic but targeting
   `device="xpu"`, with the same inputs, dtypes, and expected outputs.
3. **Diff CUDA vs XPU behavior** — supported dtypes, numerical tolerances
   (`atol`/`rtol`), operator dispatch paths, device-specific decorators/skips.
4. **Root cause** — decide whether the failure is:
   - Missing XPU kernel implementation -> fix in `src/`.
   - Incorrect test porting (wrong tolerance, missing dtype) -> fix in `test/`
     (pytorch repo).
   - Genuine behavioral difference requiring XPU-specific handling.

## Step 4: Decide the right repo for the fix

- Root cause in **device-agnostic/framework code** (`torch/`,
  `aten/src/ATen/`, `c10/`) -> fix belongs in **pytorch**. Do NOT submit a
  torch-xpu-ops PR. Surface the root cause and where the fix belongs: tell the
  user in interactive mode, or comment on the issue in pipeline mode.
- Root cause in **XPU-specific kernel/dispatch code**
  (`src/ATen/native/xpu/`, or `third_party/torch-xpu-ops/` inside the pytorch
  tree) -> fix belongs in **torch-xpu-ops**.

## Step 5: Assess fixability (verdict)

- Fix is within pytorch or torch-xpu-ops source -> `IMPLEMENTING`.
- Requires hardware changes, complex architecture redesign, or the root cause is
  genuinely unresolvable without running code -> `NEEDS_HUMAN`.

## Reproducer

Extract the reproducer command from the issue (pytest command, python script,
bash command, or just a test name). You **cannot run it** here — use it to
understand the code path, not to verify the failure. To actually reproduce
locally, use the `test-verification` skill.

## Failure categories

Classify the root cause to structure the fix strategy:

| Category | Description | Typical fix location |
|----------|-------------|---------------------|
| **XPU backend bug** | Bug in XPU kernel or backend code | `torch/_inductor/` or `third_party/torch-xpu-ops/` |
| **Tolerance too tight** | Numerical precision mismatch vs CUDA | Adjust `atol`/`rtol` to match CUDA |
| **Edge case / numerical accuracy** | NaN/Inf from extreme inputs, CPU-vs-XPU or fp32-vs-fp16 divergence, values near `finfo.max`/`min`, fuzzer-generated cases | Compare against CUDA/CPU reference; confirm it is a real bug, not expected precision behavior |
| **Skip decorator stale** | `@skipIfXpu`/`@expectedFailure` but test now passes | Remove decorator (see `fix/implement`) |
| **Upstream regression** | New upstream code broke XPU; needs XPU workaround | `torch/`, `aten/`, `test/` |
| **Test infrastructure** | Environment, import, or setup issue | Test file or CI config |

When the issue describes a newly added test, check the commit/PR that introduced
it to see if XPU support is expected — this affects the fix strategy.

## Fix-strategy principles (for the strategy field, not implementation)

- **Minimal changes** — fix only what's broken.
- **Align with CUDA** — when the feature does not depend on hardware-specific
  details (warp size, shared memory layout), the XPU strategy should match the
  CUDA implementation's logic, tolerances, and behavior. Use CUDA as reference.
- **Never skip tests** — the strategy must FIX the test, never add
  `@skipIfXpu`/`@skip`/`unittest.skip`.
- **Issue-driven, not reproducer-driven** — the strategy must address the root
  cause, not merely make one reproducer pass.

## Issue-body status (backward compatible)

**Pipeline mode only.** In interactive mode (default), return the triage result
to the user/orchestrator and do not write to the issue body. See
`issue-handler/SKILL.md` "Pipeline mode: issue body contract"
for the full contract.

This stage corresponds to legacy status stages `TRIAGING` -> `TRIAGED`. It
produces the Root Cause Analysis / Proposed Fix Strategy / Target Repository
content and the `<!-- agent:upstream-log -->` / `<!-- agent:triage-log -->` log
text. In pipeline mode the `issue-handler` orchestrator writes that content into
the issue body, advances `<!-- agent:status:... -->`, and sets the label
(`agent:active` while triaging, `agent:triaged` when done, `agent:needs-human`
on a `NEEDS_HUMAN` verdict).

## Output

**Pipeline mode:** Return ONLY this JSON block as the LAST thing in your
response, with no text after it. The `issue-handler` orchestrator is responsible
for writing it into the issue body and advancing the status marker.
```json
{
  "root_cause": "detailed analysis (2-3 sentences)",
  "fix_strategy": "specific files/functions to change",
  "target_repo": "pytorch or torch-xpu-ops",
  "verdict": "IMPLEMENTING or NEEDS_HUMAN",
  "reason": "one-line reason"
}
```

**Interactive mode:** Return the JSON above followed by the human-readable
summary below (no JSON-last constraint applies in interactive mode):
```
### Agent Summary
- **Issue type:** <kernel bug / pytorch core bug / CUDA UT porting / task>
- **Fix repo:** <pytorch / torch-xpu-ops / N/A (already fixed or task)>
- **What I found:** <root cause in one sentence>
- **Fix strategy:** <files/functions to change, or "None" for task issues>
- **CUDA alignment:** <how the strategy aligns with CUDA, or "N/A">
- **Verdict:** <IMPLEMENTING / NEEDS_HUMAN> — <reason>
```

### target_repo rules
- `"torch-xpu-ops"` — fix in `src/ATen/native/xpu/sycl/`,
  `src/ATen/native/xpu/`, or any path relative to torch-xpu-ops root. **This
  includes files under `third_party/torch-xpu-ops/` in the pytorch tree** —
  those are torch-xpu-ops source files bundled as a submodule.
- `"pytorch"` — fix in `torch/`, `aten/src/ATen/`, `test/`, `c10/`,
  `torch/_dynamo/`, `torch/_inductor/`, or any top-level pytorch path (but NOT
  `third_party/torch-xpu-ops/`).

## HARD RULES
- NEVER make code changes or create PRs for issues labeled **task**.
- NEVER submit a torch-xpu-ops PR for a bug whose root cause is in pytorch.
- NEVER recommend adding skip decorators. The strategy must FIX the test.
- NEVER conclude "already fixed" solely because a skip decorator exists.

## Next step

Once the verdict is `IMPLEMENTING`, hand off to the `fix/implement` skill to
implement and verify the fix.
